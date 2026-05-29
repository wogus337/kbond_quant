"""
phase3 신호의 진짜 백테스트 — high_conf_backtest의 P(persist)↔chg_bp 정렬 오류 보정.

signal mapping (phase3와 동일):
  regime=DOWN + P(persist)≥thr → +1 (yield 계속 down → long duration)
  regime=DOWN + P(persist)≤(1-thr) → -1
  regime=UP   + P(persist)≥thr → -1 (yield 계속 up → cut duration)
  regime=UP   + P(persist)≤(1-thr) → +1
  else → 0 (no trade)

PnL_bp = -signal × chg_bp_forward
  (signal=+1: yield 하락 베팅 → chg_bp<0이면 +)
  (signal=-1: yield 상승 베팅 → chg_bp>0이면 +)
"""
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PHASE1 = ROOT / "results" / "phase1"
PHASE3 = ROOT / "results" / "phase3"
OUT    = ROOT / "results" / "phase3"

KTB_TARGETS = ["ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y",
               "spread_ktb_3_10", "spread_ktb_5_30", "spread_ktb_10_30"]

THRESHOLDS = {"STRONG": 0.65, "MODERATE": 0.60, "WEAK": 0.55}
START_OOS = pd.Timestamp("2018-01-01")


def signal_from_persist(p_persist: pd.Series, regime: pd.Series,
                          trigger: float) -> pd.Series:
    """phase3 매핑 (reverse 미선택 시 1-p를 reverse로 대체)."""
    p = p_persist
    sig = pd.Series(0, index=p.index, dtype=int)
    high = p >= trigger
    low  = p <= (1 - trigger)
    is_down = regime == "DOWN"
    is_up   = regime == "UP"
    sig[is_down & high] = +1   # DOWN persist
    sig[is_down & low]  = -1   # DOWN reverse
    sig[is_up   & high] = -1   # UP persist
    sig[is_up   & low]  = +1   # UP reverse
    return sig


def backtest_one(t: str, p_persist: pd.Series, chg_bp: pd.Series,
                  regime: pd.Series, trigger: float) -> pd.DataFrame:
    df = pd.DataFrame({"p": p_persist, "chg": chg_bp, "regime": regime}).dropna()
    df = df[df.index >= START_OOS]
    if df.empty:
        return df
    df["signal"] = signal_from_persist(df["p"], df["regime"], trigger)
    df["pnl_bp"] = -df["signal"] * df["chg"]
    df["pnl_bp"] = df["pnl_bp"].where(df["signal"] != 0, 0.0)
    df["cum_pnl"] = df["pnl_bp"].cumsum()
    df["dd"] = df["cum_pnl"] - df["cum_pnl"].cummax()
    return df


def summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    traded = df[df["signal"] != 0]
    if traded.empty:
        return {"n_signals": 0}
    n = len(traded)
    hit = (traded["pnl_bp"] > 0).mean()
    cum = float(df["cum_pnl"].iloc[-1])
    mdd = float(df["dd"].min())
    mean_pnl = float(traded["pnl_bp"].mean())
    days = (df.index[-1] - df.index[0]).days
    ann_pnl = cum / max(days, 1) * 365
    return {
        "n_signals": n,
        "coverage": n / len(df),
        "hit_rate": float(hit),
        "mean_pnl_bp": mean_pnl,
        "cum_pnl_bp": cum,
        "ann_pnl_bp": float(ann_pnl),
        "max_dd_bp": mdd,
        "calmar": float(ann_pnl / abs(mdd)) if mdd < 0 else float("inf"),
    }


def main():
    # 데이터 로드
    sig_tbl = pd.read_parquet(PHASE3 / "signal_table.parquet")
    pred    = pd.read_parquet(PHASE1 / "predictions.parquet")
    sel     = pd.read_csv(ROOT / "results" / "phase2" / "model_selection.csv",
                            encoding="utf-8-sig")
    sel = sel[sel["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]

    # 각 타겟별 대표 run_id (가장 첫 selected) — chg_bp, regime 추출용
    rep_run = sel.groupby("target")["run_id"].first().to_dict()
    rep_lookback = sel.groupby("target")["lookback_weeks"].first().to_dict()

    all_summaries = []
    pnl_curves = {}      # for plotting
    dd_curves = {}

    for level, trigger in THRESHOLDS.items():
        print(f"\n{'='*78}")
        print(f"  Threshold = {trigger:.2f}  [{level}]")
        print(f"{'='*78}")
        for t in KTB_TARGETS:
            ens_col = f"{t}__persist__ens"
            if ens_col not in sig_tbl.columns:
                print(f"  {t}: signal_table 없음, skip"); continue
            run = rep_run.get(t)
            chg_col = f"{t}__{run}__persist__chg_bp"
            reg_col = f"{t}__{run}__persist__regime"
            if chg_col not in pred.columns or reg_col not in pred.columns:
                print(f"  {t}: chg/regime 컬럼 없음, skip"); continue

            p_persist = sig_tbl[ens_col]
            chg_bp    = pred[chg_col]
            regime    = pred[reg_col]

            df = backtest_one(t, p_persist, chg_bp, regime, trigger)
            s = summary(df)
            s.update({"target": t, "level": level, "threshold": trigger})
            all_summaries.append(s)
            print(f"  {t:<18} n={s.get('n_signals',0):>4}  cov={s.get('coverage',0):.2f}  "
                  f"hit={s.get('hit_rate',0):.3f}  cum={s.get('cum_pnl_bp',0):+.1f}bp  "
                  f"ann={s.get('ann_pnl_bp',0):+.1f}bp/y  mdd={s.get('max_dd_bp',0):.1f}  "
                  f"calmar={s.get('calmar',0):.2f}")

            if level == "MODERATE":  # 곡선 저장: 0.60 기준
                pnl_curves[t] = df["cum_pnl"]
                dd_curves[t]  = df["dd"]

    # CSV 저장
    sm = pd.DataFrame(all_summaries)
    sm = sm[["level", "threshold", "target", "n_signals", "coverage",
             "hit_rate", "mean_pnl_bp", "cum_pnl_bp", "ann_pnl_bp",
             "max_dd_bp", "calmar"]]
    sm.to_csv(OUT / "ktb_backtest_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT / 'ktb_backtest_summary.csv'}")

    # 누적 PnL plot (MODERATE, threshold=0.60)
    if pnl_curves:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for t, c in pnl_curves.items():
            axes[0].plot(c.index, c.values, label=t, lw=1.2)
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_ylabel("Cumulative PnL (bp)")
        axes[0].set_title("KTB Cumulative PnL  —  trigger P(persist)≥0.60  (signal=±1 × −Δyield)")
        axes[0].legend(fontsize=8, ncol=4)
        axes[0].grid(alpha=0.3)
        for t, d in dd_curves.items():
            axes[1].plot(d.index, d.values, label=t, lw=1.0)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("Drawdown (bp)")
        axes[1].set_xlabel("OOS Date")
        axes[1].set_title("Drawdown")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "ktb_cum_pnl_dd.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"저장: {OUT / 'ktb_cum_pnl_dd.png'}")


if __name__ == "__main__":
    main()
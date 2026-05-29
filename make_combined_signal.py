"""
make_combined_signal.py
───────────────────────
phase1 (regime persist) + direction_strategy (yield sign) 결합 신호 생성

입력
────
  results/phase1/predictions.parquet      — 컬럼: {target}__{run_id}__persist__{col}
  results/phase1/regime_history.parquet   — 컬럼: {target}__L{W}W / regime, strength
  results/phase1/metrics_by_run.csv       — phase1 best 모델 선택용
  results/direction/predictions.parquet   — 컬럼: {target}__{H}__{col}
  results/direction/leaderboard.csv       — direction best 모델 선택용

출력 (results/combined/)
─────────────────────────
  signal_2W_by_target.csv      — H=10영업일, 10일 간격 샘플, 모든 OOS
  signal_4W_by_target.csv      — H=20영업일, 20일 간격 샘플
  signal_latest.csv            — 최신 일자 한 행씩
  hit_summary.csv              — batch별 hit rate 요약

룰
──
  종합신호 = (dir_prob, regime, persist_prob, dir_AUC) → STRONG / MODERATE / WEAK / NEUTRAL
  dir_AUC < 0.50 → dir 무시, persist만 사용
  batch = (issue_date가 속한 분기). 2018Q1=1, 2018Q2=2, ...

actual_chg_bp_same : yield[t+H] - yield[t]  (이론, 동일일)
actual_chg_bp_exec : yield[t+H+1] - yield[t+1]  (1일 지연 실현)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from features import DEFAULT_CACHE, build_target_series, load_cache


PROJECT_DIR = Path(__file__).resolve().parent
PHASE1_DIR  = PROJECT_DIR / "results" / "phase1"
DIRECT_DIR  = PROJECT_DIR / "results" / "direction"
OUT_DIR     = PROJECT_DIR / "results" / "combined"

START_OOS_YEAR = 2018
H_DAYS_MAP = {2: 10, 4: 20}    # forward_weeks → 영업일

ALL_TARGETS = [
    "ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y",
    "spread_ktb_3_10", "spread_ktb_5_30", "spread_ktb_10_30",
    "spread_aa_3y", "spread_bbb_3y", "spread_kepco_5y",
    "us_10y_xl", "us_ig_oas_xl",
    "us_spr_2_10", "us_spr_5_30", "us_spr_10_30",
]

MODELS = ["lgbm", "xgb", "rf", "et", "gbm", "logreg"]


# ═══════════════════════════════════════════════════════════════
# 유틸
# ═══════════════════════════════════════════════════════════════

def quarter_batch(ts: pd.Timestamp, start_year: int = START_OOS_YEAR) -> int:
    """2018Q1=1, 2018Q2=2, ..."""
    if pd.isna(ts):
        return -1
    return (ts.year - start_year) * 4 + ((ts.month - 1) // 3) + 1


def combine_signal(dir_prob: float, regime: str, persist_prob: float,
                    dir_auc: float) -> str:
    """규칙 기반 종합 신호.
       반환: STRONG_UP / STRONG_DOWN / MODERATE_UP / MODERATE_DOWN /
             PERSIST_UP / PERSIST_DOWN / WEAK / NEUTRAL
       UP = yield 상승 / spread 와이드닝 → 듀레이션 축소 / 크레딧 축소
       DOWN = yield 하락 / spread 타이트닝 → 듀레이션 확대 / 크레딧 확대
    """
    dir_valid = (not np.isnan(dir_auc)) and dir_auc >= 0.50
    dir_known = not np.isnan(dir_prob)
    p_known   = not np.isnan(persist_prob)
    r_known   = regime in ("UP", "DOWN")

    # dir이 신뢰할 수 없으면 persist 단독
    if not (dir_valid and dir_known):
        if not (p_known and r_known):
            return "NEUTRAL"
        if persist_prob >= 0.65:
            return f"PERSIST_{regime}"
        return "NEUTRAL"

    # 둘 다 사용 가능
    if not (p_known and r_known):
        # direction 단독
        if dir_prob >= 0.60: return "MODERATE_UP"
        if dir_prob <= 0.40: return "MODERATE_DOWN"
        return "WEAK"

    high_persist = persist_prob >= 0.65
    low_persist  = persist_prob <= 0.35
    strong_up    = dir_prob >= 0.60
    strong_dn    = dir_prob <= 0.40

    # STRONG: 방향 + regime + persist 모두 합의
    if strong_up:
        if regime == "UP"   and high_persist: return "STRONG_UP"
        if regime == "DOWN" and low_persist:  return "STRONG_UP"
    if strong_dn:
        if regime == "DOWN" and high_persist: return "STRONG_DOWN"
        if regime == "UP"   and low_persist:  return "STRONG_DOWN"

    # MODERATE: 방향 신호 + persist 일치
    mid_up = dir_prob >= 0.55
    mid_dn = dir_prob <= 0.45
    if mid_up and high_persist and regime == "UP":     return "MODERATE_UP"
    if mid_up and low_persist  and regime == "DOWN":   return "MODERATE_UP"
    if mid_dn and high_persist and regime == "DOWN":   return "MODERATE_DOWN"
    if mid_dn and low_persist  and regime == "UP":     return "MODERATE_DOWN"

    return "WEAK"


def signal_to_direction(sig: str) -> int:
    """+1 = up bet, -1 = down bet, 0 = no bet."""
    if "UP" in sig: return +1
    if "DOWN" in sig: return -1
    return 0


# ═══════════════════════════════════════════════════════════════
# 데이터 로더
# ═══════════════════════════════════════════════════════════════

def load_phase1_best(metrics_path: Path,
                       forward_weeks: int) -> pd.DataFrame:
    """phase1 metrics_by_run.csv 로드 → 각 타겟별로 AUC 최고 (target,run_id,model)."""
    df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    df = df[(df["direction"] == "persist") & (df["forward_weeks"] == forward_weeks)
              & (df["model"] != "ensemble")].copy()
    df["rank_auc"] = df.groupby("target")["auc"].rank(ascending=False, method="min")
    best = df[df["rank_auc"] == 1][["target", "run_id", "model", "auc",
                                      "lookback_weeks", "accuracy",
                                      "hit@70", "cov@70"]].copy()
    best.columns = ["target", "p1_run_id", "p1_model", "p1_auc", "p1_lookback",
                     "p1_acc", "p1_hit70", "p1_cov70"]
    return best.set_index("target")


def load_direction_best(leaderboard_path: Path,
                          horizon_days: int) -> pd.DataFrame:
    """direction leaderboard.csv 로드 → 각 타겟별 AUC 최고 (target,model)."""
    df = pd.read_csv(leaderboard_path, encoding="utf-8-sig")
    df = df[(df["horizon"] == horizon_days) & (df["model"] != "ensemble")].copy()
    df["rank_auc"] = df.groupby("target")["auc"].rank(ascending=False, method="min")
    best = df[df["rank_auc"] == 1][["target", "model", "auc", "accuracy",
                                      "hit@70", "cov@70"]].copy()
    best.columns = ["target", "dir_model", "dir_auc", "dir_acc",
                     "dir_hit70", "dir_cov70"]
    return best.set_index("target")


def get_regime_series(regime_hist_df: pd.DataFrame,
                        target: str, lookback_weeks: int) -> pd.DataFrame:
    """regime_history.parquet 에서 (target, lookback) 슬라이스 → 'regime' / 'strength' DataFrame."""
    key = f"{target}__L{lookback_weeks}W"
    # MultiIndex 컬럼: (key, 'regime'), (key, 'strength')
    if isinstance(regime_hist_df.columns, pd.MultiIndex):
        if key not in regime_hist_df.columns.get_level_values(0):
            return pd.DataFrame()
        sub = regime_hist_df[key]
    else:
        # 평면: f"{key}_regime", f"{key}_strength"
        cols = {c: c.split("__", 2)[-1] for c in regime_hist_df.columns
                if c.startswith(key)}
        if not cols:
            return pd.DataFrame()
        sub = regime_hist_df[list(cols.keys())].rename(columns=cols)
    return sub


# ═══════════════════════════════════════════════════════════════
# 핵심: 타겟별 신호 테이블 생성
# ═══════════════════════════════════════════════════════════════

def build_signal_table(target: str, forward_weeks: int,
                         phase1_pred: pd.DataFrame,
                         direction_pred: pd.DataFrame,
                         regime_hist: pd.DataFrame,
                         p1_best: pd.Series,
                         dir_best: Optional[pd.Series],
                         yield_series: pd.Series) -> pd.DataFrame:
    """단일 (target, forward_weeks) 에 대한 일별 신호 시리즈 → H일 간격 샘플."""
    h_days = H_DAYS_MAP[forward_weeks]

    # --- phase1 persist 확률 ---
    p1_col = f"{target}__{p1_best['p1_run_id']}__persist__proba_{p1_best['p1_model']}"
    persist_prob = phase1_pred[p1_col] if p1_col in phase1_pred.columns \
                    else pd.Series(np.nan, index=phase1_pred.index)

    # --- direction 확률 (없을 수 있음) ---
    if dir_best is not None:
        d_col = f"{target}__{h_days}__proba_{dir_best['dir_model']}"
        dir_prob = direction_pred[d_col] if d_col in direction_pred.columns \
                    else pd.Series(np.nan, index=direction_pred.index)
        dir_auc = float(dir_best["dir_auc"])
        dir_model_name = dir_best["dir_model"]
    else:
        dir_prob = pd.Series(np.nan, index=phase1_pred.index)
        dir_auc = float("nan")
        dir_model_name = None

    # --- regime + strength ---
    reg_df = get_regime_series(regime_hist, target, int(p1_best["p1_lookback"]))
    if reg_df.empty:
        regime_series = pd.Series("", index=phase1_pred.index)
    else:
        regime_series = reg_df["regime"] if "regime" in reg_df.columns \
                          else pd.Series("", index=phase1_pred.index)
        regime_series = regime_series.reindex(phase1_pred.index).ffill().fillna("")

    # --- yield 시리즈 정렬 ---
    yield_s = yield_series.reindex(phase1_pred.index).ffill()

    # --- 통합 일별 DF ---
    daily = pd.DataFrame({
        "yield":        yield_s,
        "regime":       regime_series,
        "persist_prob": persist_prob.reindex(phase1_pred.index),
        "dir_prob":     dir_prob.reindex(phase1_pred.index),
    })

    # OOS 시작 일자 (2018-01-01 이후의 첫 영업일)
    start_dt = pd.Timestamp(f"{START_OOS_YEAR}-01-01")
    oos = daily[daily.index >= start_dt].copy()
    if oos.empty:
        return pd.DataFrame()

    # H일 간격 샘플링 (영업일 기준 — DataFrame index는 이미 영업일)
    sample_idx = oos.index[::h_days]
    samples = oos.loc[sample_idx].copy()
    samples = samples.reset_index().rename(columns={"index": "발신일"})

    # --- 실제 변화량 계산 ---
    # 이론 (same-day): yield[t+H] - yield[t]
    yld = oos["yield"]
    chg_same: List[float] = []
    chg_exec: List[float] = []
    for issue_dt in samples["발신일"]:
        # same-day chg: t to t+H
        try:
            i = oos.index.get_loc(issue_dt)
        except KeyError:
            chg_same.append(np.nan); chg_exec.append(np.nan); continue
        j = i + h_days
        if j < len(oos):
            same = (yld.iloc[j] - yld.iloc[i]) * 100.0  # bp
        else:
            same = np.nan
        # exec: t+1 to t+H+1
        i1, j1 = i + 1, i + 1 + h_days
        if j1 < len(oos) and i1 < len(oos):
            exec_ = (yld.iloc[j1] - yld.iloc[i1]) * 100.0
        else:
            exec_ = np.nan
        chg_same.append(same)
        chg_exec.append(exec_)

    samples["actual_chg_bp_same"] = chg_same
    samples["actual_chg_bp_exec"] = chg_exec

    # --- 신호 결합 ---
    samples["종합신호"] = samples.apply(
        lambda r: combine_signal(r["dir_prob"], r["regime"],
                                   r["persist_prob"], dir_auc),
        axis=1,
    )
    samples["signal_dir"] = samples["종합신호"].apply(signal_to_direction)

    # hit: 신호 방향과 실제 부호 일치?
    def _hit(sd, chg):
        if sd == 0 or pd.isna(chg): return np.nan
        return float(np.sign(chg) == sd)

    samples["hit_same"] = [_hit(sd, c) for sd, c in
                            zip(samples["signal_dir"], samples["actual_chg_bp_same"])]
    samples["hit_exec"] = [_hit(sd, c) for sd, c in
                            zip(samples["signal_dir"], samples["actual_chg_bp_exec"])]

    # --- 컬럼 정리 ---
    samples["batch"]          = samples["발신일"].apply(quarter_batch)
    samples["타겟"]            = target
    samples["dir_model"]      = dir_model_name
    samples["dir_AUC"]        = round(dir_auc, 4) if not np.isnan(dir_auc) else np.nan
    samples["persist_model"]  = p1_best["p1_model"]
    samples["persist_run_id"] = p1_best["p1_run_id"]
    samples["persist_AUC"]    = round(float(p1_best["p1_auc"]), 4)

    cols = ["batch", "발신일", "타겟",
            "dir_prob", "dir_model", "dir_AUC",
            "regime",
            "persist_prob", "persist_model", "persist_run_id", "persist_AUC",
            "종합신호", "signal_dir",
            "actual_chg_bp_same", "actual_chg_bp_exec",
            "hit_same", "hit_exec"]
    return samples[cols]


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("━" * 80)
    print("  Combined Signal Builder — phase1 persist × direction sign")
    print("━" * 80)

    # 1) 입력 로드
    p1_pred_path = PHASE1_DIR / "predictions.parquet"
    p1_metr_path = PHASE1_DIR / "metrics_by_run.csv"
    p1_regh_path = PHASE1_DIR / "regime_history.parquet"
    d_pred_path  = DIRECT_DIR / "predictions.parquet"
    d_lb_path    = DIRECT_DIR / "leaderboard.csv"

    if not p1_pred_path.exists():
        print(f"  [ERR] {p1_pred_path} 없음. phase1_train.py 먼저 실행.")
        return
    if not d_pred_path.exists():
        print(f"  [WARN] {d_pred_path} 없음. direction 신호 없이 진행.")
        direction_pred = pd.DataFrame()
        direction_lb_exists = False
    else:
        direction_pred = pd.read_parquet(d_pred_path)
        direction_lb_exists = d_lb_path.exists()

    phase1_pred = pd.read_parquet(p1_pred_path)
    regime_hist = pd.read_parquet(p1_regh_path) if p1_regh_path.exists() \
                    else pd.DataFrame()

    # yield 원시 시리즈 (실제 변화량 계산용)
    cache = load_cache(DEFAULT_CACHE)
    yield_series_map: Dict[str, pd.Series] = {}
    for t in ALL_TARGETS:
        try:
            yield_series_map[t] = build_target_series(cache, t)
        except KeyError:
            print(f"  [skip yield] {t}")

    print(f"  phase1 predictions   : shape={phase1_pred.shape}")
    print(f"  direction predictions: shape={direction_pred.shape if not direction_pred.empty else '(없음)'}")
    print(f"  yield series         : {len(yield_series_map)} / {len(ALL_TARGETS)}")

    # 2) 각 horizon 별 처리
    latest_rows: List[Dict] = []
    for fw in [2, 4]:
        h_days = H_DAYS_MAP[fw]
        print(f"\n  ── forward_weeks={fw}  (H={h_days}d)  ─────────────")

        # phase1 best for this fw
        p1_best_df = load_phase1_best(p1_metr_path, fw)
        print(f"     phase1 best: {len(p1_best_df)} 타겟")

        # direction best for this fw (h_days)
        if direction_lb_exists:
            d_best_df = load_direction_best(d_lb_path, h_days)
            print(f"     direction best: {len(d_best_df)} 타겟")
        else:
            d_best_df = pd.DataFrame()

        all_rows: List[pd.DataFrame] = []
        for t in ALL_TARGETS:
            if t not in p1_best_df.index:
                print(f"     [skip {t}] phase1 결과 없음")
                continue
            if t not in yield_series_map:
                print(f"     [skip {t}] yield series 없음")
                continue
            p1_best = p1_best_df.loc[t]
            dir_best = d_best_df.loc[t] if (not d_best_df.empty and
                                              t in d_best_df.index) else None

            tbl = build_signal_table(
                target=t, forward_weeks=fw,
                phase1_pred=phase1_pred,
                direction_pred=direction_pred,
                regime_hist=regime_hist,
                p1_best=p1_best, dir_best=dir_best,
                yield_series=yield_series_map[t],
            )
            if not tbl.empty:
                all_rows.append(tbl)
                # latest 1 row per target
                latest_rows.append({**tbl.iloc[-1].to_dict(),
                                     "forward_weeks": fw})

        if not all_rows:
            print(f"     [warn] 출력 없음")
            continue
        out = pd.concat(all_rows, ignore_index=True)
        fp = OUT_DIR / f"signal_{fw}W_by_target.csv"
        out.to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"     → {fp}  ({len(out)} rows)")

        # batch별 hit rate 요약 (현재 horizon)
        sm = (out.dropna(subset=["hit_same"])
                  .groupby("batch")
                  .agg(n=("hit_same", "size"),
                       hit_same=("hit_same", "mean"),
                       hit_exec=("hit_exec", "mean"))
                  .reset_index())
        sm["forward_weeks"] = fw
        sm.to_csv(OUT_DIR / f"hit_summary_{fw}W.csv",
                   index=False, encoding="utf-8-sig")
        print(f"     → batch hit summary saved")

    # 3) latest 통합
    if latest_rows:
        latest = pd.DataFrame(latest_rows)
        cols = ["forward_weeks", "batch", "발신일", "타겟",
                "dir_prob", "dir_model", "dir_AUC",
                "regime",
                "persist_prob", "persist_model", "persist_AUC",
                "종합신호", "signal_dir",
                "actual_chg_bp_same", "actual_chg_bp_exec"]
        latest = latest[[c for c in cols if c in latest.columns]]
        latest = latest.sort_values(["타겟", "forward_weeks"])
        fp_latest = OUT_DIR / "signal_latest.csv"
        latest.to_csv(fp_latest, index=False, encoding="utf-8-sig")
        print(f"\n  → {fp_latest}  ({len(latest)} rows)")

    print(f"\n  완료. 결과: {OUT_DIR}")


if __name__ == "__main__":
    main()
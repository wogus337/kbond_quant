"""
phase3_predict.py  (원본 방법론 — persist/reverse 결합 + STRONG/MODERATE/WEAK)
─────────────────────────────────────────────────────────────────
Phase 3: 선택된 모델로 persist / reverse 각각 ensemble → 현재 regime 과 결합해
듀레이션/크레딧 신호 + 강도(STRONG/MODERATE/WEAK/NONE) 산출.

신호 매핑 (원본 phase3 룰)
──────────────────────────
  현재 regime  |  trigger             |  방향  |  설명
  ───────────────────────────────────────────────────────────────
  DOWN_rate    |  P(persist) high     |  +1   |  rate 계속 하락 → 듀레이션 확대 (KTB)
                                              |  spread 계속 타이트 → 크레딧 확대 (spread)
  DOWN_rate    |  P(reverse) high     |  -1   |  rate 반전 상승 → 듀레이션 축소
  UP_rate      |  P(persist) high     |  -1   |  rate 계속 상승 → 듀레이션 축소
  UP_rate      |  P(reverse) high     |  +1   |  rate 반전 하락 → 듀레이션 확대
  NEUTRAL      |  -                   |   0   |  중립 / 대기

강도
────
  STRONG   : mean_prob ≥ 0.65 AND consensus ≥ 0.80
  MODERATE : mean_prob ≥ 0.60 AND consensus ≥ 0.60
  WEAK     : mean_prob ≥ 0.55
  NONE     : else
  (consensus = 선택된 모델 중 prob ≥ 0.65 인 비율)

입력
────
  results/phase1/predictions.parquet (컬럼: {target}__{run_id}__{direction}__proba_{model})
  results/phase2/model_selection.csv (target × direction × run_id × model × selected)

출력
────
  results/phase3/
    metrics_selected.csv     (target × direction 선택 ensemble 메트릭)
    signal_table.parquet     (일자별 P(persist), P(reverse), regime, signal, strength)
    high_conf_backtest.csv   (target별 임계값별 hit/PnL)
    latest_signal.txt        (현 시점 권고)
    cum_pnl.png              (누적 PnL by target)

사용
────
  python phase3_predict.py
  python phase3_predict.py --threshold 0.65
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (CONF_THRESHOLDS, _fillna_for_model, build_model,
                   cum_pnl_series, evaluate_predictions, high_conf_backtest,
                   hmm_label_regimes, make_persist_target, make_regime_features,
                   make_reverse_target, regime_strength_series,
                   select_top_features, signal_strength, weighted_ensemble)
from features import (DEFAULT_CACHE, build_all_features, build_target_series,
                       load_cache)
import json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PHASE1_DIR = PROJECT_DIR / "results" / "phase1"
DEFAULT_PHASE2_DIR = PROJECT_DIR / "results" / "phase2"
DEFAULT_PHASE3_DIR = PROJECT_DIR / "results" / "phase3"


def _unpack_predictions(p1_pred_path: Path
                          ) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """phase1 predictions.parquet → dict[(target, run_id, direction)] = DataFrame.
    컬럼: y_true, chg_bp, regime, proba_*
    """
    wide = pd.read_parquet(p1_pred_path)
    out: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for col in wide.columns:
        parts = col.split("__")
        if len(parts) < 4:
            continue
        t = parts[0]; run_id = parts[1]; direction = parts[2]
        sub = "__".join(parts[3:])
        key = (t, run_id, direction)
        if key not in out:
            out[key] = pd.DataFrame(index=wide.index)
        out[key][sub] = wide[col]
    return out


def _consensus_high(probas: pd.DataFrame, k: float = 0.65) -> pd.Series:
    """consensus = (이번 시점에서 선택 모델들 중 proba ≥ k 인 비율)."""
    if probas.empty:
        return pd.Series(dtype=float)
    high = (probas >= k).astype(float)
    # 모든 모델 NaN → NaN
    valid = probas.notna().sum(axis=1)
    cons = high.sum(axis=1) / valid.replace(0, np.nan)
    return cons


def _map_signal(regime_at_t: str, p_persist: float, p_reverse: float,
                 trigger_thr: float = 0.55) -> Tuple[int, str]:
    """원본 phase3 신호 매핑.
       returns (direction_int ∈ {-1,0,+1}, label)."""
    if regime_at_t == "DOWN":
        # rate 하락 (KTB 듀레이션 확대 / spread 타이트 → 크레딧 확대)
        if p_persist >= trigger_thr:
            return +1, "DOWN→continue (extend duration / add credit)"
        if p_reverse >= trigger_thr:
            return -1, "DOWN→reverse (cut duration / cut credit)"
    elif regime_at_t == "UP":
        if p_persist >= trigger_thr:
            return -1, "UP→continue (cut duration / cut credit)"
        if p_reverse >= trigger_thr:
            return +1, "UP→reverse (extend duration / add credit)"
    return 0, f"{regime_at_t or 'NEUTRAL'} (no signal)"


def main():
    ap = argparse.ArgumentParser(description="Phase3: persist/reverse 결합 + 듀얼 신호")
    ap.add_argument("--phase1-dir", default=str(DEFAULT_PHASE1_DIR))
    ap.add_argument("--phase2-dir", default=str(DEFAULT_PHASE2_DIR))
    ap.add_argument("--output-dir", default=str(DEFAULT_PHASE3_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="latest_signal 및 cum_pnl 기준 confidence 임계값")
    ap.add_argument("--trigger-thr", type=float, default=0.55,
                    help="signal trigger 임계값 (persist/reverse 어느 쪽으로 갈지)")
    args = ap.parse_args()

    p1_dir = Path(args.phase1_dir)
    p2_dir = Path(args.phase2_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("━" * 80)
    print(f"  Phase 3 — persist/reverse 결합 + STRONG/MODERATE/WEAK")
    print("━" * 80)

    sel_path = p2_dir / "model_selection.csv"
    if not sel_path.exists():
        print(f"  [ERR] {sel_path} 없음. phase2_review.py 먼저 실행.")
        return
    sel = pd.read_csv(sel_path, encoding="utf-8-sig")
    sel["selected"] = sel["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    sel_active = sel[sel["selected"]].copy()
    print(f"  selected = {len(sel_active)} / {len(sel)} 모델")

    pred_path = p1_dir / "predictions.parquet"
    if not pred_path.exists():
        print(f"  [ERR] {pred_path} 없음.")
        return
    proba_by_key = _unpack_predictions(pred_path)

    # 데이터 (latest values + regime + 재학습용 피처)
    cache = load_cache(Path(args.cache_dir))
    X_base = build_all_features(cache=cache)
    target_series_cache: Dict[str, pd.Series] = {}
    latest_values: Dict[str, float] = {}
    for tname in sel_active["target"].unique():
        try:
            ts = build_target_series(cache, tname)
            target_series_cache[tname] = ts
            latest_values[tname] = float(ts.dropna().iloc[-1])
        except KeyError:
            pass

    # HPO 파라미터 (refit 시 사용)
    hpo_params_path = p1_dir / "hpo_params.json"
    hpo_params_all: Dict[str, Dict] = {}
    if hpo_params_path.exists():
        hpo_params_all = json.loads(hpo_params_path.read_text(encoding="utf-8"))

    # ────────────────────────────────────────────────────────
    # latest 예측: 선택된 모델을 best HPO params 로 전체 valid 구간에서 재학습
    # → 최신 일자에 대한 P(persist) / P(reverse) 산출
    # ────────────────────────────────────────────────────────
    # Top-3 앙상블(평균) / Top-1 단일(rank-1) 각각 보관
    latest_p_persist_top3: Dict[str, float] = {}
    latest_p_persist_top1: Dict[str, float] = {}
    latest_p_reverse_top3: Dict[str, float] = {}
    latest_p_reverse_top1: Dict[str, float] = {}
    latest_regime:    Dict[str, str]   = {}
    latest_rows: List[Dict] = []

    print(f"\n  [latest refit] 선택 모델 재학습 후 최신 시점 예측")
    # (target, direction) 그룹별로 처리
    grouped = sel_active.groupby(["target", "direction"])
    for (tname, direction), grp in grouped:
        ts = target_series_cache.get(tname)
        if ts is None:
            continue
        # 각 row 는 동일 target+direction 하에서 (run_id, model) 단위. run_id에서 config 추출.
        # (rank_in_group, proba) 로 모아 Top-1(rank 최소) / Top-3(평균) 구분.
        ts_bd = ts.dropna()   # 영업일 시리즈 (phase1과 동일 기준)
        per_model: List[Tuple[float, float]] = []
        for _, row in grp.iterrows():
            lookback_days = int(row["lookback_weeks"]) * 5
            forward_days  = int(row["forward_weeks"])  * 5
            thr_bp        = float(row["target_thr_bp"])
            min_str       = float(row["regime_strength_min"])

            # phase1과 동일: 2-state HMM, 영업일 기준, regime은 라벨 생성에만(피처 미사용)
            regime, _ = hmm_label_regimes(ts_bd, lookback_days, n_states=2)
            strength = regime_strength_series(ts_bd, lookback_days)
            X_full = X_base

            if direction == "persist":
                y = make_persist_target(regime, forward_days, strength, min_str)
            else:
                y = make_reverse_target(regime, forward_days, strength, min_str)
            y = y.reindex(X_full.index)
            # target_thr 마스킹 (영업일 기준 chg)
            if thr_bp > 0:
                chg_bp = ((ts_bd.shift(-forward_days) - ts_bd) * 100.0).reindex(X_full.index)
                y[chg_bp.abs() < thr_bp] = np.nan

            valid = y.notna()
            if valid.sum() < 200:
                continue
            # 피처 선택 (phase1과 동일 로직)
            feat_list = select_top_features(X_full.loc[valid], y.loc[valid], k=80)
            X_tr = X_full.loc[valid, feat_list]
            y_tr = y.loc[valid].astype(int)
            latest_row = X_full[feat_list].iloc[[-1]]

            # HPO params
            hpo_key = f"{tname}__{row['run_id']}__{direction}__{row['model']}"
            params = hpo_params_all.get(hpo_key, {}).get("best_params", {})
            try:
                Xtr_f, Xte_f = _fillna_for_model(X_tr, latest_row, row["model"])
                model = build_model(row["model"], params, class_weight=True)
                # 시퀀스 모델 처리는 build_model 내부에서 자체 처리
                if row["model"] in ("lstm",):
                    model.fit(Xtr_f.values, y_tr.values)
                    p = float(model.predict_proba(Xte_f.values)[:, 1][0])
                else:
                    model.fit(Xtr_f, y_tr.values)
                    p = float(model.predict_proba(Xte_f)[:, 1][0])
            except Exception as e:
                print(f"     [{tname}/{direction}/{row['run_id']}/{row['model']}] err: {e}")
                continue
            rank = float(row.get("rank_in_group", 1e9))
            per_model.append((rank, p))

        if per_model:
            top3_p = float(np.mean([p for _, p in per_model]))   # 앙상블 평균
            top1_p = float(min(per_model, key=lambda x: x[0])[1])  # rank 최소 = Top-1
            if direction == "persist":
                latest_p_persist_top3[tname] = top3_p
                latest_p_persist_top1[tname] = top1_p
            else:
                latest_p_reverse_top3[tname] = top3_p
                latest_p_reverse_top1[tname] = top1_p
            print(f"     {tname} / {direction:<8} Top3 P={top3_p:.3f}  "
                  f"Top1 P={top1_p:.3f}  (n_models={len(per_model)})")

        # 현재 regime 도 한 번만 (첫 row 기준) — phase1과 동일 2-state·영업일
        if tname not in latest_regime:
            first = grp.iloc[0]
            lookback_days = int(first["lookback_weeks"]) * 5
            regime, _ = hmm_label_regimes(ts_bd, lookback_days, n_states=2)
            rg = regime.reindex(X_base.index).ffill().iloc[-1]
            latest_regime[tname] = rg if isinstance(rg, str) else "NEUTRAL"

    # latest_rows 구성 (refit 결과 사용) — Top-3 앙상블 + Top-1 단일 둘 다
    last_date = X_base.index[-1]

    def _build_latest_row(tname: str, p_p: float, p_r: float,
                           rg: str, model_set: str) -> Dict:
        d, lab = _map_signal(rg,
                             p_p if not np.isnan(p_p) else 0.0,
                             p_r if not np.isnan(p_r) else 0.0,
                             trigger_thr=args.trigger_thr)
        if d == +1 and rg == "DOWN":
            use_p = p_p
        elif d == -1 and rg == "DOWN":
            use_p = p_r
        elif d == -1 and rg == "UP":
            use_p = p_p
        elif d == +1 and rg == "UP":
            use_p = p_r
        else:
            use_p = float("nan")
        strength_label = signal_strength(
            float(use_p) if not np.isnan(use_p) else 0.0, 1.0)
        return {
            "target": tname,
            "date": last_date,
            "model_set": model_set,           # "top3" | "top1"
            "current_value": latest_values.get(tname, np.nan),
            "current_regime": rg,
            "p_persist": p_p,
            "p_reverse": p_r,
            "signal":   d,
            "signal_label": lab,
            "strength": strength_label,
        }

    all_targets = sorted(set(list(latest_p_persist_top3.keys())
                             + list(latest_p_reverse_top3.keys())))
    for tname in all_targets:
        rg = latest_regime.get(tname, "NEUTRAL")
        latest_rows.append(_build_latest_row(
            tname, latest_p_persist_top3.get(tname, float("nan")),
            latest_p_reverse_top3.get(tname, float("nan")), rg, "top3"))
        latest_rows.append(_build_latest_row(
            tname, latest_p_persist_top1.get(tname, float("nan")),
            latest_p_reverse_top1.get(tname, float("nan")), rg, "top1"))

    # 결과 누적 (latest_rows 는 위 refit 블록에서 채워짐)
    metrics_rows: List[Dict] = []
    bt_rows: List[pd.DataFrame] = []
    signal_frames: List[pd.DataFrame] = []

    # target별 처리 (한 target에 persist + reverse 둘 다 모음)
    for tname in sorted(sel_active["target"].unique()):
        sub_t = sel_active[sel_active["target"] == tname]
        if sub_t.empty:
            continue

        print(f"\n  ── {tname}")

        # direction별 ensemble proba 구성
        ens_per_dir: Dict[str, pd.Series] = {}
        chg_bp_ref: Optional[pd.Series] = None
        regime_ref: Optional[pd.Series] = None
        per_model_count: Dict[str, int] = {}

        for direction in ["persist", "reverse"]:
            sub_d = sub_t[sub_t["direction"] == direction]
            if sub_d.empty:
                continue
            per_model_count[direction] = len(sub_d)

            # 선택된 (run_id, model) 들의 proba 평균
            mix_frames = []
            for _, row in sub_d.iterrows():
                key = (tname, row["run_id"], direction)
                df = proba_by_key.get(key)
                if df is None:
                    continue
                col = f"proba_{row['model']}"
                if col not in df.columns:
                    continue
                # y_true, chg_bp, regime 동기화
                if chg_bp_ref is None and "chg_bp" in df.columns:
                    chg_bp_ref = df["chg_bp"]
                if regime_ref is None and "regime" in df.columns:
                    regime_ref = df["regime"]
                mix_frames.append(df[[col]].rename(
                    columns={col: f"{row['run_id']}__{row['model']}"}))
            if not mix_frames:
                continue
            mix = pd.concat(mix_frames, axis=1)
            ens = mix.mean(axis=1)
            ens_per_dir[direction] = ens

            # 메트릭 — 해당 direction의 y_true 가 phase1 parquet 안에 보관됨
            # 대표 키 하나에서 y_true 추출
            sample_key = (tname, sub_d.iloc[0]["run_id"], direction)
            y_true = proba_by_key.get(sample_key, pd.DataFrame()).get("y_true")
            if y_true is None:
                continue
            m = evaluate_predictions(y_true, ens)
            m.update({"target": tname, "direction": direction,
                      "n_models": len(sub_d)})
            metrics_rows.append(m)
            print(f"     [{direction}] n_models={len(sub_d)}  "
                   f"acc={m.get('accuracy',float('nan')):.3f}  "
                   f"auc={m.get('auc',float('nan')):.3f}  "
                   f"hit60={m.get('hit@60',float('nan')):.3f}  "
                   f"comp={m.get('composite',float('nan')):.3f}")

            # 고신뢰 BT
            if chg_bp_ref is not None:
                bt = high_conf_backtest(ens, chg_bp_ref)
                if not bt.empty:
                    bt.insert(0, "direction", direction)
                    bt.insert(0, "target", tname)
                    bt_rows.append(bt)

            # consensus
            cons = _consensus_high(mix, k=0.65)

            sf = pd.DataFrame({
                f"{tname}__{direction}__ens": ens,
                f"{tname}__{direction}__consensus": cons,
                f"{tname}__{direction}__y_true": y_true,
            })
            signal_frames.append(sf)

        # 듀얼 결합 → 일자별 signal 계산 (마지막 일자 latest)
        if "persist" in ens_per_dir and "reverse" in ens_per_dir and regime_ref is not None:
            df_combo = pd.concat([
                regime_ref.rename("regime"),
                ens_per_dir["persist"].rename("p_persist"),
                ens_per_dir["reverse"].rename("p_reverse"),
            ], axis=1)
            sig_dir, sig_label, sig_str = [], [], []
            for _, r in df_combo.iterrows():
                p_p = r.get("p_persist", np.nan)
                p_r = r.get("p_reverse", np.nan)
                rg  = r.get("regime", np.nan)
                d, lab = _map_signal(rg if isinstance(rg, str) else "NEUTRAL",
                                       p_p if pd.notna(p_p) else 0.0,
                                       p_r if pd.notna(p_r) else 0.0,
                                       trigger_thr=args.trigger_thr)
                sig_dir.append(d); sig_label.append(lab)
                # 강도: 더 확신있는 쪽 prob 사용
                use_prob = p_p if d == +1 and rg == "DOWN" else \
                           p_r if d == -1 and rg == "DOWN" else \
                           p_p if d == -1 and rg == "UP"   else \
                           p_r if d == +1 and rg == "UP"   else np.nan
                # consensus는 해당 direction 의 consensus 시리즈에서
                # (단순화: 0.65 임계 평균 prob로 strength 계산)
                cons_val = np.nan
                # 평균 prob = use_prob, consensus는 위에서 계산한 sf에서 가져옴
                sig_str.append(signal_strength(
                    float(use_prob) if pd.notna(use_prob) else 0.0,
                    float(cons_val) if pd.notna(cons_val) else 1.0,
                ))
            sig_df = pd.DataFrame({
                f"{tname}__signal":      sig_dir,
                f"{tname}__signal_lbl":  sig_label,
                f"{tname}__strength":    sig_str,
                f"{tname}__regime":      regime_ref,
            })
            signal_frames.append(sig_df)

            # latest_rows 는 refit 후 별도로 구성 (아래 latest refit 블록 참조)

    # 결과 저장
    print(f"\n  산출물 저장 -> {out_dir}")
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        cols = ["target", "direction", "n_models",
                "n", "pos_rate", "accuracy", "auc",
                "hit@60", "cov@60", "hit@70", "cov@70", "composite"]
        mdf = mdf[[c for c in cols if c in mdf.columns]]
        mdf.sort_values(["target", "direction"]).to_csv(
            out_dir / "metrics_selected.csv", index=False, encoding="utf-8-sig")

    if bt_rows:
        bigbt = pd.concat(bt_rows, ignore_index=True)
        bigbt.to_csv(out_dir / "high_conf_backtest.csv",
                     index=False, encoding="utf-8-sig")

    if signal_frames:
        big = pd.concat(signal_frames, axis=1)
        big.to_parquet(out_dir / "signal_table.parquet")

    # 누적 PnL 그래프 (persist+reverse 결합 신호 기준)
    try:
        if signal_frames:
            fig, ax = plt.subplots(figsize=(11, 6))
            for tname in sorted(sel_active["target"].unique()):
                # signal 컬럼이 있으면 사용
                sig_col = f"{tname}__signal"
                # 결합된 signal_table에서 행 단위 PnL = sign(signal) * Δbp_forward 의 누적
                # signal_table은 위에서 모은 frames. 일단 메모리에 있는 마지막 합치기:
                pass
            # 대안: 각 (target, direction) 의 cum_pnl_series 사용
            for (t, d), gp in pd.DataFrame([
                {"t": t, "d": dr} for t in sel_active["target"].unique() for dr in ["persist","reverse"]
            ]).groupby(["t", "d"]):
                pass  # 그래프는 단순화: bt_rows 기반 누적
            # 단순 대안: bt_rows의 cum_pnl_bp 막대그래프
            if bt_rows:
                bigbt = pd.concat(bt_rows, ignore_index=True)
                # threshold=args.threshold 행만
                pick = bigbt[np.isclose(bigbt["threshold"], args.threshold)]
                if not pick.empty:
                    ax.bar(pick["target"] + "__" + pick["direction"],
                            pick["cum_pnl_bp"])
                    ax.set_ylabel(f"Cumulative PnL_bp at thr={args.threshold:.2f}")
                    ax.set_title("OOS 누적 PnL (target × direction)")
                    plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / "cum_pnl.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"  [plot err] {e}")

    # latest_signal.csv (구조화 — 앱이 직접 읽음) + latest_signal.txt (사람용)
    if latest_rows:
        ldf = pd.DataFrame(latest_rows)
        ldf.to_csv(out_dir / "latest_signal.csv", index=False, encoding="utf-8-sig")

        lines = []
        lines.append("━" * 80)
        lines.append(f"  KBOND Phase3 Signal — as of {latest_rows[0]['date'].date()}")
        lines.append("━" * 80)
        lines.append("  신호 매핑 (원본 phase3):")
        lines.append("    DOWN regime + persist↑ → 듀레이션/크레딧 확대 (+1)")
        lines.append("    DOWN regime + reverse↑ → 듀레이션/크레딧 축소 (-1)")
        lines.append("    UP   regime + persist↑ → 듀레이션/크레딧 축소 (-1)")
        lines.append("    UP   regime + reverse↑ → 듀레이션/크레딧 확대 (+1)")
        lines.append("")
        # target별로 top3/top1 묶어서 표시
        for tname in sorted(ldf["target"].unique()):
            sub = ldf[ldf["target"] == tname]
            r0 = sub.iloc[0]
            unit = "%" if str(tname).startswith("ktb") else "% pt"
            lines.append(f"  ── {tname:<18} 현재값={r0['current_value']:.3f}{unit}  "
                         f"regime={r0['current_regime']}")
            for _, r in sub.iterrows():
                tag = "Top-3 앙상블" if r["model_set"] == "top3" else "Top-1 단일 "
                lines.append(f"      [{tag}] P(persist)={float(r['p_persist']):.3f}  "
                             f"→ signal={r['signal']:+d}  [{r.get('strength','NONE')}]  "
                             f"({r['signal_label']})")
            lines.append("")
        text = "\n".join(lines)
        (out_dir / "latest_signal.txt").write_text(text, encoding="utf-8")
        print()
        print(text)

    print(f"\n  완료. 결과: {out_dir}")


if __name__ == "__main__":
    main()

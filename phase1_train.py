"""
phase1_train.py  (원본 phase1~3 방법론 충실 복원)
─────────────────────────────────────────────
Phase 1: HMM regime labeling + persist/reverse 타겟 + (lookback × forward × retrain ×
target_thr × regime_strength) 외부 config 그리드 × 내부 Optuna HPO × 워크포워드.

원본 검색공간 (디폴트)
─────────────────────
  lookback_weeks   ∈ {4, 8, 12, 16, 20, 26}   → 코드 디폴트 [4, 8, 12, 20]
  forward_weeks    ∈ {2}                       → 코드 디폴트 [2]
  retrain_days     ∈ {63, 126, 252}            → 코드 디폴트 [63]
  target_thr_bp    ∈ {0, 5, 10, 15, 20}        → 코드 디폴트 [0, 10]
  regime_strength  ∈ {0.0, 0.5, 1.0, 1.5}     → 코드 디폴트 [0.0, 1.0]
  → outer = 4×1×1×2×2 = 16 configs / 타겟

각 config 안에서
────────────────
  · HMM 으로 lookback-day yield 변화 → DOWN/NEUTRAL/UP 라벨
  · regime_strength < threshold 인 일자 마스킹
  · persist 타겟, reverse 타겟 각각 만들기
  · 모델별 Optuna HPO (inner_trials) → best params
  · retrain_days 마다 walk-forward
  · persist/reverse 각각 OOS 예측 + 메트릭

산출물 (results/phase1/)
─────────────────────────
  outer_configs.csv         (시도한 모든 config)
  metrics_by_run.csv        (target × config × direction × model × 메트릭)
  metrics_top.csv           (target × direction × CompositeScore top-K)
  predictions.parquet       (모든 (target, run_id, direction, model) OOS 확률)
  hpo_params.json
  feature_importance.csv
  selected_features.csv
  yearly_breakdown.csv
  fold_log.csv
  model_selection_template.csv  (phase2 진입용 - direction 컬럼 포함)
  regime_history.parquet    (target별 HMM regime + strength 시계열)
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import shutil
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

from core import (evaluate_predictions, hmm_label_regimes,
                   make_persist_target, regime_strength_series,
                   select_top_features, tune_model, walk_forward_periodic,
                   weighted_ensemble, yearly_breakdown)
from features import (DEFAULT_CACHE, build_all_features, build_target_series,
                       load_cache)


PROJECT_DIR = Path(__file__).resolve().parent

ALL_TARGETS = [
    # KTB 금리 (절대값)
    "ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y",
    # KTB 장단기 스프레드 (longer - shorter)
    "spread_ktb_3_10",   # ktb_10y - ktb_3y
    "spread_ktb_5_30",   # ktb_30y - ktb_5y
    "spread_ktb_10_30",  # ktb_30y - ktb_10y
    # 회사채 스프레드 (vs KTB)
    "spread_aa_3y", "spread_bbb_3y", "spread_kepco_5y",
    # 미국 (Bloomberg / Excel)
    "us_10y_xl",         # 미 10Y
    "us_ig_oas_xl",      # 미 IG OAS
    "us_spr_2_10",       # 2s10s
    "us_spr_5_30",       # 5s30s
    "us_spr_10_30",      # 10s30s
]
ALL_MODELS = ["lgbm", "xgb", "rf", "et", "gbm", "logreg"]

# ─── 원본 방법론 외부 그리드 (PyCharm 클릭 실행용 디폴트) ───
DEFAULT_LOOKBACK_WEEKS    = [4, 8, 12, 20]
DEFAULT_FORWARD_WEEKS     = [2, 4]
DEFAULT_RETRAIN_DAYS      = [63]
DEFAULT_TARGET_THR_BP     = [0.0, 10.0]
DEFAULT_REGIME_STRENGTH   = [0.0]

# ─── 기타 디폴트 ───
DEFAULT_START             = "20100101"
DEFAULT_START_OOS_YEAR    = 2018
DEFAULT_HPO_TRIALS        = 20    # 내부 HPO (원본 inner=20)
DEFAULT_HPO_SPLITS        = 4
DEFAULT_MIN_TRAIN_DAYS    = 750
DEFAULT_N_SEEDS           = 3
DEFAULT_TOP_FEATURES      = 80
DEFAULT_ENSEMBLE_MODE     = "weighted"
DEFAULT_OUTPUT_DIR        = str(PROJECT_DIR / "results" / "phase1")


def _print_metric_row(label: str, m: Dict[str, float], extra: str = ""):
    print(f"      {label:<11} "
          f"acc={m.get('accuracy',float('nan')):.3f}  "
          f"auc={m.get('auc',float('nan')):.3f}  "
          f"hit60={m.get('hit@60',float('nan')):.3f} "
          f"cov60={m.get('cov@60',float('nan')):.3f}  "
          f"comp={m.get('composite',float('nan')):.3f}  {extra}")


def _retrain_to_period(retrain_days: int) -> str:
    """retrain_days 를 walk_forward_periodic period 인자로 매핑.
       63 → quarter, 252 → year, 그 외 가까운 쪽 (126은 quarter*2지만 quarter 처리)."""
    if retrain_days >= 252:
        return "year"
    return "quarter"


_PROG_FILE = Path(__file__).resolve().parent / ".phase1_progress.json"
_PID_FILE  = Path(__file__).resolve().parent / ".phase1_retrain.pid"


def _write_progress(**kw) -> None:
    """진행상태 heartbeat (콘솔/detached 무관하게 모니터가 읽음). 실패는 무시."""
    try:
        import os
        kw.setdefault("pid", os.getpid())
        kw["ts"] = time.time()
        _PROG_FILE.write_text(json.dumps(kw, default=str), encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Phase1 (원본 방법론): config grid × HPO × persist/reverse")
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS,
                    help=f"예측 타겟 (KTB + 스프레드)")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    choices=ALL_MODELS)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--start-oos-year", type=int, default=DEFAULT_START_OOS_YEAR)
    ap.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)

    # 원본 외부 그리드
    ap.add_argument("--lookback-weeks", nargs="+", type=int,
                    default=DEFAULT_LOOKBACK_WEEKS,
                    help="lookback 후보 (주). HMM regime 라벨링 기간.")
    ap.add_argument("--forward-weeks", nargs="+", type=int,
                    default=DEFAULT_FORWARD_WEEKS,
                    help="forward 후보 (주). persist/reverse 예측 horizon.")
    ap.add_argument("--retrain-days", nargs="+", type=int,
                    default=DEFAULT_RETRAIN_DAYS,
                    help="재학습 주기 후보 (일). 63=분기, 252=연.")
    ap.add_argument("--target-thr-bp", nargs="+", type=float,
                    default=DEFAULT_TARGET_THR_BP,
                    help="|Δforward| < target_thr_bp 인 일자는 학습/평가 제외. bp 단위.")
    ap.add_argument("--regime-strength", nargs="+", type=float,
                    default=DEFAULT_REGIME_STRENGTH,
                    help="regime_strength(|Δlookback|/std) 가 이 이상이어야 학습 포함.")

    # HPO
    ap.add_argument("--hpo-trials", type=int, default=DEFAULT_HPO_TRIALS,
                    help="모델당 내부 HPO trials")
    ap.add_argument("--hpo-splits", type=int, default=DEFAULT_HPO_SPLITS)
    ap.add_argument("--hpo-timeout", type=int, default=None)
    ap.add_argument("--skip-hpo", action="store_true")

    # 모델 관련
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--top-features", type=int, default=DEFAULT_TOP_FEATURES,
                    help="피처 선택 K. 0 이면 전체 사용")
    ap.add_argument("--ensemble-mode", choices=["weighted", "equal"],
                    default=DEFAULT_ENSEMBLE_MODE)
    ap.add_argument("--no-class-weight", action="store_true")

    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--fresh", action="store_true",
                    help="체크포인트(_ckpt) 비우고 처음부터. 미지정 시 기존 ckpt 있으면 이어서(resume)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_weight = not args.no_class_weight

    # 체크포인트 디렉터리 (중단 내성: (config×target) 조합마다 저장 → 재개 시 skip)
    ckpt_dir = out_dir / "_ckpt"
    if args.fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        print(f"  [fresh] 기존 체크포인트 삭제: {ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _done0 = len(list(ckpt_dir.glob("*.pkl")))
    if _done0 > 0:
        print(f"  [resume] 기존 완료 조합 {_done0}개 → skip하고 이어서 진행")

    cutoff = pd.Timestamp(f"{args.start_oos_year}-01-01")

    # outer grid
    outer_configs = list(product(args.lookback_weeks, args.forward_weeks,
                                   args.retrain_days, args.target_thr_bp,
                                   args.regime_strength))

    # 진행 모니터링용 PID/heartbeat
    import os as _os
    _PID_FILE.write_text(str(_os.getpid()))
    _total_combos = len(outer_configs) * len(args.targets)
    _n_models = len(args.models)
    _write_progress(phase="start", combo_no=0, total_combos=_total_combos,
                    ckpt_done=_done0, n_models=_n_models,
                    target="", config="", models_done=0, last_model="",
                    started_at=time.time())
    print("━" * 80)
    print(f"  Phase1 (원본 방법론)  start={args.start}  start_oos_year={args.start_oos_year}")
    print(f"  targets       : {args.targets}")
    print(f"  models        : {args.models}")
    print(f"  outer configs : {len(outer_configs)}  ({args.lookback_weeks} × "
           f"{args.forward_weeks} × {args.retrain_days} × {args.target_thr_bp} × "
           f"{args.regime_strength})")
    print(f"  inner HPO     : {'OFF' if args.skip_hpo else f'{args.hpo_trials} trials × {args.hpo_splits} splits'}")
    print(f"  n_seeds={args.n_seeds}  top_features={args.top_features or 'ALL'}  "
          f"ensemble={args.ensemble_mode}  class_weight={'balanced' if class_weight else 'none'}")
    print(f"  out           : {out_dir}")
    print("━" * 80)

    # 1) 데이터 + 베이스 피처
    print("\n  [1/5] 캐시 로드 + 피처 빌드...")
    cache = load_cache(Path(args.cache_dir))
    X_base = build_all_features(cache=cache)
    X_base = X_base.loc[X_base.index >= pd.to_datetime(args.start)]
    print(f"        X_base shape: {X_base.shape}  range: {X_base.index[0].date()}~{X_base.index[-1].date()}")

    # 2) target 별 series 캐시
    target_series_cache: Dict[str, pd.Series] = {}
    for tname in args.targets:
        try:
            target_series_cache[tname] = build_target_series(cache, tname)
        except KeyError as e:
            print(f"    [skip] {tname}: {e}")

    # 누적 저장소
    outer_log: List[Dict] = []
    metrics_rows: List[Dict] = []
    proba_frames: List[pd.DataFrame] = []
    hpo_log: Dict[str, Dict] = {}
    selected_features_log: List[Dict] = []
    fold_log: List[Dict] = []
    yearly_log: List[pd.DataFrame] = []
    regime_history: Dict[str, pd.DataFrame] = {}
    fi_records: List[pd.DataFrame] = []

    # 3) outer × target 루프
    print(f"\n  [2/5] outer config × target 루프  ({len(outer_configs) * len(args.targets)} 조합)")
    run_id_counter = 0
    for ci, (lb_w, fw_w, retrain_d, thr_bp, min_str) in enumerate(outer_configs, 1):
        lookback_days = int(lb_w * 5)
        forward_days  = int(fw_w * 5)
        period_str    = _retrain_to_period(retrain_d)

        for tname in args.targets:
            ts = target_series_cache.get(tname)
            if ts is None:
                continue
            # 영업일 시리즈: 주말/공휴일 NaN 제거 → lookback_days/forward_days(=주*5)가
            # 캘린더가 아닌 '영업일' 기준으로 적용됨. regime/strength/label/chg 모두 동일 기준.
            ts_bd = ts.dropna()

            run_id = f"L{lb_w}W_F{fw_w}W_R{retrain_d}d_T{thr_bp:.0f}bp_S{min_str:.1f}"
            run_id_counter += 1

            # 체크포인트: 이미 완료된 조합이면 skip (재개)
            ckpt_file = ckpt_dir / f"{run_id}__{tname}.pkl"
            if ckpt_file.exists():
                print(f"  ── [{run_id_counter}] {tname}  config={run_id}  [resume skip]")
                continue
            print(f"\n  ── [{run_id_counter}] {tname}  config={run_id}")
            # 이 조합이 전역 리스트에 추가하는 분량 추적용 스냅샷
            _snap = {
                "metrics": len(metrics_rows), "fold": len(fold_log),
                "yearly": len(yearly_log), "fi": len(fi_records),
                "selfeat": len(selected_features_log), "outer": len(outer_log),
                "proba": len(proba_frames),
            }

            # 3-a) HMM regime + strength (train-only fit, 2-state UP/DOWN) — 영업일 기준
            regime, _probs = hmm_label_regimes(ts_bd, lookback_days, n_states=2,
                                                  fit_until=cutoff)
            strength = regime_strength_series(ts_bd, lookback_days)
            regime_history[f"{tname}__L{lb_w}W"] = pd.DataFrame({
                "regime": regime, "strength": strength,
            })
            # 자기참조 차단: HMM regime은 라벨 생성에만 사용, 피처로는 안 씀
            X_full = X_base

            # 3-b) persist 타겟 (2-state에서는 reverse = 1-persist 라 별도 학습 불필요)
            y_persist = make_persist_target(regime, forward_days, strength, min_str)\
                          .reindex(X_full.index)

            # forward 변화량 (백테스트용) — 영업일 기준 10영업일 변화
            chg_bp_full = ((ts_bd.shift(-forward_days) - ts_bd) * 100.0).reindex(X_full.index)

            # forward 변화 |Δ| < thr_bp 인 일자 추가 마스킹 (target_thr)
            if thr_bp > 0:
                weak_fwd = chg_bp_full.abs() < thr_bp
                y_persist[weak_fwd] = np.nan

            # 통계
            n_per = int(y_persist.notna().sum())
            print(f"       valid persist={n_per}  "
                   f"regime DOWN/UP="
                   f"{(regime=='DOWN').sum()}/{(regime=='UP').sum()}  "
                   f"pos_rate={y_persist.mean():.3f}")
            if n_per < 200:
                print(f"       [skip] insufficient samples")
                continue

            # 3-c) 피처 선택 (persist 타겟 기준)
            if args.top_features and args.top_features > 0:
                X_hpo = X_full.loc[X_full.index < cutoff]
                y_hpo_p = y_persist.loc[X_hpo.index]
                t_fs = time.time()
                feat_list = select_top_features(X_hpo, y_hpo_p, k=args.top_features)
                print(f"       feat-select top-{args.top_features}: "
                      f"{len(feat_list)} features ({time.time()-t_fs:.1f}s)")
            else:
                feat_list = list(X_full.columns)
            selected_features_log.append({
                "target": tname, "run_id": run_id,
                "n_features": len(feat_list),
                "features": ",".join(feat_list[:25]) + (" ..." if len(feat_list) > 25 else ""),
            })

            # 3-d) model 루프 (persist 단일 direction)
            for direction, y in [("persist", y_persist)]:
                n_valid = int(y.notna().sum())
                if n_valid < 200:
                    print(f"       [{direction}] valid={n_valid} → skip")
                    continue
                print(f"       ── {direction}  valid={n_valid}  pos_rate={y.mean():.3f}")

                X_hpo = X_full.loc[X_full.index < cutoff, feat_list]
                y_hpo = y.loc[X_hpo.index]

                model_probas: Dict[str, pd.Series] = {}
                model_weights: Dict[str, float] = {}
                proba_df = pd.DataFrame(index=X_full.index)
                proba_df["y_true"] = y
                proba_df["chg_bp"] = chg_bp_full
                proba_df["regime"] = regime

                for mn in args.models:
                    t0 = time.time()
                    if args.skip_hpo:
                        best_params, best_score = {}, float("nan")
                    else:
                        print(f"          HPO  {mn:<7} ({args.hpo_trials} tr)...", end=" ", flush=True)
                        best_params, best_score = tune_model(
                            X_hpo, y_hpo, mn,
                            n_trials=args.hpo_trials,
                            n_splits=args.hpo_splits,
                            embargo=forward_days + 5,
                            timeout=args.hpo_timeout,
                            class_weight=class_weight,
                            verbose=False,
                        )
                        print(f"AUC={best_score:.3f} ({time.time()-t0:.1f}s)")
                    hpo_log[f"{tname}__{run_id}__{direction}__{mn}"] = {
                        "best_params": best_params, "best_auc": best_score,
                    }
                    model_weights[f"proba_{mn}"] = max(0.0,
                        (best_score - 0.5) if not np.isnan(best_score) else 0.0)

                    t1 = time.time()
                    proba, fi, folds = walk_forward_periodic(
                        X_full[feat_list], y, model_name=mn, params=best_params,
                        period=period_str,
                        start_oos_year=args.start_oos_year,
                        min_train_days=args.min_train_days,
                        embargo=forward_days + 5,
                        n_seeds=args.n_seeds,
                        class_weight=class_weight,
                        feature_list=feat_list,
                        verbose=False,
                    )
                    proba_df[f"proba_{mn}"] = proba
                    model_probas[f"proba_{mn}"] = proba

                    if mn == "lgbm" and not fi.empty:
                        fi2 = fi.copy()
                        fi2["target"] = tname; fi2["run_id"] = run_id
                        fi2["direction"] = direction
                        fi_records.append(fi2)
                    for fl in folds:
                        fold_log.append({
                            "target": tname, "run_id": run_id,
                            "direction": direction, "model": mn,
                            "period_label": fl.period_label,
                            "train_end": fl.train_end, "test_start": fl.test_start,
                            "test_end": fl.test_end,
                            "n_train": fl.n_train, "n_test": fl.n_test,
                            "pos_rate_train": fl.pos_rate_train,
                        })

                    m = evaluate_predictions(y, proba)
                    m.update({
                        "target": tname, "run_id": run_id,
                        "lookback_weeks": lb_w, "forward_weeks": fw_w,
                        "retrain_days": retrain_d, "target_thr_bp": thr_bp,
                        "regime_strength_min": min_str,
                        "direction": direction, "model": mn,
                        "hpo_best_auc": best_score,
                        "wf_time_s": float(time.time() - t1),
                    })
                    metrics_rows.append(m)
                    _print_metric_row(mn, m, extra=f"wf={m['wf_time_s']:.0f}s")
                    # heartbeat: 모델 1개 완료마다 진행상태 갱신
                    _write_progress(
                        phase="train", combo_no=run_id_counter,
                        total_combos=_total_combos,
                        ckpt_done=len(list(ckpt_dir.glob("*.pkl"))),
                        n_models=_n_models, target=tname, config=run_id,
                        models_done=args.models.index(mn) + 1,
                        last_model=f"{mn} auc={m.get('auc', float('nan')):.3f}")

                    yb = yearly_breakdown(y, proba)
                    if not yb.empty:
                        yb["target"] = tname; yb["run_id"] = run_id
                        yb["direction"] = direction; yb["model"] = mn
                        yearly_log.append(yb)

                # 앙상블
                if model_probas:
                    if args.ensemble_mode == "weighted" and any(w > 0 for w in model_weights.values()):
                        ens = weighted_ensemble(model_probas, model_weights)
                    else:
                        ens = weighted_ensemble(model_probas)
                    proba_df["proba_ensemble"] = ens
                    m = evaluate_predictions(y, ens)
                    m.update({
                        "target": tname, "run_id": run_id,
                        "lookback_weeks": lb_w, "forward_weeks": fw_w,
                        "retrain_days": retrain_d, "target_thr_bp": thr_bp,
                        "regime_strength_min": min_str,
                        "direction": direction, "model": "ensemble",
                        "ensemble_mode": args.ensemble_mode,
                    })
                    metrics_rows.append(m)
                    _print_metric_row("ensemble", m)

                    yb = yearly_breakdown(y, ens)
                    if not yb.empty:
                        yb["target"] = tname; yb["run_id"] = run_id
                        yb["direction"] = direction; yb["model"] = "ensemble"
                        yearly_log.append(yb)

                # proba_frames 에 wide 저장 (long → wide 변환을 위해 컬럼 prefix)
                pdfx = proba_df.copy()
                pdfx.columns = [f"{tname}__{run_id}__{direction}__{c}" for c in pdfx.columns]
                proba_frames.append(pdfx)

            outer_log.append({
                "run_id": run_id, "target": tname,
                "lookback_weeks": lb_w, "forward_weeks": fw_w,
                "retrain_days": retrain_d, "target_thr_bp": thr_bp,
                "regime_strength_min": min_str,
                "lookback_days": lookback_days, "forward_days": forward_days,
                "period": period_str,
                "n_persist": n_per,
            })

            # 체크포인트 저장: 이 조합이 추가한 분량만 묶어서 pkl로 (중단 내성)
            bundle = {
                "metrics": metrics_rows[_snap["metrics"]:],
                "fold":    fold_log[_snap["fold"]:],
                "yearly":  yearly_log[_snap["yearly"]:],
                "fi":      fi_records[_snap["fi"]:],
                "selfeat": selected_features_log[_snap["selfeat"]:],
                "outer":   outer_log[_snap["outer"]:],
                "proba":   proba_frames[_snap["proba"]:],
                "hpo":     {k: v for k, v in hpo_log.items()
                            if k.startswith(f"{tname}__{run_id}__")},
                "regime":  {f"{tname}__L{lb_w}W":
                            regime_history.get(f"{tname}__L{lb_w}W")},
            }
            with open(ckpt_file, "wb") as _f:
                pickle.dump(bundle, _f)

    # 4) 결과 저장 — 모든 체크포인트를 합쳐 산출물 생성(세션 무관 완료분 전부 반영)
    print(f"\n  [3/5] 산출물 저장 -> {out_dir}")
    ckpt_files = sorted(ckpt_dir.glob("*.pkl"))
    print(f"        체크포인트 {len(ckpt_files)}개 조합 병합")
    metrics_rows, fold_log, yearly_log, fi_records = [], [], [], []
    selected_features_log, outer_log, proba_frames = [], [], []
    hpo_log, regime_history = {}, {}
    for _cf in ckpt_files:
        try:
            with open(_cf, "rb") as _f:
                b = pickle.load(_f)
        except Exception as e:
            print(f"        [warn] ckpt 손상 skip: {_cf.name} ({e})")
            continue
        metrics_rows.extend(b["metrics"]);   fold_log.extend(b["fold"])
        yearly_log.extend(b["yearly"]);      fi_records.extend(b["fi"])
        selected_features_log.extend(b["selfeat"]); outer_log.extend(b["outer"])
        proba_frames.extend(b["proba"]);     hpo_log.update(b["hpo"])
        regime_history.update({k: v for k, v in b["regime"].items() if v is not None})

    pd.DataFrame(outer_log).to_csv(out_dir / "outer_configs.csv",
                                      index=False, encoding="utf-8-sig")

    (out_dir / "hpo_params.json").write_text(
        json.dumps(hpo_log, indent=2, default=str), encoding="utf-8")

    metric_cols = ["target", "run_id",
                   "lookback_weeks", "forward_weeks", "retrain_days",
                   "target_thr_bp", "regime_strength_min",
                   "direction", "model",
                   "n", "pos_rate", "accuracy", "precision", "recall", "f1",
                   "auc", "hit@60", "cov@60", "hit@70", "cov@70", "composite",
                   "hpo_best_auc", "wf_time_s", "ensemble_mode"]
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        mdf = mdf[[c for c in metric_cols if c in mdf.columns]]
        mdf = mdf.sort_values(["target", "direction", "run_id", "model"])\
                  .reset_index(drop=True)
        mdf.to_csv(out_dir / "metrics_by_run.csv",
                    index=False, encoding="utf-8-sig")

        # Top: target × direction 기준 composite 내림차순 (모델 단위, ensemble 포함)
        mdf["rank_in_group"] = mdf.groupby(["target", "direction"])["composite"]\
                                    .rank(ascending=False, method="min")
        top = mdf.sort_values(["target", "direction", "rank_in_group"])
        top.to_csv(out_dir / "metrics_top.csv",
                    index=False, encoding="utf-8-sig")

        # model_selection_template.csv (phase2 진입)
        # 각 (target, direction, run_id, model) 행. 자동 추천: rank_in_group == 1
        sel = mdf[mdf["model"] != "ensemble"].copy()
        sel["selected"] = (sel["rank_in_group"] == 1).astype(bool)
        sel = sel[["target", "direction", "run_id", "model", "selected",
                   "lookback_weeks", "forward_weeks", "retrain_days",
                   "target_thr_bp", "regime_strength_min",
                   "accuracy", "auc", "hit@60", "cov@60",
                   "hit@70", "cov@70", "composite", "rank_in_group"]]
        sel.to_csv(out_dir / "model_selection_template.csv",
                    index=False, encoding="utf-8-sig")

    if proba_frames:
        wide = pd.concat(proba_frames, axis=1)
        wide.to_parquet(out_dir / "predictions.parquet")

    if fi_records:
        fi_all = pd.concat(fi_records, ignore_index=True)
        fi_top = (fi_all.groupby("feature")["importance"].mean()
                  .sort_values(ascending=False).head(30)
                  .reset_index().rename(columns={"importance": "avg_importance"}))
        fi_top.to_csv(out_dir / "feature_importance.csv",
                      index=False, encoding="utf-8-sig")

    if selected_features_log:
        pd.DataFrame(selected_features_log).to_csv(
            out_dir / "selected_features.csv", index=False, encoding="utf-8-sig")

    if fold_log:
        pd.DataFrame(fold_log).to_csv(
            out_dir / "fold_log.csv", index=False, encoding="utf-8-sig")

    if yearly_log:
        ydf = pd.concat(yearly_log, ignore_index=True)
        ydf = ydf[["target", "run_id", "direction", "model", "year",
                   "n", "accuracy", "auc", "hit@60", "cov@60"]]
        ydf.to_csv(out_dir / "yearly_breakdown.csv",
                    index=False, encoding="utf-8-sig")

    if regime_history:
        rh = pd.concat({k: v for k, v in regime_history.items()}, axis=1)
        rh.to_parquet(out_dir / "regime_history.parquet")

    # 5) 콘솔 요약
    print(f"\n  [4/5] Top configs 요약 (CompositeScore 내림차순)")
    if metrics_rows:
        m = pd.DataFrame(metrics_rows)
        m = m[m["model"] != "ensemble"].copy()
        m["rank"] = m.groupby(["target", "direction"])["composite"]\
                       .rank(ascending=False, method="min")
        m = m[m["rank"] <= 3].sort_values(["target", "direction", "rank"])
        for (t, d), grp in m.groupby(["target", "direction"]):
            print(f"\n  {t}  [{d}]")
            for _, r in grp.iterrows():
                print(f"    {int(r['rank']):>2}. {r['model']:<7} run={r['run_id']:<35}"
                      f"  acc={r['accuracy']:.3f}  auc={r.get('auc',float('nan')):.3f}  "
                      f"hit60={r.get('hit@60',float('nan')):.3f} "
                      f"cov60={r.get('cov@60',float('nan')):.3f}  "
                      f"comp={r.get('composite',float('nan')):.3f}")

    print(f"\n  [5/5] 완료. 다음 단계:  python phase2_review.py")
    print(f"   결과 디렉터리: {out_dir}")
    _write_progress(phase="done", combo_no=_total_combos,
                    total_combos=_total_combos,
                    ckpt_done=len(list(ckpt_dir.glob("*.pkl"))),
                    n_models=_n_models, target="", config="",
                    models_done=_n_models, last_model="완료")
    try:
        _PID_FILE.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()

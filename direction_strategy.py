"""
direction_strategy.py
─────────────────────
KBOND ActiveQuant — KTB 금리 + 회사채 스프레드 방향성 예측 전략
(phase1~3 패턴 차용: 다중 모델 카탈로그 + Hit/Coverage 메트릭 + CompositeScore)

타겟 (모두 H일 후 부호 = sign(value[t+H] - value[t]))
─────────────────────────────────────────────────────
  ktb_3y, ktb_5y, ktb_10y         — 국고채 금리 방향 (상승=1, 하락=0)
  spread_aa_3y                    — (corp_aa_3y - ktb_3y) 방향
  spread_bbb_3y                   — (corp_bbb_3y - ktb_3y) 방향
  spread_kepco_5y                 — (kepco_5y - ktb_5y) 방향
  → 상승: 듀레이션 축소 / 크레딧 축소 / 한전채 회피
  → 하락: 듀레이션 확대 / 크레딧 확대 / 한전채 추가

모델 카탈로그 (phase1~3 차용)
─────────────────────────────
  lgbm    : LightGBM Classifier
  xgb     : XGBoost  Classifier
  rf      : RandomForest
  et      : ExtraTrees
  gbm     : sklearn GradientBoosting
  logreg  : LogisticRegression (StandardScaler + L2)

학습/검증
─────────
Walk-forward:
  min_train_days=750(약 3년), retrain_every=63(분기), embargo=H+5

메트릭 (모델별 + 앙상블)
────────────────────────
  AUC-ROC, Accuracy, Precision, Recall, F1
  Hit@K     : 신뢰도(=max(P, 1-P))≥K 인 표본에서의 정확도
  Coverage@K: 신뢰도≥K 표본 비율
  CompositeScore = AUC × (1 + Hit@60% × Coverage@60%)

산출물 (--output-dir, 기본 ./results/direction)
────────────────────────────────────────────
  metrics_by_model.csv     — target·H·모델별 전체 메트릭 (앙상블 포함)
  metrics_summary.csv      — 앙상블만 추린 요약
  leaderboard.csv          — CompositeScore 기준 모델 순위 (target·H 별)
  predictions.parquet      — 일자별 OOS 확률 + 라벨 + 변화량(bp)
  latest_signal.txt        — 현 시점 예측 + 의사결정 권고
  feature_importance.csv   — LightGBM 평균 importance (top 30)
  confusion_matrix.png     — H별 confusion matrix
  cum_accuracy.png         — 시간 흐름에 따른 누적 정확도

CLI
───
  python direction_strategy.py                          # 전체 (KTB + 스프레드) × H=10,20
  python direction_strategy.py --targets ktb_10y        # 단일 타겟
  python direction_strategy.py --targets ktb_10y spread_aa_3y
  python direction_strategy.py --horizons 10
  python direction_strategy.py --models lgbm xgb logreg # 모델 선택
  python direction_strategy.py --start 20140101
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# UTF-8 출력 (Windows 콘솔)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import (DEFAULT_CACHE, build_all_features, build_target_series,
                       load_cache, make_direction_change_bp_from_series,
                       make_direction_target_from_series)


PROJECT_DIR = Path(__file__).resolve().parent

ALL_TARGETS = [
    # KTB 절대금리
    "ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y",
    # KTB 장단기 스프레드
    "spread_ktb_3_10", "spread_ktb_5_30", "spread_ktb_10_30",
    # 회사채 스프레드
    "spread_aa_3y", "spread_bbb_3y", "spread_kepco_5y",
    # 미국
    "us_10y_xl", "us_ig_oas_xl",
    "us_spr_2_10", "us_spr_5_30", "us_spr_10_30",
]

HIT_KS = (0.60, 0.70)  # phase1~3 차용 (60%, 70% 신뢰 임계)


# ═══════════════════════════════════════════════════════════════
# 모델 팩토리
# ═══════════════════════════════════════════════════════════════

def _lgbm_factory():
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03,
        num_leaves=31, max_depth=-1,
        min_child_samples=30, reg_lambda=1.0,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
        objective="binary", n_jobs=-1, verbose=-1, random_state=42,
    )


def _xgb_factory():
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03,
        max_depth=5, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=42,
        verbosity=0,
    )


def _rf_factory():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_leaf=10, min_samples_split=20,
        max_features="sqrt",
        n_jobs=-1, random_state=42,
    )


def _et_factory():
    return ExtraTreesClassifier(
        n_estimators=400, max_depth=None,
        min_samples_leaf=10, min_samples_split=20,
        max_features="sqrt",
        n_jobs=-1, random_state=42,
    )


def _gbm_factory():
    return GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.05,
        max_depth=3, min_samples_leaf=20,
        subsample=0.85, random_state=42,
    )


def _logreg_factory():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(C=0.1, solver="liblinear", max_iter=2000, random_state=42)),
    ])


MODEL_FACTORIES = {
    "lgbm":   _lgbm_factory,
    "xgb":    _xgb_factory,
    "rf":     _rf_factory,
    "et":     _et_factory,
    "gbm":    _gbm_factory,
    "logreg": _logreg_factory,
}

# logreg는 NaN을 못 다루므로 별도 처리. 트리 계열 중 sklearn RF/ET/GBM도 NaN 못 다룸.
NAN_INTOLERANT = {"rf", "et", "gbm", "logreg"}


# ═══════════════════════════════════════════════════════════════
# Walk-forward predict
# ═══════════════════════════════════════════════════════════════

def _safe_fit_predict(factory, X_tr: pd.DataFrame, y_tr: pd.Series,
                      X_te: pd.DataFrame, model_name: str
                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """모델 학습 후 P(y=1) 반환. NaN 비허용 모델은 median으로 채움."""
    if model_name in NAN_INTOLERANT:
        med = X_tr.median(numeric_only=True)
        X_tr_f = X_tr.fillna(med).fillna(0.0)
        X_te_f = X_te.fillna(med).fillna(0.0)
        # 분산 0 컬럼 처리 (logreg)
        if model_name == "logreg":
            zv = X_tr_f.std(numeric_only=True) == 0
            if zv.any():
                cols = zv.index[zv]
                X_tr_f.loc[:, cols] = 0.0
                X_te_f.loc[:, cols] = 0.0
    else:
        X_tr_f, X_te_f = X_tr, X_te

    model = factory()
    model.fit(X_tr_f, y_tr.astype(int).values)
    proba = model.predict_proba(X_te_f)[:, 1]

    fi = None
    try:
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "named_steps") and "clf" in model.named_steps:
            coef = model.named_steps["clf"].coef_.ravel()
            fi = np.abs(coef)
    except Exception:
        fi = None

    return proba, fi


def walk_forward_predict(X: pd.DataFrame, y: pd.Series,
                          model_name: str,
                          min_train_days: int = 750,
                          retrain_every: int = 63,
                          embargo: int = 25,
                          verbose: bool = False
                          ) -> Tuple[pd.Series, pd.DataFrame]:
    """OOS 확률 시리즈와 feature importance DataFrame 리턴."""
    factory = MODEL_FACTORIES[model_name]

    valid = y.notna()
    X_v = X.loc[valid].copy()
    y_v = y.loc[valid].astype(int).copy()
    idx = X_v.index

    n = len(idx)
    if n <= min_train_days + retrain_every:
        return pd.Series(np.nan, index=X.index), pd.DataFrame()

    proba_all = pd.Series(np.nan, index=idx, dtype=float)
    fi_records: List[np.ndarray] = []
    fold_count = 0

    start = min_train_days
    while start < n:
        end_test = min(start + retrain_every, n)
        train_end = max(0, start - embargo)
        if train_end <= 50:
            start = end_test
            continue

        X_tr = X_v.iloc[:train_end]
        y_tr = y_v.iloc[:train_end]
        X_te = X_v.iloc[start:end_test]

        if y_tr.nunique() < 2:
            start = end_test
            continue

        try:
            proba, fi = _safe_fit_predict(factory, X_tr, y_tr, X_te, model_name)
        except Exception as e:
            if verbose:
                print(f"    [fold {fold_count} err] {e}")
            start = end_test
            continue

        proba_all.iloc[start:end_test] = proba
        if fi is not None and len(fi) == X.shape[1]:
            fi_records.append(fi)
        fold_count += 1
        start = end_test

    if verbose:
        print(f"    [{model_name}] folds={fold_count}, OOS coverage={proba_all.notna().sum()}/{n}")

    fi_df = pd.DataFrame()
    if fi_records:
        fi_mean = np.mean(np.stack(fi_records, axis=0), axis=0)
        fi_df = pd.DataFrame({"feature": X.columns, "importance": fi_mean})\
                  .sort_values("importance", ascending=False).reset_index(drop=True)

    return proba_all.reindex(X.index), fi_df


# ═══════════════════════════════════════════════════════════════
# 메트릭
# ═══════════════════════════════════════════════════════════════

def _hit_coverage(y_true: np.ndarray, proba: np.ndarray, k: float) -> Tuple[float, float]:
    """K=신뢰 임계. 신뢰도=max(P,1-P).
    Hit@K = 신뢰도≥K 표본에서의 정확도. Coverage@K = 그 표본 비율."""
    conf = np.maximum(proba, 1.0 - proba)
    pred = (proba >= 0.5).astype(int)
    mask = conf >= k
    cov = float(mask.mean()) if mask.size else 0.0
    if mask.sum() < 5:
        return float("nan"), cov
    hit = float((pred[mask] == y_true[mask]).mean())
    return hit, cov


def evaluate_predictions(y_true: pd.Series, proba: pd.Series,
                          thr: float = 0.5) -> Dict[str, float]:
    mask = proba.notna() & y_true.notna()
    if mask.sum() < 30:
        return {"n": int(mask.sum())}
    yt = y_true[mask].astype(int).values
    pp = proba[mask].values
    yp = (pp >= thr).astype(int)

    out = {
        "n":         int(mask.sum()),
        "pos_rate":  float(yt.mean()),
        "accuracy":  float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall":    float(recall_score(yt, yp, zero_division=0)),
        "f1":        float(f1_score(yt, yp, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(yt, pp))
    except Exception:
        out["auc"] = float("nan")

    for k in HIT_KS:
        hit, cov = _hit_coverage(yt, pp, k)
        out[f"hit@{int(k*100)}"] = hit
        out[f"cov@{int(k*100)}"] = cov

    # CompositeScore = AUC × (1 + Hit@60% × Cov@60%)  (phase1~3)
    auc = out.get("auc", float("nan"))
    h60 = out.get("hit@60", float("nan"))
    c60 = out.get("cov@60", float("nan"))
    if not (np.isnan(auc) or np.isnan(h60) or np.isnan(c60)):
        out["composite"] = float(auc * (1.0 + h60 * c60))
    else:
        out["composite"] = float("nan")

    return out


# ═══════════════════════════════════════════════════════════════
# 한 (target, horizon) 조합 실행
# ═══════════════════════════════════════════════════════════════

def run_one(X: pd.DataFrame, y: pd.Series, chg_bp: pd.Series,
             target_name: str, horizon: int,
             min_train_days: int, retrain_every: int,
             model_names: List[str],
             verbose: bool = True
             ) -> Tuple[pd.DataFrame, List[Dict], pd.DataFrame]:
    """(proba_df, metrics_rows, lgbm_fi) 리턴."""
    embargo = horizon + 5

    proba_df = pd.DataFrame(index=X.index)
    proba_df["y_true"] = y
    proba_df["chg_bp"] = chg_bp

    metrics_rows: List[Dict] = []
    lgbm_fi = pd.DataFrame()

    for mn in model_names:
        if verbose:
            print(f"  [{target_name} H={horizon}] {mn}...")
        proba, fi = walk_forward_predict(
            X, y, model_name=mn,
            min_train_days=min_train_days,
            retrain_every=retrain_every,
            embargo=embargo, verbose=verbose,
        )
        proba_df[f"proba_{mn}"] = proba
        m = evaluate_predictions(y, proba)
        m.update({"target": target_name, "horizon": horizon, "model": mn})
        metrics_rows.append(m)
        if verbose and "accuracy" in m:
            print(f"    acc={m['accuracy']:.3f} auc={m.get('auc',float('nan')):.3f} "
                  f"hit60={m.get('hit@60',float('nan')):.3f} cov60={m.get('cov@60',float('nan')):.3f} "
                  f"hit70={m.get('hit@70',float('nan')):.3f} cov70={m.get('cov@70',float('nan')):.3f} "
                  f"comp={m.get('composite',float('nan')):.3f}")
        if mn == "lgbm" and not fi.empty:
            lgbm_fi = fi.copy()

    # 앙상블 = 평균 확률
    proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
    if proba_cols:
        ens = proba_df[proba_cols].mean(axis=1)
        proba_df["proba_ensemble"] = ens
        m = evaluate_predictions(y, ens)
        m.update({"target": target_name, "horizon": horizon, "model": "ensemble"})
        metrics_rows.append(m)
        if verbose and "accuracy" in m:
            print(f"  [{target_name} H={horizon}] ensemble "
                  f"acc={m['accuracy']:.3f} auc={m.get('auc',float('nan')):.3f} "
                  f"hit60={m.get('hit@60',float('nan')):.3f} cov60={m.get('cov@60',float('nan')):.3f} "
                  f"comp={m.get('composite',float('nan')):.3f}")

    return proba_df, metrics_rows, lgbm_fi


# ═══════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrices(proba_dfs: Dict[Tuple[str, int], pd.DataFrame],
                              out_path: Path):
    items = sorted(proba_dfs.keys())
    if not items:
        return
    targets = sorted({k[0] for k in items})
    horizons = sorted({k[1] for k in items})
    fig, axes = plt.subplots(len(targets), len(horizons),
                              figsize=(4.0 * len(horizons), 3.4 * len(targets)),
                              squeeze=False)
    for i, t in enumerate(targets):
        for j, h in enumerate(horizons):
            ax = axes[i][j]
            df = proba_dfs.get((t, h))
            if df is None or "proba_ensemble" not in df.columns:
                ax.axis("off")
                continue
            mask = df["proba_ensemble"].notna() & df["y_true"].notna()
            if mask.sum() < 10:
                ax.set_title(f"{t} H={h} (insufficient)")
                ax.axis("off")
                continue
            yt = df.loc[mask, "y_true"].astype(int).values
            yp = (df.loc[mask, "proba_ensemble"].values >= 0.5).astype(int)
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            ax.imshow(cm, cmap="Blues")
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{cm[r,c]}", ha="center", va="center",
                            color="black", fontsize=11)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred DOWN", "pred UP"])
            ax.set_yticklabels(["true DOWN", "true UP"])
            ax.set_title(f"{t}  H={h}")
    fig.suptitle("Confusion Matrix (ensemble, thr=0.5)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_cum_accuracy(proba_dfs: Dict[Tuple[str, int], pd.DataFrame],
                       out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for (t, h), df in sorted(proba_dfs.items()):
        if "proba_ensemble" not in df.columns:
            continue
        mask = df["proba_ensemble"].notna() & df["y_true"].notna()
        if mask.sum() < 50:
            continue
        yt = df.loc[mask, "y_true"].astype(int).values
        yp = (df.loc[mask, "proba_ensemble"].values >= 0.5).astype(int)
        correct = (yt == yp).astype(float)
        cum_acc = pd.Series(correct, index=df.loc[mask].index).expanding().mean()
        ax.plot(cum_acc.index, cum_acc.values, label=f"{t} H={h}", linewidth=1.3)
    ax.axhline(0.5, ls="--", lw=0.7, color="gray")
    ax.set_ylim(0.35, 0.75)
    ax.set_ylabel("Cumulative accuracy (OOS)")
    ax.set_title("Direction prediction accuracy over time (ensemble)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 현재 시점 신호
# ═══════════════════════════════════════════════════════════════

def predict_latest(X: pd.DataFrame, y: pd.Series,
                    model_names: List[str]) -> Dict[str, float]:
    valid = y.notna()
    if valid.sum() < 200:
        return {}
    out: Dict[str, float] = {}
    probas = []
    latest_row = X.iloc[[-1]]
    for mn in model_names:
        factory = MODEL_FACTORIES[mn]
        try:
            X_tr = X.loc[valid]
            y_tr = y.loc[valid].astype(int)
            if mn in NAN_INTOLERANT:
                med = X_tr.median(numeric_only=True)
                X_tr_f = X_tr.fillna(med).fillna(0.0)
                X_te_f = latest_row.fillna(med).fillna(0.0)
                if mn == "logreg":
                    zv = X_tr_f.std(numeric_only=True) == 0
                    if zv.any():
                        cols = zv.index[zv]
                        X_tr_f.loc[:, cols] = 0.0
                        X_te_f.loc[:, cols] = 0.0
            else:
                X_tr_f = X_tr
                X_te_f = latest_row
            model = factory()
            model.fit(X_tr_f, y_tr.values)
            p = float(model.predict_proba(X_te_f)[:, 1][0])
        except Exception:
            p = float("nan")
        out[mn] = p
        if not np.isnan(p):
            probas.append(p)
    if probas:
        out["ensemble"] = float(np.mean(probas))
    return out


def format_signal_text(latest_signals: Dict[Tuple[str, int], Dict[str, float]],
                        latest_values: Dict[str, float],
                        latest_date: pd.Timestamp,
                        model_names: List[str],
                        neutral_band: Tuple[float, float] = (0.45, 0.55)
                        ) -> str:
    lines = []
    lines.append("━" * 78)
    lines.append(f"  KBOND Direction Signal  —  as of {latest_date.date()}")
    lines.append("━" * 78)
    lines.append("")
    lines.append("  P(UP) = H일 후 [금리 상승 / 스프레드 와이드닝] 확률")
    lines.append("  KTB   : P(UP) > 0.55 → 듀레이션 축소  /  < 0.45 → 듀레이션 확대")
    lines.append("  Spread: P(UP) > 0.55 → 크레딧 축소    /  < 0.45 → 크레딧 확대")
    lines.append("")
    targets = sorted({k[0] for k in latest_signals.keys()})
    horizons = sorted({k[1] for k in latest_signals.keys()})
    lo, hi = neutral_band

    for t in targets:
        v0 = latest_values.get(t, float("nan"))
        unit = "%" if t.startswith("ktb") else "% pt"
        lines.append(f"  ── {t}  (현재값 = {v0:.3f}{unit})")
        for h in horizons:
            sig = latest_signals.get((t, h), {})
            if not sig:
                lines.append(f"      H={h:>2}일  (예측 불가)")
                continue
            p_ens = sig.get("ensemble", float("nan"))
            parts = []
            for mn in model_names:
                p = sig.get(mn, float("nan"))
                parts.append(f"{mn}={p:.2f}")
            mp = " ".join(parts)
            if np.isnan(p_ens):
                rec = "예측불가"
            elif p_ens > hi:
                if t.startswith("ktb"):
                    rec = "▼ 듀레이션 축소"
                else:
                    rec = "▼ 크레딧 축소"
            elif p_ens < lo:
                if t.startswith("ktb"):
                    rec = "▲ 듀레이션 확대"
                else:
                    rec = "▲ 크레딧 확대"
            else:
                rec = "= 중립"
            lines.append(f"      H={h:>2}일  P(UP) ens={p_ens:.3f}  [{mp}]  → {rec}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="KTB·스프레드 방향성 예측 전략")
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS,
                    help=f"예측 타겟. 가능: {ALL_TARGETS}")
    ap.add_argument("--horizons", nargs="+", type=int, default=[10, 20],
                    help="예측 horizon (영업일)")
    ap.add_argument("--start", default="20140101",
                    help="피처/타겟 사용 시작일 (YYYYMMDD)")
    ap.add_argument("--min-train-days", type=int, default=750)
    ap.add_argument("--retrain-every", type=int, default=63)
    ap.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES.keys()),
                    choices=list(MODEL_FACTORIES.keys()))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--output-dir", default=str(PROJECT_DIR / "results" / "direction"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드 + 피처
    if verbose:
        print("━" * 78)
        print(f"  KTB Direction Strategy  start={args.start}  out={out_dir}")
        print(f"  targets : {args.targets}")
        print(f"  horizons: {args.horizons}")
        print(f"  models  : {args.models}")
        print("━" * 78)
        print("  [1/4] 캐시 로드 + 피처 빌드...")

    cache = load_cache(Path(args.cache_dir))
    X_full = build_all_features(cache=cache)
    start_dt = pd.to_datetime(args.start)
    X_full = X_full.loc[X_full.index >= start_dt]

    if verbose:
        print(f"        X shape: {X_full.shape}  range: {X_full.index[0].date()}~{X_full.index[-1].date()}")
        print(f"        NaN ratio (mean): {X_full.isna().mean().mean():.3f}")

    # 2) target × horizon 루프
    if verbose:
        print(f"\n  [2/4] Walk-forward 학습/평가 "
              f"({len(args.targets)}×{len(args.horizons)}×{len(args.models)} = "
              f"{len(args.targets)*len(args.horizons)*len(args.models)} 모델fit 단위)")

    all_metrics: List[Dict] = []
    all_proba: Dict[Tuple[str, int], pd.DataFrame] = {}
    all_fi: Dict[Tuple[str, int], pd.DataFrame] = {}
    target_series_cache: Dict[str, pd.Series] = {}

    for tname in args.targets:
        try:
            ts = build_target_series(cache, tname)
        except KeyError as e:
            print(f"    [skip] {tname}: {e}")
            continue
        target_series_cache[tname] = ts
        for h in args.horizons:
            y = make_direction_target_from_series(ts, h).reindex(X_full.index)
            chg = make_direction_change_bp_from_series(ts, h).reindex(X_full.index)
            proba_df, mrows, fi = run_one(
                X_full, y, chg, tname, h,
                min_train_days=args.min_train_days,
                retrain_every=args.retrain_every,
                model_names=args.models,
                verbose=verbose,
            )
            all_metrics.extend(mrows)
            all_proba[(tname, h)] = proba_df
            all_fi[(tname, h)]    = fi

    # 3) 산출물 저장
    if verbose:
        print(f"\n  [3/4] 산출물 저장 -> {out_dir}")

    metric_cols = ["target", "horizon", "model", "n", "pos_rate",
                    "accuracy", "precision", "recall", "f1", "auc",
                    "hit@60", "cov@60", "hit@70", "cov@70", "composite"]

    if all_metrics:
        mdf = pd.DataFrame(all_metrics)
        mdf = mdf[[c for c in metric_cols if c in mdf.columns]]
        mdf = mdf.sort_values(["target", "horizon", "model"]).reset_index(drop=True)

        # 전체 모델별
        mdf.to_csv(out_dir / "metrics_by_model.csv", index=False, encoding="utf-8-sig")

        # 앙상블만 요약
        ens = mdf[mdf["model"] == "ensemble"].reset_index(drop=True)
        ens.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

        # Leaderboard: target·H 별 모델 순위 (composite 내림차순)
        lead = mdf[mdf["model"] != "ensemble"].copy()
        lead["rank_in_group"] = lead.groupby(["target", "horizon"])["composite"]\
                                    .rank(ascending=False, method="min")
        lead = lead.sort_values(["target", "horizon", "rank_in_group"])
        lead.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")

    # predictions parquet
    if all_proba:
        big = []
        for (t, h), df in all_proba.items():
            d = df.copy()
            d.columns = pd.MultiIndex.from_product([[t], [h], d.columns])
            big.append(d)
        wide = pd.concat(big, axis=1)
        wide.columns = ["__".join([str(x) for x in col]) for col in wide.columns]
        wide.to_parquet(out_dir / "predictions.parquet")

    # LGBM feature importance — 모든 (target,H) 평균
    fi_rows = []
    for (t, h), fi in all_fi.items():
        if fi.empty:
            continue
        f = fi.copy(); f["target"] = t; f["horizon"] = h
        fi_rows.append(f)
    if fi_rows:
        fi_all = pd.concat(fi_rows, ignore_index=True)
        fi_top = (fi_all.groupby("feature")["importance"].mean()
                  .sort_values(ascending=False).head(30)
                  .reset_index().rename(columns={"importance": "avg_importance"}))
        fi_top.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    try:
        plot_confusion_matrices(all_proba, out_dir / "confusion_matrix.png")
        plot_cum_accuracy(all_proba, out_dir / "cum_accuracy.png")
    except Exception as e:
        print(f"    [plot err] {e}")

    # 4) 현 시점 신호
    if verbose:
        print(f"\n  [4/4] 현 시점 신호 생성")

    latest_values: Dict[str, float] = {}
    for tname, ts in target_series_cache.items():
        latest_values[tname] = float(ts.dropna().iloc[-1])

    latest_signals: Dict[Tuple[str, int], Dict[str, float]] = {}
    for tname, ts in target_series_cache.items():
        for h in args.horizons:
            y = make_direction_target_from_series(ts, h).reindex(X_full.index)
            sig = predict_latest(X_full, y, args.models)
            latest_signals[(tname, h)] = sig

    text = format_signal_text(latest_signals, latest_values, X_full.index[-1], args.models)
    (out_dir / "latest_signal.txt").write_text(text, encoding="utf-8")
    print()
    print(text)

    # 콘솔 leaderboard 출력
    if all_metrics:
        print("\n" + "━" * 78)
        print("  LEADERBOARD  (target · H 별 모델 순위, CompositeScore 내림차순)")
        print("━" * 78)
        lead = pd.DataFrame(all_metrics)
        lead = lead[lead["model"] != "ensemble"].copy()
        lead["rank"] = lead.groupby(["target", "horizon"])["composite"].rank(ascending=False, method="min")
        lead = lead.sort_values(["target", "horizon", "rank"])
        for (t, h), grp in lead.groupby(["target", "horizon"]):
            print(f"\n  {t}  H={h}")
            for _, r in grp.iterrows():
                print(f"    {int(r['rank']):>2}. {r['model']:<8} "
                      f"acc={r['accuracy']:.3f}  auc={r.get('auc',float('nan')):.3f}  "
                      f"hit60={r.get('hit@60',float('nan')):.3f} "
                      f"cov60={r.get('cov@60',float('nan')):.3f}  "
                      f"hit70={r.get('hit@70',float('nan')):.3f} "
                      f"cov70={r.get('cov@70',float('nan')):.3f}  "
                      f"comp={r.get('composite',float('nan')):.3f}")

        # 앙상블만 별도
        print("\n" + "━" * 78)
        print("  ENSEMBLE 요약 (모델 평균 확률)")
        print("━" * 78)
        ens = pd.DataFrame(all_metrics)
        ens = ens[ens["model"] == "ensemble"].sort_values(["target", "horizon"])
        for _, r in ens.iterrows():
            print(f"  {r['target']:<16} H={int(r['horizon']):>2}  "
                  f"n={int(r['n']):>4}  acc={r['accuracy']:.3f}  auc={r.get('auc',float('nan')):.3f}  "
                  f"hit60={r.get('hit@60',float('nan')):.3f} cov60={r.get('cov@60',float('nan')):.3f}  "
                  f"comp={r.get('composite',float('nan')):.3f}")

        print(f"\n  결과 디렉터리: {out_dir}")


if __name__ == "__main__":
    main()

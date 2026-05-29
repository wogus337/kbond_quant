"""
core.py
───────
KBOND ActiveQuant — 공유 유틸 (phase1/2/3 공통)

제공 기능
─────────
1. 모델 카탈로그 (lgbm, xgb, rf, et, gbm, logreg, lstm) + 파라미터 오버라이드
   - 모든 분류기에 class_weight='balanced' 적용 (xgb는 scale_pos_weight, gbm은 sample_weight)
2. 타겟 빌더: noise-band 적용 (|Δ| < band_bp 인 일자는 NaN으로 마스킹)
3. 주기적 walk-forward (year | quarter)
4. Optuna HPO (TimeSeriesSplit 기반, scoring=AUC)
5. 메트릭: Accuracy/AUC/Hit@K/Cov@K/CompositeScore
6. 시드 다중화 (mini-bagging)
7. 피처 선택 (LGBM importance 기반 top-K)
8. 가중 앙상블
9. 고신뢰 거래 백테스트
10. 연도별 hit / acc 분해
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# 서브프로세스(joblib loky 워커)까지 경고 억제되도록 env 먼저 설정
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning")

# 현재 프로세스: UserWarning 카테고리 광범위 차단 (sklearn parallel.delayed 등 포함)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np
import pandas as pd

from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import xgboost as xgb

# 경고 일괄 억제
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning,      module="optuna")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="optuna")
warnings.filterwarnings("ignore", category=FutureWarning,    module="pandas")
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=r".*DataFrame is highly fragmented.*")
# sklearn 카테고리 직접 잡기 (module 매칭이 불안정한 환경 대응)
try:
    from sklearn.exceptions import (ConvergenceWarning,
                                     UndefinedMetricWarning,
                                     FitFailedWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=UserWarning,      module="sklearn")
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")
warnings.filterwarnings("ignore", message=r".*should be used with `sklearn\.utils\.parallel\.Parallel`.*")
warnings.filterwarnings("ignore", message=r".*Only one class is present in y_true.*")
warnings.filterwarnings("ignore", message=r".*KMeans is known to have a memory leak.*")
# joblib / loky CPU count probe
warnings.filterwarnings("ignore", message=r".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning,      module="joblib")
# 환경변수로 loky 의 wmic probe 비활성화
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 8))


HIT_KS = (0.60, 0.70)
CONF_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)


# ═══════════════════════════════════════════════════════════════
# 모델 카탈로그 (기본 파라미터)
# ═══════════════════════════════════════════════════════════════

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "lgbm": dict(
        n_estimators=400, learning_rate=0.03,
        num_leaves=31, max_depth=-1,
        min_child_samples=30, reg_lambda=1.0,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
    ),
    "xgb": dict(
        n_estimators=400, learning_rate=0.03,
        max_depth=5, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.8,
        reg_lambda=1.0,
    ),
    "rf": dict(
        n_estimators=300, max_depth=None,
        min_samples_leaf=10, min_samples_split=20,
        max_features="sqrt",
    ),
    "et": dict(
        n_estimators=400, max_depth=None,
        min_samples_leaf=10, min_samples_split=20,
        max_features="sqrt",
    ),
    "gbm": dict(
        n_estimators=120, learning_rate=0.05,
        max_depth=3, min_samples_leaf=20,
        subsample=0.85,
    ),
    "logreg": dict(C=0.1),
    "lstm": dict(
        seq_len=30, hidden=32, epochs=20, batch=128, lr=1e-3, dropout=0.2,
    ),
}

# NaN 비허용
NAN_INTOLERANT = {"rf", "et", "gbm", "logreg", "lstm"}

# 시퀀스 모델 (학습/예측 시 시계열 윈도우 필요)
SEQUENCE_MODELS = {"lstm"}


def build_model(name: str, params: Optional[Dict[str, Any]] = None,
                  class_weight: bool = True, random_state: int = 42):
    """name + 오버라이드 params → fit/predict_proba 가능한 추정기 반환.
       class_weight=True 면 lgbm/rf/et/logreg는 class_weight='balanced',
       xgb는 fit 시점 scale_pos_weight, gbm은 fit 시점 sample_weight 로 처리.
    """
    p = dict(DEFAULT_PARAMS.get(name, {}))
    if params:
        p.update(params)

    if name == "lgbm":
        if class_weight:
            p.setdefault("class_weight", "balanced")
        return lgb.LGBMClassifier(
            objective="binary", n_jobs=-1, verbose=-1, random_state=random_state, **p,
        )
    if name == "xgb":
        # scale_pos_weight 는 _fit_predict 안에서 동적 설정
        return xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=random_state, verbosity=0, **p,
        )
    if name == "rf":
        if class_weight:
            p.setdefault("class_weight", "balanced")
        return RandomForestClassifier(n_jobs=-1, random_state=random_state, **p)
    if name == "et":
        if class_weight:
            p.setdefault("class_weight", "balanced")
        return ExtraTreesClassifier(n_jobs=-1, random_state=random_state, **p)
    if name == "gbm":
        # gbm은 class_weight 미지원. sample_weight를 _fit_predict 안에서 처리
        return GradientBoostingClassifier(random_state=random_state, **p)
    if name == "logreg":
        C = p.pop("C", 0.1)
        clf_kwargs = dict(C=C, solver="liblinear", max_iter=2000, random_state=random_state)
        if class_weight:
            clf_kwargs["class_weight"] = "balanced"
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(**clf_kwargs)),
        ])
    if name == "lstm":
        return LSTMClassifier(random_state=random_state, **p)
    raise KeyError(f"unknown model: {name}")


def suggest_params(name: str, trial) -> Dict[str, Any]:
    """Optuna trial → 모델별 하이퍼파라미터 dict."""
    if name == "lgbm":
        return dict(
            n_estimators    = trial.suggest_int("n_estimators", 200, 700, step=50),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            num_leaves      = trial.suggest_int("num_leaves", 15, 95, step=8),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 80),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.0, 5.0),
            subsample       = trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
    if name == "xgb":
        return dict(
            n_estimators    = trial.suggest_int("n_estimators", 200, 700, step=50),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 12),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.0, 5.0),
            subsample       = trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
    if name == "rf":
        return dict(
            n_estimators     = trial.suggest_int("n_estimators", 150, 500, step=50),
            max_depth        = trial.suggest_int("max_depth", 5, 30),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
            min_samples_split= trial.suggest_int("min_samples_split", 10, 40),
            max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        )
    if name == "et":
        return dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 600, step=50),
            max_depth        = trial.suggest_int("max_depth", 5, 30),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
            min_samples_split= trial.suggest_int("min_samples_split", 10, 40),
            max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        )
    if name == "gbm":
        # 80~240 (step=20 으로 나누어떨어지게)
        return dict(
            n_estimators    = trial.suggest_int("n_estimators", 80, 240, step=20),
            learning_rate   = trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            max_depth       = trial.suggest_int("max_depth", 2, 5),
            min_samples_leaf= trial.suggest_int("min_samples_leaf", 10, 40),
            subsample       = trial.suggest_float("subsample", 0.7, 1.0),
        )
    if name == "logreg":
        return dict(C = trial.suggest_float("C", 1e-3, 10.0, log=True))
    if name == "lstm":
        return dict(
            hidden  = trial.suggest_categorical("hidden", [16, 32, 64]),
            dropout = trial.suggest_float("dropout", 0.0, 0.4),
            lr      = trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            epochs  = trial.suggest_int("epochs", 10, 30, step=5),
        )
    raise KeyError(f"unknown model: {name}")


# ═══════════════════════════════════════════════════════════════
# LSTM 분류기 (torch 기반, sklearn 호환 인터페이스)
# ═══════════════════════════════════════════════════════════════

class LSTMClassifier:
    """1-layer LSTM 분류기. fit/predict_proba.
    내부에서 X(2D pandas/np) 를 시퀀스(seq_len)로 자동 변환.
    """
    def __init__(self, seq_len: int = 30, hidden: int = 32,
                  epochs: int = 20, batch: int = 128,
                  lr: float = 1e-3, dropout: float = 0.2,
                  random_state: int = 42):
        self.seq_len = int(seq_len)
        self.hidden  = int(hidden)
        self.epochs  = int(epochs)
        self.batch   = int(batch)
        self.lr      = float(lr)
        self.dropout = float(dropout)
        self.random_state = int(random_state)
        self.feature_importances_ = None  # sklearn 호환 placeholder
        self._model = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std:  Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    @staticmethod
    def _to_sequences(X2d: np.ndarray, seq_len: int) -> np.ndarray:
        """(N, F) → (N, seq_len, F). 처음 seq_len-1 시점은 자기 자신 padding."""
        N, F = X2d.shape
        if N == 0:
            return np.zeros((0, seq_len, F), dtype=np.float32)
        # pad: 앞에 (seq_len-1)개 첫 행 복제
        pad = np.repeat(X2d[:1], repeats=seq_len - 1, axis=0)
        padded = np.concatenate([pad, X2d], axis=0)  # (N+seq_len-1, F)
        # sliding
        out = np.lib.stride_tricks.sliding_window_view(padded, window_shape=seq_len, axis=0)
        # shape: (N, F, seq_len) → transpose → (N, seq_len, F)
        out = np.transpose(out, (0, 2, 1))
        return out.astype(np.float32)

    def fit(self, X, y, sample_weight=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X2d = np.asarray(X, dtype=np.float32)
        y1d = np.asarray(y).astype(np.float32).ravel()
        self._n_features = X2d.shape[1]

        # 표준화 (학습 분포 기준)
        self._scaler_mean = X2d.mean(axis=0)
        self._scaler_std  = X2d.std(axis=0)
        self._scaler_std[self._scaler_std == 0] = 1.0
        Xn = (X2d - self._scaler_mean) / self._scaler_std

        Xseq = self._to_sequences(Xn, self.seq_len)  # (N, T, F)

        device = "cpu"  # 사용자 환경 cuda False
        Xt = torch.from_numpy(Xseq)
        yt = torch.from_numpy(y1d).unsqueeze(-1)

        ds = TensorDataset(Xt, yt)
        ld = DataLoader(ds, batch_size=self.batch, shuffle=True, drop_last=False)

        class _Net(nn.Module):
            def __init__(self, F, hidden, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size=F, hidden_size=hidden,
                                     num_layers=1, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.fc   = nn.Linear(hidden, 1)
            def forward(self, x):
                _, (h, _) = self.lstm(x)
                z = self.drop(h.squeeze(0))
                return self.fc(z)

        net = _Net(self._n_features, self.hidden, self.dropout).to(device)
        # class imbalance에 대응: pos_weight
        pos = float(y1d.sum())
        neg = float(len(y1d) - pos)
        pos_w = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        net.train()
        for _ in range(self.epochs):
            for xb, yb in ld:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                logit = net(xb)
                loss = loss_fn(logit, yb)
                loss.backward()
                opt.step()
        self._model = net.eval()
        return self

    def predict_proba(self, X):
        import torch
        if self._model is None:
            raise RuntimeError("LSTM not fitted")
        X2d = np.asarray(X, dtype=np.float32)
        Xn = (X2d - self._scaler_mean) / self._scaler_std
        Xseq = self._to_sequences(Xn, self.seq_len)
        with torch.no_grad():
            logit = self._model(torch.from_numpy(Xseq)).numpy().ravel()
        p1 = 1.0 / (1.0 + np.exp(-logit))
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════
# 타겟 빌더
# ═══════════════════════════════════════════════════════════════

def make_direction_target_banded(s: pd.Series, horizon: int,
                                   noise_band_bp: float = 0.0) -> pd.Series:
    """H일 후 변화 부호. |Δ_bp| < noise_band_bp 는 NaN으로 마스킹."""
    delta = s.shift(-horizon) - s
    delta_bp = delta * 100.0
    target = (delta >= 0).astype(float)
    target[delta.isna()] = np.nan
    if noise_band_bp > 0:
        target[delta_bp.abs() < noise_band_bp] = np.nan
    return target.rename(f"y_{s.name}_h{horizon}")


def make_direction_change_bp(s: pd.Series, horizon: int) -> pd.Series:
    return ((s.shift(-horizon) - s) * 100.0).rename(f"chg_bp_{s.name}_h{horizon}")


# ═══════════════════════════════════════════════════════════════
# NaN 안전 처리 + fit/predict
# ═══════════════════════════════════════════════════════════════

def _fillna_for_model(X_tr: pd.DataFrame, X_te: pd.DataFrame,
                       model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if model_name not in NAN_INTOLERANT:
        return X_tr, X_te
    med = X_tr.median(numeric_only=True)
    X_tr_f = X_tr.fillna(med).fillna(0.0)
    X_te_f = X_te.fillna(med).fillna(0.0)
    if model_name == "logreg":
        zv = X_tr_f.std(numeric_only=True) == 0
        if zv.any():
            cols = zv.index[zv]
            X_tr_f = X_tr_f.copy(); X_te_f = X_te_f.copy()
            X_tr_f.loc[:, cols] = 0.0
            X_te_f.loc[:, cols] = 0.0
    return X_tr_f, X_te_f


def _fit_predict(model, X_tr: pd.DataFrame, y_tr: pd.Series,
                  X_te: pd.DataFrame, model_name: str = ""
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    y_arr = y_tr.astype(int).values
    # XGB: scale_pos_weight
    if model_name == "xgb":
        pos = float(y_arr.sum()); neg = float(len(y_arr) - pos)
        if pos > 0 and neg > 0:
            try:
                model.set_params(scale_pos_weight=neg / pos)
            except Exception:
                pass
    # GBM: sample_weight = balanced
    if model_name == "gbm":
        try:
            sw = compute_sample_weight("balanced", y_arr)
            model.fit(X_tr, y_arr, sample_weight=sw)
        except Exception:
            model.fit(X_tr, y_arr)
    else:
        model.fit(X_tr, y_arr)

    proba = model.predict_proba(X_te)[:, 1]
    fi = None
    try:
        if hasattr(model, "feature_importances_") and getattr(model, "feature_importances_") is not None:
            fi = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
            fi = np.abs(model.named_steps["clf"].coef_.ravel())
    except Exception:
        fi = None
    return proba, fi


# ═══════════════════════════════════════════════════════════════
# 피처 선택 (LGBM importance 기반 top-K)
# ═══════════════════════════════════════════════════════════════

def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 80
                          ) -> List[str]:
    valid = y.notna()
    if valid.sum() < 200:
        return list(X.columns)
    X_v = X.loc[valid].fillna(X.median(numeric_only=True)).fillna(0.0)
    y_v = y.loc[valid].astype(int)
    if y_v.nunique() < 2:
        return list(X.columns)
    quick = build_model("lgbm",
                          {"n_estimators": 200, "learning_rate": 0.05,
                           "num_leaves": 31, "min_child_samples": 30,
                           "class_weight": "balanced"},
                          class_weight=False)
    quick.fit(X_v.values, y_v.values)
    imp = pd.Series(quick.feature_importances_, index=X.columns)
    return imp.nlargest(max(10, k)).index.tolist()


# ═══════════════════════════════════════════════════════════════
# 주기적 Walk-forward (year | quarter) + seed bagging
# ═══════════════════════════════════════════════════════════════

@dataclass
class FoldLog:
    period_label: str       # "2021", "2021Q1" 등
    train_start:  pd.Timestamp
    train_end:    pd.Timestamp
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp
    n_train: int
    n_test:  int
    pos_rate_train: float


def _period_boundaries(idx: pd.DatetimeIndex, start_oos_year: int,
                        period: str) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """OOS 기간을 (label, start, end) 리스트로 분할."""
    if idx.empty:
        return []
    last = idx[-1]
    bounds: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if period == "year":
        for y in range(start_oos_year, last.year + 1):
            start = pd.Timestamp(f"{y}-01-01")
            end   = pd.Timestamp(f"{y}-12-31")
            if start > last:
                break
            bounds.append((f"{y}", start, min(end, last)))
    elif period == "quarter":
        for y in range(start_oos_year, last.year + 1):
            for q, (mstart, mend) in enumerate([(1,3),(4,6),(7,9),(10,12)], start=1):
                start = pd.Timestamp(f"{y}-{mstart:02d}-01")
                end   = pd.Timestamp(f"{y}-{mend:02d}-{28 if mend==2 else 30 if mend in (4,6,9,11) else 31}")
                if start > last:
                    break
                bounds.append((f"{y}Q{q}", start, min(end, last)))
    else:
        raise ValueError(f"unknown period: {period}")
    return bounds


def walk_forward_periodic(X: pd.DataFrame, y: pd.Series,
                            model_name: str,
                            params: Optional[Dict[str, Any]] = None,
                            period: str = "quarter",
                            start_oos_year: int = 2018,
                            min_train_days: int = 750,
                            embargo: int = 25,
                            n_seeds: int = 1,
                            class_weight: bool = True,
                            feature_list: Optional[List[str]] = None,
                            verbose: bool = False,
                            ) -> Tuple[pd.Series, pd.DataFrame, List[FoldLog]]:
    """매 period(year | quarter) 경계마다 재학습. n_seeds>1 이면 다중 시드 평균(bagging)."""
    valid = y.notna()
    X_v = X.loc[valid].copy()
    y_v = y.loc[valid].astype(int).copy()
    idx = X_v.index

    if feature_list:
        feature_list = [f for f in feature_list if f in X_v.columns]
        if feature_list:
            X_v = X_v[feature_list]
            X = X[feature_list]

    if len(idx) < min_train_days + 30:
        return pd.Series(np.nan, index=X.index), pd.DataFrame(), []

    bounds = _period_boundaries(idx, start_oos_year, period)
    proba_all = pd.Series(np.nan, index=idx, dtype=float)
    fi_records: List[np.ndarray] = []
    fold_logs: List[FoldLog] = []

    for label, p_start, p_end in bounds:
        test_mask = (idx >= p_start) & (idx <= p_end)
        if test_mask.sum() < 3:
            continue
        # train end = last index < p_start
        train_pos = np.where(idx < p_start)[0]
        if train_pos.size < min_train_days:
            continue
        last_train_loc = train_pos[-1]
        train_end_loc = max(0, last_train_loc - embargo)
        if train_end_loc < min_train_days:
            continue

        X_tr_full = X_v.iloc[:train_end_loc + 1]
        y_tr = y_v.iloc[:train_end_loc + 1]
        X_te_full = X_v.loc[test_mask]
        if y_tr.nunique() < 2:
            continue

        X_tr, X_te = _fillna_for_model(X_tr_full, X_te_full, model_name)

        # seed bagging
        seeds = list(range(n_seeds)) if n_seeds > 1 else [42]
        proba_stack = []
        fi_stack = []
        try:
            for sd in seeds:
                model = build_model(model_name, params,
                                       class_weight=class_weight,
                                       random_state=42 + sd)
                p_arr, fi = _fit_predict(model, X_tr.values if model_name in SEQUENCE_MODELS else X_tr,
                                            y_tr,
                                            X_te.values if model_name in SEQUENCE_MODELS else X_te,
                                            model_name=model_name)
                proba_stack.append(p_arr)
                if fi is not None and len(fi) == X_tr.shape[1]:
                    fi_stack.append(fi)
        except Exception as e:
            if verbose:
                print(f"    [{model_name} {label} err] {e}")
            continue

        proba_mean = np.mean(np.stack(proba_stack, axis=0), axis=0)
        proba_all.loc[X_te_full.index] = proba_mean
        if fi_stack:
            fi_records.append(np.mean(np.stack(fi_stack, axis=0), axis=0))

        fold_logs.append(FoldLog(
            period_label=label,
            train_start=idx[0], train_end=idx[train_end_loc],
            test_start=X_te_full.index[0], test_end=X_te_full.index[-1],
            n_train=len(X_tr_full), n_test=len(X_te_full),
            pos_rate_train=float(y_tr.mean()),
        ))

    if verbose:
        print(f"    [{model_name}] folds={len(fold_logs)} "
              f"OOS coverage={proba_all.notna().sum()}/{len(idx)}")

    fi_df = pd.DataFrame()
    if fi_records:
        fi_mean = np.mean(np.stack(fi_records, axis=0), axis=0)
        cols = X.columns
        fi_df = pd.DataFrame({"feature": cols, "importance": fi_mean})\
                  .sort_values("importance", ascending=False).reset_index(drop=True)
    return proba_all.reindex(X.index), fi_df, fold_logs


# Backwards-compat: walk_forward_annual = walk_forward_periodic(period='year')
def walk_forward_annual(X, y, model_name, params=None,
                          start_oos_year=2018, min_train_days=750,
                          embargo=25, verbose=False):
    return walk_forward_periodic(X, y, model_name, params,
                                    period="year",
                                    start_oos_year=start_oos_year,
                                    min_train_days=min_train_days,
                                    embargo=embargo, verbose=verbose)


# ═══════════════════════════════════════════════════════════════
# Optuna HPO (scoring = AUC, n_splits CV)
# ═══════════════════════════════════════════════════════════════

def tune_model(X: pd.DataFrame, y: pd.Series, model_name: str,
                n_trials: int = 50,
                n_splits: int = 4,
                embargo: int = 25,
                timeout: Optional[int] = None,
                class_weight: bool = True,
                verbose: bool = False
                ) -> Tuple[Dict[str, Any], float]:
    """학습용 데이터(OOS 이전 구간) 위에 TimeSeriesSplit 으로 HPO.
    score = mean AUC over splits.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    valid = y.notna()
    X_v = X.loc[valid].copy()
    y_v = y.loc[valid].astype(int).copy()
    if len(X_v) < 300:
        return {}, float("nan")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo)

    def objective(trial):
        params = suggest_params(model_name, trial)
        scores = []
        for tr_idx, te_idx in tscv.split(X_v):
            if len(tr_idx) < 100 or len(te_idx) < 30:
                continue
            X_tr = X_v.iloc[tr_idx]; y_tr = y_v.iloc[tr_idx]
            X_te = X_v.iloc[te_idx]; y_te = y_v.iloc[te_idx]
            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                continue
            X_tr_f, X_te_f = _fillna_for_model(X_tr, X_te, model_name)
            try:
                model = build_model(model_name, params, class_weight=class_weight)
                X_tr_in = X_tr_f.values if model_name in SEQUENCE_MODELS else X_tr_f
                X_te_in = X_te_f.values if model_name in SEQUENCE_MODELS else X_te_f
                proba, _ = _fit_predict(model, X_tr_in, y_tr, X_te_in, model_name=model_name)
            except Exception:
                continue  # 해당 폴드 skip; 다른 폴드 있을 수 있음
            yt = y_te.values
            try:
                auc = float(roc_auc_score(yt, proba))
            except Exception:
                continue
            scores.append(auc)
        if not scores:
            # 모든 폴드가 단일 클래스/실패 → 중립 점수 반환 (trial 완료 처리)
            return 0.5
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                    show_progress_bar=False, catch=(Exception,))
    try:
        best = study.best_params
        best_score = float(study.best_value)
    except (ValueError, AttributeError):
        # 모든 trial 실패 시 — 기본 파라미터로 폴백
        best = {}
        best_score = float("nan")
    if verbose:
        print(f"    [HPO {model_name}] best AUC={best_score:.4f} params={best}")
    return best, best_score


# ═══════════════════════════════════════════════════════════════
# 메트릭
# ═══════════════════════════════════════════════════════════════

def _hit_coverage(y_true: np.ndarray, proba: np.ndarray, k: float) -> Tuple[float, float]:
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
        h, c = _hit_coverage(yt, pp, k)
        out[f"hit@{int(k*100)}"] = h
        out[f"cov@{int(k*100)}"] = c
    auc = out.get("auc", float("nan"))
    h60 = out.get("hit@60", float("nan"))
    c60 = out.get("cov@60", float("nan"))
    if not (np.isnan(auc) or np.isnan(h60) or np.isnan(c60)):
        out["composite"] = float(auc * (1.0 + h60 * c60))
    else:
        out["composite"] = float("nan")
    return out


def yearly_breakdown(y_true: pd.Series, proba: pd.Series) -> pd.DataFrame:
    """연도별 hit rate / accuracy / AUC."""
    df = pd.DataFrame({"y": y_true, "p": proba}).dropna()
    if df.empty:
        return pd.DataFrame()
    df["year"] = df.index.year
    df["pred"] = (df["p"] >= 0.5).astype(int)
    rows = []
    for yr, grp in df.groupby("year"):
        n = len(grp)
        acc = float((grp["pred"] == grp["y"]).mean())
        try:
            auc = float(roc_auc_score(grp["y"].astype(int), grp["p"]))
        except Exception:
            auc = float("nan")
        h60, c60 = _hit_coverage(grp["y"].astype(int).values,
                                  grp["p"].values, 0.60)
        rows.append({"year": int(yr), "n": n, "accuracy": acc, "auc": auc,
                     "hit@60": h60, "cov@60": c60})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 가중 앙상블
# ═══════════════════════════════════════════════════════════════

def weighted_ensemble(probas: Dict[str, pd.Series],
                       weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """모델별 확률 시리즈 + 가중치 → 가중 평균. weights None 또는 모두 0이면 단순 평균."""
    keys = [k for k, s in probas.items() if s is not None]
    if not keys:
        return pd.Series(dtype=float)
    df = pd.concat([probas[k].rename(k) for k in keys], axis=1)
    if weights is None:
        return df.mean(axis=1)
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in keys}
    s = sum(w.values())
    if s <= 0:
        return df.mean(axis=1)
    arr = np.zeros(len(df))
    for k in keys:
        arr = arr + df[k].fillna(np.nan).values * (w[k] / s)
    # NaN 처리: 어떤 행이라도 모든 모델이 NaN이면 NaN, 일부만 NaN이면 가중치 재조정
    # 간단화: 단순 가중 평균 (NaN 있으면 NaN 전파)
    return pd.Series(arr, index=df.index)


# ═══════════════════════════════════════════════════════════════
# 고신뢰 거래 백테스트
# ═══════════════════════════════════════════════════════════════

def high_conf_backtest(proba: pd.Series, chg_bp: pd.Series,
                        thresholds: Tuple[float, ...] = CONF_THRESHOLDS
                        ) -> pd.DataFrame:
    mask_all = proba.notna() & chg_bp.notna()
    if mask_all.sum() < 30:
        return pd.DataFrame()
    pp = proba[mask_all].values
    cb = chg_bp[mask_all].values
    conf = np.maximum(pp, 1 - pp)
    pred_up = (pp >= 0.5)
    actual_up = (cb >= 0)
    correct = (pred_up == actual_up)
    abs_bp = np.abs(cb)
    pnl = np.where(correct, abs_bp, -abs_bp)
    rows = []
    for thr in thresholds:
        sel = conf >= thr
        n = int(sel.sum())
        if n < 5:
            rows.append({"threshold": thr, "n_trades": n,
                         "hit_rate": float("nan"),
                         "mean_pnl_bp": float("nan"),
                         "cum_pnl_bp": float("nan"),
                         "mean_abs_move_bp": float("nan")})
            continue
        rows.append({"threshold": thr, "n_trades": n,
                     "hit_rate": float(correct[sel].mean()),
                     "mean_pnl_bp": float(pnl[sel].mean()),
                     "cum_pnl_bp": float(pnl[sel].sum()),
                     "mean_abs_move_bp": float(abs_bp[sel].mean())})
    return pd.DataFrame(rows)


def cum_pnl_series(proba: pd.Series, chg_bp: pd.Series,
                    threshold: float = 0.60) -> pd.Series:
    mask = proba.notna() & chg_bp.notna()
    pp = proba[mask]; cb = chg_bp[mask]
    conf = np.maximum(pp, 1 - pp)
    sel = conf >= threshold
    pp = pp[sel]; cb = cb[sel]
    if pp.empty:
        return pd.Series(dtype=float)
    pred_up = (pp >= 0.5)
    actual_up = (cb >= 0)
    correct = (pred_up == actual_up)
    abs_bp = cb.abs()
    pnl = abs_bp.where(correct, -abs_bp)
    return pnl.cumsum().rename("cum_pnl_bp")


# ═══════════════════════════════════════════════════════════════
# HMM Regime Labeling (원본 phase1~3 방법론)
# ═══════════════════════════════════════════════════════════════
#
# 3-state Gaussian HMM 을 lookback-day yield 변화에 fit.
# state mean 순으로 DOWN(0) / NEUTRAL(1) / UP(2) 매핑.
# regime_strength = |Δlookback| / rolling_std → 이 값이 일정 이상인 일자만 "유효 regime".
# persist : 현재 regime이 forward 후에도 같으면 1
# reverse : 현재 regime이 forward 후 반대(UP↔DOWN)면 1 (NEUTRAL 경유는 제외)
# ═══════════════════════════════════════════════════════════════

def hmm_label_regimes(s: pd.Series, lookback_days: int,
                       n_states: int = 3,
                       fit_until: Optional[pd.Timestamp] = None
                       ) -> Tuple[pd.Series, pd.DataFrame]:
    """3-state Gaussian HMM. 반환: (regime_label 'DOWN/NEUTRAL/UP', state_proba DataFrame).
    fit_until 지정 시 그 시점까지 데이터만 HMM 학습 (look-ahead 방지),
    이후 전체 시리즈에 predict. 주말/공휴일 NaN 은 ffill 로 처리.
    """
    try:
        from hmmlearn import hmm  # type: ignore
    except ImportError:
        raise RuntimeError("hmmlearn 필요: pip install hmmlearn")

    # 원시 시리즈 ffill (주말/공휴일 NaN 메움) — regime 은 persistent
    s_ff = s.ffill()
    feat = s_ff.diff(lookback_days).dropna()
    if len(feat) < 100:
        return (pd.Series(index=s.index, dtype=object),
                pd.DataFrame(index=s.index))

    # fit 윈도우
    if fit_until is not None:
        feat_fit = feat.loc[feat.index < fit_until]
        if len(feat_fit) < 100:
            feat_fit = feat
    else:
        feat_fit = feat

    X_fit = feat_fit.values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                              n_iter=200, random_state=42, tol=1e-4)
    try:
        model.fit(X_fit)
    except Exception:
        return (pd.Series(index=s.index, dtype=object),
                pd.DataFrame(index=s.index))

    X_all = feat.values.reshape(-1, 1)
    states = model.predict(X_all)
    try:
        probs = model.predict_proba(X_all)
    except Exception:
        probs = np.full((len(states), n_states), 1.0 / n_states)

    means = [model.means_[i, 0] for i in range(n_states)]
    sort_idx = np.argsort(means)  # 작은→큰: DOWN, (NEUTRAL,) UP
    if n_states == 2:
        name_map = {sort_idx[0]: "DOWN", sort_idx[1]: "UP"}
    else:
        name_map = {sort_idx[0]: "DOWN", sort_idx[1]: "NEUTRAL", sort_idx[2]: "UP"}

    state_labels = pd.Series([name_map[st] for st in states], index=feat.index)
    proba_df = pd.DataFrame(probs, index=feat.index,
                              columns=[name_map[i] for i in range(n_states)])
    # 원래 인덱스 정렬 후 ffill (regime은 일자 간 지속)
    return (state_labels.reindex(s.index).ffill(),
            proba_df.reindex(s.index).ffill())


def regime_strength_series(s: pd.Series, lookback_days: int,
                             vol_window: int = 252) -> pd.Series:
    """|Δlookback| / rolling_std(Δlookback). 큰 값일수록 강한 regime. 주말 ffill 처리."""
    s_ff = s.ffill()
    change = s_ff.diff(lookback_days)
    rolling_std = change.rolling(vol_window, min_periods=60).std()
    out = (change.abs() / rolling_std.replace(0, np.nan)).rename(
        f"regime_strength_{lookback_days}d")
    return out.ffill()


def make_persist_target(regime: pd.Series, forward_days: int,
                          strength: Optional[pd.Series] = None,
                          min_strength: float = 0.0) -> pd.Series:
    """persist = 1 if regime[t+forward] == regime[t] (NEUTRAL 일자는 제외)."""
    fut = regime.shift(-forward_days)
    same = (fut == regime)
    target = same.astype(float)
    # NEUTRAL 또는 미래 NaN 제외
    target[regime == "NEUTRAL"] = np.nan
    target[fut.isna() | regime.isna()] = np.nan
    if strength is not None and min_strength > 0:
        target[strength < min_strength] = np.nan
    return target.rename("y_persist")


def make_reverse_target(regime: pd.Series, forward_days: int,
                          strength: Optional[pd.Series] = None,
                          min_strength: float = 0.0) -> pd.Series:
    """reverse = 1 if UP→DOWN or DOWN→UP (NEUTRAL 경유 제외). 분모는 UP/DOWN 일자만."""
    fut = regime.shift(-forward_days)
    flip_ud = (regime == "UP")   & (fut == "DOWN")
    flip_du = (regime == "DOWN") & (fut == "UP")
    target = (flip_ud | flip_du).astype(float)
    # 현재가 NEUTRAL이거나 미래가 NaN/NEUTRAL인 경우 제외
    target[regime == "NEUTRAL"] = np.nan
    target[fut.isna() | regime.isna()] = np.nan
    target[fut == "NEUTRAL"] = np.nan
    if strength is not None and min_strength > 0:
        target[strength < min_strength] = np.nan
    return target.rename("y_reverse")


def make_regime_features(s: pd.Series, lookback_days: int,
                           hmm_fit_until: Optional[pd.Timestamp] = None
                           ) -> pd.DataFrame:
    """원본 phase1~3 의 regime context 피처:
       - regime_DOWN / regime_NEUTRAL / regime_UP one-hot
       - regime_strength (Δlookback / rolling_std)
       - state probabilities (proba_DOWN / proba_NEUTRAL / proba_UP)
    """
    regime, probs = hmm_label_regimes(s, lookback_days, n_states=3,
                                          fit_until=hmm_fit_until)
    strength = regime_strength_series(s, lookback_days)

    out = pd.DataFrame(index=s.index)
    out["regime_DOWN"]    = (regime == "DOWN").astype(float)
    out["regime_NEUTRAL"] = (regime == "NEUTRAL").astype(float)
    out["regime_UP"]      = (regime == "UP").astype(float)
    out["regime_strength"] = strength
    for col in ["DOWN", "NEUTRAL", "UP"]:
        if col in probs.columns:
            out[f"proba_regime_{col}"] = probs[col]
    return out


# ═══════════════════════════════════════════════════════════════
# Signal strength labeling (Phase3 — 원본 룰)
# ═══════════════════════════════════════════════════════════════

def signal_strength(mean_prob: float, consensus: float) -> str:
    """원본 phase3 의 STRONG/MODERATE/WEAK/NONE 결정 룰."""
    if np.isnan(mean_prob) or np.isnan(consensus):
        return "NONE"
    if mean_prob >= 0.65 and consensus >= 0.80:
        return "STRONG"
    if mean_prob >= 0.60 and consensus >= 0.60:
        return "MODERATE"
    if mean_prob >= 0.55:
        return "WEAK"
    return "NONE"

"""
공통 데이터 로딩 · 백테스트 로직 — Streamlit 앱 전역에서 import.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# 프로젝트 루트 sys.path 추가 (data_fetcher, features 등 import용)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PHASE1_DIR = PROJECT_ROOT / "results" / "phase1"
PHASE2_DIR = PROJECT_ROOT / "results" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "results" / "phase3"
CACHE_DIR  = PROJECT_ROOT / "data_cache"

KTB_TARGETS = [
    "ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y",
    "spread_ktb_3_10", "spread_ktb_5_30", "spread_ktb_10_30",
]
CREDIT_TARGETS = ["spread_aa_3y", "spread_bbb_3y"]   # 회사채 스프레드
TARGET_LABEL = {
    "ktb_3y": "KTB 3Y 금리",
    "ktb_5y": "KTB 5Y 금리",
    "ktb_10y": "KTB 10Y 금리",
    "ktb_30y": "KTB 30Y 금리",
    "spread_ktb_3_10": "KTB 10Y−3Y 스프레드",
    "spread_ktb_5_30": "KTB 30Y−5Y 스프레드",
    "spread_ktb_10_30": "KTB 30Y−10Y 스프레드",
    "spread_aa_3y": "회사채 AA− 3Y 스프레드",
    "spread_bbb_3y": "회사채 BBB− 3Y 스프레드",
}
TARGET_UNIT = {t: "%" if t.startswith("ktb") else "%pt"
               for t in KTB_TARGETS + CREDIT_TARGETS}

START_OOS = pd.Timestamp("2018-01-01")
TRIGGER_DEFAULT = 0.60


# ───────────────────────────────────────────────────────────
# 캐시된 로더 (Streamlit 재실행 시 빠르게)
# ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_metrics_top() -> pd.DataFrame:
    p = PHASE1_DIR / "metrics_top.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_phase1_predictions() -> pd.DataFrame:
    p = PHASE1_DIR / "predictions.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_signal_table() -> pd.DataFrame:
    p = PHASE3_DIR / "signal_table.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_yearly_breakdown() -> pd.DataFrame:
    p = PHASE1_DIR / "yearly_breakdown.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_target_series(target: str) -> pd.Series:
    """캐시에서 raw target yield 시리즈 로드."""
    from features import build_target_series, load_cache
    cache = load_cache(CACHE_DIR)
    try:
        return build_target_series(cache, target)
    except KeyError:
        return pd.Series(dtype=float)


@st.cache_data(show_spinner=False)
def load_latest_signal_text() -> str:
    p = PHASE3_DIR / "latest_signal.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_latest_signal_df() -> pd.DataFrame:
    """phase3 latest_signal.csv (구조화). 컬럼: target, model_set(top3/top1),
       current_value, current_regime, p_persist, p_reverse, signal, strength."""
    p = PHASE3_DIR / "latest_signal.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


# ───────────────────────────────────────────────────────────
# Top-N 모델 선정 + 앙상블 proba
# ───────────────────────────────────────────────────────────

def get_topn_models(target: str, n: int) -> pd.DataFrame:
    """metrics_top.csv 기준 ensemble 제외 후 composite 내림차순 Top-N."""
    mt = load_metrics_top()
    if mt.empty:
        return pd.DataFrame()
    sub = mt[(mt["target"] == target)
              & (mt["direction"] == "persist")
              & (mt["model"] != "ensemble")].copy()
    sub["rank2"] = sub["composite"].rank(ascending=False, method="min")
    return sub.sort_values("rank2").head(n).reset_index(drop=True)


def ensemble_proba(target: str, models_df: pd.DataFrame) -> pd.Series:
    """models_df 의 (run_id, model) 들의 phase1 OOS proba 평균."""
    pred = load_phase1_predictions()
    if pred.empty or models_df.empty:
        return pd.Series(dtype=float)
    cols = []
    for _, r in models_df.iterrows():
        c = f"{target}__{r['run_id']}__persist__proba_{r['model']}"
        if c in pred.columns:
            cols.append(c)
    if not cols:
        return pd.Series(dtype=float)
    return pred[cols].mean(axis=1)


def get_chg_regime(target: str, run_id: str) -> Tuple[pd.Series, pd.Series]:
    pred = load_phase1_predictions()
    chg_col = f"{target}__{run_id}__persist__chg_bp"
    reg_col = f"{target}__{run_id}__persist__regime"
    chg = pred[chg_col] if chg_col in pred.columns else pd.Series(dtype=float)
    reg = pred[reg_col] if reg_col in pred.columns else pd.Series(dtype=str)
    return chg, reg


# ───────────────────────────────────────────────────────────
# 시그널 매핑 + PnL
# ───────────────────────────────────────────────────────────

def signal_from_proba(p: pd.Series, regime: pd.Series, trigger: float) -> pd.Series:
    """phase3 매핑 (reverse 미선택 시 1-p를 reverse로 대체)."""
    sig = pd.Series(0, index=p.index, dtype=int)
    high = p >= trigger
    low  = p <= (1 - trigger)
    isd = regime == "DOWN"
    isu = regime == "UP"
    sig[isd & high] = +1
    sig[isd & low]  = -1
    sig[isu & high] = -1
    sig[isu & low]  = +1
    return sig


FORWARD_BD = 10   # 예측 horizon = 10영업일(≈2주)


@st.cache_data(show_spinner=False)
def forward_change_bp(target: str, horizon_bd: int = FORWARD_BD) -> pd.Series:
    """현재 raw 시리즈 기준 '영업일' forward 변화(bp).
       phase1 chg_bp는 캘린더-day 인덱스에 shift(-10)을 적용해 ~10캘린더일(≈7영업일)을
       측정하는 문제가 있어, 앱 레벨에서 영업일 기준으로 재계산해 차트와 일치시킴."""
    ts = load_target_series(target).dropna()
    if ts.empty:
        return pd.Series(dtype=float)
    return (ts.shift(-horizon_bd) - ts) * 100.0


def compute_backtest(target: str, models_df: pd.DataFrame,
                       trigger: float = TRIGGER_DEFAULT,
                       sample_every: int = 1) -> pd.DataFrame:
    """전체 백테스트 일별 시리즈 반환. sample_every=10이면 비중첩 10일 샘플.
       chg는 현재 데이터의 10영업일 forward 변화로 재계산(차트 색 세그먼트와 동일)."""
    if models_df.empty:
        return pd.DataFrame()
    p = ensemble_proba(target, models_df)
    run = models_df.iloc[0]["run_id"]
    _chg_old, reg = get_chg_regime(target, run)   # regime만 phase1에서 사용
    chg = forward_change_bp(target).reindex(p.index)   # 영업일 forward로 재계산
    df = pd.DataFrame({"p": p, "chg": chg, "regime": reg}).dropna()
    df = df[df.index >= START_OOS]
    if df.empty:
        return df
    if sample_every > 1:
        df = df.iloc[::sample_every].copy()
    df["signal"] = signal_from_proba(df["p"], df["regime"], trigger)
    df["pnl_bp"] = (-df["signal"] * df["chg"]).where(df["signal"] != 0, 0.0)
    df["cum_pnl"] = df["pnl_bp"].cumsum()
    df["dd"] = df["cum_pnl"] - df["cum_pnl"].cummax()
    return df


def _consolidate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """연속 동일방향 시그널을 1트레이드로 묶음.
       - 같은 방향(±1)이 이어지면 포지션 유지 = 1트레이드
       - signal=0(중립)이 끼면 청산 → 다음 시그널은 새 트레이드
       - 방향이 +↔- 뒤집히면 close-and-reverse → 별개 트레이드
       반환: 트레이드별 (start, end, signal, n_rebal, pnl_bp).
    """
    if df.empty or (df["signal"] == 0).all():
        return pd.DataFrame(columns=["start", "end", "signal", "n_rebal", "pnl_bp"])
    s = df["signal"]
    grp = (s != s.shift(1)).cumsum()
    trades = []
    for _, g in df.groupby(grp):
        sig = int(g["signal"].iloc[0])
        if sig == 0:
            continue
        trades.append({
            "start": g.index[0], "end": g.index[-1],
            "signal": sig, "n_rebal": len(g),
            "pnl_bp": float(g["pnl_bp"].sum()),
        })
    return pd.DataFrame(trades)


def summarize_backtest(df: pd.DataFrame) -> Dict:
    """리밸런스 단위(per-2주) + 트레이드 단위(연속 동일방향=1트레이드) 메트릭 동시 산출."""
    if df.empty:
        return {}
    traded = df[df["signal"] != 0]
    if traded.empty:
        return {"n_signals": 0, "n_trades": 0}

    # 리밸런스 단위 (2주마다 평가)
    n_rebal = len(traded)
    hit_rebal = (traded["pnl_bp"] > 0).mean()

    # 트레이드 단위 (연속 동일방향 = 1 포지션)
    tr = _consolidate_trades(df)
    n_trades = len(tr)
    hit_trade = (tr["pnl_bp"] > 0).mean() if n_trades else 0.0
    avg_hold_rebal = float(tr["n_rebal"].mean()) if n_trades else 0.0

    cum = float(df["cum_pnl"].iloc[-1])
    mdd = float(df["dd"].min())
    days = max((df.index[-1] - df.index[0]).days, 1)
    ann = cum / days * 365

    return {
        # 리밸런스 단위
        "n_signals": int(n_rebal),         # backward-compat alias
        "n_rebalances": int(n_rebal),
        "coverage": float(n_rebal / len(df)),
        "hit_rate_rebal": float(hit_rebal),
        "hit_rate": float(hit_rebal),       # backward-compat alias
        "mean_pnl_bp": float(traded["pnl_bp"].mean()),
        # 트레이드 단위 (연속 동일방향 통합)
        "n_trades": int(n_trades),
        "hit_rate_trade": float(hit_trade),
        "mean_pnl_per_trade": float(tr["pnl_bp"].mean()) if n_trades else 0.0,
        "avg_hold_rebal": avg_hold_rebal,
        "avg_hold_weeks": avg_hold_rebal * 2.0,   # 1 리밸런스 = 2주
        # 공통
        "cum_pnl_bp": cum,
        "ann_pnl_bp": float(ann),
        "max_dd_bp": mdd,
        "calmar": float(ann / abs(mdd)) if mdd < 0 else float("inf"),
    }


def yearly_pnl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(df.index.year).apply(lambda x: pd.Series({
        "n_trades": int((x.signal != 0).sum()),
        "hit_rate": float(((x["pnl_bp"] > 0).sum())
                           / max((x.signal != 0).sum(), 1)),
        "pnl_bp": float(x["pnl_bp"].sum()),
    }))
    out.index.name = "year"
    return out.reset_index()


# ───────────────────────────────────────────────────────────
# 메타 정보
# ───────────────────────────────────────────────────────────

def cache_file_status() -> pd.DataFrame:
    """data_cache/*.parquet 파일별 최종 수정시각 + 행수."""
    rows = []
    if not CACHE_DIR.exists():
        return pd.DataFrame()
    for p in sorted(CACHE_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            last_idx = df.index.max() if len(df) else None
            rows.append({
                "group": p.stem,
                "rows": len(df),
                "data_end": str(last_idx.date()) if last_idx is not None else "",
                "file_mtime": datetime.fromtimestamp(p.stat().st_mtime)
                                .strftime("%Y-%m-%d %H:%M"),
            })
        except Exception as e:
            rows.append({
                "group": p.stem, "rows": 0, "data_end": "",
                "file_mtime": f"error: {e}",
            })
    return pd.DataFrame(rows)


def phase_artifacts_status() -> pd.DataFrame:
    items = [
        ("phase1/predictions.parquet",         PHASE1_DIR / "predictions.parquet"),
        ("phase1/metrics_top.csv",             PHASE1_DIR / "metrics_top.csv"),
        ("phase1/regime_history.parquet",      PHASE1_DIR / "regime_history.parquet"),
        ("phase2/model_selection.csv",         PHASE2_DIR / "model_selection.csv"),
        ("phase3/signal_table.parquet",        PHASE3_DIR / "signal_table.parquet"),
        ("phase3/latest_signal.txt",           PHASE3_DIR / "latest_signal.txt"),
    ]
    rows = []
    for name, p in items:
        if p.exists():
            rows.append({
                "artifact": name,
                "size_kb": round(p.stat().st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime)
                            .strftime("%Y-%m-%d %H:%M"),
            })
        else:
            rows.append({"artifact": name, "size_kb": 0, "mtime": "(없음)"})
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────
# 시그널 매핑 텍스트 (UI 표시용)
# ───────────────────────────────────────────────────────────

def parse_latest_signal(text: str) -> pd.DataFrame:
    """phase3/latest_signal.txt → DataFrame."""
    import re
    rows = []
    blocks = re.split(r"  ── ", text)
    for b in blocks[1:]:
        m_name = re.match(r"(\S+)\s+현재값=([-\d\.]+)(%|% pt)\s+regime=(\S+)", b)
        m_prob = re.search(r"P\(persist\)=([-\d\.]+).*P\(reverse\)=([-\dnan\.]+).*signal=([+-]?\d+).*\[(\w+)\]", b)
        if not m_name:
            continue
        if m_prob:
            p_p = float(m_prob.group(1))
            sig = int(m_prob.group(3))
            strength = m_prob.group(4)
        else:
            p_p = float("nan"); sig = 0; strength = "NONE"
        rows.append({
            "target": m_name.group(1),
            "current": float(m_name.group(2)),
            "unit": m_name.group(3),
            "regime": m_name.group(4),
            "p_persist": p_p,
            "signal": sig,
            "strength": strength,
        })
    return pd.DataFrame(rows)


SIGNAL_MAPPING_MD = """
| 현재 regime | trigger | 신호 | 의미 |
|---|---|---|---|
| DOWN | P(persist) ≥ trigger | **+1** | 금리/스프레드 계속 하락 → **듀레이션 확대 / 크레딧 확대** |
| DOWN | P(persist) ≤ 1−trigger | **−1** | 반전 상승 → 듀레이션 축소 |
| UP   | P(persist) ≥ trigger | **−1** | 계속 상승 → 듀레이션 축소 / 크레딧 축소 |
| UP   | P(persist) ≤ 1−trigger | **+1** | 반전 하락 → 듀레이션 확대 |
| NEUTRAL or 미달 | — | 0 | 대기 |

**강도** (P(persist) 기준)
- STRONG: P ≥ 0.65
- MODERATE: 0.60 ≤ P < 0.65
- WEAK: 0.55 ≤ P < 0.60
- NONE: P < 0.55
"""
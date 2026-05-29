"""
features.py
───────────
KBOND ActiveQuant — KTB 방향성 예측용 피처 엔지니어링

입력: data_cache/*.parquet (data_fetcher.load_all_data 산출물)
출력: 일별 wide DataFrame (DatetimeIndex × feature columns)

설계
────
1) 금리 커브 (rates_ktb + rates_credit)
   - 레벨, 슬로프(10y-2y, 30y-10y, 5y-2y), 커버처(2-5-10)
   - 모멘텀: d5/d20/d60 (bp 단위)
   - 정규화: z20/z60/z252 (rolling z-score)
   - 변동성: 20일 daily-change std
   - 크레딧 스프레드: corp_aa-ktb_3y, corp_bbb-ktb_3y, kepco-ktb_5y

2) 미국 금리 (us_rates + us_global)
   - us_2y, us_10y, us_10y2y, us_bei_10y, us_real_10y 레벨/변화/z

3) FX·주식·글로벌 (fx + equity + global_yahoo)
   - usdkrw 로그수익률, kospi 로그수익률, 외인 순매수 z
   - vix 레벨/z, dxy 변화, wti 변화

4) 매크로 (월간→일별 ffill + 1개월 지연으로 lookahead 차단)
   - CPI/PPI/IP YoY, fx_reserve YoY, csi z, us_cpi YoY, us_unrate level

NaN 처리: 피처별로 forward-fill 후 warm-up 252일은 호출 측에서 절단.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 캐시 로더
# ═══════════════════════════════════════════════════════════════

DEFAULT_CACHE = Path(__file__).resolve().parent / "data_cache"

PARQUET_GROUPS = [
    "rates_ktb", "rates_credit",
    "macro_growth", "macro_price", "macro_money", "macro_sentiment",
    "fx", "equity",
    "us_rates", "us_macro", "us_global", "us_credit",
    "global_yahoo",
    "us_xl",
]


def load_cache(cache_dir: Path = DEFAULT_CACHE) -> Dict[str, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    out: Dict[str, pd.DataFrame] = {}
    for g in PARQUET_GROUPS:
        p = cache_dir / f"{g}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            out[g] = df.sort_index()
        else:
            out[g] = pd.DataFrame()
    return out


# ═══════════════════════════════════════════════════════════════
# 헬퍼: 롤링 z-score / diff
# ═══════════════════════════════════════════════════════════════

def _rolling_z(s: pd.Series, window: int) -> pd.Series:
    mp = max(20, window // 4)
    mu  = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    z = (s - mu) / std.replace(0, np.nan)
    return z


def _add_momentum_z(df: pd.DataFrame, col: str,
                     d_windows=(5, 20, 60),
                     z_windows=(20, 60, 252),
                     prefix: Optional[str] = None) -> pd.DataFrame:
    """col 시리즈에 대해 차분/z-score 피처를 df에 추가."""
    p = prefix or col
    s = df[col]
    out = {}
    for w in d_windows:
        out[f"d{w}_{p}"] = s.diff(w)
    for w in z_windows:
        out[f"z{w}_{p}"] = _rolling_z(s, w)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


# ═══════════════════════════════════════════════════════════════
# (1) 금리 커브 + 크레딧
# ═══════════════════════════════════════════════════════════════

def build_rate_features(rates_ktb: pd.DataFrame,
                         rates_credit: pd.DataFrame) -> pd.DataFrame:
    if rates_ktb.empty:
        return pd.DataFrame()

    df = rates_ktb.copy()
    # ktb_2y는 2021년부터만 있음 → ktb_1y/3y로 보강
    # 그대로 컬럼 사용. 결측은 ffill로 채우진 않고 모델에 NaN 그대로 (LGBM 처리 가능)

    keys = [c for c in ["ktb_1y", "ktb_3y", "ktb_5y", "ktb_10y", "ktb_20y", "ktb_30y", "cd_91d"]
            if c in df.columns]

    out = pd.DataFrame(index=df.index)
    # 레벨
    for k in keys:
        out[f"lvl_{k}"] = df[k]

    # 슬로프 / 커버처
    if {"ktb_2y", "ktb_10y"}.issubset(df.columns):
        out["slope_10y2y"] = df["ktb_10y"] - df["ktb_2y"]
    elif {"ktb_3y", "ktb_10y"}.issubset(df.columns):
        out["slope_10y3y"] = df["ktb_10y"] - df["ktb_3y"]
    if {"ktb_10y", "ktb_30y"}.issubset(df.columns):
        out["slope_30y10y"] = df["ktb_30y"] - df["ktb_10y"]
    if {"ktb_5y", "ktb_30y"}.issubset(df.columns):
        out["slope_30y5y"] = df["ktb_30y"] - df["ktb_5y"]
    if {"ktb_3y", "ktb_10y"}.issubset(df.columns):
        out["slope_10y3y"] = df["ktb_10y"] - df["ktb_3y"]
    if {"ktb_3y", "ktb_5y", "ktb_10y"}.issubset(df.columns):
        out["bfly_3_5_10"] = 2 * df["ktb_5y"] - df["ktb_3y"] - df["ktb_10y"]
    if {"ktb_3y", "base_rate"}.issubset(df.columns):
        out["ktb3y_base_spread"] = df["ktb_3y"] - df["base_rate"]
    if {"cd_91d", "base_rate"}.issubset(df.columns):
        out["cd_base_spread"] = df["cd_91d"] - df["base_rate"]

    # 모멘텀/Z — 금리 레벨은 df에서, 슬로프/버터플라이는 out에서
    level_sources = [(k, df[k], k) for k in keys]  # (tag, series, _)
    derived_keys = [c for c in ["slope_10y2y", "slope_10y3y", "slope_30y10y",
                                  "slope_30y5y", "bfly_3_5_10",
                                  "ktb3y_base_spread", "cd_base_spread"] if c in out.columns]
    derived_sources = [(c, out[c], c) for c in derived_keys]

    for tag, src, _ in level_sources + derived_sources:
        for w in (5, 20, 60):
            out[f"d{w}_{tag}"] = src.diff(w)
        for w in (20, 60, 252):
            out[f"z{w}_{tag}"] = _rolling_z(src, w)

    # 일별 변화의 20일 std (realized vol)
    for k in ["ktb_3y", "ktb_5y", "ktb_10y"]:
        if k in df.columns:
            out[f"vol20_{k}"] = df[k].diff().rolling(20, min_periods=10).std()

    # 크레딧 스프레드
    if not rates_credit.empty:
        cr = rates_credit.reindex(df.index)
        if {"corp_aa_3y"}.issubset(cr.columns) and "ktb_3y" in df.columns:
            out["sp_corp_aa_3y"] = cr["corp_aa_3y"] - df["ktb_3y"]
        if {"corp_bbb_3y"}.issubset(cr.columns) and "ktb_3y" in df.columns:
            out["sp_corp_bbb_3y"] = cr["corp_bbb_3y"] - df["ktb_3y"]
        if {"kepco_5y"}.issubset(cr.columns) and "ktb_5y" in df.columns:
            out["sp_kepco_5y"] = cr["kepco_5y"] - df["ktb_5y"]
        if {"ind_fin_1y"}.issubset(cr.columns) and "ktb_1y" in df.columns:
            out["sp_indfin_1y"] = cr["ind_fin_1y"] - df["ktb_1y"]
        for c in ["sp_corp_aa_3y", "sp_corp_bbb_3y", "sp_kepco_5y", "sp_indfin_1y"]:
            if c in out.columns:
                out[f"d20_{c}"] = out[c].diff(20)
                out[f"z60_{c}"] = _rolling_z(out[c], 60)

    return out


# ═══════════════════════════════════════════════════════════════
# (2) 미국 금리
# ═══════════════════════════════════════════════════════════════

def build_us_features(us_rates: pd.DataFrame,
                       us_global: pd.DataFrame,
                       global_yahoo: pd.DataFrame,
                       us_credit: pd.DataFrame = None,
                       us_xl: pd.DataFrame = None) -> pd.DataFrame:
    out = pd.DataFrame()
    if not us_rates.empty:
        out = pd.DataFrame(index=us_rates.index)
        for c in ["us_ffr", "us_2y", "us_5y", "us_10y", "us_30y",
                  "us_10y2y", "us_10y3m", "us_bei_10y", "us_bei_5y5y", "us_real_10y"]:
            if c in us_rates.columns:
                out[f"lvl_{c}"] = us_rates[c]
                out[f"d5_{c}"]  = us_rates[c].diff(5)
                out[f"d20_{c}"] = us_rates[c].diff(20)
                out[f"z60_{c}"] = _rolling_z(us_rates[c], 60)

    if not us_global.empty:
        gg = us_global.reindex(out.index if not out.empty else us_global.index)
        if out.empty:
            out = pd.DataFrame(index=gg.index)
        for c in ["dxy_broad", "vixcls", "wti", "brent", "tedrate"]:
            if c in gg.columns:
                out[f"lvl_{c}"] = gg[c]
                out[f"d5_{c}"]  = gg[c].diff(5)
                out[f"d20_{c}"] = gg[c].diff(20)
                out[f"z60_{c}"] = _rolling_z(gg[c], 60)

    # 미국 회사채 OAS (FRED us_credit)
    if us_credit is not None and not us_credit.empty:
        uc = us_credit.reindex(out.index if not out.empty else us_credit.index)
        if out.empty:
            out = pd.DataFrame(index=uc.index)
        for c in ["us_ig_oas", "us_bbb_oas", "us_hy_oas"]:
            if c in uc.columns:
                out[f"lvl_{c}"]  = uc[c]
                out[f"d5_{c}"]   = uc[c].diff(5)
                out[f"d20_{c}"]  = uc[c].diff(20)
                out[f"z60_{c}"]  = _rolling_z(uc[c], 60)
                out[f"z252_{c}"] = _rolling_z(uc[c], 252)
        # HY-IG 스프레드 차 = 신용 스트레스 지표
        if {"us_hy_oas", "us_ig_oas"}.issubset(uc.columns):
            stress = uc["us_hy_oas"] - uc["us_ig_oas"]
            out["us_hy_ig_diff"]    = stress
            out["d20_us_hy_ig"]     = stress.diff(20)
            out["z60_us_hy_ig"]     = _rolling_z(stress, 60)
        # BBB-AAA 스프레드 차
        if {"us_bbb_oas", "us_aaa_oas"}.issubset(uc.columns):
            ig_curve = uc["us_bbb_oas"] - uc["us_aaa_oas"]
            out["us_bbb_aaa_diff"]  = ig_curve
            out["d20_us_bbb_aaa"]   = ig_curve.diff(20)

    # 미국 장단기 스프레드 (Excel us_xl) — 사용자 지정 3종만
    if us_xl is not None and not us_xl.empty:
        ux = us_xl.reindex(out.index if not out.empty else us_xl.index)
        if out.empty:
            out = pd.DataFrame(index=ux.index)
        for c in ["us_spr_2_10", "us_spr_5_30", "us_spr_10_30"]:
            if c in ux.columns:
                out[f"lvl_{c}"]  = ux[c]
                out[f"d5_{c}"]   = ux[c].diff(5)
                out[f"d20_{c}"]  = ux[c].diff(20)
                out[f"z60_{c}"]  = _rolling_z(ux[c], 60)
                out[f"z252_{c}"] = _rolling_z(ux[c], 252)
        # Bloomberg us_10y_xl / us_ig_oas_xl 도 피처로
        for c in ["us_10y_xl", "us_ig_oas_xl"]:
            if c in ux.columns:
                out[f"lvl_{c}"]  = ux[c]
                out[f"d5_{c}"]   = ux[c].diff(5)
                out[f"d20_{c}"]  = ux[c].diff(20)
                out[f"z60_{c}"]  = _rolling_z(ux[c], 60)

    # Yahoo VIX는 평일 더 잘 채워짐 (FRED VIXCLS 보조)
    if not global_yahoo.empty:
        gy = global_yahoo.reindex(out.index if not out.empty else global_yahoo.index)
        if out.empty:
            out = pd.DataFrame(index=gy.index)
        for c in ["yh_vix", "yh_dxy", "yh_sp500", "yh_gold"]:
            if c in gy.columns:
                out[f"lvl_{c}"] = gy[c]
                if c == "yh_sp500":
                    ret = np.log(gy[c]).diff()
                    out["ret1_sp500"]  = ret
                    out["ret20_sp500"] = ret.rolling(20).sum()
                    out["z60_sp500"]   = _rolling_z(gy[c], 60)
                else:
                    out[f"d5_{c}"]  = gy[c].diff(5)
                    out[f"d20_{c}"] = gy[c].diff(20)
                    out[f"z60_{c}"] = _rolling_z(gy[c], 60)

    return out


# ═══════════════════════════════════════════════════════════════
# (3) FX + 주식
# ═══════════════════════════════════════════════════════════════

def build_fx_equity_features(fx: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    if not fx.empty:
        out = pd.DataFrame(index=fx.index)
        for c in ["usdkrw", "cnykrw", "jpykrw", "eurkrw"]:
            if c in fx.columns:
                out[f"lvl_{c}"] = fx[c]
                ret = np.log(fx[c]).diff()
                out[f"ret1_{c}"]  = ret
                out[f"ret20_{c}"] = ret.rolling(20).sum()
                out[f"z60_{c}"]   = _rolling_z(fx[c], 60)
        # USD/KRW intraday range (변동성 proxy)
        if {"usdkrw_h", "usdkrw_l", "usdkrw_o"}.issubset(fx.columns):
            out["usdkrw_range"] = (fx["usdkrw_h"] - fx["usdkrw_l"]) / fx["usdkrw_o"]
            out["d20_usdkrw_range"] = out["usdkrw_range"].rolling(20, min_periods=10).mean()

    if not equity.empty:
        eq = equity.reindex(out.index if not out.empty else equity.index)
        if out.empty:
            out = pd.DataFrame(index=eq.index)
        if "kospi" in eq.columns:
            ret = np.log(eq["kospi"]).diff()
            out["ret1_kospi"]  = ret
            out["ret20_kospi"] = ret.rolling(20).sum()
            out["vol20_kospi"] = ret.rolling(20, min_periods=10).std()
            out["z60_kospi"]   = _rolling_z(eq["kospi"], 60)
        if "fgn_kospi" in eq.columns:
            out["fgn_kospi"]      = eq["fgn_kospi"]
            out["fgn_kospi_z60"]  = _rolling_z(eq["fgn_kospi"], 60)
            out["fgn_kospi_cum20"] = eq["fgn_kospi"].rolling(20, min_periods=5).sum()
        if "vol_kospi" in eq.columns:
            out["z60_vol_kospi"] = _rolling_z(eq["vol_kospi"], 60)

    return out


# ═══════════════════════════════════════════════════════════════
# (4) 매크로 (월간→일별 ffill, 1개월 지연으로 lookahead 차단)
# ═══════════════════════════════════════════════════════════════

PUBLICATION_LAG = pd.Timedelta(days=30)


def _monthly_to_daily(s: pd.Series, daily_index: pd.DatetimeIndex,
                       lag: pd.Timedelta = PUBLICATION_LAG) -> pd.Series:
    """월간 시리즈를 1개월 lag 적용한 뒤 daily index에 ffill로 정렬."""
    if s.empty:
        return pd.Series(index=daily_index, dtype=float)
    s2 = s.copy()
    s2.index = s2.index + lag
    s2 = s2.sort_index()
    return s2.reindex(daily_index, method="ffill")


def build_macro_features(macro_growth: pd.DataFrame,
                          macro_price: pd.DataFrame,
                          macro_money: pd.DataFrame,
                          macro_sentiment: pd.DataFrame,
                          us_macro: pd.DataFrame,
                          daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=daily_index)

    def yoy(s: pd.Series) -> pd.Series:
        return s.pct_change(12, fill_method=None) * 100  # 월간 시리즈 가정

    def mom(s: pd.Series) -> pd.Series:
        return s.pct_change(1, fill_method=None) * 100

    for src, prefix in [
        (macro_growth, "kr_growth"),
        (macro_price,  "kr_price"),
        (macro_money,  "kr_money"),
        (macro_sentiment, "kr_sent"),
    ]:
        if src is None or src.empty:
            continue
        for c in src.columns:
            s = src[c]
            out[f"{prefix}_{c}_lvl"]   = _monthly_to_daily(s, daily_index)
            out[f"{prefix}_{c}_yoy"]   = _monthly_to_daily(yoy(s), daily_index)
            out[f"{prefix}_{c}_mom"]   = _monthly_to_daily(mom(s), daily_index)

    # 미국 매크로 — 핵심만
    if us_macro is not None and not us_macro.empty:
        for c in ["us_cpi", "us_cpi_core", "us_pce_core", "us_indpro",
                  "us_unrate", "us_payems", "us_ism_mfg"]:
            if c in us_macro.columns:
                s = us_macro[c]
                out[f"{c}_lvl"] = _monthly_to_daily(s, daily_index)
                out[f"{c}_yoy"] = _monthly_to_daily(yoy(s), daily_index)

    return out


# ═══════════════════════════════════════════════════════════════
# (5) 통합
# ═══════════════════════════════════════════════════════════════

def build_all_features(cache: Optional[Dict[str, pd.DataFrame]] = None,
                        cache_dir: Path = DEFAULT_CACHE,
                        warmup_days: int = 252) -> pd.DataFrame:
    """모든 그룹 통합 후 일별 wide DataFrame 리턴."""
    if cache is None:
        cache = load_cache(cache_dir)

    # 기준 인덱스 = rates_ktb (가장 풍부한 일별 시리즈)
    rates_ktb = cache.get("rates_ktb", pd.DataFrame())
    if rates_ktb.empty:
        raise RuntimeError("rates_ktb.parquet 없음 또는 비어있음. run_fetch_data.py 먼저 실행하세요.")
    base_idx = rates_ktb.index

    parts = []
    parts.append(build_rate_features(rates_ktb, cache.get("rates_credit", pd.DataFrame())))
    parts.append(build_us_features(cache.get("us_rates", pd.DataFrame()),
                                    cache.get("us_global", pd.DataFrame()),
                                    cache.get("global_yahoo", pd.DataFrame()),
                                    cache.get("us_credit", pd.DataFrame()),
                                    cache.get("us_xl", pd.DataFrame())))
    parts.append(build_fx_equity_features(cache.get("fx", pd.DataFrame()),
                                           cache.get("equity", pd.DataFrame())))
    parts.append(build_macro_features(cache.get("macro_growth", pd.DataFrame()),
                                       cache.get("macro_price", pd.DataFrame()),
                                       cache.get("macro_money", pd.DataFrame()),
                                       cache.get("macro_sentiment", pd.DataFrame()),
                                       cache.get("us_macro", pd.DataFrame()),
                                       base_idx))

    # 모든 part를 base_idx에 정렬
    aligned = [p.reindex(base_idx) for p in parts if p is not None and not p.empty]
    X = pd.concat(aligned, axis=1)

    # 일별 ffill (주말/공휴일 갭 메움) — 그래도 시작 부분의 진짜 결측은 NaN 유지
    X = X.ffill(limit=5)

    # warmup 절단
    if warmup_days > 0 and len(X) > warmup_days:
        X = X.iloc[warmup_days:]

    return X


# ═══════════════════════════════════════════════════════════════
# (6) 타겟 생성
# ═══════════════════════════════════════════════════════════════

def make_direction_target(rates_ktb: pd.DataFrame,
                           tenor: str,
                           horizon: int) -> pd.Series:
    """tenor 금리의 H일 후 방향 (1=상승, 0=하락). NaN은 학습에서 제외.

    경계 사례:
    - Δ가 정확히 0이면 1(상승)로 처리 (실무 영향 미미, 학습 안정성 우선)
    """
    if tenor not in rates_ktb.columns:
        raise KeyError(f"{tenor} not in rates_ktb columns")
    y_full = rates_ktb[tenor]
    delta = y_full.shift(-horizon) - y_full
    target = (delta >= 0).astype(float)
    target[delta.isna()] = np.nan
    return target.rename(f"y_{tenor}_h{horizon}")


def make_direction_change_bp(rates_ktb: pd.DataFrame,
                              tenor: str,
                              horizon: int) -> pd.Series:
    """진단용: H일 후 변화량 (bp 단위 = % × 100)."""
    if tenor not in rates_ktb.columns:
        raise KeyError(f"{tenor} not in rates_ktb columns")
    y_full = rates_ktb[tenor]
    return ((y_full.shift(-horizon) - y_full) * 100).rename(f"chg_bp_{tenor}_h{horizon}")


# ═══════════════════════════════════════════════════════════════
# 타겟 시리즈 빌더 (KTB 금리 + 회사채 스프레드 일원화)
# ═══════════════════════════════════════════════════════════════

def build_target_series(cache: Dict[str, pd.DataFrame],
                         name: str) -> pd.Series:
    """name → 일별 시리즈 (DatetimeIndex). 타겟 정의:
       - ktb_3y / ktb_5y / ktb_10y : 해당 만기 금리 레벨 (%)
       - spread_aa_3y   : corp_aa_3y  - ktb_3y
       - spread_bbb_3y  : corp_bbb_3y - ktb_3y
       - spread_kepco_5y: kepco_5y    - ktb_5y
    """
    rk = cache.get("rates_ktb", pd.DataFrame())
    rc = cache.get("rates_credit", pd.DataFrame())

    if name in ("ktb_3y", "ktb_5y", "ktb_10y", "ktb_2y", "ktb_1y", "ktb_20y", "ktb_30y"):
        if rk.empty or name not in rk.columns:
            raise KeyError(f"{name} not in rates_ktb")
        return rk[name].rename(name)

    # KTB 장단기 스프레드 (= longer - shorter, US 명명규칙과 동일)
    if name == "spread_ktb_3_10":
        if not {"ktb_3y", "ktb_10y"}.issubset(rk.columns):
            raise KeyError("spread_ktb_3_10: ktb_3y / ktb_10y 필요")
        return (rk["ktb_10y"] - rk["ktb_3y"]).rename(name)
    if name == "spread_ktb_5_30":
        if not {"ktb_5y", "ktb_30y"}.issubset(rk.columns):
            raise KeyError("spread_ktb_5_30: ktb_5y / ktb_30y 필요")
        return (rk["ktb_30y"] - rk["ktb_5y"]).rename(name)
    if name == "spread_ktb_10_30":
        if not {"ktb_10y", "ktb_30y"}.issubset(rk.columns):
            raise KeyError("spread_ktb_10_30: ktb_10y / ktb_30y 필요")
        return (rk["ktb_30y"] - rk["ktb_10y"]).rename(name)

    if name == "spread_aa_3y":
        if "corp_aa_3y" not in rc.columns or "ktb_3y" not in rk.columns:
            raise KeyError("spread_aa_3y: required series missing")
        s = rc["corp_aa_3y"].reindex(rk.index) - rk["ktb_3y"]
        return s.rename(name)

    if name == "spread_bbb_3y":
        if "corp_bbb_3y" not in rc.columns or "ktb_3y" not in rk.columns:
            raise KeyError("spread_bbb_3y: required series missing")
        s = rc["corp_bbb_3y"].reindex(rk.index) - rk["ktb_3y"]
        return s.rename(name)

    if name == "spread_kepco_5y":
        if "kepco_5y" not in rc.columns or "ktb_5y" not in rk.columns:
            raise KeyError("spread_kepco_5y: required series missing")
        s = rc["kepco_5y"].reindex(rk.index) - rk["ktb_5y"]
        return s.rename(name)

    # ── US 타겟 (FRED) ────────────────────────────────
    us_rates  = cache.get("us_rates",  pd.DataFrame())
    us_credit = cache.get("us_credit", pd.DataFrame())

    # US 10Y Treasury yield
    if name in ("us_10y", "us_2y", "us_5y", "us_30y", "us_3m",
                 "us_real_10y", "us_bei_10y"):
        if us_rates.empty or name not in us_rates.columns:
            raise KeyError(f"{name} not in us_rates")
        return us_rates[name].rename(name)

    # US 회사채 OAS (이미 스프레드 형태 — 별도 차감 불필요)
    if name in ("us_ig_oas", "us_aaa_oas", "us_aa_oas", "us_a_oas",
                 "us_bbb_oas", "us_hy_oas"):
        if us_credit.empty or name not in us_credit.columns:
            raise KeyError(f"{name} not in us_credit (run_fetch_data.py 추가 실행 필요)")
        return us_credit[name].rename(name)

    # ── Excel(BB) 출처 미국 데이터 ────────────────────────────
    us_xl = cache.get("us_xl", pd.DataFrame())
    if name in ("us_10y_xl", "us_ig_oas_xl",
                 "us_spr_2_10", "us_spr_2_5", "us_spr_2_30",
                 "us_spr_10_30", "us_spr_5_30"):
        if us_xl.empty or name not in us_xl.columns:
            raise KeyError(f"{name} not in us_xl (run_fetch_data.py 추가 실행 필요)")
        return us_xl[name].rename(name)

    raise KeyError(f"unknown target name: {name}")


def make_direction_target_from_series(s: pd.Series, horizon: int) -> pd.Series:
    """시리즈의 H일 후 변화 부호 (1=상승/와이드닝, 0=하락/타이트닝). NaN→NaN."""
    delta = s.shift(-horizon) - s
    target = (delta >= 0).astype(float)
    target[delta.isna()] = np.nan
    return target.rename(f"y_{s.name}_h{horizon}")


def make_direction_change_bp_from_series(s: pd.Series, horizon: int) -> pd.Series:
    return ((s.shift(-horizon) - s) * 100).rename(f"chg_bp_{s.name}_h{horizon}")

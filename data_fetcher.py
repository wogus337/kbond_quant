"""
data_fetcher.py
────────────────
KBOND ActiveQuant — KTB(2Y/5Y/10Y) 방향성 예측용 데이터 수집 모듈

수집 소스
─────────
1. BOK ECOS API (무료, 키 내장)
   - rates_ktb     : 국고채 1Y~50Y, 기준금리, CD/CP
   - rates_credit  : 회사채 AA-/BBB-, 산금채, 한전채
   - macro_growth  : GDP, 광공업생산, 선행지수, 수출입
   - macro_price   : CPI(헤드라인/근원), PPI, 수입물가, 기대인플레이션
   - macro_money   : M1/M2, 외환보유액
   - macro_sentiment: BSI, CSI
   - fx            : USD/KRW(매매기준율·OHLC), CNY, JPY, EUR
   - equity        : KOSPI, KOSDAQ, 외인 순매수, 거래대금, 시가총액

2. FRED API (무료, 키 내장 / CSV 폴백)
   - us_rates      : DGS2/5/10/30, DFF, T10Y2Y, T10Y3M, T10YIE (BEI), T5YIFR
   - us_macro      : CPIAUCSL, PCEPI, INDPRO, UNRATE, PAYEMS, ISM proxy
   - us_global     : DCOILWTICO, DTWEXBGS(USD index), VIXCLS, DEXKOUS(KRW/USD)

3. Yahoo Finance (무료, raw chart API)
   - global_yahoo  : ^TNX, ^VIX, DX-Y.NYB, ^GSPC, ^KS200, CL=F, GC=F

캐시: ./data_cache/{group}.parquet
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════

BOK_API_KEY   = "Q9NMPP58WSJBI3BYD6AW"
FRED_API_KEY  = "1bdc46db405f39815c783e1525cfd1cb"
BOK_BASE_URL  = "https://ecos.bok.or.kr/api"
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_CSV_URL  = "https://fred.stlouisfed.org/graph/fredgraph.csv"

REQUEST_DELAY = 0.25

PROJECT_DIR  = Path(__file__).resolve().parent
CACHE_DIR    = PROJECT_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# ECOS 시리즈 정의
# (stat_code, cycle, item_code, label)
# ═══════════════════════════════════════════════════════════════

ECOS_RATES_KTB = {
    "base_rate" : ("722Y001", "D", "0101000",  "기준금리"),
    "ktb_1y"    : ("817Y002", "D", "010190000", "국고채 1Y"),
    "ktb_2y"    : ("817Y002", "D", "010195000", "국고채 2Y"),
    "ktb_3y"    : ("817Y002", "D", "010200000", "국고채 3Y"),
    "ktb_5y"    : ("817Y002", "D", "010200001", "국고채 5Y"),
    "ktb_10y"   : ("817Y002", "D", "010210000", "국고채 10Y"),
    "ktb_20y"   : ("817Y002", "D", "010220000", "국고채 20Y"),
    "ktb_30y"   : ("817Y002", "D", "010230000", "국고채 30Y"),
    "ktb_50y"   : ("817Y002", "D", "010240000", "국고채 50Y"),
    "cd_91d"    : ("817Y002", "D", "010502000", "CD 91일"),
    "cp_91d"    : ("817Y002", "D", "010503000", "CP 91일"),
}

ECOS_RATES_CREDIT = {
    "corp_aa_3y"   : ("817Y002", "D", "010300000", "회사채 AA- 3Y"),
    "corp_aa_un_3y": ("817Y002", "D", "010310000", "회사채 AA- 3Y 무보증"),
    "corp_bbb_3y"  : ("817Y002", "D", "010320000", "회사채 BBB- 3Y"),
    "ind_fin_1y"   : ("817Y002", "D", "010260000", "산금채 1Y"),
    "kepco_5y"     : ("817Y002", "D", "010503500", "한전채 5Y"),
}

ECOS_MACRO_GROWTH = {
    "gdp_real_qoq" : ("200Y001", "Q", "10111",  "실질GDP (계절조정, 전기비)"),
    "ip_total"     : ("901Y033", "M", "A00",    "광공업생산지수 (월)"),
    "cli_lead"     : ("901Y067", "M", "I16C",   "경기선행지수 순환변동치"),
    "exports"      : ("901Y011", "M", "FIEAA",  "수출 (월간)"),
    "imports"      : ("901Y011", "M", "FIEAB",  "수입 (월간)"),
}

ECOS_MACRO_PRICE = {
    "cpi_total"    : ("901Y009", "M", "0",     "소비자물가지수 (총지수)"),
    "cpi_core"     : ("901Y009", "M", "A0",    "근원 CPI (농산물·석유류 제외)"),
    "ppi_total"    : ("404Y014", "M", "*AA",   "생산자물가지수 (총지수)"),
    "imp_price"    : ("401Y015", "M", "*AA",   "수입물가지수 (계약통화 기준)"),
    "exp_inflation": ("511Y002", "M", "FMC",   "기대인플레이션 (가계 1년)"),
}

ECOS_MACRO_MONEY = {
    "m1_avg"     : ("101Y003", "M", "BBHA00", "M1 평잔"),
    "m2_avg"     : ("101Y003", "M", "BBHS00", "M2 평잔"),
    "fx_reserve" : ("902Y006", "M", "KR",     "외환보유액 (한국)"),
}

ECOS_MACRO_SENTIMENT = {
    "bsi_total_actual" : ("512Y014", "M", "AX1AA", "BSI 전산업 업황실적"),
    "bsi_total_outlook": ("512Y014", "M", "AX2AA", "BSI 전산업 업황전망"),
    "csi_total"        : ("511Y002", "M", "FME",   "CSI 종합소비자심리지수"),
}

ECOS_FX = {
    "usdkrw"   : ("731Y001", "D", "0000001", "원/달러 매매기준율"),
    "cnykrw"   : ("731Y001", "D", "0000053", "원/위안 매매기준율"),
    "jpykrw"   : ("731Y001", "D", "0000002", "원/엔(100)"),
    "eurkrw"   : ("731Y001", "D", "0000003", "원/유로"),
    "usdkrw_o" : ("731Y003", "D", "0000002", "원/달러 시가"),
    "usdkrw_h" : ("731Y003", "D", "0000005", "원/달러 고가"),
    "usdkrw_l" : ("731Y003", "D", "0000004", "원/달러 저가"),
    "usdkrw_c" : ("731Y003", "D", "0000003", "원/달러 종가15:30"),
}

ECOS_EQUITY = {
    "kospi"        : ("802Y001", "D", "0001000", "KOSPI 종합"),
    "kosdaq"       : ("802Y001", "D", "0089000", "KOSDAQ 종합"),
    "fgn_kospi"    : ("802Y001", "D", "0030000", "외인 순매수(KOSPI)"),
    "fgn_kosdaq"   : ("802Y001", "D", "0113000", "외인 순매수(KOSDAQ)"),
    "vol_kospi"    : ("802Y001", "D", "0088000", "거래대금(KOSPI)"),
    "mktcap_kospi" : ("802Y001", "D", "0183000", "시가총액(KOSPI)"),
}

ECOS_GROUPS: Dict[str, Dict[str, Tuple[str, str, str, str]]] = {
    "rates_ktb"       : ECOS_RATES_KTB,
    "rates_credit"    : ECOS_RATES_CREDIT,
    "macro_growth"    : ECOS_MACRO_GROWTH,
    "macro_price"     : ECOS_MACRO_PRICE,
    "macro_money"     : ECOS_MACRO_MONEY,
    "macro_sentiment" : ECOS_MACRO_SENTIMENT,
    "fx"              : ECOS_FX,
    "equity"          : ECOS_EQUITY,
}


# ═══════════════════════════════════════════════════════════════
# FRED 시리즈 정의 — (series_id, frequency, label)
# frequency는 진단용. FRED는 원본 빈도로 반환.
# ═══════════════════════════════════════════════════════════════

FRED_US_RATES = {
    "us_ffr"    : ("DFF",     "D", "Effective Federal Funds Rate"),
    "us_2y"     : ("DGS2",    "D", "US Treasury 2Y"),
    "us_5y"     : ("DGS5",    "D", "US Treasury 5Y"),
    "us_10y"    : ("DGS10",   "D", "US Treasury 10Y"),
    "us_30y"    : ("DGS30",   "D", "US Treasury 30Y"),
    "us_3m"     : ("DGS3MO",  "D", "US Treasury 3M"),
    "us_10y2y"  : ("T10Y2Y",  "D", "US 10Y-2Y Spread"),
    "us_10y3m"  : ("T10Y3M",  "D", "US 10Y-3M Spread"),
    "us_bei_10y": ("T10YIE",  "D", "10Y Breakeven Inflation"),
    "us_bei_5y5y": ("T5YIFR", "D", "5Y5Y Forward Inflation Expectation"),
    "us_real_10y": ("DFII10", "D", "10Y TIPS (Real Yield)"),
}

FRED_US_MACRO = {
    "us_cpi"     : ("CPIAUCSL",   "M", "US CPI (SA)"),
    "us_cpi_core": ("CPILFESL",   "M", "US Core CPI (SA)"),
    "us_pce"     : ("PCEPI",      "M", "US PCE Price Index"),
    "us_pce_core": ("PCEPILFE",   "M", "US Core PCE"),
    "us_indpro"  : ("INDPRO",     "M", "US Industrial Production"),
    "us_unrate"  : ("UNRATE",     "M", "US Unemployment Rate"),
    "us_payems"  : ("PAYEMS",     "M", "US Nonfarm Payrolls"),
    "us_retail"  : ("RSAFS",      "M", "US Retail Sales"),
    "us_ism_mfg" : ("MANEMP",     "M", "US Manufacturing Employment (ISM proxy)"),
}

FRED_GLOBAL = {
    "wti"        : ("DCOILWTICO", "D", "WTI Crude Oil"),
    "brent"      : ("DCOILBRENTEU", "D", "Brent Crude Oil"),
    "dxy_broad"  : ("DTWEXBGS",   "D", "Broad USD Index"),
    "vixcls"     : ("VIXCLS",     "D", "VIX (FRED daily close)"),
    "usdkrw_fred": ("DEXKOUS",    "D", "KRW/USD (noon NY)"),
    "tedrate"    : ("TEDRATE",    "D", "TED Spread"),
}

FRED_US_CREDIT = {
    # ICE BofA Option-Adjusted Spread (단위: %; 1.0 = 100bp)
    "us_ig_oas"   : ("BAMLC0A0CMOAS", "D", "US IG Corp OAS (전체 투자등급)"),
    "us_aaa_oas"  : ("BAMLC0A1CAAA",  "D", "US AAA Corp OAS"),
    "us_aa_oas"   : ("BAMLC0A2CAA",   "D", "US AA Corp OAS"),
    "us_a_oas"    : ("BAMLC0A3CA",    "D", "US A Corp OAS"),
    "us_bbb_oas"  : ("BAMLC0A4CBBB",  "D", "US BBB Corp OAS"),
    "us_hy_oas"   : ("BAMLH0A0HYM2",  "D", "US HY Corp OAS"),
    "us_ig_yield" : ("BAMLC0A0CMEY",  "D", "US IG Corp Effective Yield"),
    "us_hy_yield" : ("BAMLH0A0HYM2EY","D", "US HY Corp Effective Yield"),
}

FRED_GROUPS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "us_rates" : FRED_US_RATES,
    "us_macro" : FRED_US_MACRO,
    "us_global": FRED_GLOBAL,
    "us_credit": FRED_US_CREDIT,
}


# ═══════════════════════════════════════════════════════════════
# Yahoo Finance 정의 — 일별 종가
# ═══════════════════════════════════════════════════════════════

YAHOO_SERIES = {
    "yh_us10y"  : ("^TNX",     "US 10Y T-Note Yield (Yahoo, %)"),
    "yh_us2y"   : ("^IRX",     "US 13W T-Bill (proxy)"),
    "yh_vix"    : ("^VIX",     "VIX"),
    "yh_dxy"    : ("DX-Y.NYB", "DXY"),
    "yh_sp500"  : ("^GSPC",    "S&P 500"),
    "yh_kospi"  : ("^KS11",    "KOSPI"),
    "yh_kospi200": ("^KS200",  "KOSPI 200"),
    "yh_wti"    : ("CL=F",     "WTI Futures"),
    "yh_gold"   : ("GC=F",     "Gold Futures"),
    "yh_usdkrw" : ("KRW=X",    "USD/KRW (Yahoo FX)"),
}


# ═══════════════════════════════════════════════════════════════
# ECOS Fetcher
# ═══════════════════════════════════════════════════════════════

def _fmt_period(dt: datetime, cycle: str) -> str:
    if cycle == "D":
        return dt.strftime("%Y%m%d")
    if cycle == "M":
        return dt.strftime("%Y%m")
    if cycle == "Q":
        return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"
    return dt.strftime("%Y%m%d")


def _bok_fetch_series(stat: str, cycle: str, item: str,
                       start: str, end: str,
                       verbose: bool = False) -> Optional[pd.Series]:
    s_start = pd.to_datetime(start) if len(start) == 8 else pd.to_datetime(start, format="%Y%m")
    s_end   = pd.to_datetime(end)   if len(end)   == 8 else pd.to_datetime(end,   format="%Y%m")
    p_start = _fmt_period(s_start, cycle)
    p_end   = _fmt_period(s_end,   cycle)

    url = (f"{BOK_BASE_URL}/StatisticSearch/{BOK_API_KEY}/json/kr"
           f"/1/99999/{stat}/{cycle}/{p_start}/{p_end}/{item}")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        d = r.json()
        if "StatisticSearch" not in d:
            if verbose:
                print(f"    [no StatisticSearch] {d}")
            return None
        rows = d["StatisticSearch"].get("row", [])
        records: Dict[pd.Timestamp, float] = {}
        for row in rows:
            val = row.get("DATA_VALUE", "").strip()
            if not val or val in ("-", ".."):
                continue
            try:
                t = row["TIME"]
                if cycle == "D":
                    dt = pd.to_datetime(t, format="%Y%m%d")
                elif cycle == "M":
                    dt = pd.to_datetime(t, format="%Y%m")
                elif cycle == "Q":
                    yr = int(t[:4]); q = int(t[-1])
                    dt = pd.Timestamp(year=yr, month=q * 3, day=1)
                else:
                    dt = pd.to_datetime(t)
                records[dt] = float(val)
            except Exception:
                continue
        return pd.Series(records).sort_index() if records else None
    except Exception as e:
        if verbose:
            print(f"    [exception] {e}")
        return None


def fetch_ecos_group(group_name: str,
                      start: str = "20140101",
                      end: Optional[str] = None,
                      verbose: bool = True
                      ) -> Tuple[pd.DataFrame, List[Tuple[str, str, str, str]]]:
    if end is None:
        end = datetime.today().strftime("%Y%m%d")
    if group_name not in ECOS_GROUPS:
        raise KeyError(f"unknown ECOS group: {group_name}")

    series_defs = ECOS_GROUPS[group_name]
    frames: Dict[str, pd.Series] = {}
    log: List[Tuple[str, str, str, str]] = []

    if verbose:
        print(f"  [ECOS:{group_name}] {len(series_defs)} 시리즈 시도...")

    for key, (stat, cyc, item, label) in series_defs.items():
        s = _bok_fetch_series(stat, cyc, item, start, end, verbose=False)
        if s is not None and not s.empty:
            frames[key] = s
            log.append((key, "OK", str(len(s)), label))
            if verbose:
                rng = f"{s.index[0].date()}~{s.index[-1].date()}"
                print(f"    ✓ {key:<18} {len(s):>5}obs  {rng}  {label}")
        else:
            log.append((key, "FAIL", "0", label))
            if verbose:
                print(f"    ✗ {key:<18} (stat={stat} item={item})  {label}")
        time.sleep(REQUEST_DELAY)

    if not frames:
        return pd.DataFrame(), log
    return pd.DataFrame(frames).sort_index(), log


# ═══════════════════════════════════════════════════════════════
# FRED Fetcher — API 키 우선, CSV 엔드포인트 폴백
# ═══════════════════════════════════════════════════════════════

def _fred_api_fetch(series_id: str, start: str, end: str,
                     api_key: str, verbose: bool = False) -> Optional[pd.Series]:
    s_start = pd.to_datetime(start).strftime("%Y-%m-%d")
    s_end   = pd.to_datetime(end).strftime("%Y-%m-%d")
    url = f"{FRED_BASE_URL}/series/observations"
    params = {
        "series_id"        : series_id,
        "api_key"          : api_key,
        "file_type"        : "json",
        "observation_start": s_start,
        "observation_end"  : s_end,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        records: Dict[pd.Timestamp, float] = {}
        for row in obs:
            val = row.get("value", "")
            if val in (".", "", None):
                continue
            try:
                dt = pd.to_datetime(row["date"])
                records[dt] = float(val)
            except Exception:
                continue
        return pd.Series(records).sort_index() if records else None
    except Exception as e:
        if verbose:
            print(f"    [FRED API exception] {series_id}: {e}")
        return None


def _fred_csv_fetch(series_id: str, start: str, end: str,
                     verbose: bool = False) -> Optional[pd.Series]:
    """공식 키 없이 fredgraph.csv 엔드포인트 사용 (rate-limit 더 빡셈)."""
    s_start = pd.to_datetime(start).strftime("%Y-%m-%d")
    s_end   = pd.to_datetime(end).strftime("%Y-%m-%d")
    params = {"id": series_id, "cosd": s_start, "coed": s_end}
    try:
        r = requests.get(FRED_CSV_URL, params=params,
                         headers={"User-Agent": "Mozilla/5.0"},
                         timeout=30)
        if r.status_code != 200 or not r.text.strip().startswith("observation_date") and not r.text.strip().startswith("DATE"):
            if verbose:
                print(f"    [FRED CSV bad response] {series_id} status={r.status_code}")
            # 그래도 파싱 시도
        df = pd.read_csv(StringIO(r.text))
        if df.empty or df.shape[1] < 2:
            return None
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col, val_col])
        if df.empty:
            return None
        s = pd.Series(df[val_col].values, index=df[date_col].values).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s
    except Exception as e:
        if verbose:
            print(f"    [FRED CSV exception] {series_id}: {e}")
        return None


def _fred_fetch(series_id: str, start: str, end: str,
                 api_key: Optional[str], verbose: bool = False) -> Optional[pd.Series]:
    if api_key:
        s = _fred_api_fetch(series_id, start, end, api_key, verbose=verbose)
        if s is not None and not s.empty:
            return s
        if verbose:
            print(f"    [FRED] API 빈응답 → CSV 폴백: {series_id}")
    return _fred_csv_fetch(series_id, start, end, verbose=verbose)


def fetch_fred_group(group_name: str,
                      start: str = "20140101",
                      end: Optional[str] = None,
                      api_key: Optional[str] = None,
                      verbose: bool = True
                      ) -> Tuple[pd.DataFrame, List[Tuple[str, str, str, str]]]:
    if end is None:
        end = datetime.today().strftime("%Y%m%d")
    if group_name not in FRED_GROUPS:
        raise KeyError(f"unknown FRED group: {group_name}")
    if api_key is None:
        api_key = FRED_API_KEY

    series_defs = FRED_GROUPS[group_name]
    frames: Dict[str, pd.Series] = {}
    log: List[Tuple[str, str, str, str]] = []

    if verbose:
        mode = "API" if api_key else "CSV(no-key)"
        print(f"  [FRED:{group_name}] {len(series_defs)} 시리즈 ({mode})...")

    for key, (sid, freq, label) in series_defs.items():
        s = _fred_fetch(sid, start, end, api_key=api_key, verbose=False)
        if s is not None and not s.empty:
            frames[key] = s
            log.append((key, "OK", str(len(s)), label))
            if verbose:
                rng = f"{s.index[0].date()}~{s.index[-1].date()}"
                print(f"    ✓ {key:<14} {len(s):>5}obs  {rng}  {label}")
        else:
            log.append((key, "FAIL", "0", label))
            if verbose:
                print(f"    ✗ {key:<14} (id={sid})  {label}")
        time.sleep(REQUEST_DELAY)

    if not frames:
        return pd.DataFrame(), log
    return pd.DataFrame(frames).sort_index(), log


# ═══════════════════════════════════════════════════════════════
# Yahoo Finance Fetcher
# ═══════════════════════════════════════════════════════════════

def _yahoo_chart(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    if len(start) == 8 and "-" not in start:
        start = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    if len(end) == 8 and "-" not in end:
        end = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": p1, "period2": p2, "interval": "1d", "events": "history"}
    try:
        r = requests.get(url, params=params,
                         headers={"User-Agent": "Mozilla/5.0"},
                         timeout=20)
        if r.status_code != 200:
            return None
        j = r.json()
        res = j.get("chart", {}).get("result")
        if not res:
            return None
        res = res[0]
        ts = res.get("timestamp", [])
        quote = res.get("indicators", {}).get("quote", [{}])[0]
        closes = quote.get("close", [])
        if not ts or not closes:
            return None
        idx = pd.to_datetime(ts, unit="s").normalize()
        s = pd.Series(closes, index=idx, name=symbol).dropna()
        s = s[~s.index.duplicated(keep="last")]
        return s
    except Exception:
        return None


def fetch_yahoo(start: str = "20140101",
                 end: Optional[str] = None,
                 series: Optional[Dict[str, Tuple[str, str]]] = None,
                 verbose: bool = True
                 ) -> Tuple[pd.DataFrame, List[Tuple[str, str, str, str]]]:
    if end is None:
        end = datetime.today().strftime("%Y%m%d")
    if series is None:
        series = YAHOO_SERIES

    if verbose:
        print(f"  [Yahoo] {len(series)} 티커 (raw chart API)...")

    frames: Dict[str, pd.Series] = {}
    log: List[Tuple[str, str, str, str]] = []
    for key, (ticker, label) in series.items():
        s = _yahoo_chart(ticker, start, end)
        if s is None or s.empty:
            log.append((key, "FAIL", "0", label))
            if verbose:
                print(f"    ✗ {key:<14} ({ticker})  {label}")
        else:
            s.name = key
            frames[key] = s
            log.append((key, "OK", str(len(s)), label))
            if verbose:
                rng = f"{s.index[0].date()}~{s.index[-1].date()}"
                print(f"    ✓ {key:<14} {len(s):>5}obs  {rng}  ({ticker})")
        time.sleep(0.15)

    if not frames:
        return pd.DataFrame(), log
    return pd.DataFrame(frames).sort_index(), log


# ═══════════════════════════════════════════════════════════════
# Excel — 채권운용부문 공유 파일 (Bloomberg 출처 미국 금리/스프레드)
# ═══════════════════════════════════════════════════════════════

US_XL_PATH = r"\\172.16.130.210\채권운용부문\USIGAIQ\83020N\Trading tools\금리와스프레드_BB_update.xlsx"

US_XL_SHEET3_USECOLS = "E:G"   # DATE, USGG10YR(미 10Y), LUACOAS(IG OAS)
US_XL_SHEET5_USECOLS = "A:F"   # DATE, 5종 장단기 스프레드


def fetch_us_xl(path: str = US_XL_PATH,
                  verbose: bool = True
                  ) -> Tuple[pd.DataFrame, List[Tuple[str, str, str, str]]]:
    """Sheet3(E:G) 미 10Y / IG OAS + Sheet5(A:F) 미 장단기 스프레드 5종 통합.

    컬럼 매핑
    ─────────
      us_10y_xl       ← USGG10YR Index   (Bloomberg)
      us_ig_oas_xl    ← LUACOAS  Index   (Bloomberg)
      us_spr_2_10     ← SPR_02_10
      us_spr_2_5      ← SPR_02_05
      us_spr_2_30     ← SPR_02_30
      us_spr_10_30    ← SPR_10_30
      us_spr_5_30     ← SPR_05_30
    """
    log: List[Tuple[str, str, str, str]] = []
    if not Path(path).exists():
        if verbose:
            print(f"  [Excel] 파일 접근 불가: {path}")
        log.append(("us_xl_file", "FAIL", "0", path))
        return pd.DataFrame(), log

    if verbose:
        print(f"  [Excel] {path}")

    try:
        s3 = pd.read_excel(path, sheet_name="Sheet3",
                            usecols=US_XL_SHEET3_USECOLS, header=0)
        s3.columns = ["date", "us_10y_xl", "us_ig_oas_xl"]
        s3["date"] = pd.to_datetime(s3["date"], errors="coerce")
        s3 = s3.dropna(subset=["date"]).set_index("date").sort_index()
        s3 = s3.apply(pd.to_numeric, errors="coerce")
        log.append(("us_xl_sheet3", "OK", str(len(s3)),
                    "us_10y_xl + us_ig_oas_xl"))
        if verbose:
            print(f"    ✓ Sheet3  {len(s3)}obs  "
                   f"{s3.index[0].date()}~{s3.index[-1].date()}")
    except Exception as e:
        log.append(("us_xl_sheet3", "FAIL", "0", str(e)[:80]))
        s3 = pd.DataFrame()
        if verbose:
            print(f"    ✗ Sheet3 read err: {e}")

    try:
        s5 = pd.read_excel(path, sheet_name="Sheet5",
                            usecols=US_XL_SHEET5_USECOLS, header=0)
        s5.columns = ["date", "us_spr_2_10", "us_spr_2_5",
                      "us_spr_2_30", "us_spr_10_30", "us_spr_5_30"]
        s5["date"] = pd.to_datetime(s5["date"], errors="coerce")
        s5 = s5.dropna(subset=["date"]).set_index("date").sort_index()
        s5 = s5.apply(pd.to_numeric, errors="coerce")
        log.append(("us_xl_sheet5", "OK", str(len(s5)),
                    "us_spr_2_10/2_5/2_30/10_30/5_30"))
        if verbose:
            print(f"    ✓ Sheet5  {len(s5)}obs  "
                   f"{s5.index[0].date()}~{s5.index[-1].date()}")
    except Exception as e:
        log.append(("us_xl_sheet5", "FAIL", "0", str(e)[:80]))
        s5 = pd.DataFrame()
        if verbose:
            print(f"    ✗ Sheet5 read err: {e}")

    if s3.empty and s5.empty:
        return pd.DataFrame(), log
    if s3.empty:
        df = s5
    elif s5.empty:
        df = s3
    else:
        df = pd.concat([s3, s5], axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, log


# ═══════════════════════════════════════════════════════════════
# 통합 로더
# ═══════════════════════════════════════════════════════════════

# 증분 수집 시 마지막 날짜로부터 거슬러 다시 받는 버퍼(일).
# 일별: 최근 수정/지연 반영. 월별: 직전 1~2개월 재확인.
INCREMENTAL_BUFFER_DAYS = 45


def _merge_incremental(cached: pd.DataFrame,
                        fresh: Optional[pd.DataFrame]) -> pd.DataFrame:
    """기존 캐시 + 신규를 인덱스·컬럼 기준 병합.
       - 신규 우선, 단 신규가 NaN인 칸은 기존 값 유지 (combine_first).
       - 시리즈가 일시 실패해 fresh에서 컬럼/값이 빠져도 과거 데이터 보존.
    """
    if fresh is None or fresh.empty:
        return cached
    combined = fresh.combine_first(cached).sort_index()
    # 컬럼 순서: 기존 캐시 순서 우선 + fresh에만 있는 신규 컬럼 뒤에
    ordered = list(cached.columns) + [c for c in fresh.columns
                                       if c not in cached.columns]
    combined = combined.reindex(columns=ordered)
    return combined


def _load_group(group: str, cache: Path, fetch_fn,
                 start: str, end: str, refresh: bool, verbose: bool,
                 source: str) -> Tuple[pd.DataFrame, list]:
    """그룹 단위 로드.
       - refresh=True 또는 캐시 없음 → 전체 수집
       - refresh=False & 캐시 有     → 증분(마지막일자 − buffer ~ end)만 받아 병합
       fetch_fn(start, end, verbose) -> (df, log)
    """
    if refresh or not cache.exists():
        if verbose:
            mode = "전체 재수집" if refresh else "최초 수집"
            print(f"\n  [{group}] {source} {mode}:")
        df, log = fetch_fn(start, end, verbose)
        if not df.empty:
            df.to_parquet(cache)
        return df, log

    # 증분
    try:
        cached = pd.read_parquet(cache)
    except Exception as e:
        if verbose:
            print(f"\n  [{group}] 캐시 읽기 실패({e}) → 전체 재수집")
        df, log = fetch_fn(start, end, verbose)
        if not df.empty:
            df.to_parquet(cache)
        return df, log

    if cached.empty:
        df, log = fetch_fn(start, end, verbose)
        if not df.empty:
            df.to_parquet(cache)
        return df, log

    last_date = cached.index.max()
    inc_start = (last_date - timedelta(days=INCREMENTAL_BUFFER_DAYS)).strftime("%Y%m%d")
    if verbose:
        print(f"\n  [{group}] {source} 증분 (캐시 end={last_date.date()}, "
               f"{inc_start}~{end} 재요청):")
    fresh, log = fetch_fn(inc_start, end, verbose)
    merged = _merge_incremental(cached, fresh)
    n_new = len(merged) - len(cached)
    new_last = merged.index.max()
    if n_new > 0 or new_last > last_date:
        merged.to_parquet(cache)
    if verbose:
        print(f"      → +{n_new}행, data_end {last_date.date()} → {new_last.date()}")
    log = [("(incremental)", "OK", f"+{n_new}", f"end={new_last.date()}")] + (log or [])
    return merged, log


def load_all_data(start: str = "20140101",
                   end: Optional[str] = None,
                   cache_dir: Path = CACHE_DIR,
                   refresh: bool = False,
                   verbose: bool = True
                   ) -> Dict[str, pd.DataFrame]:
    """ECOS + FRED + Yahoo 전체 그룹 수집. 캐시 우선, refresh=True면 재수집.

    Returns
    -------
    {
      "rates_ktb": DataFrame, "rates_credit": DataFrame,
      "macro_growth": DataFrame, "macro_price": DataFrame,
      "macro_money": DataFrame, "macro_sentiment": DataFrame,
      "fx": DataFrame, "equity": DataFrame,
      "us_rates": DataFrame, "us_macro": DataFrame, "us_global": DataFrame,
      "global_yahoo": DataFrame,
      "_logs": {group: [(key, status, n_obs, label)]},
    }
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, pd.DataFrame] = {}
    logs: Dict[str, list] = {}

    if verbose:
        print("━" * 70)
        print(f"  KBOND 데이터 수집  start={start}  end={end or 'today'}")
        print(f"  캐시: {cache_dir}")
        print(f"  FRED: {'API key (inline)' if FRED_API_KEY else 'CSV (no key)'}")
        print("━" * 70)

    # ECOS
    for group in ECOS_GROUPS.keys():
        cache = cache_dir / f"{group}.parquet"
        df, log = _load_group(
            group, cache,
            lambda s, e, v, g=group: fetch_ecos_group(g, start=s, end=e, verbose=v),
            start, end, refresh, verbose, source="ECOS")
        out[group] = df
        logs[group] = log

    # FRED
    for group in FRED_GROUPS.keys():
        cache = cache_dir / f"{group}.parquet"
        df, log = _load_group(
            group, cache,
            lambda s, e, v, g=group: fetch_fred_group(g, start=s, end=e, verbose=v),
            start, end, refresh, verbose, source="FRED")
        out[group] = df
        logs[group] = log

    # Yahoo
    cache = cache_dir / "global_yahoo.parquet"
    df, log = _load_group(
        "global_yahoo", cache,
        lambda s, e, v: fetch_yahoo(start=s, end=e, verbose=v),
        start, end, refresh, verbose, source="Yahoo")
    out["global_yahoo"] = df
    logs["global_yahoo"] = log

    # Excel us_xl (Bloomberg 미 금리·OAS·장단기 스프레드)
    cache = cache_dir / "us_xl.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        out["us_xl"] = df
        logs["us_xl"] = [("(cache)", "OK", str(df.shape[0]),
                          f"{df.shape[1]}컬럼")]
        if verbose:
            print(f"\n  [us_xl] (cache) {df.shape}")
    else:
        if verbose:
            print(f"\n  [us_xl] Excel 수집:")
        df, log = fetch_us_xl(verbose=verbose)
        out["us_xl"] = df
        logs["us_xl"] = log
        if not df.empty:
            df.to_parquet(cache)

    out["_logs"] = logs

    if verbose:
        print("\n" + "━" * 70)
        print("  수집 요약")
        print("━" * 70)
        for group, df in out.items():
            if group == "_logs":
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                rng = f"{df.index[0].date()}~{df.index[-1].date()}"
                print(f"  {group:<18} shape={str(df.shape):<14}  {rng}")
            else:
                print(f"  {group:<18} (비어있음)")

    return out


if __name__ == "__main__":
    import sys, io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    data = load_all_data(start="20140101", refresh=False)

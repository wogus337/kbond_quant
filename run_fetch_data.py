"""
run_fetch_data.py
─────────────────
KBOND ActiveQuant 데이터 수집 실행 스크립트.

사용
────
    # 캐시 우선 (빠름, 두번째 실행부터)
    python run_fetch_data.py

    # 전체 재수집 (캐시 무시)
    python run_fetch_data.py --refresh

    # 기간/끝 지정
    python run_fetch_data.py --start 20100101 --end 20260512

    # FRED API 키
    set FRED_API_KEY=xxxx
    python run_fetch_data.py

출력
────
    data_cache/{group}.parquet            — 원본 parquet 캐시
    data_cache/_logs.csv                  — 시리즈별 수집 상태표
    data_cache/_summary.csv               — 그룹별 (rows, cols, start, end) 요약
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# UTF-8 출력 (Windows 콘솔)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)
except Exception:
    pass

from data_fetcher import CACHE_DIR, load_all_data


def main():
    ap = argparse.ArgumentParser(description="KBOND 데이터 수집")
    ap.add_argument("--start", default="20100101", help="시작일 YYYYMMDD")
    ap.add_argument("--end",   default=None,       help="종료일 YYYYMMDD (기본=오늘)")
    ap.add_argument("--refresh", action="store_true",
                    help="전체 재수집(start부터). 미지정 시 증분(캐시 마지막일자 이후만)")
    ap.add_argument("--quiet", action="store_true", help="진행 로그 최소화")
    args = ap.parse_args()

    end = args.end or datetime.today().strftime("%Y%m%d")

    data = load_all_data(
        start=args.start, end=end,
        cache_dir=CACHE_DIR,
        refresh=args.refresh,
        verbose=not args.quiet,
    )

    # ── 진단 표 저장 ───────────────────────────────────────────
    logs = data.get("_logs", {})
    rows = []
    for group, items in logs.items():
        for it in items:
            key, status, n_obs, label = it
            rows.append({
                "group": group, "key": key, "status": status,
                "n_obs": n_obs, "label": label,
            })
    if rows:
        log_df = pd.DataFrame(rows)
        log_path = CACHE_DIR / "_logs.csv"
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"\n  [diag] 시리즈별 상태표: {log_path}")

    # 그룹 요약
    sum_rows = []
    for group, df in data.items():
        if group == "_logs":
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            sum_rows.append({
                "group": group,
                "rows": df.shape[0],
                "cols": df.shape[1],
                "start": df.index[0].date().isoformat(),
                "end"  : df.index[-1].date().isoformat(),
                "cols_list": ",".join(df.columns.astype(str)),
            })
        else:
            sum_rows.append({
                "group": group, "rows": 0, "cols": 0,
                "start": "", "end": "", "cols_list": "",
            })
    if sum_rows:
        sum_df = pd.DataFrame(sum_rows)
        sum_path = CACHE_DIR / "_summary.csv"
        sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")
        print(f"  [diag] 그룹별 요약    : {sum_path}")

    # 실패 시리즈 강조
    fails = [(g, it) for g, items in logs.items() for it in items if it[1] == "FAIL"]
    if fails:
        print(f"\n  [경고] 실패 시리즈 {len(fails)}개:")
        for g, it in fails:
            print(f"    - [{g}] {it[0]}  ({it[3]})")
    else:
        print(f"\n  [OK] 모든 시리즈 수집 성공.")


if __name__ == "__main__":
    main()

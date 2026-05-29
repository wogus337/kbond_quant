"""데이터 업로드 · 갱신 · phase3 재실행 · phase1 재학습."""
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from lib import PROJECT_ROOT, CACHE_DIR, cache_file_status, phase_artifacts_status
from theme import apply_theme, banner, ORANGE, NAVY
from auth import require_login

apply_theme()
require_login("데이터 · 모델 갱신")
banner("데이터 · 모델 갱신")

PHASE1_LOG = PROJECT_ROOT / "phase1_retrain.log"
PHASE1_PID = PROJECT_ROOT / ".phase1_retrain.pid"


def _pid_alive(pid: int) -> bool:
    """Windows에서 PID 살아있는지 (psutil 없으면 os.kill 트릭)."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, PermissionError):
            return False


import re


def _run_streaming(args: list, timeout_s: int = 3600,
                     total_pat: str = None, item_pat: str = None,
                     log_height: int = 380) -> int:
    """subprocess를 실시간 스트리밍 + 옵션 진행 % 표시.

    timeout_s 기본 1시간 (FRED API 느릴 때 대비).
    total_pat: 라인에서 추출한 정수를 '총 단위'에 누적 가산하는 regex (group 1).
    item_pat : 매칭되면 '완료 단위' +1.
    로그는 st.code(height=…)로 전체 보존 + 스크롤·드래그 가능.
    """
    full_args = [sys.executable, "-u"] + args
    progress_widget = st.empty() if (total_pat and item_pat) else None
    log_widget = st.empty()
    lines: list[str] = []
    total, done = 0, 0
    fails = 0
    try:
        proc = subprocess.Popen(
            full_args, cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except Exception as e:
        st.error(f"프로세스 실행 실패: {type(e).__name__}: {e}")
        return -1

    start = datetime.now()
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            # 진행 파싱
            if total_pat:
                m = re.search(total_pat, line)
                if m:
                    try: total += int(m.group(1))
                    except Exception: pass
            if item_pat and re.search(item_pat, line):
                done += 1
                if "✗" in line:
                    fails += 1
            # 진행바 갱신
            if progress_widget is not None:
                elapsed = int((datetime.now() - start).total_seconds())
                if total > 0:
                    pct = min(done / total, 1.0)
                    progress_widget.progress(
                        pct, text=f"{done}/{total} ({pct*100:.0f}%) · 실패 {fails} · 경과 {elapsed}s")
                else:
                    progress_widget.progress(
                        0.0, text=f"수집 시작... 경과 {elapsed}s")
            # 로그(전체 보존, 스크롤·드래그 가능)
            elapsed = int((datetime.now() - start).total_seconds())
            log_widget.code(
                f"[경과 {elapsed}s · {len(lines)}줄]\n" + "\n".join(lines),
                language="text", height=log_height)
            if (datetime.now() - start).total_seconds() > timeout_s:
                proc.kill()
                lines.append(f"[timeout] {timeout_s}s 초과 → 중단")
                log_widget.code("\n".join(lines), language="text", height=log_height)
                return -1
        proc.wait(timeout=30)
    except Exception as e:
        lines.append(f"[error] {e}")
        log_widget.code("\n".join(lines), language="text", height=log_height)
        return -1

    if progress_widget is not None and total > 0:
        progress_widget.progress(min(done/total, 1.0),
                                   text=f"완료 {done}/{total} (실패 {fails})")
    return proc.returncode


def _launch_phase1_detached(extra_args: list) -> int:
    """phase1_train.py 를 detached 백그라운드 실행. PID 반환."""
    log_f = open(PHASE1_LOG, "w", encoding="utf-8")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [sys.executable, "-u", str(PROJECT_ROOT / "phase1_train.py"), *extra_args],
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log_f, stderr=subprocess.STDOUT,
        creationflags=creationflags,
        close_fds=True,
    )
    PHASE1_PID.write_text(str(proc.pid))
    return proc.pid

# ─── Bloomberg Excel 업로드 ───
st.markdown("### 1. Bloomberg Excel 업로드")
st.caption("`금리와스프레드_BB_update.xlsx` 업로드 → `data_cache/us_xl.parquet` 갱신")

uploaded = st.file_uploader("Excel 파일", type=["xlsx"], key="xlsx",
                              label_visibility="collapsed")
if uploaded is not None:
    if st.button("➤ us_xl.parquet 갱신", type="primary"):
        with st.spinner("Excel 파싱 중..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                from data_fetcher import fetch_us_xl
                df, log = fetch_us_xl(path=tmp_path, verbose=False)
                if df.empty:
                    st.error("파싱 실패. 시트 구조를 확인해주세요.")
                    st.write(pd.DataFrame(log, columns=["key","status","n","note"]))
                else:
                    out = CACHE_DIR / "us_xl.parquet"
                    df.to_parquet(out)
                    st.success(f"성공: {len(df)} 행 → {out.name}")
                    st.dataframe(df.tail(8), use_container_width=True)
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"오류: {e}")
            finally:
                try: os.unlink(tmp_path)
                except Exception: pass

st.divider()


# ─── 캐시 클리어 (산출물 갱신 후 화면 즉시 반영) ───
st.markdown("### 화면 캐시 클리어")
st.caption("phase1 재학습/phase3 재실행 후 새 결과가 안 보이면 클릭. "
            "Streamlit 캐시(`@st.cache_data`)를 비워 모든 페이지가 디스크의 최신 산출물을 다시 읽음.")
if st.button("🔄 캐시 클리어 + 새로고침"):
    st.cache_data.clear()
    st.success("캐시 비움. 자동 새로고침...")
    st.rerun()

st.divider()


# ─── BOK/FRED/Yahoo 자동 갱신 ───
st.markdown("### 2. API 데이터 자동 갱신")
st.caption("BOK ECOS · FRED · Yahoo Finance 전부 실행됩니다.")

c1, c2 = st.columns(2)
incr = c1.button("↻ 증분 갱신", use_container_width=True, type="primary")
full = c2.button("⟳ 전체 재수집", use_container_width=True)
st.caption("• **증분 갱신**: 각 그룹 캐시 마지막 날짜 이후만 받아 병합(최근 45일 재확인). "
            "약 90개 시리즈 순차 호출. 일반적으로 **2~4분**이지만 FRED가 느리면 **수십 분**까지 걸릴 수 있음 (timeout 1시간).  \n"
            "• **전체 재수집**: 2010년부터 전체 히스토리 재다운로드. **5~10분**. 데이터 오류 의심 시에만.  \n"
            "_진행률 바 + 전체 로그(스크롤·드래그 가능)가 아래에 표시됩니다 — 멈춘 게 아니라 API 응답 대기 중입니다._")

if incr or full:
    args = [str(PROJECT_ROOT / "run_fetch_data.py")]
    if full:
        args.append("--refresh")
    with st.spinner("API 호출 중... (약 90개 시리즈 순차 수집, 진행상황 아래 표시)"):
        # 시리즈 시도 카운트(예: "11 시리즈 시도") + 개별 ✓/✗ 라인으로 진행% 산출
        rc = _run_streaming(
            args, timeout_s=3600,
            total_pat=r"\]\s+(\d+)\s+시리즈\s+시도",
            item_pat=r"^\s+(✓|✗)\s")
    if rc == 0:
        st.success("완료. 캐시 갱신됨.")
        st.cache_data.clear()
    else:
        st.error(f"실패 또는 중단 (exit {rc})")

st.divider()


# ─── phase3 재실행 ───
st.markdown("### 3. phase3 재실행 (최신 시그널 갱신)")
st.caption("`model_selection.csv` 의 selected 모델을 최신 데이터로 재학습 → P(persist) 갱신 · 3~5분")

if st.button("▶ phase3_predict.py 실행", type="primary"):
    with st.spinner("phase3 실행 중... (선택 모델 재학습 + 시그널 산출, 진행상황 아래 표시)"):
        rc = _run_streaming([str(PROJECT_ROOT / "phase3_predict.py")], timeout_s=900)
    if rc == 0:
        st.success("완료. 듀레이션 / 장단기스프레드 페이지에서 새 시그널 확인.")
        st.cache_data.clear()
    else:
        st.error(f"실패 또는 중단 (exit {rc})")

st.divider()


# ─── phase1 재학습 ───
st.markdown("### 4. phase1 전체 재학습 (모델 자체)")
st.caption("HMM regime + 6개 분류기 × HPO × walk-forward — 분기 1회 권장.")

# 실행 중 상태 체크
running = False
if PHASE1_PID.exists():
    try:
        pid = int(PHASE1_PID.read_text().strip())
        running = _pid_alive(pid)
    except Exception:
        running = False

if running:
    log_mtime = datetime.fromtimestamp(PHASE1_LOG.stat().st_mtime) if PHASE1_LOG.exists() else None
    st.success(f"⚡ phase1 재학습 백그라운드 실행 중  (PID {pid})  "
                 f"마지막 로그 갱신: {log_mtime.strftime('%H:%M:%S') if log_mtime else '?'}")
    if st.button("진행 로그 새로고침"):
        st.rerun()
    if PHASE1_LOG.exists():
        tail = "\n".join(PHASE1_LOG.read_text(encoding="utf-8", errors="replace")
                            .splitlines()[-30:])
        st.code(tail or "(아직 출력 없음)", language="text")
else:
    # 실행 옵션 표
    st.markdown("""
| 옵션 | 대상 | HPO trials | 예상 시간 (단일 코어) |
|---|---|---|---|
| **빠른 테스트** | ktb_10y 1개 | 5 | 약 **15-20분** |
| **KTB 4타겟 (금리만)** | ktb_3y/5y/10y/30y | 20 | 약 **12-18시간** |
| **KTB 7타겟 (금리+스프레드)** | KTB 7개 | 20 | 약 **24-30시간** |
| **전체 15타겟** | 모든 시리즈 | 20 | 약 **48-60시간** |
""")
    st.info("💡 백그라운드 detached 실행. 앱·PC를 닫아도 학습은 계속 진행됨. "
              "진행은 `phase1_retrain.log` 파일에 누적 (로그는 이 페이지에서도 확인).")
    st.warning("⚠ 실행 중에는 `predictions.parquet` 등 산출물이 계속 덮어쓰여집니다. "
                 "학습 완료 후 **데이터 → 3. phase3 재실행** 한 번 더 돌려야 시그널이 새로 갱신됩니다.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        quick = st.button("▶ 빠른 테스트", use_container_width=True)
    with c2:
        ktb4 = st.button("▶ KTB 금리 4개", use_container_width=True, type="primary")
    with c3:
        ktb7 = st.button("▶ KTB 7타겟", use_container_width=True)
    with c4:
        all15 = st.button("▶ 전체 15타겟", use_container_width=True)

    args = None
    if quick:
        args = ["--targets", "ktb_10y", "--hpo-trials", "5"]
    elif ktb4:
        args = ["--targets", "ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y"]
    elif ktb7:
        args = ["--targets", "ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y",
                  "spread_ktb_3_10", "spread_ktb_5_30", "spread_ktb_10_30"]
    elif all15:
        args = []  # 기본 ALL_TARGETS

    if args is not None:
        try:
            pid = _launch_phase1_detached(args)
            st.success(f"phase1 재학습 시작 (PID {pid}). 로그: `phase1_retrain.log`")
            st.rerun()
        except Exception as e:
            st.error(f"실행 실패: {e}")

    # 이전 실행 로그 (있으면)
    if PHASE1_LOG.exists():
        with st.expander("이전 실행 로그 보기"):
            log_mtime = datetime.fromtimestamp(PHASE1_LOG.stat().st_mtime)
            st.caption(f"마지막 수정: {log_mtime:%Y-%m-%d %H:%M}")
            tail = "\n".join(PHASE1_LOG.read_text(encoding="utf-8", errors="replace")
                                .splitlines()[-40:])
            st.code(tail or "(empty)", language="text")

st.divider()


# ─── 상태표 ───
st.markdown("### 5. 캐시 상태")
tab1, tab2 = st.tabs(["데이터 캐시", "산출물"])
with tab1:
    df = cache_file_status()
    if df.empty:
        st.warning("data_cache/ 가 비어있습니다.")
    else:
        st.dataframe(df, hide_index=True, use_container_width=True)
with tab2:
    st.dataframe(phase_artifacts_status(), hide_index=True,
                   use_container_width=True)
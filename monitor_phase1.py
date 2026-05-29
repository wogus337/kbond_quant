"""
phase1 재학습 진행상황 모니터 (실행 중인 학습은 건드리지 않고 heartbeat·ckpt만 읽음).

phase1_train.py 가 모델 1개 완료마다 .phase1_progress.json 에 진행상태를 기록 →
콘솔 실행/detached 무관하게 추적됨.

사용:
  python monitor_phase1.py            # 1회 스냅샷
  python monitor_phase1.py --follow   # 진행이 있을 때만 한 줄씩 출력 (Ctrl+C 종료)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PROG_FILE = ROOT / ".phase1_progress.json"
CKPT_DIR = ROOT / "results" / "phase1" / "_ckpt"
ALIVE_GAP = 600   # heartbeat가 이 초 이내 갱신이면 '실행 중'으로 간주


def read_state() -> dict:
    """heartbeat + ckpt 종합 상태."""
    prog = {}
    if PROG_FILE.exists():
        try:
            prog = json.loads(PROG_FILE.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    # ckpt_done: heartbeat 값 우선(phase1이 실제 ckpt_dir에서 계산), 없으면 기본 경로 glob
    if "ckpt_done" in prog:
        ckpt_done = int(prog.get("ckpt_done") or 0)
    else:
        ckpt_done = len(list(CKPT_DIR.glob("*.pkl"))) if CKPT_DIR.exists() else 0
    total = int(prog.get("total_combos", 240) or 240)
    n_models = int(prog.get("n_models", 6) or 6)
    models_done = int(prog.get("models_done", 0) or 0)
    # 전체 진행률(모델 단위): 완료조합×모델수 + 현재조합 진행모델
    done_units = ckpt_done * n_models + (models_done if prog.get("phase") == "train" else 0)
    total_units = total * n_models
    pct = (done_units / total_units * 100) if total_units else 0.0
    ts = float(prog.get("ts", 0) or 0)
    age = time.time() - ts if ts else 1e9
    phase = prog.get("phase", "")
    alive = (phase in ("start", "train") and age < ALIVE_GAP)
    return {
        "prog": prog, "ckpt_done": ckpt_done, "total": total,
        "n_models": n_models, "pct": pct, "age": age, "alive": alive,
        "phase": phase, "done_units": done_units, "total_units": total_units,
    }


def status_line(s: dict) -> str:
    p = s["prog"]
    if s["phase"] == "done":
        return f"✅ 완료  ({s['ckpt_done']}/{s['total']} 조합)"
    if not p:
        return "⚪ 시작 기록 없음 (아직 실행 안 됨)"
    if s["alive"]:
        tag = "초기화 중" if s["phase"] == "start" else "실행 중"
        st = f"🟢 {tag} (PID {p.get('pid','?')}, {int(s['age'])}s 전 갱신)"
    else:
        st = f"🔴 중단/정지 추정 (마지막 갱신 {int(s['age'])}s 전)"
    return st


def fmt_dur(sec: float) -> str:
    sec = int(max(sec, 0)); h, sec = divmod(sec, 3600); m, sec = divmod(sec, 60)
    return f"{h}시간 {m}분" if h else (f"{m}분 {sec}초" if m else f"{sec}초")


def render_snapshot(s: dict) -> None:
    p = s["prog"]
    bar_n = int(s["pct"] / 5)
    bar = "█" * bar_n + "░" * (20 - bar_n)
    print("━" * 60)
    print(f"  phase1 재학습 모니터   {datetime.now():%H:%M:%S}")
    print("━" * 60)
    print(f"  상태    : {status_line(s)}")
    print(f"  진행률  : [{bar}] {s['pct']:.1f}%  "
          f"(완료 {s['ckpt_done']}/{s['total']} 조합, 모델단위 {s['done_units']}/{s['total_units']})")
    if p and s["phase"] in ("train", "start"):
        print(f"  현재    : 조합 [{p.get('combo_no','?')}/{s['total']}] "
              f"{p.get('target','')} {p.get('config','')}")
        print(f"            모델 {p.get('models_done',0)}/{s['n_models']}  "
              f"최근완료: {p.get('last_model','')}")
        started = float(p.get("started_at", 0) or 0)
        if started and s["ckpt_done"] > 0:
            elapsed = time.time() - started
            per = elapsed / s["ckpt_done"]
            remain = (s["total"] - s["ckpt_done"]) * per
            print(f"  경과    : {fmt_dur(elapsed)}  |  조합당 평균 {fmt_dur(per)}")
            print(f"  예상잔여: {fmt_dur(remain)}  "
                  f"(완료예상 {datetime.fromtimestamp(time.time()+remain):%m-%d %H:%M})")
        elif started:
            print(f"  경과    : {fmt_dur(time.time()-started)}  (첫 조합 완료 후 ETA 표시)")
    print("━" * 60)


def follow(interval: int = 5) -> None:
    """진행(모델완료/조합완료/상태변화)이 있을 때만 한 줄 출력."""
    print("진행 추적 시작 (Ctrl+C 종료). 변화가 있을 때만 출력합니다.\n")
    last_sig = None
    last_alive = None
    try:
        while True:
            s = read_state()
            p = s["prog"]
            # 변화 시그니처: 완료조합수 + 현재조합 + 진행모델수 + phase
            sig = (s["ckpt_done"], p.get("combo_no"), p.get("models_done"),
                   s["phase"], p.get("last_model"))
            if sig != last_sig:
                ts = datetime.now().strftime("%H:%M:%S")
                if s["phase"] == "done":
                    print(f"[{ts}] ✅ 전체 완료 — {s['ckpt_done']}/{s['total']} 조합")
                    break
                bar_n = int(s["pct"] / 5)
                bar = "█" * bar_n + "░" * (20 - bar_n)
                print(f"[{ts}] [{bar}] {s['pct']:5.1f}% | "
                      f"조합 {p.get('combo_no','?')}/{s['total']} "
                      f"{p.get('target',''):<16} | "
                      f"모델 {p.get('models_done',0)}/{s['n_models']} "
                      f"{p.get('last_model','')}")
                last_sig = sig
            # 살아있다가 멈춘 경우 한 번 경고
            if last_alive is True and not s["alive"] and s["phase"] != "done":
                print(f"[{datetime.now():%H:%M:%S}] ⚠ heartbeat 끊김 "
                      f"({int(s['age'])}s) — 중단됐을 수 있음. 재개: python phase1_train.py")
            last_alive = s["alive"]
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n모니터 종료.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true",
                    help="진행이 있을 때만 한 줄씩 출력 (이벤트 방식)")
    ap.add_argument("--interval", type=int, default=5,
                    help="--follow 시 내부 폴링 간격(초). 출력은 변화 시에만.")
    args = ap.parse_args()
    if args.follow:
        follow(args.interval)
    else:
        render_snapshot(read_state())


if __name__ == "__main__":
    main()
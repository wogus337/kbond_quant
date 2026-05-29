"""
phase2_review.py  (원본 방법론 — direction별 모델 선택)
─────────────────────────────────────────────────
Phase 1 산출 metrics_top / model_selection_template 검토 후
results/phase2/model_selection.csv 생성. 각 행은 (target, direction, run_id, model) 단위.

흐름
────
1. results/phase1/metrics_top.csv 로드
2. 콘솔에 target × direction × Top-K 모델 보고
3. results/phase2/model_selection.csv 생성
   - selected 컬럼: Top-1(기본) 자동 추천. 사용자가 True/False 편집.
4. 편집 후  python phase3_predict.py  실행

사용 예
──────
  python phase2_review.py
  python phase2_review.py --top 2          # Top-2 자동 추천
  python phase2_review.py --regenerate     # 기존 selection 덮어쓰기
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PHASE1_DIR = PROJECT_DIR / "results" / "phase1"
DEFAULT_PHASE2_DIR = PROJECT_DIR / "results" / "phase2"


def main():
    ap = argparse.ArgumentParser(description="Phase2: 모델 선택 CSV (direction 포함)")
    ap.add_argument("--phase1-dir", default=str(DEFAULT_PHASE1_DIR))
    ap.add_argument("--phase2-dir", default=str(DEFAULT_PHASE2_DIR))
    ap.add_argument("--top", type=int, default=1,
                    help="(target × direction) 별 자동 selected=True 표기할 상위 N개")
    ap.add_argument("--regenerate", action="store_true",
                    help="기존 model_selection.csv 덮어쓰기")
    args = ap.parse_args()

    p1_dir = Path(args.phase1_dir)
    p2_dir = Path(args.phase2_dir)
    p2_dir.mkdir(parents=True, exist_ok=True)

    template_path = p1_dir / "model_selection_template.csv"
    if not template_path.exists():
        print(f"  [ERR] {template_path} 없음. phase1_train.py 먼저 실행.")
        return
    tmpl = pd.read_csv(template_path, encoding="utf-8-sig")

    print("━" * 80)
    print(f"  Phase 2 — direction별 모델 선택  (top-{args.top} 자동 추천)")
    print("━" * 80)

    # 콘솔 leaderboard
    for (t, d), grp in tmpl.groupby(["target", "direction"]):
        print(f"\n  {t}  [{d}]")
        top_n = grp.sort_values("rank_in_group").head(min(8, len(grp)))
        for _, r in top_n.iterrows():
            star = "★" if r["rank_in_group"] <= args.top else " "
            print(f"   {star} {int(r['rank_in_group']):>2}. {r['model']:<7} "
                  f"run={r['run_id']:<32}"
                  f"  acc={r['accuracy']:.3f}  auc={r['auc']:.3f}  "
                  f"hit60={r['hit@60']:.3f} cov60={r['cov@60']:.3f}  "
                  f"comp={r['composite']:.3f}")

    out = p2_dir / "model_selection.csv"
    if out.exists() and not args.regenerate:
        print(f"\n  [skip] {out} 이미 존재. 덮어쓰려면 --regenerate")
        existing = pd.read_csv(out, encoding="utf-8-sig")
        sel_cnt = int(existing["selected"].astype(str)
                       .str.lower().isin(["true", "1", "yes"]).sum())
        print(f"  현재 selected={sel_cnt} 모델")
    else:
        sel = tmpl.copy()
        sel["selected"] = sel["rank_in_group"] <= args.top
        # 컬럼 순서 정리
        ordered = ["target", "direction", "run_id", "model", "selected",
                   "lookback_weeks", "forward_weeks", "retrain_days",
                   "target_thr_bp", "regime_strength_min",
                   "accuracy", "auc", "hit@60", "cov@60",
                   "hit@70", "cov@70", "composite", "rank_in_group"]
        sel = sel[[c for c in ordered if c in sel.columns]]
        sel.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  [OK] model_selection.csv 생성: {out}")
        print(f"        총 {len(sel)} 행 / selected={int(sel['selected'].sum())} 모델")

    print(f"\n  사용자 편집 가이드:")
    print(f"    1. {out} 를 엑셀/텍스트에디터로 열어")
    print(f"    2. selected 컬럼을 True/False 로 조정")
    print(f"       (target × direction 당 1개 이상 True 권장)")
    print(f"    3. 저장 후  python phase3_predict.py  실행")


if __name__ == "__main__":
    main()

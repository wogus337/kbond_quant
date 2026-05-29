"""수익률곡선 전략 — 추후 구현."""
import streamlit as st

from theme import apply_theme, banner, placeholder

apply_theme()
banner("수익률곡선 전략")

placeholder(
    icon="◆",
    title="준비 중",
    desc=("수익률곡선 형태(레벨·슬로프·커버처) 변화를 활용한 전략. "
            "Butterfly·NS factor·PCA 기반 곡선 노출 모델링 예정."),
)

with st.expander("계획 (요약)"):
    st.markdown("""
- **타겟**: 2-5-10 butterfly, 3-10-30 butterfly, 곡선 PCA factor (level/slope/curvature)
- **시그널**: 곡선 형태가 균형 상태에서 이탈했을 때 mean-reversion 베팅
- **포지션**: 3개 만기 동시 매매 (DV01-neutral)
- **모델 후보**: HMM regime + factor mean-reversion + ML 보조
""")
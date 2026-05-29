"""상대가치 전략 — 추후 구현."""
import streamlit as st

from theme import apply_theme, banner, placeholder

apply_theme()
banner("상대가치 전략 (Relative Value)")

placeholder(
    icon="◆",
    title="준비 중",
    desc=("이론값 대비 시장가의 괴리 또는 동종 채권 간 상대가치를 이용한 전략. "
            "Cheapest-to-deliver, 스왑 스프레드, asset swap 등 분석 예정."),
)

with st.expander("계획 (요약)"):
    st.markdown("""
- **분석 대상**: 국고채 vs 통안채, 국고채 vs 스왑, 신·구 종목 스프레드
- **시그널**: z-score 기반 mean-reversion + 캐리 trade-off
- **모델**: 정상성 검정 + Ornstein-Uhlenbeck 모수 추정 + ML 보조
""")
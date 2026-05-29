"""PairTrading 전략 — 추후 구현."""
import streamlit as st

from theme import apply_theme, banner, placeholder

apply_theme()
banner("Pair Trading 전략")

placeholder(
    icon="◆",
    title="준비 중",
    desc=("두 채권 간 스프레드의 통계적 mean-reversion 을 이용한 long-short 전략."),
)

with st.expander("계획 (요약)"):
    st.markdown("""
- **페어 후보**: 같은 발행사 다른 만기, 동일 만기 다른 발행사 (AAA/AA/A), KTB vs 산금채
- **시그널**: cointegration test → z-score → 진입/청산 threshold
- **모델**: Engle-Granger 또는 Johansen 검정 + Kalman filter beta 추정
""")
"""섹터 전략 — 추후 구현."""
import streamlit as st

from theme import apply_theme, banner, placeholder

apply_theme()
banner("섹터 전략")

placeholder(
    icon="◆",
    title="준비 중",
    desc=("산업·신용등급별 섹터 스프레드 사이클을 활용한 over/under-weight 전략. "
            "회사채 AA/BBB 스프레드, KEPCO·산금채, 은행채 등 섹터별 비중 조절."),
)

with st.expander("계획 (요약)"):
    st.markdown("""
- **섹터 정의**: 은행채 · 카드/캐피탈 · 공기업(KEPCO·도공) · 보증채 · 회사채 등급별
- **시그널**: 섹터 스프레드 z-score + 매크로 사이클 (BBB-AA 스프레드, VIX 등) 조합
- **포지션**: 섹터 ETF/지수 또는 대표종목 매매
""")
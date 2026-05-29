"""듀레이션 전략 — KTB 절대금리 (ktb_3y/5y/10y/30y)."""
import streamlit as st

from theme import apply_theme, banner
from views.strategy_common import (render_signal_section,
                                       render_recent_vs_actual,
                                       render_target_analysis,
                                       render_backtest,
                                       render_models_table)

apply_theme()
banner("듀레이션 전략 · KTB 절대금리 시그널 (3Y/5Y/10Y/30Y)")

KTB_RATES = ["ktb_3y", "ktb_5y", "ktb_10y", "ktb_30y"]

tab1, tab2, tab3 = st.tabs(["현재 시그널", "타겟 분석", "백테스트"])

with tab1:
    render_signal_section(KTB_RATES, "KTB 절대금리 (4개 타겟)")
    st.divider()
    st.caption("""
**신호 해석** — `▲ LONG` = 듀레이션 확대(receive), `▼ SHORT` = 듀레이션 축소(pay).
강도(좌측 컬러바): <span style='color:#F58220;font-weight:700'>● STRONG</span>(P≥0.65) ·
<span style='color:#043B72;font-weight:700'>● MODERATE</span>(0.60≤P<0.65) ·
WEAK(0.55≤P<0.60) · NONE(P<0.55).
""", unsafe_allow_html=True)
    st.divider()
    render_recent_vs_actual(KTB_RATES, key_prefix="dur")

with tab2:
    target = render_target_analysis(KTB_RATES, key_prefix="dur")
    st.divider()
    render_models_table(target)

with tab3:
    render_backtest(KTB_RATES, key_prefix="dur", default_target="ktb_3y")
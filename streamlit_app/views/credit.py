"""회사채 스프레드 전략 — AA−·BBB− 3Y 스프레드 (vs KTB 3Y)."""
import streamlit as st

from theme import apply_theme, banner, CREDIT_ARROW, CREDIT_DESC
from views.strategy_common import (render_signal_section,
                                       render_recent_vs_actual,
                                       render_target_analysis,
                                       render_backtest,
                                       render_models_table)

apply_theme()
banner("회사채 스프레드 전략 · AA−/BBB− 3Y vs KTB 3Y")

CREDIT = ["spread_aa_3y", "spread_bbb_3y"]

tab1, tab2, tab3 = st.tabs(["현재 시그널", "타겟 분석", "백테스트"])

with tab1:
    render_signal_section(CREDIT, "회사채 스프레드 (2개 타겟)",
                            arrow_map=CREDIT_ARROW, desc_map=CREDIT_DESC)
    st.divider()
    st.caption("""
**신호 해석** — `▲ 와이드닝` = 스프레드 확대 예상 → **크레딧 축소(sell)** (방어).
`▼ 타이트닝` = 스프레드 축소 예상 → **크레딧 확대(add)** (위험감수).
스프레드 = 회사채 yield − KTB 3Y yield 기준.
""")
    st.divider()
    render_recent_vs_actual(CREDIT, key_prefix="cre",
                              arrow_map=CREDIT_ARROW, desc_map=CREDIT_DESC)

with tab2:
    target = render_target_analysis(CREDIT, key_prefix="cre")
    st.divider()
    render_models_table(target)

with tab3:
    render_backtest(CREDIT, key_prefix="cre", default_target="spread_aa_3y")
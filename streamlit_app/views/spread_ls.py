"""장단기스프레드 전략 — KTB 커브 스프레드."""
import streamlit as st

from theme import apply_theme, banner, SPREAD_ARROW, SPREAD_DESC
from views.strategy_common import (render_signal_section,
                                       render_recent_vs_actual,
                                       render_target_analysis,
                                       render_backtest,
                                       render_models_table)

apply_theme()
banner("장단기 스프레드 전략 · KTB Curve (10-3 / 30-5 / 30-10)")

KTB_SPREADS = ["spread_ktb_3_10", "spread_ktb_5_30", "spread_ktb_10_30"]

tab1, tab2, tab3 = st.tabs(["현재 시그널", "타겟 분석", "백테스트"])

with tab1:
    render_signal_section(KTB_SPREADS, "KTB 커브 스프레드 (3개 타겟)",
                            arrow_map=SPREAD_ARROW, desc_map=SPREAD_DESC)
    st.divider()
    st.caption("""
**신호 해석** — `▲ STEEPEN` = 스프레드 확대 베팅(스티프너), `▼ FLATTEN` = 스프레드 축소 베팅(플래트너).
스프레드 = 장기물 금리 − 단기물 금리 기준. DV01-neutral 2-leg 포지션.
""")
    st.warning("⚠ 백테스트상 KTB 커브 스프레드는 단독 트레이드 비추천 — 자세한 내용은 모델 설명 참고.")
    st.divider()
    render_recent_vs_actual(KTB_SPREADS, key_prefix="spr",
                              arrow_map=SPREAD_ARROW, desc_map=SPREAD_DESC)

with tab2:
    target = render_target_analysis(KTB_SPREADS, key_prefix="spr")
    st.divider()
    render_models_table(target)

with tab3:
    render_backtest(KTB_SPREADS, key_prefix="spr", default_target="spread_ktb_3_10")
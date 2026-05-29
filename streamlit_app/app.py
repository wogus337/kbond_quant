"""KBond ActiveQuant Strategy Dashboard — 메인 라우터."""
import streamlit as st

from theme import apply_theme
from auth import render_logout_sidebar

st.set_page_config(
    page_title="KBond ActiveQuant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


pages = {
    "전략": [
        st.Page("views/duration.py",       title="듀레이션",        default=True),
        st.Page("views/spread_ls.py",      title="장단기스프레드"),
        st.Page("views/credit.py",         title="회사채스프레드"),
        st.Page("views/yield_curve.py",    title="수익률곡선"),
        st.Page("views/relative_value.py", title="상대가치"),
        st.Page("views/pair_trading.py",   title="PairTrading"),
        st.Page("views/sector.py",         title="섹터"),
    ],
    "관리": [
        st.Page("views/data.py",  title="데이터"),
        st.Page("views/model.py", title="모델 설명"),
    ],
}

# 사이드바 상단 브랜드
with st.sidebar:
    st.markdown("""
<div style='padding:14px 8px 10px 8px; border-bottom:1px solid #cdd6e2; margin-bottom:6px;'>
  <div style='font-size:14px; font-weight:800; color:#043B72; line-height:1.25;'>
    KBond<br/>ActiveQuant
  </div>
  <div style='font-size:10px; color:#475569; margin-top:3px; letter-spacing:0.5px;'>
    STRATEGY DASHBOARD
  </div>
  <div style='height:3px; background:#F58220; width:38px; margin-top:6px; border-radius:2px;'></div>
</div>
""", unsafe_allow_html=True)

render_logout_sidebar()
pg = st.navigation(pages, position="sidebar")
pg.run()
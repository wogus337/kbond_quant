"""디자인 시스템 — 컬러 팔레트 + CSS 주입."""
import streamlit as st

# Palette
NAVY        = "#043B72"   # rgb(4, 59, 114)
NAVY_50     = "#67809e"
NAVY_20     = "#cdd6e2"
NAVY_08     = "#e8edf3"
ORANGE      = "#F58220"   # rgb(245, 130, 32)
ORANGE_50   = "#fac190"
ORANGE_20   = "#fde0c8"
ORANGE_08   = "#fef3e8"
WHITE       = "#ffffff"
GRAY_TEXT   = "#475569"
GRAY_LIGHT  = "#cbd5e1"
GRAY_BG     = "#f8fafc"

# Plotly 공통 색
PLOTLY_NAVY   = NAVY
PLOTLY_ORANGE = ORANGE


CSS = f"""
<style>
/* ─── 전체 폰트 축소 ─── */
html, body, [class*="css"]  {{
    font-size: 13px !important;
}}

/* 헤더 */
h1 {{ font-size: 22px !important; font-weight: 700 !important;
      color: {NAVY} !important; margin-bottom: 0.4rem !important; }}
h2 {{ font-size: 17px !important; font-weight: 700 !important;
      color: {NAVY} !important; margin-top: 1rem !important;
      margin-bottom: 0.4rem !important; padding-top: 0.2rem !important; }}
h3 {{ font-size: 14px !important; font-weight: 700 !important;
      color: {NAVY} !important; margin-top: 0.6rem !important;
      margin-bottom: 0.2rem !important; }}
h4 {{ font-size: 13px !important; font-weight: 600 !important;
      color: {GRAY_TEXT} !important; }}

p, li, span, label, div[data-testid="stMarkdownContainer"] p {{
    font-size: 13px !important;
}}
.stCaption, .stMarkdown small {{ font-size: 11px !important; color: {GRAY_TEXT}; }}

/* 메트릭 카드 */
[data-testid="stMetric"] {{
    background: {WHITE};
    border: 1px solid {NAVY_20};
    border-radius: 6px;
    padding: 10px 12px;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 11px !important; color: {GRAY_TEXT};
}}
[data-testid="stMetricValue"] {{
    font-size: 18px !important; font-weight: 700; color: {NAVY};
}}
[data-testid="stMetricDelta"] {{ font-size: 11px !important; }}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0px;
    border-bottom: 2px solid {NAVY_20};
}}
.stTabs [data-baseweb="tab"] {{
    height: 36px; padding: 0 18px;
    font-size: 13px; font-weight: 500;
    color: {GRAY_TEXT}; background: transparent;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {NAVY} !important; font-weight: 700 !important;
    border-bottom: 2px solid {ORANGE} !important;
    background: transparent !important;
}}

/* 라디오/셀렉트 */
.stRadio label, .stSelectbox label, .stSlider label {{
    font-size: 12px !important; font-weight: 600; color: {NAVY};
}}

/* 버튼 */
.stButton button {{
    font-size: 13px;
    border: 1px solid {NAVY};
    color: {NAVY};
    background: {WHITE};
    border-radius: 4px;
    padding: 4px 14px;
}}
.stButton button:hover {{
    background: {NAVY}; color: {WHITE};
}}
.stButton button[kind="primary"] {{
    background: {ORANGE}; color: {WHITE}; border: 1px solid {ORANGE};
}}
.stButton button[kind="primary"]:hover {{
    background: #d96c11; border: 1px solid #d96c11;
}}

/* 데이터프레임 */
[data-testid="stDataFrame"] {{ font-size: 12px; }}

/* 사이드바 */
[data-testid="stSidebar"] {{
    background: {GRAY_BG}; border-right: 1px solid {NAVY_20};
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {NAVY} !important; font-size: 13px !important;
}}
[data-testid="stSidebarNav"] {{ padding-top: 0; }}
[data-testid="stSidebarNav"] a {{
    color: {GRAY_TEXT} !important;
    font-size: 13px !important;
    border-radius: 4px;
    padding: 6px 10px !important;
}}
[data-testid="stSidebarNav"] a:hover {{
    background: {NAVY_08} !important;
    color: {NAVY} !important;
}}

/* 컨테이너 */
.block-container {{ padding-top: 1.2rem !important; max-width: 1400px; }}

/* expander */
[data-testid="stExpander"] summary {{ font-size: 12px; color: {NAVY}; }}

/* divider */
hr {{ margin: 0.4rem 0 !important; border-color: {NAVY_20} !important; }}

/* 상단 헤더 배너 */
.kbond-banner {{
    background: linear-gradient(90deg, {NAVY} 0%, #0a5da8 100%);
    color: {WHITE};
    padding: 14px 22px;
    border-radius: 6px;
    margin-bottom: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    border-left: 5px solid {ORANGE};
}}
.kbond-banner .title {{
    font-size: 18px; font-weight: 700; letter-spacing: 0.2px;
    margin: 0;
}}
.kbond-banner .subtitle {{
    font-size: 11px; opacity: 0.85; margin-top: 2px;
}}

/* 시그널 카드 */
.signal-card {{
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    background: {WHITE};
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border-left: 4px solid {GRAY_LIGHT};
}}
.signal-card .label {{ font-size: 11px; color: {GRAY_TEXT}; }}
.signal-card .arrow {{
    font-size: 18px; font-weight: 700;
    margin: 2px 0 4px 0;
}}
.signal-card .desc {{ font-size: 11px; color: {GRAY_TEXT}; line-height: 1.4; }}
.signal-card .stats {{
    font-size: 10.5px; color: {GRAY_TEXT};
    margin-top: 6px; padding-top: 6px;
    border-top: 1px dashed {NAVY_20};
    line-height: 1.5;
}}
.signal-card.strong   {{ border-left-color: {ORANGE}; }}
.signal-card.moderate {{ border-left-color: {NAVY}; }}
.signal-card.weak     {{ border-left-color: {NAVY_50}; }}
.signal-card.none     {{ border-left-color: {GRAY_LIGHT}; }}
.signal-card .arrow.long    {{ color: {ORANGE}; }}
.signal-card .arrow.short   {{ color: {NAVY}; }}
.signal-card .arrow.neutral {{ color: {GRAY_LIGHT}; }}

/* 배지 */
.badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.3px;
}}
.badge.strong   {{ background: {ORANGE}; color: {WHITE}; }}
.badge.moderate {{ background: {NAVY}; color: {WHITE}; }}
.badge.weak     {{ background: {NAVY_20}; color: {NAVY}; }}
.badge.none     {{ background: {GRAY_LIGHT}; color: {GRAY_TEXT}; }}

/* placeholder 카드 */
.placeholder {{
    background: {GRAY_BG};
    border: 1px dashed {NAVY_20};
    border-radius: 8px;
    padding: 30px;
    text-align: center;
    color: {GRAY_TEXT};
}}
.placeholder .icon {{ font-size: 32px; color: {ORANGE}; }}
.placeholder .title {{ font-size: 16px; font-weight: 700; color: {NAVY}; margin: 8px 0; }}
.placeholder .desc {{ font-size: 12px; }}
</style>
"""


def apply_theme():
    """모든 페이지 상단에서 호출."""
    st.markdown(CSS, unsafe_allow_html=True)


def banner(subtitle: str = ""):
    """상단 헤더 배너."""
    sub_html = f"<div class='subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
<div class='kbond-banner'>
  <div class='title'>KBond ActiveQuant Strategy Dashboard</div>
  {sub_html}
</div>
""", unsafe_allow_html=True)


DURATION_ARROW = {
    +1: ("▲ LONG",  "long"),
    -1: ("▼ SHORT", "short"),
     0: ("● 대기",  "neutral"),
}
DURATION_DESC = {
    +1: "듀레이션 확대 (receive)",
    -1: "듀레이션 축소 (pay)",
     0: "NEUTRAL — 무포지션",
}
SPREAD_ARROW = {
    -1: ("▲ STEEPEN", "long"),   # spread 확대 → 스티프너 (positive side에 색)
    +1: ("▼ FLATTEN", "short"),  # spread 축소 → 플래트너
     0: ("● 대기",    "neutral"),
}
SPREAD_DESC = {
    -1: "스프레드 확대 베팅 (steepener)",
    +1: "스프레드 축소 베팅 (flattener)",
     0: "NEUTRAL — 무포지션",
}
# 회사채 스프레드 (credit) — spread 상승 = 와이드닝 = 크레딧 위험 확대 → 크레딧 축소
CREDIT_ARROW = {
    -1: ("▲ 와이드닝", "long"),    # spread 상승 예상
    +1: ("▼ 타이트닝", "short"),   # spread 하락 예상
     0: ("● 대기",    "neutral"),
}
CREDIT_DESC = {
    -1: "크레딧 축소 (sell credit, defensive)",
    +1: "크레딧 확대 (add credit, risk-on)",
     0: "NEUTRAL — 무포지션",
}


def render_signal_card(label: str, current: str, regime: str,
                         signal: int, strength: str,
                         p_persist: float,
                         arrow_map: dict = None,
                         desc_map: dict = None):
    """시그널 카드 (전략별 arrow/desc 매핑 주입 가능)."""
    arrow_map = arrow_map or DURATION_ARROW
    desc_map  = desc_map  or DURATION_DESC
    arrow_text, arrow_class = arrow_map.get(signal, ("● 대기", "neutral"))
    desc = desc_map.get(signal, "")
    card_class = strength.lower() if strength in ("STRONG","MODERATE","WEAK") else "none"
    badge_class = card_class

    st.markdown(f"""
<div class='signal-card {card_class}'>
  <div class='label'>{label}</div>
  <div class='arrow {arrow_class}'>{arrow_text}</div>
  <div class='desc'>{desc}</div>
  <div class='stats'>
    현재값 <b>{current}</b> · regime <b>{regime}</b><br/>
    P(persist) <b>{p_persist:.3f}</b>
    <span class='badge {badge_class}' style='margin-left:6px;'>{strength}</span>
  </div>
</div>
""", unsafe_allow_html=True)


def _signal_inline(signal: int, strength: str, p_persist: float,
                    arrow_map: dict, desc_map: dict) -> str:
    """카드 내부 한 모델셋(top3/top1)의 시그널 HTML — 단일 라인(들여쓰기 X)."""
    arrow_text, arrow_class = arrow_map.get(signal, ("● 대기", "neutral"))
    desc = desc_map.get(signal, "")
    badge_class = strength.lower() if strength in ("STRONG", "MODERATE", "WEAK") else "none"
    return (
        f"<div class='arrow {arrow_class}' style='font-size:16px;'>{arrow_text}"
        f"<span class='badge {badge_class}' style='margin-left:6px;'>{strength}</span></div>"
        f"<div class='desc'>{desc} · P {p_persist:.3f}</div>"
    )


def render_signal_card_dual(label: str, current: str, regime: str,
                              top3: dict, top1: dict,
                              arrow_map: dict = None, desc_map: dict = None):
    """Top-3 앙상블(운용 기준) + Top-1 단일(참고)을 한 카드에 표시.
       top3/top1 = {'signal':int, 'strength':str, 'p_persist':float}
       마크다운 코드블록 오인 방지를 위해 전체를 들여쓰기 없는 단일 문자열로 구성."""
    arrow_map = arrow_map or DURATION_ARROW
    desc_map  = desc_map  or DURATION_DESC
    s3 = top3.get("strength", "NONE")
    card_class = s3.lower() if s3 in ("STRONG", "MODERATE", "WEAK") else "none"

    html = (
        f"<div class='signal-card {card_class}'>"
        f"<div class='label'>{label}</div>"
        f"<div style='font-size:10px;color:{ORANGE};font-weight:700;margin-top:4px;'>TOP-3 앙상블 · 운용기준</div>"
        f"{_signal_inline(int(top3['signal']), s3, float(top3['p_persist']), arrow_map, desc_map)}"
        f"<div style='border-top:1px dashed {NAVY_20};margin:6px 0;'></div>"
        f"<div style='font-size:10px;color:{NAVY_50};font-weight:700;'>TOP-1 단일 · 참고</div>"
        f"{_signal_inline(int(top1['signal']), top1.get('strength','NONE'), float(top1['p_persist']), arrow_map, desc_map)}"
        f"<div class='stats'>현재값 <b>{current}</b> · regime <b>{regime}</b></div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def placeholder(icon: str, title: str, desc: str):
    st.markdown(f"""
<div class='placeholder'>
  <div class='icon'>{icon}</div>
  <div class='title'>{title}</div>
  <div class='desc'>{desc}</div>
</div>
""", unsafe_allow_html=True)
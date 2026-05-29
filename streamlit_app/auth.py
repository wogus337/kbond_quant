"""관리 메뉴 로그인 게이트 — st.secrets 기반."""
import streamlit as st


def _admin_creds():
    """secrets.toml 의 [admin] 섹션에서 id/pw 로드. 없으면 (None, None)."""
    try:
        sec = st.secrets["admin"]
        return sec.get("id"), sec.get("pw")
    except Exception:
        return None, None


def require_login(page_label: str = "관리 메뉴"):
    """관리 영역 진입 게이트. 인증 안 됐으면 로그인 폼만 보이고 st.stop()."""
    if st.session_state.get("admin_authed"):
        return

    st.markdown(f"### 🔒 {page_label} — 로그인 필요")
    st.caption("이 영역은 관리자만 접근할 수 있습니다.")
    with st.form("admin_login", clear_on_submit=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            uid = st.text_input("ID", autocomplete="username")
        with c2:
            pw = st.text_input("Password", type="password",
                                 autocomplete="current-password")
        ok = st.form_submit_button("로그인", type="primary",
                                     use_container_width=True)

    if ok:
        admin_id, admin_pw = _admin_creds()
        if not admin_id:
            st.error("관리자 인증 정보 없음. 로컬: `.streamlit/secrets.toml` 작성. "
                       "Cloud: 앱 Settings → Secrets 에 `[admin] id, pw` 설정.")
            st.stop()
        if uid == admin_id and pw == admin_pw:
            st.session_state["admin_authed"] = True
            st.success("로그인 성공.")
            st.rerun()
        else:
            st.error("ID 또는 비밀번호가 올바르지 않습니다.")
    st.stop()


def render_logout_sidebar():
    """앱 전역에서 호출. authed 상태면 사이드바에 로그아웃 박스 표시."""
    if not st.session_state.get("admin_authed"):
        return
    uid, _ = _admin_creds()
    with st.sidebar:
        st.markdown("---")
        st.caption(f"🔓 관리자: **{uid or 'admin'}**")
        if st.button("로그아웃", use_container_width=True, key="_logout"):
            st.session_state["admin_authed"] = False
            st.rerun()
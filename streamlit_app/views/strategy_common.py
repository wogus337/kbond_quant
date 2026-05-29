"""듀레이션 · 장단기스프레드 view 공통 로직."""
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from lib import (TARGET_LABEL, TARGET_UNIT, PHASE3_DIR, TRIGGER_DEFAULT,
                  load_latest_signal_df,
                  load_target_series, get_topn_models,
                  ensemble_proba, get_chg_regime,
                  compute_backtest, summarize_backtest, yearly_pnl)
from theme import (NAVY, ORANGE, NAVY_20, ORANGE_20, GRAY_LIGHT, GRAY_TEXT,
                    render_signal_card_dual, banner)


def render_signal_section(targets: list, section_title: str,
                            arrow_map: dict = None,
                            desc_map: dict = None):
    """현재 시그널 카드들 — Top-3 앙상블(운용기준) + Top-1 단일(참고) 동시 표시."""
    df = load_latest_signal_df()
    if df.empty:
        st.info("phase3가 아직 실행되지 않았습니다(또는 구버전 산출물). "
                  "**데이터 → 3. phase3 재실행**을 먼저 실행하세요.")
        return

    df = df[df["target"].isin(targets)]
    if df.empty:
        st.warning("시그널 데이터가 없습니다.")
        return

    unit_of = lambda t: "%" if str(t).startswith("ktb") else "%pt"
    sig_date = pd.to_datetime(df["date"].iloc[0]).date() if "date" in df.columns else "?"
    latest_csv = PHASE3_DIR / "latest_signal.csv"
    mtime = datetime.fromtimestamp(latest_csv.stat().st_mtime)

    # Top-3 기준 시그널 발생 수
    top3_all = df[df["model_set"] == "top3"]
    n_fire = int((top3_all["signal"] != 0).sum())

    st.markdown(f"#### {section_title}")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("시그널 기준일", str(sig_date))
    c2.metric("최신 갱신", mtime.strftime("%m-%d %H:%M"))
    c3.metric("시그널 발생 (Top-3)", f"{n_fire} / {len(top3_all)}")

    ordered = [t for t in targets if t in df["target"].unique()]
    cols = st.columns(len(ordered))
    for col, t in zip(cols, ordered):
        sub = df[df["target"] == t]
        top3 = sub[sub["model_set"] == "top3"]
        top1 = sub[sub["model_set"] == "top1"]
        if top3.empty:
            continue
        r3 = top3.iloc[0]
        r1 = top1.iloc[0] if not top1.empty else r3
        with col:
            render_signal_card_dual(
                label=TARGET_LABEL.get(t, t),
                current=f"{r3['current_value']:.3f}{unit_of(t)}",
                regime=r3["current_regime"],
                top3={"signal": r3["signal"], "strength": r3["strength"],
                       "p_persist": r3["p_persist"]},
                top1={"signal": r1["signal"], "strength": r1["strength"],
                       "p_persist": r1["p_persist"]},
                arrow_map=arrow_map, desc_map=desc_map,
            )


def render_recent_vs_actual(targets: list, key_prefix: str,
                              arrow_map: dict = None, desc_map: dict = None,
                              months: int = 36):
    """최근 N개월 시그널(Top-3 앙상블, 2주 비중첩) vs 실제 타겟 등락 비교.
       색은 '예측 타겟 방향' 기준 — signal=-1(상승 예상)=빨강▲, +1(하락 예상)=파랑▼.
       (듀레이션 SHORT=금리 상승 예상=빨강, 스프레드 STEEPEN=상승 예상=빨강 — 공통)."""
    RED, BLUE, GRAY = "#d62728", "#1f5fbf", GRAY_LIGHT   # 상승=빨강 / 하락=파랑 / 중립=회색
    HOLD_BD = 10

    st.markdown("### 최근 시그널과 실제 등락 비교")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        target = st.selectbox("타겟", targets,
                                format_func=lambda t: f"{t}  —  {TARGET_LABEL[t]}",
                                key=f"{key_prefix}_recent_target")
    with c2:
        MONTH_OPTS = [6, 12, 24, 36, 60, 120]
        sel = st.selectbox("조회 개월수", MONTH_OPTS,
                             index=MONTH_OPTS.index(months),
                             format_func=lambda m: ("전체(2018~)" if m >= 120
                                                      else f"{m}개월"),
                             key=f"{key_prefix}_recent_months")
        months = sel
    with c3:
        trigger = st.slider("시그널 강도(trigger)", 0.55, 0.70, 0.60, 0.01,
                              key=f"{key_prefix}_recent_trigger",
                              help="P(persist) 컷오프. 높일수록 강한 신호만 표시")

    is_spread = str(target).startswith("spread")
    up_lbl   = "스프레드 상승" if is_spread else "금리 상승"
    down_lbl = "스프레드 하락" if is_spread else "금리 하락"

    # trigger → 강도 등급
    strg_name = ("STRONG" if trigger >= 0.65 else
                 "MODERATE+" if trigger >= 0.60 else "WEAK+")
    st.caption(f"최근 {'전체' if months>=120 else str(months)+'개월'} · "
                f"**Top-3 앙상블 운용기준** · 2주 비중첩 · "
                f"trigger P(persist)≥**{trigger:.2f}** ({strg_name} 이상). "
                f"세모 = 시그널 진입일, 진입 후 **10영업일 보유구간** 라인 색칠 — "
                f"**{up_lbl} 예상 ▲ 빨강 · {down_lbl} 예상 ▼ 파랑**, 중립/무포지션=회색. "
                "실제 등락이 확정된 발신일까지만 표시(라이브 시그널은 위 카드).")
    with st.expander("ℹ️ trigger·강도 설명"):
        st.markdown("""
- **trigger** = P(persist) 컷오프. `P(persist) ≥ trigger`(또는 반대편 `≤ 1−trigger`) **이고**
  regime이 UP/DOWN일 때만 시그널(세모)이 찍히고, 그 사이는 **중립(●)** = 무포지션.
- **강도 등급** (신뢰도 = max(P, 1−P) 기준):
  STRONG ≥ 0.65 · MODERATE 0.60–0.65 · WEAK 0.55–0.60 · NONE < 0.55
- **slider 효과**: 0.65 → STRONG만 / 0.60 → STRONG+MODERATE / 0.55 → WEAK까지 포함.
  높일수록 신호 수는 줄고 신뢰도는 올라감.
- **방향**: `signal=−1`(듀레이션 축소 / 스프레드 STEEPEN) = 타겟 **상승** 예상,
  `signal=+1`(듀레이션 확대 / FLATTEN) = 타겟 **하락** 예상.
""")

    m3 = get_topn_models(target, 3)
    bt = compute_backtest(target, m3, trigger=trigger, sample_every=10)
    if bt.empty:
        st.info("시그널 이력이 없습니다.")
        return

    cutoff = bt.index.max() - pd.DateOffset(months=months)
    recent = bt[bt.index >= cutoff].copy()
    # 예측 타겟 방향: signal=-1 → 상승(up), +1 → 하락(down), 0 → 중립
    recent["pdir"] = recent["signal"].map({-1: "up", 1: "down", 0: "flat"})
    unit = TARGET_UNIT[target]

    ts_bd = load_target_series(target).dropna()
    ts_recent = ts_bd[ts_bd.index >= cutoff]
    idx = ts_bd.index

    fig = go.Figure()
    # 1) 회색 베이스 라인
    fig.add_trace(go.Scatter(
        x=ts_recent.index, y=ts_recent.values, name="레벨 (무포지션)",
        line=dict(color=GRAY, width=1.3),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f} " + unit + "<extra></extra>"))

    # 2) 보유구간 색 세그먼트 (예측 방향 기준)
    seg = {"up": ([], []), "down": ([], [])}
    for d, r in recent.iterrows():
        pdir = r["pdir"]
        if pdir not in ("up", "down"):
            continue
        pos = idx.searchsorted(d)
        sub = ts_bd.iloc[pos: pos + HOLD_BD + 1]
        if sub.empty:
            continue
        xs, ys = seg[pdir]
        xs.extend(list(sub.index) + [None])
        ys.extend(list(sub.values) + [None])
    if seg["up"][0]:
        fig.add_trace(go.Scatter(
            x=seg["up"][0], y=seg["up"][1], mode="lines",
            name=f"{up_lbl} 예상 보유구간", line=dict(color=RED, width=3.2),
            connectgaps=False, hoverinfo="skip"))
    if seg["down"][0]:
        fig.add_trace(go.Scatter(
            x=seg["down"][0], y=seg["down"][1], mode="lines",
            name=f"{down_lbl} 예상 보유구간", line=dict(color=BLUE, width=3.2),
            connectgaps=False, hoverinfo="skip"))

    # 3) 진입 마커 (예측 방향 기준)
    style = {
        "up":   dict(color=RED,  symbol="triangle-up",   name=f"{up_lbl} 예상",   show=False),
        "down": dict(color=BLUE, symbol="triangle-down", name=f"{down_lbl} 예상", show=False),
        "flat": dict(color=GRAY, symbol="circle",        name="중립",            show=True),
    }
    lvl_at = ts_bd.reindex(recent.index).ffill()
    for pdir, stl in style.items():
        sub = recent[recent["pdir"] == pdir]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub.index, y=lvl_at.reindex(sub.index).values,
            mode="markers", name=stl["name"], showlegend=stl["show"],
            marker=dict(color=stl["color"], symbol=stl["symbol"], size=11,
                          line=dict(width=1, color="white")),
            customdata=sub[["p", "chg"]].values,
            hovertemplate=("%{x|%Y-%m-%d}<br>레벨 %{y:.3f} " + unit +
                            "<br>P(persist)=%{customdata[0]:.3f}"
                            "<br>2주후 Δ=%{customdata[1]:+.1f}bp<extra></extra>")))

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=380, hovermode="closest",
                       margin=dict(l=20, r=20, t=30, b=20),
                       plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                       font=dict(size=11), yaxis_title=unit,
                       legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1,
                                      font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    # 표 — 발신일별 예측 vs 실제
    dir_txt = {"up": f"▲ {up_lbl}", "down": f"▼ {down_lbl}", "flat": "중립"}
    rows = []
    for dt_, r in recent.iloc[::-1].iterrows():   # 최신순
        sig = int(r["signal"])
        hit = "" if sig == 0 else ("✓ 적중" if r["pnl_bp"] > 0 else "✗ 빗나감")
        rows.append({
            "발신일": dt_.strftime("%Y-%m-%d"),
            "regime": r["regime"],
            "P(persist)": round(float(r["p"]), 3),
            "예측방향": dir_txt[r["pdir"]],
            "2주후 Δ(bp)": round(float(r["chg"]), 1),
            "결과": hit,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    fired = recent[recent["signal"] != 0]
    if not fired.empty:
        hit_rate = (fired["pnl_bp"] > 0).mean()
        # 연속 동일방향 = 1 포지션으로 묶은 트레이드 수
        from lib import _consolidate_trades
        tr = _consolidate_trades(recent)
        n_trade = len(tr)
        st.caption(f"리밸런스 **{len(fired)}건** (2주마다 평가), "
                    f"실제 포지션 진입은 연속 동일방향을 묶어 **{n_trade}건**. "
                    f"리밸런스 적중률 **{hit_rate*100:.0f}%** (각 2주 예측의 부호 적중). "
                    "Δ는 발신일 대비 2주 후 변화(+면 상승). "
                    "결과 적중 = 예측방향과 실제 Δ 부호 일치.")
    else:
        st.caption(f"최근 {months}개월 발생 시그널 없음 (전부 중립).")


def render_target_analysis(targets: list, key_prefix: str):
    """타겟 선택 → 시계열 + regime + P(persist) 차트."""
    st.markdown("### 시계열 확인 및 국면지속확률 추이")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        target = st.selectbox(
            "타겟", targets,
            format_func=lambda t: f"{t}  —  {TARGET_LABEL[t]}",
            key=f"{key_prefix}_target")
    with c2:
        n_models = st.radio(
            "모델", [1, 3], index=1, horizontal=True,
            format_func=lambda n: f"Top-{n}" + (" 앙상블" if n>1 else ""),
            key=f"{key_prefix}_n")
    with c3:
        from_year = st.slider(
            "시작연도", 2014, 2026, 2018,
            key=f"{key_prefix}_year")

    st.caption("**기본값**: 모델 = Top-3 앙상블(운용기준), 시작연도 = 2018(OOS 시작). "
                "상단 = 타겟 레벨 + regime 음영(주황 UP·남색 DOWN), "
                "하단 = P(persist) 추이와 강도 임계선(0.55/0.60/0.65). "
                "여기 차트엔 매매 trigger가 없음 — 시그널 컷오프는 위 '최근 비교' 또는 '백테스트' 탭에서 조정.")

    ts = load_target_series(target)
    models = get_topn_models(target, n_models)
    proba = ensemble_proba(target, models)
    run = models.iloc[0]["run_id"] if not models.empty else None
    chg, regime = get_chg_regime(target, run) if run else (pd.Series(), pd.Series())

    start = pd.Timestamp(f"{from_year}-01-01")
    ts2 = ts[ts.index >= start].dropna() if not ts.empty else ts
    proba2 = proba[proba.index >= start].dropna() if not proba.empty else proba
    regime2 = regime[regime.index >= start].dropna() if not regime.empty else regime
    if not regime2.empty:
        regime2 = regime2[regime2.astype(str).str.len() > 0]

    unit = TARGET_UNIT[target]
    set_label = "Top-3 앙상블" if n_models > 1 else "Top-1 단일"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.65, 0.35], vertical_spacing=0.09,
                         subplot_titles=[f"{TARGET_LABEL[target]} 시계열 ({unit})",
                                          f"국면지속확률 P(persist) — {set_label}"])

    # yield
    fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, name=target,
                              line=dict(color=NAVY, width=1.5),
                              hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f} "+unit+"<extra></extra>"),
                   row=1, col=1)

    # regime 음영
    if not regime2.empty:
        reg_align = regime2.reindex(ts2.index).ffill()
        # UP=orange, DOWN=navy alpha
        bg = {"UP": "rgba(245,130,32,0.10)",
              "DOWN": "rgba(4,59,114,0.10)",
              "NEUTRAL": "rgba(148,163,184,0.05)"}
        cur = None; start_dt = None
        for dt_, rg in reg_align.items():
            if rg != cur:
                if cur in bg and start_dt is not None:
                    fig.add_vrect(x0=start_dt, x1=dt_, fillcolor=bg[cur],
                                    line_width=0, row=1, col=1)
                cur = rg; start_dt = dt_
        if cur in bg and start_dt is not None:
            fig.add_vrect(x0=start_dt, x1=reg_align.index[-1],
                            fillcolor=bg[cur], line_width=0, row=1, col=1)

    # P(persist)
    if not proba2.empty:
        fig.add_trace(go.Scatter(x=proba2.index, y=proba2.values,
                                  name="P(persist)",
                                  line=dict(color=ORANGE, width=1.2),
                                  fill="tozeroy", fillcolor="rgba(245,130,32,0.10)",
                                  hovertemplate="%{x|%Y-%m-%d}<br>P=%{y:.3f}<extra></extra>"),
                       row=2, col=1)
        for thr, lbl, dash in [(0.65,"0.65 STRONG","dash"),
                                 (0.60,"0.60 MODERATE","dot"),
                                 (0.55,"0.55 WEAK","dot")]:
            fig.add_hline(y=thr, line_dash=dash, line_color=NAVY,
                           opacity=0.4, row=2, col=1,
                           annotation_text=lbl, annotation_position="right",
                           annotation_font_size=9)
        fig.add_hline(y=0.5, line_color=GRAY_LIGHT, row=2, col=1)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text=unit, row=1, col=1, title_font_size=11)
    fig.update_yaxes(title_text="P(persist)", row=2, col=1, range=[0,1],
                       title_font_size=11)
    fig.update_layout(height=470, hovermode="x unified",
                       margin=dict(l=20, r=20, t=46, b=20),
                       showlegend=False, plot_bgcolor="#ffffff",
                       paper_bgcolor="#ffffff",
                       font=dict(size=11))
    # subplot 제목(annotation) 폰트/색
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=NAVY)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("음영: regime 국면 (주황=UP 상승국면 · 남색=DOWN 하락국면). "
                "하단 P(persist)는 현재 국면이 2주 후에도 유지될 확률 — 점선은 STRONG/MODERATE/WEAK 임계.")

    return target


def render_backtest(targets: list, key_prefix: str, default_target: str = None):
    """Top-1 vs Top-3 백테스트 비교. 2주(10영업일) 비중첩 고정."""
    every = 10  # forward_weeks=2 = 10영업일. 비중첩으로 포지션 1개씩만 보유.
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        idx = targets.index(default_target) if default_target in targets else 0
        target = st.selectbox(
            "타겟", targets, index=idx,
            format_func=lambda t: f"{t}  —  {TARGET_LABEL[t]}",
            key=f"{key_prefix}_bt_target")
    with c2:
        trigger = st.slider("trigger", 0.50, 0.75, TRIGGER_DEFAULT, 0.01,
                              key=f"{key_prefix}_bt_trigger",
                              help="P(persist) 컷오프. 기본 0.60(MODERATE+)")
    with c3:
        compare = st.checkbox("Top-1 동시표시", value=True,
                                 key=f"{key_prefix}_bt_compare")

    st.caption(f"**기본값**: trigger P(persist)≥**{TRIGGER_DEFAULT:.2f}**(MODERATE 이상), "
                "샘플링 **2주(10영업일) 비중첩**, 비교 = Top-1 vs Top-3 앙상블.  \n"
                "• trigger ↑(예 0.65)=STRONG만 → 신호 수 감소·신뢰도 ↑ / ↓(0.55)=WEAK까지 포함.  \n"
                "• 2주 비중첩: 예측 horizon이 2주이므로 10영업일마다 1회만 진입/청산(포지션 1개). "
                "일별 진입은 2주 포지션이 10겹 중첩돼 비현실적이라 제외.")
    with st.expander("ℹ️ 백테스트 기준 · 가정"):
        st.markdown("""
- **신호 정의**: `P(persist) ≥ trigger`(또는 ≤ 1−trigger) & regime UP/DOWN → 진입, 그 외 무포지션.
- **PnL** = `−signal × Δ(2주 후 bp)`. 방향 맞으면 |Δ|만큼 +, 틀리면 −. **거래비용 차감 전**.
- **기간**: 2018-01-01 OOS 시작 ~ 데이터 확정 시점. 1단위 포지션당 bp 누적.
- **Top-1 vs Top-3**: Top-1 = composite 최상위 단일 모델, Top-3 = 상위 3개 proba 평균(운용기준).
- **거래비용 미반영** — 실제는 KTB 선물 round-trip ~0.1~0.3bp + 슬리피지 차감 필요.
""")

    m1 = get_topn_models(target, 1)
    m3 = get_topn_models(target, 3)
    df1 = compute_backtest(target, m1, trigger=trigger, sample_every=every)
    df3 = compute_backtest(target, m3, trigger=trigger, sample_every=every)
    s1 = summarize_backtest(df1)
    s3 = summarize_backtest(df3)

    # 요약 카드 — 트레이드 단위(연속 동일방향=1포지션) 기준
    c_left, c_right = st.columns(2)
    for col, (label, s) in zip([c_left, c_right],
                                   [("Top-1 단일", s1), ("Top-3 앙상블", s3)]):
        with col:
            st.markdown(f"##### {label}")
            if not s or s.get("n_trades", 0) == 0:
                st.caption("신호 없음"); continue
            cc = st.columns(4)
            cc[0].metric("Hit (트레이드)", f"{s['hit_rate_trade']*100:.1f}%",
                          help="포지션 단위 적중률 (포지션의 누적 PnL>0)")
            cc[1].metric("누적", f"{s['cum_pnl_bp']:+.0f}bp")
            cc[2].metric("연환산", f"{s['ann_pnl_bp']:+.0f}bp/y")
            cc[3].metric("Calmar", f"{s['calmar']:.2f}")
            cc2 = st.columns(4)
            cc2[0].metric("트레이드 수", f"{s['n_trades']}",
                           help="연속 동일방향 시그널을 1포지션으로 묶은 실제 진입 횟수")
            cc2[1].metric("리밸런스", f"{s['n_signals']}",
                           help="2주마다 평가한 시그널 발생 건수 (포지션 holding 포함)")
            cc2[2].metric("평균 보유", f"{s['avg_hold_weeks']:.1f}주",
                           help="포지션당 평균 보유 기간")
            cc2[3].metric("Hit (리밸런스)", f"{s['hit_rate_rebal']*100:.1f}%",
                           help="2주 예측 적중률 (포지션 holding 동안의 각 2주 평가)")

    # PnL + Drawdown
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.6, 0.4], vertical_spacing=0.06,
                         subplot_titles=["Cumulative PnL (bp)", "Drawdown (bp)"])
    if compare and not df1.empty:
        fig.add_trace(go.Scatter(x=df1.index, y=df1["cum_pnl"], name="Top-1",
                                   line=dict(color=NAVY, width=1.5, dash="dot")),
                       row=1, col=1)
        fig.add_trace(go.Scatter(x=df1.index, y=df1["dd"], showlegend=False,
                                   line=dict(color=NAVY, width=1, dash="dot"),
                                   fill="tozeroy", fillcolor="rgba(4,59,114,0.05)"),
                       row=2, col=1)
    if not df3.empty:
        fig.add_trace(go.Scatter(x=df3.index, y=df3["cum_pnl"], name="Top-3 앙상블",
                                   line=dict(color=ORANGE, width=1.8)),
                       row=1, col=1)
        fig.add_trace(go.Scatter(x=df3.index, y=df3["dd"], showlegend=False,
                                   line=dict(color=ORANGE, width=1.2),
                                   fill="tozeroy", fillcolor="rgba(245,130,32,0.10)"),
                       row=2, col=1)
    fig.add_hline(y=0, line_color=GRAY_LIGHT, row=1, col=1)
    fig.add_hline(y=0, line_color=GRAY_LIGHT, row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="bp", row=1, col=1, title_font_size=11)
    fig.update_yaxes(title_text="bp", row=2, col=1, title_font_size=11)
    fig.update_layout(height=440, hovermode="x unified",
                       margin=dict(l=20, r=20, t=36, b=20),
                       plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                       font=dict(size=11),
                       legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1,
                                      font=dict(size=10)))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=11, color=NAVY)
    st.plotly_chart(fig, use_container_width=True)

    # 연도별 막대
    y3 = yearly_pnl(df3)
    y1 = yearly_pnl(df1) if compare else pd.DataFrame()
    fig_y = go.Figure()
    if not y1.empty:
        fig_y.add_trace(go.Bar(x=y1["year"], y=y1["pnl_bp"], name="Top-1",
                                  marker_color=NAVY_20))
    if not y3.empty:
        fig_y.add_trace(go.Bar(x=y3["year"], y=y3["pnl_bp"], name="Top-3 앙상블",
                                  marker_color=ORANGE))
    fig_y.add_hline(y=0, line_color=GRAY_LIGHT)
    fig_y.update_layout(height=250, barmode="group",
                          margin=dict(l=20, r=20, t=10, b=20),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                          xaxis_title="연도", yaxis_title="PnL (bp)",
                          font=dict(size=11),
                          legend=dict(orientation="h", yanchor="bottom",
                                          y=1.02, xanchor="right", x=1,
                                          font=dict(size=10)))
    st.plotly_chart(fig_y, use_container_width=True)

    # 7개 타겟 비교표 — 트레이드 단위(연속 동일방향=1포지션)
    st.markdown("##### 타겟별 Top-3 앙상블 백테스트 비교")
    rows = []
    for t in targets:
        mm = get_topn_models(t, 3)
        bt = compute_backtest(t, mm, trigger=trigger, sample_every=every)
        s = summarize_backtest(bt)
        if not s or s.get("n_trades", 0) == 0:
            continue
        rows.append({
            "target": t,
            "trades": s["n_trades"],
            "리밸런스": s["n_rebalances"],
            "avg 보유(주)": round(s["avg_hold_weeks"], 1),
            "hit% (trade)": round(s["hit_rate_trade"]*100, 1),
            "cum (bp)": round(s["cum_pnl_bp"], 0),
            "ann (bp/y)": round(s["ann_pnl_bp"], 1),
            "max DD (bp)": round(s["max_dd_bp"], 0),
            "Calmar": round(s["calmar"], 2),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Calmar", ascending=False),
                       hide_index=True, use_container_width=True)
        st.caption(
            "**컬럼 설명** (거래비용 차감 전, 1단위 포지션당 bp 기준)  \n"
            "• **trades**: 실제 진입 횟수. **연속 동일방향 시그널 = 1포지션**으로 묶음 "
            "(중간에 중립/0 끼거나 방향 뒤집히면 별개 트레이드).  \n"
            "• **리밸런스**: 2주마다 평가한 시그널 발생 건수 (포지션 holding 평가 포함).  \n"
            "• **avg 보유(주)**: 포지션당 평균 보유 기간 (= 1포지션의 연속 리밸런스 × 2주).  \n"
            "• **hit% (trade)**: 포지션 단위 적중률 — 포지션의 누적 PnL>0 비율.  \n"
            "• **cum (bp)**: 누적 손익. 모든 리밸런스 PnL 합 (트레이드 합산과 동일).  \n"
            "• **ann (bp/y)**: 연환산 손익 (cum ÷ 기간 × 365).  \n"
            "• **max DD (bp)**: 최대 낙폭 (누적 손익 고점 대비).  \n"
            "• **Calmar**: 연환산 ÷ |max DD|. 높을수록 위험대비 수익 우수 "
            "(≥0.5 양호, <0 손실).")


def render_models_table(target: str):
    """선택된 타겟의 Top-3 모델 정보."""
    st.markdown("##### Top-3 모델 (composite 내림차순)")
    m3 = get_topn_models(target, 3)
    if m3.empty:
        st.caption("metrics_top.csv 가 없습니다.")
        return
    show = m3[["rank2","run_id","model","accuracy","auc",
                 "hit@60","cov@60","composite"]].copy()
    show.columns = ["rank","run_id","model","accuracy","AUC",
                     "hit@60","cov@60","composite"]
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(
        "**컬럼 설명** (phase1 walk-forward OOS 2018~)  \n"
        "• **rank**: composite 기준 순위 (Top-3 = 운용 앙상블 구성 모델)  \n"
        "• **run_id**: 설정 `L{lookback}W_F{forward}W_R{retrain}d_T{thr}bp_S{strength}`  \n"
        "• **model**: 분류기 (lgbm/xgb/rf/et/gbm/logreg)  \n"
        "• **accuracy**: 전체 정확도 / **AUC**: 판별력(0.5=무작위, 클수록 우수)  \n"
        "• **hit@60**: P(persist)≥0.6 고신뢰 표본의 정확도 / **cov@60**: 그 표본 비율  \n"
        "• **composite**: `AUC × (1 + hit@60 × cov@60)` — 모델 선정 종합점수")
"""모델 · 트레이딩 시스템 설명 (인증 불필요, 일반 공개)."""
import streamlit as st

from lib import SIGNAL_MAPPING_MD
from theme import apply_theme, banner

apply_theme()
banner("모델 · 트레이딩 시스템 설명")

st.markdown("### 1. 시스템 개요")
st.markdown("""
**예측 대상**: 향후 2주(10영업일) 동안 현재의 금리/스프레드 regime이 **유지될 확률** (`P(persist)`)

**HMM regime 라벨링** — 과거 N주(`lookback_weeks`) yield 변화량에 2-state Gaussian HMM을 적합하여
각 시점을 **UP (상승국면)** / **DOWN (하락국면)** 으로 분류. `regime_strength = |Δlookback|/std` 가
일정 이상인 일자만 "유효 regime"으로 마스킹.

**학습 라벨** — regime이 H일 후에도 동일하면 `persist=1`, 반전이면 `0`.

**피처** (`features.py`)
- 금리 커브 레벨/슬로프/커버처 (10y−2y, 30y−10y 등)
- 모멘텀/Z-score (d5/d20/d60, z20/z60/z252)
- 변동성 (20일 daily-change std)
- 크레딧 스프레드 (AA−KTB3y, BBB−KTB3y, KEPCO−KTB5y)
- 미국 금리 (us_2y/10y, BEI, real_10y)
- FX·주식·글로벌 (USD/KRW, KOSPI, VIX, DXY, WTI)
- 매크로 (CPI/PPI/IP YoY 등 — 월간 매크로는 1개월 lag)
""")

st.markdown("### 2. 학습 · 검증")
st.markdown("""
- **Walk-forward + embargo (H+5영업일)** → train/test 사이 누설 차단
- **OOS 시작 2018-01-01**, 최소 학습 750일, 재학습 주기 63영업일 (분기)
- **HPO**: Optuna `hpo_trials=20`
- **외부 그리드**: lookback {4,8,12,20} × forward {2,4} × thr_bp {0,10}
- **모델 카탈로그**: LightGBM · XGBoost · RandomForest · ExtraTrees · GradientBoosting · LogReg
- **선정 기준**: `Composite = AUC × (1 + Hit@60 × Cov@60)` 내림차순
""")

st.markdown("### 3. 시그널 매핑 (phase3)")
st.markdown(SIGNAL_MAPPING_MD)

st.markdown("### 4. 모델 선정 방식")
st.markdown("""
타겟별 Top-3 모델은 **운용 앙상블**로 사용됨. 각 타겟의 구체적 모델 구성 + 메트릭은
**듀레이션 / 장단기스프레드 / 회사채스프레드** 페이지의 **'타겟 분석'** 탭 하단에서 확인.

`run_id` 표기: `L{lookback_weeks}W_F{forward_weeks}W_R{retrain_days}d_T{target_thr_bp}bp_S{regime_strength_min}`
""")

st.markdown("### 5. 트레이딩 운용 가이드")
st.markdown("""
**포지션 사이즈**

| 강도 | P(persist) | 권고 사이즈 |
|---|---|---|
| STRONG | ≥ 0.65 | full tilt (±0.3년 듀레이션) |
| MODERATE | 0.60–0.65 | half tilt (±0.15년) |
| WEAK | 0.55–0.60 | quarter tilt (±0.075년) |
| NONE | < 0.55 | 무포지션 |

**운용 시 유의사항**
- 연도별 안정성에 차이 (2024년 long-end 약점)
- 거래비용 차감 전. KTB 선물 round-trip ~0.1–0.3bp + 슬리피지 고려
- **KTB 커브 스프레드(spread_ktb_*)는 백테스트상 단독 트레이드 비추천**
- 매 분기 phase1 재학습 권장 — regime drift 빠름
- STRONG 시그널이라도 BOK MPC · FOMC · CPI 일정 확인 후 진입
""")
# KBOND ActiveQuant — S01_DUR (듀레이션·크레딧 방향성 예측)

> 머신러닝 기반으로 **국고채 금리·장단기 스프레드·회사채 스프레드의 미래 방향**을 예측하여,
> 액티브 채권 운용의 **듀레이션(Duration) 및 크레딧(Credit) 포지션 의사결정**을 보조하는 파이프라인입니다.

---

## 1. 프로젝트 목적

한국·미국의 금리/스프레드 시계열에 대해 다음 두 가지 형태의 "방향성 신호"를 생성하는 것이 목표입니다.

| 신호 종류 | 설명 | 산출 스크립트 |
|---|---|---|
| **Regime persist/reverse 신호** | HMM으로 라벨링한 현재 regime이 유지/반전될 확률 (forward 1~4주) | `phase1_train.py` → `phase3_predict.py` |
| **순수 방향(부호) 신호** | H일 후 sign(value[t+H] − value[t]) 자체를 직접 분류 | `direction_strategy.py` |
| **결합 신호** | 위 두 채널을 결합한 STRONG/MODERATE/WEAK/NEUTRAL 종합신호 | `make_combined_signal.py` |

최종 권고는 KTB·스프레드 별로 다음과 같이 매핑됩니다.
- **KTB 금리**: P(UP)↑ → **듀레이션 축소** / P(UP)↓ → **듀레이션 확대**
- **회사채 스프레드**: P(UP)↑ → **크레딧 축소** / P(UP)↓ → **크레딧 확대**

---

## 2. 예측 타겟

`ALL_TARGETS` (phase1_train.py·direction_strategy.py 공통):

```
KTB 절대금리      : ktb_3y, ktb_5y, ktb_10y, ktb_30y
KTB 장단기 스프레드 : spread_ktb_3_10, spread_ktb_5_30, spread_ktb_10_30
회사채 스프레드     : spread_aa_3y, spread_bbb_3y, spread_kepco_5y
미국             : us_10y_xl, us_ig_oas_xl, us_spr_2_10, us_spr_5_30, us_spr_10_30
```

스프레드 정의는 `features.build_target_series` 참조 (`spread_aa_3y = corp_aa_3y − ktb_3y` 등).

---

## 3. 데이터 파이프라인

### 3-1. 수집 (`data_fetcher.py` + `run_fetch_data.py`)

| 소스 | 그룹 (parquet) | 설명 |
|---|---|---|
| **BOK ECOS** | `rates_ktb`, `rates_credit` | 국고채 1Y~50Y, 기준금리, CD/CP, 회사채 AA-/BBB-, 산금채, 한전채 |
| BOK ECOS | `macro_growth`, `macro_price`, `macro_money`, `macro_sentiment` | GDP/IP, CPI/PPI, M1/M2/외환보유액, BSI/CSI |
| BOK ECOS | `fx`, `equity` | USD/KRW·CNY·JPY·EUR, KOSPI/KOSDAQ + 외인 순매수 |
| **FRED** | `us_rates`, `us_macro`, `us_global`, `us_credit` | DGS2/5/10/30, T10Y2Y, BEI, CPI/UNRATE/PAYEMS, WTI, DXY, VIX |
| **Yahoo Finance** | `global_yahoo` | ^TNX, ^VIX, DX-Y.NYB, ^GSPC, ^KS200, CL=F, GC=F |
| **Bloomberg/Excel** | `us_xl` | 미 10Y, US IG OAS 등 (`us_10y_xl`, `us_ig_oas_xl`) |

캐시: `data_cache/*.parquet` (+ `_logs.csv`, `_summary.csv`)

### 3-2. 피처 엔지니어링 (`features.py`)

`build_all_features()` 산출 wide DataFrame:

1. **금리 커브** — 레벨, 슬로프(10y−2y / 30y−10y / 5y−2y), 커버처(2-5-10)
2. **모멘텀/정규화** — d5/d20/d60 (bp), z20/z60/z252 (rolling z-score)
3. **변동성** — 20일 daily-change std
4. **크레딧 스프레드 피처** — corp_aa−ktb_3y, corp_bbb−ktb_3y, kepco−ktb_5y
5. **미국 금리** — us_2y/10y/10y2y, BEI, real_10y (레벨/변화/z)
6. **FX·주식·글로벌** — USD/KRW 로그수익률, KOSPI 로그수익률, 외인 순매수 z, VIX, DXY 변화, WTI 변화
7. **매크로** — CPI/PPI/IP YoY, 외환보유액 YoY, CSI z, US CPI YoY, US 실업률 level
   - **월간 매크로는 1개월 lag 적용 + ffill** (lookahead 차단)
8. NaN 처리: 피처별 forward-fill, warm-up 252일 절단

타겟 빌더:
- `build_target_series(cache, name)` — 절대 시리즈
- `make_persist_target` — regime이 H일 후에도 유지되면 1
- `make_reverse_target` — regime이 반전되면 1
- `make_direction_target_from_series(ts, H)` — sign(ts[t+H] − ts[t])

---

## 4. 모델 카탈로그

phase1·direction 공통으로 6개 분류기 + 평균 앙상블:

| 모델 | 라이브러리 | 특징 |
|---|---|---|
| `lgbm`   | LightGBM        | NaN 허용, feature_importance 추출 |
| `xgb`    | XGBoost (hist)  | NaN 허용 |
| `rf`     | sklearn RandomForest      | NaN 비허용 → median fill |
| `et`     | sklearn ExtraTrees        | NaN 비허용 |
| `gbm`    | sklearn GradientBoosting  | NaN 비허용 |
| `logreg` | StandardScaler + L2 LogReg | NaN 비허용, 분산 0 컬럼 제거 |

phase1은 추가로 **Optuna HPO (inner)** 를 모델별로 돌립니다 (`hpo_trials=20`, `hpo_splits=4`).

---

## 5. 학습·검증 (Walk-Forward)

공통 원칙:
- **시계열 분할 + embargo** — train↔test 사이에 `H + 5` 영업일 간격을 둬서 라벨 누설 차단
- **min_train_days = 750** (≈ 3년) 부터 OOS 시작
- **retrain_every = 63** (분기) 또는 252 (연)
- `start_oos_year = 2018` 부터 평가

### phase1 — 외부 그리드 × 내부 HPO

```
lookback_weeks   ∈ {4, 8, 12, 20}      # HMM regime 라벨링 기간
forward_weeks    ∈ {2, 4}              # persist/reverse 예측 horizon
retrain_days     ∈ {63}                # 재학습 주기
target_thr_bp    ∈ {0, 10}             # |Δforward| < thr 인 일자 마스킹
regime_strength  ∈ {0.0}               # |Δlookback|/std 강도 컷오프
→ outer = 4 × 2 × 1 × 2 × 1 = 16 configs / target
```

각 config 내부에서:
1. HMM 2-state (UP/DOWN) regime 라벨링 (train-only fit, cutoff=2018-01-01)
2. regime_strength 마스킹 → persist 타겟 생성 (2-state라 reverse = 1 − persist)
3. forward |Δ| < `target_thr_bp` 인 일자 추가 마스킹
4. LGBM 기반 top-K (=80) 피처 선택
5. 모델별 Optuna HPO → walk_forward_periodic
6. weighted ensemble (가중치 = max(AUC−0.5, 0))

### direction_strategy — 단순 부호 분류

- horizons: `[10, 20]` 영업일 (≈ 2주, 4주)
- HMM 없이 `sign(ts[t+H] − ts[t])` 라벨을 직접 분류
- 6개 모델 + 평균 앙상블
- HPO 없음 (고정 하이퍼파라미터)

### 메트릭 (양 파이프라인 공통)

- Accuracy / Precision / Recall / F1
- AUC-ROC
- **Hit@K** : 신뢰도 max(P, 1−P) ≥ K (K=0.6, 0.7) 표본의 정확도
- **Coverage@K** : 신뢰도 ≥ K 표본의 비율
- **CompositeScore = AUC × (1 + Hit@60 × Cov@60)** — 최종 모델 선정 기준

---

## 6. 파이프라인 전체 흐름

```
┌─────────────────┐
│ run_fetch_data  │  BOK + FRED + Yahoo + Excel → data_cache/*.parquet
└───────┬─────────┘
        │
        ▼
┌──────────────────────────┐        ┌──────────────────────────────┐
│ phase1_train.py          │        │ direction_strategy.py        │
│  HMM regime → persist    │        │  sign(Δ_H) 직접 분류          │
│  config grid × HPO × WF  │        │  Walk-forward, H=10/20       │
│  results/phase1/         │        │  results/direction/          │
└───────┬──────────────────┘        └──────────────┬───────────────┘
        │                                          │
        ▼                                          │
┌──────────────────────────┐                       │
│ phase2_review.py         │                       │
│  metrics_top → 모델 선택   │                      │
│  model_selection.csv     │                       │
└───────┬──────────────────┘                       │
        │                                          │
        ▼                                          │
┌──────────────────────────┐                       │
│ phase3_predict.py        │                       │
│  persist+reverse ensemble│                       │
│  + 현재 regime → 신호+강도 │                       │
│  results/phase3/         │                       │
└───────┬──────────────────┘                       │
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
                ┌──────────────────────────────┐
                │ make_combined_signal.py      │
                │  phase1(persist) + direction │
                │  → STRONG/MODERATE/WEAK      │
                │  results/combined/           │
                └──────────────────────────────┘
```

---

## 7. 산출물

### `results/phase1/`
| 파일 | 내용 |
|---|---|
| `outer_configs.csv` | 시도한 모든 outer config |
| `metrics_by_run.csv` | (target × config × direction × model) 전 메트릭 |
| `metrics_top.csv` | (target × direction) CompositeScore 내림차순 |
| `predictions.parquet` | 모든 OOS 확률 (wide, 컬럼명 `{target}__{run_id}__{direction}__proba_{model}`) |
| `hpo_params.json` | Optuna best params |
| `feature_importance.csv` | LGBM 평균 importance top-30 |
| `selected_features.csv`, `fold_log.csv`, `yearly_breakdown.csv` | 진단용 |
| `model_selection_template.csv` | phase2 입력 템플릿 (Top-1 자동 추천) |
| `regime_history.parquet` | target × lookback별 HMM regime/strength 시계열 |

### `results/phase2/`
- `model_selection.csv` — 사용자가 `selected` 컬럼을 True/False로 편집

### `results/phase3/`
- `metrics_selected.csv`, `signal_table.parquet` (일자별 P(persist), P(reverse), regime, signal, strength)
- `high_conf_backtest.csv`, `latest_signal.txt`, `cum_pnl.png`

### `results/direction/`
- `metrics_by_model.csv`, `metrics_summary.csv` (앙상블만), `leaderboard.csv`
- `predictions.parquet`, `feature_importance.csv`
- `latest_signal.txt` — 현 시점 P(UP) + 듀레이션/크레딧 권고
- `confusion_matrix.png`, `cum_accuracy.png`

### `results/combined/`
- `signal_2W_by_target.csv` (H=10영업일, 10일 샘플링), `signal_4W_by_target.csv` (H=20)
- `signal_latest.csv`, `hit_summary.csv`

---

## 8. 실행 순서 (전체 리프레시)

```powershell
# 1) 데이터 캐시 갱신
python run_fetch_data.py                       # 증분 (기본)
python run_fetch_data.py --refresh             # 전체 재수집

# 2-A) Regime persist 파이프라인 (phase1~3)
python phase1_train.py                         # ★ 현재 실행 중
python phase2_review.py                        # 모델 선택 CSV 생성
# (필요 시 results/phase2/model_selection.csv 의 selected 컬럼 편집)
python phase3_predict.py                       # 신호 + 강도

# 2-B) Direction 부호 파이프라인 (별도 채널)
python direction_strategy.py

# 3) 결합 신호
python make_combined_signal.py
```

### 단일 타겟·빠른 실행 예시

```powershell
python phase1_train.py --targets ktb_10y --models lgbm xgb --hpo-trials 5
python direction_strategy.py --targets ktb_10y spread_aa_3y --horizons 10
python phase3_predict.py --threshold 0.65
```

---

## 9. 디렉터리 구조

```
S01_DUR/
├── core.py                      # 공유: 모델 팩토리, HPO, WF, 메트릭, 앙상블, HMM
├── features.py                  # 피처 빌더, 타겟 빌더, 캐시 로더
├── data_fetcher.py              # BOK/FRED/Yahoo 시리즈 정의 및 fetch 함수
├── run_fetch_data.py            # 데이터 수집 실행
├── phase1_train.py              # ★ HMM regime + persist 학습 (현재 실행 중)
├── phase2_review.py             # 모델 선택 CSV
├── phase3_predict.py            # persist+reverse → 신호+강도
├── direction_strategy.py        # 순수 방향(부호) 학습/평가/신호
├── make_combined_signal.py      # 두 채널 결합 → 종합신호
├── requirements.txt
├── data_cache/                  # parquet 캐시 (BOK/FRED/Yahoo/Excel)
└── results/
    ├── phase1/                  # 학습 산출물
    ├── phase2/                  # 모델 선택
    ├── phase3/                  # 최종 신호
    ├── direction/               # direction_strategy 산출물
    └── combined/                # 결합 신호
```

---

## 10. 핵심 설계 결정

- **시계열 누설 차단**: walk-forward + embargo(H+5), 매크로는 1개월 lag + ffill, HMM은 cutoff 이전 데이터만으로 fit
- **HMM regime은 라벨에만 사용** (자기참조 차단을 위해 피처로는 포함 X)
- **자동 추천 + 사용자 검토 단계 분리**: phase2에서 사용자가 모델 선택을 편집 가능
- **두 개의 독립 채널**:
  - phase1 = "regime 지속성" 관점 (체계적/거시 흐름)
  - direction = "순간 부호" 관점 (단기 기술적)
  - 두 신호를 `make_combined_signal.py`에서 합쳐 신뢰도 보강
- **CompositeScore 기반 모델 선정** — AUC 단독이 아닌 "고신뢰 표본에서의 정확도 × 표본 비율" 가중

---

## 11. 외부 의존성

- Python 패키지: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `optuna`, `hmmlearn`, `requests`, `matplotlib`, `pyarrow`
- API 키: BOK ECOS, FRED (현재 `data_fetcher.py`에 하드코딩됨 — 운영 환경에서는 환경변수 전환 권장)
- Bloomberg/엑셀 입력 (`us_xl.parquet`): 별도 수기 갱신
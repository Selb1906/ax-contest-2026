# 전기요금 과다발생 사전 예측모델 개발
## [Plan A — 1차+2차 분석 통합]

---

## 1. 과제 정의 및 가설 수립

### 1.1 과제 개요
고객의 과거 전력 사용 데이터를 기반으로, 검침일 기준 +10일 및 +20일 시점에 해당 월 전력사용량을 예측하고, 과다발생 가능성을 사전에 알리는 모델을 개발한다.

**알림 조건** (1건 이상 충족 시):
- 전월 대비 +30% 초과
- 전년 동월 대비 +30% 초과
- 직전 3개월 평균 대비 +50% 초과

### 1.2 핵심 가설
> "검침일 이후 부분 실측 사용량(partial_kwh)은 해당 월 전체 사용량과 r > 0.99의 상관관계를 가진다. 이를 기반으로 한 선형 외삽(partial_linear)을 베이스라인으로 삼고, LightGBM으로 잔차를 보정하면 정확도와 알림 성능을 동시에 개선할 수 있다."

실 데이터 검증:

| 관측 일수 | 상관계수 | 전체 대비 비율 | 표본 수 |
|----------|---------|--------------|--------|
| 10일 | 0.990 | 34.2% | 54,273 |
| 15일 | 0.994 | 50.4% | 54,337 |
| 20일 | 0.994 | 64.8% | 57,367 |

### 1.3 2단계 분석 전략

| | 1차 분석 | 2차 분석 |
|---|---------|---------|
| 목적 | 가설 검증 + 베이스라인 수립 | 잔차 학습 모델 + 실운영 재현성 |
| 모델 | LightGBM (직접 예측) | partial_linear + LightGBM (잔차 보정) |
| 기상 | ASOS 실측 (전체 월) | observed=ASOS, remainder=TMY |
| AMI | 미반영 | 역률, TOU, load_factor 반영 |
| 평가 기준 | MAPE | CVRMSE median (고객별 중위값) |
| 알림 | F1 전체 | F1 + 규모별 Precision/Recall |

1차 분석에서 partial_linear(r=0.99)의 강력한 성능을 확인하고, 2차 분석에서 이를 베이스라인으로 활용하는 잔차 학습 구조로 전환하였다.

---

## 2. 데이터 특성 분석

- 일반용(갑)저압 약 3,000호, 22개월 (2024-02 ~ 2025-11), 15분 단위 LP
- 데이터 품질: 결측/중복 0.001% 미만
- cold start 고객 존재 — 모델이 partial_kwh만으로도 예측 가능 (강건성)

### 알림 기저율

| 알림 조건 | 기저율 |
|----------|-------|
| 전월 대비 +30% | 19.8% |
| 전년동월 대비 +30% | 19.2% |
| 3개월 평균 +50% | 14.4% |
| **1건 이상** | **30.2%** |

### 역률 분포 (15분 단위 한전 약관 기준)

| 통계량 | 값 |
|-------|---|
| 평균 | 0.881 |
| 중위값 | 0.944 |

---

## 3. 외부 데이터 융합

| 데이터 | 출처 | 활용 |
|-------|------|------|
| ASOS 기상 실측 | 기상청 data.kma.go.kr | 관측 기간 기온/강수/일사 |
| TMY 표준기상년 | 기상청 | 잔여 기간 기상 추정 (3종: raw/bias/forecast) |
| 공휴일/특수일 | 공공데이터포털 data.go.kr | 캘린더 피처 |

### 기상 데이터의 observed/remainder 분리

예측 시점(+10d)에서 잔여 기간(11~말일)의 기온은 미래 데이터이므로 실측값 사용 불가. TMY(표준기상년)로 대체하여 실 운영 환경을 재현:

- **observed HDD/CDD**: ASOS 실측 (예측 시점까지 확정된 기온)
- **remainder HDD/CDD**: TMY 기반 추정 (실측 불가 구간)

---

## 4. 모델 구조: partial_linear + LightGBM 잔차 학습

### 4.1 1차 분석: 직접 예측

LightGBM이 `full_month_kwh`를 직접 예측. partial_linear 대비:

| 모델 | +10d MAE | +10d CVRMSE | +10d F1 |
|------|---------|-------------|---------|
| partial_linear | 143.0 | 21.1% | 0.817 |
| LightGBM (직접) | 187.7 | 29.8% | 0.771 |

partial_linear이 우세 → **가설 검증**: 부분 실측이 예측의 핵심.

### 4.2 2차 분석: 잔차 학습

1차 분석의 인사이트를 반영하여 모델 구조 전환:

```
최종 예측 = partial_linear 예측 + LightGBM 잔차 보정
         = (partial_kwh / days_observed × days_in_month) + LightGBM(잔차)
```

LightGBM이 `full_month_kwh - partial_linear`(잔차)만 학습:
- 타겟이 작아져 학습 용이
- partial_linear의 정확도를 기본으로 보존
- 물리적 제약: `max(예측, partial_kwh)` 클리핑 적용

<!-- 사진: fig02_pred_vs_actual.png -->

---

## 5. 피처 설계

### 5.1 피처 구성 (2차 분석)

| 그룹 | 주요 피처 | 소스 |
|------|---------|------|
| 부분월 실측 | partial_kwh, partial_rate | LP 15분 집계 |
| 이력 | prev_month_kwh, ma3_kwh, customer_mean_history | 월별 집계 |
| 캘린더 | month, n_workday, n_holiday, disruption_ratio | 공휴일 DB |
| 기상 | hdd_observed, cdd_observed, hdd_remainder(TMY), cdd_remainder(TMY) | ASOS + TMY |
| 기상(station) | temp_mean, cdh, hdh, solar_mj_mean, precip_sum, humidity, wind_speed | ASOS 월별 |
| AMI 패턴 | observed_power_factor, peak_ratio, load_factor, TOU 비율 | LP 15분 사전 계산 |
| 고객 속성 | contract_power_kw, industry_code | 고객 정보 |

### 5.2 SHAP 분석 (1차 분석 기준)

| 순위 | 피처 | SHAP 비중 |
|-----|------|----------|
| 1 | partial_rate | 79.7% |
| 2 | partial_kwh | 3.7% |
| 3 | prev_month_kwh | 3.1% |
| 4 | contract_power_kw | 2.7% |
| 5 | solar_mj_mean | 1.8% |

<!-- 사진: shap_summary.png -->

### 5.3 Ablation Study (2차 분석 — 피처 그룹 기여도)

| 그룹 | 제거 시 CVRMSE_med 변화 | 기여 방향 |
|------|----------------------|---------|
| {{abl_group_1}} | {{abl_contrib_1}} | {{abl_dir_1}} |
| {{abl_group_2}} | {{abl_contrib_2}} | {{abl_dir_2}} |
| {{abl_group_3}} | {{abl_contrib_3}} | {{abl_dir_3}} |
| {{abl_group_4}} | {{abl_contrib_4}} | {{abl_dir_4}} |

기여도 ≤ 0인 그룹은 자동 제거 후 튜닝에 반영.

---

## 6. 기상 방식 비교 (TMY Ablation)

| 방식 | remainder 소스 | CVRMSE_med |
|------|--------------|------------|
| regional (ASOS 실측) | ASOS 실측 | {{weather_regional}} |
| tmy_forecast | TMY + 예보 보정 | {{weather_tmy_forecast}} |
| tmy_bias | TMY + 편차 보정 | {{weather_tmy_bias}} |
| tmy_raw | TMY 순정 | {{weather_tmy_raw}} |

> "ASOS 실측(regional)이 최고 성능이나, 예측 시점에 잔여 기간 기온을 알 수 없으므로 실 운영에서는 TMY 기반 추정이 필요하다. TMY 대체 시 CVRMSE 증가가 {{weather_delta}}%p로 제한적이다."

**최종 모델**: remainder에 {{weather_best}} 적용

---

## 7. 모델 최적화

### 7.1 하이퍼파라미터 튜닝 (Optuna, {{tune_n_trials}} trial)

- 목적함수: 고객별 CVRMSE median 최소화
- 조기 종료: 25 trial 연속 미갱신 시

| 지표 | 기본값 | 최적값 |
|------|-------|-------|
| CVRMSE median | {{tune_baseline}} | {{tune_best}} |

### 7.2 잔차 보정 (RidgeCV)

| 지표 | 보정 전 | 보정 후 |
|------|--------|--------|
| MAE | {{resid_before_mae}} | {{resid_after_mae}} |
| RMSE | {{resid_before_rmse}} | {{resid_after_rmse}} |
| Ridge alpha (CV) | {{ridge_alpha}} | |

---

## 8. 최종 성능 평가

### 8.1 전체 모델 비교

| 모델 | MAPE | CVRMSE | CVRMSE_med | MAE | Bias | Precision | Recall | F1 |
|------|------|--------|------------|-----|------|-----------|--------|-----|
| partial_linear | {{pl_mape}} | {{pl_cvrmse}} | {{pl_cvmed}} | {{pl_mae}} | {{pl_bias}} | {{pl_prec}} | {{pl_recall}} | {{pl_f1}} |
| lgbm_tuned | {{tuned_mape}} | {{tuned_cvrmse}} | {{tuned_cvmed}} | {{tuned_mae}} | {{tuned_bias}} | {{tuned_prec}} | {{tuned_recall}} | {{tuned_f1}} |
| lgbm_corrected | {{corr_mape}} | {{corr_cvrmse}} | {{corr_cvmed}} | {{corr_mae}} | {{corr_bias}} | {{corr_prec}} | {{corr_recall}} | {{corr_f1}} |

### 8.2 규모별 성능 (lgbm_tuned)

| 규모 | Horizon | MAPE | CVRMSE | CVRMSE_med | Precision | Recall | F1 |
|------|---------|------|--------|------------|-----------|--------|-----|
| 0-100 | +10d | {{t_0100_10_mape}} | {{t_0100_10_cvrmse}} | {{t_0100_10_cvmed}} | {{t_0100_10_prec}} | {{t_0100_10_recall}} | {{t_0100_10_f1}} |
| 0-100 | +20d | {{t_0100_20_mape}} | {{t_0100_20_cvrmse}} | {{t_0100_20_cvmed}} | {{t_0100_20_prec}} | {{t_0100_20_recall}} | {{t_0100_20_f1}} |
| 100-1K | +10d | {{t_1001K_10_mape}} | {{t_1001K_10_cvrmse}} | {{t_1001K_10_cvmed}} | {{t_1001K_10_prec}} | {{t_1001K_10_recall}} | {{t_1001K_10_f1}} |
| 100-1K | +20d | {{t_1001K_20_mape}} | {{t_1001K_20_cvrmse}} | {{t_1001K_20_cvmed}} | {{t_1001K_20_prec}} | {{t_1001K_20_recall}} | {{t_1001K_20_f1}} |
| 1K-10K | +10d | {{t_1K10K_10_mape}} | {{t_1K10K_10_cvrmse}} | {{t_1K10K_10_cvmed}} | {{t_1K10K_10_prec}} | {{t_1K10K_10_recall}} | {{t_1K10K_10_f1}} |
| 1K-10K | +20d | {{t_1K10K_20_mape}} | {{t_1K10K_20_cvrmse}} | {{t_1K10K_20_cvmed}} | {{t_1K10K_20_prec}} | {{t_1K10K_20_recall}} | {{t_1K10K_20_f1}} |
| 10K+ | +10d | {{t_10K_10_mape}} | {{t_10K_10_cvrmse}} | {{t_10K_10_cvmed}} | {{t_10K_10_prec}} | {{t_10K_10_recall}} | {{t_10K_10_f1}} |
| 10K+ | +20d | {{t_10K_20_mape}} | {{t_10K_20_cvrmse}} | {{t_10K_20_cvmed}} | {{t_10K_20_prec}} | {{t_10K_20_recall}} | {{t_10K_20_f1}} |

### 8.3 규모별 성능 (partial_linear 대비)

| 규모 | Horizon | partial_linear CVRMSE | lgbm_tuned CVRMSE | 개선 |
|------|---------|----------------------|-------------------|------|
| 100-1K | +10d | {{pl_1001K_10_cvrmse}} | {{t_1001K_10_cvrmse}} | {{imp_1001K_10}} |
| 1K-10K | +10d | {{pl_1K10K_10_cvrmse}} | {{t_1K10K_10_cvrmse}} | {{imp_1K10K_10}} |
| 10K+ | +10d | {{pl_10K_10_cvrmse}} | {{t_10K_10_cvrmse}} | {{imp_10K_10}} |

<!-- 사진: fig04_monthly_error_boxplot.png, fig10_mape_heatmap.png -->

---

## 9. 스케일 아웃 설계

| 규모 | 예상 메모리 | 예상 시간 | 병렬화 |
|------|-----------|---------|-------|
| 10,000호 | 5.1 GB | 2.8분 | 가능 |
| 100,000호 | 51.4 GB | 28.5분 | 가능 |
| 1,000,000호 | 514.2 GB | 284.5분 | 가능 |

- 고객 축 청크 분할 (청크 간 독립, 병렬화 가능)
- 글로벌 모델 1개 — 고객 수 증가해도 모델 크기 불변
- 학습/추론 분리 — 추론은 검침일 주기 배치
- Config 기반 경로 주입 — 환경 전환 시 코드 변경 불필요
- partial_linear + 잔차 구조 — 잔차 모델만 재학습하면 되므로 효율적

---

## 10. 활용 시나리오 및 정책 실현성

### 운영 흐름
```
매 검침일 (한전 7차 검침 주기)
  → +10일: partial_linear + LightGBM 잔차 → 1차 알림
  → +20일: 동일 구조 (정확도 향상) → 2차 알림
  → 검침 완료: 실측값 기반 다음 주기 학습 데이터 갱신
```

### 물리적 제약 반영
- 예측값 ≥ partial_kwh 클리핑 — 이미 실측된 사용량보다 낮은 예측 방지
- 잔여 기간 기상은 TMY 기반 — 실 운영 환경 재현

### cold start 대응
- 과거 데이터 없는 신규 고객도 partial_kwh만으로 예측 가능
- LightGBM이 NaN 피처를 자체 처리 → 별도 분류 없이 동일 모델로 추론

---

## 11. 한계 및 향후 개선

- 22개월 데이터 제약: val 미분리, 계절 1사이클만 학습
- 0 고객 MAPE 한계: CVRMSE median 등 대안 지표 사용
- Walk-forward training 미적용: 스케일 아웃 비용 고려, 주기적 배치 재학습 권장
- 잔차 보정(Ridge) 효과 제한적: 데이터 증가 시 효과 기대
- 앙상블(partial_linear + LightGBM 가중 평균) 미검토

# 보고서 핵심 포인트 모음

## 1. 가설 검증
- partial_kwh와 full_month_kwh 상관계수 0.990(10일), 0.994(20일)
- 부분월 안정성: 역률 상관 0.958(10일), 일변동계수 0.855(10일)
- partial_linear(선형 외삽)이 MAE/CVRMSE에서 강력한 베이스라인

## 2. 잔차 학습 구조 (2차 분석 핵심)
- 최종 예측 = partial_linear + LightGBM(잔차 보정)
- partial_linear의 정확도를 기본으로 보존하면서 LightGBM이 비선형 패턴만 학습
- 물리적 제약: max(예측, partial_kwh) 클리핑 — 이미 실측된 사용량보다 낮은 예측 방지
- 로컬 테스트에서 partial_linear 대비 MAPE 19.6% → 7.3%, F1 0.80 → 0.91

## 3. TMY 잔여기간 대체
- 예측 시점(+10d)에 잔여기간(11~말일) 기온은 미래 데이터 → 실측 사용 불가
- observed HDD/CDD: ASOS 실측 (예측 시점까지 확정)
- remainder HDD/CDD: TMY 기반 추정 (3종: raw/bias/forecast)
- ablation에서 TMY 간 비교 + regional(실측)은 optimal 후보에서 제외
- "TMY 대체 시 성능 저하가 제한적" → 실 운영 가능성 입증

## 4. AMI 피처 (15분 데이터 고유 가치)
- 역률(observed_power_factor), 부하율(load_factor), TOU 비율, 일변동계수 등
- 일별 집계에서는 산출 불가, 15분 LP에서만 계산 가능
- 한전 약관 기준 역률 계산 (지상/진상 미합산, 구간별 독립)
- booster importance에서 load_factor, daily_cv, TOU 등 상위권

## 5. 검침일 수렴 분석 (스케일 아웃 근거)
- 2일 [1,15] → 6일 [1,5,10,15,20,25] → 12일 [1,3,...,28]
- 검침일 수를 늘려도 결과가 수렴하면: "2일 샘플링으로 충분, 평가 시간 6배 단축"
- 스케일 아웃 시 전수 평가(31일) 불필요 → 대표 검침일 샘플링으로 효율적 평가
- 심사 기준 "스케일 아웃을 고려한 코드 설계의 논리적 정합성"에 직접 부합

## 6. 알림 성능
- 사전 알림 서비스 특성상 Recall이 Precision보다 중요
- 미탐지(놓침): 고객이 예고 없이 과다 청구 → 불만
- 과잉 알림: "높을 수 있습니다" → 확인 후 괜찮음 → 경미
- 조건별 P/R/F1 산출: prev+30%, yoy+30%, ma3+50% 개별 평가

## 7. cold start 대응
- 과거 데이터 없는 신규 고객도 partial_kwh만으로 예측 가능
- LightGBM이 NaN 피처를 자체 처리 → 별도 분류 없이 동일 모델
- 보고서: "신규 계약 고객에 대해서도 검침일 이후 부분 실측만으로 예측 가능"

## 8. 평가 방법론
- CVRMSE median: 고객별 CVRMSE의 중위값 → 0 고객 왜곡 방지, 공정한 비교
- 규모별(0-100/100-1K/1K-10K/10K+) × horizon별(+10d/+20d) 체계적 평가
- sliding window evaluation (walk-forward evaluation)

## 9. 1차→2차 스토리
- 1차: 직접 예측 → partial_linear이 LightGBM보다 우수 발견
- 1차 인사이트: partial_kwh가 SHAP 79.7% → "이걸 베이스라인으로 쓰자"
- 2차: 잔차 학습으로 전환 → partial_linear 상회
- 2차 추가: TMY 재현성, AMI 피처, 체계적 평가, 수렴 분석

## 10. 설계 강점
- config 기반 경로 주입 (하드코딩 없음)
- 로컬 합성 → DSZ 실데이터 분리 구조 (코드 재현성 자연 보장)
- 청크 기반 전처리 (34GB 대응, 매 청크 중간 저장)
- 인코딩 방어 (encoding_errors="replace" 항상 적용)
- 글로벌 모델 1개 → 고객 수 증가해도 모델 크기 불변
- 100만호 추정 284분, 병렬화 가능

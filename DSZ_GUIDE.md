# DSZ 3차 방문 실행 가이드

## 사전 준비
- `dsz_code_v3.zip` USB 반입
- `dsz_results_template.csv` 폰에 미리 전송
- 이전 실행 결과(data/preprocessed/, ASOS/, checkpoints/, weights/) 유지 확인

## 압축 해제 후 확인
```cmd
dir data\preprocessed\daily.parquet
dir data\preprocessed\ami_features.parquet
dir ASOS\asos_all.parquet
dir data\weather\station_*.csv
```
모두 있어야 함. 없으면 이전 방문 폴더에서 복사.

## 실행 순서

### 1. Ablation (약 15-20분)
```cmd
python -m scripts.ablation_study --source configs/source_dsz.yaml --train-end 2024-12
```
확인:
- `[weather_aggs] ASOS 직접 로드` 출력되는지
- `hdd_remainder 비null` 0이 아닌지
- `[AMI 조인] 13개 피처 추가` 출력되는지
- TMY 비교에서 값이 서로 다른지
- Phase 3 optimal이 regional이 아닌 TMY인지

결과: `ablation_results/best_config.json`, `ablation_results/ablation_results.csv`

### 2. 튜닝 (약 10-15분)
```cmd
python -m scripts.tune_lgbm --source configs/source_dsz.yaml --train-end 2024-12
```
확인:
- `[피처 제거]` 에서 뭐가 제거되었는지
- `잔차 학습` 출력되는지
- `CVRMSE median` 값이 출력되는지
- 25 trial 연속 미갱신 시 조기 종료되는지

결과: `weights/dsz_lgbm_tuned/`, `tuning_results/best_params.json`

### 3. 잔차 보정 (약 1-2분)
```cmd
python -m scripts.run_residual_correction --source configs/source_dsz.yaml --train-end 2024-12
```
확인:
- `[Ridge] CV 최적 alpha=` 값
- 보정 전후 MAE/RMSE 차이

결과: `residual_correction/residual_summary.json`

### 4. 슬라이딩 평가 (약 10-15분)
```cmd
python -m scripts.run_sliding_eval --source configs/source_dsz.yaml --train-end 2024-12
```
확인:
- 4개 모델 전부 성공하는지 (partial_linear, lgbm_base, lgbm_tuned, lgbm_corrected)
- `실패` 문구 없는지

결과:
- `sliding_results_partial_linear/error_overall.csv`, `error_by_scale.csv`
- `sliding_results_lgbm_base/error_overall.csv`, `error_by_scale.csv`
- `sliding_results_lgbm_tuned/error_overall.csv`, `error_by_scale.csv`
- `sliding_results_lgbm_corrected/error_overall.csv`, `error_by_scale.csv`

## 결과 기록 (폰에 수기 입력)

### error_overall.csv 열 순서
n, mae, rmse, nrmse_pct, cvrmse_pct, mape_pct, smape_pct, bias, std_error, cvrmse_median_pct, cvrmse_q25_pct, cvrmse_q75_pct, alarm_precision, alarm_recall, alarm_f1

### 필수 기록 항목
1. 4개 모델의 error_overall.csv → MAPE, CVRMSE, CVRMSE_med, Bias, Precision, Recall, F1
2. lgbm_tuned + partial_linear의 error_by_scale.csv → 규모별 위 지표
3. ablation_results.csv → 각 실험의 cvrmse_med_10d
4. best_params.json → 최적 하이퍼파라미터
5. residual_summary.json → 보정 전후 + alpha
6. 10_alarm_base_rates.json → 기저율 (이전 반출과 동일하면 스킵)

### 빠르게 보는 명령어
```cmd
python -c "import pandas as pd; df=pd.read_csv('sliding_results_lgbm_tuned/error_overall.csv'); print(df.to_string())"
python -c "import pandas as pd; df=pd.read_csv('sliding_results_lgbm_tuned/error_by_scale.csv'); print(df.to_string())"
python -c "import pandas as pd; df=pd.read_csv('sliding_results_partial_linear/error_overall.csv'); print(df.to_string())"
python -c "import pandas as pd; df=pd.read_csv('sliding_results_partial_linear/error_by_scale.csv'); print(df.to_string())"
python -c "import pandas as pd; df=pd.read_csv('sliding_results_lgbm_corrected/error_overall.csv'); print(df.to_string())"
python -c "import pandas as pd; df=pd.read_csv('ablation_results/ablation_results.csv'); print(df[['name','cvrmse_med_10d','cvrmse_med_20d','mape_10d']].to_string())"
python -c "import json; print(json.dumps(json.load(open('tuning_results/best_params.json',encoding='utf-8')),indent=2,ensure_ascii=False))"
python -c "import json; print(json.dumps(json.load(open('residual_correction/residual_summary.json',encoding='utf-8')),indent=2,ensure_ascii=False))"
```

## 에러 발생 시
- `ASOS 파일 없음` → `dir /s asos_all.parquet`으로 위치 확인 후 ASOS/ 폴더에 복사
- `ami_features.parquet 없음` → AMI 없이 진행 (자동 스킵)
- `categorical_feature do not match` → 이전 모델과 새 코드 불일치. checkpoints/ 삭제 후 재실행
- `cannot convert the series to float` → monthly 중복. DSZ 단일 contract_type이면 발생 안 함

## 시간 예상
| 단계 | 예상 소요 |
|------|---------|
| 1. Ablation | 15-20분 |
| 2. 튜닝 | 10-15분 |
| 3. 잔차 보정 | 1-2분 |
| 4. 슬라이딩 평가 | 10-15분 |
| 5. 결과 기록 | 10-15분 |
| **합계** | **약 50-70분** |

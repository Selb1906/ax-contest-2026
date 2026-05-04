# 안심구역 환경 설정 가이드

## 1. 패키지 설치 (오프라인)

```bash
# 이미 설치된 패키지 확인
pip list

# 부족한 패키지만 오프라인 설치
pip install --no-index --find-links=wheels/ lightgbm pandas pyarrow scipy scikit-learn pyyaml numpy
```

## 2. 데이터 소스 설정

```bash
# source_dsz.yaml 의 path 를 실제 LP 데이터 경로로 수정
# column_map 은 실제 컬럼명 확인 후 수정
```

## 3. 실행 순서

```bash
# Step 1: 데이터 스키마 확인
python -c "import pandas as pd; df = pd.read_csv('LP경로', nrows=5); print(df.columns.tolist()); print(df.dtypes)"

# Step 2: source_dsz.yaml column_map 수정 후 로딩 테스트
python -c "from src.io_adapter import load_from_yaml; df = load_from_yaml('configs/source_dsz.yaml', validate=False); print(df.shape, df.dtypes)"

# Step 3: BTM 탐지
python -m scripts.detect_btm --source configs/source_dsz.yaml

# Step 4: 프로파일러 (반출용 통계)
python -m scripts.profile --source configs/source_dsz.yaml --out profile_stats/dsz

# Step 5: 베이스라인 평가
python -m scripts.run_baselines --source configs/source_dsz.yaml

# Step 6: LightGBM 학습 + 평가
python -m scripts.train_lgbm --source configs/source_dsz.yaml --train-end 2023-12 --compare-baselines

# Step 7: UI 데이터 생성 (선택)
python -m scripts.prepare_ui_data
```

## 4. 반출 대상 파일

- `profile_stats/dsz/` — 집계 통계 (MANIFEST.json 포함)
- `btm_results/btm_summary.json` — BTM 집계 (btm_flags.csv는 반출 불가)
- `eval_results/` — 모델 평가 결과
- `weights/` — 학습된 모델 가중치

# 로컬 테스트 가이드 (팀원용)

합성 데이터로 전체 파이프라인을 로컬에서 테스트하는 방법입니다.

## 1. 환경 준비

### 저장소 클론
```bash
git clone https://github.com/lemon19967-lab/-.git
cd -
```

### Python 의존성 설치
```bash
pip install -r requirements.txt
```

matplotlib이 빠져있으면:
```bash
pip install matplotlib tqdm
```

SHAP 분석도 하려면:
```bash
pip install shap
```

## 2. 환경 검증
```bash
python -m scripts.verify_env
```
전부 통과해야 합니다. shap이 없으면 `pip install shap`으로 설치.

## 3. 합성 데이터 생성
```bash
python -m scripts.gen_synth
```
`data/synth/lp_v0.parquet` 생성됨 (100호, 6개월).

## 4. 전체 파이프라인 실행
```bash
python -m scripts.run_full_analysis --source configs/source_synth.yaml --train-end 2022-04 --skip-profile
```

### 옵션 설명
- `--source`: 데이터 소스 YAML
- `--train-end`: 학습 종료 기간 (이후는 테스트)
- `--skip-profile`: 프로파일러 건너뛰기 (시간 절약)
- `--skip-shap`: SHAP 건너뛰기 (shap 미설치 시)
- `--resume`: 중단 후 이어서 실행

### 예상 시간 (합성 100호)
- Step 1~5: 5분
- Step 5.5 Sliding: 5분
- Step 6 SHAP: 2분 (shap 설치 시)
- Step 7~11: 10분
- 전체: 약 20~30분

## 5. Ablation 별도 실행
```bash
python -m scripts.ablation_study --source configs/source_synth.yaml --train-end 2022-04 --skip-phase4
```
`--skip-phase4`: Optuna 튜닝 건너뛰기 (시간 절약)

## 6. 대시보드 생성
```bash
python -m scripts.prepare_ui_data --source configs/source_synth.yaml
python -m scripts.generate_dashboard
```
`dashboard.html` 열어서 확인.

## 7. 보고서용 이미지 생성
```bash
python -m scripts.generate_figures
```
`figures/` 폴더에 PNG 생성.

## 8. 반출용 파일 수집
```bash
python -m scripts.export_summary --dashboard
```
`export/` 폴더에 모든 반출 파일 수집.

## 주의사항
- 합성 데이터는 6개월뿐이라 성능 지표가 실제와 다름 (MAPE 높음, F1 낮음)
- seasonal_yoy, partial_seasonal_yoy는 전년 데이터 없어서 NaN
- Ablation에서 BTM 미발견 (합성 데이터 특성)
- 대시보드의 고객 ID는 합성 (GEN_000000 등)

## 디렉토리 구조 (실행 후)
```
eval_results/       — 모델 평가 결과
sliding_results/    — 검침일별 상세 분석
explain_results/    — SHAP 이미지
ablation_results/   — Ablation 결과
residual_correction/ — 잔차 보정
weights/            — 모델 가중치
figures/            — 보고서용 이미지
export/             — 반출 파일 (export_summary 실행 후)
data/ui/            — 대시보드용 데이터
dashboard.html      — 대시보드
```

## 문제 해결
- `ModuleNotFoundError`: `pip install <패키지명>`
- stdout 버퍼링 (출력 안 보임): 스크립트에 `line_buffering=True` 확인
- Windows 한글 깨짐: 터미널에서 `chcp 65001` 실행 후 재시도

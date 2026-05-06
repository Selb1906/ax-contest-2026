# 안심구역 반입 번들

## [관리자용] 패키지 설치 요청

`wheels/` 디렉토리의 Python 패키지를 분석 환경(Python 3.9)에 설치 부탁드립니다.

```bash
pip install --no-index --find-links=wheels/ wheels/*.whl
```

포함 패키지 (58개 wheel): numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm,
shap, matplotlib, optuna, pyyaml, openpyxl, tqdm + 의존성.

---

## [분석자용] 실행 가이드

### 작업 디렉토리 구성

`code/` 디렉토리를 작업 위치에 복사하고, 그 안에서 모든 명령을 실행합니다.
모든 경로는 **프로젝트 루트(code/) 기준 상대경로**로 동작합니다.

```
code/               ← 여기가 프로젝트 루트 (cd 여기서 실행)
├── src/            ← 핵심 모듈
├── scripts/        ← 실행 스크립트
├── configs/        ← YAML 설정 (경로 수정 필요)
├── ui/             ← 대시보드
└── run.py          ← 대화형 실행기
```

### 1. 환경 검증

```bash
cd code/
python -m scripts.verify_env
```

### 2. LP 데이터 찾기 + 경로 설정

**LP 경로를 모를 때** — 파일 시스템 자동 탐색:

```bash
python -m scripts.inspect_lp --scan              # 전체 드라이브 탐색
python -m scripts.inspect_lp --scan --root D:/   # 특정 드라이브만
```

CSV/parquet 중 "전력", "검침", "LP", "AMI" 등 키워드가 포함되거나
10MB 이상인 파일을 찾아서 목록으로 보여줍니다.

**LP 경로를 알 때** — 구조 탐색 + yaml 자동 생성:

```bash
python -m scripts.inspect_lp --path /data/lp/
# → 컬럼 구조 출력 + configs/source_dsz.yaml 자동 생성
```

자동 생성된 yaml의 컬럼 매핑을 확인·수정:

```yaml
source:
  kind: dsz_lp
  path: "/data/lp/"             # ← 탐색에서 확인한 경로
  column_map:
    customer_id: "고객번호"      # ← 실제 컬럼명으로 수정
    ts: "검침일시"
    contract_type: "계약종별"
    p_active_kwh: "유효전력량"
    p_reactive_kwh: "무효전력량"
    max_demand_kw: "최대수요전력"
```

### 3. 분석 실행

```bash
# 대화형 (팀 작업 분배 + 메뉴)
python run.py

# 또는 원스톱 파이프라인
python -m scripts.run_full_analysis --source configs/source_dsz.yaml --train-end 2023-12
```

### 4. 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--resume` | 중단된 지점에서 이어서 실행 |
| `--skip-shap` | SHAP 분석 건너뛰기 (시간 절약) |
| `--skip-profile` | 프로파일러 건너뛰기 |
| `--meter-days 1,15` | 검침일 2개만 평가 (기본: 1~28 전수) |

### 5. 반출 대상

| 경로 | 내용 | 반출 |
|------|------|------|
| `profile_stats/` | 집계 통계 | O |
| `eval_results/` | 모델 평가 지표 | O |
| `sliding_results/` | 검침일별 상세 평가 | O |
| `explain_results/` | SHAP 피처 중요도 | O |
| `weights/` | 모델 가중치 | O |
| `btm_results/btm_summary.json` | BTM 집계 | O |
| `btm_results/btm_flags.csv` | 고객별 플래그 | X (고객ID) |

### 6. 트러블슈팅

- **한글 깨짐**: matplotlib 폰트 문제 → Malgun Gothic (Win) / NanumGothic (Linux)
- **메모리 부족**: `--meter-days 1,15`로 축소, `--skip-shap`
- **중단 후 재개**: `--resume` (체크포인트 자동 저장됨)
- **컬럼 매핑 오류**: `python -m scripts.inspect_lp --path <경로>`로 재탐색

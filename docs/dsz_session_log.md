# 데이터안심구역 1차 방문 기록 (2026-05-07)

## 환경 정보
- PC 사양: RAM 128GB, Python 3.9.7 (64bit), Windows
- LP 데이터: 34GB CSV 단일 파일, 약 4200만 행, UTF-8 (BOM)
- 파일명: `(제공) 26년 AX 아이디어 경진대회 일반용+상업용 데이터.csv`
- 경로에 한글 포함 → 영문 경로(`E:\lpdata\data.csv`)로 복사하여 해결

## 해결한 이슈

### 1. 의존성 설치
- 시스템 Python에 기존 패키지와 충돌 발생 → 가상환경(venv) 생성으로 해결
- `--no-deps` 플래그로 충돌 무시하고 설치
- **Python 3.9 전용 하위 의존성 5개 누락** 발견:
  - `tomli`, `importlib-resources`, `importlib-metadata`, `zipp`, `exceptiongroup`
  - 원인: 로컬 3.11에서는 내장이라 wheel에 미포함
  - `missing_py39_wheels.zip` (141KB) 별도 제작하여 메일로 전달 요청

### 2. 터미널 환경
- `python` 명령 안 됨 → `py` 사용 (Windows Python Launcher)
- 가상환경 활성화 후에는 `python` 직접 사용 가능
- 따옴표 문제: 큰따옴표/작은따옴표 모두 문제 → `python -c` 대신 스크립트 파일로 실행

### 3. 파일 경로 인코딩
- yaml path에 한글 경로 → `cp949 codec can't decode` 에러
- 해결: LP 파일을 `E:\lpdata\data.csv`로 복사, yaml 경로 수정

### 4. CSV 읽기 에러
- C 엔진: `Error tokenizing data. Calling read(nbytes) on source failed`
- 원인: UTF-8 파일 내 일부 깨진 바이트
- `engine="python"` → 동작하지만 34GB에서 극도로 느림
- **최종 해결 시도**: `fix_csv.py`로 깨진 바이트 제거한 clean.csv 생성 중 시간 종료

## 다음 방문 시 해야 할 것

### 사전 준비 (로컬에서)
1. `missing_py39_wheels.zip` 직원에게 사전 설치 요청
2. `io_adapter.py` 수정본 반영 (encoding_errors 처리)
3. `fix_csv.py` 코드 zip에 포함

### 안심구역 도착 후 순서
1. 가상환경 활성화: `E:\반입데이터\venv\Scripts\activate`
2. 누락 패키지 설치 확인 (이미 직원이 설치했을 수 있음)
3. `python -m scripts.verify_env` 전체 통과 확인
4. `fix_csv.py` 실행 → `clean.csv` 생성 (약 30~60분 예상)
5. yaml path를 `clean.csv`로 설정
6. `python -m scripts.run_full_analysis --source configs/source_dsz.yaml`
7. 이후 pipeline 순서대로 진행

### io_adapter.py 수정 사항 (165줄)
```python
# 현재 (에러 발생):
                d = pd.read_csv(sp, encoding="utf-8-sig")

# 수정 후:
                d = pd.read_csv(sp, on_bad_lines="skip")
```
clean.csv는 이미 UTF-8이므로 encoding 지정 불필요.

## 데이터 특성 (확인된 것)
- 파일 포맷: CSV, UTF-8 BOM
- 행 수: 약 4200만 행
- 용량: 34GB
- 계약종: 일반용 + 상업용 (주택용 미제공)
- 엑셀 최대 104만 행 → 전체 확인 불가, Python으로만 처리

## 미해결
- clean.csv 생성 미완료 (fix_csv.py 실행 중 시간 종료)
- 실제 파이프라인 Step 1 이후 진행 안 됨
- 컬럼 구조 미확인 (로딩 실패로 컬럼 확인 불가)

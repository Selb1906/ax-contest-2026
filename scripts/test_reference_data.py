"""자료실 샘플 데이터 확인 + 모의 테스트."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np

ref_dir = ROOT / "자료실"
files = list(ref_dir.glob("*.xlsx"))
print(f"파일 {len(files)}개: {[f.name for f in files]}")

for fpath in files:
    print(f"\n{'='*60}")
    print(f"{fpath.name}")
    print(f"{'='*60}")

    # 헤더 행(row 6) + 데이터 읽기
    header = pd.read_excel(fpath, header=None, skiprows=6, nrows=1)
    header_vals = [str(v) for v in header.iloc[0] if str(v) != 'nan']
    print(f"헤더: {header_vals}")

    # 타입 행(row 7) 스킵, 데이터는 row 8부터
    df = pd.read_excel(fpath, header=None, skiprows=8)
    # 빈 행 제거
    df = df.dropna(how='all')
    print(f"행: {len(df):,}")
    print(f"컬럼 수: {df.shape[1]}")

    # 컬럼명 부여
    if len(header_vals) <= df.shape[1]:
        # 컬럼 수가 맞으면 헤더 적용
        for i, h in enumerate(header_vals):
            df = df.rename(columns={i: h})

    print(f"\n컬럼: {df.columns.tolist()}")
    print(f"\ndtypes:")
    for col in df.columns[:10]:
        print(f"  {col}: {df[col].dtype}  sample={df[col].iloc[0] if len(df)>0 else 'N/A'}")

    print(f"\n상위 5행:")
    print(df.head().to_string(index=False))

    # 고객 수, 기간
    if '고객번호' in df.columns:
        print(f"\n고객 수: {df['고객번호'].nunique()}")
    if '년도' in df.columns:
        print(f"년도 범위: {sorted(df['년도'].unique())}")
    if '월' in df.columns:
        print(f"월 범위: {sorted(df['월'].unique())}")

    # 마스킹 확인
    for col in df.columns:
        sample = str(df[col].iloc[0]) if len(df) > 0 else ''
        if '*' in sample:
            print(f"\n⚠ 마스킹 발견: {col} = '{sample}'")
            break
    else:
        print("\n✅ 마스킹 없음 — 실제 값 사용 가능")

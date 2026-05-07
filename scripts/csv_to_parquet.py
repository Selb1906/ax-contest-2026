"""대용량 CSV → Parquet 변환 (인코딩 정리 + 용량 축소 + 읽기 속도 개선).

34GB CSV를 청크 단위로 읽어서 Parquet 디렉토리로 변환합니다.
- 인코딩 에러 자동 처리
- 숫자 컬럼 자동 변환 (문자열 → 숫자)
- 34GB CSV → ~3-5GB Parquet (10배 축소)
- 이후 pd.read_parquet()로 1~2분 내 로딩

사용법:
  python scripts/csv_to_parquet.py "E:\\경로\\data.csv" "E:\\lpdata\\data.parquet"
"""
import sys
import time
from pathlib import Path

if len(sys.argv) < 3:
    print("사용법: python scripts/csv_to_parquet.py <입력CSV> <출력parquet>")
    sys.exit(1)

src = sys.argv[1]
dst = sys.argv[2]

print(f"입력: {src}")
print(f"출력: {dst}")

try:
    import pandas as pd
    import pyarrow
except ImportError:
    print("pandas, pyarrow 필요: pip install pandas pyarrow")
    sys.exit(1)

CHUNK_SIZE = 500_000

# 인코딩 감지
encoding = None
for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
    try:
        test = pd.read_csv(src, encoding=enc, nrows=5)
        encoding = enc
        print(f"  인코딩: {enc}")
        print(f"  컬럼 ({len(test.columns)}개): {test.columns.tolist()}")
        break
    except Exception:
        continue

if encoding is None:
    encoding = "utf-8"
    print("  인코딩 감지 실패 -> utf-8")

# 숫자 변환 함수
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > len(df) * 0.5:
                    df[col] = converted
            except Exception:
                pass
    return df

# 청크 단위 변환
print(f"청크 크기: {CHUNK_SIZE:,}행")
print("변환 시작...")

t0 = time.time()
total_rows = 0
chunk_num = 0

# 단일 parquet 파일로 저장 (pyarrow ParquetWriter 사용)
import pyarrow as pa
import pyarrow.parquet as pq

writer = None
schema = None

reader = pd.read_csv(
    src,
    encoding=encoding,
    on_bad_lines="skip",
    chunksize=CHUNK_SIZE,
)

for i, chunk in enumerate(reader):
    chunk_num = i + 1
    total_rows += len(chunk)
    elapsed = time.time() - t0

    chunk = optimize_dtypes(chunk)
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        schema = table.schema
        writer = pq.ParquetWriter(dst, schema, compression="snappy")

    # 스키마 맞추기 (청크마다 타입이 다를 수 있음)
    try:
        table = table.cast(schema)
    except Exception:
        pass

    writer.write_table(table)

    if chunk_num % 5 == 0 or chunk_num <= 3:
        rate = total_rows / elapsed if elapsed > 0 else 0
        print(f"  청크 {chunk_num}: {total_rows:,}행 ({elapsed:.0f}초, {rate:,.0f}행/초)")

if writer:
    writer.close()

# 결과
src_size = Path(src).stat().st_size / (1024**3)
dst_size = Path(dst).stat().st_size / (1024**3)
total_time = time.time() - t0

print(f"\n{'='*50}")
print(f"  CSV:     {src_size:.1f} GB")
print(f"  Parquet: {dst_size:.1f} GB ({dst_size/src_size*100:.0f}%)")
print(f"  행 수:   {total_rows:,}")
print(f"  소요:    {total_time:.0f}초 ({total_time/60:.1f}분)")
print(f"{'='*50}")
print(f"\nyaml에서 path를 변경:")
print(f'  path: "{dst}"')

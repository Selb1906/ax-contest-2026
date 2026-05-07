"""대용량 CSV → Parquet 분할 변환.

1.3억 행 / 34GB CSV를 100만 행 단위로 분할하여 Parquet 디렉토리로 저장.
- 34GB CSV → ~3-5GB Parquet (10배 축소)
- 분할 저장으로 메모리 부하 최소화
- pd.read_parquet("디렉토리/") 로 한번에 읽기 가능

사용법:
  python scripts/csv_to_parquet.py "E:\\경로\\data.csv" "E:\\lpdata\\lp_parquet"
"""
import sys
import time
from pathlib import Path

if len(sys.argv) < 3:
    print("사용법: python scripts/csv_to_parquet.py <입력CSV> <출력디렉토리>")
    sys.exit(1)

src = sys.argv[1]
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

print(f"입력: {src}")
print(f"출력: {dst}/")

try:
    import pandas as pd
except ImportError:
    print("pandas 필요: pip install pandas pyarrow")
    sys.exit(1)

CHUNK_SIZE = 1_000_000  # 100만 행씩

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

print(f"  청크: {CHUNK_SIZE:,}행")
print("  변환 시작...")

t0 = time.time()
total_rows = 0

reader = pd.read_csv(
    src,
    encoding=encoding,
    on_bad_lines="skip",
    chunksize=CHUNK_SIZE,
)

for i, chunk in enumerate(reader):
    total_rows += len(chunk)
    elapsed = time.time() - t0

    # 숫자 변환
    for col in chunk.columns:
        if chunk[col].dtype == object:
            try:
                converted = pd.to_numeric(chunk[col], errors="coerce")
                if converted.notna().sum() > len(chunk) * 0.5:
                    chunk[col] = converted
            except Exception:
                pass

    # 파트 저장
    part_path = dst / f"part_{i:04d}.parquet"
    chunk.to_parquet(part_path, index=False, engine="pyarrow", compression="snappy")

    rate = total_rows / elapsed if elapsed > 0 else 0
    print(f"  part_{i:04d}: {total_rows:,}행 ({elapsed:.0f}초, {rate:,.0f}행/초)")

# 결과
src_size = Path(src).stat().st_size / (1024**3)
dst_size = sum(f.stat().st_size for f in dst.glob("*.parquet")) / (1024**3)
n_parts = len(list(dst.glob("*.parquet")))
total_time = time.time() - t0

print(f"\n{'='*50}")
print(f"  CSV:     {src_size:.1f} GB")
print(f"  Parquet: {dst_size:.1f} GB ({n_parts}개 파트)")
print(f"  행 수:   {total_rows:,}")
print(f"  소요:    {total_time:.0f}초 ({total_time/60:.1f}분)")
print(f"{'='*50}")
print(f"\nyaml에서 path를 변경:")
print(f'  path: "{dst}"')

"""CSV 인코딩 정리 — 깨진 바이트 제거.

안심구역에서 LP CSV의 깨진 바이트를 제거한 clean 버전을 생성.
UTF-8 BOM 파일에서 일부 깨진 바이트가 있을 때 사용.

사용법:
  python fix_csv.py E:\\lpdata\\data.csv E:\\lpdata\\clean.csv
  python fix_csv.py <입력파일> <출력파일>
"""
import sys

if len(sys.argv) < 3:
    print("사용법: python fix_csv.py <입력파일> <출력파일>")
    print("예: python fix_csv.py E:\\lpdata\\data.csv E:\\lpdata\\clean.csv")
    sys.exit(1)

src = sys.argv[1]
dst = sys.argv[2]

print(f"입력: {src}")
print(f"출력: {dst}")
print("reading...")

count = 0
with open(src, "r", encoding="utf-8", errors="replace") as fin:
    with open(dst, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            fout.write(line)
            count += 1
            if count % 100000 == 0:
                print(f"  {count:,} lines done")

print(f"done: {count:,} lines -> {dst}")

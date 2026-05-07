"""ASOS CSV 디렉토리 → 단일 Parquet 변환.

ASOS/ 디렉토리 내 모든 CSV 파일을 읽어 asos_all.parquet 으로 저장.
src/weather/asos.py 의 read_asos_csv() 로직을 그대로 사용.

사용법:
  python scripts/convert_asos_parquet.py ASOS/
  python scripts/convert_asos_parquet.py "E:\\경로\\ASOS"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.weather.asos import read_asos_csv  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python scripts/convert_asos_parquet.py <ASOS 디렉토리>",
              flush=True)
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"오류: 디렉토리가 존재하지 않습니다: {directory}", flush=True)
        sys.exit(1)

    output_path = directory / "asos_all.parquet"

    csv_files = sorted(f for f in directory.glob("*.csv") if not f.name.startswith("_"))
    if not csv_files:
        print(f"오류: {directory} 에 CSV 파일이 없습니다.", flush=True)
        sys.exit(1)

    print(f"ASOS CSV → Parquet 변환", flush=True)
    print(f"  입력 디렉토리: {directory}", flush=True)
    print(f"  CSV 파일 수:   {len(csv_files)}", flush=True)
    print(f"  출력:          {output_path}", flush=True)
    print("", flush=True)

    import pandas as pd

    frames = []
    t0 = time.time()
    for i, f in enumerate(csv_files, 1):
        t_file = time.time()
        df = read_asos_csv(f)
        elapsed = time.time() - t_file
        print(f"  [{i}/{len(csv_files)}] {f.name}: {len(df):,}행 ({elapsed:.1f}초)",
              flush=True)
        frames.append(df)

    print("", flush=True)
    print("  합치는 중...", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["station_id", "ts"]).reset_index(drop=True)

    print(f"  전체: {len(combined):,}행", flush=True)
    print(f"  Parquet 저장 중: {output_path}", flush=True)
    combined.to_parquet(output_path, index=False, engine="pyarrow")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    total_time = time.time() - t0
    print("", flush=True)
    print(f"  완료: {file_size_mb:.1f} MB, {total_time:.1f}초", flush=True)


if __name__ == "__main__":
    main()

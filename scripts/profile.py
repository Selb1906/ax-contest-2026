"""프로파일러 CLI.

사용:
    python -m scripts.profile --source configs/source_synth.yaml --out profile_stats/synth
    python -m scripts.profile --source configs/source_public_apt.yaml --out profile_stats/public_apt
    python -m scripts.profile --source configs/source_dsz.yaml       --out profile_stats/dsz
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from src import io_adapter, profiler


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="소스 YAML 경로")
    p.add_argument("--out", required=True, help="profile_stats 출력 디렉터리")
    p.add_argument("--min-n", type=int, default=profiler.MIN_N_DEFAULT)
    p.add_argument("--skip", default="", help="쉼표 구분 skip 목록")
    args = p.parse_args()

    print(f"[load] source={args.source}")
    df = io_adapter.load_smart(args.source, validate=False)
    print(
        f"[load] rows={len(df):,}  customers={df['customer_id'].nunique()}  "
        f"cols={list(df.columns)}"
    )
    # public_apt 는 _temp_c 사이드카를 temp_c 로 승격 (프로파일러가 인식하는 이름)
    if "_temp_c" in df.columns and "temp_c" not in df.columns:
        df = df.rename(columns={"_temp_c": "temp_c"})

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    profiler.run(df, args.out, min_n=args.min_n, skip=skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

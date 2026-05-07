"""합성 LP 생성 CLI.

사용:
    python -m scripts.gen_synth --config configs/synth.yaml
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from src import schemas
from src.synth import GeneratorConfig, generate, write_parquet


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="generator YAML 경로")
    p.add_argument("--preview-rows", type=int, default=5, help="미리보기 행 수")
    args = p.parse_args()

    cfg = GeneratorConfig.from_yaml(args.config)
    t0 = time.time()
    df = generate(cfg)
    t_gen = time.time() - t0

    report = schemas.validate(df)
    print(f"[gen] rows={report.n_rows:,}  customers={report.n_customers}  "
          f"range=[{report.ts_min}, {report.ts_max}]  "
          f"contracts={report.contract_type_counts}  ({t_gen:.1f}s)")
    print(f"[gen] optional cols present: {report.present_optional}")
    print(df.head(args.preview_rows).to_string())

    t1 = time.time()
    out = write_parquet(df, cfg)
    t_write = time.time() - t1
    print(f"[write] {out} ({t_write:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

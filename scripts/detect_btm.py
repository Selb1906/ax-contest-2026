"""BTM (Behind-The-Meter) 고객 탐지 — 안심구역 스카우팅용.

데이터안심구역에서 실 LP 데이터에 처음 돌려볼 스크립트.
자가발전(태양광 등) 설치 고객을 식별하고 모델 전략 결정에 활용.

사용법:
  python -m scripts.detect_btm --source configs/source_dsz.yaml
  python -m scripts.detect_btm --source configs/source_synth_v1.yaml  # 로컬 테스트

출력:
  btm_results/
    btm_flags.csv          ← DSZ 내부 전용 (고객별 플래그, 반출 불가)
    btm_summary.json       ← 반출 가능 (집계 통계만)
"""
from __future__ import annotations

import argparse
import io as _stdio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from src import io_adapter
from src.btm_detect import detect

OUT_DIR = Path("btm_results")


def main() -> int:
    parser = argparse.ArgumentParser(description="BTM 고객 탐지")
    parser.add_argument(
        "--source", default="configs/source_dsz.yaml",
        help="데이터 소스 YAML (default: configs/source_dsz.yaml)",
    )
    parser.add_argument(
        "--neg-threshold", type=float, default=0.001,
        help="음수 비율 임계값 (default: 0.001 = 0.1%%)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.source}")
    # strict=False: 음수 값이 있을 수 있으므로 스키마 검증 완화
    df = io_adapter.load_from_yaml(args.source, validate=False)
    print(f"  rows={len(df):,}  customers={df['customer_id'].nunique()}")

    print("[detect] BTM 신호 탐지 중...")
    result = detect(df, neg_threshold=args.neg_threshold)
    flags = result.customer_flags
    summary = result.summary

    # 결과 출력
    n_btm = summary["n_btm_detected"]
    n_total = summary["n_total_customers"]
    rate = summary["btm_rate"] * 100
    print(f"\n{'='*50}")
    print(f"BTM 탐지 결과: {n_btm} / {n_total} 고객 ({rate:.1f}%)")
    print(f"{'='*50}")

    by_sig = summary["by_signal"]
    print(f"  음수 전력 (역송):     {by_sig['negative_power']}명")
    print(f"  낮 시간 골짜기:       {by_sig['daytime_valley']}명")
    print(f"  수준 급변 (설치 추정): {by_sig['level_shift']}명")
    print(f"  여름 역전:            {by_sig['summer_inversion']}명")

    if summary["by_contract"]:
        print("\n계약종별:")
        for ct, info in summary["by_contract"].items():
            print(f"  {ct}: {info['n_btm']}/{info['n']} ({info['btm_rate']*100:.1f}%)")

    # 저장
    flags.to_csv(OUT_DIR / "btm_flags.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "btm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[saved]")
    print(f"  내부용 (반출불가): {OUT_DIR / 'btm_flags.csv'}")
    print(f"  반출용 (집계):     {OUT_DIR / 'btm_summary.json'}")

    if n_btm > 0:
        print(f"\n[action] BTM 고객 {n_btm}명 발견!")
        print("  → 모델 학습 시 btm_flags.csv 의 is_btm 컬럼으로 필터링 또는 별도 그룹 처리")
        print("  → btm_summary.json 은 반출하여 보고서에 활용 가능")
    else:
        print("\n[info] BTM 고객 미발견 — 별도 처리 불필요")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

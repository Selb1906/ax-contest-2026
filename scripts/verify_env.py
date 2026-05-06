"""안심구역 환경 검증 — 패키지 임포트 + 파이프라인 스모크 테스트.

가장 먼저 실행하여 빠진 패키지나 호환 문제를 잡음.
문제가 있으면 즉시 출력하여 시간 낭비 방지.

사용법:
  python -m scripts.verify_env
"""
from __future__ import annotations

import io as _stdio
import sys
import time

sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("matplotlib", "matplotlib"),
    ("optuna", "optuna"),
    ("pyyaml", "yaml"),
    ("openpyxl", "openpyxl"),
    ("tqdm", "tqdm"),
]

OPTIONAL = [
    ("streamlit", "streamlit"),
    ("polars", "polars"),
]


def check_imports():
    print("=" * 60)
    print("  [1/3] 패키지 임포트 검증")
    print("=" * 60)

    ok, fail = [], []
    for name, module in REQUIRED:
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            ok.append((name, ver))
            print(f"  ✅ {name:15s} {ver}")
        except ImportError as e:
            fail.append((name, str(e)))
            print(f"  ❌ {name:15s} MISSING — {e}")

    print()
    for name, module in OPTIONAL:
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            print(f"  ℹ️  {name:15s} {ver} (선택)")
        except ImportError:
            print(f"  ⚠️  {name:15s} 미설치 (선택 — 없어도 진행 가능)")

    return fail


def check_pipeline():
    print("\n" + "=" * 60)
    print("  [2/3] 파이프라인 모듈 임포트 검증")
    print("=" * 60)

    modules = [
        "src.schemas",
        "src.io_adapter",
        "src.preprocess",
        "src.baselines",
        "src.eval",
        "src.eval_detailed",
        "src.btm_detect",
        "src.models.lgbm",
        "src.models.explain",
        "src.profiler",
        "src.checkpoint",
        "src.result_saver",
        "src.tariff",
        "src.tou",
        "src.special_days",
        "src.weather.asos",
        "src.weather.aggregate",
        "src.weather.optimize",
        "src.split",
        "src.feature_selection",
        "src.residual_correction",
    ]

    fail = []
    for mod in modules:
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except Exception as e:
            fail.append((mod, str(e)))
            print(f"  ❌ {mod} — {e}")
    return fail


def check_smoke():
    print("\n" + "=" * 60)
    print("  [3/3] 미니 스모크 테스트 (합성 데이터 10호, 1개월)")
    print("=" * 60)

    try:
        import numpy as np
        import pandas as pd
        from src.schemas import CUSTOMER_ID, TS, P_ACTIVE_KWH, CONTRACT_TYPE
        from src import baselines, eval as ev

        n_cust = 10
        days = 30
        intervals_per_day = 96  # 15분
        rows = []
        for cid in range(n_cust):
            for d in range(days):
                for i in range(intervals_per_day):
                    rows.append({
                        CUSTOMER_ID: f"TEST_{cid:03d}",
                        TS: pd.Timestamp("2024-01-01") + pd.Timedelta(days=d, minutes=15*i),
                        P_ACTIVE_KWH: max(0.1, np.random.normal(2.0, 0.5)),
                        CONTRACT_TYPE: "residential" if cid < 5 else "commercial",
                    })
        df = pd.DataFrame(rows)
        print(f"  합성 데이터 생성: {len(df):,}행, {n_cust}호")

        daily = ev.daily_by_customer(df)
        monthly = ev.monthly_by_customer(daily)
        horizon = ev.build_horizon_table(daily, horizons=(10, 20))
        print(f"  월별 집계: {len(monthly)}행, horizon: {len(horizon)}행")

        ctx = ev.attach_alarm_context(horizon, monthly)
        print(f"  알림 컨텍스트 부착 완료")

        # matplotlib 차트 테스트
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("테스트 차트")
        plt.close(fig)
        print(f"  matplotlib 렌더링 OK")

        # LightGBM 미니 학습
        import lightgbm as lgb
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        ds = lgb.Dataset(X, label=y)
        bst = lgb.train({"objective": "regression", "verbose": -1},
                        ds, num_boost_round=5)
        pred = bst.predict(X[:5])
        print(f"  LightGBM 학습/추론 OK (pred 샘플: {pred[:3].round(2)})")

        print(f"\n  ✅ 스모크 테스트 통과!")
        return []

    except Exception as e:
        print(f"\n  ❌ 스모크 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return [("smoke_test", str(e))]


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║     안심구역 환경 검증                               ║
╚══════════════════════════════════════════════════════╝""")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    start = time.time()

    f1 = check_imports()
    f2 = check_pipeline()
    f3 = check_smoke()

    elapsed = time.time() - start
    all_fail = f1 + f2 + f3

    print("\n" + "=" * 60)
    if not all_fail:
        print(f"  ✅ 전체 검증 통과 ({elapsed:.1f}초)")
        print(f"  → 파이프라인 실행 준비 완료")
    else:
        print(f"  ❌ {len(all_fail)}건 실패:")
        for name, err in all_fail:
            print(f"     - {name}: {err[:80]}")
        print(f"\n  해결: pip install --no-index --find-links=wheels/ <패키지명>")
    print("=" * 60)

    return 1 if all_fail else 0


if __name__ == "__main__":
    sys.exit(main())

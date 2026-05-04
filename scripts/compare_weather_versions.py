"""기상 데이터 버전별 예측 성능 비교 — 안심구역용.

5개 가중평균 버전 + 8개 스테이션 개별 투입을 비교하여
최적 기상 전략을 선택.

사용법:
  python -m scripts.compare_weather_versions --source configs/source_dsz.yaml --train-end 2023-12
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

from src import eval as ev, io_adapter
from src.eval import daily_by_customer, monthly_by_customer, build_horizon_table, attach_alarm_context
from src.schemas import CUSTOMER_ID

WEATHER_DIR = Path("data/weather")


def load_weather_version(name: str) -> pd.DataFrame:
    """가중평균 CSV 로드."""
    p = WEATHER_DIR / f"{name}.csv"
    df = pd.read_csv(p)
    df["year_month"] = pd.Period.astype if "year_month" in df.columns else None
    # year_month를 Period로 변환
    df["year_month"] = df["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
    return df


def attach_weather(horizon: pd.DataFrame, weather: pd.DataFrame,
                   horizon_days_col: str = "horizon_days") -> pd.DataFrame:
    """horizon 테이블에 기상 피처 조인."""
    h = horizon.copy()
    h = h.merge(weather, on="year_month", how="left")
    return h


def train_and_eval(features: pd.DataFrame, train_end: pd.Period,
                   weather_cols: list[str]) -> dict:
    """간단 LightGBM 학습 + 평가."""
    import lightgbm as lgb

    base_cols = [
        "month", "days_in_month", "days_observed",
        "partial_kwh", "partial_rate",
        "prev_month_kwh", "yoy_month_kwh", "ma3_kwh",
        "customer_mean_history", "contract_type",
    ]
    feature_cols = base_cols + weather_cols

    available = [c for c in feature_cols if c in features.columns]
    cat_cols = ["contract_type"] if "contract_type" in available else []

    df = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()
    for c in cat_cols:
        df[c] = df[c].astype("category")

    train_mask = df["year_month"] <= train_end
    test_mask = ~train_mask

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        return {"mape": float("nan"), "n_test": 0}

    X_tr = df.loc[train_mask, available]
    y_tr = df.loc[train_mask, "full_month_kwh"]
    X_te = df.loc[test_mask, available]
    y_te = df.loc[test_mask, "full_month_kwh"]

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    params = {
        "objective": "regression", "metric": "mape",
        "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 40, "verbose": -1,
    }
    booster = lgb.train(
        params, dtr, num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    pred = booster.predict(X_te, num_iteration=booster.best_iteration)
    actual = y_te.values
    mape = np.mean(np.abs(pred - actual) / actual) * 100
    mae = np.mean(np.abs(pred - actual))

    return {"mape": float(mape), "mae": float(mae), "n_test": int(len(y_te))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    args = parser.parse_args()

    print("[load] LP data...")
    df = io_adapter.load_from_yaml(args.source, validate=False)

    daily = daily_by_customer(df)
    monthly = monthly_by_customer(daily)
    horizon = build_horizon_table(daily, horizons=(10, 20))

    # 래그 피처
    mlook = monthly.set_index([CUSTOMER_ID, "year_month"])["monthly_kwh"]
    horizon["month"] = horizon["year_month"].dt.month
    horizon["partial_rate"] = horizon["partial_kwh"] / horizon["days_observed"].replace(0, np.nan)
    horizon["prev_month_kwh"] = [
        mlook.get((c, ym - 1), np.nan)
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]
    horizon["yoy_month_kwh"] = [
        mlook.get((c, ym - 12), np.nan)
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]
    horizon["ma3_kwh"] = [
        np.nanmean([mlook.get((c, ym - k), np.nan) for k in (1, 2, 3)])
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]
    mean_hist = (
        monthly.groupby(CUSTOMER_ID, observed=True)["monthly_kwh"]
        .expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    monthly_mh = monthly.copy()
    monthly_mh["customer_mean_history"] = mean_hist.values
    hist_lookup = monthly_mh.set_index([CUSTOMER_ID, "year_month"])["customer_mean_history"]
    horizon["customer_mean_history"] = [
        hist_lookup.get((c, ym), np.nan)
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]

    train_end = pd.Period(args.train_end, freq="M") if args.train_end else None
    if train_end is None:
        max_ym = horizon["year_month"].max()
        train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")

    # 1) 기상 없이 (baseline)
    print("\n[compare] no weather (baseline)...")
    r_none = train_and_eval(horizon, train_end, [])
    print(f"  MAPE: {r_none['mape']:.3f}%")

    # 2) 가중평균 버전별
    versions = ["national_v1_paper", "national_v2_power", "national_v3_population",
                "national_v4_equal", "national_v5_seoul"]
    results = [{"version": "no_weather", **r_none}]

    for vname in versions:
        try:
            w = load_weather_version(vname)
            w_cols = [c for c in w.columns if c != "year_month"]
            h_with_w = attach_weather(horizon, w)
            r = train_and_eval(h_with_w, train_end, w_cols)
            print(f"  {vname:30s}: MAPE {r['mape']:.3f}%")
            results.append({"version": vname, **r})
        except Exception as e:
            print(f"  {vname:30s}: FAILED ({e})")

    # 3) 8개 스테이션 개별 투입
    try:
        station_ids = [108, 112, 114, 119, 133, 143, 156, 159]
        all_stations = []
        for sid in station_ids:
            sw = pd.read_csv(WEATHER_DIR / f"station_{sid}.csv")
            sw["year_month"] = sw["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
            rename_cols = {c: f"{c}_s{sid}" for c in sw.columns
                          if c not in ("station_id", "year_month")}
            sw = sw.rename(columns=rename_cols)
            sw = sw.drop(columns=["station_id"], errors="ignore")
            all_stations.append(sw)

        h_multi = horizon.copy()
        for sw in all_stations:
            h_multi = h_multi.merge(sw, on="year_month", how="left")

        multi_cols = [c for c in h_multi.columns if any(f"_s{s}" in c for s in station_ids)]
        r_multi = train_and_eval(h_multi, train_end, multi_cols)
        print(f"  {'8_stations_individual':30s}: MAPE {r_multi['mape']:.3f}%")
        results.append({"version": "8_stations_individual", **r_multi})
    except Exception as e:
        print(f"  8_stations_individual: FAILED ({e})")

    # 결과 저장
    result_df = pd.DataFrame(results).sort_values("mape")
    Path("weather_opt").mkdir(exist_ok=True)
    result_df.to_csv("weather_opt/version_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("기상 버전별 MAPE 순위:")
    print("=" * 60)
    for _, row in result_df.iterrows():
        marker = " <-- best" if row.name == result_df.index[0] else ""
        print(f"  {row['version']:30s}: {row['mape']:.3f}%{marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

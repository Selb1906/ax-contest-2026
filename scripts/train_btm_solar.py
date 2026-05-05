"""BTM 태양광 발전량 예측 모델 학습 — 기상→월별 BTM 출력.

KPX 자가용 태양광 출력 추계(시간별) + ASOS 기상 → 월별 BTM 발전량 예측.
운영 시: 기상 예보/평년값 → BTM 예측 → 메인 모델에 투입.

사용법:
  python -m scripts.train_btm_solar
"""
from __future__ import annotations

import io as _stdio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.weather.asos import read_asos_directory, monthly_weather_features, EIGHT_CITY_WEIGHTS

OUT_DIR = Path("btm_solar_model")


def load_btm_data() -> pd.DataFrame:
    """KPX BTM 데이터 로드 → 월별 집계."""
    df = pd.read_csv("kpx_BTM.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    # 01~24시 → 00~23시 보정
    if df["datetime"].dt.hour.min() >= 1:
        df["datetime"] = df["datetime"] - pd.Timedelta(hours=1)
    df["year_month"] = df["datetime"].dt.to_period("M")

    # BTM = total - market = kepco_ppa + self_use
    df["btm_solar_mw"] = df["total_solar_mw"] - df["market_solar_mw"]

    # 월별 집계: MWh (시간별 MW → 합산 = MWh)
    monthly = df.groupby("year_month").agg(
        btm_mwh=("btm_solar_mw", "sum"),
        btm_peak_mw=("btm_solar_mw", "max"),
        btm_mean_mw=("btm_solar_mw", "mean"),
        total_solar_mwh=("total_solar_mw", "sum"),
        market_mwh=("market_solar_mw", "sum"),
        n_hours=("btm_solar_mw", "count"),
    ).reset_index()

    # BTM 비율
    monthly["btm_ratio"] = monthly["btm_mwh"] / monthly["total_solar_mwh"].clip(lower=1)

    return monthly


def build_features(btm_monthly: pd.DataFrame, weather_monthly: pd.DataFrame) -> pd.DataFrame:
    """BTM 월별 + 기상 월별 → 학습용 피처."""
    # 기상 가중평균
    from src.weather.asos import weighted_national_average
    weather_avg = weighted_national_average(weather_monthly, EIGHT_CITY_WEIGHTS)

    merged = btm_monthly.merge(weather_avg, on="year_month", how="inner")
    merged["month"] = merged["year_month"].apply(lambda x: x.month)
    merged["year"] = merged["year_month"].apply(lambda x: x.year)

    # 태양광 설치량 트렌드 (연도가 지남에 따라 BTM 증가)
    merged["months_since_start"] = range(len(merged))

    # 일조 가능 시간 근사
    merged["daylight_hours"] = merged["month"].map({
        1: 9.8, 2: 10.8, 3: 12.0, 4: 13.2, 5: 14.2, 6: 14.8,
        7: 14.5, 8: 13.6, 9: 12.4, 10: 11.2, 11: 10.1, 12: 9.5,
    })

    # 태양광 효율 온도 보정 (25°C 이상에서 효율 저하)
    if "temp_mean" in merged.columns:
        merged["temp_loss"] = 1 - 0.004 * np.maximum(merged["temp_mean"] - 25, 0)

    return merged


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] BTM 데이터...")
    btm = load_btm_data()
    print(f"  기간: {btm['year_month'].min()} ~ {btm['year_month'].max()}")
    print(f"  행: {len(btm)}")

    print("\n[load] ASOS 기상 데이터...")
    asos = read_asos_directory("ASOS")
    weather = monthly_weather_features(asos)
    print(f"  관측소: {sorted(weather['station_id'].unique())}")

    print("\n[features] 피처 빌드...")
    features = build_features(btm, weather)
    print(f"  행: {len(features)}, 컬럼: {features.columns.tolist()}")

    # 학습/테스트 분할 (마지막 12개월 test)
    target_col = "btm_mwh"
    feature_cols = [c for c in features.columns
                    if c not in ("year_month", target_col, "btm_peak_mw", "btm_mean_mw",
                                 "total_solar_mwh", "market_mwh", "n_hours", "btm_ratio")]

    # 2026-01은 0값이므로 제외, 마지막 완전 연도를 test로
    features = features[features[target_col] > 0]
    max_ym = features["year_month"].max()
    train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")

    train = features[features["year_month"] <= train_end]
    test = features[features["year_month"] > train_end]

    print(f"\n[split] train: {len(train)} ({train['year_month'].min()}~{train['year_month'].max()})")
    print(f"  test:  {len(test)} ({test['year_month'].min()}~{test['year_month'].max()})")

    X_tr, y_tr = train[feature_cols], train[target_col]
    X_te, y_te = test[feature_cols], test[target_col]

    # LightGBM 학습
    print(f"\n[train] features: {feature_cols}")
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dte = lgb.Dataset(X_te, label=y_te, reference=dtr)

    params = {
        "objective": "regression",
        "metric": "mape",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 3,
        "verbose": -1,
    }
    booster = lgb.train(
        params, dtr, num_boost_round=300,
        valid_sets=[dtr, dte], valid_names=["train", "test"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    # 평가
    pred_test = booster.predict(X_te, num_iteration=booster.best_iteration)
    mape = np.mean(np.abs(pred_test - y_te.values) / y_te.values) * 100
    rmse = np.sqrt(np.mean((pred_test - y_te.values) ** 2))

    print(f"\n[result]")
    print(f"  Test MAPE: {mape:.2f}%")
    print(f"  Test RMSE: {rmse:.0f} MWh")

    # 월별 비교
    print(f"\n  월별 비교:")
    for i, (_, row) in enumerate(test.iterrows()):
        actual = row[target_col]
        pred = pred_test[i]
        err = abs(pred - actual) / actual * 100
        print(f"    {row['year_month']}: 실측 {actual:,.0f}  예측 {pred:,.0f}  오차 {err:.1f}%")

    # 피처 중요도
    importance = booster.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    print(f"\n  피처 중요도:")
    for _, row in imp_df.iterrows():
        print(f"    {row['feature']:25s}: {row['importance']:.0f}")

    # 저장
    booster.save_model(str(OUT_DIR / "btm_solar_model.txt"))
    meta = {
        "feature_cols": feature_cols,
        "target": target_col,
        "train_end": str(train_end),
        "test_mape": round(mape, 2),
        "test_rmse": round(rmse, 0),
        "best_iteration": booster.best_iteration,
    }
    with open(OUT_DIR / "btm_solar_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 전체 기간 예측 (반출용 — 개인정보 없음, 전국 집계)
    pred_all = booster.predict(features[feature_cols], num_iteration=booster.best_iteration)
    result_df = features[["year_month"]].copy()
    result_df["actual_btm_mwh"] = features[target_col].values
    result_df["predicted_btm_mwh"] = pred_all
    result_df["error_pct"] = abs(pred_all - features[target_col].values) / features[target_col].clip(lower=1).values * 100
    result_df.to_csv(OUT_DIR / "btm_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"\n[saved] {OUT_DIR}/")
    print(f"  btm_solar_model.txt — 모델 가중치")
    print(f"  btm_solar_meta.json — 메타 정보")
    print(f"  btm_predictions.csv — 전체 기간 예측 vs 실측")
    print(f"  feature_importance.csv — 피처 중요도")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

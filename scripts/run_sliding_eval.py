"""슬라이딩 평가 독립 실행 — LightGBM(base/tuned) + partial_linear.

Usage:
    python -m scripts.run_sliding_eval --source configs/source_dsz.yaml
    python -m scripts.run_sliding_eval --source configs/source_dsz.yaml --meter-days 1,15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
import os; os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.schemas import CUSTOMER_ID
from src.eval import (
    build_horizon_table, daily_by_customer, monthly_by_customer,
    attach_alarm_context,
)
from src.eval_detailed import save_detailed_evaluation
from src.models import lgbm


def _sliding_eval(name, predict_fn, daily, monthly, spec, train_end, meter_days, horizons=(10, 20), suffix=""):
    print(f"\n{'='*50}")
    print(f"[{name}] Sliding 평가 시작")
    print(f"{'='*50}")
    results = []
    for md in meter_days:
        try:
            h = build_horizon_table(daily, horizons=horizons, meter_day=md)
            ctx = attach_alarm_context(h, monthly)
            test = ctx[ctx["year_month"] > train_end].copy()
            if len(test) == 0:
                continue
            test["pred_monthly_kwh"] = predict_fn(test)
            test["meter_day"] = md
            test["error"] = test["pred_monthly_kwh"] - test["full_month_kwh"]
            test["abs_error"] = test["error"].abs()
            test["pct_error"] = test["abs_error"] / test["full_month_kwh"].clip(lower=1e-9) * 100
            test["month"] = test["year_month"].apply(lambda x: x.month)
            results.append(test)
            print(f"  md={md}: {len(test)}건", flush=True)
        except Exception as e:
            print(f"  md={md}: 실패 ({e})", flush=True)

    if results:
        all_df = pd.concat(results, ignore_index=True)
        out_dir = f"sliding_results_{name}{suffix}"
        dims = save_detailed_evaluation(all_df, out_dir)
        print(f"  → {out_dir}/ 저장 완료 ({len(all_df)}건)")
        return dims
    print(f"  → 결과 없음")
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--meter-days", default="1,5,10,15,20,25")
    parser.add_argument("--suffix", default="", help="출력 폴더 접미사 (예: _2d, _6d, _12d)")
    parser.add_argument("--models", default="all", help="평가할 모델 (all 또는 콤마구분: partial_linear,lgbm_corrected)")
    args = parser.parse_args()

    meter_days = [int(x) for x in args.meter_days.split(",")]
    run_models = args.models.split(",") if args.models != "all" else ["partial_linear", "lgbm_base", "lgbm_tuned", "lgbm_corrected"]

    # 필수 파일 검증
    if not Path(args.source).exists():
        print(f"[ERROR] source 파일 없음: {args.source}"); return 1

    # 데이터 로드
    print("[1/4] 데이터 로드...", flush=True)
    from src.io_adapter import load_smart
    df = load_smart(args.source)
    daily = daily_by_customer(df)
    monthly = monthly_by_customer(daily)

    if args.train_end:
        train_end = pd.Period(args.train_end, freq="M")
    else:
        max_ym = monthly["year_month"].max()
        train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")
    print(f"  train_end={train_end}, meter_days={meter_days}")

    # features 로드/빌드
    print("[2/4] 피처 로드...", flush=True)
    fc = Path("checkpoints/features_cache.parquet")
    if fc.exists():
        features = pd.read_parquet(fc)
        features["year_month"] = features["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
        _, spec = lgbm.build_features(df)
        print(f"  features_cache 로드: {len(features)}행, spec: {len(spec.all)}개")
    else:
        features, spec = lgbm.build_features(df)
        print(f"  build_features: {len(features)}행, spec: {len(spec.all)}개")

    # AMI 피처 조인 (preprocess_lp에서 생성된 것)
    ami_path = Path("data/preprocessed/ami_features.parquet")
    if ami_path.exists():
        ami_df = pd.read_parquet(ami_path)
        ami_df["year_month_str"] = ami_df["year_month"].astype(str)
        features["year_month_str"] = features["year_month"].astype(str)
        ami_new = [c for c in ami_df.columns if c not in features.columns and c not in ("year_month_str",)]
        if ami_new:
            features = features.merge(
                ami_df[["customer_id", "year_month_str"] + ami_new],
                on=["customer_id", "year_month_str"], how="left")
            spec.extra_numeric.extend([c for c in ami_new if c in features.columns])
            print(f"  [AMI] {len(ami_new)}개 피처 조인 완료")
        features = features.drop(columns=["year_month_str"], errors="ignore")
    else:
        print("  [AMI] ami_features.parquet 없음 — AMI 피처 없이 진행")

    # ablation 최적 config 적용 (TMY remainder + 피처 제거)
    from src.utils import load_best_config, apply_tmy_remainder
    _best_cfg = load_best_config()
    if _best_cfg:
        _wv = _best_cfg.get("best_per_factor", {}).get("weather_version", "regional")
        features = apply_tmy_remainder(features, _wv)
        _exclude = _best_cfg.get("exclude_features", [])
        if _exclude:
            _excluded = [c for c in _exclude if c in spec.extra_numeric]
            spec.extra_numeric = [c for c in spec.extra_numeric if c not in _exclude]
            if _excluded:
                print(f"  [피처 제거] {_excluded}")

    test_features = features[features["year_month"] > train_end].copy()
    for c in spec.categorical:
        if c in test_features.columns:
            test_features[c] = test_features[c].astype("category")
    nan_filled = []
    for c in spec.all:
        if c not in test_features.columns:
            test_features[c] = np.nan
            nan_filled.append(c)
    if nan_filled:
        print(f"  [DEBUG] NaN 채운 피처 {len(nan_filled)}개: {nan_filled}", flush=True)
    _debug_cols = ["hdd_remainder", "cdd_remainder", "hdd_observed", "observed_power_factor"]
    for c in _debug_cols:
        if c in test_features.columns:
            nna = test_features[c].notna().sum()
            print(f"  [DEBUG] {c}: {nna}/{len(test_features)} non-null", flush=True)
    print(f"  [DEBUG] test_features: {len(test_features)}행, spec: {len(spec.all)}개", flush=True)

    # 1) partial_linear
    if "partial_linear" in run_models:
        print("\n[3/4] partial_linear 평가...", flush=True)
        def _pl_predict(ctx):
            return ctx["partial_kwh"] / ctx["days_observed"].clip(lower=1) * ctx["days_in_month"]
        _sliding_eval("partial_linear", _pl_predict, daily, monthly, spec, train_end, meter_days, suffix=args.suffix)

    # 2) LightGBM base
    base_path = Path("weights/dsz_lgbm/model.txt")
    if "lgbm_base" in run_models and base_path.exists():
        print("\n[3/4] LightGBM base 평가...", flush=True)
        booster_base, spec_base = lgbm.load("weights/dsz_lgbm")
        def _lgbm_base_predict(ctx):
            keys = ctx[[CUSTOMER_ID, "year_month", "horizon_days"]]
            matched = test_features.merge(keys, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="inner")
            if len(matched) == 0:
                return pd.Series(np.nan, index=ctx.index)
            for c in spec_base.all:
                if c not in matched.columns:
                    matched[c] = np.nan
            for c in spec_base.categorical:
                if c in matched.columns:
                    matched[c] = matched[c].astype("category")
            residual = booster_base.predict(matched[spec_base.all], num_iteration=booster_base.best_iteration)
            pl_base = matched["partial_kwh"].values / matched["days_observed"].clip(lower=1).values * matched["days_in_month"].values
            preds = pl_base + residual
            if "partial_kwh" in matched.columns:
                preds = np.maximum(preds, matched["partial_kwh"].values)
            result = ctx[[CUSTOMER_ID, "year_month", "horizon_days"]].merge(
                matched[[CUSTOMER_ID, "year_month", "horizon_days"]].assign(pred=preds),
                on=[CUSTOMER_ID, "year_month", "horizon_days"], how="left"
            )
            return result["pred"].values
        _sliding_eval("lgbm_base", _lgbm_base_predict, daily, monthly, spec_base, train_end, meter_days, suffix=args.suffix)
    else:
        print("  [SKIP] weights/dsz_lgbm/ 없음")

    # 3) LightGBM tuned
    tuned_path = Path("weights/dsz_lgbm_tuned/model.txt")
    if ("lgbm_tuned" in run_models or "lgbm_corrected" in run_models) and tuned_path.exists():
        print("\n[4/4] LightGBM tuned 평가...", flush=True)
        booster_tuned, spec_tuned = lgbm.load("weights/dsz_lgbm_tuned")
        def _lgbm_tuned_predict(ctx):
            keys = ctx[[CUSTOMER_ID, "year_month", "horizon_days"]]
            matched = test_features.merge(keys, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="inner")
            if len(matched) == 0:
                return pd.Series(np.nan, index=ctx.index)
            for c in spec_tuned.all:
                if c not in matched.columns:
                    matched[c] = np.nan
            for c in spec_tuned.categorical:
                if c in matched.columns:
                    matched[c] = matched[c].astype("category")
            residual = booster_tuned.predict(matched[spec_tuned.all], num_iteration=booster_tuned.best_iteration)
            pl_base = matched["partial_kwh"].values / matched["days_observed"].clip(lower=1).values * matched["days_in_month"].values
            preds = pl_base + residual
            if "partial_kwh" in matched.columns:
                preds = np.maximum(preds, matched["partial_kwh"].values)
            result = ctx[[CUSTOMER_ID, "year_month", "horizon_days"]].merge(
                matched[[CUSTOMER_ID, "year_month", "horizon_days"]].assign(pred=preds),
                on=[CUSTOMER_ID, "year_month", "horizon_days"], how="left"
            )
            return result["pred"].values
        if "lgbm_tuned" in run_models:
            _sliding_eval("lgbm_tuned", _lgbm_tuned_predict, daily, monthly, spec_tuned, train_end, meter_days, suffix=args.suffix)
    else:
        print("  [SKIP] weights/dsz_lgbm_tuned/ 없음")

    # 4) LightGBM tuned + 잔차 보정
    if "lgbm_corrected" in run_models and tuned_path.exists():
        print("\n[5/5] LightGBM tuned + 잔차 보정 평가...", flush=True)
        try:
            from src.residual_correction import build_residual_features, train_residual_model, predict_residual

            # train 구간에서 Ridge 학습
            train_features = features[features["year_month"] <= train_end].copy()
            for c in spec_tuned.categorical:
                if c in train_features.columns:
                    train_features[c] = train_features[c].astype("category")
            for c in spec_tuned.all:
                if c not in train_features.columns:
                    train_features[c] = np.nan

            residual_train = booster_tuned.predict(
                train_features[spec_tuned.all], num_iteration=booster_tuned.best_iteration)
            pl_train = (train_features["partial_kwh"].values
                        / train_features["days_observed"].clip(lower=1).values
                        * train_features["days_in_month"].values)
            pred_train = pl_train + residual_train
            actual_train = train_features["full_month_kwh"].values
            res_feat_train, residuals_train = build_residual_features(train_features, pred_train, actual_train)
            ridge_model, scaler, feature_cols = train_residual_model(res_feat_train, residuals_train)
            print(f"  Ridge 학습 완료 (train {len(train_features)}건)")

            def _lgbm_corrected_predict(ctx):
                keys = ctx[[CUSTOMER_ID, "year_month", "horizon_days"]]
                matched = test_features.merge(keys, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="inner")
                if len(matched) == 0:
                    return pd.Series(np.nan, index=ctx.index)
                for c in spec_tuned.all:
                    if c not in matched.columns:
                        matched[c] = np.nan
                for c in spec_tuned.categorical:
                    if c in matched.columns:
                        matched[c] = matched[c].astype("category")
                residual = booster_tuned.predict(matched[spec_tuned.all], num_iteration=booster_tuned.best_iteration)
                pl_base = matched["partial_kwh"].values / matched["days_observed"].clip(lower=1).values * matched["days_in_month"].values
                preds = pl_base + residual
                res_feat, _ = build_residual_features(matched, preds, np.zeros_like(preds))
                correction = predict_residual(ridge_model, scaler, feature_cols, res_feat)
                corrected = preds + correction
                if "partial_kwh" in matched.columns:
                    corrected = np.maximum(corrected, matched["partial_kwh"].values)
                result = ctx[[CUSTOMER_ID, "year_month", "horizon_days"]].merge(
                    matched[[CUSTOMER_ID, "year_month", "horizon_days"]].assign(pred=corrected),
                    on=[CUSTOMER_ID, "year_month", "horizon_days"], how="left"
                )
                return result["pred"].values
            _sliding_eval("lgbm_corrected", _lgbm_corrected_predict, daily, monthly, spec_tuned, train_end, meter_days, suffix=args.suffix)
        except Exception as e:
            print(f"  [잔차 보정 실패] {e}")

    print("\n[완료] 결과는 sliding_results_*/ 폴더에 저장됨")
    return 0


if __name__ == "__main__":
    sys.exit(main())

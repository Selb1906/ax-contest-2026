"""LightGBM 하이퍼파라미터 튜닝 — Optuna 기반.

최적 모델(BTM 전략 + 피처 선택 확정 후)에 대해 실행.
중단해도 그 시점까지의 최적 파라미터를 사용 가능.

사용법:
  python -m scripts.tune_lgbm --source configs/source_dsz.yaml --train-end 2023-12 --timeout 600
  python -m scripts.tune_lgbm --source configs/source_dsz.yaml --train-end 2023-12 --n-trials 50

출력:
  tuning_results/
    best_params.json     — 최적 파라미터
    study_history.csv    — 전체 탐색 이력
    importance.csv       — 파라미터 중요도
"""
from __future__ import annotations

import argparse
import io as _stdio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import io_adapter
from src.preprocess import preprocess
from src.models.lgbm import build_features, FeatureSpec

OUT_DIR = Path("tuning_results")


def objective(trial, X_tr, y_tr, X_val, y_val, spec):
    """Optuna objective — MAPE 최소화."""
    params = {
        "objective": "regression",
        "metric": "mape",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
    }

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=spec.categorical)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=spec.categorical)

    booster = lgb.train(
        params, dtr,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    actual = y_val.values
    mape = np.mean(np.abs(pred - actual) / actual) * 100

    # best_iteration 저장
    trial.set_user_attr("best_iteration", booster.best_iteration)
    trial.set_user_attr("num_trees", booster.num_trees())

    # 진행 상황 출력
    n = trial.number + 1
    best_so_far = min(mape, trial.study.best_value if trial.study.best_trial else mape)
    print(f"  #{n:>3d}  MAPE {mape:.3f}%  trees {booster.best_iteration}  best {best_so_far:.3f}%")

    return mape


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGBM 하이퍼파라미터 튜닝")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--timeout", type=int, default=600, help="최대 시간 (초, 기본 600=10분)")
    parser.add_argument("--n-trials", type=int, default=100, help="최대 시도 횟수")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 준비
    print("[load]")
    df = io_adapter.load_from_yaml(args.source, validate=False)
    df, _ = preprocess(df)

    print("[features]")
    features, spec = build_features(df)
    features = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()
    for c in spec.categorical:
        features[c] = features[c].astype("category")

    train_end = pd.Period(args.train_end, freq="M") if args.train_end else None
    if train_end is None:
        max_ym = features["year_month"].max()
        train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")

    train_mask = features["year_month"] <= train_end
    X_tr = features.loc[train_mask, spec.all]
    y_tr = features.loc[train_mask, "full_month_kwh"]
    X_val = features.loc[~train_mask, spec.all]
    y_val = features.loc[~train_mask, "full_month_kwh"]

    print(f"  train={len(X_tr):,}  val={len(X_val):,}  features={len(spec.all)}")

    # 기본값 MAPE 측정
    print("\n[baseline]")
    default_params = {
        "objective": "regression", "metric": "mape",
        "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 40, "verbose": -1,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=spec.categorical)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=spec.categorical)
    base_booster = lgb.train(
        default_params, dtr, num_boost_round=400,
        valid_sets=[dval], valid_names=["valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    base_pred = base_booster.predict(X_val, num_iteration=base_booster.best_iteration)
    base_mape = np.mean(np.abs(base_pred - y_val.values) / y_val.values) * 100
    print(f"  기본값 MAPE: {base_mape:.4f}%")

    # Optuna 튜닝
    print(f"\n[tune] timeout={args.timeout}s, max_trials={args.n_trials}")
    print(f"  중단(Ctrl+C)해도 그 시점 최적값 저장됨\n")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize", study_name="lgbm_tune")
    t0 = time.time()

    try:
        study.optimize(
            lambda trial: objective(trial, X_tr, y_tr, X_val, y_val, spec),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n  [중단됨] 현재까지 최적값 저장 중...")

    elapsed = time.time() - t0
    n_completed = len(study.trials)
    best = study.best_trial

    print(f"\n{'='*60}")
    print(f"튜닝 완료: {n_completed}회 시도, {elapsed:.0f}초")
    print(f"{'='*60}")
    print(f"  기본값 MAPE:  {base_mape:.4f}%")
    print(f"  최적값 MAPE:  {best.value:.4f}%")
    print(f"  개선:         {base_mape - best.value:.4f}% ({(base_mape - best.value)/base_mape*100:.1f}%)")
    print(f"\n  최적 파라미터:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # 저장
    result = {
        "baseline_mape": round(base_mape, 4),
        "best_mape": round(best.value, 4),
        "improvement_pct": round((base_mape - best.value) / base_mape * 100, 2),
        "n_trials": n_completed,
        "elapsed_sec": round(elapsed, 1),
        "best_params": best.params,
        "best_iteration": best.user_attrs.get("best_iteration"),
    }
    with open(OUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 탐색 이력
    history = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "mape": t.value}
            row.update(t.params)
            history.append(row)
    pd.DataFrame(history).to_csv(OUT_DIR / "study_history.csv", index=False, encoding="utf-8-sig")

    # 파라미터 중요도
    try:
        importance = optuna.importance.get_param_importances(study)
        imp_df = pd.DataFrame([{"param": k, "importance": v} for k, v in importance.items()])
        imp_df.to_csv(OUT_DIR / "param_importance.csv", index=False, encoding="utf-8-sig")
        print(f"\n  파라미터 중요도:")
        for k, v in importance.items():
            print(f"    {k}: {v:.4f}")
    except Exception:
        pass

    # 최적 파라미터로 최종 모델 학습 + 저장
    print(f"\n[retrain] 최적 파라미터로 최종 모델 학습...")
    final_params = {**default_params, **best.params}
    final_booster = lgb.train(
        final_params, dtr,
        num_boost_round=500,
        valid_sets=[dval], valid_names=["valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    from src.models.lgbm import save
    save(final_booster, spec, "weights/dsz_lgbm_tuned",
         notes=f"tuned: {n_completed} trials, MAPE {best.value:.4f}%")
    print(f"  저장: weights/dsz_lgbm_tuned/")

    print(f"\n[done] {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

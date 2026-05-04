"""SHAP 기반 모델 해석 — 피처 기여도 분석 + 개별 고객 설명.

산출물:
  - shap_global.csv: 피처별 mean(|SHAP|) 순위 (반출 안전)
  - shap_summary.png: beeswarm plot (반출 안전)
  - shap_dependence_*.png: 주요 피처 dependence plot (반출 안전)
  - shap_customer_examples/: 개별 고객 waterfall (반출 안전 — 고객ID 비식별)
  - feature_correlation.csv: 피처 간 상관행렬 (반출 안전)
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..schemas import CUSTOMER_ID


def compute_shap_values(
    booster: lgb.Booster,
    X: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """TreeExplainer 로 SHAP values 산출."""
    import shap
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X[feature_names])
    return shap_values


def shap_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """피처별 mean(|SHAP|) 테이블."""
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["pct_of_total"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum() * 100
    return df


def shap_summary_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Beeswarm summary plot 저장."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(
        shap_values,
        X[feature_names],
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def shap_dependence_plots(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    out_dir: Path,
    top_n: int = 5,
) -> list[str]:
    """상위 N개 피처의 dependence plot 저장."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(importance)[-top_n:][::-1]
    saved = []

    for idx in top_indices:
        fname = feature_names[idx]
        shap.dependence_plot(
            idx,
            shap_values,
            X[feature_names],
            feature_names=feature_names,
            show=False,
        )
        p = out_dir / f"shap_dep_{fname}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(str(p))
    return saved


def shap_waterfall_examples(
    booster: lgb.Booster,
    X: pd.DataFrame,
    feature_names: list[str],
    labels: pd.DataFrame,
    out_dir: Path,
    n_examples: int = 5,
) -> list[str]:
    """알림 대상 고객 중 SHAP waterfall 예시 저장.

    labels 에 is_alarm 컬럼이 있으면 알림 대상 위주로 선택.
    고객ID는 익명화(idx 기반).
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(booster)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if "is_alarm" in labels.columns:
        alarm_idx = labels[labels["is_alarm"]].index.tolist()
        if len(alarm_idx) > n_examples:
            alarm_idx = np.random.choice(alarm_idx, n_examples, replace=False).tolist()
    else:
        alarm_idx = np.random.choice(len(X), min(n_examples, len(X)), replace=False).tolist()

    for i, idx in enumerate(alarm_idx):
        row = X[feature_names].iloc[[idx]]
        sv = explainer(row)
        shap.waterfall_plot(sv[0], show=False, max_display=12)
        p = out_dir / f"waterfall_example_{i+1}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(str(p))
    return saved


def feature_correlation_matrix(
    X: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """피처 간 Pearson 상관행렬."""
    numeric_cols = [c for c in feature_names if X[c].dtype != "category"]
    return X[numeric_cols].corr()


def correlation_heatmap(
    corr: pd.DataFrame,
    out_path: Path,
) -> None:
    """상관행렬 히트맵 저장."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            matplotlib.rcParams["font.family"] = f
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_full_explanation(
    booster: lgb.Booster,
    features: pd.DataFrame,
    spec,
    out_dir: str | Path,
    *,
    alarm_labels: pd.DataFrame | None = None,
) -> dict:
    """전체 SHAP 분석 실행. 반출 안전 산출물 생성."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = spec.all
    X = features.copy()
    for c in spec.categorical:
        X[c] = X[c].astype("category")

    print("[shap] computing SHAP values...")
    shap_values = compute_shap_values(booster, X, feature_names)

    # 1. Global importance
    global_imp = shap_global_importance(shap_values, feature_names)
    global_imp.to_csv(out_dir / "shap_global.csv", index=False, encoding="utf-8-sig")
    print(f"  shap_global.csv — top 5: {global_imp['feature'].head().tolist()}")

    # 2. Summary plot
    shap_summary_plot(shap_values, X, feature_names, out_dir / "shap_summary.png")
    print("  shap_summary.png")

    # 3. Dependence plots
    dep_paths = shap_dependence_plots(shap_values, X, feature_names, out_dir)
    print(f"  dependence plots: {len(dep_paths)}")

    # 4. Waterfall examples
    if alarm_labels is not None:
        wf_paths = shap_waterfall_examples(
            booster, X, feature_names, alarm_labels, out_dir / "waterfall"
        )
        print(f"  waterfall examples: {len(wf_paths)}")
    else:
        wf_paths = []

    # 5. Feature correlation
    corr = feature_correlation_matrix(X, feature_names)
    corr.to_csv(out_dir / "feature_correlation.csv", encoding="utf-8-sig")
    correlation_heatmap(corr, out_dir / "correlation_heatmap.png")
    print("  feature_correlation.csv + correlation_heatmap.png")

    return {
        "global_importance": str(out_dir / "shap_global.csv"),
        "summary_plot": str(out_dir / "shap_summary.png"),
        "dependence_plots": dep_paths,
        "waterfall_examples": wf_paths,
        "correlation_matrix": str(out_dir / "feature_correlation.csv"),
        "correlation_heatmap": str(out_dir / "correlation_heatmap.png"),
    }

"""분석 보고서용 시각화 자동 생성.

파이프라인 결과 파일을 읽어서 고품질 PNG를 figures/ 디렉터리에 저장한다.
데이터안심구역에서 한 번만 실행하므로 가능한 모든 차트를 일괄 생성.

사용법:
  python -m scripts.generate_figures
"""
from __future__ import annotations

import io as _stdio
import json
import sys
from pathlib import Path

# -- 출력 버퍼링 방지 + 프로젝트 루트 sys.path --
sys.stdout = _stdio.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", line_buffering=True
)
import os
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────
# 공통 설정
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DPI = 300
FACECOLOR = "white"

# 색상 팔레트 (보고서 친화)
COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#059669",
    "warm": "#D97706",
    "purple": "#7C3AED",
    "gray": "#6B7280",
    "horizon_10": "#2563EB",
    "horizon_20": "#DC2626",
}
SEASON_COLORS = {
    "Spring": "#22C55E",
    "Summer": "#EF4444",
    "Fall": "#F59E0B",
    "Winter": "#3B82F6",
}


def _month_to_season(m: int) -> str:
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    if m in (9, 10, 11):
        return "Fall"
    return "Winter"


def _savefig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
    plt.close(fig)
    print(f"    -> {path.relative_to(ROOT)}")


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


# =============================================================
# 1. 모델별 성능 비교 막대 차트
# =============================================================
def fig01_model_comparison():
    """eval_results/metrics.csv에서 베이스라인 + 제안모델의 MAE/MAPE grouped bar."""
    print("  [1/10] 모델별 성능 비교 막대 차트")

    # 여러 후보 경로 시도
    df = None
    for p in [ROOT / "eval_results" / "lgbm_vs_baselines.csv",
              ROOT / "eval_results" / "metrics.csv"]:
        df = _safe_read_csv(p)
        if df is not None:
            break
    if df is None:
        print("    SKIP: 메트릭 CSV 없음")
        return

    if "model" not in df.columns or "mape_pct" not in df.columns:
        print("    SKIP: 필수 컬럼 없음")
        return

    # horizon별 평균 (contract_type 통합)
    group_cols = ["model"]
    if "horizon_days" in df.columns:
        group_cols.append("horizon_days")

    agg = df.groupby(group_cols, observed=True).agg(
        mape_pct=("mape_pct", "mean"),
        mae=("mae", "mean") if "mae" in df.columns else ("mape_pct", "mean"),
    ).reset_index()

    if "horizon_days" not in agg.columns:
        agg["horizon_days"] = 10  # fallback

    models = agg["model"].unique()
    horizons = sorted(agg["horizon_days"].unique())
    n_models = len(models)
    n_horizons = len(horizons)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=FACECOLOR)
    bar_width = 0.35
    x = np.arange(n_models)

    for ax_idx, (metric, ylabel) in enumerate([("mape_pct", "MAPE (%)"), ("mae", "MAE (kWh)")]):
        ax = axes[ax_idx]
        if metric not in agg.columns:
            ax.set_visible(False)
            continue
        for i, h in enumerate(horizons):
            subset = agg[agg["horizon_days"] == h]
            vals = []
            for m in models:
                row = subset[subset["model"] == m]
                vals.append(float(row[metric].iloc[0]) if len(row) > 0 else 0)
            offset = (i - (n_horizons - 1) / 2) * bar_width
            color = COLORS["horizon_10"] if h == 10 else COLORS["horizon_20"]
            bars = ax.bar(x + offset, vals, bar_width * 0.9, label=f"+{h}d",
                          color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "fig01_model_comparison")


# =============================================================
# 2. 예측 vs 실측 산점도
# =============================================================
def fig02_pred_vs_actual():
    """checkpoints/lgbm_preds.csv + data/ui/ctx.parquet => scatter."""
    print("  [2/10] 예측 vs 실측 산점도")

    preds = _safe_read_parquet(ROOT / "data" / "ui" / "preds.parquet")
    if preds is None:
        preds = _safe_read_csv(ROOT / "checkpoints" / "lgbm_preds.csv")
    if preds is None:
        preds = _safe_read_parquet(ROOT / "checkpoints" / "lgbm_preds.parquet")
    ctx = _safe_read_parquet(ROOT / "data" / "ui" / "ctx.parquet")

    if preds is None or ctx is None:
        print("    SKIP: 예측/실측 데이터 없음")
        return

    # year_month 타입 통일
    for df in [preds, ctx]:
        if "year_month" in df.columns:
            df["year_month"] = df["year_month"].astype(str)

    merge_cols = ["customer_id", "year_month", "horizon_days"]
    available = [c for c in merge_cols if c in preds.columns and c in ctx.columns]
    if len(available) < 2:
        print("    SKIP: 조인 컬럼 부족")
        return

    merged = preds.merge(ctx, on=available, how="inner")
    if "pred_monthly_kwh" not in merged.columns or "full_month_kwh" not in merged.columns:
        print("    SKIP: pred_monthly_kwh 또는 full_month_kwh 없음")
        return

    actual = merged["full_month_kwh"].values
    predicted = merged["pred_monthly_kwh"].values
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    actual, predicted = actual[mask], predicted[mask]

    if len(actual) == 0:
        print("    SKIP: 유효 데이터 0건")
        return

    # R^2 계산
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=FACECOLOR)

    if "horizon_days" in merged.columns:
        horizons = sorted(merged.loc[mask, "horizon_days"].unique()) if "horizon_days" in merged.columns else [0]
        h_vals = merged.loc[mask, "horizon_days"].values if "horizon_days" in merged.columns else np.zeros(len(actual))
        for h in horizons:
            hmask = h_vals == h
            color = COLORS["horizon_10"] if h == 10 else COLORS["horizon_20"]
            ax.scatter(actual[hmask], predicted[hmask], s=8, alpha=0.3,
                       color=color, label=f"+{h}d", rasterized=True)
    else:
        ax.scatter(actual, predicted, s=8, alpha=0.3, color=COLORS["primary"], rasterized=True)

    # y=x 대각선
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1, alpha=0.7, label="y = x")

    ax.set_xlabel("Actual Monthly Usage (kWh)", fontsize=11)
    ax.set_ylabel("Predicted Monthly Usage (kWh)", fontsize=11)
    ax.set_title(f"Predicted vs Actual  (R² = {r2:.4f}, n = {len(actual):,})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _savefig(fig, "fig02_pred_vs_actual")


# =============================================================
# 3. 잔차 분포 히스토그램
# =============================================================
def fig03_residual_histogram():
    """(predicted - actual) 분포 히스토그램."""
    print("  [3/10] 잔차 분포 히스토그램")

    preds = _safe_read_parquet(ROOT / "data" / "ui" / "preds.parquet")
    if preds is None:
        preds = _safe_read_csv(ROOT / "checkpoints" / "lgbm_preds.csv")
    if preds is None:
        preds = _safe_read_parquet(ROOT / "checkpoints" / "lgbm_preds.parquet")
    ctx = _safe_read_parquet(ROOT / "data" / "ui" / "ctx.parquet")

    if preds is None or ctx is None:
        print("    SKIP: 데이터 없음")
        return

    for df in [preds, ctx]:
        if "year_month" in df.columns:
            df["year_month"] = df["year_month"].astype(str)

    merge_cols = ["customer_id", "year_month", "horizon_days"]
    available = [c for c in merge_cols if c in preds.columns and c in ctx.columns]
    merged = preds.merge(ctx, on=available, how="inner")

    if "pred_monthly_kwh" not in merged.columns or "full_month_kwh" not in merged.columns:
        print("    SKIP: 필수 컬럼 없음")
        return

    residuals = merged["pred_monthly_kwh"].values - merged["full_month_kwh"].values
    mask = np.isfinite(residuals)
    residuals = residuals[mask]

    if len(residuals) == 0:
        print("    SKIP: 유효 잔차 0건")
        return

    mu = np.mean(residuals)
    sigma = np.std(residuals)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=FACECOLOR)
    n_bins = min(100, max(30, len(residuals) // 50))
    ax.hist(residuals, bins=n_bins, color=COLORS["primary"], alpha=0.7,
            edgecolor="white", linewidth=0.5, density=True)

    # 평균선
    ax.axvline(mu, color=COLORS["secondary"], linewidth=2, linestyle="-",
               label=f"Mean = {mu:.1f} kWh")
    # +/- sigma
    ax.axvline(mu + sigma, color=COLORS["warm"], linewidth=1.5, linestyle="--",
               label=f"+1$\\sigma$ = {mu + sigma:.1f}")
    ax.axvline(mu - sigma, color=COLORS["warm"], linewidth=1.5, linestyle="--",
               label=f"$-$1$\\sigma$ = {mu - sigma:.1f}")

    ax.set_xlabel("Residual (Predicted - Actual, kWh)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Residual Distribution  ($\\mu$={mu:.1f}, $\\sigma$={sigma:.1f}, n={len(residuals):,})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _savefig(fig, "fig03_residual_distribution")


# =============================================================
# 4. 월별 예측 오차 박스플롯
# =============================================================
def fig04_monthly_error_boxplot():
    """sliding_results/sliding_raw.csv에서 월별 pct_error 분포."""
    print("  [4/10] 월별 예측 오차 박스플롯")

    df = _safe_read_csv(ROOT / "sliding_results" / "sliding_raw.csv")
    if df is None:
        df = _safe_read_parquet(ROOT / "sliding_results" / "sliding_raw.parquet")
    if df is None:
        print("    SKIP: sliding_raw 없음")
        return

    if "pct_error" not in df.columns or "month" not in df.columns:
        # month 복원 시도
        if "year_month" in df.columns and "month" not in df.columns:
            try:
                df["month"] = pd.to_datetime(df["year_month"].astype(str)).dt.month
            except Exception:
                print("    SKIP: month 컬럼 없음")
                return
        if "pct_error" not in df.columns:
            print("    SKIP: pct_error 컬럼 없음")
            return

    months = sorted(df["month"].dropna().unique())
    data_by_month = [df[df["month"] == m]["pct_error"].dropna().values for m in months]

    # 계절별 색
    month_colors = [SEASON_COLORS[_month_to_season(m)] for m in months]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=FACECOLOR)
    bp = ax.boxplot(
        data_by_month,
        positions=list(range(len(months))),
        widths=0.6,
        patch_artist=True,
        showfliers=False,  # 이상치 생략 (깔끔한 보고서 스타일)
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], month_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([f"M{m}" for m in months], fontsize=9)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("Monthly Error Distribution (Sliding Eval)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 범례 (계절)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=s)
                       for s, c in SEASON_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    _savefig(fig, "fig04_monthly_error_boxplot")


# =============================================================
# 5. Ablation 워터폴 차트
# =============================================================
def fig05_ablation_waterfall():
    """ablation_results/feature_group_contribution.csv -> waterfall."""
    print("  [5/10] Ablation 워터폴 차트")

    df = _safe_read_csv(ROOT / "ablation_results" / "feature_group_contribution.csv")
    if df is None:
        print("    SKIP: feature_group_contribution.csv 없음")
        return

    if "group" not in df.columns or "contribution" not in df.columns:
        print("    SKIP: 필수 컬럼 없음")
        return

    # contribution: 해당 그룹 제거 시 MAPE 변화 (양수 = 제거하면 나빠짐 = 기여 있음)
    df = df.sort_values("contribution", ascending=False).reset_index(drop=True)
    groups = df["group"].tolist()
    contributions = df["contribution"].values

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=FACECOLOR)

    # 워터폴: 각 바의 위치를 누적
    cumulative = 0
    for i, (g, c) in enumerate(zip(groups, contributions)):
        color = COLORS["secondary"] if c > 0 else COLORS["accent"]
        ax.barh(i, c, left=0, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        # 값 레이블
        ha = "left" if c >= 0 else "right"
        offset = 0.01 * max(abs(contributions)) if c >= 0 else -0.01 * max(abs(contributions))
        ax.text(c + offset, i, f"{c:+.3f}%p", va="center", ha=ha, fontsize=9)

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=10)
    ax.set_xlabel("MAPE Change (%p, when removed)", fontsize=11)
    ax.set_title("Feature Group Contribution (Ablation)", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["secondary"], alpha=0.8, label="Degrades when removed"),
        Patch(facecolor=COLORS["accent"], alpha=0.8, label="Improves when removed"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    _savefig(fig, "fig05_ablation_waterfall")


# =============================================================
# 6. 학습 곡선 (LightGBM train/valid)
# =============================================================
def fig06_learning_curve():
    """weights/dsz_lgbm/ 에 학습 로그가 있으면 train/valid 곡선."""
    print("  [6/10] 학습 곡선")

    # LightGBM 학습 로그 탐색
    log_candidates = [
        ROOT / "weights" / "dsz_lgbm" / "training_log.json",
        ROOT / "weights" / "dsz_lgbm" / "evals_result.json",
        ROOT / "weights" / "dsz_lgbm" / "training_log.csv",
        ROOT / "weights" / "pretrained_lgbm" / "training_log.json",
        ROOT / "weights" / "pretrained_lgbm" / "evals_result.json",
    ]

    log_data = None
    for p in log_candidates:
        if p.exists():
            if p.suffix == ".json":
                try:
                    with open(p, encoding="utf-8") as f:
                        log_data = json.load(f)
                    break
                except Exception:
                    continue
            elif p.suffix == ".csv":
                try:
                    log_data = pd.read_csv(p)
                    break
                except Exception:
                    continue

    if log_data is None:
        print("    SKIP: 학습 로그 없음")
        return

    # JSON 형태 파싱: {"training": {"mape": [...]}, "valid_1": {"mape": [...]}}
    train_loss = None
    valid_loss = None
    metric_name = "MAPE"

    if isinstance(log_data, dict):
        for train_key in ["training", "train", "train_set"]:
            if train_key in log_data:
                inner = log_data[train_key]
                for metric_key in ["mape", "l1", "mae", "rmse"]:
                    if metric_key in inner:
                        train_loss = inner[metric_key]
                        metric_name = metric_key.upper()
                        break
                break
        for val_key in ["valid_1", "valid", "validation", "valid_set"]:
            if val_key in log_data:
                inner = log_data[val_key]
                for metric_key in ["mape", "l1", "mae", "rmse"]:
                    if metric_key in inner:
                        valid_loss = inner[metric_key]
                        break
                break
    elif isinstance(log_data, pd.DataFrame):
        for col_t in ["train_mape", "train_loss", "train_mae"]:
            if col_t in log_data.columns:
                train_loss = log_data[col_t].tolist()
                metric_name = col_t.replace("train_", "").upper()
                break
        for col_v in ["valid_mape", "valid_loss", "valid_mae"]:
            if col_v in log_data.columns:
                valid_loss = log_data[col_v].tolist()
                break

    if train_loss is None and valid_loss is None:
        print("    SKIP: 학습 로그 파싱 실패")
        return

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=FACECOLOR)
    if train_loss is not None:
        ax.plot(range(1, len(train_loss) + 1), train_loss,
                color=COLORS["primary"], linewidth=1.5, alpha=0.8, label="Train")
    if valid_loss is not None:
        ax.plot(range(1, len(valid_loss) + 1), valid_loss,
                color=COLORS["secondary"], linewidth=1.5, alpha=0.8, label="Validation")
        # best iteration 표시
        best_iter = int(np.argmin(valid_loss)) + 1
        best_val = min(valid_loss)
        ax.axvline(best_iter, color=COLORS["gray"], linewidth=1, linestyle=":",
                   alpha=0.7)
        ax.annotate(f"Best: iter {best_iter}\n{metric_name}={best_val:.4f}",
                    xy=(best_iter, best_val), fontsize=9,
                    xytext=(best_iter + len(valid_loss) * 0.05, best_val),
                    arrowprops=dict(arrowstyle="->", color=COLORS["gray"]),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"]))

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"LightGBM Learning Curve ({metric_name})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _savefig(fig, "fig06_learning_curve")


# =============================================================
# 7. 알림 정밀도-재현율 scatter + F1 등고선
# =============================================================
def fig07_precision_recall():
    """eval_results/metrics.csv에서 모델별 precision/recall scatter."""
    print("  [7/10] 알림 정밀도-재현율")

    df = None
    for p in [ROOT / "eval_results" / "lgbm_vs_baselines.csv",
              ROOT / "eval_results" / "metrics.csv"]:
        df = _safe_read_csv(p)
        if df is not None:
            break
    if df is None:
        print("    SKIP: 메트릭 CSV 없음")
        return

    prec_col = "alarm_precision" if "alarm_precision" in df.columns else "precision"
    rec_col = "alarm_recall" if "alarm_recall" in df.columns else "recall"
    if prec_col not in df.columns or rec_col not in df.columns:
        print("    SKIP: precision/recall 컬럼 없음")
        return

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=FACECOLOR)

    # F1 등고선 배경
    p_grid = np.linspace(0.01, 1.0, 200)
    r_grid = np.linspace(0.01, 1.0, 200)
    P, R = np.meshgrid(p_grid, r_grid)
    F1 = 2 * P * R / (P + R)
    contour = ax.contourf(P, R, F1, levels=np.arange(0, 1.05, 0.1),
                          cmap="YlOrRd", alpha=0.15)
    contour_lines = ax.contour(P, R, F1, levels=np.arange(0.1, 1.0, 0.1),
                               colors="gray", linewidths=0.5, alpha=0.4)
    ax.clabel(contour_lines, fmt="F1=%.1f", fontsize=7, inline=True)

    # 모델별 점
    models = df["model"].unique() if "model" in df.columns else ["model"]
    marker_list = ["o", "s", "D", "^", "v", "P", "*", "X"]
    color_list = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  COLORS["warm"], COLORS["purple"], COLORS["gray"],
                  "#EC4899", "#06B6D4"]

    for i, m in enumerate(models):
        sub = df[df["model"] == m] if "model" in df.columns else df
        prec = sub[prec_col].values
        rec = sub[rec_col].values
        mask = np.isfinite(prec) & np.isfinite(rec)
        if not mask.any():
            continue
        marker = marker_list[i % len(marker_list)]
        color = color_list[i % len(color_list)]
        ax.scatter(prec[mask], rec[mask], s=80, marker=marker, color=color,
                   edgecolors="white", linewidth=0.5, label=m, zorder=5)

    ax.set_xlabel("Precision", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("Alarm Precision vs Recall (F1 Contour)", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.15, linestyle="--")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _savefig(fig, "fig07_precision_recall")


# =============================================================
# 8. 기상 변수 반응 곡선
# =============================================================
def fig08_weather_response():
    """HDD/CDD vs monthly_kwh 산점도 + 회귀선."""
    print("  [8/10] 기상 변수 반응 곡선")

    df = _safe_read_parquet(ROOT / "data" / "ui" / "monthly.parquet")
    if df is None:
        df = _safe_read_csv(ROOT / "data" / "ui" / "monthly.csv")
    if df is None:
        print("    SKIP: monthly 데이터 없음")
        return

    # HDD/CDD 컬럼 탐색
    hdd_col = None
    cdd_col = None
    kwh_col = None
    for c in df.columns:
        if "hdd" in c.lower():
            hdd_col = c
        if "cdd" in c.lower():
            cdd_col = c
        if c in ("monthly_kwh", "kwh", "full_month_kwh"):
            kwh_col = c

    if kwh_col is None:
        print("    SKIP: kWh 컬럼 없음")
        return
    if hdd_col is None and cdd_col is None:
        # features parquet 에서 찾기
        feat_df = _safe_read_parquet(ROOT / "data" / "ui" / "features.parquet")
        if feat_df is None:
            print("    SKIP: HDD/CDD 컬럼 없음")
            return
        for c in feat_df.columns:
            if "hdd" in c.lower() and hdd_col is None:
                hdd_col = c
            if "cdd" in c.lower() and cdd_col is None:
                cdd_col = c
        if hdd_col is None and cdd_col is None:
            print("    SKIP: HDD/CDD 컬럼 없음")
            return
        # merge
        common = list(set(df.columns) & set(feat_df.columns) - {kwh_col})
        if common:
            df = df.merge(feat_df[common + [hdd_col or "", cdd_col or ""]].dropna(axis=1),
                          on=common, how="left")

    plots = []
    if hdd_col and hdd_col in df.columns:
        plots.append((hdd_col, "HDD (난방도일)", COLORS["primary"]))
    if cdd_col and cdd_col in df.columns:
        plots.append((cdd_col, "CDD (냉방도일)", COLORS["secondary"]))

    if not plots:
        print("    SKIP: 기상 컬럼 최종 없음")
        return

    n_plots = len(plots)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6), facecolor=FACECOLOR,
                             squeeze=False)

    for idx, (col, label, color) in enumerate(plots):
        ax = axes[0, idx]
        x = df[col].values
        y = df[kwh_col].values
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x, y = x[mask], y[mask]

        if len(x) < 5:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        ax.scatter(x, y, s=10, alpha=0.3, color=color, rasterized=True)

        # 회귀선
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), color="black", linewidth=2, linestyle="--",
                label=f"y = {z[0]:.1f}x + {z[1]:.0f}")

        # 상관계수
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}\nn = {len(x):,}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", fc="white", ec=COLORS["gray"], alpha=0.9))

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Monthly Usage (kWh)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Weather Variable Response", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "fig08_weather_response")


# =============================================================
# 9. 시간대별 부하 프로파일
# =============================================================
def fig09_hourly_load_profile():
    """data/ui/daily.parquet에서 시간대별 평균 부하 (계약종별)."""
    print("  [9/10] 시간대별 부하 프로파일")

    # daily.parquet에는 시간대 정보가 없을 수 있으므로
    # 원본 LP 15분 데이터에서 시간대별 집계된 결과가 있는지 확인
    hourly = _safe_read_parquet(ROOT / "data" / "ui" / "hourly.parquet")
    if hourly is None:
        hourly = _safe_read_csv(ROOT / "data" / "ui" / "hourly.csv")

    # 없으면 daily에서 hour 컬럼이 있는 경우
    if hourly is None:
        daily = _safe_read_parquet(ROOT / "data" / "ui" / "daily.parquet")
        if daily is None:
            print("    SKIP: 시간대 데이터 없음")
            return

        # daily에 date + day_kwh 만 있으면 시간대별 분리 불가
        # 대안: LP 원본에서 시간대별 집계
        if "hour" not in daily.columns:
            # daily.parquet 은 일별 합계이므로 시간대별 분리 불가능
            # 그 대신 계약종별 일별 패턴이라도 그림
            if "contract_type" in daily.columns and "day_kwh" in daily.columns:
                ct_agg = daily.groupby("contract_type", observed=True)["day_kwh"].agg(
                    ["mean", "std", "count"]
                ).reset_index()
                if len(ct_agg) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor=FACECOLOR)
                    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                              COLORS["warm"], COLORS["purple"]]
                    bars = ax.bar(range(len(ct_agg)), ct_agg["mean"],
                                  yerr=ct_agg["std"], capsize=4,
                                  color=[colors[i % len(colors)] for i in range(len(ct_agg))],
                                  alpha=0.8, edgecolor="white")
                    ax.set_xticks(range(len(ct_agg)))
                    ax.set_xticklabels(ct_agg["contract_type"], fontsize=9, rotation=15)
                    ax.set_ylabel("Avg Daily Usage (kWh)", fontsize=11)
                    ax.set_title("Daily Usage by Contract Type", fontsize=13, fontweight="bold")
                    ax.grid(axis="y", alpha=0.2, linestyle="--")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    for bar, val in zip(bars, ct_agg["mean"]):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height(), f"{val:.0f}",
                                ha="center", va="bottom", fontsize=8)
                    _savefig(fig, "fig09_load_profile_by_contract")
                    return

            print("    SKIP: hour 컬럼 없음 (시간대별 분리 불가)")
            return

        hourly = daily

    # hour 컬럼이 있는 경우
    hour_col = "hour"
    kwh_col = None
    for c in ["kwh", "p_active_kwh", "day_kwh", "hourly_kwh"]:
        if c in hourly.columns:
            kwh_col = c
            break
    if kwh_col is None:
        print("    SKIP: kWh 컬럼 없음")
        return

    ct_col = "contract_type" if "contract_type" in hourly.columns else None
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=FACECOLOR)

    if ct_col:
        contract_types = hourly[ct_col].unique()
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  COLORS["warm"], COLORS["purple"]]
        for i, ct in enumerate(sorted(contract_types)):
            sub = hourly[hourly[ct_col] == ct]
            hourly_avg = sub.groupby(hour_col)[kwh_col].mean()
            color = colors[i % len(colors)]
            ax.plot(hourly_avg.index, hourly_avg.values, "-o", markersize=4,
                    linewidth=2, color=color, label=ct, alpha=0.85)
    else:
        hourly_avg = hourly.groupby(hour_col)[kwh_col].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, "-o", markersize=4,
                linewidth=2, color=COLORS["primary"], alpha=0.85)

    ax.set_xlabel("Hour", fontsize=11)
    ax.set_ylabel("Avg Usage (kWh)", fontsize=11)
    ax.set_title("Hourly Load Profile", fontsize=13, fontweight="bold")
    ax.set_xticks(range(24))
    if ct_col:
        ax.legend(fontsize=9)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 피크/오프피크 영역 표시
    ax.axvspan(10, 17, alpha=0.05, color=COLORS["secondary"], label="On-peak")
    ax.axvspan(23, 24, alpha=0.05, color=COLORS["primary"])
    ax.axvspan(0, 9, alpha=0.05, color=COLORS["primary"], label="Off-peak")

    _savefig(fig, "fig09_hourly_load_profile")


# =============================================================
# 10. 검침일별 MAPE 히트맵
# =============================================================
def fig10_mape_heatmap():
    """sliding_results/error_by_date.csv에서 (year_month x meter_day) 히트맵."""
    print("  [10/10] 검침일별 MAPE 히트맵")

    df = _safe_read_csv(ROOT / "sliding_results" / "error_by_date.csv")
    if df is None:
        df = _safe_read_parquet(ROOT / "sliding_results" / "error_by_date.parquet")
    if df is None:
        print("    SKIP: error_by_date 없음")
        return

    if "year_month" not in df.columns or "meter_day" not in df.columns:
        print("    SKIP: 필수 컬럼 없음")
        return

    metric_col = "mape_pct" if "mape_pct" in df.columns else "pct_error"
    if metric_col not in df.columns:
        print("    SKIP: MAPE 컬럼 없음")
        return

    # 피벗 테이블
    pivot = df.pivot_table(
        index="year_month", columns="meter_day", values=metric_col, aggfunc="mean"
    )
    if pivot.empty:
        print("    SKIP: 피벗 결과 비어있음")
        return

    # year_month 정렬
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(16, max(6, len(pivot) * 0.4)), facecolor=FACECOLOR)
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   interpolation="nearest")

    # 축 레이블
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=7, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index], fontsize=8)

    ax.set_xlabel("Meter Day", fontsize=11)
    ax.set_ylabel("Year-Month", fontsize=11)
    ax.set_title("MAPE Heatmap by Meter Day (%)", fontsize=13, fontweight="bold")

    # 컬러바
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("MAPE (%)", fontsize=10)

    # 셀 값 표시 (셀이 적을 때만)
    n_cells = pivot.shape[0] * pivot.shape[1]
    if n_cells <= 300:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    text_color = "white" if val > pivot.values[np.isfinite(pivot.values)].mean() else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=6, color=text_color)

    fig.tight_layout()
    _savefig(fig, "fig10_mape_heatmap")


# =============================================================
# 메인 실행
# =============================================================
def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"""
===================================================
  분석 보고서용 시각화 자동 생성
  출력: {FIG_DIR}/
  DPI: {DPI}
===================================================
""")

    figures = [
        ("fig01", "모델별 성능 비교 막대 차트",       fig01_model_comparison),
        ("fig02", "예측 vs 실측 산점도",              fig02_pred_vs_actual),
        ("fig03", "잔차 분포 히스토그램",              fig03_residual_histogram),
        ("fig04", "월별 예측 오차 박스플롯",           fig04_monthly_error_boxplot),
        ("fig05", "Ablation 워터폴 차트",             fig05_ablation_waterfall),
        ("fig06", "학습 곡선",                        fig06_learning_curve),
        ("fig07", "알림 정밀도-재현율",                fig07_precision_recall),
        ("fig08", "기상 변수 반응 곡선",              fig08_weather_response),
        ("fig09", "시간대별 부하 프로파일",            fig09_hourly_load_profile),
        ("fig10", "검침일별 MAPE 히트맵",             fig10_mape_heatmap),
    ]

    success = 0
    skipped = 0
    failed = 0

    for fig_id, title, func in figures:
        try:
            func()
            # 파일 생성 여부로 success/skip 판단
            any_fig = any((FIG_DIR / f).exists()
                          for f in FIG_DIR.glob(f"{fig_id}*"))
            if any_fig:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"    FAIL: {e}")

    print(f"""
===================================================
  완료: {success} 생성 / {skipped} 스킵 / {failed} 실패
  경로: {FIG_DIR}/
===================================================
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

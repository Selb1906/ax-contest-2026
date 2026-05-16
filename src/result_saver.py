"""분석 결과 이중 저장 — 이미지 + 데이터.

이미지(PNG)는 보고서용이지만 깨질 수 있으므로,
항상 원본 데이터(CSV/JSON)도 함께 저장하여 반출 후 복구 가능.

저장 구조:
  results/
    {name}.png          — 시각화 (깨질 수 있음)
    {name}.csv          — 원본 데이터 (항상 정확)
    {name}.json         — 메타데이터 (설정, 통계 등)
    {name}.parquet      — 대용량 데이터 (빠른 로드)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_plot_setup():
    """폰트 깨짐 방지 — 여러 폰트 시도."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.facecolor"] = "white"
    return matplotlib, plt


def save_dataframe(
    df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    description: str = "",
):
    """DataFrame을 CSV + Parquet + 메타 JSON으로 저장."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(out / f"{name}.parquet", index=False)
    except Exception:
        pass

    meta = {
        "name": name,
        "description": description,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    with open(out / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_chart(
    plot_func,
    out_dir: str | Path,
    name: str,
    data: pd.DataFrame | dict | None = None,
    description: str = "",
    figsize: tuple = (10, 6),
):
    """차트를 PNG + 원본 데이터(CSV/JSON)로 이중 저장.

    Parameters
    ----------
    plot_func : (fig, ax) → None 형태의 함수
    data : 차트의 원본 데이터 (반출 후 재생성용)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 이미지 저장 시도
    try:
        matplotlib, plt = _safe_plot_setup()
        fig, ax = plt.subplots(figsize=figsize)
        plot_func(fig, ax)
        plt.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] {name}.png 저장 실패: {e}")
        # 이미지 실패해도 데이터는 저장

    # 원본 데이터 저장
    if data is not None:
        if isinstance(data, pd.DataFrame):
            save_dataframe(data, out, name, description)
        elif isinstance(data, dict):
            with open(out / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def save_comparison_table(
    results: list[dict],
    out_dir: str | Path,
    name: str,
    sort_by: str | None = None,
    ascending: bool = True,
    description: str = "",
):
    """비교 테이블을 CSV + 콘솔 출력 + JSON."""
    df = pd.DataFrame(results)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    save_dataframe(df, out_dir, name, description)

    # 콘솔 출력
    print(f"\n  {name}:")
    print(df.to_string(index=False))
    return df


def save_full_results(
    out_dir: str | Path,
    metrics: pd.DataFrame,
    params: dict | None = None,
    feature_importance: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    description: str = "",
):
    """모델 평가 결과를 종합 저장.

    - metrics.csv/parquet — 평가 지표
    - params.json — 모델 파라미터
    - feature_importance.csv — 피처 중요도
    - predictions.csv/parquet — 예측값 (반출 주의)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_dataframe(metrics, out, "metrics", description)

    if params:
        with open(out / "params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2, default=str)

    if feature_importance is not None:
        save_dataframe(feature_importance, out, "feature_importance")

    if predictions is not None:
        save_dataframe(predictions, out, "predictions", "예측값 (반출 시 개인정보 확인)")

    # 요약 텍스트
    summary_lines = [f"# {description}", ""]
    if "mape_pct" in metrics.columns:
        for _, row in metrics.iterrows():
            line = f"- {row.get('model','')}"
            if "horizon_days" in row:
                line += f" +{int(row['horizon_days'])}d"
            line += f": MAPE {row['mape_pct']:.3f}%"
            if "alarm_f1" in row:
                line += f", F1 {row['alarm_f1']:.3f}"
            summary_lines.append(line)

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

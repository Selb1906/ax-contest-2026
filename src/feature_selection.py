"""피처 선택 파이프라인 — 상관 순위 + 다중공선성 제거.

단계:
  1. 모든 후보 피처 vs 목표변수(월 사용량) 상관계수 산출 → 순위
  2. 피처 간 상관행렬 → 높은 쌍(>threshold) 중 목표 상관이 낮은 쪽 제거
  3. VIF 순차 제거 (VIF > threshold인 피처 반복 제거)
  4. 최종 선택된 피처 목록 반환
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    selected: list[str]
    ranking: pd.DataFrame          # 전체 순위 (피처, 상관, VIF)
    removed_correlation: list[str] # 상관 중복으로 제거된 피처
    removed_vif: list[str]         # VIF로 제거된 피처
    inter_correlation: pd.DataFrame # 피처 간 상관행렬


def target_correlation_ranking(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "full_month_kwh",
) -> pd.DataFrame:
    """모든 피처 vs 목표변수 상관계수 순위."""
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        valid = df[[col, target_col]].dropna()
        if len(valid) < 10:
            continue
        corr = valid[col].corr(valid[target_col])
        rows.append({
            "feature": col,
            "corr_with_target": float(corr),
            "abs_corr": float(abs(corr)),
            "n_valid": int(len(valid)),
        })
    ranking = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking


def remove_correlated_pairs(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "full_month_kwh",
    threshold: float = 0.85,
) -> tuple[list[str], list[str]]:
    """피처 간 상관이 threshold 이상인 쌍에서 목표 상관이 낮은 쪽 제거."""
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return available, []

    corr_matrix = df[available].corr().abs()
    target_corr = {c: abs(df[c].corr(df[target_col])) for c in available
                   if df[c].notna().sum() > 10}

    removed = set()
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            a, b = available[i], available[j]
            if a in removed or b in removed:
                continue
            if corr_matrix.loc[a, b] >= threshold:
                # 목표 상관이 낮은 쪽 제거
                corr_a = target_corr.get(a, 0)
                corr_b = target_corr.get(b, 0)
                drop = b if corr_a >= corr_b else a
                removed.add(drop)

    selected = [c for c in available if c not in removed]
    return selected, list(removed)


def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """피처별 VIF 산출."""
    available = [c for c in feature_cols if c in df.columns]
    data = df[available].dropna()
    if len(data) < len(available) + 1:
        return pd.DataFrame(columns=["feature", "vif"])

    X = data.values.astype(float)
    X = X - X.mean(axis=0)

    vif_rows = []
    for i, col in enumerate(available):
        others = [j for j in range(len(available)) if j != i]
        if not others:
            continue
        X_others = X[:, others]
        y = X[:, i]
        try:
            coef, *_ = np.linalg.lstsq(X_others, y, rcond=None)
            y_hat = X_others @ coef
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        except Exception:
            vif = float("nan")
        vif_rows.append({"feature": col, "vif": float(vif)})

    return pd.DataFrame(vif_rows)


def remove_high_vif(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "full_month_kwh",
    vif_threshold: float = 10.0,
    max_iterations: int = 20,
) -> tuple[list[str], list[str]]:
    """VIF가 threshold 이상인 피처를 순차 제거.

    제거 시 목표 상관이 가장 낮은 피처부터 제거.
    """
    current = [c for c in feature_cols if c in df.columns]
    removed = []

    target_corr = {c: abs(df[c].corr(df[target_col])) for c in current
                   if df[c].notna().sum() > 10}

    for _ in range(max_iterations):
        if len(current) < 2:
            break

        vif_df = compute_vif(df, current)
        if vif_df.empty:
            break

        high_vif = vif_df[vif_df["vif"] > vif_threshold]
        if high_vif.empty:
            break

        # 목표 상관이 가장 낮은 high-VIF 피처 제거
        candidates = high_vif["feature"].tolist()
        worst = min(candidates, key=lambda c: target_corr.get(c, 0))
        current.remove(worst)
        removed.append(worst)

    return current, removed


def auto_select_features(
    df: pd.DataFrame,
    candidate_cols: list[str],
    target_col: str = "full_month_kwh",
    top_n: int | None = None,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    min_corr: float = 0.01,
) -> SelectionResult:
    """전체 피처 선택 파이프라인.

    Parameters
    ----------
    df : 피처 + 목표변수 포함 DataFrame
    candidate_cols : 후보 피처 목록
    target_col : 목표변수 컬럼
    top_n : 상관 순위 상위 N개만 사용 (None이면 전부)
    corr_threshold : 피처 간 상관 제거 임계값
    vif_threshold : VIF 제거 임계값
    min_corr : 최소 상관 컷오프 — 이 미만인 피처는 처음부터 제외
    """
    available = [c for c in candidate_cols if c in df.columns and c != target_col]

    # 1. 목표 상관 순위
    ranking = target_correlation_ranking(df, available, target_col)

    # 최소 상관 컷오프 적용
    before_cutoff = len(ranking)
    ranking_cut = ranking[ranking["abs_corr"] >= min_corr]
    n_cut = before_cutoff - len(ranking_cut)
    if n_cut > 0:
        cut_features = ranking[ranking["abs_corr"] < min_corr]["feature"].tolist()
        print(f"  [cutoff] 상관 < {min_corr}: {n_cut}개 제외 — {cut_features}")

    # 상위 N개 선택
    if top_n is not None:
        top_features = ranking_cut.head(top_n)["feature"].tolist()
    else:
        top_features = ranking_cut["feature"].tolist()

    # 2. 피처 간 상관 중복 제거
    after_corr, removed_corr = remove_correlated_pairs(
        df, top_features, target_col, corr_threshold
    )

    # 3. VIF 순차 제거
    after_vif, removed_vif = remove_high_vif(
        df, after_corr, target_col, vif_threshold
    )

    # 피처 간 상관행렬 (최종 선택 기준)
    inter_corr = df[after_vif].corr() if after_vif else pd.DataFrame()

    # ranking에 VIF 추가
    final_vif = compute_vif(df, after_vif)
    ranking = ranking.merge(final_vif, on="feature", how="left")
    ranking["selected"] = ranking["feature"].isin(after_vif)

    return SelectionResult(
        selected=after_vif,
        ranking=ranking,
        removed_correlation=removed_corr,
        removed_vif=removed_vif,
        inter_correlation=inter_corr,
    )


def print_selection_report(result: SelectionResult) -> None:
    """선택 결과 출력."""
    print(f"\n{'='*60}")
    print(f"피처 선택 결과: {len(result.selected)}개 선택")
    print(f"{'='*60}")

    print(f"\n상관 중복 제거: {len(result.removed_correlation)}개")
    for f in result.removed_correlation:
        print(f"  - {f}")

    print(f"\nVIF 제거: {len(result.removed_vif)}개")
    for f in result.removed_vif:
        print(f"  - {f}")

    print(f"\n최종 선택 피처 (상관 순):")
    selected_rank = result.ranking[result.ranking["selected"]].sort_values("abs_corr", ascending=False)
    for _, row in selected_rank.iterrows():
        vif_str = f"VIF={row['vif']:.1f}" if pd.notna(row.get("vif")) else ""
        print(f"  {row['rank']:>3d}. {row['feature']:35s} corr={row['corr_with_target']:+.4f}  {vif_str}")

    if result.removed_correlation or result.removed_vif:
        print(f"\n제거된 피처:")
        removed_rank = result.ranking[~result.ranking["selected"]].sort_values("abs_corr", ascending=False)
        for _, row in removed_rank.iterrows():
            reason = "corr" if row["feature"] in result.removed_correlation else "VIF"
            print(f"  {row['rank']:>3d}. {row['feature']:35s} corr={row['corr_with_target']:+.4f}  ({reason})")

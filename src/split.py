"""시계열 데이터 분할 — 누수 방지.

시간 순서 기반 Train/Val/Test 분할.
미래 데이터가 과거 예측에 영향을 주지 않도록 보장.

분할 전략:
  Train: 학습 (모델 파라미터)
  Val:   검증 (피처 선택, 기준온도, Ablation, 하이퍼파라미터)
  Test:  최종 평가 (한 번만 사용, 보고서 결과)

누수 체크리스트:
  ✅ 래그 피처: shift로 과거만 참조
  ✅ partial_kwh: 해당 월 관측분만
  ⚠️ 피처 선택: train+val에서만 수행 (test 미포함)
  ⚠️ 기준온도 최적화: train+val에서만 수행
  ⚠️ 프로파일러: 전체 데이터 사용 (설계 결정에만, 모델 학습에 직접 사용 안 함)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def time_split(
    features: pd.DataFrame,
    train_end: pd.Period,
    val_end: pd.Period | None = None,
) -> dict:
    """시간 순서 기반 분할.

    Parameters
    ----------
    features : year_month 컬럼 포함 DataFrame
    train_end : 학습 종료 기간 (이 기간 포함)
    val_end : 검증 종료 기간 (None이면 val 없이 train/test만)

    Returns
    -------
    {"train": mask, "val": mask, "test": mask} boolean arrays
    """
    ym = features["year_month"]

    if val_end is not None:
        train_mask = ym <= train_end
        val_mask = (ym > train_end) & (ym <= val_end)
        test_mask = ym > val_end
    else:
        train_mask = ym <= train_end
        val_mask = np.zeros(len(features), dtype=bool)
        test_mask = ym > train_end

    splits = {
        "train": train_mask.values if hasattr(train_mask, "values") else train_mask,
        "val": val_mask.values if hasattr(val_mask, "values") else val_mask,
        "test": test_mask.values if hasattr(test_mask, "values") else test_mask,
    }

    print(f"  분할: train={splits['train'].sum():,} "
          f"val={splits['val'].sum():,} "
          f"test={splits['test'].sum():,}")

    return splits


def suggest_split(
    features: pd.DataFrame,
) -> tuple[pd.Period, pd.Period]:
    """데이터 기간에 따라 분할 기간 자동 추천.

    3년: train 1.5년, val 0.5년, test 1년
    2년: train 1년, val 없음, test 1년
    """
    ym_min = features["year_month"].min()
    ym_max = features["year_month"].max()
    total_months = (ym_max.year - ym_min.year) * 12 + (ym_max.month - ym_min.month) + 1

    if total_months >= 30:
        # 3년 이상: train ~18개월, val ~6개월, test ~12개월
        train_end = ym_min + 17  # 18개월
        val_end = train_end + 6
        print(f"  데이터 {total_months}개월 → train ~18m, val ~6m, test ~{total_months-24}m")
    elif total_months >= 18:
        # 1.5~2.5년: train 절반, val 없음, test 절반
        mid = total_months // 2
        train_end = ym_min + mid - 1
        val_end = None
        print(f"  데이터 {total_months}개월 → train {mid}m, test {total_months-mid}m (val 없음)")
    else:
        # 1.5년 미만: train 2/3, test 1/3
        train_months = total_months * 2 // 3
        train_end = ym_min + train_months - 1
        val_end = None
        print(f"  데이터 {total_months}개월 (짧음) → train {train_months}m, test {total_months-train_months}m")

    return train_end, val_end


def check_leakage(features: pd.DataFrame, splits: dict) -> list[str]:
    """데이터 누수 체크 — 경고 목록 반환."""
    warnings = []
    train = features[splits["train"]]
    test = features[splits["test"]]

    # 1. train과 test에 같은 year_month가 있으면 누수
    overlap = set(train["year_month"].unique()) & set(test["year_month"].unique())
    if overlap:
        warnings.append(f"train/test year_month 겹침: {overlap}")

    # 2. test의 year_month가 train보다 이전이면 시간 역전
    if not train.empty and not test.empty and test["year_month"].min() <= train["year_month"].max():
        warnings.append(f"시간 역전: test min={test['year_month'].min()} <= train max={train['year_month'].max()}")

    # 3. customer_id가 test에만 있고 train에 없으면 cold start
    if "customer_id" in features.columns:
        train_custs = set(train["customer_id"].unique())
        test_custs = set(test["customer_id"].unique())
        test_only = test_custs - train_custs
        if test_only:
            warnings.append(f"cold start 고객 {len(test_only)}명 (test에만 존재)")

    if warnings:
        print(f"  ⚠ 누수 경고:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"  ✅ 누수 체크 통과")

    return warnings

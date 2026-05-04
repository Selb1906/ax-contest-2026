"""스케일 아웃 지원 — 대용량 처리 + 배치 추론.

3,500호에서는 pandas 단일 프로세스로 충분하지만,
수만~수십만 호 확장 시를 위한 설계 패턴.

핵심 전략:
  1. 고객 청크 분할 — 메모리 제한 내에서 순차 처리
  2. 피처 빌드 → 저장 → 추론 분리 (학습과 추론을 독립 실행)
  3. 추론은 학습된 모델 로드 후 배치 실행
  4. 병렬화 가능 구조 (multiprocessing / Dask 전환 용이)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .schemas import CUSTOMER_ID


def customer_chunks(
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> Iterator[pd.DataFrame]:
    """고객 ID 기준으로 청크 분할.

    메모리 제한 시 전체 데이터를 한 번에 올리지 않고
    고객 단위로 나눠 처리.
    """
    cust_ids = df[CUSTOMER_ID].unique()
    for i in range(0, len(cust_ids), chunk_size):
        chunk_ids = cust_ids[i:i + chunk_size]
        yield df[df[CUSTOMER_ID].isin(chunk_ids)]


def build_features_chunked(
    df: pd.DataFrame,
    chunk_size: int = 500,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """청크 단위 피처 빌드 — 대용량 데이터 대응.

    고객을 chunk_size씩 나눠 피처 생성 → 합치기.
    out_dir 지정 시 중간 결과를 parquet으로 저장 (메모리 절약).
    """
    from .models.lgbm import build_features

    n_customers = df[CUSTOMER_ID].nunique()
    n_chunks = (n_customers + chunk_size - 1) // chunk_size
    print(f"[scale] {n_customers} customers → {n_chunks} chunks (size={chunk_size})")

    frames = []
    spec = None
    t0 = time.time()

    for i, chunk_df in enumerate(customer_chunks(df, chunk_size)):
        t = time.time()
        chunk_features, chunk_spec = build_features(chunk_df)
        elapsed = time.time() - t

        if spec is None:
            spec = chunk_spec

        if out_dir:
            p = Path(out_dir) / f"chunk_{i:04d}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            chunk_features.to_parquet(p, index=False)
        else:
            frames.append(chunk_features)

        n_cust = chunk_df[CUSTOMER_ID].nunique()
        print(f"  chunk {i+1}/{n_chunks}: {n_cust} customers, "
              f"{len(chunk_features)} rows, {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"[scale] total: {total_time:.1f}s")

    if out_dir:
        all_features = pd.concat(
            [pd.read_parquet(p) for p in sorted(Path(out_dir).glob("chunk_*.parquet"))],
            ignore_index=True,
        )
    else:
        all_features = pd.concat(frames, ignore_index=True)

    return all_features, spec


def batch_predict(
    weights_dir: str | Path,
    features: pd.DataFrame,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """배치 추론 — 학습된 모델로 대량 고객 예측.

    학습과 분리되어 독립 실행 가능.
    chunk_size 행씩 나눠 predict (메모리 안전).
    """
    from .models.lgbm import load, predict

    booster, spec = load(weights_dir)
    print(f"[predict] model loaded from {weights_dir}")
    print(f"  features: {spec.all}")

    preds = []
    for i in range(0, len(features), chunk_size):
        chunk = features.iloc[i:i + chunk_size]
        pred = predict(booster, chunk, spec)
        preds.append(pred)

    result = pd.concat(preds, ignore_index=True)
    print(f"[predict] {len(result)} predictions")
    return result


def estimate_scale_metrics(
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> dict:
    """스케일 아웃 지표 추정 — 보고서용.

    현재 데이터 크기 기준으로 10만/100만 호 확장 시
    소요 시간과 메모리를 추정.
    """
    n_customers = df[CUSTOMER_ID].nunique()
    n_rows = len(df)
    rows_per_customer = n_rows / n_customers if n_customers > 0 else 0
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    mb_per_customer = memory_mb / n_customers if n_customers > 0 else 0

    # 청크 하나 처리 시간 측정
    sample = df[df[CUSTOMER_ID].isin(df[CUSTOMER_ID].unique()[:min(100, n_customers)])]
    t0 = time.time()
    from .models.lgbm import build_features
    try:
        build_features(sample)
        sec_per_100 = time.time() - t0
    except Exception:
        sec_per_100 = float("nan")

    projections = {}
    for target_n in [10_000, 100_000, 1_000_000]:
        proj_memory_gb = mb_per_customer * target_n / 1024
        proj_time_min = sec_per_100 * (target_n / 100) / 60
        proj_chunks = (target_n + chunk_size - 1) // chunk_size
        projections[f"{target_n:,} customers"] = {
            "estimated_memory_gb": round(proj_memory_gb, 1),
            "estimated_time_min": round(proj_time_min, 1),
            "n_chunks": proj_chunks,
            "parallelizable": True,
        }

    return {
        "current": {
            "n_customers": n_customers,
            "n_rows": n_rows,
            "rows_per_customer": round(rows_per_customer),
            "memory_mb": round(memory_mb, 1),
            "mb_per_customer": round(mb_per_customer, 3),
            "sec_per_100_customers": round(sec_per_100, 2),
        },
        "projections": projections,
        "design_notes": {
            "chunking": "고객 축 분할 — 메모리 선형 증가, 청크 간 독립",
            "parallelism": "각 청크를 독립 프로세스/노드에서 병렬 처리 가능",
            "model": "글로벌 모델 1개 — 고객 수 증가해도 모델 크기 불변",
            "inference": "학습/추론 분리 — 추론은 모델 로드 후 배치 실행",
            "storage": "중간 결과 parquet 저장 — 재시작 시 이어서 처리",
        },
    }

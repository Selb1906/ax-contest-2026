import sys; sys.path.insert(0, 'E:/AX_Contest')
import pandas as pd
import numpy as np

df = pd.read_csv('E:/AX_Contest/kpx_BTM.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print('=== 스키마 ===')
print(f'rows: {len(df):,}')
print(f'period: {df["datetime"].min()} ~ {df["datetime"].max()}')
print(f'columns: {df.columns.tolist()}')
print()

print('=== dtypes ===')
print(df.dtypes)
print()

print('=== 기초 통계 ===')
print(df.describe().round(2).to_string())
print()

print('=== 결측 ===')
print(df.isna().sum())
print()

# 시간 간격 확인
diffs = df['datetime'].diff().dt.total_seconds().dropna()
print(f'=== 시간 간격 ===')
print(f'min: {diffs.min()}s, max: {diffs.max()}s, median: {diffs.median()}s')
interval = diffs.mode().iloc[0]
print(f'가장 빈번: {interval}s = {interval/3600}h')
print()

# 연도별
print('=== 연도별 행수 ===')
for y in sorted(df['datetime'].dt.year.unique()):
    n = len(df[df['datetime'].dt.year == y])
    print(f'  {y}: {n:,}')
print()

# self_use_solar_mw 월별 패턴 (BTM 핵심)
df['year_month'] = df['datetime'].dt.to_period('M')
monthly = df.groupby('year_month')['self_use_solar_mw'].agg(['mean', 'sum', 'max']).reset_index()
print('=== self_use_solar_mw 월별 ===')
print(monthly.tail(12).to_string(index=False))
print()

# 샘플
print('=== 상위 5행 ===')
print(df.head().to_string(index=False))

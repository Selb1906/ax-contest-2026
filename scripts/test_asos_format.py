import sys; sys.path.insert(0, 'E:/AX_Contest')
from src.weather.asos import read_asos_directory

df = read_asos_directory('E:/AX_Contest/ASOS')
print(f'rows: {len(df):,}')
print(f'period: {df["ts"].min()} ~ {df["ts"].max()}')
print(f'years: {sorted(df["ts"].dt.year.unique())}')

print(f'\nhour range: {df["ts"].dt.hour.min()} ~ {df["ts"].dt.hour.max()}')

midnight = df[df["ts"].dt.hour == 0].head(3)
print(f'\n00시 samples:')
print(midnight[["station_id", "ts", "temp_c"]].to_string(index=False))

last = df[df["ts"].dt.hour == 23].head(3)
print(f'\n23시 samples:')
print(last[["station_id", "ts", "temp_c"]].to_string(index=False))

print(f'\ntemp_c null: {df["temp_c"].isna().sum():,} / {len(df):,} ({df["temp_c"].isna().mean()*100:.2f}%)')

print(f'\n연도별:')
for y in sorted(df["ts"].dt.year.unique()):
    n = len(df[df["ts"].dt.year == y])
    expected = 8 * 365 * 24
    print(f'  {y}: {n:,} rows (expected ~{expected:,})')

# 하루 24시간 모두 있는지 확인
sample_day = df[(df["station_id"] == 108) & (df["ts"].dt.date.astype(str) == "2022-07-15")]
print(f'\n서울 2022-07-15 시간대:')
print(sorted(sample_day["ts"].dt.hour.tolist()))

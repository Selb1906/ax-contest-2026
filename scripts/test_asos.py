"""ASOS 파서 테스트."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.weather.asos import (
    read_asos_directory, monthly_degree_hours,
    monthly_weather_features, weighted_national_average,
)

ROOT = Path(__file__).resolve().parents[1]
df = read_asos_directory(str(ROOT / "ASOS"))
sids = sorted(df["station_id"].unique())
print(f"rows: {len(df):,}  stations: {sids}")
print(f"period: {df['ts'].min()} ~ {df['ts'].max()}")

mdh = monthly_degree_hours(df)
print(f"\nmonthly degree-hours: {mdh.shape}")
seoul = mdh[mdh["station_id"] == 108].head(6)
print(seoul[["station_id", "year_month", "cdh_total", "hdh_total",
             "cdh_daytime", "cdh_evening", "temp_mean"]].to_string(index=False))

natl = weighted_national_average(mdh)
print(f"\nnational weighted avg: {natl.shape}")
print(natl.head(6)[["year_month", "cdh_total", "hdh_total", "temp_mean"]].to_string(index=False))

# Jul 2022 vs Jan 2022 comparison
jul = natl[natl["year_month"] == "2022-07"].iloc[0]
jan = natl[natl["year_month"] == "2022-01"].iloc[0]
print(f"\n2022-01: temp={jan['temp_mean']:.1f}  HDH={jan['hdh_total']:.0f}  CDH={jan['cdh_total']:.0f}")
print(f"2022-07: temp={jul['temp_mean']:.1f}  HDH={jul['hdh_total']:.0f}  CDH={jul['cdh_total']:.0f}")

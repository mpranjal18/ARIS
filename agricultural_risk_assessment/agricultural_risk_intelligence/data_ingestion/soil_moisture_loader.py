from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_soil_moisture(
    weather_df: pd.DataFrame,
    raw_dir: str | Path,
    file_name: str = "soil_moisture_data.csv",
    window: int = 7,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file_name

    df = weather_df.copy()
    rain_roll = df["rainfall_mm"].rolling(window, min_periods=1).mean()
    humidity = df["humidity_pct"] / 100

    ndwi_proxy = 0.7 * rain_roll + 0.3 * humidity
    ndwi_min = ndwi_proxy.min()
    ndwi_max = ndwi_proxy.max()
    ndwi_norm = (ndwi_proxy - ndwi_min) / (ndwi_max - ndwi_min + 1e-6)

    out = pd.DataFrame({"date": df["date"], "soil_moisture_index": ndwi_norm.round(4)})
    out.to_csv(file_path, index=False)
    return out


if __name__ == "__main__":
    from weather_loader import load_weather_data

    weather = load_weather_data(Path("../data/raw"))
    df = compute_soil_moisture(weather, Path("../data/raw"))
    print(df.tail())

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _generate_weather(start_date: pd.Timestamp, days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=days, freq="D")
    seasonal = 22 + 8 * np.sin(np.linspace(0, 3 * np.pi, days))
    tmax = seasonal + rng.normal(6, 2.0, days)
    tmin = seasonal - rng.normal(2, 1.5, days)
    humidity = 60 + 20 * np.sin(np.linspace(0, 2 * np.pi, days)) + rng.normal(0, 5, days)
    humidity = np.clip(humidity, 30, 95)
    rainfall = rng.gamma(2.0, 3.0, days) * (humidity / 100)
    rainfall = np.clip(rainfall, 0, None)

    return pd.DataFrame(
        {
            "date": dates,
            "rainfall_mm": rainfall.round(2),
            "tmax_c": tmax.round(2),
            "tmin_c": tmin.round(2),
            "humidity_pct": humidity.round(2),
        }
    )


def load_weather_data(
    raw_dir: str | Path,
    days_history: int = 180,
    forecast_days: int = 15,
    file_name: str = "weather_data.csv",
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file_name

    if file_path.exists():
        history = pd.read_csv(file_path, parse_dates=["date"])
    else:
        start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_history - 1)
        history = _generate_weather(start_date, days_history, seed=7)
        history.to_csv(file_path, index=False)

    last_date = history["date"].max()
    forecast_start = last_date + pd.Timedelta(days=1)
    forecast = _generate_weather(forecast_start, forecast_days, seed=17)

    history = history.copy()
    history["is_forecast"] = False
    forecast["is_forecast"] = True

    combined = pd.concat([history, forecast], ignore_index=True)
    return combined


if __name__ == "__main__":
    df = load_weather_data(Path("../data/raw"))
    print(df.tail())

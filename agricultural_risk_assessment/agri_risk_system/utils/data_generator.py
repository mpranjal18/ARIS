from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REGION_COORDS: Dict[str, Tuple[float, float]] = {
    "Bhopal": (23.2599, 77.4126),
    "Sehore": (23.2032, 77.0850),
    "Ashta": (23.0175, 76.7221),
}


def _bounded(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(arr, low, high)


def generate_synthetic_data(days: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    today = datetime.today().date()
    dates = pd.date_range(end=today, periods=days, freq="D")

    rows: List[dict] = []
    for region, (lat, lon) in REGION_COORDS.items():
        shift = {"Bhopal": 0.0, "Sehore": 1.8, "Ashta": 2.5}[region]
        t = np.linspace(0, 3 * np.pi, days)

        rainfall = _bounded(18 + 10 * np.sin(t + shift) + rng.normal(0, 4, days), 0, 70)
        temperature = _bounded(25 + 8 * np.sin(t + 0.7 + shift) + rng.normal(0, 2.3, days), 10, 45)
        humidity = _bounded(62 + 16 * np.sin(t + 1.6 - shift) + rng.normal(0, 5, days), 25, 98)
        ndvi = _bounded(0.62 + 0.12 * np.sin(t - 0.4) - 0.002 * (temperature - 30) + rng.normal(0, 0.03, days), 0.2, 0.9)

        pest_signal = 0.4 * (humidity / 100) + 0.35 * (temperature / 40) + 0.25 * ((1 - ndvi))
        pest_level = _bounded((pest_signal + rng.normal(0, 0.08, days)) * 100, 0, 100)

        for i in range(days):
            rows.append(
                {
                    "date": dates[i],
                    "region": region,
                    "lat": lat,
                    "lon": lon,
                    "rainfall": round(float(rainfall[i]), 2),
                    "temperature": round(float(temperature[i]), 2),
                    "humidity": round(float(humidity[i]), 2),
                    "ndvi": round(float(ndvi[i]), 4),
                    "pest_level": round(float(pest_level[i]), 2),
                    "crop": "Wheat",
                    "state": "Madhya Pradesh",
                    "cluster": "Bhopal-Sehore-Ashta",
                }
            )

    df = pd.DataFrame(rows).sort_values(["region", "date"]).reset_index(drop=True)
    return df

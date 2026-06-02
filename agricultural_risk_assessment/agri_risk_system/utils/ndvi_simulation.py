from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


CROP_BASE = {
    "Wheat": {"base": 0.44, "amp": 0.24, "peak": 70},
    "Maize": {"base": 0.38, "amp": 0.30, "peak": 92},
}


def _smooth_curve(days: int, base: float, amp: float, peak_day: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    doy = np.arange(1, days + 1)
    sigma = 24
    curve = base + amp * np.exp(-((doy - peak_day) ** 2) / (2 * sigma**2))
    noise = rng.normal(0, 0.012, days)
    return np.clip(curve + noise, 0.15, 0.92)


def simulate_crop_ndvi(days: int = 180, seed: int = 123) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today().date(), periods=days, freq="D")
    rows: list[dict] = []

    for idx, (crop, params) in enumerate(CROP_BASE.items()):
        vals = _smooth_curve(days, params["base"], params["amp"], params["peak"], seed + idx)
        for i, dt in enumerate(dates):
            rows.append({"date": dt, "doy": int(dt.dayofyear), "crop": crop, "ndvi": round(float(vals[i]), 4)})

    out = pd.DataFrame(rows)
    return out.sort_values(["crop", "date"]).reset_index(drop=True)


def interpolate_ndvi(ndvi_df: pd.DataFrame, points_per_day: int = 3) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for crop, group in ndvi_df.groupby("crop"):
        g = group.sort_values("doy")
        x_old = g["doy"].to_numpy()
        y_old = g["ndvi"].to_numpy()
        x_new = np.linspace(x_old.min(), x_old.max(), len(x_old) * points_per_day)
        y_new = np.interp(x_new, x_old, y_old)
        records.append(pd.DataFrame({"crop": crop, "doy": x_new, "ndvi": y_new}))

    return pd.concat(records, ignore_index=True)


def forecast_ndvi_by_crop(ndvi_df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    rows: list[dict] = []
    for crop, group in ndvi_df.groupby("crop"):
        g = group.sort_values("date").copy()
        slope = (g["ndvi"].iloc[-1] - g["ndvi"].iloc[-8]) / 7 if len(g) > 8 else 0
        last_date = g["date"].iloc[-1]
        last_ndvi = g["ndvi"].iloc[-1]
        for i in range(1, horizon + 1):
            dt = last_date + pd.Timedelta(days=i)
            pred = float(np.clip(last_ndvi + slope * i, 0.15, 0.92))
            rows.append({"date": dt, "crop": crop, "predicted_ndvi": round(pred, 4), "doy": int(dt.dayofyear)})

    return pd.DataFrame(rows)

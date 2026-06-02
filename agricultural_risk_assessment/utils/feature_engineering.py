"""Feature engineering for climate, vegetation, and pest risk modelling."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


BASE_FEATURES: List[str] = [
    "rainfall",
    "temperature",
    "humidity",
    "soil_moisture",
    "ndvi",
    "aphid_population",
    "rust_severity",
    "armyworm_presence",
]


def merge_raw_data(weather: pd.DataFrame, ndvi: pd.DataFrame, pest: pd.DataFrame) -> pd.DataFrame:
    """Merges weather, NDVI, and pest tables on date and region."""
    merged = weather.merge(ndvi, on=["date", "region"], how="inner")
    merged = merged.merge(pest, on=["date", "region"], how="inner")
    merged = merged.sort_values(["region", "date"]).reset_index(drop=True)
    return merged


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds temporal, anomaly, and interaction features for model performance."""
    out = df.copy()

    # Backward compatibility for older weather/NDVI schemas.
    if "temperature" not in out.columns and {"temp_min", "temp_max"}.issubset(out.columns):
        out["temperature"] = (out["temp_min"] + out["temp_max"]) / 2.0

    if "soil_moisture" not in out.columns:
        if "lswi" in out.columns:
            out["soil_moisture"] = out["lswi"].clip(0.05, 0.70)
        else:
            # Coarse fallback if soil moisture is unavailable.
            out["soil_moisture"] = (
                0.15
                + 0.02 * out["rainfall"].fillna(0)
                + 0.001 * out["humidity"].fillna(0)
            ).clip(0.05, 0.70)

    out["day_of_year"] = out["date"].dt.dayofyear
    out["month"] = out["date"].dt.month

    regional_rain_mean = out.groupby("region")["rainfall"].transform("mean")
    out["rainfall_deviation"] = out["rainfall"] - regional_rain_mean

    regional_temp_mean = out.groupby("region")["temperature"].transform("mean")
    out["temperature_anomaly"] = out["temperature"] - regional_temp_mean

    out["heatwave_flag"] = (out["temperature"] >= 36.0).astype(int)
    out["humidity_stress"] = np.where(out["humidity"] > 80, 1, 0)
    out["soil_moisture_stress"] = np.where(out["soil_moisture"] < 0.20, 1, 0)

    out["ndvi_7d_ma"] = out.groupby("region")["ndvi"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    out["pest_pressure_index"] = (
        0.45 * out["aphid_population"]
        + 0.35 * out["rust_severity"]
        + 0.20 * out["armyworm_presence"]
    )

    return out


def build_feature_columns() -> List[str]:
    """Returns model input columns used across all model families."""
    return BASE_FEATURES + [
        "day_of_year",
        "month",
        "rainfall_deviation",
        "temperature_anomaly",
        "heatwave_flag",
        "humidity_stress",
        "soil_moisture_stress",
        "ndvi_7d_ma",
        "pest_pressure_index",
    ]

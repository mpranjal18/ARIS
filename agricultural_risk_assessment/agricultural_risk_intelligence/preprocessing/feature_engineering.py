from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=2).std().fillna(0)


def build_features(
    weather_df: pd.DataFrame,
    ndvi_df: pd.DataFrame,
    soil_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    df = weather_df.merge(ndvi_df, on="date", how="left")
    df = df.merge(soil_df, on="date", how="left")
    df = df.merge(price_df, on="date", how="left")

    tavg = (df["tmax_c"] + df["tmin_c"]) / 2
    rain_mean = _rolling_mean(df["rainfall_mm"], 30)
    rain_dev = (df["rainfall_mm"] - rain_mean) / (rain_mean + 1e-6) * 100

    heatwave = (df["tmax_c"] > 32).astype(int)
    heatwave_index = heatwave.rolling(7, min_periods=1).sum()

    temp_anom = tavg - _rolling_mean(tavg, 30)
    humidity_stress = np.clip((df["humidity_pct"] - 70) / 30, 0, 1)

    ndvi_anom = df["ndvi"] - _rolling_mean(df["ndvi"], 30)
    soil_moisture_index = df["soil_moisture_index"].fillna(method="ffill").fillna(0)

    price_volatility = _rolling_std(df["price_inr_per_qtl"], 14)

    pest_suitability = 1 / (1 + np.exp(-(0.08 * (df["humidity_pct"] - 65) + 0.1 * (df["tmax_c"] - 30))))

    features = pd.DataFrame(
        {
            "date": df["date"],
            "rainfall_mm": df["rainfall_mm"],
            "tmax_c": df["tmax_c"],
            "tmin_c": df["tmin_c"],
            "humidity_pct": df["humidity_pct"],
            "ndvi": df["ndvi"],
            "soil_moisture_index": soil_moisture_index,
            "price_inr_per_qtl": df["price_inr_per_qtl"],
            "rainfall_deviation_pct": rain_dev,
            "heatwave_index": heatwave_index,
            "temperature_anomaly": temp_anom,
            "humidity_stress_index": humidity_stress,
            "ndvi_anomaly": ndvi_anom,
            "price_volatility": price_volatility,
            "pest_suitability_score": pest_suitability,
            "is_forecast": df.get("is_forecast", False),
        }
    )

    numeric_cols = features.select_dtypes(include="number").columns
    features[numeric_cols] = features[numeric_cols].fillna(0)
    return features


if __name__ == "__main__":
    print("Feature engineering module")

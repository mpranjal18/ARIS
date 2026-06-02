from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_mandi_prices(days: int = 240, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime.today().date(), periods=days, freq="D")

    rows: list[dict] = []
    crops = ["Wheat", "Soybean"]
    regions = ["Bhopal", "Sehore", "Ashta"]
    base_price = {"Wheat": 2250, "Soybean": 4950}
    season_amp = {"Wheat": 170, "Soybean": 210}
    region_bias = {"Bhopal": 0, "Sehore": 45, "Ashta": -25}

    for crop in crops:
        for region in regions:
            t = np.linspace(0, 4 * np.pi, days)
            trend = np.linspace(-60, 210, days)
            seasonal = season_amp[crop] * np.sin(t + (0.3 if crop == "Wheat" else 0.8))
            noise = rng.normal(0, 48, days)
            series = np.clip(base_price[crop] + region_bias[region] + trend + seasonal + noise, 1200, 7500)

            for i, dt in enumerate(dates):
                rows.append(
                    {
                        "date": dt,
                        "region": region,
                        "crop": crop,
                        "price_inr_qtl": round(float(series[i]), 2),
                    }
                )

    return pd.DataFrame(rows)


def forecast_prices(price_df: pd.DataFrame, crop: str = "Wheat", region: str | None = None, horizon: int = 30) -> pd.DataFrame:
    crop_df = price_df[price_df["crop"] == crop].copy()
    if region is not None and "region" in crop_df.columns:
        crop_df = crop_df[crop_df["region"] == region].copy()
    crop_df = crop_df.groupby("date", as_index=False)["price_inr_qtl"].mean().sort_values("date").reset_index(drop=True)
    x = np.arange(len(crop_df)).reshape(-1, 1)
    y = crop_df["price_inr_qtl"].to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    future_x = np.arange(len(crop_df), len(crop_df) + horizon).reshape(-1, 1)
    trend_pred = model.predict(future_x)

    seasonality = np.sin(np.linspace(0, 1.3 * np.pi, horizon)) * 45
    forecast_values = np.clip(trend_pred + seasonality, 1200, 3800)

    last_date = crop_df["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    out = pd.DataFrame(
        {
            "date": future_dates,
            "crop": crop,
            "region": region if region is not None else "All",
            "predicted_price": forecast_values.round(2),
        }
    )
    return out


def selling_window_recommendation(forecast_df: pd.DataFrame) -> dict:
    top = forecast_df.sort_values("predicted_price", ascending=False).head(5).sort_values("date")
    start_dt = top["date"].min()
    end_dt = top["date"].max()

    expected_price = float(top["predicted_price"].mean())
    baseline_price = float(forecast_df["predicted_price"].iloc[0])
    expected_gain = expected_price - baseline_price

    signal = "Sell" if expected_gain > 0 else "Hold"
    message = f"Recommended selling period: {start_dt.strftime('%d %b')} - {end_dt.strftime('%d %b')}"

    return {
        "signal": signal,
        "message": message,
        "expected_price": round(expected_price, 2),
        "baseline_price": round(baseline_price, 2),
        "expected_gain": round(expected_gain, 2),
    }

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from models.model_trainer import forecast_next_7_days, train_and_compare_models
from utils.market_intelligence import forecast_prices, generate_mandi_prices, selling_window_recommendation
from utils.ndvi_simulation import forecast_ndvi_by_crop, simulate_crop_ndvi
from utils.data_generator import generate_synthetic_data
from utils.risk_calculator import add_targets_for_training, compute_risk_scores


def run_pipeline(days: int = 90, seed: int = 42, save: bool = True) -> Dict[str, pd.DataFrame]:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_df = generate_synthetic_data(days=days, seed=seed)
    scored_df = compute_risk_scores(raw_df)
    train_df = add_targets_for_training(scored_df, seed=seed)

    perf_df, best = train_and_compare_models(train_df, seed=seed)
    forecast_df = forecast_next_7_days(train_df, best.model)
    crop_ndvi_df = simulate_crop_ndvi(days=max(150, days), seed=seed + 101)
    ndvi_forecast_df = forecast_ndvi_by_crop(crop_ndvi_df, horizon=14)
    mandi_price_df = generate_mandi_prices(days=max(200, days * 2), seed=seed + 202)
    price_forecast_df = forecast_prices(mandi_price_df, crop="Wheat", horizon=30)
    selling_window = selling_window_recommendation(price_forecast_df)
    selling_window_df = pd.DataFrame([selling_window])

    if save:
        train_df.to_csv(data_dir / "synthetic_risk_data.csv", index=False)
        perf_df.to_csv(data_dir / "model_performance.csv", index=False)
        forecast_df.to_csv(data_dir / "risk_forecast_7d.csv", index=False)
        best.feature_importance.to_csv(data_dir / "feature_importance.csv", index=False)
        crop_ndvi_df.to_csv(data_dir / "crop_ndvi.csv", index=False)
        ndvi_forecast_df.to_csv(data_dir / "ndvi_forecast.csv", index=False)
        mandi_price_df.to_csv(data_dir / "mandi_prices.csv", index=False)
        price_forecast_df.to_csv(data_dir / "price_forecast.csv", index=False)
        selling_window_df.to_csv(data_dir / "selling_window.csv", index=False)

    return {
        "data": train_df,
        "performance": perf_df,
        "forecast": forecast_df,
        "feature_importance": best.feature_importance,
        "best_model": pd.DataFrame([{"best_model": best.model_name}]),
        "crop_ndvi": crop_ndvi_df,
        "ndvi_forecast": ndvi_forecast_df,
        "mandi_prices": mandi_price_df,
        "price_forecast": price_forecast_df,
        "selling_window": selling_window_df,
    }


if __name__ == "__main__":
    out = run_pipeline()
    print("Pipeline completed.")
    print(out["performance"])

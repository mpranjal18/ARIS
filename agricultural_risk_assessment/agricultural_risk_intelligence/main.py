from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_ingestion.mandi_price_loader import load_mandi_prices
from data_ingestion.ndvi_loader import load_ndvi_data
from data_ingestion.soil_moisture_loader import compute_soil_moisture
from data_ingestion.weather_loader import load_weather_data
from models.climate_risk_model import ClimateRiskModel
from models.pest_risk_model import PestRiskModel
from models.risk_fusion import fuse_risks
from preprocessing.clean_data import clean_time_series
from preprocessing.feature_engineering import build_features


def run_pipeline(base_dir: str | Path | None = None) -> pd.DataFrame:
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    weather = load_weather_data(raw_dir)
    ndvi = load_ndvi_data(raw_dir)
    prices = load_mandi_prices(raw_dir)
    soil = compute_soil_moisture(weather, raw_dir)

    weather = clean_time_series(weather)
    ndvi = clean_time_series(ndvi)
    prices = clean_time_series(prices)
    soil = clean_time_series(soil)

    features = build_features(weather, ndvi, soil, prices)

    climate_model = ClimateRiskModel(seq_len=14)
    pest_model = PestRiskModel()

    climate_risk = climate_model.fit_predict(features)
    pest_risk = pest_model.fit_predict(features)

    risk_df = fuse_risks(climate_risk, pest_risk)
    risk_df.insert(0, "date", features["date"].values)

    features.to_csv(processed_dir / "features.csv", index=False)
    risk_df.to_csv(processed_dir / "risk_results.csv", index=False)
    return risk_df


if __name__ == "__main__":
    run_pipeline()

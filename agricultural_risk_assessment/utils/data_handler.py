"""Data ingestion and synthetic dataset generation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from config.config import (
    DATA_DIR,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    PRIMARY_REGIONS,
    SECONDARY_REGIONS,
    SEED,
)


@dataclass
class DatasetBundle:
    weather: pd.DataFrame
    ndvi: pd.DataFrame
    pest: pd.DataFrame
    price: pd.DataFrame


class DataHandler:
    """Handles data generation, loading, and persistence for the project."""

    def __init__(self, data_dir: str = DATA_DIR, seed: int = SEED) -> None:
        self.data_dir = data_dir
        self.regions = PRIMARY_REGIONS + SECONDARY_REGIONS
        self.rng = np.random.default_rng(seed)
        os.makedirs(self.data_dir, exist_ok=True)

    def _date_range(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        return pd.date_range(start=start_date, end=end_date, freq="D")

    def generate_synthetic_weather_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
    ) -> pd.DataFrame:
        """Generates synthetic weather variables used for climate risk prediction."""
        date_range = self._date_range(start_date, end_date)
        records = []
        for region in self.regions:
            day_idx = np.arange(len(date_range))
            seasonal = np.sin(2 * np.pi * day_idx / 365.25)
            rainfall = np.clip(6 + 5 * seasonal + self.rng.normal(0, 3, len(date_range)), 0, None)
            temp = 24 + 9 * seasonal + self.rng.normal(0, 2, len(date_range))
            humidity = np.clip(58 + 18 * seasonal + self.rng.normal(0, 6, len(date_range)), 20, 100)
            soil_moisture = np.clip(0.32 + 0.12 * seasonal + self.rng.normal(0, 0.05, len(date_range)), 0.05, 0.7)

            for i, dt in enumerate(date_range):
                records.append(
                    {
                        "date": dt,
                        "region": region,
                        "rainfall": round(float(rainfall[i]), 3),
                        "temperature": round(float(temp[i]), 3),
                        "humidity": round(float(humidity[i]), 3),
                        "soil_moisture": round(float(soil_moisture[i]), 4),
                    }
                )
        return pd.DataFrame(records)

    def generate_synthetic_ndvi_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
    ) -> pd.DataFrame:
        """Generates synthetic NDVI signals for vegetation health tracking."""
        date_range = self._date_range(start_date, end_date)
        records = []
        for region in self.regions:
            day_idx = np.arange(len(date_range))
            seasonal = np.sin(2 * np.pi * (day_idx - 45) / 365.25)
            ndvi = np.clip(0.5 + 0.22 * seasonal + self.rng.normal(0, 0.04, len(date_range)), 0.1, 0.95)
            for i, dt in enumerate(date_range):
                records.append({"date": dt, "region": region, "ndvi": round(float(ndvi[i]), 4)})
        return pd.DataFrame(records)

    def generate_synthetic_pest_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
    ) -> pd.DataFrame:
        """Generates synthetic pest and disease indicators."""
        date_range = self._date_range(start_date, end_date)
        records = []
        for region in self.regions:
            day_idx = np.arange(len(date_range))
            seasonal = np.sin(2 * np.pi * (day_idx - 20) / 365.25)
            aphid = np.clip(0.25 + 0.20 * seasonal + self.rng.normal(0, 0.05, len(date_range)), 0, 1)
            rust = np.clip(0.20 + 0.18 * seasonal + self.rng.normal(0, 0.05, len(date_range)), 0, 1)
            armyworm = np.clip(0.15 + 0.12 * seasonal + self.rng.normal(0, 0.04, len(date_range)), 0, 1)
            for i, dt in enumerate(date_range):
                records.append(
                    {
                        "date": dt,
                        "region": region,
                        "aphid_population": round(float(aphid[i]), 4),
                        "rust_severity": round(float(rust[i]), 4),
                        "armyworm_presence": round(float(armyworm[i]), 4),
                    }
                )
        return pd.DataFrame(records)

    def generate_synthetic_price_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
    ) -> pd.DataFrame:
        """Generates synthetic wheat mandi price data for contextual analysis."""
        date_range = self._date_range(start_date, end_date)
        records = []
        for region in self.regions:
            trend = np.linspace(2200, 2650, len(date_range))
            seasonal = 120 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25)
            noise = self.rng.normal(0, 40, len(date_range))
            price = np.clip(trend + seasonal + noise, 1500, None)
            for i, dt in enumerate(date_range):
                records.append({"date": dt, "region": region, "wheat_price": round(float(price[i]), 2)})
        return pd.DataFrame(records)

    def save_data(self, data: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.data_dir, filename)
        data.to_csv(path, index=False)
        return path

    def load_data(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        return pd.read_csv(path, parse_dates=["date"])

    def load_or_generate_bundle(self) -> DatasetBundle:
        """Loads existing CSVs or generates and stores synthetic datasets."""
        file_map: Dict[str, str] = {
            "weather": "weather_data.csv",
            "ndvi": "ndvi_data.csv",
            "pest": "pest_data.csv",
            "price": "price_data.csv",
        }
        datasets = {}
        try:
            for key, filename in file_map.items():
                datasets[key] = self.load_data(filename)
        except FileNotFoundError:
            datasets = {
                "weather": self.generate_synthetic_weather_data(),
                "ndvi": self.generate_synthetic_ndvi_data(),
                "pest": self.generate_synthetic_pest_data(),
                "price": self.generate_synthetic_price_data(),
            }
            for key, filename in file_map.items():
                self.save_data(datasets[key], filename)
        return DatasetBundle(**datasets)
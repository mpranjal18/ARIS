from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _generate_ndvi(start_date: pd.Timestamp, days: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=days, freq="D")
    seasonal = 0.45 + 0.12 * np.sin(np.linspace(0, 2 * np.pi, days))
    noise = rng.normal(0, 0.02, days)
    ndvi = np.clip(seasonal + noise, 0.2, 0.8)
    return pd.DataFrame({"date": dates, "ndvi": ndvi.round(4)})


def load_ndvi_data(
    raw_dir: str | Path,
    days_history: int = 180,
    file_name: str = "ndvi_data.csv",
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file_name

    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["date"])
    else:
        start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_history - 1)
        df = _generate_ndvi(start_date, days_history, seed=13)
        df.to_csv(file_path, index=False)

    return df


if __name__ == "__main__":
    df = load_ndvi_data(Path("../data/raw"))
    print(df.tail())

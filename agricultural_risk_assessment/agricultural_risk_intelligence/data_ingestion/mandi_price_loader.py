from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _generate_prices(start_date: pd.Timestamp, days: int, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=days, freq="D")
    trend = 2100 + np.linspace(-30, 40, days)
    noise = rng.normal(0, 35, days)
    price = np.clip(trend + noise, 1700, 2600)
    return pd.DataFrame({"date": dates, "price_inr_per_qtl": price.round(2)})


def load_mandi_prices(
    raw_dir: str | Path,
    days_history: int = 180,
    file_name: str = "mandi_price_data.csv",
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file_name

    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["date"])
    else:
        start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_history - 1)
        df = _generate_prices(start_date, days_history, seed=23)
        df.to_csv(file_path, index=False)

    return df


if __name__ == "__main__":
    df = load_mandi_prices(Path("../data/raw"))
    print(df.tail())

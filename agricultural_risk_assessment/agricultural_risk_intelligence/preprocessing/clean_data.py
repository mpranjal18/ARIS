from __future__ import annotations

import pandas as pd


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].interpolate(limit_direction="both")
    out[numeric_cols] = out[numeric_cols].fillna(method="bfill").fillna(method="ffill")
    return out


def clean_time_series(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)
    out = clean_numeric(out)
    return out


if __name__ == "__main__":
    sample = pd.DataFrame({"date": ["2024-01-02", "2024-01-01"], "x": [1.0, None]})
    print(clean_time_series(sample))

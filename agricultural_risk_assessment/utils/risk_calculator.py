"""Risk score utilities and label mapping on a 0-100 scale."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import RISK_LEVEL_THRESHOLDS, RISK_WEIGHTS


def _minmax_0_100(series: pd.Series) -> pd.Series:
    min_val = float(series.min())
    max_val = float(series.max())
    if max_val - min_val < 1e-9:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return ((series - min_val) / (max_val - min_val) * 100).clip(0, 100)


def add_risk_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adds climate, pest, and combined risk targets used for supervised learning."""
    out = df.copy()

    climate_raw = (
        0.25 * out["rainfall_deviation"].abs()
        + 0.25 * out["temperature_anomaly"].abs()
        + 0.15 * out["heatwave_flag"]
        + 0.10 * out["humidity_stress"]
        + 0.10 * out["soil_moisture_stress"]
        + 0.15 * (1 - out["ndvi"])
    )
    pest_raw = (
        0.45 * out["aphid_population"]
        + 0.35 * out["rust_severity"]
        + 0.20 * out["armyworm_presence"]
    )

    out["climate_risk_score"] = _minmax_0_100(climate_raw)
    out["pest_disease_risk_score"] = _minmax_0_100(pest_raw)

    out["total_risk_score"] = (
        RISK_WEIGHTS["climate"] * out["climate_risk_score"]
        + RISK_WEIGHTS["pest_disease"] * out["pest_disease_risk_score"]
    ).clip(0, 100)

    out["climate_risk_level"] = out["climate_risk_score"].map(risk_level)
    out["pest_disease_risk_level"] = out["pest_disease_risk_score"].map(risk_level)
    out["total_risk_level"] = out["total_risk_score"].map(risk_level)
    return out


def risk_level(score: float) -> str:
    """Maps a score to Low/Medium/High buckets."""
    if score < RISK_LEVEL_THRESHOLDS["low"]:
        return "Low"
    if score < RISK_LEVEL_THRESHOLDS["medium"]:
        return "Medium"
    return "High"

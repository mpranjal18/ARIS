from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series - low) / (high - low + 1e-6)).clip(0, 1)


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rain_stress = (_normalize(30 - out["rainfall"], 0, 30) * 100).clip(0, 100)
    temp_stress = (_normalize(out["temperature"], 18, 38) * 100).clip(0, 100)
    humidity_stress = (_normalize(abs(out["humidity"] - 60), 0, 35) * 100).clip(0, 100)
    ndvi_stress = (_normalize(0.75 - out["ndvi"], 0, 0.5) * 100).clip(0, 100)

    climate_risk = 0.35 * rain_stress + 0.30 * temp_stress + 0.20 * humidity_stress + 0.15 * ndvi_stress
    pest_risk = (0.50 * out["pest_level"] + 0.25 * temp_stress + 0.15 * humidity_stress + 0.10 * ndvi_stress).clip(0, 100)
    total_risk = (0.6 * climate_risk + 0.4 * pest_risk).clip(0, 100)

    out["climate_risk"] = climate_risk.round(2)
    out["pest_risk"] = pest_risk.round(2)
    out["total_risk"] = total_risk.round(2)

    out["outbreak_flag"] = ((out["pest_level"] > 70) & (out["humidity"] > 70) & (out["temperature"] > 30)).astype(int)
    return out


def risk_band(score: float) -> str:
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Moderate"
    return "High"


def risk_color(score: float) -> str:
    if score <= 30:
        return "#27ae60"
    if score <= 60:
        return "#f1c40f"
    return "#e74c3c"


def add_targets_for_training(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    noisy_target = (0.7 * out["total_risk"] + 0.3 * out["pest_level"] + rng.normal(0, 5, len(out))).clip(0, 100)
    out["risk_target"] = noisy_target.round(2)
    out["risk_label"] = pd.cut(out["risk_target"], bins=[-1, 30, 60, 100], labels=[0, 1, 2]).astype(int)
    return out

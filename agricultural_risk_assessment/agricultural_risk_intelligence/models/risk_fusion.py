from __future__ import annotations

import numpy as np
import pandas as pd


def fuse_risks(climate_risk: pd.Series, pest_risk: pd.Series) -> pd.DataFrame:
    overall = 0.6 * climate_risk + 0.4 * pest_risk
    overall = np.clip(overall, 0, 100)

    def _category(value: float) -> str:
        if value <= 30:
            return "Low"
        if value <= 60:
            return "Moderate"
        return "High"

    categories = overall.apply(_category)
    return pd.DataFrame(
        {
            "climate_risk": climate_risk.round(2),
            "pest_risk": pest_risk.round(2),
            "overall_risk": overall.round(2),
            "risk_category": categories,
        }
    )

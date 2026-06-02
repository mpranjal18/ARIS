from __future__ import annotations

from typing import List

import pandas as pd

from utils.risk_calculator import risk_band


def generate_ai_insights(filtered_df: pd.DataFrame) -> List[str]:
    insights: List[str] = []
    if filtered_df.empty:
        return ["No data available for current filter selection."]

    latest = filtered_df.sort_values("date").iloc[-1]
    avg_total = filtered_df["total_risk"].mean()
    ndvi_drop = filtered_df["ndvi"].diff().lt(-0.05).sum()
    pest_spike = filtered_df["pest_level"].diff().gt(12).sum()

    insights.append(
        f"Current overall risk is {latest['total_risk']:.1f} ({risk_band(float(latest['total_risk']))}) in {latest['region']}."
    )
    insights.append(
        f"Average total risk across selected window is {avg_total:.1f}; climate risk contribution is {filtered_df['climate_risk'].mean():.1f}."
    )

    if ndvi_drop > 0:
        insights.append(f"Detected {int(ndvi_drop)} meaningful NDVI drops. Vegetation stress may be increasing.")
    else:
        insights.append("NDVI trend is stable with no significant drop events detected.")

    if pest_spike > 0:
        insights.append(f"Detected {int(pest_spike)} pest-level spikes. Keep surveillance active for early outbreak response.")

    if (filtered_df["outbreak_flag"] == 1).any():
        insights.append("Outbreak conditions detected (high humidity + high temperature + high pest level).")

    return insights

from __future__ import annotations

from typing import List

import pandas as pd


def generate_advanced_insights(risk_df: pd.DataFrame, ndvi_df: pd.DataFrame, price_forecast_df: pd.DataFrame) -> List[str]:
    insights: List[str] = []

    latest_risk = risk_df.sort_values("date").iloc[-1]
    prev_risk = risk_df.sort_values("date").iloc[-2] if len(risk_df) > 1 else latest_risk

    if float(prev_risk["ndvi"]) - float(latest_risk["ndvi"]) > 0.03:
        insights.append("NDVI declining, which indicates increasing crop stress in selected region.")

    if float(latest_risk["humidity"]) > 72 and float(latest_risk["temperature"]) > 31:
        insights.append("Pest likelihood is increasing due to high humidity and warm temperature.")

    if float(price_forecast_df["predicted_price"].iloc[-1]) > float(price_forecast_df["predicted_price"].iloc[0]) + 25:
        insights.append("Mandi prices are expected to rise; delayed selling may improve returns.")

    if not insights:
        insights.append("Risk and market signals are stable; continue regular monitoring and phased selling.")

    return insights

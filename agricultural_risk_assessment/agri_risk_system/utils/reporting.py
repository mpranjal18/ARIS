from __future__ import annotations

import pandas as pd


def build_report_csv(
    latest: pd.Series,
    insight: str,
    recommendation: str,
    lang: str,
    price_context: dict | None = None,
) -> bytes:
    report = pd.DataFrame(
        [
            {"metric": "language", "value": lang},
            {"metric": "climate_risk", "value": round(float(latest["climate_risk"]), 2)},
            {"metric": "pest_risk", "value": round(float(latest["pest_risk"]), 2)},
            {"metric": "forecast_risk", "value": round(float(latest["total_risk"]), 2)},
            {"metric": "insight", "value": insight},
            {"metric": "recommendation", "value": recommendation},
        ]
    )
    if price_context:
        price_rows = pd.DataFrame(
            [{"metric": str(k), "value": str(v)} for k, v in price_context.items()]
        )
        report = pd.concat([report, price_rows], ignore_index=True)
    return report.to_csv(index=False).encode("utf-8")

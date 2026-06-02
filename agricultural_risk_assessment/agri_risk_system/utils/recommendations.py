from __future__ import annotations

from typing import List


def get_recommendations(total_risk: float, climate_risk: float, pest_risk: float) -> List[str]:
    actions: List[str] = []

    if climate_risk > 60:
        actions.append("Schedule irrigation balancing and moisture conservation to reduce temperature-water stress.")
        actions.append("Use mulching in vulnerable plots to retain soil moisture and limit heat shock.")

    if pest_risk > 60:
        actions.append("Increase field scouting frequency to every 48 hours for early pest hotspot detection.")
        actions.append("Deploy integrated pest management with targeted bio-control where infestation signs appear.")

    if total_risk <= 30:
        actions.append("Risk is low; continue routine monitoring and maintain preventive agronomic practices.")
    elif total_risk <= 60:
        actions.append("Risk is moderate; monitor weather-pest signals daily and prepare contingency inputs.")
    else:
        actions.append("Risk is high; initiate protective interventions immediately and coordinate advisory support.")

    return actions

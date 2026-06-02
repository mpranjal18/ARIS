from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.common_data import load_payload
from utils.i18n import ensure_language_state, t
from utils.style import apply_glassmorphism_css
from visualization.advanced_charts import climate_pest_risk_chart, rainfall_heatwave_chart


st.set_page_config(page_title="Agri Risk Intelligence System (ARIS)", page_icon="🐛", layout="wide")
ensure_language_state()
apply_glassmorphism_css()

payload = load_payload(seed=42, days=120)
df = payload["data"].copy()

st.markdown(f"## 🐛 {t('pest_climate_page')}")

regions = sorted(df["region"].unique())
selected = st.multiselect(t("region"), regions, default=regions)
filtered = df[df["region"].isin(selected)].copy()

if filtered.empty:
    st.error("No data available for selected regions.")
    st.stop()

st.plotly_chart(climate_pest_risk_chart(filtered), use_container_width=True)
st.plotly_chart(rainfall_heatwave_chart(filtered), use_container_width=True)

with st.expander("Correlation Insights", expanded=True):
    daily = filtered.groupby("date", as_index=False).agg(
        rainfall=("rainfall", "mean"),
        heatwave=("temperature", lambda x: (x > 32).mean() * 100),
        climate=("climate_risk", "mean"),
        pest=("pest_risk", "mean"),
    )
    corr_rh = daily["rainfall"].corr(daily["heatwave"])
    corr_cp = daily["climate"].corr(daily["pest"])

    st.write(f"Rainfall vs Heatwave correlation: {corr_rh:.2f}")
    st.write(f"Climate Risk vs Pest Risk correlation: {corr_cp:.2f}")

    latest = filtered.sort_values("date").iloc[-1]
    if float(latest["pest_risk"]) > 65 and float(latest["humidity"]) > 70:
        st.error("Alert: Pest risk is elevated under humid conditions.")
        st.warning("🔥 Heatwave detected. Increase watering frequency.")
    elif float(latest["climate_risk"]) > 60:
        st.warning("Alert: Climate risk is elevated due to weather stress.")
        st.info("AI Recommendation: Increase irrigation checks and protect vulnerable plots.")
    else:
        st.success("Current pest-climate interaction remains manageable.")

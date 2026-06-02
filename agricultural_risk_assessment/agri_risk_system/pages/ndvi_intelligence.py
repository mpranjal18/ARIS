from __future__ import annotations

import streamlit as st

from utils.common_data import load_payload
from utils.i18n import ensure_language_state, t
from utils.style import apply_glassmorphism_css
from visualization.advanced_charts import ndvi_research_chart


st.set_page_config(page_title="Agri Risk Intelligence System (ARIS)", page_icon="🌿", layout="wide")
ensure_language_state()
apply_glassmorphism_css()

payload = load_payload(seed=42, days=150)
ndvi_df = payload["crop_ndvi"].copy()
ndvi_forecast = payload["ndvi_forecast"].copy()

st.markdown(f"## 🌿 {t('ndvi_page')}")

crop_focus = st.selectbox("Crop Focus", ["Wheat", "Maize"], index=0)
fig = ndvi_research_chart(ndvi_df, selected_crop=crop_focus)
st.plotly_chart(fig, use_container_width=True)

with st.expander("NDVI Trend Analysis", expanded=True):
    series = ndvi_df[ndvi_df["crop"] == crop_focus].sort_values("date")
    drop_events = int((series["ndvi"].diff() < -0.03).sum())
    avg_ndvi = float(series["ndvi"].mean())
    st.write(f"Average NDVI ({crop_focus}): {avg_ndvi:.3f}")
    if drop_events > 0:
        st.warning(f"Detected {drop_events} abnormal NDVI drop points. Crop stress monitoring recommended.")
        st.error("⚠️ Vegetation stress detected. Consider irrigation.")
    else:
        st.success("No abnormal NDVI drop found in current observation period.")

with st.expander("NDVI Forecast", expanded=False):
    fc = ndvi_forecast[ndvi_forecast["crop"] == crop_focus].sort_values("date")
    st.line_chart(fc.set_index("date")["predicted_ndvi"])
    if len(fc) > 1 and float(fc["predicted_ndvi"].iloc[-1]) < float(fc["predicted_ndvi"].iloc[0]):
        st.warning("Insight: Forecast NDVI is trending downward, indicating possible crop stress.")
    else:
        st.info("Insight: Forecast NDVI is stable to improving.")

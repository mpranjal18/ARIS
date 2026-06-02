from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is importable when Streamlit executes from dashboard context.
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import run_pipeline
from visualization.plots import (
    plot_ndvi_series,
    plot_overall_risk,
    plot_price_trend,
    plot_rainfall_heatwave,
    plot_risk_comparison,
)


st.set_page_config(page_title="ARIS Phase I", layout="wide")

PROCESSED_DIR = BASE_DIR / "data" / "processed"

st.title("Agri Risk Intelligence System (ARIS) - Phase I")

if st.button("Run Pipeline"):
    run_pipeline(base_dir=BASE_DIR)
    st.success("Pipeline executed.")

risk_path = PROCESSED_DIR / "risk_results.csv"
features_path = PROCESSED_DIR / "features.csv"

if not risk_path.exists() or not features_path.exists():
    st.warning("Run the pipeline to generate results.")
    st.stop()

features = pd.read_csv(features_path, parse_dates=["date"])
risk_df = pd.read_csv(risk_path, parse_dates=["date"])

latest = risk_df.iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Climate Risk", f"{latest['climate_risk']:.1f}")
col2.metric("Pest Risk", f"{latest['pest_risk']:.1f}")
col3.metric("Overall Risk", f"{latest['overall_risk']:.1f}")

st.subheader(f"Risk Category: {latest['risk_category']}")

fig1 = plot_ndvi_series(features)
fig2 = plot_rainfall_heatwave(features)
fig3 = plot_risk_comparison(risk_df)
fig4 = plot_price_trend(features)
fig5 = plot_overall_risk(risk_df)

st.pyplot(fig1)
st.pyplot(fig2)
st.plotly_chart(fig3, use_container_width=True)
st.plotly_chart(fig4, use_container_width=True)
st.plotly_chart(fig5, use_container_width=True)

from __future__ import annotations

import streamlit as st

from utils.common_data import load_payload
from utils.i18n import ensure_language_state, t
from utils.recommendations import get_recommendations
from utils.style import apply_glassmorphism_css
from visualization.charts import feature_importance_chart, performance_bar_chart


st.set_page_config(page_title="Agri Risk Intelligence System (ARIS)", page_icon="📊", layout="wide")
ensure_language_state()
apply_glassmorphism_css()

payload = load_payload(seed=42, days=120)
perf_df = payload["performance"].copy()
imp_df = payload["feature_importance"].copy()
risk_df = payload["data"].copy()
best_model = str(payload["best_model"].iloc[0]["best_model"])

st.markdown(f"## 📊 {t('model_page')}")
st.caption("Model performance and AI-driven action advice")

latest = risk_df.sort_values("date").iloc[-1]
rec = get_recommendations(float(latest["total_risk"]), float(latest["climate_risk"]), float(latest["pest_risk"]))

c1, c2 = st.columns(2)
with c1:
    st.metric("Best Model", best_model)
with c2:
    st.metric("Best RMSE", f"{float(perf_df.iloc[0]['RMSE']):.4f}")

st.plotly_chart(performance_bar_chart(perf_df), use_container_width=True)
st.plotly_chart(feature_importance_chart(imp_df), use_container_width=True)

with st.expander("AI Recommendation", expanded=True):
    if float(latest["total_risk"]) > 60:
        st.error(rec[-1])
    elif float(latest["total_risk"]) > 30:
        st.warning(rec[-1])
    else:
        st.success(rec[-1])

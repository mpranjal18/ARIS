from __future__ import annotations

import streamlit as st

from utils.i18n import LANGUAGES, ensure_language_state, set_language, t
from utils.style import apply_glassmorphism_css


st.set_page_config(page_title="Agri Risk Intelligence System (ARIS)", page_icon="⚙️", layout="wide")
ensure_language_state()
apply_glassmorphism_css()

st.markdown(f"## ⚙️ {t('settings_page')}")

current = st.session_state.get("aris_language", "English")
choice = st.radio(t("language"), LANGUAGES, index=LANGUAGES.index(current), horizontal=True)
set_language(choice)

st.markdown("### Preview")

c1, c2 = st.columns(2)
c1.metric(t("risk_score"), "68.4")
c2.metric(t("climate_risk"), "61.2")

st.info(f"Language switched to: {choice}")
st.caption("All pages dynamically read shared language state from the Settings page.")

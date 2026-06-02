from __future__ import annotations

import pandas as pd
import streamlit as st

from main import run_pipeline


@st.cache_data(show_spinner=False)
def load_payload(seed: int = 42, days: int = 120) -> dict[str, pd.DataFrame]:
    return run_pipeline(days=days, seed=seed, save=True)

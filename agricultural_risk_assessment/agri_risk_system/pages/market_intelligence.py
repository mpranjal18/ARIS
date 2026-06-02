from __future__ import annotations

import streamlit as st

from utils.common_data import load_payload
from utils.i18n import ensure_language_state, t
from utils.market_intelligence import forecast_prices, selling_window_recommendation
from utils.style import apply_glassmorphism_css
from visualization.advanced_charts import live_mandi_comparison_chart, mandi_price_chart, price_forecast_chart


st.set_page_config(page_title="Agri Risk Intelligence System (ARIS)", page_icon="💰", layout="wide")
ensure_language_state()
apply_glassmorphism_css()

payload = load_payload(seed=42, days=140)
price_df = payload["mandi_prices"].copy()
base_forecast = payload["price_forecast"].copy()
window_df = payload["selling_window"].copy()

if "region" not in price_df.columns:
    price_df = price_df.copy()
    price_df["region"] = "All"

if "crop" in price_df.columns and "Soybean" not in set(price_df["crop"].unique()) and "Maize" in set(price_df["crop"].unique()):
    price_df = price_df.copy()
    price_df["crop"] = price_df["crop"].replace({"Maize": "Soybean"})

st.markdown(f"## 💰 {t('market_page')}")

crop = st.selectbox("Crop", ["Wheat", "Soybean"], index=0)
if crop == "Wheat":
    fc = base_forecast.copy()
    rec = window_df.iloc[0].to_dict()
else:
    fc = forecast_prices(price_df, crop="Soybean", horizon=30)
    rec = selling_window_recommendation(fc)

hist = price_df[price_df["crop"] == crop].sort_values("date")
current_price = float(hist["price_inr_qtl"].iloc[-1])
end_price = float(fc["predicted_price"].iloc[-1])

k1, k2, k3 = st.columns(3)
k1.metric(t("current_price"), f"INR {current_price:.2f}/qtl")
k2.metric("30-Day Forecast", f"INR {end_price:.2f}/qtl", delta=f"{end_price - current_price:.2f}")
k3.metric("Expected Gain", f"INR {float(rec['expected_gain']):.2f}/qtl")

st.plotly_chart(live_mandi_comparison_chart(price_df), use_container_width=True)
st.plotly_chart(mandi_price_chart(price_df, crop=crop), use_container_width=True)
st.plotly_chart(price_forecast_chart(hist, fc, crop=crop), use_container_width=True)

st.success(f"📈 {t('best_sell_text')}")
if float(rec["expected_gain"]) > 0:
    st.info(f"{t('wait_days')} | Expected avg price: INR {float(rec['expected_price']):.2f}/qtl")
else:
    st.warning(f"{t('sell_now')} | Signal: Sell now")

with st.expander("Profit Estimation", expanded=True):
    qty = st.slider("Estimated quantity (quintal)", 10, 500, 100, step=10)
    profit_now = current_price * qty
    profit_window = float(rec["expected_price"]) * qty
    extra = profit_window - profit_now
    st.write(f"Sell now: INR {profit_now:,.0f}")
    st.write(f"Sell in recommended window: INR {profit_window:,.0f}")
    if extra > 0:
        st.success(f"Estimated additional profit: INR {extra:,.0f}")
    else:
        st.warning(f"Estimated difference: INR {extra:,.0f}")

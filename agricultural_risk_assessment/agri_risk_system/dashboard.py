from __future__ import annotations

from datetime import datetime

import streamlit as st

from utils.ai_insights import generate_advanced_insights
from utils.common_data import load_payload
from utils.i18n import LANGUAGES, ensure_language_state, get_language, set_language, t
from utils.market_intelligence import forecast_prices
from utils.recommendations import get_recommendations
from utils.reporting import build_report_csv
from utils.risk_calculator import risk_band
from utils.style import apply_glassmorphism_css
from visualization.advanced_charts import live_mandi_comparison_chart
from visualization.charts import risk_trend_chart


st.set_page_config(page_title="Agri Risk Intelligence System (ARIS)", page_icon="🏠", layout="wide")
ensure_language_state()
apply_glassmorphism_css()

payload = load_payload(seed=42, days=120)
risk_df = payload["data"].copy()
mandi_df = payload["mandi_prices"].copy()
price_forecast_df = payload["price_forecast"].copy()

if "region" not in mandi_df.columns:
    mandi_df = mandi_df.copy()
    mandi_df["region"] = "All"

if "crop" in mandi_df.columns and "Soybean" not in set(mandi_df["crop"].unique()) and "Maize" in set(mandi_df["crop"].unique()):
    mandi_df = mandi_df.copy()
    mandi_df["crop"] = mandi_df["crop"].replace({"Maize": "Soybean"})

regions = sorted(risk_df["region"].unique())


def _mandi_region_view(df, selected: list[str]):
    if "region" not in df.columns:
        return df
    if "All" in df["region"].unique():
        return df
    return df[df["region"].isin(selected)]

with st.sidebar:
    st.markdown(f"### {t('app_name')}")
    st.caption(t("home_page"))
    current_lang = get_language()
    lang_choice = st.selectbox(t("language"), LANGUAGES, index=LANGUAGES.index(current_lang))
    set_language(lang_choice)
    selected_regions = st.multiselect(t("region"), regions, default=regions)


def _greeting_text() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return t("greet_morning")
    if hour < 17:
        return t("greet_afternoon")
    return t("greet_evening")


min_date = risk_df["date"].min().date()
max_date = risk_df["date"].max().date()
date_window = st.slider(t("date_range"), min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")

filtered = risk_df[
    risk_df["region"].isin(selected_regions)
    & (risk_df["date"].dt.date >= date_window[0])
    & (risk_df["date"].dt.date <= date_window[1])
].copy()

if filtered.empty:
    st.error("No data for selected filters.")
    st.stop()

latest = filtered.sort_values("date").iloc[-1]
prev = filtered.sort_values("date").iloc[-2] if len(filtered) > 1 else latest
overall = float(latest["total_risk"])
band = risk_band(overall)

wheat_now = float(
    _mandi_region_view(mandi_df, selected_regions)[_mandi_region_view(mandi_df, selected_regions)["crop"] == "Wheat"]
    .sort_values("date")
    .groupby("date")["price_inr_qtl"]
    .mean()
    .iloc[-1]
)
soy_now = float(
    _mandi_region_view(mandi_df, selected_regions)[_mandi_region_view(mandi_df, selected_regions)["crop"] == "Soybean"]
    .sort_values("date")
    .groupby("date")["price_inr_qtl"]
    .mean()
    .iloc[-1]
)

wheat_fc = forecast_prices(_mandi_region_view(mandi_df, selected_regions), crop="Wheat", horizon=7)
market_delta = float(wheat_fc["predicted_price"].iloc[-1] - wheat_fc["predicted_price"].iloc[0])
market_risk = max(0.0, min(100.0, 55 - (market_delta / 8)))

st.markdown(f"### {_greeting_text()}")

if band == "Low":
    pill = "background:#e8f8ef;color:#166534;"
elif band == "Moderate":
    pill = "background:#fff8dc;color:#7c5d00;"
else:
    pill = "background:#fdecec;color:#b91c1c;"

st.markdown(
    f"""
    <div class='glass-card hero-card'>
        <h1>{t('app_name')}</h1>
        <div style='font-size:1rem;color:#6b7280;'>{t('subtitle')}</div>
        <div style='font-size:0.92rem;color:#6b7280;margin-top:0.2rem;'>{t('subtitle_hi_line')}</div>
        <div class='hero-score'>{overall:.1f}</div>
        <span class='pill' style='{pill}'>{band}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

climate_delta = float(latest["climate_risk"] - prev["climate_risk"])
pest_delta = float(latest["pest_risk"] - prev["pest_risk"])
forecast_val = float(filtered.groupby("date")["total_risk"].mean().tail(7).mean())
prev_window = filtered.groupby("date")["total_risk"].mean().tail(14).head(7)
prev_window_mean = float(prev_window.mean()) if len(prev_window) > 0 else forecast_val
forecast_delta = float(forecast_val - prev_window_mean)

for col, title, value, delta in [
    (c1, t("climate_risk"), float(latest["climate_risk"]), climate_delta),
    (c2, t("pest_risk"), float(latest["pest_risk"]), pest_delta),
    (c3, t("forecast_risk"), market_risk, -market_delta / 10),
]:
    sign = "+" if delta >= 0 else ""
    col.markdown(
        f"""
        <div class='glass-card'>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value:.1f}</div>
            <div class='kpi-sub'>{sign}{delta:.2f} vs previous</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

insight = generate_advanced_insights(filtered, payload["crop_ndvi"], price_forecast_df)[0]
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
if band == "High":
    st.error(f"{t('smart_insight')}: {insight}")
elif band == "Moderate":
    st.warning(f"{t('smart_insight')}: {insight}")
else:
    st.success(f"{t('smart_insight')}: {insight}")

if market_delta > 0:
    st.info("📉 Wheat prices expected to rise in next 3 days")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown(f"#### {t('live_mandi')}")
p1, p2 = st.columns(2)
p1.metric(f"{t('current_price')} - Wheat", f"INR {wheat_now:.2f}/qtl")
p2.metric(f"{t('current_price')} - Soybean", f"INR {soy_now:.2f}/qtl")

mandi_filtered = _mandi_region_view(mandi_df, selected_regions)
st.plotly_chart(live_mandi_comparison_chart(mandi_filtered), use_container_width=True)

chart = risk_trend_chart(filtered, risk_type="total_risk")
chart.update_layout(title=t("combined_risk_graph"), template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=55, b=10))
for trace in chart.data:
    trace.line.width = 3
st.plotly_chart(chart, use_container_width=True)

if band == "High":
    st.error(t("alert_high"))
elif band == "Moderate":
    st.warning(t("alert_med"))
else:
    st.success(t("alert_low"))

st.markdown(f"#### {t('quick_actions')}")
q1, q2, q3, q4 = st.columns(4)
with q1:
    if st.button("🌿 NDVI Intelligence", use_container_width=True):
        st.switch_page("pages/1_NDVI_Intelligence.py")
with q2:
    if st.button("🐛 Pest & Climate", use_container_width=True):
        st.switch_page("pages/2_Pest_Climate.py")
with q3:
    if st.button("💰 Market Intelligence", use_container_width=True):
        st.switch_page("pages/3_Market_Intelligence.py")
with q4:
    if st.button("⚙️ Settings", use_container_width=True):
        st.switch_page("pages/settings.py")

rec = get_recommendations(float(latest["total_risk"]), float(latest["climate_risk"]), float(latest["pest_risk"]))[-1]
report_csv = build_report_csv(
    latest,
    insight,
    rec,
    get_language(),
    price_context={
        "wheat_price_inr_qtl": f"{wheat_now:.2f}",
        "soybean_price_inr_qtl": f"{soy_now:.2f}",
        "market_signal": t("wait_days") if market_delta > 0 else t("sell_now"),
    },
)

st.download_button(
    label=f"📥 {t('download_report')}",
    data=report_csv,
    file_name="aris_report.csv",
    mime="text/csv",
)

st.caption("Use sidebar to explore NDVI Intelligence, Pest & Climate Analysis, Market Intelligence, and Settings.")

"""Premium startup-grade Streamlit dashboard for agricultural risk intelligence."""

from __future__ import annotations

import io
import os
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.backends.backend_pdf import PdfPages

from config.config import PRIMARY_REGIONS, RISK_LEVEL_THRESHOLDS
from main import run_pipeline
from utils.feature_engineering import build_feature_columns


st.set_page_config(page_title="Agri Risk Intelligence System", page_icon="🌾", layout="wide")

RISK_COLUMNS = {
    "Climate": "climate_risk_score",
    "Pest": "pest_disease_risk_score",
    "Combined": "predicted_total_risk",
}

REGION_COORDS = {
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Sehore": {"lat": 23.2050, "lon": 77.0850},
    "Ashta": {"lat": 23.0170, "lon": 76.7200},
}


def inject_custom_css(theme_mode: str) -> None:
    """Injects theme-aware glassmorphism styles and micro-interactions."""
    dark = theme_mode == "Dark"
    bg = "#0d1117" if dark else "#edf2f7"
    text = "#f2f6fa" if dark else "#0f172a"
    panel = "rgba(255,255,255,0.08)" if dark else "rgba(255,255,255,0.58)"
    border = "rgba(255,255,255,0.18)" if dark else "rgba(255,255,255,0.52)"
    shadow = "0 10px 32px rgba(0,0,0,0.35)" if dark else "0 10px 28px rgba(20,40,80,0.16)"
    hero_grad = "linear-gradient(130deg,#0a7d45,#1ea867,#2b5ea8)" if dark else "linear-gradient(130deg,#2d9b63,#5fbf7e,#6aa7ff)"

    st.markdown(
        f"""
        <style>
            .stApp {{
                color: {text};
                background:
                    radial-gradient(circle at 10% 10%, rgba(46, 204, 113, 0.17), transparent 35%),
                    radial-gradient(circle at 86% 16%, rgba(59, 130, 246, 0.18), transparent 30%),
                    linear-gradient(180deg, {bg} 0%, {bg} 100%);
                transition: all 0.35s ease;
            }}
            .main .block-container {{
                padding-top: 1.3rem;
            }}
            h1, h2, h3 {{
                position: relative;
                z-index: 10;
                clear: both;
            }}
            .safe-heading {{
                margin-top: 1.2rem;
                margin-bottom: 0.65rem;
                padding-top: 0.2rem;
                line-height: 1.25;
            }}
            .section-gap {{
                height: 0.55rem;
            }}
            .hero {{
                background: {hero_grad};
                color: #ffffff;
                border-radius: 18px;
                padding: 1.2rem 1.3rem;
                box-shadow: {shadow};
                margin-bottom: 1rem;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .hero h1 {{ margin: 0; font-size: 1.6rem; }}
            .hero p {{ margin: 0.45rem 0 0 0; opacity: 0.95; }}
            .glass {{
                background: {panel};
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border-radius: 16px;
                border: 1px solid {border};
                box-shadow: {shadow};
                padding: 0.9rem 1rem;
                margin-bottom: 0.75rem;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .glass:hover {{
                transform: translateY(-2px);
                box-shadow: 0 14px 30px rgba(0,0,0,0.25);
            }}
            .chip {{
                display: inline-block;
                padding: 0.24rem 0.6rem;
                margin-right: 0.4rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
                background: rgba(255,255,255,0.18);
                border: 1px solid rgba(255,255,255,0.24);
            }}
            .kpi-title {{ font-size: 0.84rem; opacity: 0.9; font-weight: 700; }}
            .kpi-value {{ font-size: 2rem; font-weight: 800; margin: 0.15rem 0; }}
            .kpi-delta {{ font-size: 0.86rem; font-weight: 600; opacity: 0.95; }}
            .risk-badge {{
                display: inline-block;
                border-radius: 10px;
                padding: 0.3rem 0.55rem;
                font-size: 0.78rem;
                font-weight: 700;
            }}
            .badge-low {{ background: rgba(74, 222, 128, 0.2); color: #16a34a; }}
            .badge-med {{ background: rgba(250, 204, 21, 0.2); color: #d97706; }}
            .badge-high {{ background: rgba(248, 113, 113, 0.2); color: #dc2626; }}
            .small-note {{ font-size: 0.78rem; opacity: 0.8; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads cached data outputs or runs pipeline once when files are missing."""
    results_path = os.path.join("data", "risk_assessment_results.csv")
    comparison_path = "model_comparison_table.csv"

    if not (os.path.exists(results_path) and os.path.exists(comparison_path)):
        output = run_pipeline()
        return output["results_df"], output["comparison_df"]

    results = pd.read_csv(results_path, parse_dates=["date"])
    comparison = pd.read_csv(comparison_path)
    return results, comparison


def level_from_score(score: float) -> str:
    """Converts score to low/medium/high band."""
    if score < RISK_LEVEL_THRESHOLDS["low"]:
        return "Low"
    if score < RISK_LEVEL_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def level_badge(level: str) -> str:
    """Returns HTML badge for risk level."""
    if level == "Low":
        return '<span class="risk-badge badge-low">Low Risk</span>'
    if level == "Medium":
        return '<span class="risk-badge badge-med">Medium Risk</span>'
    return '<span class="risk-badge badge-high">High Risk</span>'


def latest_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Returns latest record per region."""
    idx = df.groupby("region")["date"].idxmax()
    return df.loc[idx].copy().reset_index(drop=True)


def simulate_realtime_data(base_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Adds realistic noise and updates dates to current day for live simulation."""
    rng = np.random.default_rng(seed)
    df = base_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    day_shift = (datetime.today().date() - df["date"].max().date()).days
    if day_shift != 0:
        df["date"] = df["date"] + pd.to_timedelta(day_shift, unit="D")

    noise_map = {
        "climate_risk_score": 2.3,
        "pest_disease_risk_score": 2.8,
        "predicted_total_risk": 2.0,
        "total_risk_score": 2.0,
        "ndvi": 0.02,
        "temperature": 0.45,
        "humidity": 1.2,
    }
    for col, std in noise_map.items():
        if col in df.columns:
            df[col] = df[col] + rng.normal(0, std, len(df))

    if "temperature" not in df.columns and {"temp_min", "temp_max"}.issubset(df.columns):
        df["temperature"] = (df["temp_min"] + df["temp_max"]) / 2.0

    for col in ["climate_risk_score", "pest_disease_risk_score", "predicted_total_risk", "total_risk_score"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 100)

    if "ndvi" in df.columns:
        df["ndvi"] = df["ndvi"].clip(0.05, 0.95)
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(15, 100)

    df["climate_risk_level"] = df["climate_risk_score"].apply(level_from_score)
    df["pest_disease_risk_level"] = df["pest_disease_risk_score"].apply(level_from_score)
    df["total_risk_level"] = df["predicted_total_risk"].apply(level_from_score)
    return df.sort_values(["date", "region"]).reset_index(drop=True)


def forecast_next_7_days(df: pd.DataFrame, region: str, risk_col: str) -> pd.DataFrame:
    """Creates a simple trend-based 7-day forecast for selected region."""
    region_df = df[df["region"] == region].sort_values("date")[["date", risk_col]].dropna()
    if len(region_df) < 10:
        return pd.DataFrame(columns=["date", risk_col])

    y = region_df[risk_col].tail(30).to_numpy()
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + 7)
    pred = intercept + slope * future_x
    pred = np.clip(pred, 0, 100)

    start = region_df["date"].max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start, periods=7, freq="D")
    return pd.DataFrame({"date": future_dates, risk_col: pred, "region": region})


def make_kpi_card(title: str, icon: str, value: float, delta_pct: float, series: pd.Series) -> None:
    """Renders KPI card with sparkline and micro animation counter."""
    trend_color = "#ef4444" if delta_pct > 0 else "#22c55e"
    st.markdown(
        f"""
        <div class="glass">
            <div class="kpi-title">{icon} {title}</div>
            <div class="kpi-value">{value:.1f}</div>
            <div class="kpi-delta" style="color:{trend_color};">{delta_pct:+.2f}% vs previous window</div>
            <div class="small-note">Auto-updating simulated feed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    spark = px.area(y=series.tail(40), labels={"value": "Score", "index": "Step"})
    spark.update_traces(line_color=trend_color, fillcolor="rgba(16,185,129,0.15)")
    spark.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=85,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(spark, use_container_width=True)


def get_feature_importance_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Returns feature importance from CatBoost artifact or correlation fallback."""
    features = [c for c in build_feature_columns() if c in filtered_df.columns]
    if not features:
        return pd.DataFrame(columns=["Feature", "Importance"])

    artifact_path = os.path.join("models", "artifacts", "catboost_risk.joblib")
    if os.path.exists(artifact_path):
        try:
            model = joblib.load(artifact_path)
            if hasattr(model, "get_feature_importance"):
                vals = model.get_feature_importance()
                return pd.DataFrame({"Feature": features, "Importance": vals}).sort_values("Importance", ascending=False)
        except Exception:
            pass

    target = filtered_df["predicted_total_risk"]
    values = []
    for col in features:
        corr = filtered_df[col].corr(target)
        values.append(0.0 if pd.isna(corr) else abs(corr) * 100)
    return pd.DataFrame({"Feature": features, "Importance": values}).sort_values("Importance", ascending=False)


def build_ai_insights(df: pd.DataFrame, selected_region: str, risk_col: str) -> list[str]:
    """Creates plain-language AI insights for data storytelling."""
    insights = []
    region_df = df[df["region"] == selected_region].sort_values("date")
    if len(region_df) < 8:
        return ["Insufficient recent data for deep insights."]

    last7 = region_df.tail(7)
    prev7 = region_df.tail(14).head(7)

    risk_delta = float(last7[risk_col].mean() - prev7[risk_col].mean())
    temp_delta = float(last7.get("temperature", pd.Series([0])).mean() - prev7.get("temperature", pd.Series([0])).mean())
    ndvi_delta = float(last7["ndvi"].mean() - prev7["ndvi"].mean()) if "ndvi" in region_df.columns else 0.0
    pest_delta = float(last7["pest_disease_risk_score"].mean() - prev7["pest_disease_risk_score"].mean())

    if risk_delta > 2 and temp_delta > 0.4:
        insights.append("Risk increased due to a recent temperature spike and drier crop conditions.")
    elif risk_delta < -1:
        insights.append("Risk is easing as weather variability reduced over the last week.")
    else:
        insights.append("Risk trend is mostly stable with mild short-term fluctuations.")

    if ndvi_delta < -0.02:
        insights.append("NDVI drop detected, indicating possible crop stress in active plots.")
    else:
        insights.append("NDVI remains steady, suggesting vegetation health is currently resilient.")

    if pest_delta > 4:
        insights.append("Pest outbreak probability is rising; field scouting should be intensified.")
    else:
        insights.append("Pest pressure is not showing unusual acceleration this week.")

    top_region = latest_by_region(df).sort_values(risk_col, ascending=False).iloc[0]["region"]
    insights.append(f"Most affected region currently: {top_region}.")

    return insights


def make_report_pdf(latest_df: pd.DataFrame, comparison_df: pd.DataFrame, risk_col: str) -> bytes:
    """Creates a lightweight PDF report for download."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax.axis("off")

        lines = [
            "Agricultural Risk Assessment Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Regional Snapshot:",
        ]
        for _, row in latest_df.iterrows():
            lines.append(
                f"- {row['region']}: {risk_col}={row[risk_col]:.2f}, Climate={row['climate_risk_score']:.2f}, Pest={row['pest_disease_risk_score']:.2f}"
            )

        best = comparison_df.sort_values("RMSE").iloc[0]
        lines.extend([
            "",
            f"Best model: {best['Model']} | RMSE={best['RMSE']:.3f} | MAE={best['MAE']:.3f} | R2={best['R2']:.3f}",
        ])

        ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return buf.getvalue()


def render_dashboard() -> None:
    """Renders startup-level interactive dashboard experience."""
    if "refresh_count" not in st.session_state:
        st.session_state["refresh_count"] = 0
    if "timeline_idx" not in st.session_state:
        st.session_state["timeline_idx"] = 0

    st.sidebar.title("Experience Controls")
    theme_mode = st.sidebar.toggle("Dark mode", value=True, help="Switch between dark and light themes")
    theme_name = "Dark" if theme_mode else "Light"
    inject_custom_css(theme_name)

    if st.sidebar.button("Refresh Live Data"):
        st.session_state["refresh_count"] += 1

    auto_refresh = st.sidebar.checkbox("Auto-refresh simulation", value=False)
    refresh_interval = st.sidebar.slider("Refresh every (seconds)", 4, 20, 8)

    risk_type = st.sidebar.selectbox("Risk Lens", ["Climate", "Pest", "Combined"], index=2)
    risk_col = RISK_COLUMNS[risk_type]

    results_df, comparison_df = load_outputs()
    sim_seed = st.session_state["refresh_count"] + int(time.time() // max(refresh_interval, 1))
    data_df = simulate_realtime_data(results_df, sim_seed)

    min_date = data_df["date"].min().date()
    max_date = data_df["date"].max().date()
    default_start = max(min_date, max_date - pd.Timedelta(days=150))
    date_range = st.sidebar.slider("Date range", min_value=min_date, max_value=max_date, value=(default_start, max_date))
    selected_regions = st.sidebar.multiselect("Regions", PRIMARY_REGIONS, default=PRIMARY_REGIONS)

    filtered_df = data_df[
        (data_df["region"].isin(selected_regions))
        & (data_df["date"].dt.date >= date_range[0])
        & (data_df["date"].dt.date <= date_range[1])
    ].copy()

    if filtered_df.empty:
        st.warning("No data for selected filters.")
        return

    if auto_refresh:
        components.html(
            f"""
            <script>
                setTimeout(function() {{
                    window.parent.location.reload();
                }}, {refresh_interval * 1000});
            </script>
            """,
            height=0,
        )

    st.markdown(
        """
        <div class="hero">
            <h1>Agri Risk Intelligence System</h1>
            <p>
                Premium AI decision cockpit for wheat farming in Madhya Pradesh.
                Real-time risk simulation, forecast storytelling, and actionable advisories for Bhopal, Sehore, and Ashta.
            </p>
            <div style="margin-top:0.6rem;">
                <span class="chip">AI-native</span>
                <span class="chip">Forecast Ready</span>
                <span class="chip">Farmer-Centric</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Overview", "NDVI Analysis", "Pest Insights", "Model Performance"])

    with tabs[0]:
        latest = latest_by_region(filtered_df)
        selected_region = st.selectbox("Region Focus", selected_regions, index=0)
        region_df = filtered_df[filtered_df["region"] == selected_region].sort_values("date")

        c1, c2, c3 = st.columns(3)
        for col, title, icon, box in [
            ("predicted_total_risk", "Combined Risk", "🌾", c1),
            ("climate_risk_score", "Climate Stress", "🌡️", c2),
            ("pest_disease_risk_score", "Pest Pressure", "🐛", c3),
        ]:
            recent = region_df[col].tail(14)
            prev = region_df[col].tail(28).head(14)
            delta_pct = 0.0 if prev.mean() == 0 else ((recent.mean() - prev.mean()) / prev.mean()) * 100
            with box:
                make_kpi_card(title, icon, float(recent.mean()), float(delta_pct), region_df[col])

        st.markdown("### AI Insights")
        for insight in build_ai_insights(filtered_df, selected_region, risk_col):
            if "rising" in insight.lower() or "drop" in insight.lower() or "increased" in insight.lower():
                st.warning(insight)
            elif "stable" in insight.lower() or "resilient" in insight.lower() or "easing" in insight.lower():
                st.success(insight)
            else:
                st.info(insight)

        st.markdown("### Interactive Risk Map")
        map_df = latest.copy()
        map_df["lat"] = map_df["region"].map(lambda x: REGION_COORDS[x]["lat"])
        map_df["lon"] = map_df["region"].map(lambda x: REGION_COORDS[x]["lon"])
        map_df["risk_val"] = map_df[risk_col]

        fig_map = go.Figure()
        fig_map.add_trace(
            go.Densitymap(
                lat=map_df["lat"],
                lon=map_df["lon"],
                z=map_df["risk_val"],
                radius=35,
                coloraxis="coloraxis",
                hovertemplate="Heat Risk: %{z:.1f}<extra></extra>",
            )
        )
        fig_map.add_trace(
            go.Scattermap(
                lat=map_df["lat"],
                lon=map_df["lon"],
                mode="markers+text",
                text=map_df["region"],
                customdata=np.stack([map_df["region"], map_df["risk_val"]], axis=-1),
                marker=dict(size=map_df["risk_val"] / 4 + 12, color=map_df["risk_val"], colorscale="Turbo", showscale=False),
                textposition="top center",
                hovertemplate="Region: %{customdata[0]}<br>Risk: %{customdata[1]:.1f}<extra></extra>",
            )
        )
        fig_map.update_layout(
            map=dict(style="carto-darkmatter" if theme_mode else "open-street-map", center=dict(lat=23.16, lon=77.08), zoom=7),
            margin=dict(l=0, r=0, t=0, b=0),
            height=460,
            coloraxis=dict(colorscale="RdYlGn_r"),
        )

        map_selection = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="regional_map")
        if map_selection and map_selection.get("selection") and map_selection["selection"].get("points"):
            point = map_selection["selection"]["points"][0]
            if "customdata" in point and len(point["customdata"]) > 0:
                clicked_region = point["customdata"][0]
                st.info(f"Clicked region details: {clicked_region}")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="safe-heading">Forecast (Next 7 Days)</h3>', unsafe_allow_html=True)
        fc = forecast_next_7_days(filtered_df, selected_region, risk_col)
        hist = region_df[["date", risk_col]].tail(60)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=hist["date"], y=hist[risk_col], mode="lines", name="Historical", line=dict(width=2.6)))
        if not fc.empty:
            fig_fc.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc[risk_col],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(width=2.3, dash="dot", color="#f97316"),
                )
            )
        fig_fc.update_layout(height=360, hovermode="x unified", yaxis_title="Risk Score")
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("### Timeline Player")
        all_dates = sorted(filtered_df["date"].dt.date.unique())
        play_col, stop_col, slider_col = st.columns([1, 1, 6])
        with play_col:
            play = st.button("Play")
        with stop_col:
            stop = st.button("Stop")
        if stop:
            st.session_state["timeline_play"] = False
        if play:
            st.session_state["timeline_play"] = True

        st.session_state["timeline_idx"] = st.slider(
            "Timeline",
            min_value=0,
            max_value=max(len(all_dates) - 1, 0),
            value=min(st.session_state["timeline_idx"], max(len(all_dates) - 1, 0)),
        )
        frame_date = all_dates[st.session_state["timeline_idx"]]
        frame_df = filtered_df[filtered_df["date"].dt.date == frame_date]
        frame_fig = px.bar(frame_df, x="region", y=risk_col, color=risk_col, color_continuous_scale="Turbo", title=f"Risk on {frame_date}")
        st.plotly_chart(frame_fig, use_container_width=True)

        if st.session_state.get("timeline_play", False):
            time.sleep(0.4)
            st.session_state["timeline_idx"] = (st.session_state["timeline_idx"] + 1) % len(all_dates)
            st.rerun()

        st.markdown("### Alert Center")
        hot = latest[latest["predicted_total_risk"] > 70]["region"].tolist()
        if hot:
            st.error("High overall risk detected in: " + ", ".join(hot))
        if latest["pest_disease_risk_score"].max() > 68:
            st.warning("High pest activity detected. Immediate scouting and biological control advisory suggested.")
        if latest["climate_risk_score"].max() > 72:
            st.warning("Heatwave warning: climate stress is elevated in at least one region.")
        if not hot and latest["pest_disease_risk_score"].max() <= 68:
            st.success("No critical alerts at this moment. Monitoring continues in real-time.")

        st.markdown("### Farmer Action Panel")
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("View Recommendations"):
                score = float(region_df[risk_col].tail(7).mean())
                if score > 70:
                    st.error("High risk action plan: preventive spray window, moisture retention, and daily field checks.")
                elif score >= 40:
                    st.warning("Moderate risk plan: scouting every 2 days, monitor leaf color, and localized treatment.")
                else:
                    st.success("Low risk plan: continue irrigation schedule and routine monitoring.")
        with a2:
            report_pdf = make_report_pdf(latest, comparison_df, risk_col)
            st.download_button("Download Advisory", data=report_pdf, file_name="krishi_advisory.pdf", mime="application/pdf")
        with a3:
            if st.button("Simulate Risk Reduction"):
                reduced = max(0.0, float(region_df[risk_col].tail(7).mean()) - 8.5)
                st.info(f"If interventions are applied, expected {risk_type.lower()} risk could reduce to ~{reduced:.1f}.")

        st.markdown("### Farm Health Score")
        health_score = int(np.clip(100 - region_df[risk_col].tail(14).mean(), 0, 100))
        badge = "Excellent 🌟" if health_score >= 75 else ("Moderate ⚠️" if health_score >= 45 else "Critical 🚨")
        st.progress(health_score / 100)
        st.markdown(f"**Health Score:** {health_score}/100 | **Badge:** {badge}")

    with tabs[1]:
        st.markdown("### NDVI Storyboard")
        ndvi_df = filtered_df.sort_values(["region", "date"]).copy()
        ndvi_df["ndvi_drop"] = ndvi_df.groupby("region")["ndvi"].diff()
        anomalies = ndvi_df[ndvi_df["ndvi_drop"] <= -0.06]

        ndvi_fig = px.line(
            ndvi_df,
            x="date",
            y="ndvi",
            color="region",
            line_shape="spline",
            title="Vegetation Health Timeline",
            hover_data=["temperature", "humidity", "rainfall"],
        )
        if not anomalies.empty:
            an_fig = px.scatter(anomalies, x="date", y="ndvi", color="region")
            for tr in an_fig.data:
                tr.update(marker=dict(size=8, color="#ef4444"))
                ndvi_fig.add_trace(tr)
        ndvi_fig.update_layout(hovermode="x unified")
        st.plotly_chart(ndvi_fig, use_container_width=True)

        if not anomalies.empty:
            st.warning("NDVI drop detected in recent periods, possible crop stress pockets emerging.")
        else:
            st.success("NDVI trend remains healthy and stable for selected regions.")

    with tabs[2]:
        st.markdown("### Pest Outbreak Intelligence")
        pest_fig = px.line(
            filtered_df,
            x="date",
            y="pest_disease_risk_score",
            color="region",
            line_shape="spline",
            title="Pest Risk Evolution",
            hover_data=["aphid_population", "rust_severity", "armyworm_presence", "humidity"],
        )
        pest_fig.update_layout(hovermode="x unified")
        st.plotly_chart(pest_fig, use_container_width=True)

        st.info("Data Story: Pest risk spikes often coincide with high humidity and warm nights; monitor canopy zones after rainfall events.")

    with tabs[3]:
        st.markdown("### Model Hub")
        best = comparison_df.sort_values("RMSE").iloc[0]
        st.success(f"Best model detected: {best['Model']} | RMSE {best['RMSE']:.3f} | R2 {best['R2']:.3f}")
        st.dataframe(comparison_df.style.highlight_min(subset=["RMSE"], color="#a7f3d0"), use_container_width=True)

        fig_rmse = px.bar(comparison_df.sort_values("RMSE"), x="Model", y="RMSE", color="RMSE", color_continuous_scale="Sunset")
        st.plotly_chart(fig_rmse, use_container_width=True)

        cmp = comparison_df.copy()
        cmp["Accuracy"] = (cmp["R2"] * 100).clip(0, 100)
        fig_acc = px.bar(cmp.sort_values("Accuracy", ascending=False), x="Model", y="Accuracy", color="Accuracy", color_continuous_scale="Greens")
        st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown("### Explainability (SHAP-like Summary)")
        imp_df = get_feature_importance_df(filtered_df).head(6)
        if imp_df.empty:
            st.info("Explainability data unavailable for this selection.")
        else:
            total_imp = imp_df["Importance"].sum() or 1.0
            imp_df["Contribution (%)"] = (imp_df["Importance"] / total_imp) * 100
            explain_fig = px.bar(
                imp_df.sort_values("Contribution (%)", ascending=True),
                x="Contribution (%)",
                y="Feature",
                orientation="h",
                color="Contribution (%)",
                color_continuous_scale="Teal",
            )
            st.plotly_chart(explain_fig, use_container_width=True)
            top = imp_df.iloc[0]
            st.info(f"{top['Feature']} contributed approximately {top['Contribution (%)']:.1f}% to current risk behavior.")

        report_pdf = make_report_pdf(latest_by_region(filtered_df), comparison_df, risk_col)
        st.download_button("Download Report (PDF)", data=report_pdf, file_name="krishi_risk_report.pdf", mime="application/pdf")


if __name__ == "__main__":
    render_dashboard()

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.ndvi_simulation import interpolate_ndvi


def ndvi_research_chart(crop_ndvi: pd.DataFrame, selected_crop: str = "Wheat") -> go.Figure:
    smooth = interpolate_ndvi(crop_ndvi, points_per_day=4)

    fig = go.Figure()
    for crop in ["Wheat", "Maize"]:
        data = smooth[smooth["crop"] == crop]
        fig.add_trace(
            go.Scatter(
                x=data["doy"],
                y=data["ndvi"],
                mode="lines",
                name=f"{crop} (smoothed)",
                line=dict(width=3 if crop == selected_crop else 2),
            )
        )

    single = smooth[smooth["crop"] == selected_crop]
    fig.add_trace(
        go.Scatter(
            x=single["doy"],
            y=single["ndvi"],
            mode="lines",
            name=f"{selected_crop} focus",
            line=dict(color="#16a34a", width=4),
        )
    )

    fig.update_layout(
        title="Crop-wise NDVI Analysis (DOY vs NDVI)",
        xaxis_title="Day of Year (DOY)",
        yaxis_title="NDVI",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def rainfall_heatwave_chart(df: pd.DataFrame) -> go.Figure:
    summary = (
        df.groupby("date", as_index=False)
        .agg(rainfall=("rainfall", "mean"), heatwave=("temperature", lambda x: (x > 32).mean() * 100))
        .sort_values("date")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary["date"], y=summary["rainfall"], mode="lines", name="Rainfall", line=dict(color="#16a34a", width=3)))
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["heatwave"],
            mode="lines",
            name="Heatwave Intensity",
            line=dict(color="#ef4444", width=2, dash="dot"),
            yaxis="y2",
        )
    )

    extreme = summary[(summary["rainfall"] < summary["rainfall"].quantile(0.15)) | (summary["heatwave"] > summary["heatwave"].quantile(0.85))]
    fig.add_trace(
        go.Scatter(
            x=extreme["date"],
            y=extreme["rainfall"],
            mode="markers",
            name="Extreme Weather",
            marker=dict(color="#dc2626", size=8),
        )
    )

    fig.update_layout(
        title="Rainfall vs Heatwave Intensity",
        xaxis_title="Date",
        yaxis=dict(title="Rainfall (mm)"),
        yaxis2=dict(title="Heatwave Intensity (%)", overlaying="y", side="right", range=[0, 100]),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def climate_pest_risk_chart(df: pd.DataFrame) -> go.Figure:
    trend = df.groupby("date", as_index=False).agg(climate_risk=("climate_risk", "mean"), pest_risk=("pest_risk", "mean"))
    fig = px.line(trend, x="date", y=["climate_risk", "pest_risk"], title="Climate vs Pest Risk Interaction")
    fig.update_layout(template="plotly_white", yaxis_title="Risk Score", hovermode="x unified")
    return fig


def mandi_price_chart(price_df: pd.DataFrame, crop: str = "Wheat") -> go.Figure:
    d = (
        price_df[price_df["crop"] == crop]
        .groupby("date", as_index=False)["price_inr_qtl"]
        .mean()
        .sort_values("date")
    )
    trend = d["price_inr_qtl"].rolling(14, min_periods=1).mean()
    seasonal = d["price_inr_qtl"] - trend

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["price_inr_qtl"], mode="lines", name="Price", line=dict(color="#16a34a", width=2.8)))
    fig.add_trace(go.Scatter(x=d["date"], y=trend, mode="lines", name="Trend (14D)", line=dict(color="#14532d", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=d["date"], y=(trend + seasonal.rolling(30, min_periods=1).mean()), mode="lines", name="Seasonal Pattern", line=dict(color="#4b5563", width=1.5, dash="dot")))
    fig.update_layout(title=f"Mandi Price Analysis - {crop}", xaxis_title="Date", yaxis_title="INR / Quintal", template="plotly_white", hovermode="x unified")
    return fig


def live_mandi_comparison_chart(price_df: pd.DataFrame) -> go.Figure:
    d = price_df.copy().sort_values("date")
    if "region" not in d.columns:
        d["region"] = "All"
    fig = px.line(
        d,
        x="date",
        y="price_inr_qtl",
        color="region",
        line_dash="crop",
        title="Live Mandi Price Trend (Wheat & Soybean) - Region Comparison",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="INR / Quintal", hovermode="x unified")
    return fig


def price_forecast_chart(price_df: pd.DataFrame, forecast_df: pd.DataFrame, crop: str = "Wheat") -> go.Figure:
    hist = price_df[price_df["crop"] == crop].sort_values("date")
    fut = forecast_df.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["price_inr_qtl"], mode="lines", name="Historical Price", line=dict(color="#166534", width=2.7)))
    fig.add_trace(go.Scatter(x=fut["date"], y=fut["predicted_price"], mode="lines", name="Forecast Price", line=dict(color="#16a34a", width=2.5, dash="dash")))

    top_zone = fut.sort_values("predicted_price", ascending=False).head(5).sort_values("date")
    fig.add_vrect(
        x0=top_zone["date"].min(),
        x1=top_zone["date"].max(),
        fillcolor="rgba(34,197,94,0.12)",
        line_width=0,
        annotation_text="Best Selling Window",
        annotation_position="top left",
    )
    fig.update_layout(title="Price Forecast and Selling Window", xaxis_title="Date", yaxis_title="INR / Quintal", template="plotly_white", hovermode="x unified")
    return fig

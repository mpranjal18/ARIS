from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def risk_trend_chart(df: pd.DataFrame, risk_type: str = "total_risk") -> go.Figure:
    fig = px.line(
        df,
        x="date",
        y=risk_type,
        color="region",
        hover_data=["temperature", "humidity", "ndvi"],
        title="Risk Trend",
    )
    fig.update_layout(hovermode="x unified", legend_title_text="Region")
    return fig


def ndvi_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="date", y="ndvi", color="region", title="NDVI Time Series")

    drop_df = df[df.groupby("region")["ndvi"].diff() < -0.05]
    if not drop_df.empty:
        fig.add_trace(
            go.Scatter(
                x=drop_df["date"],
                y=drop_df["ndvi"],
                mode="markers",
                marker=dict(color="#e74c3c", size=9, symbol="diamond"),
                name="NDVI Drop",
            )
        )

    return fig


def pest_trend_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="date", y="pest_level", color="region", title="Pest Trend")
    outbreak = df[df["outbreak_flag"] == 1]
    if not outbreak.empty:
        fig.add_trace(
            go.Scatter(
                x=outbreak["date"],
                y=outbreak["pest_level"],
                mode="markers",
                marker=dict(color="#c0392b", size=10, symbol="x"),
                name="Outbreak Signal",
            )
        )
    return fig


def performance_bar_chart(perf_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(perf_df, x="Model", y="RMSE", color="Model", title="Model RMSE Comparison")
    return fig


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        importance_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        title="Feature Importance (Best Model)",
    )
    return fig


def region_map(df: pd.DataFrame) -> go.Figure:
    latest = df.sort_values("date").groupby("region").tail(1)
    fig = px.scatter_geo(
        latest,
        lat="lat",
        lon="lon",
        color="total_risk",
        size="total_risk",
        hover_name="region",
        hover_data={"total_risk": ":.1f", "lat": False, "lon": False},
        color_continuous_scale=["#27ae60", "#f1c40f", "#e74c3c"],
        title="Regional Risk Map: Bhopal, Sehore, Ashta",
    )
    fig.update_geos(
        lataxis_range=[22.5, 23.8],
        lonaxis_range=[76.4, 77.8],
        showcountries=False,
        showland=True,
        landcolor="#f9f5e9",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for region in sorted(history_df["region"].unique()):
        h = history_df[history_df["region"] == region]
        f = forecast_df[forecast_df["region"] == region]
        fig.add_trace(
            go.Scatter(
                x=h["date"],
                y=h["total_risk"],
                mode="lines",
                name=f"{region} - Historical",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=f["date"],
                y=f["predicted_total_risk"],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{region} - Forecast",
            )
        )

    fig.update_layout(title="Forecast (Next 7 Days)", hovermode="x unified")
    return fig

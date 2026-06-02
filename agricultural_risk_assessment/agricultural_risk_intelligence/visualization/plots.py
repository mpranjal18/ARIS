from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def _ensure_dir(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_ndvi_series(features: pd.DataFrame, output_dir: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(features["date"], features["ndvi"], color="#2c7a7b", label="NDVI")
    ax.set_title("NDVI Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.grid(True, alpha=0.2)
    ax.legend()

    out_dir = _ensure_dir(output_dir)
    if out_dir:
        fig.savefig(out_dir / "ndvi_series.png", dpi=150, bbox_inches="tight")
    return fig


def plot_rainfall_heatwave(features: pd.DataFrame, output_dir: str | Path | None = None):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(features["date"], features["rainfall_mm"], color="#3182bd", alpha=0.6, label="Rainfall")
    ax1.set_ylabel("Rainfall (mm)")

    ax2 = ax1.twinx()
    ax2.plot(features["date"], features["heatwave_index"], color="#e53e3e", label="Heatwave Index")
    ax2.set_ylabel("Heatwave Index (7d)")

    ax1.set_title("Rainfall and Heatwave Index")
    ax1.grid(True, axis="y", alpha=0.2)

    out_dir = _ensure_dir(output_dir)
    if out_dir:
        fig.savefig(out_dir / "rainfall_heatwave.png", dpi=150, bbox_inches="tight")
    return fig


def plot_risk_comparison(risk_df: pd.DataFrame, output_dir: str | Path | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risk_df["date"], y=risk_df["climate_risk"], name="Climate Risk"))
    fig.add_trace(go.Scatter(x=risk_df["date"], y=risk_df["pest_risk"], name="Pest Risk"))
    fig.update_layout(title="Climate vs Pest Risk", xaxis_title="Date", yaxis_title="Risk")

    out_dir = _ensure_dir(output_dir)
    if out_dir:
        fig.write_image(str(out_dir / "risk_comparison.png"))
    return fig


def plot_price_trend(features: pd.DataFrame, output_dir: str | Path | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=features["date"], y=features["price_inr_per_qtl"], name="Mandi Price"))
    fig.update_layout(title="Mandi Price Trend", xaxis_title="Date", yaxis_title="INR per Quintal")

    out_dir = _ensure_dir(output_dir)
    if out_dir:
        fig.write_image(str(out_dir / "price_trend.png"))
    return fig


def plot_overall_risk(risk_df: pd.DataFrame, output_dir: str | Path | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risk_df["date"], y=risk_df["overall_risk"], name="Overall Risk"))
    fig.update_layout(title="Overall Risk Trend", xaxis_title="Date", yaxis_title="Risk")

    out_dir = _ensure_dir(output_dir)
    if out_dir:
        fig.write_image(str(out_dir / "overall_risk.png"))
    return fig

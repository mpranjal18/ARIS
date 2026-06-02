"""Visualization layer for risk trends, NDVI behavior, and model comparison."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from config.config import VISUALIZATION_OUTPUT_DIR


class RiskVisualizer:
    """Creates static and interactive charts for the risk assessment workflow."""

    def __init__(self, output_dir: str = VISUALIZATION_OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _save_fig(self, fig, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_climate_risk_graph(self, df: pd.DataFrame, region: str) -> str:
        region_df = df[df["region"] == region].sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(region_df["date"], region_df["climate_risk_score"], color="#1f77b4", linewidth=2)
        ax.set_title(f"Climate Risk Trend - {region}")
        ax.set_ylabel("Score (0-100)")
        return self._save_fig(fig, f"climate_risk_{region}.png")

    def save_pest_risk_graph(self, df: pd.DataFrame, region: str) -> str:
        region_df = df[df["region"] == region].sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(region_df["date"], region_df["pest_disease_risk_score"], color="#ff7f0e", linewidth=2)
        ax.set_title(f"Pest & Disease Risk Trend - {region}")
        ax.set_ylabel("Score (0-100)")
        return self._save_fig(fig, f"pest_risk_{region}.png")

    def save_combined_risk_graph(self, df: pd.DataFrame, region: str) -> str:
        region_df = df[df["region"] == region].sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(region_df["date"], region_df["climate_risk_score"], label="Climate", linewidth=1.8)
        ax.plot(region_df["date"], region_df["pest_disease_risk_score"], label="Pest", linewidth=1.8)
        ax.plot(region_df["date"], region_df["predicted_total_risk"], label="Predicted Total", linewidth=2.2)
        ax.set_title(f"Combined Risk Trend - {region}")
        ax.set_ylabel("Score (0-100)")
        ax.legend()
        return self._save_fig(fig, f"combined_risk_{region}.png")

    def save_ndvi_timeseries_graph(self, df: pd.DataFrame, region: str) -> str:
        region_df = df[df["region"] == region].sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(region_df["date"], region_df["ndvi"], color="#2ca02c", linewidth=1.8)
        ax.set_title(f"NDVI Time Series - {region}")
        ax.set_ylabel("NDVI")
        return self._save_fig(fig, f"ndvi_time_series_{region}.png")

    def save_model_comparison_chart(self, comparison_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=comparison_df, x="Model", y="RMSE", ax=ax)
        ax.set_title("Model Comparison (RMSE)")
        ax.tick_params(axis="x", rotation=25)
        return self._save_fig(fig, "model_comparison_rmse.png")

    def save_regional_comparison(self, df: pd.DataFrame) -> str:
        grouped = (
            df.groupby("region", as_index=False)[
                ["climate_risk_score", "pest_disease_risk_score", "predicted_total_risk"]
            ]
            .mean()
            .round(2)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        grouped.set_index("region").plot(kind="bar", ax=ax)
        ax.set_title("Regional Average Risk Comparison")
        ax.set_ylabel("Score (0-100)")
        return self._save_fig(fig, "regional_risk_comparison.png")

    def make_interactive_combined_plot(self, df: pd.DataFrame, region: str):
        """Returns plotly figure for Streamlit rendering."""
        region_df = df[df["region"] == region].sort_values("date")
        plot_df = region_df[["date", "climate_risk_score", "pest_disease_risk_score", "predicted_total_risk"]]
        melted = plot_df.melt(id_vars="date", var_name="Risk Type", value_name="Score")
        return px.line(melted, x="date", y="Score", color="Risk Type", title=f"Combined Risk - {region}")

    def create_all_visualizations(self, results_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
        """Creates all required visual outputs for primary regions."""
        for region in ["Bhopal", "Sehore", "Ashta"]:
            self.save_climate_risk_graph(results_df, region)
            self.save_pest_risk_graph(results_df, region)
            self.save_combined_risk_graph(results_df, region)
            self.save_ndvi_timeseries_graph(results_df, region)
        self.save_model_comparison_chart(comparison_df)
        self.save_regional_comparison(results_df)

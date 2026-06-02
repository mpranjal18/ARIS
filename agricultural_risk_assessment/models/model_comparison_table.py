"""Creates and stores model comparison tables for reporting and dashboarding."""

from __future__ import annotations

import os
import pandas as pd

from models.model_comparison import add_rank, to_comparison_dataframe


def generate_model_comparison_table(metrics_by_model: dict, output_path: str) -> pd.DataFrame:
    """Generates ranked model comparison table and writes CSV."""
    comparison_df = to_comparison_dataframe(metrics_by_model)
    comparison_df = add_rank(comparison_df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    return comparison_df

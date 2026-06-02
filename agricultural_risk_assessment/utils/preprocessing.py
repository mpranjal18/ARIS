"""Preprocessing utilities for tabular and deep-learning model inputs."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_tabular_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into train and test sets for tabular models."""
    x = df[feature_cols]
    y = df[target_col]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=True)


def scale_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardizes train/test features and returns fitted scaler."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def create_sequence_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_len: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates rolling sequences for LSTM training."""
    x_values = df[feature_cols].values
    y_values = df[target_col].values
    x_seq, y_seq = [], []
    for i in range(sequence_len, len(df)):
        x_seq.append(x_values[i - sequence_len : i])
        y_seq.append(y_values[i])
    return np.array(x_seq), np.array(y_seq)


def create_cnn_image_like_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_size: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds 2D pseudo-images from temporal windows for CNN models."""
    x_values = df[feature_cols].values
    y_values = df[target_col].values
    x_images, y_image_targets = [], []

    for i in range(window_size, len(df)):
        window = x_values[i - window_size : i]
        image = window.reshape(window_size, len(feature_cols), 1)
        x_images.append(image)
        y_image_targets.append(y_values[i])

    return np.array(x_images), np.array(y_image_targets)


def summarize_region_latest(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    """Returns latest score by region as a dict for dashboard KPIs."""
    latest = (
        df.sort_values("date")
        .groupby("region", as_index=False)
        .tail(1)[["region", score_col]]
    )
    return {row["region"]: float(row[score_col]) for _, row in latest.iterrows()}

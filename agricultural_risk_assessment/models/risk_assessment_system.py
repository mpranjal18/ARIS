"""Central training/evaluation engine for the Agricultural Risk Assessment System."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import MODEL_ARTIFACTS_DIR, SEED
from models.catboost_model import CatBoostRiskModel
from models.cnn_model import CNNRiskModel
from models.hybrid_lstm_catboost_model import HybridLSTMCatBoostRiskModel
from models.lstm_model import LSTMRiskModel
from models.random_forest_model import RandomForestRiskModel
from models.xgboost_model import XGBoostRiskModel


class AgriculturalRiskAssessmentSystem:
    """Trains all required models and returns metrics and production predictions."""

    def __init__(self, sequence_len: int = 14) -> None:
        self.sequence_len = sequence_len
        self.tabular_models = {
            "RandomForest": RandomForestRiskModel(),
            "XGBoost": XGBoostRiskModel(),
            "CatBoost": CatBoostRiskModel(),
        }
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.failed_models: Dict[str, str] = {}

    @staticmethod
    def _metrics(y_true, y_pred) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def _build_sequence_and_aligned_tabular(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates aligned sequence data and matching tabular rows for hybrid training."""
        x_seq_all, y_all, x_tab_all = [], [], []

        for _, g in df.sort_values(["region", "date"]).groupby("region"):
            g = g.reset_index(drop=True)
            x = g[feature_cols].values
            y = g[target_col].values
            for i in range(self.sequence_len, len(g)):
                x_seq_all.append(x[i - self.sequence_len : i])
                x_tab_all.append(x[i])
                y_all.append(y[i])

        return np.array(x_seq_all), np.array(y_all), np.array(x_tab_all)

    def train_and_evaluate(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, Dict[str, float]]:
        """Trains all required model families and evaluates them on held-out data."""
        x = df[feature_cols]
        y = df[target_col]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=SEED, shuffle=True
        )

        os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)

        for model_name, model in self.tabular_models.items():
            model.train(x_train, y_train)
            preds = model.predict(x_test)
            self.metrics[model_name] = model.evaluate(y_test, preds)
            model.save(os.path.join(MODEL_ARTIFACTS_DIR, f"{model_name.lower()}_risk.joblib"))

        x_seq, y_seq, x_tab_aligned = self._build_sequence_and_aligned_tabular(df, feature_cols, target_col)
        split_idx = int(0.8 * len(x_seq))
        x_seq_train, x_seq_test = x_seq[:split_idx], x_seq[split_idx:]
        y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
        x_tab_train, x_tab_test = x_tab_aligned[:split_idx], x_tab_aligned[split_idx:]

        try:
            lstm_model = LSTMRiskModel(n_features=len(feature_cols))
            lstm_model.train(x_seq_train, y_seq_train)
            lstm_preds = lstm_model.predict(x_seq_test)
            self.metrics["LSTM"] = self._metrics(y_seq_test, lstm_preds)
            lstm_model.model.save(os.path.join(MODEL_ARTIFACTS_DIR, "lstm_risk.keras"))
        except Exception as exc:  # pragma: no cover
            self.failed_models["LSTM"] = str(exc)

        try:
            cnn_model = CNNRiskModel(image_shape=x_seq_train.shape[1:])
            cnn_model.train(x_seq_train, y_seq_train)
            cnn_preds = cnn_model.predict(x_seq_test)
            self.metrics["CNN"] = self._metrics(y_seq_test, cnn_preds)
            cnn_model.model.save(os.path.join(MODEL_ARTIFACTS_DIR, "cnn_risk.keras"))
        except Exception as exc:  # pragma: no cover
            self.failed_models["CNN"] = str(exc)

        try:
            hybrid_model = HybridLSTMCatBoostRiskModel(n_features=len(feature_cols))
            hybrid_model.train(x_tab_train, y_seq_train, x_seq_train)
            hybrid_preds = hybrid_model.predict(x_tab_test, x_seq_test)
            self.metrics["Hybrid_LSTM_CatBoost"] = self._metrics(y_seq_test, hybrid_preds)
            hybrid_model.model.save_model(os.path.join(MODEL_ARTIFACTS_DIR, "hybrid_catboost_risk.cbm"))
            hybrid_model.lstm.model.save(os.path.join(MODEL_ARTIFACTS_DIR, "hybrid_lstm_risk.keras"))
        except Exception as exc:  # pragma: no cover
            self.failed_models["Hybrid_LSTM_CatBoost"] = str(exc)

        return self.metrics

    def infer_total_risk(self, df: pd.DataFrame, feature_cols: List[str], model_name: str = "CatBoost") -> np.ndarray:
        """Returns predictions from a selected trained tabular model for dashboard usage."""
        selected = self.tabular_models.get(model_name, self.tabular_models["CatBoost"])
        return selected.predict(df[feature_cols])

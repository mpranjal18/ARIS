from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class PestRiskModel:
    def __init__(self) -> None:
        self.rf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
        self.xgb = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )

    def _simulate_labels(self, features: pd.DataFrame) -> np.ndarray:
        base_prob = features["pest_suitability_score"].values
        labels = (base_prob > 0.6).astype(int)
        return labels

    def fit_predict(self, features: pd.DataFrame) -> pd.Series:
        cols = [
            "humidity_pct",
            "tmax_c",
            "rainfall_mm",
            "ndvi",
            "soil_moisture_index",
            "pest_suitability_score",
        ]
        X = features[cols].values
        y = self._simulate_labels(features)

        self.rf.fit(X, y)
        self.xgb.fit(X, y)

        rf_prob = self.rf.predict_proba(X)[:, 1]
        xgb_prob = self.xgb.predict_proba(X)[:, 1]
        prob = (rf_prob + xgb_prob) / 2

        trf_trigger = (features["humidity_pct"] > 70) & (features["tmax_c"] > 32)
        prob = np.where(trf_trigger, np.minimum(prob + 0.15, 1.0), prob)

        risk = np.clip(prob * 100, 0, 100)
        return pd.Series(risk, index=features.index, name="pest_risk")

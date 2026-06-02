from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


class ClimateRiskModel:
    def __init__(self, seq_len: int = 14) -> None:
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.cat_model = CatBoostRegressor(
            iterations=60,
            depth=6,
            learning_rate=0.1,
            loss_function="RMSE",
            verbose=False,
        )
        self.lstm_model: Sequential | None = None

    def _simulate_risk_target(self, features: pd.DataFrame) -> np.ndarray:
        heat = np.clip(features["heatwave_index"] / 7, 0, 1)
        humidity = features["humidity_stress_index"].values
        ndvi_penalty = np.clip(-features["ndvi_anomaly"], 0, None)
        rain_penalty = np.clip(-features["rainfall_deviation_pct"] / 100, 0, None)
        raw = 0.4 * heat + 0.3 * humidity + 0.2 * ndvi_penalty + 0.1 * rain_penalty
        return np.clip(raw * 100, 0, 100)

    def _build_lstm(self, input_shape: tuple[int, int]) -> Sequential:
        model = Sequential(
            [
                LSTM(16, input_shape=input_shape),
                Dense(8, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit_predict(self, features: pd.DataFrame) -> pd.Series:
        structured_cols = [
            "rainfall_deviation_pct",
            "heatwave_index",
            "temperature_anomaly",
            "humidity_stress_index",
            "ndvi_anomaly",
            "soil_moisture_index",
            "price_volatility",
            "pest_suitability_score",
        ]
        X_struct = features[structured_cols].values
        y_target = self._simulate_risk_target(features)

        X_scaled = self.scaler.fit_transform(X_struct)
        self.cat_model.fit(X_scaled, y_target)
        cat_pred = self.cat_model.predict(X_scaled)

        seq_features = features[["ndvi", "rainfall_mm", "tmax_c", "tmin_c", "humidity_pct"]].values
        if len(seq_features) > self.seq_len + 2:
            X_seq, y_seq = [], []
            for i in range(self.seq_len, len(seq_features)):
                X_seq.append(seq_features[i - self.seq_len : i])
                y_seq.append(y_target[i])
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            self.lstm_model = self._build_lstm((self.seq_len, X_seq.shape[-1]))
            self.lstm_model.fit(X_seq, y_seq, epochs=5, batch_size=16, verbose=0)
            lstm_pred = self.lstm_model.predict(X_seq, verbose=0).reshape(-1)
            lstm_full = np.concatenate([np.full(self.seq_len, np.nan), lstm_pred])
        else:
            lstm_full = np.full(len(features), np.nan)

        combined = np.nanmean(np.vstack([cat_pred, lstm_full]), axis=0)
        combined = np.clip(combined, 0, 100)
        return pd.Series(combined, index=features.index, name="climate_risk")

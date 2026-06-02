from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


FEATURES: List[str] = ["rainfall", "temperature", "humidity", "ndvi", "pest_level", "climate_risk", "pest_risk"]


@dataclass
class TrainingResult:
    model_name: str
    model: object
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame


def _metric_frame(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(float(mae), 4), "RMSE": round(rmse, 4), "R2": round(float(r2), 4)}


def _importance_from_model(model: object, model_name: str) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.zeros(len(FEATURES))
    imp = pd.DataFrame({"feature": FEATURES, "importance": values})
    imp["model"] = model_name
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def train_and_compare_models(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, TrainingResult]:
    X = df[FEATURES]
    y = df["risk_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=seed)

    registry: Dict[str, object] = {
        "Random Forest": RandomForestRegressor(n_estimators=220, max_depth=8, random_state=seed),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=280,
            depth=6,
            learning_rate=0.06,
            loss_function="RMSE",
            verbose=False,
            random_seed=seed,
        ),
    }

    rows = []
    winners: List[TrainingResult] = []

    for name, model in registry.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = _metric_frame(y_test.to_numpy(), np.array(pred))
        rows.append({"Model": name, **metrics})
        winners.append(
            TrainingResult(
                model_name=name,
                model=model,
                metrics=metrics,
                feature_importance=_importance_from_model(model, name),
            )
        )

    perf = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_name = perf.iloc[0]["Model"]
    best = [w for w in winners if w.model_name == best_name][0]
    return perf, best


def forecast_next_7_days(df: pd.DataFrame, best_model: object) -> pd.DataFrame:
    base = df.sort_values(["region", "date"]).groupby("region").tail(1).copy()
    future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=7, freq="D")

    rows = []
    rng = np.random.default_rng(123)
    for _, row in base.iterrows():
        for dt in future_dates:
            new_row = row.copy()
            new_row["date"] = dt
            new_row["rainfall"] = max(0, float(row["rainfall"] + rng.normal(0, 3)))
            new_row["temperature"] = float(np.clip(row["temperature"] + rng.normal(0.2, 1.8), 10, 45))
            new_row["humidity"] = float(np.clip(row["humidity"] + rng.normal(0, 4), 20, 98))
            new_row["ndvi"] = float(np.clip(row["ndvi"] + rng.normal(-0.002, 0.015), 0.2, 0.9))
            new_row["pest_level"] = float(np.clip(row["pest_level"] + rng.normal(1.2, 5), 0, 100))
            new_row["climate_risk"] = float(np.clip(row["climate_risk"] + rng.normal(0.8, 4), 0, 100))
            new_row["pest_risk"] = float(np.clip(row["pest_risk"] + rng.normal(1.0, 4), 0, 100))
            rows.append(new_row)

    fdf = pd.DataFrame(rows)
    fdf["predicted_total_risk"] = best_model.predict(fdf[FEATURES]).clip(0, 100)
    return fdf[["date", "region", "predicted_total_risk"]].reset_index(drop=True)

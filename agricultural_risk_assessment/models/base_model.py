"""Base model abstraction and shared utilities for all model wrappers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaseRiskModel(ABC):
    """Defines common interface and metrics used by all risk prediction models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, x_train, y_train) -> None:
        """Train model on provided inputs and targets."""

    @abstractmethod
    def predict(self, x_input):
        """Run inference for provided model inputs."""

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """Computes standard regression metrics for model comparison."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def save(self, filepath: str) -> None:
        """Persists model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Loads model artifact from disk."""
        self.model = joblib.load(filepath)
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models
    """
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model
        """
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """
        Make predictions
        """
        pass
    
    def preprocess_data(self, X, fit_scaler=False):
        """
        Preprocess the data (scaling)
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # This is an abstract class, so we can't instantiate it directly
    pass
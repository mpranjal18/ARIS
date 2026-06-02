"""Hybrid model that combines LSTM sequence signal with CatBoost tabular strength."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor

from config.config import SEED
from models.base_model import BaseRiskModel
from models.lstm_model import LSTMRiskModel


class HybridLSTMCatBoostRiskModel(BaseRiskModel):
    """Uses LSTM predictions as additional features for CatBoost."""

    def __init__(self, n_features: int) -> None:
        super().__init__("Hybrid_LSTM_CatBoost")
        self.lstm = LSTMRiskModel(n_features=n_features)
        self.model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=SEED,
            loss_function="RMSE",
            verbose=False,
        )

    def train(self, x_train_tabular, y_train, x_train_seq) -> None:
        self.lstm.train(x_train_seq, y_train)
        lstm_preds = self.lstm.predict(x_train_seq).reshape(-1, 1)
        hybrid_input = np.hstack([x_train_tabular, lstm_preds])
        self.model.fit(hybrid_input, y_train)

    def predict(self, x_input_tabular, x_input_seq):
        lstm_preds = self.lstm.predict(x_input_seq).reshape(-1, 1)
        hybrid_input = np.hstack([x_input_tabular, lstm_preds])
        return self.model.predict(hybrid_input)
import numpy as np
import pandas as pd
from models.base_model import BaseModel
from models.lstm_model import LSTMModel
from models.catboost_model import CatBoostModel

class HybridLSTMCatBoostModel(BaseModel):
    """
    Hybrid model combining LSTM and CatBoost for agricultural risk prediction
    """
    
    def __init__(self, name="HybridLSTM-CatBoost", sequence_length=30):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.lstm_model = LSTMModel("LSTM_Component", sequence_length)
        self.catboost_model = CatBoostModel("CatBoost_Component")
        self.is_trained = False
        
    def train(self, X_train, y_train, lstm_epochs=50, catboost_grid_search=False):
        """
        Train the hybrid model
        """
        # For this hybrid approach, we'll use LSTM for time series features
        # and CatBoost for static/tabular features
        
        # Assuming X_train is a DataFrame with mixed time series and static features
        # We need to separate them
        
        # For simplicity in this implementation, we'll treat the first part as time series
        # and the rest as static features
        n_time_series_features = min(5, X_train.shape[1] // 2)  # Adjust as needed
        
        # Split features
        X_train_ts = X_train.iloc[:, :n_time_series_features]
        X_train_static = X_train.iloc[:, n_time_series_features:]
        y_train_values = y_train
        
        # Train LSTM component on time series features
        print("Training LSTM component...")
        self.lstm_model.train(
            X_train_ts.values, 
            y_train_values, 
            epochs=lstm_epochs
        )
        
        # Generate LSTM predictions to use as features for CatBoost
        print("Generating LSTM predictions for CatBoost...")
        lstm_predictions = []
        for i in range(len(X_train_ts)):
            if i >= self.sequence_length:
                pred = self.lstm_model.predict(X_train_ts.iloc[:i+1].values)
                lstm_predictions.append(pred[0])
            else:
                # For initial points, use actual values or simple interpolation
                lstm_predictions.append(y_train_values[i] if i < len(y_train_values) else np.mean(y_train_values))
        
        # Combine LSTM predictions with static features
        X_combined = np.column_stack([
            X_train_static.values,
            lstm_predictions
        ])
        
        # Train CatBoost component
        print("Training CatBoost component...")
        self.catboost_model.train(
            X_combined, 
            y_train_values, 
            use_grid_search=catboost_grid_search
        )
        
        self.is_trained = True
        print(f"{self.name} model trained successfully")
    
    def predict(self, X_test):
        """
        Make predictions using the trained hybrid model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Split features (same logic as in training)
        n_time_series_features = min(5, X_test.shape[1] // 2)
        X_test_ts = X_test.iloc[:, :n_time_series_features]
        X_test_static = X_test.iloc[:, n_time_series_features:]
        
        # Generate LSTM predictions
        lstm_predictions = []
        for i in range(len(X_test_ts)):
            if i >= self.sequence_length:
                pred = self.lstm_model.predict(X_test_ts.iloc[:i+1].values)
                lstm_predictions.append(pred[0])
            else:
                # For initial points, use simple interpolation or mean
                lstm_predictions.append(np.mean(lstm_predictions) if lstm_predictions else 0)
        
        # Combine LSTM predictions with static features
        X_combined = np.column_stack([
            X_test_static.values,
            lstm_predictions
        ])
        
        # Make final predictions with CatBoost
        final_predictions = self.catboost_model.predict(X_combined)
        return final_predictions
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the CatBoost component
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.catboost_model.get_feature_importance(feature_names)
    
    def save_model(self, filepath):
        """
        Save the trained hybrid model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save both components
        self.lstm_model.save_model(filepath + "_lstm_component")
        self.catboost_model.save_model(filepath + "_catboost_component")
        print(f"Hybrid model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained hybrid model
        """
        # Load both components
        self.lstm_model.load_model(filepath + "_lstm_component")
        self.catboost_model.load_model(filepath + "_catboost_component")
        self.is_trained = True
        print(f"Hybrid model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # This would be run from a training script
    pass
"""LSTM model wrapper for time-series agricultural risk prediction."""

from __future__ import annotations

import numpy as np

from models.base_model import BaseRiskModel

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:  # pragma: no cover
    Sequential = None


class LSTMRiskModel(BaseRiskModel):
    """LSTM regressor wrapper for sequence inputs."""

    def __init__(self, n_features: int) -> None:
        super().__init__("LSTM")
        self.n_features = n_features
        self.model = None
        if Sequential is not None:
            self.model = self._build_model()

    def _build_model(self):
        model = Sequential(
            [
                LSTM(32, input_shape=(None, self.n_features), return_sequences=False),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, x_train, y_train) -> None:
        if self.model is None:
            raise RuntimeError("TensorFlow is unavailable. Install tensorflow to run LSTM model.")
        self.model.fit(x_train, y_train, epochs=8, batch_size=32, validation_split=0.1, verbose=0)

    def predict(self, x_input):
        preds = self.model.predict(x_input, verbose=0)
        return np.asarray(preds).reshape(-1)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from models.base_model import BaseModel

class LSTMModel(BaseModel):
    """
    LSTM model for time series prediction of agricultural risks
    """
    
    def __init__(self, name="LSTM", sequence_length=30):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def create_sequences(self, data, target):
        """
        Create sequences for LSTM training
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model
        """
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        
        # Reshape input to be 3D [samples, timesteps, features]
        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train.shape[1]))
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        
        # Train the model
        history = self.model.fit(
            X_train_seq, 
            y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_trained = True
        print(f"{self.name} model trained successfully")
        return history
    
    def predict(self, X_test):
        """
        Make predictions using the trained LSTM model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale the data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create sequences (we need to have enough data points)
        if len(X_test_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        # For simplicity, we'll just use the last sequence_length points
        X_test_seq = X_test_scaled[-self.sequence_length:].reshape(1, self.sequence_length, X_test.shape[1])
        
        # Make predictions
        predictions = self.model.predict(X_test_seq, verbose=0)
        
        # Inverse transform to get actual values
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def save_model(self, filepath):
        """
        Save the trained LSTM model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model_weights': self.model.get_weights(),
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length
        }
        import joblib
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained LSTM model
        """
        import joblib
        model_data = joblib.load(filepath)
        
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.sequence_length = model_data['sequence_length']
        
        # Rebuild model and load weights
        # Note: This is a simplified version. In practice, you'd need to store the input shape
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # This would be run from a training script
    pass
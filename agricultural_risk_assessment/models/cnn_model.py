"""CNN model wrapper for pseudo-image agricultural risk modelling."""

from __future__ import annotations

import numpy as np

from models.base_model import BaseRiskModel

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
except Exception:  # pragma: no cover
    Sequential = None


class CNNRiskModel(BaseRiskModel):
    """CNN regressor wrapper for windowed image-like features."""

    def __init__(self, image_shape) -> None:
        super().__init__("CNN")
        self.image_shape = image_shape
        self.model = None
        if Sequential is not None:
            self.model = self._build_model()

    def _build_model(self):
        model = Sequential(
            [
                Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=self.image_shape),
                MaxPool2D((2, 2)),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                MaxPool2D((2, 2)),
                Flatten(),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, x_train, y_train) -> None:
        if self.model is None:
            raise RuntimeError("TensorFlow is unavailable. Install tensorflow to run CNN model.")
        self.model.fit(x_train, y_train, epochs=8, batch_size=32, validation_split=0.1, verbose=0)

    def predict(self, x_input):
        preds = self.model.predict(x_input, verbose=0)
        return np.asarray(preds).reshape(-1)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from models.base_model import BaseModel

class CNNModel(BaseModel):
    """
    CNN model for satellite image classification and risk assessment
    """
    
    def __init__(self, name="CNN", input_shape=(64, 64, 3)):
        super().__init__(name)
        self.input_shape = input_shape
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self, num_classes=1):
        """
        Build the CNN model architecture
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(2, 2),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            
            # Output layer
            Dense(num_classes, activation='linear' if num_classes == 1 else 'softmax')
        ])
        
        # Compile model
        if num_classes == 1:
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, num_classes=1):
        """
        Train the CNN model
        """
        # Normalize pixel values to [0, 1]
        X_train_scaled = X_train.astype('float32') / 255.0
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(num_classes)
        
        # Print model summary
        print(self.model.summary())
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, 
            y_train,
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
        Make predictions using the trained CNN model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Normalize pixel values to [0, 1]
        X_test_scaled = X_test.astype('float32') / 255.0
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled, verbose=0)
        return predictions
    
    def save_model(self, filepath):
        """
        Save the trained CNN model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save the entire model
        self.model.save(filepath + '_model.h5')
        
        model_data = {
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'input_shape': self.input_shape
        }
        import joblib
        joblib.dump(model_data, filepath + '_data.pkl')
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained CNN model
        """
        from tensorflow.keras.models import load_model
        import joblib
        
        # Load the model
        self.model = load_model(filepath + '_model.h5')
        
        # Load additional data
        model_data = joblib.load(filepath + '_data.pkl')
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.input_shape = model_data['input_shape']
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # This would be run from a training script
    pass
"""CatBoost model wrapper for agricultural risk prediction.

This module provides a CatBoost regressor wrapper optimized for agricultural 
risk assessment tasks with configurable hyperparameters.

Contributed by: Satwik Agrawal
"""

from catboost import CatBoostRegressor
import numpy as np

from config.config import SEED
from models.base_model import BaseRiskModel


class CatBoostRiskModel(BaseRiskModel):
    """CatBoost regressor wrapper."""

    def __init__(self) -> None:
        super().__init__("CatBoost")
        self.model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=SEED,
            verbose=False,
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train CatBoost model.
        
        Args:
            x_train: Training feature array
            y_train: Training target array
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Generate predictions using trained model.
        
        Args:
            x_input: Input feature array
            
        Returns:
            Predicted risk values
        """
        return self.model.predict(x_input)
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from models.base_model import BaseModel

class CatBoostModel(BaseModel):
    """
    CatBoost Regressor model for agricultural risk prediction
    """
    
    def __init__(self, name="CatBoost"):
        super().__init__(name)
        self.model = CatBoostRegressor(random_state=42, verbose=False)
        
    def train(self, X_train, y_train, use_grid_search=False):
        """
        Train the CatBoost model
        """
        # Preprocess the data
        X_train_scaled = self.preprocess_data(X_train, fit_scaler=True)
        
        if use_grid_search:
            # Define hyperparameter grid
            param_grid = {
                'iterations': [100, 200, 500],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train with default parameters
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        print(f"{self.name} model trained successfully")
    
    def predict(self, X_test):
        """
        Make predictions using the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the data
        X_test_scaled = self.preprocess_data(X_test, fit_scaler=False)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        return predictions
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importances = self.model.get_feature_importance()
        
        if feature_names is not None:
            # Return as dictionary with feature names
            return dict(zip(feature_names, importances))
        else:
            # Return as array
            return importances

# Example usage
if __name__ == "__main__":
    # This would be run from a training script
    pass
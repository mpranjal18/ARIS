"""XGBoost model wrapper for agricultural risk prediction."""

from xgboost import XGBRegressor

from config.config import SEED
from models.base_model import BaseRiskModel


class XGBoostRiskModel(BaseRiskModel):
    """XGBoost regressor wrapper."""

    def __init__(self) -> None:
        super().__init__("XGBoost")
        self.model = XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            objective="reg:squarederror",
        )

    def train(self, x_train, y_train) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_input):
        return self.model.predict(x_input)
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost Regressor model for agricultural risk prediction
    """
    
    def __init__(self, name="XGBoost"):
        super().__init__(name)
        self.model = XGBRegressor(random_state=42)
        
    def train(self, X_train, y_train, use_grid_search=False):
        """
        Train the XGBoost model
        """
        # Preprocess the data
        X_train_scaled = self.preprocess_data(X_train, fit_scaler=True)
        
        if use_grid_search:
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
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
        
        importances = self.model.feature_importances_
        
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
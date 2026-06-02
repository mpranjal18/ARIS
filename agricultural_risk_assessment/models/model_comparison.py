"""Utilities for aggregating and ranking model evaluation metrics."""

from __future__ import annotations

import pandas as pd


def to_comparison_dataframe(metrics_by_model: dict) -> pd.DataFrame:
    """Converts metrics dictionary into sortable comparison table."""
    rows = [{"Model": name, **metrics} for name, metrics in metrics_by_model.items()]
    df = pd.DataFrame(rows)
    if not df.empty and "RMSE" in df.columns:
        df = df.sort_values("RMSE").reset_index(drop=True)
    return df


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rank column based on RMSE (lower is better)."""
    out = df.copy()
    if "RMSE" in out.columns:
        out["Rank"] = out["RMSE"].rank(method="dense", ascending=True).astype(int)
        out = out.sort_values("Rank")
    return out
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import our models
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.hybrid_lstm_catboost_model import HybridLSTMCatBoostModel

class ModelComparison:
    """
    Class for comparing different machine learning models
    """
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestModel(),
            'XGBoost': XGBoostModel(),
            'CatBoost': CatBoostModel(),
            'LSTM': LSTMModel(),
            'HybridLSTMCatBoost': HybridLSTMCatBoostModel()
            # Note: CNN is not included here as it requires image data
        }
        self.results = {}
        
    def prepare_data(self, data, target_column, test_size=0.2):
        """
        Prepare data for training and testing
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            try:
                # Train model
                if name == 'LSTM':
                    # LSTM requires special handling
                    model.train(X_train.values, y_train.values, epochs=10)
                    predictions = model.predict(X_test.values)
                elif name == 'HybridLSTMCatBoost':
                    # Hybrid model
                    model.train(X_train, y_train, lstm_epochs=10)
                    predictions = model.predict(X_test)
                else:
                    # Traditional models
                    model.train(X_train, y_train)
                    predictions = model.predict(X_test)
                
                # Evaluate model
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                results[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'RMSE': np.sqrt(mse)
                }
                
                print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'MSE': np.inf, 'MAE': np.inf, 'R2': -np.inf, 'RMSE': np.inf}
        
        self.results = results
        return results
    
    def plot_comparison(self, save_path=None):
        """
        Plot model comparison
        """
        if not self.results:
            print("No results to plot. Run train_and_evaluate first.")
            return
        
        # Convert results to DataFrame for easier plotting
        df_results = pd.DataFrame(self.results).T
        df_results.reset_index(inplace=True)
        df_results.rename(columns={'index': 'Model'}, inplace=True)
        
        # Melt for easier plotting
        df_melted = df_results.melt(id_vars=['Model'], var_name='Metric', value_name='Value')
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Plot each metric
        metrics = ['MSE', 'MAE', 'R2', 'RMSE']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            data = df_results.sort_values(by=metric, ascending=(metric != 'R2'))
            bars = ax.bar(data['Model'], data[metric], color=plt.cm.viridis(np.linspace(0, 1, len(data))))
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_best_model(self):
        """
        Get the best performing model based on RMSE
        """
        if not self.results:
            print("No results available. Run train_and_evaluate first.")
            return None
        
        # Find model with lowest RMSE
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['RMSE'])
        return best_model, self.results[best_model]
    
    def generate_report(self):
        """
        Generate a comprehensive model comparison report
        """
        if not self.results:
            print("No results available. Run train_and_evaluate first.")
            return
        
        print("\n" + "="*60)
        print("MODEL COMPARISON REPORT")
        print("="*60)
        
        # Create DataFrame for better display
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.round(4)
        
        print(df_results)
        
        # Best model
        best_model, best_metrics = self.get_best_model()
        print(f"\nBest Model: {best_model}")
        print(f"Metrics: {best_metrics}")
        
        return df_results

# Example usage
if __name__ == "__main__":
    # This would be run from a training script
    pass
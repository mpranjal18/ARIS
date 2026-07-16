"""Main entrypoint for training, evaluation, artifacts, and visual outputs."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from config.config import DATA_DIR
from models.model_comparison_table import generate_model_comparison_table
from models.risk_assessment_system import AgriculturalRiskAssessmentSystem
from utils.data_handler import DataHandler
from utils.feature_engineering import add_engineered_features, build_feature_columns, merge_raw_data
from utils.risk_calculator import add_risk_targets
from visualization.data_fusion_architecture import create_data_fusion_architecture_diagram
from visualization.risk_visualizer import RiskVisualizer


def run_pipeline(sequence_len: int = 14, infer_model: str = "CatBoost") -> dict:
    """Runs complete production pipeline and returns key objects for app usage."""
    data_handler = DataHandler()
    visualizer = RiskVisualizer()
    risk_system = AgriculturalRiskAssessmentSystem(sequence_len=sequence_len)

    bundle = data_handler.load_or_generate_bundle()
    merged = merge_raw_data(bundle.weather, bundle.ndvi, bundle.pest)
    merged = add_engineered_features(merged)
    dataset = add_risk_targets(merged)

    feature_cols = build_feature_columns()
    metrics = risk_system.train_and_evaluate(dataset, feature_cols, "total_risk_score")

    dataset["predicted_total_risk"] = np.clip(
        risk_system.infer_total_risk(dataset, feature_cols, model_name=infer_model), 0, 100
    )
    dataset["prediction_gap"] = dataset["total_risk_score"] - dataset["predicted_total_risk"]

    results_path = data_handler.save_data(dataset, "risk_assessment_results.csv")
    comparison_path = os.path.join(os.path.dirname(results_path), "..", "model_comparison_table.csv")
    comparison_df = generate_model_comparison_table(metrics, os.path.abspath(comparison_path))

    visualizer.create_all_visualizations(dataset, comparison_df)
    create_data_fusion_architecture_diagram()

    return {
        "results_df": dataset,
        "comparison_df": comparison_df,
        "metrics": metrics,
        "failed_models": risk_system.failed_models,
        "results_path": results_path,
        "comparison_path": os.path.abspath(comparison_path),
    }


def main() -> None:
    """CLI execution wrapper with parameter arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="🌾 Agricultural Risk Assessment System CLI Pipeline")
    parser.add_argument(
        "--sequence-len", 
        type=int, 
        default=14, 
        help="Sequence length for memory-based models (e.g., LSTM)"
    )
    parser.add_argument(
        "--infer-model", 
        type=str, 
        default="CatBoost", 
        help="Model name to run final inference on the dataset (e.g., CatBoost, RandomForest, XGBoost)"
    )
    args = parser.parse_args()

    print("🌾 Agricultural Risk Assessment System")
    print("-" * 52)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: sequence_len={args.sequence_len}, infer_model={args.infer_model}")

    output = run_pipeline(sequence_len=args.sequence_len, infer_model=args.infer_model)

    print("\nTraining complete.")
    print(f"Risk results saved: {output['results_path']}")
    print(f"Model comparison saved: {output['comparison_path']}")
    print("Visuals saved in: visualizations/")
    print(f"\nRows processed: {len(output['results_df'])}")
    print("\nTop model metrics:")
    print(output["comparison_df"].head(3).to_string(index=False))

    if output["failed_models"]:
        print("\nModels skipped due to environment/runtime issues:")
        for model_name, reason in output["failed_models"].items():
            print(f" - {model_name}: {reason}")

    print("\nRun dashboard using: streamlit run dashboard.py")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")
VISUALIZATION_OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")

PRIMARY_REGIONS = ["Bhopal", "Sehore", "Ashta"]
SECONDARY_REGIONS = ["Raisen", "Vidisha"]

CROP_TYPE = "Wheat"
SEASON = "Rabi"

RISK_WEIGHTS = {
    "climate": 0.6,
    "pest_disease": 0.4,
}

RISK_LEVEL_THRESHOLDS = {
    "low": 33,
    "medium": 66,
}

SEED = 42

DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2025-12-31"
"""Project settings: paths, dataset description and experiment configurations."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "irrigation_prediction.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset source (pinned version, verified by SHA-256)
KAGGLE_DATASET = "miadul/irrigation-water-requirement-prediction-dataset/versions/1"
DATA_SHA256 = "3d311f2999a5857e6612fdd673762f565fe9cb788923d78abd93ad04e4ce1ab7"

# Dataset columns
TARGET = "Irrigation_Need"
CLASS_LABELS = ["Low", "Medium", "High"]

# Feature order as in the CSV file (CatBoost results depend on column order)
FEATURES = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Field_Area_hectare",
    "Mulching_Used",
    "Previous_Irrigation_mm",
    "Region",
]
CATEGORICAL_FEATURES = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# Reproducibility and evaluation protocol (same as the original notebook)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Experiments: hyperparameters of the original study, plus class-weighted
# variants that address the High class imbalance (3.36% of observations).
EXPERIMENTS = {
    "logistic_regression": {
        "model_type": "logistic_regression",
        "params": {"max_iter": 2000},
    },
    "logistic_regression_balanced": {
        "model_type": "logistic_regression",
        "params": {"max_iter": 2000, "class_weight": "balanced"},
    },
    "decision_tree": {
        "model_type": "decision_tree",
        "params": {},
    },
    "random_forest": {
        "model_type": "random_forest",
        "params": {"n_estimators": 300, "n_jobs": -1},
    },
    "catboost": {
        "model_type": "catboost",
        "params": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "MultiClass",
            "thread_count": -1,
        },
    },
    "catboost_balanced": {
        "model_type": "catboost",
        "params": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "MultiClass",
            "thread_count": -1,
            "auto_class_weights": "Balanced",
        },
    },
}

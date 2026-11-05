"""Predict irrigation need for new observations with a saved model.

Usage:
    python -m src.predict --model models/catboost/model.joblib --input new_fields.csv --output predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES


def predict_file(model_path: Path, input_path: Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    result = df.copy()
    result["Predicted_Irrigation_Need"] = np.asarray(model.predict(df[FEATURES])).ravel()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict irrigation need from a CSV file.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("predictions.csv"), type=Path)
    args = parser.parse_args()

    result = predict_file(args.model, args.input)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")
    print(result["Predicted_Irrigation_Need"].value_counts().to_string())


if __name__ == "__main__":
    main()

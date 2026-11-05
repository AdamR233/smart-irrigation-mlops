"""Train and evaluate one or all experiments from src/config.py.

Protocol (same as the original notebook):
  1. stratified 80/20 train/test split (random_state=42),
  2. stratified 5-fold cross-validation on the training part only,
  3. final fit on the whole training part and evaluation on the test part.
Preprocessing is part of the saved model, so it is fitted only on training data.

Usage:
    python -m src.train --experiment catboost
    python -m src.train --experiment all --cv-folds 0   # skip cross-validation
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    EXPERIMENTS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from src.data import load_data, prepare_data, split_data
from src.evaluate import compute_metrics, confusion_matrix_table, save_confusion_matrix_plot

SKLEARN_MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def build_model(model_type: str, params: dict):
    """Return an unfitted model that accepts the raw feature DataFrame."""
    if model_type == "catboost":
        from catboost import CatBoostClassifier  # imported only when needed

        # CatBoost handles categorical features natively, no encoding needed.
        return CatBoostClassifier(
            **params,
            cat_features=CATEGORICAL_FEATURES,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    classifier = SKLEARN_MODELS[model_type](**params, random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def predict(model, X) -> np.ndarray:
    # CatBoost returns a column vector, sklearn a flat array.
    return np.asarray(model.predict(X)).ravel()


def cross_validate(config: dict, X_train, y_train, n_folds: int) -> dict:
    """Stratified k-fold CV on the training part; returns mean and std of each metric.

    A fresh model is built for every fold, so preprocessing never sees validation data.
    """
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for train_idx, valid_idx in folds.split(X_train, y_train):
        fold_model = build_model(config["model_type"], config["params"])
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred = predict(fold_model, X_train.iloc[valid_idx])
        fold_metrics.append(compute_metrics(y_train.iloc[valid_idx], y_pred))

    table = pd.DataFrame(fold_metrics)
    summary = {}
    for metric in table.columns:
        summary[f"cv_{metric}_mean"] = float(table[metric].mean())
        summary[f"cv_{metric}_std"] = float(table[metric].std(ddof=0))
    return summary


def run_experiment(name: str, cv_folds: int = CV_FOLDS, data_path: Path = DATA_PATH,
                   output_dir: Path = MODELS_DIR) -> dict:
    """Train one experiment, save the model and its results, return the results."""
    config = EXPERIMENTS[name]
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    cv_metrics = cross_validate(config, X_train, y_train, cv_folds) if cv_folds > 1 else {}

    model = build_model(config["model_type"], config["params"])

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    y_pred = predict(model, X_test)
    test_metrics = compute_metrics(y_test, y_pred)

    experiment_dir = Path(output_dir) / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, experiment_dir / "model.joblib")
    save_confusion_matrix_plot(y_test, y_pred, experiment_dir / "confusion_matrix.png", f"{name} (test)")

    results = {
        "experiment": name,
        "model_type": config["model_type"],
        "params": config["params"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cv_folds": cv_folds,
        "training_time_s": training_time,
        "test": test_metrics,
        "cv": cv_metrics,
        "confusion_matrix": confusion_matrix_table(y_test, y_pred).to_dict(orient="index"),
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train irrigation need classifiers.")
    parser.add_argument("--experiment", default="catboost", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS, help="0 or 1 disables CV")
    args = parser.parse_args()

    prepare_data()  # download (if missing) and verify the dataset before training
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    summary = []
    for name in names:
        print(f"Training: {name}", flush=True)
        results = run_experiment(name, cv_folds=args.cv_folds)
        row = {"experiment": name, **{f"test_{k}": v for k, v in results["test"].items()}}
        if results["cv"]:
            row["cv_balanced_accuracy"] = results["cv"]["cv_balanced_accuracy_mean"]
            row["cv_macro_f1"] = results["cv"]["cv_macro_f1_mean"]
        row["time_s"] = results["training_time_s"]
        summary.append(row)

    columns = ["experiment", "test_accuracy", "test_balanced_accuracy", "test_macro_f1",
               "test_recall_High", "cv_balanced_accuracy", "cv_macro_f1", "time_s"]
    table = pd.DataFrame(summary).reindex(columns=columns).dropna(axis=1, how="all")
    print()
    print(table.sort_values("test_balanced_accuracy", ascending=False).round(4).to_string(index=False))
    print(f"\nModels and metrics saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()

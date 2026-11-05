import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    CLASS_LABELS,
    DATA_PATH,
    DATA_SHA256,
    FEATURES,
    KAGGLE_DATASET,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_data(destination: Path = DATA_PATH) -> Path:
    import kagglehub

    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = list(download_dir.rglob("*.csv"))
    if len(csv_files) != 1:
        raise RuntimeError(f"Expected one CSV file in {download_dir}, found {len(csv_files)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_files[0], destination)
    return destination


def validate_data(df: pd.DataFrame) -> None:
    expected_columns = set(FEATURES + [TARGET])
    missing = expected_columns - set(df.columns)
    unexpected = set(df.columns) - expected_columns
    if missing or unexpected:
        raise ValueError(
            f"Wrong columns. Missing: {sorted(missing)}, unexpected: {sorted(unexpected)}"
        )

    if df.isna().any().any():
        raise ValueError(f"Missing values in columns: {df.columns[df.isna().any()].tolist()}")

    non_numeric = [c for c in NUMERIC_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric values in numeric columns: {non_numeric}")

    unknown_labels = set(df[TARGET].unique()) - set(CLASS_LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown target labels: {sorted(unknown_labels)}")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_data(df)
    return df


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def prepare_data(force_download: bool = False) -> pd.DataFrame:
    if force_download or not DATA_PATH.exists():
        print(f"Downloading {KAGGLE_DATASET} ...", flush=True)
        download_data(DATA_PATH)

    checksum = file_sha256(DATA_PATH)
    if checksum != DATA_SHA256:
        raise ValueError(f"SHA-256 mismatch: expected {DATA_SHA256}, got {checksum}")
    return load_data(DATA_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate the dataset.")
    parser.add_argument("--download", action="store_true", help="force download from Kaggle")
    args = parser.parse_args()

    df = prepare_data(force_download=args.download)
    print(f"Data OK: {DATA_PATH}")
    print(f"Rows: {len(df)}, features: {len(FEATURES)} "
          f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    print("Class distribution [%]:")
    shares = df[TARGET].value_counts(normalize=True).reindex(CLASS_LABELS) * 100
    print(shares.round(2).to_string())


if __name__ == "__main__":
    main()

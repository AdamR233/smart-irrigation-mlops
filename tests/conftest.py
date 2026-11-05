import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def make_dataset(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 100, n_rows).round(2) for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice([f"{col}_A", f"{col}_B", f"{col}_C"], n_rows)

    moisture = data["Soil_Moisture"]
    data[TARGET] = np.where(moisture < 15, "High", np.where(moisture < 50, "Medium", "Low"))
    return pd.DataFrame(data)[FEATURES + [TARGET]]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return make_dataset()


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    return path

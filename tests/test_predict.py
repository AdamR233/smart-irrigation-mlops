import pandas as pd
import pytest

from src.config import CLASS_LABELS
from src.predict import predict_file
from src.train import run_experiment


@pytest.fixture
def model_path(sample_csv, tmp_path):
    run_experiment("decision_tree", cv_folds=0, data_path=sample_csv, output_dir=tmp_path)
    return tmp_path / "decision_tree" / "model.joblib"


def test_predict_adds_column_with_valid_labels(model_path, sample_csv):
    result = predict_file(model_path, sample_csv)

    assert len(result) == len(pd.read_csv(sample_csv))
    assert set(result["Predicted_Irrigation_Need"]).issubset(CLASS_LABELS)


def test_predict_rejects_missing_columns(model_path, sample_csv, tmp_path):
    incomplete = tmp_path / "incomplete.csv"
    pd.read_csv(sample_csv).drop(columns="Region").to_csv(incomplete, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        predict_file(model_path, incomplete)

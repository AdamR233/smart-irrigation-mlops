import json

import pytest

from src.config import EXPERIMENTS
from src.train import run_experiment


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_experiment_runs_and_saves_outputs(name, sample_csv, tmp_path):
    if EXPERIMENTS[name]["model_type"] == "catboost":
        pytest.importorskip("catboost")

    results = run_experiment(name, cv_folds=2, data_path=sample_csv, output_dir=tmp_path)

    experiment_dir = tmp_path / name
    for file_name in ("model.joblib", "metrics.json", "confusion_matrix.png"):
        assert (experiment_dir / file_name).exists()

    saved = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["test_rows"] == 60
    assert "cv_balanced_accuracy_mean" in saved["cv"]
    for value in results["test"].values():
        assert 0.0 <= value <= 1.0


def test_model_learns_simple_rule(sample_csv, tmp_path):
    results = run_experiment("decision_tree", cv_folds=0, data_path=sample_csv, output_dir=tmp_path)
    assert results["test"]["balanced_accuracy"] > 0.9

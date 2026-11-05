import pytest

from src.config import CLASS_LABELS
from src.evaluate import compute_metrics, confusion_matrix_table, save_confusion_matrix_plot

Y_TRUE = ["Low", "Low", "Medium", "High"]
Y_PRED = ["Low", "Medium", "Medium", "Low"]


def test_perfect_predictions():
    metrics = compute_metrics(Y_TRUE, Y_TRUE)
    assert all(value == 1.0 for value in metrics.values())


def test_metrics_on_known_example():
    metrics = compute_metrics(Y_TRUE, Y_PRED)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["recall_Low"] == pytest.approx(0.5)
    assert metrics["recall_Medium"] == pytest.approx(1.0)
    assert metrics["recall_High"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_metrics_contain_every_class():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    for label in CLASS_LABELS:
        for name in ("precision", "recall", "f1"):
            assert f"{name}_{label}" in metrics


def test_confusion_matrix_order_and_values():
    table = confusion_matrix_table(Y_TRUE, Y_PRED)

    assert list(table.index) == CLASS_LABELS
    assert list(table.columns) == CLASS_LABELS
    assert table.loc["Low", "Medium"] == 1
    assert table.loc["High", "Low"] == 1
    assert table.to_numpy().sum() == len(Y_TRUE)


def test_confusion_matrix_plot_is_saved(tmp_path):
    path = tmp_path / "cm.png"
    save_confusion_matrix_plot(Y_TRUE, Y_PRED, path, "test")
    assert path.exists() and path.stat().st_size > 0

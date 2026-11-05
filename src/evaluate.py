"""Classification metrics and confusion matrix (class order: Low, Medium, High)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (Docker, Jenkins)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import CLASS_LABELS


def compute_metrics(y_true, y_pred) -> dict:
    """Overall metrics plus precision, recall and F1 for each class."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    for i, label in enumerate(CLASS_LABELS):
        metrics[f"precision_{label}"] = precision[i]
        metrics[f"recall_{label}"] = recall[i]
        metrics[f"f1_{label}"] = f1[i]
    return {name: float(value) for name, value in metrics.items()}


def confusion_matrix_table(y_true, y_pred) -> pd.DataFrame:
    """Rows: true class, columns: predicted class."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    return pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS)


def save_confusion_matrix_plot(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=CLASS_LABELS, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

"""Evaluation of the 5-class ride classifier.

The classes are very imbalanced (GBG500: 416 Safe, 38 Safe overall, 22 Bad weather, 18 Tandem,
6 Reckless). Accuracy is misleading (always answering "Safe" gives 83 %), so the headline metric
is the **macro-F1** (every class weighs the same), plus balanced accuracy and per-class recall.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from fraud_detection.config import LABELS  # noqa: E402

_ALL = list(range(len(LABELS)))


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=_ALL, average=None, zero_division=0)
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=_ALL, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    metrics.update({f"recall_{_slug(label)}": float(r) for label, r in zip(LABELS, recalls, strict=True)})
    return metrics


def majority_baseline_macro_f1(y_true: np.ndarray) -> float:
    """Macro-F1 of a model that always predicts the most frequent class ("Safe")."""
    majority = np.bincount(y_true, minlength=len(LABELS)).argmax()
    return float(f1_score(y_true, np.full_like(y_true, majority), labels=_ALL, average="macro", zero_division=0))


def report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, labels=_ALL, target_names=LABELS, zero_division=0)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=_ALL), display_labels=LABELS)
    disp.plot(values_format="d", xticks_rotation=30, colorbar=False)
    disp.ax_.set_title(title)
    path = out_dir / "confusion_matrix.png"
    disp.figure_.savefig(path, bbox_inches="tight")
    plt.close(disp.figure_)
    return path

import numpy as np

from fraud_detection.config import LABELS
from fraud_detection.evaluate import (
    compute_metrics,
    majority_baseline_macro_f1,
    report_text,
    save_confusion_matrix,
)


def test_perfect_predictions():
    y = np.arange(len(LABELS))
    m = compute_metrics(y, y)
    assert m["macro_f1"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert all(m[f"recall_{label.lower().replace(' ', '_')}"] == 1.0 for label in LABELS)


def test_always_safe_has_high_accuracy_but_low_macro_f1():
    """Why accuracy is not the headline metric on GBG500."""
    y = np.array([0] * 83 + [1, 1, 2, 2, 3, 3, 4] + [0] * 10)
    m = compute_metrics(y, np.zeros_like(y))
    assert m["accuracy"] > 0.9
    assert m["macro_f1"] < 0.25
    assert m["recall_reckless"] == 0.0


def test_majority_baseline_matches_always_safe():
    y = np.array([0, 0, 0, 1, 4])
    assert majority_baseline_macro_f1(y) == compute_metrics(y, np.zeros_like(y))["macro_f1"]


def test_report_and_confusion_matrix(tmp_path):
    y = np.array([0, 1, 2, 3, 4, 0])
    assert "Reckless" in report_text(y, y)
    path = save_confusion_matrix(y, y, tmp_path, "test")
    assert path.exists() and path.stat().st_size > 0

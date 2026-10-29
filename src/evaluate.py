"""Evaluation helpers for binary churn classification."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true: Any, y_probability: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate binary labels and predicted probabilities at an explicit threshold."""
    y = np.asarray(y_true).reshape(-1)
    probability = np.asarray(y_probability, dtype=float).reshape(-1)

    if len(y) == 0 or len(y) != len(probability):
        raise ValueError("y_true and y_probability must have the same non-zero length.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if len(np.unique(y)) < 2:
        raise ValueError("Evaluation requires both classes to be present in y_true.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("y_probability must contain finite values between 0 and 1.")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1.")

    y_pred = (probability >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    return {
        "threshold": float(threshold),
        "metrics": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_model(model: Any, X: Any, y_true: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate an already-fitted binary classifier with predict_proba support."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("model must expose predict_proba for probability-based evaluation.")
    probabilities = model.predict_proba(X)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("predict_proba must return an array with a positive-class column.")
    return evaluate_predictions(y_true, probabilities[:, 1], threshold=threshold)


def evaluate_calibration(
    y_true: Any,
    y_probability: Any,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Summarize probability calibration on a separate validation sample."""
    y = np.asarray(y_true).reshape(-1)
    probability = np.asarray(y_probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(probability):
        raise ValueError("y_true and y_probability must have the same non-zero length.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("y_probability must contain finite values between 0 and 1.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    bin_ids = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    bins = []
    expected_calibration_error = 0.0
    max_calibration_error = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(probability[mask].mean())
        observed_rate = float(y[mask].mean())
        calibration_error = abs(mean_probability - observed_rate)
        expected_calibration_error += (count / len(y)) * calibration_error
        max_calibration_error = max(max_calibration_error, calibration_error)
        bins.append(
            {
                "bin": bin_index,
                "count": count,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_error": calibration_error,
            }
        )

    return {
        "sample_count": int(len(y)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "expected_calibration_error": float(expected_calibration_error),
        "maximum_calibration_error": float(max_calibration_error),
        "n_bins_requested": int(n_bins),
        "n_bins_nonempty": int(len(bins)),
        "reliability_bins": bins,
        "interpretation": (
            "Diagnostics are computed on the supplied validation sample. Lower Brier/log loss/ECE "
            "are generally preferable, but these estimates are sample-dependent and do not prove "
            "perfect calibration."
        ),
    }

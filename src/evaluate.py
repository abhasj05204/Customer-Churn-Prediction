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

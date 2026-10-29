"""Tests for validation-only threshold and one-standard-error selection."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.finalize import (
    compare_calibration_methods,
    select_f1_threshold,
    select_model_one_standard_error,
)


class FinalizationTests(unittest.TestCase):
    def test_one_standard_error_rule_prefers_simpler_model_when_close(self) -> None:
        baseline = {
            "Logistic Regression (balanced)": {"roc_auc": {"mean": 0.846, "std": 0.013}},
            "Random Forest (balanced)": {"roc_auc": {"mean": 0.827, "std": 0.012}},
            "XGBoost (no resampling)": {"roc_auc": {"mean": 0.842, "std": 0.010}},
        }
        selected, details = select_model_one_standard_error(
            baseline_results=baseline,
            tuned_xgb_mean=0.850,
            tuned_xgb_std=0.013,
            n_splits=5,
        )
        self.assertEqual(selected, "Logistic Regression (balanced)")
        self.assertIn(selected, details["eligible_models"])
        self.assertAlmostEqual(details["best_mean_cv_roc_auc"], 0.850)

    def test_threshold_selection_uses_f1_and_returns_metrics(self) -> None:
        y = [0, 0, 1, 1]
        probabilities = [0.10, 0.40, 0.35, 0.80]
        threshold, details = select_f1_threshold(
            y, probabilities, thresholds=[0.20, 0.30, 0.35, 0.40, 0.50, 0.60]
        )
        self.assertIn(threshold, [0.30, 0.35])
        self.assertEqual(details["selection_metric"], "F1")
        self.assertEqual(details["candidate_threshold_count"], 6)
        self.assertGreaterEqual(details["selected_validation_metrics"]["f1"], 0.75)

    def test_calibration_method_selection_uses_training_only_oof_predictions(self) -> None:
        size = 90
        tenure = np.arange(size, dtype=float)
        X = pd.DataFrame({
            "tenure": tenure,
            "monthly": 20.0 + (tenure % 20),
            "contract": np.where(tenure % 3 == 0, "Month-to-month", "One year"),
        })
        y = pd.Series((tenure > 44).astype("int8"), name="State")
        method, report = compare_calibration_methods(
            X,
            y,
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
            n_splits=3,
            random_state=42,
        )
        self.assertIn(method, {"none", "sigmoid", "isotonic"})
        self.assertEqual(report["selected_method"], method)
        self.assertEqual(report["selection_data"], "training-only out-of-fold probabilities from X_model")
        self.assertEqual(set(report["candidate_metrics"]), {"none", "sigmoid", "isotonic"})
        for metrics in report["candidate_metrics"].values():
            self.assertEqual(metrics["oof_sample_count"], len(X))
            self.assertGreaterEqual(metrics["brier_score"], 0.0)
            self.assertGreaterEqual(metrics["log_loss"], 0.0)

    def test_rejects_invalid_threshold_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            select_f1_threshold([0, 1], [0.1])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            select_f1_threshold([0, 1], [0.1, 1.4])
        with self.assertRaisesRegex(ValueError, "strictly between"):
            select_f1_threshold([0, 1], [0.1, 0.9], thresholds=[0.0, 0.5])


if __name__ == "__main__":
    unittest.main()

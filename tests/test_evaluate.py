"""Tests for classifier evaluation helpers."""
import unittest

import numpy as np

from src.evaluate import evaluate_calibration, evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_confusion_matrix(self) -> None:
        result = evaluate_predictions(
            [0, 0, 1, 1],
            [0.1, 0.7, 0.4, 0.9],
            threshold=0.5,
        )
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(result["confusion_counts"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertAlmostEqual(result["threshold"], 0.5)
        self.assertIn("average_precision", result["metrics"])
        self.assertIn("brier_score", result["metrics"])


    def test_calibration_reports_brier_logloss_and_reliability_bins(self) -> None:
        report = evaluate_calibration(
            [0, 0, 1, 1, 1, 0],
            [0.05, 0.25, 0.65, 0.80, 0.90, 0.35],
            n_bins=4,
        )
        self.assertEqual(report["sample_count"], 6)
        self.assertIn("brier_score", report)
        self.assertIn("log_loss", report)
        self.assertIn("expected_calibration_error", report)
        self.assertGreaterEqual(report["expected_calibration_error"], 0.0)
        self.assertLessEqual(report["expected_calibration_error"], 1.0)
        self.assertGreater(report["n_bins_nonempty"], 0)

    def test_calibration_rejects_bad_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            evaluate_calibration([0, 1], [0.2])
        with self.assertRaisesRegex(ValueError, "n_bins must be at least 2"):
            evaluate_calibration([0, 1], [0.1, 0.9], n_bins=1)

    def test_rejects_invalid_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate_predictions([0, 1], [0.2, 1.5])
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            evaluate_predictions([0, 1], [0.2])
        with self.assertRaisesRegex(ValueError, "both classes"):
            evaluate_predictions([1, 1], [0.2, 0.9])
        with self.assertRaisesRegex(ValueError, "strictly between"):
            evaluate_predictions([0, 1], [0.2, 0.9], threshold=1.0)


if __name__ == "__main__":
    unittest.main()

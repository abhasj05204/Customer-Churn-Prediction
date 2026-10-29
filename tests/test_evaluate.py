"""Tests for classifier evaluation helpers."""
import unittest

import numpy as np

from src.evaluate import evaluate_predictions


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

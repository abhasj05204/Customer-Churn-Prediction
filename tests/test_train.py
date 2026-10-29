"""Tests for the training orchestration without running the full CV experiment."""
import unittest

import pandas as pd

from src.train import candidate_estimators, cross_validate_candidates


class TrainingTests(unittest.TestCase):
    def test_candidate_models_are_defined(self) -> None:
        estimators = candidate_estimators()
        self.assertIn("Logistic Regression (balanced)", estimators)
        self.assertIn("Random Forest (balanced)", estimators)
        self.assertIn("XGBoost (no resampling)", estimators)

    def test_cross_validation_validates_fold_counts(self) -> None:
        X = pd.DataFrame({"tenure": [1, 2, 3, 4, 5, 6], "contract": ["m", "y", "t", "m", "y", "t"]})
        y = pd.Series([0, 1, 0, 1, 0, 1], dtype="int8")
        with self.assertRaisesRegex(ValueError, "no greater than the smallest class count"):
            cross_validate_candidates(X, y, n_splits=4)


if __name__ == "__main__":
    unittest.main()

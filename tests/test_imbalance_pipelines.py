"""Tests for imbalance-aware preprocessing pipeline builders."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocess import build_smote_pipeline, build_smotenc_pipeline


def mixed_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rows = 60
    X = pd.DataFrame({
        "tenure": np.arange(rows, dtype=float),
        "MonthlyCharges": np.linspace(20.0, 110.0, rows),
        "TotalCharges": np.linspace(20.0, 2500.0, rows),
        "Contract": (["Month-to-month", "One year", "Two year"] * 20),
        "InternetService": (["DSL", "Fiber optic", "No"] * 20),
    })
    y = pd.Series([0] * 40 + [1] * 20, name="State", dtype="int8")
    return X, y


class ImbalancePipelineTests(unittest.TestCase):
    def test_ordinary_smote_pipeline_fits_only_during_fit(self) -> None:
        X, y = mixed_training_data()
        model = build_smote_pipeline(
            X, LogisticRegression(max_iter=2000, random_state=42), random_state=42
        )
        model.fit(X, y)
        probabilities = model.predict_proba(X[:7])
        self.assertEqual(probabilities.shape, (7, 2))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_smotenc_preserves_mixed_feature_contract(self) -> None:
        X, y = mixed_training_data()
        model = build_smotenc_pipeline(
            X, LogisticRegression(max_iter=2000, random_state=42), random_state=42
        )
        model.fit(X, y)
        probabilities = model.predict_proba(X[:7])
        self.assertEqual(probabilities.shape, (7, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertIn("sampler", model.named_steps)


if __name__ == "__main__":
    unittest.main()

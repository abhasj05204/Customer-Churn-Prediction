"""Tests for leakage-safe preprocessing."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import build_model_pipeline, build_preprocessor


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = pd.DataFrame(
            {
                "tenure": [1.0, 12.0, 24.0, np.nan, 48.0, 60.0],
                "contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year"],
                "segment": pd.Categorical(["A", "A", "B", "B", "C", "C"]),
            }
        )
        self.y = pd.Series([1, 1, 0, 1, 0, 0], name="State")

    def test_preprocessor_is_unfitted_until_fit(self) -> None:
        preprocessor = build_preprocessor(self.X)
        self.assertFalse(hasattr(preprocessor, "transformers_"))
        transformed = preprocessor.fit_transform(self.X)
        self.assertEqual(transformed.shape[0], len(self.X))
        self.assertTrue(np.isfinite(transformed).all())

    def test_unknown_categories_are_supported(self) -> None:
        preprocessor = build_preprocessor(self.X)
        preprocessor.fit(self.X)
        new_rows = pd.DataFrame(
            {"tenure": [36.0], "contract": ["New contract"], "segment": ["Z"]}
        )
        transformed = preprocessor.transform(new_rows)
        self.assertEqual(transformed.shape[0], 1)
        self.assertEqual(transformed.shape[1], preprocessor.transform(self.X).shape[1])

    def test_model_pipeline_fits_and_predicts(self) -> None:
        pipeline = build_model_pipeline(
            self.X,
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
        self.assertIsInstance(pipeline, Pipeline)
        pipeline.fit(self.X, self.y)
        predictions = pipeline.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_rejects_empty_features_and_duplicate_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one row and one feature"):
            build_preprocessor(pd.DataFrame())
        duplicated = pd.DataFrame([[1, 2]], columns=["x", "x"])
        with self.assertRaisesRegex(ValueError, "duplicate feature names"):
            build_preprocessor(duplicated)


if __name__ == "__main__":
    unittest.main()

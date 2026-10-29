"""Unit tests for SHAP's probability-prediction adapter."""
import unittest

import numpy as np
import pandas as pd

from src.explain import make_probability_predictor


class FixedProbabilityModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = frame["tenure"].to_numpy(dtype=float) / 10.0
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))


class ExplainabilityTests(unittest.TestCase):
    def test_adapter_preserves_columns_and_predicts_from_array(self) -> None:
        template = pd.DataFrame(
            {
                "tenure": [1.0, 5.0],
                "contract": pd.Series(["Month-to-month", "One year"], dtype="object"),
            }
        )
        predictor = make_probability_predictor(FixedProbabilityModel(), template)
        probabilities = predictor(
            np.array([[2.0, "Month-to-month"], [8.0, "One year"]], dtype=object)
        )
        np.testing.assert_allclose(probabilities, [0.2, 0.8])

    def test_adapter_rejects_wrong_feature_width(self) -> None:
        template = pd.DataFrame({"tenure": [1.0], "contract": ["Month-to-month"]})
        predictor = make_probability_predictor(FixedProbabilityModel(), template)
        with self.assertRaisesRegex(ValueError, "Expected rows with 2 columns"):
            predictor(np.array([[1.0]]))

    def test_adapter_rejects_invalid_probability_output(self) -> None:
        class BadModel:
            def predict_proba(self, frame):
                return np.array([[0.5, 1.5]] * len(frame))

        template = pd.DataFrame({"tenure": [1.0], "contract": ["m"]})
        predictor = make_probability_predictor(BadModel(), template)
        with self.assertRaisesRegex(ValueError, "invalid positive-class probabilities"):
            predictor(np.array([[1.0, "m"]], dtype=object))


if __name__ == "__main__":
    unittest.main()

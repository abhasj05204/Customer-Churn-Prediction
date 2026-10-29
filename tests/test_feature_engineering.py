"""Tests for reusable churn feature engineering."""
import unittest

import pandas as pd

from src.feature_engineering import engineer_features, engineer_inference_features, split_features_target


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [2, 0],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["Yes", "No phone service"],
            "InternetService": ["DSL", "No"],
            "OnlineSecurity": ["Yes", "No internet service"],
            "OnlineBackup": ["No", "No internet service"],
            "DeviceProtection": ["No", "No internet service"],
            "TechSupport": ["Yes", "No internet service"],
            "StreamingTV": ["No", "No internet service"],
            "StreamingMovies": ["Yes", "No internet service"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [80.0, 25.0],
            "TotalCharges": [160.0, 1397.475],
            "State": ["Churn", "No Churn"],
            "StateNextMonth": ["Churn", "No Churn"],
        }
    )


class FeatureEngineeringTests(unittest.TestCase):
    def test_engineers_features_without_mutating_source(self) -> None:
        df = sample_frame()
        original = df.copy(deep=True)
        engineered = engineer_features(df)
        pd.testing.assert_frame_equal(df, original)
        self.assertNotIn("customerID", engineered.columns)
        self.assertNotIn("State", engineered.columns)
        self.assertEqual(engineered["StateNextMonth"].tolist(), [1, 0])
        self.assertEqual(engineered["ServiceCount"].tolist(), [5, 0])
        self.assertEqual(engineered["SecuritySupportCount"].tolist(), [2, 0])
        self.assertEqual(engineered["AvgMonthlyCharges"].tolist(), [80.0, 25.0])

    def test_split_features_target_excludes_target_and_identifier(self) -> None:
        engineered = engineer_features(sample_frame())
        X, y = split_features_target(engineered)
        self.assertNotIn("State", X.columns)
        self.assertNotIn("customerID", X.columns)
        self.assertNotIn("StateNextMonth", X.columns)
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(len(X), len(y))


    def test_inference_features_do_not_require_target(self) -> None:
        raw = sample_frame().drop(columns=["customerID", "State", "StateNextMonth"])
        features = engineer_inference_features(raw)
        self.assertNotIn("State", features.columns)
        self.assertNotIn("customerID", features.columns)
        self.assertEqual(len(features), 2)
        self.assertIn("TenureGroup", features.columns)
        self.assertIn("AvgMonthlyCharges", features.columns)


    def test_unreviewed_extra_columns_do_not_enter_features(self) -> None:
        raw = sample_frame().drop(columns=["customerID", "State", "StateNextMonth"])
        raw["future_churn_label"] = ["Churn", "No Churn"]
        features = engineer_inference_features(raw)
        self.assertNotIn("future_churn_label", features.columns)
        self.assertEqual(len(features.columns), len(raw.columns) + 4)

    def test_rejects_invalid_state_next_month(self) -> None:
        df = sample_frame()
        df.loc[0, "StateNextMonth"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Invalid StateNextMonth target"):
            engineer_features(df)

    def test_rejects_missing_required_feature(self) -> None:
        df = sample_frame().drop(columns=["Contract"])
        with self.assertRaisesRegex(ValueError, "Missing required"):
            engineer_features(df)


if __name__ == "__main__":
    unittest.main()

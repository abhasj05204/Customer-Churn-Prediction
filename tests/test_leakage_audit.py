"""Tests for structural target-leakage and temporal audit flags."""
import unittest

import pandas as pd

from src.audit_leakage import audit_feature_leakage
from src.feature_engineering import REQUIRED_INPUT_COLUMNS


def valid_source_frame(include_next_month: bool = False) -> pd.DataFrame:
    row = {
        "customerID": "C-1",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "State": "No Churn",
    }
    df = pd.DataFrame([row], columns=REQUIRED_INPUT_COLUMNS)
    if include_next_month:
        df["StateNextMonth"] = "Churn"
    return df


class LeakageAuditTests(unittest.TestCase):
    def test_known_labels_and_identifiers_are_excluded(self) -> None:
        report = audit_feature_leakage(valid_source_frame(include_next_month=True))
        self.assertEqual(report["structural_leakage_check"]["status"], "pass")
        self.assertEqual(report["blocked_columns_in_predictor_frame"], [])
        self.assertNotIn("customerID", report["predictor_columns"])
        self.assertNotIn("State", report["predictor_columns"])
        self.assertNotIn("StateNextMonth", report["predictor_columns"])
        self.assertTrue(report["state_next_month_present_in_source"])

    def test_temporal_availability_is_not_overclaimed(self) -> None:
        report = audit_feature_leakage(valid_source_frame())
        self.assertEqual(report["feature_timing_check"]["status"], "not_verifiable")
        self.assertFalse(report["state_next_month_present_in_source"])
        self.assertTrue(any("Temporal leakage cannot be ruled out" in w for w in report["warnings"]))

    def test_rejects_invalid_target_labels(self) -> None:
        df = valid_source_frame()
        df.loc[0, "State"] = "Unknown"
        report = audit_feature_leakage(df)
        self.assertEqual(report["structural_leakage_check"]["status"], "fail")
        self.assertEqual(report["target_label_errors"]["unknown_values"], ["Unknown"])


    def test_flags_unreviewed_source_columns(self) -> None:
        df = valid_source_frame()
        df["future_churn_label"] = "Churn"
        report = audit_feature_leakage(df)
        self.assertEqual(report["unexpected_source_columns"], ["future_churn_label"])
        self.assertEqual(report["structural_leakage_check"]["status"], "fail")
        self.assertNotIn("future_churn_label", report["predictor_columns"])

    def test_rejects_missing_predictor_columns(self) -> None:
        df = valid_source_frame().drop(columns=["Contract"])
        with self.assertRaisesRegex(ValueError, "Missing raw predictor columns"):
            audit_feature_leakage(df)


if __name__ == "__main__":
    unittest.main()

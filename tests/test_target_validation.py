"""Unit tests for validation of the supported recorded-State target."""
import unittest
import pandas as pd
from src.validate_target import audit_target_columns


class TargetAuditTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "customerID": ["A", "B", "C", "D"],
            "State": ["No Churn", "No Churn", "Churn", "No Churn"],
        })

    def test_audits_recorded_state_without_future_label(self):
        report = audit_target_columns(self.df)
        self.assertEqual(report["target_column"], "State")
        self.assertEqual(report["target_counts"], {"No Churn": 3, "Churn": 1})
        self.assertAlmostEqual(report["target_churn_rate"], 0.25)
        self.assertFalse(report["state_next_month_available"])

    def test_future_label_is_optional_and_unverified(self):
        df = self.df.copy()
        df["StateNextMonth"] = ["Churn", "No Churn", "Churn", "No Churn"]
        report = audit_target_columns(df)
        self.assertTrue(report["state_next_month_available"])
        self.assertEqual(report["future_label_provenance"]["status"], "unverified")
        self.assertEqual(report["target_column"], "State")

    def test_future_label_is_not_required_or_validated_for_training(self):
        df = self.df.copy()
        df["StateNextMonth"] = ["synthetic", None, "unknown", "future"]
        report = audit_target_columns(df)
        self.assertEqual(report["target_column"], "State")
        self.assertFalse(report["state_next_month_valid_labels"])

    def test_rejects_missing_required_column(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            audit_target_columns(self.df.drop(columns=["State"]))

    def test_rejects_unknown_state_label(self):
        df = self.df.copy()
        df.loc[0, "State"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Unexpected/missing State labels"):
            audit_target_columns(df)

    def test_reports_duplicate_customer_ids(self):
        duplicated = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        report = audit_target_columns(duplicated)
        self.assertEqual(report["rows_with_duplicate_customer_id"], 2)


if __name__ == "__main__":
    unittest.main()

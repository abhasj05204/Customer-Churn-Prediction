"""Unit tests for target validation (standard-library unittest)."""
import unittest

import pandas as pd

from src.validate_target import audit_target_columns


class TargetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "customerID": ["A", "B", "C", "D"],
                "State": ["No Churn", "No Churn", "Churn", "No Churn"],
            }
        )

    def test_audits_dataset_without_future_label(self) -> None:
        report = audit_target_columns(self.df)
        self.assertEqual(report["row_count"], 4)
        self.assertEqual(report["column_count"], 2)
        self.assertEqual(report["target_counts"], {"No Churn": 3, "Churn": 1})
        self.assertAlmostEqual(report["target_churn_rate"], 0.25)
        self.assertFalse(report["state_next_month_available"])
        self.assertFalse(report["next_month_analysis"]["available"])

    def test_optional_future_label_is_audited_when_present(self) -> None:
        with_future = self.df.copy()
        with_future["StateNextMonth"] = ["Churn", "No Churn", "Churn", "No Churn"]
        report = audit_target_columns(with_future)
        self.assertTrue(report["state_next_month_available"])
        self.assertTrue(report["next_month_analysis"]["available"])
        self.assertEqual(report["next_month_analysis"]["candidate_cohort_rows"], 3)
        self.assertEqual(report["next_month_analysis"]["next_month_churn_count"], 1)
        self.assertAlmostEqual(
            report["next_month_analysis"]["observed_next_month_churn_rate"], 1 / 3
        )

    def test_rejects_missing_required_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            audit_target_columns(self.df.drop(columns=["State"]))

    def test_rejects_unknown_target_label(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "State"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Unexpected/missing State labels"):
            audit_target_columns(bad)

    def test_reports_duplicate_customer_ids(self) -> None:
        duplicated = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        report = audit_target_columns(duplicated)
        self.assertEqual(report["rows_with_duplicate_customer_id"], 2)


if __name__ == "__main__":
    unittest.main()

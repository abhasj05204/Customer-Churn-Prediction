"""Unit tests for the target-label audit (standard-library unittest)."""
import unittest

import pandas as pd

from src.validate_target import audit_target_columns


class TargetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "customerID": ["A", "B", "C", "D"],
                "State": ["No Churn", "No Churn", "Churn", "No Churn"],
                "StateNextMonth": ["Churn", "No Churn", "Churn", "No Churn"],
            }
        )

    def test_counts_and_candidate_active_cohort(self) -> None:
        report = audit_target_columns(self.df)
        self.assertEqual(report["row_count"], 4)
        self.assertEqual(report["state_counts"], {"No Churn": 3, "Churn": 1})
        self.assertEqual(report["state_next_month_counts_all_rows"], {"No Churn": 2, "Churn": 2})
        self.assertEqual(report["state_by_state_next_month"]["No Churn"], {
            "No Churn": 2, "Churn": 1
        })
        self.assertEqual(report["candidate_active_cohort"]["row_count"], 3)
        self.assertEqual(report["candidate_active_cohort"]["next_month_churn_count"], 1)
        self.assertAlmostEqual(
            report["candidate_active_cohort"]["next_month_churn_rate"], 1 / 3
        )

    def test_rejects_missing_required_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required"):
            audit_target_columns(self.df.drop(columns=["StateNextMonth"]))

    def test_rejects_unknown_target_label(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "State"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Unexpected/missing target labels"):
            audit_target_columns(bad)

    def test_reports_duplicate_customer_rows(self) -> None:
        duplicated = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        report = audit_target_columns(duplicated)
        self.assertEqual(report["duplicate_customer_rows"], 2)


if __name__ == "__main__":
    unittest.main()

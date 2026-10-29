"""Tests for the dataset quality audit."""
import unittest

import pandas as pd

from src.validate_data import audit_dataset


class DataAuditTests(unittest.TestCase):
    def test_basic_schema_and_target_counts(self) -> None:
        df = pd.DataFrame({
            "customerID": ["A", "B", "C"],
            "State": ["No Churn", "Churn", "No Churn"],
            "tenure": [1, 4, 0],
            "TotalCharges": [30.0, 150.0, 1397.475],
        })
        report = audit_dataset(df)
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["target_counts"], {"No Churn": 2, "Churn": 1})
        self.assertEqual(report["anomalies"]["zero_tenure_rows"], 1)
        self.assertEqual(report["anomalies"]["zero_tenure_with_positive_total_charges"], 1)

    def test_flags_duplicates_and_missing_values(self) -> None:
        df = pd.DataFrame({
            "customerID": ["A", "A", "B"],
            "State": ["No Churn", "No Churn", None],
        })
        report = audit_dataset(df)
        self.assertEqual(report["exact_duplicate_rows"], 1)
        self.assertEqual(report["rows_with_duplicate_customer_id"], 2)
        self.assertEqual(report["missing_cells_total"], 1)
        self.assertEqual(report["missing_target_labels"], 1)

    def test_reports_invalid_numeric_values(self) -> None:
        df = pd.DataFrame({
            "customerID": ["A", "B"],
            "State": ["No Churn", "Churn"],
            "tenure": ["not-a-number", -2],
            "TotalCharges": [10.0, 15.0],
        })
        report = audit_dataset(df)
        self.assertEqual(report["nonnumeric_counts"]["tenure"], 1)
        self.assertEqual(report["negative_counts"]["tenure"], 1)

    def test_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            audit_dataset(pd.DataFrame({"State": ["No Churn"]}))


if __name__ == "__main__":
    unittest.main()

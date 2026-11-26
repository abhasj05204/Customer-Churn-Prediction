"""Run a descriptive quality audit of the selected Telco churn dataset.

The audit flags suspicious combinations for review but does not automatically
change or drop records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import REQUIRED_INPUT_COLUMNS

# Keep validation and the training/inference contract synchronized.
EXPECTED_COLUMNS = REQUIRED_INPUT_COLUMNS
REQUIRED_COLUMNS = ("customerID", "State")
NUMERIC_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
NONNEGATIVE_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
VALID_TARGET_LABELS = ("No Churn", "Churn")
VALID_CATEGORICAL_VALUES = {
    "gender": ("Female", "Male"),
    "Partner": ("Yes", "No"),
    "Dependents": ("Yes", "No"),
    "PhoneService": ("Yes", "No"),
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("Yes", "No", "No internet service"),
    "OnlineBackup": ("Yes", "No", "No internet service"),
    "DeviceProtection": ("Yes", "No", "No internet service"),
    "TechSupport": ("Yes", "No", "No internet service"),
    "StreamingTV": ("Yes", "No", "No internet service"),
    "StreamingMovies": ("Yes", "No", "No internet service"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("Yes", "No"),
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return schema, missingness, duplicate, numeric, target, and anomaly checks."""
    if df.empty:
        raise ValueError("Cannot audit an empty dataset.")

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected_columns = [str(c) for c in df.columns if c not in EXPECTED_COLUMNS]
    missing_counts = {str(c): int(n) for c, n in df.isna().sum().items() if n > 0}
    duplicate_rows = int(df.duplicated().sum())

    customer_ids = df["customerID"].astype("string").str.strip()
    duplicate_id_rows = int(customer_ids.duplicated(keep=False).sum())
    target = df["State"].astype("string").str.strip()
    target_counts = {label: int(target.eq(label).sum()) for label in VALID_TARGET_LABELS}
    unknown_targets = sorted(
        str(v) for v in target[~target.isin(VALID_TARGET_LABELS)].dropna().unique()
    )
    missing_target = int(target.isna().sum())

    numeric_summary: dict[str, Any] = {}
    negative_counts: dict[str, int] = {}
    nonnumeric_counts: dict[str, int] = {}
    nonfinite_counts: dict[str, int] = {}
    numeric_domain_errors: dict[str, dict[str, int]] = {}
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        nonnumeric_counts[column] = int((df[column].notna() & converted.isna()).sum())
        finite_mask = np.isfinite(converted)
        finite = converted[finite_mask]
        nonfinite_counts[column] = int((converted.notna() & ~finite_mask).sum())
        numeric_summary[column] = {
            "min": float(finite.min()) if not finite.empty else None,
            "max": float(finite.max()) if not finite.empty else None,
            "missing_or_non_numeric": int(converted.isna().sum()),
            "non_finite": nonfinite_counts[column],
        }
        if column in NONNEGATIVE_COLUMNS:
            negative_counts[column] = int((converted < 0).sum())
        domain_errors: dict[str, int] = {}
        if column == "SeniorCitizen":
            domain_errors["not_zero_or_one"] = int(
                (~converted.isin([0, 1]) & converted.notna()).sum()
            )
        if column == "tenure":
            domain_errors["not_integer"] = int(
                ((converted.notna()) & (converted % 1 != 0)).sum()
            )
            domain_errors["above_72_months"] = int((converted > 72).sum())
        if column == "MonthlyCharges":
            domain_errors["not_positive"] = int((converted <= 0).sum())
        numeric_domain_errors[column] = domain_errors

    categorical_domain_errors: dict[str, dict[str, Any]] = {}
    for column, allowed_values in VALID_CATEGORICAL_VALUES.items():
        if column not in df.columns:
            continue
        values = df[column].astype("string").str.strip()
        unexpected = sorted(
            str(value) for value in values[~values.isin(allowed_values)].dropna().unique()
        )
        if unexpected:
            categorical_domain_errors[column] = {
                "unexpected_values": unexpected,
                "rows_with_unexpected_values": int(values.isin(unexpected).sum()),
            }

    anomaly: dict[str, Any] = {"zero_tenure_rows": None}
    if "tenure" in df.columns:
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        zero_tenure = tenure.eq(0)
        anomaly["zero_tenure_rows"] = int(zero_tenure.sum())
        if "TotalCharges" in df.columns:
            total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
            positive_mask = zero_tenure & total_charges.gt(0)
            positive_values = total_charges[positive_mask]
            distinct_values = sorted(float(v) for v in positive_values.dropna().unique())
            anomaly["zero_tenure_with_positive_total_charges"] = int(positive_mask.sum())
            anomaly["positive_total_charges_distinct_values_for_zero_tenure"] = distinct_values
            anomaly["all_positive_zero_tenure_charges_identical"] = (
                len(distinct_values) == 1 and int(positive_mask.sum()) > 1
            )

    warnings: list[str] = []
    if missing_expected:
        warnings.append(f"Expected schema columns missing: {missing_expected}")
    if unexpected_columns:
        warnings.append(f"Unexpected schema columns present: {unexpected_columns}")
    if missing_counts:
        warnings.append("One or more columns contain missing values.")
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} exact duplicate rows.")
    if duplicate_id_rows:
        warnings.append(f"{duplicate_id_rows} rows belong to repeated customer IDs.")
    if unknown_targets or missing_target:
        warnings.append(
            f"Target labels need review: unknown={unknown_targets}, missing={missing_target}."
        )
    if any(nonnumeric_counts.values()):
        warnings.append(f"Numeric columns contain non-numeric values: {nonnumeric_counts}")
    if any(nonfinite_counts.values()):
        warnings.append(f"Numeric columns contain non-finite values: {nonfinite_counts}")
    if any(any(count > 0 for count in details.values()) for details in numeric_domain_errors.values()):
        warnings.append(f"Numeric values outside expected domain: {numeric_domain_errors}")
    if categorical_domain_errors:
        warnings.append(f"Categorical values outside the declared schema: {categorical_domain_errors}")
    if any(count > 0 for count in negative_counts.values()):
        warnings.append(f"Non-negative numeric fields contain negative values: {negative_counts}")
    positive_zero_tenure = anomaly.get("zero_tenure_with_positive_total_charges", 0)
    if positive_zero_tenure:
        warnings.append(
            f"{positive_zero_tenure} rows have tenure == 0 and TotalCharges > 0; "
            "inspect source provenance/imputation rather than silently changing these values."
        )
    if anomaly.get("all_positive_zero_tenure_charges_identical"):
        warnings.append(
            "All positive TotalCharges values among zero-tenure rows are identical; "
            "this may indicate prior imputation, but the file alone cannot prove that."
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "matches_expected_column_order": tuple(df.columns) == EXPECTED_COLUMNS,
        "missing_expected_columns": missing_expected,
        "unexpected_columns": unexpected_columns,
        "missing_cells_total": int(df.isna().sum().sum()),
        "missing_counts_by_column": missing_counts,
        "exact_duplicate_rows": duplicate_rows,
        "unique_customer_ids": int(customer_ids.nunique(dropna=True)),
        "rows_with_duplicate_customer_id": duplicate_id_rows,
        "target_column": "State",
        "target_counts": target_counts,
        "unknown_target_labels": unknown_targets,
        "missing_target_labels": missing_target,
        "numeric_summary": numeric_summary,
        "negative_counts": negative_counts,
        "nonnumeric_counts": nonnumeric_counts,
        "nonfinite_counts": nonfinite_counts,
        "numeric_domain_errors": numeric_domain_errors,
        "categorical_domain_errors": categorical_domain_errors,
        "anomalies": anomaly,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data quality for the churn dataset.")
    default_csv = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
    parser.add_argument("--csv", type=Path, default=default_csv, help=f"CSV path (default: {default_csv})")
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {args.csv}")
    print(json.dumps(audit_dataset(pd.read_csv(args.csv)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

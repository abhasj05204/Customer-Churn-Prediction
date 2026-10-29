"""Audit the target column and dataset schema used by the churn project.

StateNextMonth is optional. If absent, this module explicitly reports that the
provided data cannot support a next-month outcome analysis by itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = ("No Churn", "Churn")
REQUIRED_COLUMNS = ("customerID", "State")


def audit_target_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Audit label counts and identifier quality without guessing label timing."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Cannot audit an empty dataset.")

    state = df["State"].astype("string").str.strip()
    invalid_state = sorted(
        str(value) for value in state[~state.isin(LABELS)].dropna().unique()
    )
    missing_state = int(state.isna().sum())
    if invalid_state or missing_state:
        raise ValueError(
            "Unexpected/missing State labels. "
            f"Invalid={invalid_state}, missing={missing_state}; "
            f"expected only {list(LABELS)}."
        )

    customer_ids = df["customerID"].astype("string").str.strip()
    target_counts = {label: int(state.eq(label).sum()) for label in LABELS}

    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(column) for column in df.columns],
        "unique_customer_count": int(customer_ids.nunique(dropna=True)),
        "rows_with_duplicate_customer_id": int(
            customer_ids.duplicated(keep=False).sum()
        ),
        "target_column": "State",
        "target_counts": target_counts,
        "target_churn_rate": float(state.eq("Churn").mean()),
        "state_next_month_available": "StateNextMonth" in df.columns,
        "interpretation_warning": (
            "This report describes the provided labels; it does not establish when "
            "State was measured. Verify dataset documentation before describing the "
            "model as a future-churn predictor."
        ),
    }

    if "StateNextMonth" not in df.columns:
        report["next_month_analysis"] = {
            "available": False,
            "reason": (
                "StateNextMonth is not present in this dataset. Future/next-month churn "
                "prediction cannot be trained or validated from this file alone."
            ),
        }
        return report

    next_state = df["StateNextMonth"].astype("string").str.strip()
    invalid_next = sorted(
        str(value) for value in next_state[~next_state.isin(LABELS)].dropna().unique()
    )
    missing_next = int(next_state.isna().sum())
    if invalid_next or missing_next:
        raise ValueError(
            "Unexpected/missing StateNextMonth labels. "
            f"Invalid={invalid_next}, missing={missing_next}; "
            f"expected only {list(LABELS)}."
        )

    transition = pd.crosstab(state, next_state).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    active_mask = state.eq("No Churn")
    active_next = next_state[active_mask]
    active_count = int(active_mask.sum())
    next_churn_count = int(active_next.eq("Churn").sum())

    report["state_next_month_counts_all_rows"] = {
        label: int(next_state.eq(label).sum()) for label in LABELS
    }
    report["state_by_state_next_month"] = {
        current: {future: int(transition.loc[current, future]) for future in LABELS}
        for current in LABELS
    }
    report["next_month_analysis"] = {
        "available": True,
        "candidate_active_cohort_definition": 'State == "No Churn"',
        "candidate_cohort_rows": active_count,
        "next_month_no_churn_count": int(active_next.eq("No Churn").sum()),
        "next_month_churn_count": next_churn_count,
        "observed_next_month_churn_rate": (
            float(next_churn_count / active_count) if active_count else None
        ),
        "warning": (
            "Descriptive only; confirm label definitions and feature timestamps before "
            "using StateNextMonth as a genuine future outcome."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit target labels in the Telco churn dataset."
    )
    default_csv = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Dataset CSV path (default: {default_csv})",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    print(json.dumps(audit_target_columns(df), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

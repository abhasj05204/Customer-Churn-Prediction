"""Audit churn labels and the candidate next-month prediction cohort.

This module describes label distributions; it does not infer label chronology.
Confirm the dataset's data dictionary and feature timing before model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = ("No Churn", "Churn")
REQUIRED_COLUMNS = ("customerID", "State", "StateNextMonth")


def audit_target_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Return label counts, transition table, and active-customer cohort stats.

    State and StateNextMonth are treated as observed labels by name only.
    This function does not assume when the labels were generated.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required target-audit columns: {missing}")
    if df.empty:
        raise ValueError("Cannot audit an empty dataset.")

    state = df["State"].astype("string").str.strip()
    next_state = df["StateNextMonth"].astype("string").str.strip()

    invalid_state = sorted(str(value) for value in state[~state.isin(LABELS)].dropna().unique())
    invalid_next_state = sorted(
        str(value) for value in next_state[~next_state.isin(LABELS)].dropna().unique()
    )
    missing_state = int(state.isna().sum())
    missing_next_state = int(next_state.isna().sum())

    if invalid_state or invalid_next_state or missing_state or missing_next_state:
        raise ValueError(
            "Unexpected/missing target labels. "
            f"State invalid={invalid_state}, missing={missing_state}; "
            f"StateNextMonth invalid={invalid_next_state}, missing={missing_next_state}. "
            f"Expected only {list(LABELS)}."
        )

    def counts(series: pd.Series) -> dict[str, int]:
        return {label: int(series.eq(label).sum()) for label in LABELS}

    transition = pd.crosstab(state, next_state).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )

    active_mask = state.eq("No Churn")
    active_next = next_state[active_mask]
    active_count = int(active_mask.sum())
    active_churn_count = int(active_next.eq("Churn").sum())

    customer_ids = df["customerID"].astype("string").str.strip()
    unique_customer_count = int(customer_ids.nunique(dropna=True))
    duplicate_customer_rows = int(customer_ids.duplicated(keep=False).sum())

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "unique_customer_count": unique_customer_count,
        "duplicate_customer_rows": duplicate_customer_rows,
        "state_counts": counts(state),
        "state_next_month_counts_all_rows": counts(next_state),
        "state_by_state_next_month": {
            current: {future: int(transition.loc[current, future]) for future in LABELS}
            for current in LABELS
        },
        "candidate_active_cohort": {
            "definition_used_for_audit": 'State == "No Churn"',
            "row_count": active_count,
            "next_month_no_churn_count": int(active_next.eq("No Churn").sum()),
            "next_month_churn_count": active_churn_count,
            "next_month_churn_rate": (
                float(active_churn_count / active_count) if active_count else None
            ),
        },
        "interpretation_warning": (
            "Counts are descriptive only. Verify label definitions, feature timestamps, "
            "and outcome-generation procedure before selecting a production target."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit State and StateNextMonth labels in the Telco churn dataset."
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
    report = audit_target_columns(df)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

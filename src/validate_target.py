"""Audit the supported State target and optional, unverified future-label metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = ("No Churn", "Churn")
REQUIRED_COLUMNS = ("customerID", "State")


def audit_target_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Validate recorded State; never require or infer StateNextMonth."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Cannot audit an empty dataset.")

    target = df["State"].astype("string").str.strip()
    invalid = sorted(str(v) for v in target[~target.isin(LABELS)].dropna().unique())
    missing = int(target.isna().sum())
    if invalid or missing:
        raise ValueError(
            f"Unexpected/missing State labels. Invalid={invalid}, missing={missing}; expected only {list(LABELS)}."
        )
    customer_ids = df["customerID"].astype("string").str.strip()
    future_available = "StateNextMonth" in df.columns
    future_valid = False
    future_counts: dict[str, int] | None = None
    future_warning = "StateNextMonth is absent; no future-label claim is supported."
    if future_available:
        future = df["StateNextMonth"].astype("string").str.strip()
        future_invalid = sorted(str(v) for v in future[~future.isin(LABELS)].dropna().unique())
        future_missing = int(future.isna().sum())
        future_valid = not future_invalid and future_missing == 0
        if future_valid:
            future_counts = {label: int(future.eq(label).sum()) for label in LABELS}
        future_warning = (
            "Optional StateNextMonth values are present but their provenance and temporal alignment are unverified; "
            "they are not used as the training target."
        )
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(column) for column in df.columns],
        "unique_customer_count": int(customer_ids.nunique(dropna=True)),
        "rows_with_duplicate_customer_id": int(customer_ids.duplicated(keep=False).sum()),
        "target_column": "State",
        "target_counts": {label: int(target.eq(label).sum()) for label in LABELS},
        "target_churn_rate": float(target.eq("Churn").mean()),
        "state_next_month_available": future_available,
        "state_next_month_valid_labels": future_valid,
        "state_next_month_counts": future_counts,
        "future_label_provenance": {
            "status": "unverified" if future_available else "absent",
            "warning": future_warning,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit recorded State target labels.")
    default_csv = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
    parser.add_argument("--csv", type=Path, default=default_csv, help=f"Dataset CSV path (default: {default_csv})")
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {args.csv}")
    print(json.dumps(audit_target_columns(pd.read_csv(args.csv)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

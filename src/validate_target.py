"""Audit StateNextMonth labels and record their unverified provenance explicitly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = ("No Churn", "Churn")
REQUIRED_COLUMNS = ("customerID", "State", "StateNextMonth")


def audit_target_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Validate supplied target labels without assuming they are observed future outcomes."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Cannot audit an empty dataset.")

    current = df["State"].astype("string").str.strip()
    target = df["StateNextMonth"].astype("string").str.strip()
    for name, values in (("State", current), ("StateNextMonth", target)):
        invalid = sorted(str(v) for v in values[~values.isin(LABELS)].dropna().unique())
        missing = int(values.isna().sum())
        if invalid or missing:
            raise ValueError(
                f"Unexpected/missing {name} labels. Invalid={invalid}, missing={missing}; "
                f"expected only {list(LABELS)}."
            )

    customer_ids = df["customerID"].astype("string").str.strip()
    transition = pd.crosstab(current, target).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    target_counts = {label: int(target.eq(label).sum()) for label in LABELS}
    active_mask = current.eq("No Churn")
    active_target = target[active_mask]
    active_count = int(active_mask.sum())

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(column) for column in df.columns],
        "unique_customer_count": int(customer_ids.nunique(dropna=True)),
        "rows_with_duplicate_customer_id": int(customer_ids.duplicated(keep=False).sum()),
        "target_column": "StateNextMonth",
        "target_counts": target_counts,
        "target_churn_rate": float(target.eq("Churn").mean()),
        "current_state_counts": {label: int(current.eq(label).sum()) for label in LABELS},
        "state_by_state_next_month": {
            current_label: {
                target_label: int(transition.loc[current_label, target_label])
                for target_label in LABELS
            }
            for current_label in LABELS
        },
        "candidate_active_cohort": {
            "definition": 'State == "No Churn"',
            "rows": active_count,
            "next_month_churn_count": int(active_target.eq("Churn").sum()),
            "next_month_churn_rate": (
                float(active_target.eq("Churn").mean()) if active_count else None
            ),
        },
        "label_provenance": {
            "status": "unverified",
            "warning": (
                "The CSV contains StateNextMonth labels, but file contents alone cannot prove "
                "they were measured from actual next-month observations. Keep this as a supplied "
                "target label until a source, date range, and label-generation method are documented."
            ),
        },
        "temporal_alignment": {
            "status": "not_verifiable",
            "warning": (
                "No trusted observation timestamp or outcome date is available in the declared "
                "schema, so this audit does not establish a real next-month prediction horizon."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit StateNextMonth target labels.")
    default_csv = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
    parser.add_argument("--csv", type=Path, default=default_csv, help=f"Dataset CSV path (default: {default_csv})")
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {args.csv}")
    print(json.dumps(audit_target_columns(pd.read_csv(args.csv)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Structural target-leakage and feature-timing audit for churn modelling.

Static checks can catch identifiers and label-like columns included as predictors.
They cannot prove temporal availability without a documented observation timestamp.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_engineering import (
    RAW_FEATURE_COLUMNS,
    engineer_inference_features,
)

TARGET_COLUMN = "StateNextMonth"
BLOCKED_PREDICTOR_COLUMNS = {
    "customerID",
    "State",
    "StateNextMonth",
    "Churn",
    "ChurnLabel",
    "target",
    "label",
}
TEMPORAL_METADATA_COLUMNS = (
    "observation_date",
    "snapshot_date",
    "as_of_date",
    "churn_date",
    "outcome_date",
    "target_date",
)


def audit_feature_leakage(df: pd.DataFrame) -> dict[str, Any]:
    """Audit whether the known identifier/target fields enter the feature pipeline."""
    if df.empty:
        raise ValueError("Cannot audit an empty dataset.")
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing raw predictor columns required by this project: {missing}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    engineered_features = engineer_inference_features(df)
    predictor_columns = list(engineered_features.columns)
    blocked_present = sorted(BLOCKED_PREDICTOR_COLUMNS.intersection(df.columns))
    blocked_in_predictors = sorted(BLOCKED_PREDICTOR_COLUMNS.intersection(predictor_columns))
    temporal_columns_present = sorted(set(TEMPORAL_METADATA_COLUMNS).intersection(df.columns))
    temporal_columns_missing = [
        column for column in TEMPORAL_METADATA_COLUMNS if column not in df.columns
    ]
    allowed_source_columns = (
        set(RAW_FEATURE_COLUMNS)
        | BLOCKED_PREDICTOR_COLUMNS
        | set(TEMPORAL_METADATA_COLUMNS)
    )
    unexpected_source_columns = sorted(
        str(column) for column in df.columns if column not in allowed_source_columns
    )

    target_values = df[TARGET_COLUMN].astype("string").str.strip()
    target_labels = sorted(str(value) for value in target_values.dropna().unique())
    unknown_target_values = sorted(
        str(value)
        for value in target_values[
            ~target_values.isin(["No Churn", "Churn"])
        ].dropna().unique()
    )
    target_missing = int(target_values.isna().sum())

    state_next_month_in_file = "StateNextMonth" in df.columns
    structural_pass = (
        not blocked_in_predictors
        and not unknown_target_values
        and target_missing == 0
        and not unexpected_source_columns
    )

    return {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "target_column": TARGET_COLUMN,
        "target_labels_observed": target_labels,
        "target_label_errors": {
            "unknown_values": unknown_target_values,
            "missing_rows": target_missing,
        },
        "raw_predictor_count": int(len(RAW_FEATURE_COLUMNS)),
        "engineered_predictor_count": int(len(predictor_columns)),
        "predictor_columns": predictor_columns,
        "blocked_columns_present_in_source": blocked_present,
        "blocked_columns_in_predictor_frame": blocked_in_predictors,
        "unexpected_source_columns": unexpected_source_columns,
        "state_next_month_present_in_source": state_next_month_in_file,
        "structural_leakage_check": {
            "status": "pass" if structural_pass else "fail",
            "meaning": (
                "Known identifier/target columns are excluded from the model feature frame."
                if structural_pass
                else "A blocked column entered predictors, target labels are invalid/missing, or unexpected source columns need explicit review."
            ),
        },
        "feature_timing_check": {
            "status": "not_verifiable",
            "temporal_metadata_columns_present": temporal_columns_present,
            "expected_metadata_columns_not_present": temporal_columns_missing,
            "explanation": (
                "The selected dataset has no documented observation cut-off or outcome date. "
                "Code cannot establish whether billing/service features were recorded before "
                "the churn label. Do not claim a future prediction horizon based on this file alone."
            ),
        },
        "warnings": [
            (
                "StateNextMonth is present as the supplied target, but its provenance and temporal alignment are unverified."
                if not state_next_month_in_file
                else "StateNextMonth exists in the source, but it is excluded from predictors and its semantics require provenance."
            ),
            (
                f"Unexpected source columns require explicit review: {unexpected_source_columns}"
                if unexpected_source_columns
                else "Source columns match the declared schema plus known labels/metadata."
            ),
            (
                "State and StateNextMonth target columns plus customerID are excluded by the shared inference feature builder."
                if not blocked_in_predictors
                else f"Blocked columns found in engineered predictors: {blocked_in_predictors}"
            ),
            "Temporal leakage cannot be ruled out without trusted feature and target timestamps.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit target leakage and feature-timing limitations."
    )
    default_csv = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
    parser.add_argument("--csv", type=Path, default=default_csv, help=f"CSV path (default: {default_csv})")
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    print(json.dumps(audit_feature_leakage(df), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

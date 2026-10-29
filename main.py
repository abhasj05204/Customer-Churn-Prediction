"""Command-line entry point for the Customer Churn Prediction project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.validate_data import audit_dataset
from src.validate_target import audit_target_columns

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "telco_churn.csv"


def load_dataset(csv_path: Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load a CSV from an explicit path and raise a clear error if it is absent."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction: dataset checks and baseline modelling."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("dataset", help="Print the dataset shape and first five rows.")
    subparsers.add_parser("audit", help="Print target and data-quality audit reports.")

    train_parser = subparsers.add_parser(
        "train", help="Run leakage-safe cross-validated baseline model comparison."
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Local directory for the JSON report and fitted pipeline (default: outputs).",
    )
    train_parser.add_argument("--test-size", type=float, default=0.20)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--folds", type=int, default=5)

    args = parser.parse_args()
    command = args.command or "dataset"

    if command == "dataset":
        df = load_dataset()
        print("Dataset loaded successfully.")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\nFirst five rows:")
        print(df.head().to_string(index=False))
        return

    if command == "audit":
        df = load_dataset()
        print("=== Target audit ===")
        print(json.dumps(audit_target_columns(df), indent=2, ensure_ascii=False))
        print("\n=== Data quality audit ===")
        print(json.dumps(audit_dataset(df), indent=2, ensure_ascii=False))
        return

    if command == "train":
        from src.train import run_baseline_experiment

        report = run_baseline_experiment(
            csv_path=DEFAULT_DATA,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_splits=args.folds,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    parser.error(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()

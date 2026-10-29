"""Command-line entry point for the Customer Churn Prediction project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.audit_leakage import audit_feature_leakage
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
    subparsers.add_parser("audit", help="Print target, data-quality, and leakage audit reports.")

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

    tune_parser = subparsers.add_parser(
        "tune", help="Tune XGBoost with training-only stratified cross-validation."
    )
    tune_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    tune_parser.add_argument("--test-size", type=float, default=0.20)
    tune_parser.add_argument("--random-state", type=int, default=42)
    tune_parser.add_argument("--folds", type=int, default=5)
    tune_parser.add_argument("--trials", type=int, default=30)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Select model and threshold on development data, then evaluate the holdout once.",
    )
    finalize_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    finalize_parser.add_argument("--test-size", type=float, default=0.20)
    finalize_parser.add_argument("--threshold-validation-fraction", type=float, default=0.25)
    finalize_parser.add_argument("--random-state", type=int, default=42)
    finalize_parser.add_argument("--folds", type=int, default=5)
    finalize_parser.add_argument("--trials", type=int, default=30)

    imbalance_parser = subparsers.add_parser(
        "imbalance", help="Compare class weighting, SMOTE and SMOTENC with training-only CV."
    )
    imbalance_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    imbalance_parser.add_argument("--test-size", type=float, default=0.20)
    imbalance_parser.add_argument("--random-state", type=int, default=42)
    imbalance_parser.add_argument("--folds", type=int, default=5)

    explain_parser = subparsers.add_parser(
        "explain", help="Generate SHAP explanations for a saved final pipeline."
    )
    explain_parser.add_argument("--model", type=Path, default=Path("outputs/final_pipeline.joblib"))
    explain_parser.add_argument("--output-dir", type=Path, default=Path("outputs/explainability"))
    explain_parser.add_argument("--background-size", type=int, default=100)
    explain_parser.add_argument("--explain-size", type=int, default=40)
    explain_parser.add_argument("--random-state", type=int, default=42)

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
        print("\n=== Leakage and feature timing audit ===")
        print(json.dumps(audit_feature_leakage(df), indent=2, ensure_ascii=False))
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

    if command == "tune":
        from src.optimize import run_optuna_experiment

        report = run_optuna_experiment(
            csv_path=DEFAULT_DATA,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_splits=args.folds,
            n_trials=args.trials,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if command == "finalize":
        from src.finalize import run_final_evaluation

        report = run_final_evaluation(
            csv_path=DEFAULT_DATA,
            output_dir=args.output_dir,
            test_size=args.test_size,
            threshold_validation_fraction=args.threshold_validation_fraction,
            random_state=args.random_state,
            n_splits=args.folds,
            n_trials=args.trials,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if command == "imbalance":
        from src.compare_imbalance import run_imbalance_experiment

        report = run_imbalance_experiment(
            csv_path=DEFAULT_DATA,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_splits=args.folds,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if command == "explain":
        from src.explain import run_shap_explanations

        report = run_shap_explanations(
            model_path=args.model,
            csv_path=DEFAULT_DATA,
            output_dir=args.output_dir,
            background_size=args.background_size,
            explain_size=args.explain_size,
            random_state=args.random_state,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    parser.error(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()

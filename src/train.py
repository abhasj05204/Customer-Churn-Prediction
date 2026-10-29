"""Leakage-safe baseline model comparison for the StateNextMonth target."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.experiment_metadata import collect_environment_metadata
from src.feature_engineering import engineer_features, split_features_target
from src.preprocess import build_model_pipeline

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def load_dataset(csv_path: Path = DEFAULT_DATA) -> tuple[pd.DataFrame, str]:
    """Read the CSV and return its frame plus a SHA-256 content fingerprint."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    content_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return pd.read_csv(csv_path), content_sha256


def candidate_estimators(random_state: int = 42) -> dict[str, Any]:
    """Return a small, explicit set of baselines for fair cross-validation."""
    from xgboost import XGBClassifier

    return {
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "XGBoost (no resampling)": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
        ),
    }


def cross_validate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare pipelines using training-only stratified CV; no holdout data is used."""
    if y_train.nunique() != 2:
        raise ValueError("Training target must contain both binary classes.")
    if n_splits < 2 or int(y_train.value_counts().min()) < n_splits:
        raise ValueError("n_splits must be at least 2 and no greater than the smallest class count.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    estimators = candidate_estimators(random_state=random_state)
    pipelines = {
        name: build_model_pipeline(X_train, estimator)
        for name, estimator in estimators.items()
    }

    results: dict[str, Any] = {}
    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        results[name] = {
            metric: {
                "mean": float(pd.Series(scores[f"test_{metric}"]).mean()),
                "std": float(pd.Series(scores[f"test_{metric}"]).std(ddof=1)),
                "fold_scores": [float(v) for v in scores[f"test_{metric}"]],
            }
            for metric in SCORING
        }

    selected_name = max(results, key=lambda model_name: results[model_name]["roc_auc"]["mean"])
    return results, {selected_name: pipelines[selected_name]}


def run_baseline_experiment(
    csv_path: Path = DEFAULT_DATA,
    output_dir: Path | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Select by training CV ROC-AUC and leave holdout data unevaluated."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    from sklearn.model_selection import train_test_split

    raw_df, dataset_sha256 = load_dataset(csv_path)
    engineered = engineer_features(raw_df)
    X, y = split_features_target(engineered)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cv_results, selected = cross_validate_candidates(
        X_train, y_train, random_state=random_state, n_splits=n_splits
    )
    selected_name, selected_pipeline = next(iter(selected.items()))
    selected_pipeline.fit(X_train, y_train)

    result: dict[str, Any] = {
        "environment": collect_environment_metadata(),
        "dataset": {
            "path": str(csv_path),
            "sha256": dataset_sha256,
            "rows": int(len(raw_df)),
            "columns": int(len(raw_df.columns)),
            "target_column": "StateNextMonth",
            "target_counts": {str(k): int(v) for k, v in raw_df["StateNextMonth"].value_counts().items()},
        },
        "split": {
            "test_size": float(test_size),
            "random_state": int(random_state),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "cross_validation_folds": int(n_splits),
        },
        "selection_rule": f"highest mean training-only {n_splits}-fold ROC-AUC",
        "cross_validation": cv_results,
        "selected_model": selected_name,
        "holdout_test": {
            "rows": int(len(X_test)),
            "metrics_evaluated": False,
            "status": "reserved for final evaluation after model and threshold selection",
        },
        "warning": (
            "This experiment selects a baseline using training-only cross-validation and "
            "does not calculate holdout metrics. It classifies the dataset State label; "
            "StateNextMonth is the supplied target, but its real-world provenance and temporal alignment "
            "have not been independently verified."
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "baseline_evaluation.json"
        model_path = output_dir / "baseline_pipeline.joblib"
        result["saved_model_path"] = str(model_path)
        result["saved_report_path"] = str(report_path)
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        import joblib

        joblib.dump(selected_pipeline, model_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe baseline CV and one holdout evaluation."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA, help="Dataset CSV path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for JSON report and model.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = run_baseline_experiment(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_splits=args.folds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

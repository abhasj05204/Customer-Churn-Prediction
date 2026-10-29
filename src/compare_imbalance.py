"""Training-only CV comparison of class weighting, SMOTE, and SMOTENC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.experiment_metadata import collect_environment_metadata
from src.feature_engineering import engineer_features, split_features_target
from src.preprocess import build_model_pipeline, build_smote_pipeline, build_smotenc_pipeline
from src.train import DEFAULT_DATA, load_dataset

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def candidate_imbalance_pipelines(
    X_train: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Any]:
    """Use the same base Logistic Regression across imbalance strategies."""
    def logistic(class_weight: str | None = None) -> LogisticRegression:
        return LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
        )

    return {
        "Logistic Regression (no balancing)": build_model_pipeline(
            X_train, logistic(class_weight=None)
        ),
        "Logistic Regression (class_weight=balanced)": build_model_pipeline(
            X_train, logistic(class_weight="balanced")
        ),
        "Logistic Regression + ordinary SMOTE (experimental)": build_smote_pipeline(
            X_train, logistic(class_weight=None), random_state=random_state
        ),
        "Logistic Regression + SMOTENC": build_smotenc_pipeline(
            X_train, logistic(class_weight=None), random_state=random_state
        ),
    }


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Compare candidates using identical training-only stratified CV."""
    if y_train.nunique() != 2:
        raise ValueError("Training target must contain both binary classes.")
    if n_splits < 2 or int(y_train.value_counts().min()) < n_splits:
        raise ValueError("n_splits must be at least 2 and no greater than the smallest class count.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidates = candidate_imbalance_pipelines(X_train, random_state=random_state)
    results: dict[str, Any] = {}
    for name, pipeline in candidates.items():
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
                "fold_scores": [float(value) for value in scores[f"test_{metric}"]],
            }
            for metric in SCORING
        }
    return results


def run_imbalance_experiment(
    csv_path: Path = DEFAULT_DATA,
    output_dir: Path | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Compare imbalance strategies with CV on development rows; do not score holdout."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    raw_df, dataset_sha256 = load_dataset(csv_path)
    engineered = engineer_features(raw_df)
    X, y = split_features_target(engineered)
    X_train, X_holdout, y_train, _y_holdout = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    results = compare_imbalance_strategies(
        X_train, y_train, random_state=random_state, n_splits=n_splits
    )
    report: dict[str, Any] = {
        "environment": collect_environment_metadata(),
        "dataset": {
            "path": str(csv_path),
            "sha256": dataset_sha256,
            "rows": int(len(raw_df)),
            "target_column": "State",
        },
        "split": {
            "random_state": int(random_state),
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_holdout)),
            "cv_folds": int(n_splits),
        },
        "comparison_protocol": {
            "base_estimator": "LogisticRegression(max_iter=2000)",
            "scoring": list(SCORING),
            "preprocessing": "fit inside each CV training fold",
            "ordinary_smote": "experimental only; one-hot interpolation may create fractional indicators",
            "smotenc": "preserves categorical values until after resampling",
            "holdout_scored": False,
        },
        "cross_validation": results,
        "warning": (
            "Scores are training-only CV estimates, not final holdout results. "
            "The target is State, not a verified next-month outcome."
        ),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "imbalance_comparison.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["saved_report_path"] = str(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare class weighting, ordinary SMOTE and SMOTENC with training-only CV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    report = run_imbalance_experiment(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_splits=args.folds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Select a model/threshold without using the holdout, then evaluate once.

The dataset target is State. This pipeline does not claim next-month churn
prediction because the selected CSV has no future outcome label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.evaluate import evaluate_calibration, evaluate_model, evaluate_predictions
from src.feature_engineering import engineer_features, split_features_target
from src.optimize import tune_xgb_params
from src.preprocess import build_model_pipeline
from src.train import DEFAULT_DATA, candidate_estimators, cross_validate_candidates, load_dataset


MODEL_PREFERENCE = (
    "Logistic Regression (balanced)",
    "Random Forest (balanced)",
    "XGBoost (no resampling)",
    "XGBoost (Optuna tuned)",
)


def select_model_one_standard_error(
    baseline_results: dict[str, Any],
    tuned_xgb_mean: float,
    tuned_xgb_std: float,
    n_splits: int,
) -> tuple[str, dict[str, Any]]:
    """Prefer the simplest model within one SE of the highest mean CV ROC-AUC."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    candidates: dict[str, dict[str, float]] = {}
    for name, metrics in baseline_results.items():
        candidates[name] = {
            "mean": float(metrics["roc_auc"]["mean"]),
            "std": float(metrics["roc_auc"]["std"]),
        }
    candidates["XGBoost (Optuna tuned)"] = {
        "mean": float(tuned_xgb_mean),
        "std": float(tuned_xgb_std),
    }

    best_name = max(candidates, key=lambda name: candidates[name]["mean"])
    best_mean = candidates[best_name]["mean"]
    standard_error = candidates[best_name]["std"] / float(np.sqrt(n_splits))
    cutoff = best_mean - standard_error
    eligible = [
        name for name, score in candidates.items()
        if score["mean"] >= cutoff
    ]
    preference_index = {name: index for index, name in enumerate(MODEL_PREFERENCE)}
    selected = min(eligible, key=lambda name: preference_index.get(name, len(MODEL_PREFERENCE)))
    return selected, {
        "rule": "one-standard-error rule on training-only CV ROC-AUC; prefer simpler baseline",
        "best_mean_cv_model": best_name,
        "best_mean_cv_roc_auc": best_mean,
        "best_model_standard_error": standard_error,
        "eligibility_cutoff": cutoff,
        "eligible_models": eligible,
        "selected_model": selected,
        "candidate_scores": candidates,
    }


def select_f1_threshold(
    y_true: Any,
    probabilities: Any,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Select an F1 threshold on a separate validation split, not on the holdout."""
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and probabilities must have the same non-zero length.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1.")
    candidate_thresholds = thresholds or [round(index / 100, 2) for index in range(5, 96)]
    if not candidate_thresholds:
        raise ValueError("At least one threshold is required.")
    if any(not 0.0 < threshold < 1.0 for threshold in candidate_thresholds):
        raise ValueError("All candidate thresholds must be strictly between 0 and 1.")

    evaluations = []
    for threshold in candidate_thresholds:
        result = evaluate_predictions(y, p, threshold=float(threshold))
        evaluations.append({
            "threshold": float(threshold),
            "metrics": result["metrics"],
            "confusion_counts": result["confusion_counts"],
        })

    # Maximize F1, then recall, then prefer a threshold closer to 0.5.
    chosen = max(
        evaluations,
        key=lambda item: (
            item["metrics"]["f1"],
            item["metrics"]["recall"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    return chosen["threshold"], {
        "selection_metric": "F1",
        "tie_breakers": ["recall", "distance from 0.5"],
        "selected_threshold": chosen["threshold"],
        "selected_validation_metrics": chosen["metrics"],
        "selected_validation_confusion_counts": chosen["confusion_counts"],
        "candidate_threshold_count": len(evaluations),
        "candidate_metrics": evaluations,
    }


def run_final_evaluation(
    csv_path: Path = DEFAULT_DATA,
    output_dir: Path | None = None,
    test_size: float = 0.20,
    threshold_validation_fraction: float = 0.25,
    random_state: int = 42,
    n_splits: int = 5,
    n_trials: int = 30,
) -> dict[str, Any]:
    """Select model/threshold using development data; evaluate final holdout once."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    if not 0.1 <= threshold_validation_fraction <= 0.4:
        raise ValueError("threshold_validation_fraction must be between 0.1 and 0.4.")

    raw_df, dataset_sha256 = load_dataset(csv_path)
    engineered = engineer_features(raw_df)
    X, y = split_features_target(engineered)

    # Outer holdout remains isolated from model selection and threshold selection.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_model, X_threshold, y_model, y_threshold = train_test_split(
        X_development,
        y_development,
        test_size=threshold_validation_fraction,
        random_state=random_state + 1,
        stratify=y_development,
    )

    baseline_cv, _ = cross_validate_candidates(
        X_model, y_model, random_state=random_state, n_splits=n_splits
    )
    study, best_xgb_params = tune_xgb_params(
        X_model,
        y_model,
        random_state=random_state,
        n_splits=n_splits,
        n_trials=n_trials,
        study_name="churn_xgboost_final_selection_cv",
    )
    tuned_std = float(study.best_trial.user_attrs.get("std_roc_auc", 0.0))
    selected_name, selection_report = select_model_one_standard_error(
        baseline_cv,
        tuned_xgb_mean=float(study.best_value),
        tuned_xgb_std=tuned_std,
        n_splits=n_splits,
    )

    if selected_name == "XGBoost (Optuna tuned)":
        selected_estimator = XGBClassifier(**best_xgb_params)
    else:
        selected_estimator = candidate_estimators(random_state=random_state)[selected_name]

    # Fit only on model-training data to choose threshold on a separate validation split.
    threshold_model = build_model_pipeline(X_model, clone(selected_estimator))
    threshold_model.fit(X_model, y_model)
    validation_probabilities = threshold_model.predict_proba(X_threshold)[:, 1]
    calibration_report = evaluate_calibration(
        y_threshold, validation_probabilities, n_bins=10
    )
    chosen_threshold, threshold_report = select_f1_threshold(
        y_threshold, validation_probabilities
    )

    # After model and threshold selection are frozen, refit on all development data.
    final_pipeline = build_model_pipeline(X_development, clone(selected_estimator))
    final_pipeline.fit(X_development, y_development)
    holdout_report = evaluate_model(
        final_pipeline, X_test, y_test, threshold=chosen_threshold
    )

    report: dict[str, Any] = {
        "dataset": {
            "path": str(csv_path),
            "sha256": dataset_sha256,
            "rows": int(len(raw_df)),
            "columns": int(len(raw_df.columns)),
            "target_column": "State",
            "target_counts": {
                str(label): int(count)
                for label, count in raw_df["State"].value_counts().items()
            },
        },
        "split": {
            "random_state": int(random_state),
            "development_rows": int(len(X_development)),
            "model_training_rows": int(len(X_model)),
            "threshold_validation_rows": int(len(X_threshold)),
            "holdout_rows": int(len(X_test)),
            "cv_folds": int(n_splits),
        },
        "baseline_cross_validation": baseline_cv,
        "optuna_tuning": {
            "trials_completed": len(study.trials),
            "mean_cv_roc_auc": float(study.best_value),
            "std_cv_roc_auc": tuned_std,
            "mean_average_precision": study.best_trial.user_attrs.get(
                "mean_average_precision"
            ),
            "best_parameters": best_xgb_params,
        },
        "model_selection": selection_report,
        "threshold_selection": threshold_report,
        "probability_calibration_on_threshold_validation": calibration_report,
        "calibration_decision": (
            "Diagnostics are reported for review; no calibration layer is applied automatically."
        ),
        "final_model": selected_name,
        "final_threshold": float(chosen_threshold),
        "holdout_test_evaluation": holdout_report,
        "methodology": {
            "holdout_used_for_selection": False,
            "preprocessing_fit_inside_cv": True,
            "threshold_selected_on_separate_validation_split": True,
            "resampling": "none",
            "threshold_objective": "validation F1; ties use recall then distance from 0.5",
        },
        "warning": (
            "Final metrics describe classification of State, not verified next-month churn. "
            "The holdout was excluded from model and threshold selection and evaluated only "
            "after those choices were frozen."
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "final_evaluation.json"
        model_path = output_dir / "final_pipeline.joblib"
        report["saved_report_path"] = str(report_path)
        report["saved_model_path"] = str(model_path)
        import joblib

        joblib.dump(final_pipeline, model_path)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select model and threshold without holdout leakage; then evaluate once."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--threshold-validation-fraction", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    report = run_final_evaluation(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        threshold_validation_fraction=args.threshold_validation_fraction,
        random_state=args.random_state,
        n_splits=args.folds,
        n_trials=args.trials,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

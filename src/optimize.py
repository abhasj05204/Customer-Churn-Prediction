"""Leakage-safe Optuna tuning for XGBoost using training-only CV.

The holdout split is created for later final evaluation but is never scored here.
No resampling is used: one-hot categorical features stay valid, and the baseline
experiment compares weighting approaches separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

from src.feature_engineering import engineer_features, split_features_target
from src.preprocess import build_model_pipeline
from src.train import DEFAULT_DATA, load_dataset

SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision"}


def validate_tuning_config(y: pd.Series, n_trials: int, n_splits: int) -> None:
    """Validate configuration before spending time on optimization."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if y.nunique() != 2:
        raise ValueError("Target must contain both binary classes.")
    smallest_class = int(y.value_counts().min())
    if smallest_class < n_splits:
        raise ValueError(
            "n_splits cannot exceed the smallest class count in the training split."
        )


def suggest_xgb_params(trial: Any, random_state: int = 42) -> dict[str, Any]:
    """Suggest XGBoost parameters without changing feature or validation data."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 4.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0, log=True),
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": 1,
    }


def tune_xgb_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
    n_trials: int = 30,
    study_name: str = "churn_xgboost_training_cv",
) -> tuple[optuna.Study, dict[str, Any]]:
    """Tune XGBoost on the provided training portion only, returning study and params."""
    validate_tuning_config(y_train, n_trials=n_trials, n_splits=n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial, random_state=random_state)
        pipeline = build_model_pipeline(X_train, XGBClassifier(**params))
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
        roc_scores = [float(value) for value in scores["test_roc_auc"]]
        ap_scores = [float(value) for value in scores["test_average_precision"]]
        trial.set_user_attr("fold_roc_auc", roc_scores)
        trial.set_user_attr("fold_average_precision", ap_scores)
        trial.set_user_attr("mean_average_precision", float(sum(ap_scores) / len(ap_scores)))
        trial.set_user_attr("std_roc_auc", float(pd.Series(roc_scores).std(ddof=1)))
        return float(sum(roc_scores) / len(roc_scores))

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)
    best_params = dict(study.best_params)
    best_params.update({"random_state": random_state, "eval_metric": "logloss", "n_jobs": 1})
    return study, best_params


def run_optuna_experiment(
    csv_path: Path = DEFAULT_DATA,
    output_dir: Path | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    n_splits: int = 5,
    n_trials: int = 30,
) -> dict[str, Any]:
    """Tune XGBoost on the development split and leave holdout data unscored."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")

    raw_df, dataset_sha256 = load_dataset(csv_path)
    engineered = engineer_features(raw_df)
    X, y = split_features_target(engineered)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    study, best_params = tune_xgb_params(
        X_train, y_train, random_state=random_state,
        n_splits=n_splits, n_trials=n_trials,
    )
    best_pipeline = build_model_pipeline(X_train, XGBClassifier(**best_params))
    best_pipeline.fit(X_train, y_train)

    result: dict[str, Any] = {
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
            "test_size": float(test_size),
            "random_state": int(random_state),
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_test)),
            "cross_validation_folds": int(n_splits),
        },
        "optimization": {
            "study_name": study.study_name,
            "direction": "maximize",
            "objective": "mean training-only stratified CV ROC-AUC",
            "n_trials_requested": int(n_trials),
            "n_trials_completed": len(study.trials),
            "best_mean_cv_roc_auc": float(study.best_value),
            "best_trial_std_cv_roc_auc": study.best_trial.user_attrs.get("std_roc_auc"),
            "best_trial_mean_average_precision": study.best_trial.user_attrs.get(
                "mean_average_precision"
            ),
            "best_parameters": best_params,
        },
        "holdout_test": {
            "rows": int(len(X_test)),
            "metrics_evaluated": False,
            "status": "reserved for final model evaluation only",
        },
        "methodology": {
            "resampling": "none",
            "preprocessing": "fitted independently inside each CV training fold",
            "categorical_features": "one-hot encoded after fold-local preprocessing",
            "random_seed": int(random_state),
        },
        "warning": (
            "This experiment predicts State, not next-month churn. It does not score "
            "the holdout set. Do not compare the best CV score directly with historical "
            "test ROC-AUC or treat it as final test performance."
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "optuna_xgboost_cv.json"
        model_path = output_dir / "optuna_xgboost_pipeline.joblib"
        result["saved_report_path"] = str(report_path)
        result["saved_model_path"] = str(model_path)
        import joblib

        joblib.dump(best_pipeline, model_path)
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune XGBoost with stratified CV on the training split only."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    result = run_optuna_experiment(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_splits=args.folds,
        n_trials=args.trials,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

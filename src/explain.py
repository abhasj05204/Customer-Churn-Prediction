"""SHAP explainability for a fitted raw-feature churn pipeline.

Explanations are sampled from the development/training partition, never from
the reserved final holdout. The model pipeline must accept the engineered raw
DataFrame produced by src.feature_engineering.engineer_features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import engineer_features, split_features_target
from src.train import DEFAULT_DATA, load_dataset


def make_probability_predictor(
    model: Any,
    feature_template: pd.DataFrame,
) -> Callable[[Any], np.ndarray]:
    """Wrap a pipeline to preserve DataFrame column names/dtypes for SHAP arrays."""
    feature_names = list(feature_template.columns)
    template_dtypes = feature_template.dtypes.to_dict()
    categorical_categories = {
        name: feature_template[name].cat.categories
        for name in feature_names
        if isinstance(feature_template[name].dtype, pd.CategoricalDtype)
    }

    def predict_positive_probability(rows: Any) -> np.ndarray:
        if isinstance(rows, pd.DataFrame):
            frame = rows.reindex(columns=feature_names).copy()
        else:
            array = np.asarray(rows, dtype=object)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[1] != len(feature_names):
                raise ValueError(
                    f"Expected rows with {len(feature_names)} columns; got shape {array.shape}."
                )
            frame = pd.DataFrame(array, columns=feature_names)

        for name, dtype in template_dtypes.items():
            if name in categorical_categories:
                frame[name] = pd.Categorical(
                    frame[name], categories=categorical_categories[name]
                )
            elif pd.api.types.is_numeric_dtype(dtype):
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
            elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
                frame[name] = frame[name].astype("string")
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("Loaded model must return positive-class probabilities.")
        result = probabilities[:, 1]
        if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise ValueError("Model returned invalid positive-class probabilities.")
        return result

    return predict_positive_probability


def _encoded_model_components(model: Any) -> tuple[Any, Any | None]:
    """Return the fitted base pipeline and optional positive-class calibrator."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        return model, None

    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        calibrated = calibrated_classifiers[0]
        base_pipeline = getattr(calibrated, "estimator", None)
        calibrators = getattr(calibrated, "calibrators", None)
        if (
            base_pipeline is not None
            and hasattr(base_pipeline, "named_steps")
            and calibrators
        ):
            # Binary sklearn calibration stores one mapping for the positive class.
            return base_pipeline, calibrators[0]

    raise TypeError(
        "Expected a fitted sklearn Pipeline or CalibratedClassifierCV wrapping a pipeline."
    )


def make_encoded_probability_predictor(
    model: Any,
) -> tuple[Callable[[Any], np.ndarray], list[str], Any]:
    """Explain a model after its fitted preprocessing, using numeric/one-hot inputs."""
    base_pipeline, calibrator = _encoded_model_components(model)
    preprocessor = base_pipeline.named_steps["preprocessor"]
    estimator = base_pipeline.named_steps["model"]
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    def predict_positive_probability(rows: Any) -> np.ndarray:
        values = rows.to_numpy() if isinstance(rows, pd.DataFrame) else np.asarray(rows)
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected encoded rows with {len(feature_names)} columns; got shape {values.shape}."
            )
        if calibrator is not None and hasattr(estimator, "decision_function"):
            # sklearn calibrates decision_function scores when the estimator exposes one.
            raw_scores = np.asarray(estimator.decision_function(values), dtype=float).reshape(-1)
            result = np.asarray(calibrator.predict(raw_scores), dtype=float)
        else:
            probabilities = np.asarray(estimator.predict_proba(values), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                raise ValueError("Underlying estimator must return positive-class probabilities.")
            result = probabilities[:, 1]
            if calibrator is not None:
                result = np.asarray(calibrator.predict(result), dtype=float)
        result = np.asarray(result, dtype=float)
        if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise ValueError("Model returned invalid positive-class probabilities.")
        return result

    return predict_positive_probability, feature_names, preprocessor


def run_shap_explanations(
    model_path: Path,
    csv_path: Path = DEFAULT_DATA,
    output_dir: Path = Path("outputs/explainability"),
    background_size: int = 100,
    explain_size: int = 40,
    random_state: int = 42,
) -> dict[str, Any]:
    """Write global/local SHAP artifacts in the fitted numeric/one-hot feature space."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if background_size < 2 or explain_size < 1:
        raise ValueError("background_size must be >= 2 and explain_size must be >= 1.")

    import joblib
    import matplotlib.pyplot as plt
    import shap

    raw_df, dataset_sha256 = load_dataset(csv_path)
    engineered = engineer_features(raw_df)
    X, y = split_features_target(engineered)

    # Recreate the deterministic 80/20 boundary used by finalization; explain development rows only.
    X_development, _X_holdout, y_development, _y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    if len(X_development) < 5:
        raise ValueError("Not enough development rows for SHAP explanation.")

    model = joblib.load(model_path)
    predict_encoded, feature_names, preprocessor = make_encoded_probability_predictor(model)
    encoded_all = np.asarray(preprocessor.transform(X_development), dtype=float)
    if encoded_all.ndim != 2 or encoded_all.shape[1] != len(feature_names):
        raise ValueError(f"Unexpected encoded feature shape: {encoded_all.shape}")

    encoded_df = pd.DataFrame(encoded_all, columns=feature_names, index=X_development.index)
    background_count = min(int(background_size), max(2, len(X_development) // 4))
    background = encoded_df.sample(n=background_count, random_state=random_state)
    explain_pool = encoded_df.drop(index=background.index)
    if explain_pool.empty:
        explain_pool = encoded_df
    sample_count = min(int(explain_size), len(explain_pool))
    explain_rows = explain_pool.sample(n=sample_count, random_state=random_state + 1)

    # Numeric transformed data avoids np.isclose/string subtraction in SHAP's tabular masker.
    explainer = shap.Explainer(
        predict_encoded,
        background,
        algorithm="permutation",
        feature_names=feature_names,
    )
    explanation = explainer(
        explain_rows,
        max_evals=(2 * encoded_df.shape[1] + 1),
    )

    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2 or values.shape != (len(explain_rows), encoded_df.shape[1]):
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_value": np.mean(np.abs(values), axis=0),
        }
    ).sort_values("mean_abs_shap_value", ascending=False, ignore_index=True)
    importance_path = output_dir / "shap_global_importance.csv"
    global_importance.to_csv(importance_path, index=False)

    local_rows = []
    encoded_values = explain_rows.to_numpy(dtype=float)
    base_values = np.asarray(explanation.base_values)
    for row_position, row_index in enumerate(explain_rows.index):
        base_value = np.asarray(base_values[row_position]).reshape(-1)
        base_scalar = float(base_value[0]) if base_value.size else float("nan")
        for feature_position, feature_name in enumerate(feature_names):
            local_rows.append(
                {
                    "example_row_index": int(row_index),
                    "base_value": base_scalar,
                    "feature": feature_name,
                    "feature_value_encoded": float(encoded_values[row_position, feature_position]),
                    "shap_value": float(values[row_position, feature_position]),
                }
            )
    local_path = output_dir / "shap_local_explanations.csv"
    pd.DataFrame(local_rows).to_csv(local_path, index=False)

    shap.plots.bar(explanation, max_display=min(20, encoded_df.shape[1]), show=False)
    global_plot_path = output_dir / "shap_global_bar.png"
    plt.savefig(global_plot_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    shap.plots.beeswarm(explanation, max_display=min(20, encoded_df.shape[1]), show=False)
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    report_path = output_dir / "shap_report.json"
    report = {
        "dataset_sha256": dataset_sha256,
        "target_column": "StateNextMonth",
        "model_path": str(model_path),
        "rows_in_development_partition": int(len(X_development)),
        "background_rows": int(len(background)),
        "explained_rows": int(len(explain_rows)),
        "holdout_rows_used_for_explanations": 0,
        "raw_feature_count": int(X_development.shape[1]),
        "encoded_feature_count": int(encoded_df.shape[1]),
        "feature_space": (
            "Fitted numeric scaling and one-hot encoding. Numeric SHAP values are in "
            "model-input space; one-hot feature names identify individual category levels."
        ),
        "global_importance_csv": str(importance_path),
        "local_explanations_csv": str(local_path),
        "global_bar_plot": str(global_plot_path),
        "beeswarm_plot": str(beeswarm_path),
        "report_path": str(report_path),
        "top_features": global_importance.head(10).to_dict(orient="records"),
        "warning": (
            "SHAP values explain model associations on the development sample; they do not "
            "prove causation or a future next-month churn outcome."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate global and local SHAP explanations for a saved final pipeline."
    )
    parser.add_argument("--model", type=Path, default=Path("outputs/final_pipeline.joblib"))
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explainability"))
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--explain-size", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    report = run_shap_explanations(
        model_path=args.model,
        csv_path=args.csv,
        output_dir=args.output_dir,
        background_size=args.background_size,
        explain_size=args.explain_size,
        random_state=args.random_state,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

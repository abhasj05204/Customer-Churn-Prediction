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


def run_shap_explanations(
    model_path: Path,
    csv_path: Path = DEFAULT_DATA,
    output_dir: Path = Path("outputs/explainability"),
    background_size: int = 100,
    explain_size: int = 40,
    random_state: int = 42,
) -> dict[str, Any]:
    """Write global/local SHAP artifacts using only development rows."""
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
    background_count = min(int(background_size), max(2, len(X_development) // 4))
    background = X_development.sample(n=background_count, random_state=random_state)
    explain_pool = X_development.drop(index=background.index)
    if explain_pool.empty:
        explain_pool = X_development
    sample_count = min(int(explain_size), len(explain_pool))
    explain_rows = explain_pool.sample(n=sample_count, random_state=random_state + 1)

    predictor = make_probability_predictor(model, X_development)
    explainer = shap.Explainer(
        predictor,
        background,
        algorithm="permutation",
        feature_names=list(X_development.columns),
    )
    explanation = explainer(
        explain_rows,
        max_evals=(2 * X_development.shape[1] + 1),
    )

    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2 or values.shape != (len(explain_rows), X_development.shape[1]):
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    global_importance = pd.DataFrame(
        {
            "feature": list(X_development.columns),
            "mean_abs_shap_value": np.mean(np.abs(values), axis=0),
        }
    ).sort_values("mean_abs_shap_value", ascending=False, ignore_index=True)
    importance_path = output_dir / "shap_global_importance.csv"
    global_importance.to_csv(importance_path, index=False)

    local_rows = []
    raw_values = explain_rows.to_numpy(dtype=object)
    base_values = np.asarray(explanation.base_values)
    for row_position, row_index in enumerate(explain_rows.index):
        base_value = np.asarray(base_values[row_position]).reshape(-1)
        base_scalar = float(base_value[0]) if base_value.size else float("nan")
        for feature_position, feature_name in enumerate(X_development.columns):
            local_rows.append(
                {
                    "example_row_index": int(row_index),
                    "base_value": base_scalar,
                    "feature": str(feature_name),
                    "feature_value": str(raw_values[row_position, feature_position]),
                    "shap_value": float(values[row_position, feature_position]),
                }
            )
    local_path = output_dir / "shap_local_explanations.csv"
    pd.DataFrame(local_rows).to_csv(local_path, index=False)

    shap.plots.bar(explanation, max_display=min(20, X_development.shape[1]), show=False)
    global_plot_path = output_dir / "shap_global_bar.png"
    plt.savefig(global_plot_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    shap.plots.beeswarm(explanation, max_display=min(20, X_development.shape[1]), show=False)
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    report_path = output_dir / "shap_report.json"
    report = {
        "dataset_sha256": dataset_sha256,
        "target_column": "State",
        "model_path": str(model_path),
        "rows_in_development_partition": int(len(X_development)),
        "background_rows": int(len(background)),
        "explained_rows": int(len(explain_rows)),
        "holdout_rows_used_for_explanations": 0,
        "feature_count": int(X_development.shape[1]),
        "global_importance_csv": str(importance_path),
        "local_explanations_csv": str(local_path),
        "global_bar_plot": str(global_plot_path),
        "beeswarm_plot": str(beeswarm_path),
        "report_path": str(report_path),
        "top_features": global_importance.head(10).to_dict(orient="records"),
        "warning": (
            "SHAP values explain model associations on the selected development sample; "
            "they do not prove causation or a future next-month churn outcome."
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

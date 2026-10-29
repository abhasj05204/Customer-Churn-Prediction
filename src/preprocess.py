"""Leakage-safe preprocessing factories for raw engineered feature frames."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create an unfitted transformer; fit it only inside training/CV pipelines."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature.")
    if X.columns.duplicated().any():
        duplicates = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"X contains duplicate feature names: {duplicates}")

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    """Return preprocessing + estimator as one cloneable sklearn Pipeline.

    Passing this pipeline to cross-validation ensures imputers, scaling, and
    category encoding are fit independently inside each training fold.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", estimator),
        ]
    )

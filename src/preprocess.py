"""Leakage-safe preprocessing and imbalance-handling pipeline factories."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, TransformerMixin
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


class MixedFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Impute/scale numerics while preserving raw categorical values for SMOTENC."""

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MixedFeaturePreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MixedFeaturePreprocessor requires a pandas DataFrame.")
        if X.empty or X.shape[1] == 0:
            raise ValueError("X must contain at least one row and one feature.")
        if X.columns.duplicated().any():
            raise ValueError("X contains duplicate feature names.")

        self.feature_names_in_ = list(X.columns)
        self.numeric_columns_ = (
            list(self.numeric_columns)
            if self.numeric_columns is not None
            else X.select_dtypes(include=[np.number]).columns.tolist()
        )
        self.categorical_columns_ = (
            list(self.categorical_columns)
            if self.categorical_columns is not None
            else [column for column in X.columns if column not in self.numeric_columns_]
        )
        declared = set(self.numeric_columns_) | set(self.categorical_columns_)
        if declared != set(self.feature_names_in_):
            raise ValueError("Numeric and categorical columns must partition all feature columns.")
        if not self.numeric_columns_ or not self.categorical_columns_:
            raise ValueError("SMOTENC requires at least one numeric and one categorical feature.")

        self.numeric_imputer_ = SimpleImputer(strategy="median")
        numeric_values = self.numeric_imputer_.fit_transform(X[self.numeric_columns_])
        self.scaler_ = StandardScaler().fit(numeric_values)

        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
        self.categorical_imputer_.fit(X[self.categorical_columns_])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MixedFeaturePreprocessor requires a pandas DataFrame.")
        missing = [column for column in self.feature_names_in_ if column not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns during transform: {missing}")

        frame = pd.DataFrame(index=X.index)
        if self.numeric_columns_:
            numeric = self.scaler_.transform(
                self.numeric_imputer_.transform(X[self.numeric_columns_])
            )
            for index, column in enumerate(self.numeric_columns_):
                frame[column] = numeric[:, index]
        if self.categorical_columns_:
            categorical = self.categorical_imputer_.transform(X[self.categorical_columns_])
            for index, column in enumerate(self.categorical_columns_):
                frame[column] = categorical[:, index].astype(object)
        return frame.loc[:, self.feature_names_in_]


def build_smote_pipeline(
    X: pd.DataFrame,
    estimator: Any,
    random_state: int = 42,
) -> ImbalancedPipeline:
    """Experimental ordinary SMOTE after one-hot encoding, fitted inside each fold.

    This option is included for a controlled comparison only. Interpolation across
    one-hot flags can create fractional category indicators; prefer class weighting
    or build_smotenc_pipeline when this is unsuitable.
    """
    return ImbalancedPipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("sampler", SMOTE(random_state=random_state)),
            ("model", estimator),
        ]
    )


def build_smotenc_pipeline(
    X: pd.DataFrame,
    estimator: Any,
    random_state: int = 42,
) -> ImbalancedPipeline:
    """Build a SMOTENC pipeline that preserves categorical feature values.

    Numeric fields are imputed and scaled fold-locally; categorical values are
    imputed but remain unencoded through SMOTENC. One-hot encoding occurs after
    sampling, still within each training fold. The sampler is automatically skipped
    by imbalanced-learn's pipeline during predict/predict_proba.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty or X.columns.duplicated().any():
        raise ValueError("X must be non-empty with unique feature names.")

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    if not numeric_columns or not categorical_columns:
        raise ValueError("SMOTENC requires at least one numeric and one categorical feature.")

    mixed_preprocessor = MixedFeaturePreprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    post_sampling_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ImbalancedPipeline(
        steps=[
            ("mixed_preprocessor", mixed_preprocessor),
            ("sampler", SMOTENC(
                categorical_features=categorical_columns,
                random_state=random_state,
            )),
            ("post_sampling_preprocessor", post_sampling_preprocessor),
            ("model", estimator),
        ]
    )

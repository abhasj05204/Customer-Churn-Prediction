"""Reusable feature engineering for training/inference on StateNextMonth."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_MAP = {"No Churn": 0, "Churn": 1}
REQUIRED_INPUT_COLUMNS = (
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "State", "StateNextMonth",
)
RAW_FEATURE_COLUMNS = tuple(
    column for column in REQUIRED_INPUT_COLUMNS if column not in {"customerID", "State", "StateNextMonth"}
)
SERVICE_COLUMNS = (
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
)
SECURITY_SUPPORT_COLUMNS = (
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
)


def engineer_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw customer features without requiring or using target labels.

    Accepts the 19 declared raw feature columns or a source dataset that also
    contains identifiers/labels. Only the whitelisted feature contract is retained;
    all identifiers, targets, timestamps, and unreviewed extra columns are excluded.
    """
    if df.empty:
        raise ValueError("Cannot engineer features from an empty dataframe.")
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required input feature columns: {missing}")

    # Whitelist only the declared predictor contract. This prevents an unreviewed
    # extra column (including a label-like field) from silently entering training
    # or inference features.
    out = df.loc[:, list(RAW_FEATURE_COLUMNS)].copy()

    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        converted = pd.to_numeric(out[column], errors="coerce")
        bad = int((out[column].notna() & converted.isna()).sum())
        if bad:
            raise ValueError(f"Column {column!r} has {bad} non-numeric values.")
        out[column] = converted

    out["TenureGroup"] = pd.cut(
        out["tenure"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-5 Years", "5+ Years"],
    )

    avg_charges = out["TotalCharges"] / out["tenure"].replace(0, np.nan)
    out["AvgMonthlyCharges"] = avg_charges.fillna(out["MonthlyCharges"])

    out["ServiceCount"] = out[list(SERVICE_COLUMNS)].eq("Yes").sum(axis=1).astype(int)
    out["SecuritySupportCount"] = (
        out[list(SECURITY_SUPPORT_COLUMNS)].eq("Yes").sum(axis=1).astype(int)
    )
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return an engineered training frame with a binary StateNextMonth target."""
    if "StateNextMonth" not in df.columns:
        raise ValueError("Training dataframe must contain the StateNextMonth target column.")
    target = df["StateNextMonth"].astype("string").str.strip()
    invalid = sorted(
        str(value) for value in target[~target.isin(TARGET_MAP)].dropna().unique()
    )
    missing_target = int(target.isna().sum())
    if invalid or missing_target:
        raise ValueError(
            f"Invalid StateNextMonth target values: {invalid}; missing target rows: {missing_target}."
        )

    out = engineer_inference_features(df)
    out["StateNextMonth"] = target.map(TARGET_MAP).astype("int8")
    return out


def split_features_target(engineered_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split engineered data into predictors and binary StateNextMonth target."""
    if "StateNextMonth" not in engineered_df.columns:
        raise ValueError("Engineered dataframe must contain the StateNextMonth target column.")
    if engineered_df["StateNextMonth"].isna().any():
        raise ValueError("StateNextMonth target contains missing values.")
    X = engineered_df.drop(columns=["StateNextMonth"]).copy()
    y = engineered_df["StateNextMonth"].copy().astype("int8")
    forbidden = [c for c in ("customerID", "State", "StateNextMonth") if c in X.columns]
    if forbidden:
        raise ValueError(
            f"Identifier/current-target columns must not be predictors: {forbidden}"
        )
    return X, y

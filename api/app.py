"""FastAPI endpoints for the saved customer churn pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.feature_engineering import engineer_inference_features

DEFAULT_MODEL_PATH = Path("outputs/final_pipeline.joblib")
DEFAULT_REPORT_PATH = Path("outputs/final_evaluation.json")


class CustomerFeatures(BaseModel):
    """Raw features only; the target and customer identifier are not accepted."""
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    predicted_state: Literal["No Churn", "Churn"]
    decision_threshold: float = Field(gt=0, lt=1)
    model_name: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


def _read_threshold(report_path: Path, fallback: float) -> float:
    if not report_path.is_file():
        return fallback
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        threshold = float(data.get("final_threshold", fallback))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback
    return threshold if 0.0 < threshold < 1.0 else fallback


def _model_name(model: object) -> str:
    """Show the underlying classifier even when probabilities are calibrated."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return type(model.named_steps["model"]).__name__
    base_estimator = getattr(model, "estimator", None)
    if base_estimator is not None and hasattr(base_estimator, "named_steps"):
        if "model" in base_estimator.named_steps:
            return f"{type(base_estimator.named_steps['model']).__name__} (calibrated)"
    return type(model).__name__


def create_app(
    model: object | None = None,
    model_path: Path | None = None,
    threshold: float | None = None,
    report_path: Path | None = None,
) -> FastAPI:
    """Create the API; model injection is supported for tests."""
    resolved_model_path = model_path or Path(os.getenv("CHURN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    resolved_report_path = report_path or Path(os.getenv("CHURN_REPORT_PATH", str(DEFAULT_REPORT_PATH)))
    fallback_threshold = float(os.getenv("CHURN_THRESHOLD", "0.5"))
    chosen_threshold = fallback_threshold if threshold is None else float(threshold)
    if not 0.0 < chosen_threshold < 1.0:
        raise ValueError("decision threshold must be strictly between 0 and 1.")

    loaded_model = model
    if loaded_model is None and resolved_model_path.is_file():
        loaded_model = joblib.load(resolved_model_path)
    if threshold is None:
        chosen_threshold = _read_threshold(resolved_report_path, fallback_threshold)

    application = FastAPI(
        title="Customer Churn Prediction API",
        version="1.0.0",
        description=(
            "Predicts the supplied StateNextMonth label from customer service and billing features. "
            "The target provenance and temporal alignment have not been independently verified."
        ),
    )
    application.state.model = loaded_model
    application.state.model_path = str(resolved_model_path)
    application.state.threshold = chosen_threshold
    application.state.model_name = _model_name(loaded_model) if loaded_model is not None else None

    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "CHURN_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @application.get("/health")
    def health() -> dict[str, object]:
        model_loaded = application.state.model is not None
        return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded}

    @application.get("/model-info")
    def model_info() -> dict[str, object]:
        return {
            "model_loaded": application.state.model is not None,
            "model_name": application.state.model_name,
            "target": "StateNextMonth",
            "decision_threshold": float(application.state.threshold),
            "prediction_horizon": "not established by the selected dataset",
        }

    def predict_one(customer: CustomerFeatures) -> PredictionResponse:
        if application.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifact unavailable. Run 'python -m src.finalize --output-dir outputs' first.",
            )
        try:
            raw_features = pd.DataFrame([customer.model_dump()])
            engineered = engineer_inference_features(raw_features)
            probabilities = np.asarray(application.state.model.predict_proba(engineered), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] < 2:
                raise ValueError("Model returned an invalid probability array.")
            churn_probability = float(probabilities[0, 1])
            if not np.isfinite(churn_probability) or not 0.0 <= churn_probability <= 1.0:
                raise ValueError("Model returned an invalid churn probability.")
        except HTTPException:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=f"Could not process customer input: {exc}") from exc

        label = "Churn" if churn_probability >= application.state.threshold else "No Churn"
        return PredictionResponse(
            churn_probability=churn_probability,
            predicted_state=label,
            decision_threshold=float(application.state.threshold),
            model_name=application.state.model_name or "unknown",
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(customer: CustomerFeatures) -> PredictionResponse:
        return predict_one(customer)

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        predictions = [predict_one(customer) for customer in request.customers]
        return BatchPredictionResponse(predictions=predictions, count=len(predictions))

    return application


app = create_app()

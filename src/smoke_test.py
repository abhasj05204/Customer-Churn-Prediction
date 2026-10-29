"""End-to-end smoke test for a newly saved final model and the FastAPI inference path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.app import create_app
from src.train import DEFAULT_DATA


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        raise ValueError("Smoke-test row contains a missing feature value.")
    return value


def payloads_from_dataset(csv_path: Path, sample_count: int = 2, random_state: int = 42) -> list[dict[str, Any]]:
    """Build API requests from real rows while stripping identifier and target fields."""
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    selected = df.sample(n=min(sample_count, len(df)), random_state=random_state)
    forbidden = ["customerID", "State", "StateNextMonth"]
    selected = selected.drop(columns=[column for column in forbidden if column in selected.columns])
    payloads = []
    for _, row in selected.iterrows():
        payload = {column: _json_scalar(value) for column, value in row.items()}
        payload["SeniorCitizen"] = int(payload["SeniorCitizen"])
        payload["tenure"] = int(payload["tenure"])
        payload["MonthlyCharges"] = float(payload["MonthlyCharges"])
        payload["TotalCharges"] = float(payload["TotalCharges"])
        payloads.append(payload)
    return payloads


def run_smoke_test(
    model_path: Path = Path("outputs/final_pipeline.joblib"),
    report_path: Path = Path("outputs/final_evaluation.json"),
    csv_path: Path = DEFAULT_DATA,
) -> dict[str, Any]:
    """Load a trained artifact and verify single/batch API predictions."""
    for path, label in ((model_path, "model"), (report_path, "evaluation report")):
        if not path.is_file():
            raise FileNotFoundError(f"Saved {label} not found: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = float(report["final_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid threshold in report: {threshold}")

    model = joblib.load(model_path)
    payloads = payloads_from_dataset(csv_path, sample_count=2)
    client = TestClient(create_app(model=model, threshold=threshold))

    health = client.get("/health")
    if health.status_code != 200 or not health.json().get("model_loaded"):
        raise RuntimeError(f"Health check failed: {health.status_code} {health.text}")

    single_response = client.post("/predict", json=payloads[0])
    if single_response.status_code != 200:
        raise RuntimeError(f"Single prediction failed: {single_response.status_code} {single_response.text}")
    single = single_response.json()
    if single["predicted_state"] not in {"No Churn", "Churn"}:
        raise RuntimeError(f"Unexpected prediction label: {single['predicted_state']}")
    if not 0.0 <= float(single["churn_probability"]) <= 1.0:
        raise RuntimeError(f"Invalid probability: {single['churn_probability']}")

    batch_response = client.post("/predict/batch", json={"customers": payloads})
    if batch_response.status_code != 200:
        raise RuntimeError(f"Batch prediction failed: {batch_response.status_code} {batch_response.text}")
    batch = batch_response.json()
    if batch["count"] != len(payloads) or len(batch["predictions"]) != len(payloads):
        raise RuntimeError("Batch API response count does not match input rows.")

    return {
        "status": "passed",
        "model_artifact": str(model_path),
        "model_type": type(model).__name__,
        "target": "StateNextMonth",
        "threshold": threshold,
        "single_prediction": single,
        "batch_prediction_count": batch["count"],
        "source_rows_used": len(payloads),
        "target_and_identifier_fields_sent_to_api": False,
        "warning": (
            "This validates model/API wiring with source rows. It does not verify the provenance "
            "or temporal alignment of StateNextMonth labels, or establish business performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the saved model through FastAPI.")
    parser.add_argument("--model", type=Path, default=Path("outputs/final_pipeline.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/final_evaluation.json"))
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(
        run_smoke_test(model_path=args.model, report_path=args.report, csv_path=args.csv),
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()

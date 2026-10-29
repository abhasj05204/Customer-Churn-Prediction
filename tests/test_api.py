"""Endpoint tests using an injected deterministic model."""
import unittest

import numpy as np
from fastapi.testclient import TestClient

from api.app import create_app


class DummyModel:
    def predict_proba(self, frame):
        positive = np.clip(frame["tenure"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


def sample_customer() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 80,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 7600.0,
    }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(model=DummyModel(), threshold=0.5))

    def test_health_and_model_info(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_loaded"])
        info = self.client.get("/model-info")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["target"], "State")

    def test_single_prediction_does_not_require_target_or_customer_id(self) -> None:
        response = self.client.post("/predict", json=sample_customer())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertAlmostEqual(body["churn_probability"], 0.8)
        self.assertEqual(body["predicted_state"], "Churn")
        self.assertEqual(body["decision_threshold"], 0.5)

    def test_batch_prediction(self) -> None:
        response = self.client.post(
            "/predict/batch",
            json={"customers": [sample_customer(), {**sample_customer(), "tenure": 10}]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["count"], 2)
        self.assertEqual(len(response.json()["predictions"]), 2)

    def test_missing_model_returns_service_unavailable(self) -> None:
        from pathlib import Path
        client = TestClient(create_app(model=None, model_path=Path("/missing/model.joblib")))
        response = client.post("/predict", json=sample_customer())
        self.assertEqual(response.status_code, 503)

    def test_rejects_unknown_categories_and_extra_target_field(self) -> None:
        bad = sample_customer()
        bad["Contract"] = "Monthly"
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 422)
        bad = sample_customer()
        bad["State"] = "Churn"
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

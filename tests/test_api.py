from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.inference import DEFAULT_CHAMPION_PATH


pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_CHAMPION_PATH).exists(),
    reason="Champion model artifact is required for API prediction smoke tests.",
)


def test_health_reports_ready_model() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True
    assert payload["champion_model"]


def test_predict_sentiment_returns_required_fields() -> None:
    client = TestClient(app)
    text = "\u0b87\u0ba8\u0bcd\u0ba4 service romba nalla irukku"

    response = client.post("/predict/sentiment", json={"text": text})

    assert response.status_code == 200
    payload = response.json()
    assert {"sentiment", "confidence", "model", "model_name", "cleaned_text", "top_probabilities"} <= set(payload)
    assert isinstance(payload["sentiment"], str)
    assert 0 <= payload["confidence"] <= 1
    assert payload["model"] in {"baseline", "transformer"}
    assert "service" in payload["cleaned_text"]
    assert 1 <= len(payload["top_probabilities"]) <= 3

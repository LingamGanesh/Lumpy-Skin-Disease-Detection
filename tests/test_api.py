from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
import numpy as np

from api import app, get_bundle


def test_health_reports_unavailable_without_training_data(monkeypatch):
    def fail():
        raise FileNotFoundError("fixture: no dataset")
    get_bundle.cache_clear()
    monkeypatch.setattr("api.train_model", fail)
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    get_bundle.cache_clear()


def test_predict_rejects_unsupported_content_type():
    response = TestClient(app).post(
        "/predict", files={"file": ("note.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415


def test_predict_rejects_invalid_image_bytes(monkeypatch):
    class FakeModel:
        pass
    # Exercise validation before any model access.
    response = TestClient(app).post(
        "/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 400

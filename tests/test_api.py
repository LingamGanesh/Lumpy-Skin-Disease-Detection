from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import api
import lumpycare.database as database
from lumpycare.model import ModelBundle


class FakeClassifier:
    def predict(self, _features):
        return np.array([0])

    def predict_proba(self, _features):
        return np.array([[0.9, 0.1]])


def _jpeg_bytes():
    pixels = np.random.default_rng(11).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_unavailable_when_model_fails(monkeypatch):
    def fail():
        raise FileNotFoundError("fixture: no dataset")
    api.get_bundle.cache_clear()
    monkeypatch.setattr(api, "train_model", fail)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 503
    api.get_bundle.cache_clear()


def test_predict_rejects_unsupported_content_type():
    response = TestClient(api.app).post(
        "/predict", files={"file": ("note.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415


def test_predict_rejects_invalid_image_bytes(monkeypatch):
    monkeypatch.setattr(
        api, "get_bundle",
        lambda: ModelBundle(FakeClassifier(), {}, "test fixture")
    )
    response = TestClient(api.app).post(
        "/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 400


def test_predict_returns_prediction_for_valid_image(monkeypatch):
    monkeypatch.setattr(
        api, "get_bundle",
        lambda: ModelBundle(FakeClassifier(), {}, "test fixture")
    )
    response = TestClient(api.app).post(
        "/predict",
        files={"file": ("cattle.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "Lumpy skin"
    assert payload["confidence"] == 0.9
    assert "disclaimer" in payload



def test_analytics_stats_counts_persisted_scans(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_lumpycare.db")
    database.init_db()
    database.save_scan("Lumpy skin", 0.91, image_name="sample.jpg", source="test")
    database.save_scan("Normal skin", 0.84, image_name="healthy.jpg", source="test")
    response = TestClient(api.app).get("/analytics/stats?include_demo=false")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_scans"] == 2
    assert payload["infected"] == 1
    assert payload["healthy"] == 1
    assert payload["infection_rate"] == 0.5


def test_cattle_profile_create_and_list(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "cattle.db")
    response = TestClient(api.app).post(
        "/cattle",
        json={"cattle_id": "COW-001", "name": "Lakshmi", "age_years": 4, "breed": "Gir"},
    )
    assert response.status_code == 201
    listing = TestClient(api.app).get("/cattle")
    assert listing.status_code == 200
    assert listing.json()["items"][0]["cattle_id"] == "COW-001"

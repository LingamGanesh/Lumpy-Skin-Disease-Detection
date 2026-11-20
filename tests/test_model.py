import numpy as np
import pytest

from lumpycare.model import ModelBundle, extract_features, predict_rgb


class FakeClassifier:
    def predict(self, _features):
        return np.array([1])

    def predict_proba(self, _features):
        return np.array([[0.15, 0.85]])


def test_extract_features_returns_expected_vector():
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    vector = extract_features(image)
    assert vector.shape == (128 * 128 * 3,)
    assert vector.dtype == np.float32


def test_extract_features_is_deterministic():
    image = np.random.default_rng(7).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    first = extract_features(image)
    second = extract_features(image)
    np.testing.assert_array_equal(first, second)


def test_extract_features_rejects_empty_input():
    with pytest.raises(ValueError):
        extract_features(np.empty((0, 0, 3), dtype=np.uint8))


def test_predict_rgb_returns_class_and_probability():
    bundle = ModelBundle(FakeClassifier(), {}, "test fixture")
    result = predict_rgb(bundle, np.zeros((128, 128, 3), dtype=np.uint8))
    assert result["prediction"] == "Normal skin"
    assert result["class_id"] == 1
    assert result["confidence"] == pytest.approx(0.85)
    assert result["probabilities"]["Lumpy skin"] == pytest.approx(0.15)

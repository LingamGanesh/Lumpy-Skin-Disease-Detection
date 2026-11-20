import numpy as np
import pytest

from lumpycare.model import extract_features


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

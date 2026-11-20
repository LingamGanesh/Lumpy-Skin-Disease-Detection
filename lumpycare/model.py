"""Shared, deterministic SVM inference and evaluation for LumpyCare."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("Lumpy skin", "Normal skin")
IMAGE_SIZE = (128, 128)
FEATURES_FILE = ROOT / "features" / "features.txt.npy"
LABELS_FILE = ROOT / "features" / "labels.txt.npy"
DATASET_DIR = ROOT / "SkinDataset"


@dataclass
class ModelBundle:
    model: SVC
    metrics: dict
    source: str


def extract_features(rgb_image: np.ndarray) -> np.ndarray:
    """Resize RGB pixels and apply the project's six-cluster K-means transform."""
    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("The supplied image is empty.")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected a three-channel RGB image.")
    resized = cv2.resize(rgb_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    pixels = np.float32(resized.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # Match the original notebook's RANDOM_CENTERS strategy, but make inference repeatable.
    cv2.setRNGSeed(42)
    _compactness, labels, centers = cv2.kmeans(
        pixels, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    return np.uint8(centers)[labels.flatten()].reshape(-1).astype(np.float32)


def _load_training_data() -> tuple[np.ndarray, np.ndarray, str]:
    """Prefer raw labelled images so training uses the same deterministic transform as inference."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, class_name in enumerate(CLASSES):
        folder = DATASET_DIR / class_name
        if not folder.is_dir():
            xs = []
            ys = []
            break
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xs.append(extract_features(rgb))
            ys.append(label)
    if xs:
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), "SkinDataset (reprocessed)"

    if FEATURES_FILE.is_file() and LABELS_FILE.is_file():
        X = np.load(FEATURES_FILE, allow_pickle=False)
        y = np.load(LABELS_FILE, allow_pickle=False)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64).ravel(), "precomputed feature arrays"
    raise FileNotFoundError(
        "Training data is missing. Add SkinDataset/Lumpy skin and SkinDataset/Normal skin, "
        "or features/features.txt.npy and features/labels.txt.npy."
    )


def train_model() -> ModelBundle:
    X, y, source = _load_training_data()
    if X.ndim != 2 or X.shape[1] != IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3:
        raise ValueError(f"Unexpected feature matrix shape {X.shape}; expected (*, 49152).")
    if len(X) != len(y) or len(np.unique(y)) != 2:
        raise ValueError("Training data must have matching samples and both class labels (0 and 1).")
    if len(X) < 10:
        raise ValueError("At least 10 labelled images are required for a stratified 80/20 evaluation.")
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2 or counts.min() < 2:
        raise ValueError("Each class needs at least two images for a stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classifier = SVC(C=12, gamma="scale", kernel="rbf", probability=True, random_state=42)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    report = classification_report(
        y_test, predicted, labels=[0, 1], target_names=CLASSES,
        output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]),
        "classification_report": report,
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "total_count": int(len(X)),
        "class_counts": {CLASSES[int(k)]: int(v) for k, v in zip(unique, counts)},
    }
    return ModelBundle(classifier, metrics, source)


def predict_rgb(bundle: ModelBundle, rgb_image: np.ndarray) -> dict:
    vector = extract_features(rgb_image).reshape(1, -1)
    predicted = int(bundle.model.predict(vector)[0])
    probabilities = bundle.model.predict_proba(vector)[0]
    confidence = float(probabilities[predicted])
    return {
        "prediction": CLASSES[predicted] if predicted in (0, 1) else str(predicted),
        "class_id": predicted,
        "confidence": confidence,
        "probabilities": {CLASSES[int(i)]: float(p) for i, p in enumerate(probabilities)},
    }

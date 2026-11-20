"""Optional FastAPI inference service for LumpyCare."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from lumpycare.model import ModelBundle, predict_rgb, train_model


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return train_model()


app = FastAPI(
    title="LumpyCare API",
    version="1.0.0",
    description="Image-based Lumpy Skin Disease screening research API. Not a veterinary diagnosis.",
)


@app.get("/health")
def health():
    try:
        bundle = get_bundle()
        return {"status": "ok", "model": "SVM", "dataset_source": bundle.source}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")
    raw = await file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
        rgb = np.asarray(image)
        result = predict_rgb(get_bundle(), rgb)
        return {
            **result,
            "disclaimer": "Screening support only. A qualified veterinarian must confirm diagnosis.",
        }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction unavailable: {exc}") from exc

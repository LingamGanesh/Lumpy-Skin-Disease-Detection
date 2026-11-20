"""FastAPI inference and analytics service for LumpyCare."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import sqlite3

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from lumpycare.database import (
    analytics_activity, analytics_alerts, analytics_distribution, analytics_geo,
    analytics_stats, analytics_trends, create_cattle, get_cattle, init_db, list_cattle,
)
from lumpycare.model import ModelBundle, predict_rgb, train_model


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return train_model()


app = FastAPI(
    title="LumpyCare API",
    version="1.1.0",
    description="Image-based Lumpy Skin Disease screening research API. Not a veterinary diagnosis.",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


class CattleCreate(BaseModel):
    cattle_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    age_years: float | None = Field(default=None, ge=0, le=30)
    breed: str = Field(default="", max_length=80)
    village: str = Field(default="", max_length=120)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


def _filters(
    start_date: str | None, end_date: str | None, result_type: str | None,
    min_confidence: float, max_confidence: float, cattle_id: str | None,
    location: str | None, include_demo: bool,
) -> dict:
    if min_confidence > max_confidence:
        raise HTTPException(status_code=422, detail="min_confidence cannot exceed max_confidence")
    if result_type not in {None, "Lumpy skin", "Normal skin"}:
        raise HTTPException(status_code=422, detail="result_type must be Lumpy skin or Normal skin")
    return {
        "start_date": start_date, "end_date": end_date, "result_type": result_type,
        "min_confidence": min_confidence, "max_confidence": max_confidence,
        "cattle_id": cattle_id, "location": location, "include_demo": include_demo,
    }


@app.get("/health")
def health():
    try:
        bundle = get_bundle()
        init_db()
        return {"status": "ok", "model": "SVM", "dataset_source": bundle.source, "database": "sqlite"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}") from exc


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


@app.get("/cattle")
def cattle_list():
    init_db()
    return {"items": list_cattle()}


@app.post("/cattle", status_code=201)
def cattle_create(payload: CattleCreate):
    try:
        return create_cattle(**payload.dict())
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Cattle ID already exists or related data is invalid.") from exc


@app.get("/cattle/{cattle_id}")
def cattle_detail(cattle_id: str):
    animal = get_cattle(cattle_id)
    if not animal:
        raise HTTPException(status_code=404, detail="Cattle profile not found.")
    return animal


def filter_parameters(
    start_date: str | None = None,
    end_date: str | None = None,
    result_type: str | None = None,
    min_confidence: float = Query(default=0, ge=0, le=1),
    max_confidence: float = Query(default=1, ge=0, le=1),
    cattle_id: str | None = None,
    location: str | None = None,
    include_demo: bool = True,
) -> dict:
    return _filters(start_date, end_date, result_type, min_confidence, max_confidence,
                    cattle_id, location, include_demo)


@app.get("/analytics/stats")
def analytics_stats_route(filters: dict = Depends(filter_parameters)):
    return analytics_stats(**filters)


@app.get("/analytics/trends")
def analytics_trends_route(filters: dict = __import__("fastapi").Depends(filter_parameters)):
    return {"items": analytics_trends(**filters)}


@app.get("/analytics/distribution")
def analytics_distribution_route(filters: dict = __import__("fastapi").Depends(filter_parameters)):
    return analytics_distribution(**filters)


@app.get("/analytics/activity")
def analytics_activity_route(
    limit: int = Query(default=20, ge=1, le=100),
    filters: dict = __import__("fastapi").Depends(filter_parameters),
):
    return {"items": analytics_activity(limit=limit, **filters)}


@app.get("/analytics/geo")
def analytics_geo_route(filters: dict = __import__("fastapi").Depends(filter_parameters)):
    return {"items": analytics_geo(**filters)}


@app.get("/analytics/alerts")
def analytics_alerts_route(
    threshold: float = Query(default=.30, ge=0, le=1),
    filters: dict = __import__("fastapi").Depends(filter_parameters),
):
    return {"threshold": threshold, "items": analytics_alerts(threshold=threshold, **filters)}

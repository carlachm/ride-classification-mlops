"""Scoring API: classifies one IMU ride into Safe / Safe overall / Bad weather / Tandem / Reckless.

    uv run uvicorn fraud_detection.api.main:app --port 8000     # Swagger UI: http://localhost:8000/docs

Endpoints: GET /health, POST /v1/predict, POST /v1/feedback, GET /metrics (Prometheus).
Every prediction is appended to PREDICTION_LOG_PATH (JSON lines) for drift tests and monitoring.
The API starts even without a model: /health says "degraded" and /v1/predict answers 503.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app

from fraud_detection.api import metrics as m
from fraud_detection.api import model_store
from fraud_detection.api.features import ride_fingerprint, ride_summary
from fraud_detection.api.ride_store import save_ride
from fraud_detection.api.schemas import CHANNELS, CLASSES, Feedback, Health, Prediction, Ride

logger = logging.getLogger("fraud_detection.api")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.propagate = False

state: dict = {"loaded": None}


def log_event(event: dict) -> None:
    """One JSON line on stdout and, if PREDICTION_LOG_PATH is set, in that file."""
    line = json.dumps(event, default=str)
    logger.info(line)
    path = os.getenv("PREDICTION_LOG_PATH")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    uri = os.getenv("MODEL_URI") or model_store.default_model_uri()
    try:
        loaded = model_store.load(uri)
        state["loaded"] = loaded
        m.MODEL_INFO.labels(version=loaded.version).set(1)
        logger.info(json.dumps({"event": "model_loaded", "uri": uri, "version": loaded.version}))
    except Exception as exc:  # keep serving /health so that Kubernetes can show the problem
        state["loaded"] = None
        logger.error(json.dumps({"event": "model_load_failed", "uri": uri, "error": str(exc)}))
    yield


app = FastAPI(
    title="Ride classification API",
    version="0.1.0",
    description="Send the IMU samples of one scooter ride, get the probability of each of the 5 ride categories.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path
        if not path.startswith("/metrics"):
            m.REQUESTS.labels(endpoint=path, status=str(status)).inc()
            m.LATENCY.labels(endpoint=path).observe(time.perf_counter() - start)


@app.get("/health", response_model=Health)
def health() -> Health:
    loaded = state["loaded"]
    return Health(
        status="ok" if loaded else "degraded",
        model_loaded=loaded is not None,
        model_version=loaded.version if loaded else "none",
        classes=CLASSES,
    )


@app.get("/ready")
def ready() -> dict:
    """Readiness probe: 200 only once a model is loaded (Kubernetes sends no traffic before that)."""
    if state["loaded"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/v1/predict", response_model=Prediction)
def predict(ride: Ride) -> Prediction:
    loaded = state["loaded"]
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    frame = pd.DataFrame([s.model_dump() for s in ride.samples], columns=CHANNELS).astype(float)
    if frame.isna().all().any():
        raise HTTPException(status_code=422, detail="A channel has no value at all in this ride")

    proba = np.asarray(loaded.model.predict(frame), dtype=float).ravel()
    if proba.shape != (len(CLASSES),):
        raise HTTPException(status_code=500, detail=f"Model returned {proba.shape}, expected ({len(CLASSES)},)")
    best = int(proba.argmax())
    summary = ride_summary(frame.to_numpy())
    ride_hash = ride_fingerprint(frame.to_numpy())
    save_ride(frame, ride_hash)

    m.INPUT_SAMPLES.observe(len(frame))
    m.INPUT_MISSING_RATIO.observe(float(frame.isna().to_numpy().mean()))
    for channel in m.MONITORED_CHANNELS:
        m.INPUT_CHANNEL_STD.labels(channel=channel).observe(summary[f"{channel}_std"])
    m.PREDICTED_CLASS.labels(predicted_class=CLASSES[best]).inc()
    m.PREDICTION_CONFIDENCE.observe(float(proba[best]))

    log_event(
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "prediction",
            "ride_id": ride.ride_id,
            "ride_hash": ride_hash,
            "n_samples": len(frame),
            **{k: round(v, 6) for k, v in summary.items()},
            "predicted_class": CLASSES[best],
            "probabilities": {c: round(float(p), 6) for c, p in zip(CLASSES, proba, strict=True)},
            "model_version": loaded.version,
            "inference_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    )
    return Prediction(
        ride_id=ride.ride_id,
        predicted_class=CLASSES[best],
        confidence=float(proba[best]),
        probabilities={c: float(p) for c, p in zip(CLASSES, proba, strict=True)},
        n_samples=len(frame),
        model_version=loaded.version,
        ride_hash=ride_hash,
    )


@app.post("/v1/feedback", status_code=204)
def feedback(fb: Feedback) -> None:
    unknown = {fb.predicted_class, fb.actual_class} - set(CLASSES)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown class(es) {sorted(unknown)}; expected {CLASSES}")
    m.FEEDBACK.labels(predicted_class=fb.predicted_class, actual_class=fb.actual_class).inc()
    log_event({"ts": datetime.now(UTC).isoformat(), "event": "feedback", **fb.model_dump()})

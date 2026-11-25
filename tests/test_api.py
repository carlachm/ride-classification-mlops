"""API tests: contract, safe failure, logging, metrics. The model is the registered placeholder."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import dummy_model, main
from fraud_detection.api.features import ride_fingerprint
from fraud_detection.api.schemas import CHANNELS, CLASSES, MIN_SAMPLES


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)


def make_ride(n: int = 300, seed: int = 0, ride_id: str | None = "r-1") -> dict:
    x = np.random.default_rng(seed).normal(size=(n, len(CHANNELS)))
    return {"ride_id": ride_id, "samples": [dict(zip(CHANNELS, map(float, row), strict=True)) for row in x]}


@pytest.fixture
def client(local_mlflow, tmp_path, monkeypatch):
    dummy_model.register()
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "logs" / "predictions.jsonl"))
    with TestClient(main.app) as c:
        yield c


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "1", "classes": CLASSES}


def test_predict_returns_one_probability_per_class_and_logs(client, tmp_path):
    r = client.post("/v1/predict", json=make_ride())
    assert r.status_code == 200
    body = r.json()
    assert list(body["probabilities"]) == CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["predicted_class"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["confidence"] == pytest.approx(body["probabilities"][body["predicted_class"]])
    assert body["n_samples"] == 300 and body["ride_id"] == "r-1"

    line = json.loads((tmp_path / "logs" / "predictions.jsonl").read_text().splitlines()[-1])
    assert line["event"] == "prediction" and line["model_version"] == "1"
    assert "ax_std" in line and line["predicted_class"] == body["predicted_class"]


def test_predict_is_deterministic(client):
    ride = make_ride()
    assert client.post("/v1/predict", json=ride).json() == client.post("/v1/predict", json=ride).json()


def test_null_sensor_values_are_tolerated(client):
    ride = make_ride()
    for sample in ride["samples"][:20]:
        sample["ax"] = None
    assert client.post("/v1/predict", json=ride).status_code == 200


def test_channel_entirely_null_is_rejected(client):
    ride = make_ride()
    for sample in ride["samples"]:
        sample["rz"] = None
    assert client.post("/v1/predict", json=ride).status_code == 422


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(samples=r["samples"][: MIN_SAMPLES - 1]),  # too short
        lambda r: r.update(samples=[]),
        lambda r: r["samples"][0].update(ax="abc"),  # wrong type
        lambda r: r["samples"][0].pop("rz"),  # missing channel
        lambda r: r.pop("samples"),
    ],
)
def test_invalid_input_is_rejected_with_422(client, mutate):
    ride = make_ride()
    mutate(ride)
    assert client.post("/v1/predict", json=ride).status_code == 422


def test_metrics_endpoint_counts_requests_and_predictions(client):
    client.post("/v1/predict", json=make_ride())
    text = client.get("/metrics/").text
    assert 'api_requests_total{endpoint="/v1/predict",status="200"}' in text
    assert "prediction_class_total{" in text
    assert 'input_channel_std_bucket{channel="ax"' in text
    assert "model_info{" in text


def test_feedback_updates_outcome_counter(client):
    r = client.post("/v1/feedback", json={"predicted_class": "Safe", "actual_class": "Reckless"})
    assert r.status_code == 204
    assert 'feedback_total{actual_class="Reckless",predicted_class="Safe"}' in client.get("/metrics/").text


def test_feedback_with_unknown_class_is_rejected(client):
    r = client.post("/v1/feedback", json={"predicted_class": "Safe", "actual_class": "Banana"})
    assert r.status_code == 422


def test_feedback_is_linked_to_the_scored_ride(client, tmp_path):
    ride_hash = client.post("/v1/predict", json=make_ride()).json()["ride_hash"]
    other = client.post("/v1/predict", json=make_ride(seed=1)).json()["ride_hash"]
    assert ride_hash != other and len(ride_hash) == 32

    feedback = {"predicted_class": "Safe", "actual_class": "Reckless", "ride_hash": ride_hash}
    assert client.post("/v1/feedback", json=feedback).status_code == 204
    lines = [json.loads(x) for x in (tmp_path / "logs" / "predictions.jsonl").read_text().splitlines()]
    assert lines[0]["ride_hash"] == ride_hash == lines[-1]["ride_hash"] and lines[-1]["event"] == "feedback"


@pytest.mark.parametrize("bad", ["../../etc/passwd", "ABC", "g" * 32, ""])
def test_feedback_with_a_malformed_ride_hash_is_rejected(client, bad):
    r = client.post("/v1/feedback", json={"predicted_class": "Safe", "actual_class": "Safe", "ride_hash": bad})
    assert r.status_code == 422


def test_raw_ride_is_stored_only_when_enabled(client, tmp_path, monkeypatch):
    monkeypatch.delenv("RIDE_STORE_DIR", raising=False)
    client.post("/v1/predict", json=make_ride())
    assert not (tmp_path / "rides").exists()

    monkeypatch.setenv("RIDE_STORE_DIR", str(tmp_path / "rides"))
    ride_hash = client.post("/v1/predict", json=make_ride()).json()["ride_hash"]
    stored = pd.read_parquet(tmp_path / "rides" / f"{ride_hash}.parquet")
    assert list(stored.columns) == CHANNELS and len(stored) == 300
    assert ride_fingerprint(stored.to_numpy()) == ride_hash  # same id after the float32 round trip


def test_a_broken_ride_store_does_not_break_predictions(client, tmp_path, monkeypatch):
    (tmp_path / "not-a-dir").write_text("x")
    monkeypatch.setenv("RIDE_STORE_DIR", str(tmp_path / "not-a-dir"))
    assert client.post("/v1/predict", json=make_ride()).status_code == 200


def test_every_feedback_combination_is_exported_from_the_start(client):
    text = client.get("/metrics/").text
    assert 'feedback_total{actual_class="Tandem",predicted_class="Bad weather"}' in text


def test_api_degrades_safely_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "does-not-exist"))
    with TestClient(main.app) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        assert c.post("/v1/predict", json=make_ride()).status_code == 503
        assert c.get("/ready").status_code == 503  # Kubernetes must not route traffic to this pod


def test_ready_once_model_is_loaded(client):
    assert client.get("/ready").status_code == 200

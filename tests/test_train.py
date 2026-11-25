"""Integration test: full training run against a throw-away local MLflow store."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from fraud_detection.api.schemas import CHANNELS, CLASSES
from fraud_detection.data import prepare
from fraud_detection.train import class_weights, run


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)


def test_class_weights_favour_rare_classes():
    y = np.array([0] * 80 + [1] * 10 + [2] * 5 + [3] * 4 + [4] * 1)
    w = class_weights(y, "balanced")
    assert w[4] > w[1] > w[0]
    assert np.isclose(float(w.mean()), 1.0)
    assert (class_weights(y, "none") == 1).all()


def test_training_run_logs_and_registers_ride_classifier(params, local_mlflow):
    prepare(params)
    params["quality_gate"] = {"min_test_macro_f1": 0.0}
    summary = run(params)

    assert json.loads(Path(params["metrics_path"]).read_text())["run_id"] == summary["run_id"]
    assert summary["promoted"] is True  # first model always becomes champion

    client = MlflowClient()
    name = params["mlflow"]["registered_model_name"]
    assert name == "ride-classifier"  # name loaded by the API (docs/api-contract.md)
    champion = client.get_model_version_by_alias(name, params["mlflow"]["champion_alias"])
    assert str(champion.version) == summary["model_version"]

    run_data = client.get_run(summary["run_id"]).data
    for metric in ("test_macro_f1", "test_recall_reckless", "baseline_majority_macro_f1", "best_val_macro_f1"):
        assert metric in run_data.metrics

    # Exactly what the API does: load with pyfunc, send one ride, get (1, 5) probabilities
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    ride = pd.DataFrame(np.random.default_rng(0).normal(size=(450, 6)), columns=CHANNELS)
    ride.iloc[5:15, 0] = np.nan
    proba = np.asarray(model.predict(ride))
    assert proba.shape == (1, len(CLASSES))
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)


def test_quality_gate_blocks_bad_model(params, local_mlflow):
    prepare(params)
    params["quality_gate"] = {"min_test_macro_f1": 1.01}  # impossible bar
    summary = run(params)
    assert summary["quality_gate_passed"] is False
    assert summary["promoted"] is False
    version = MlflowClient().get_model_version(params["mlflow"]["registered_model_name"], summary["model_version"])
    assert version.tags["promotion"] == "gate_failed"

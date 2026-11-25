"""Drift detection and traffic simulator, on synthetic features (no real data, no model)."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.api.schemas import CHANNELS, CLASSES
from fraud_detection.config import load_params
from fraud_detection.data import prepare
from fraud_detection.monitoring import drift, simulate
from fraud_detection.monitoring.logs import read_events, window

CFG = load_params()["drift"]
# Typical per-ride values measured on the GBG500 training rides
MEANS = {"ax": -0.12, "ay": -9.14, "az": -2.27, "rx": -0.55, "ry": -0.09, "rz": 0.05}
STDS = {"ax": 0.76, "ay": 1.06, "az": 0.92, "rx": 5.5, "ry": 21.1, "rz": 8.3}


def features(n: int, seed: int, std_factor: float = 1.0, accel_divisor: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for c in CHANNELS:
        divisor = accel_divisor if c.startswith("a") else 1.0
        data[f"{c}_mean"] = rng.normal(MEANS[c], 0.2, n) / divisor
        data[f"{c}_std"] = rng.lognormal(np.log(STDS[c]), 0.25, n) * std_factor / divisor
    data["n_samples"] = rng.normal(3000, 800, n).round()
    return pd.DataFrame(data)


def reference(n: int = 400) -> pd.DataFrame:
    return features(n, seed=1).assign(part="train")


def test_psi_is_zero_on_identical_data_and_large_after_a_shift():
    x = np.random.default_rng(0).normal(size=1000)
    assert drift.psi(x, x, bins=5) == pytest.approx(0.0)
    assert drift.psi(x, x + 2, bins=5) > 1


def test_same_population_is_not_drift():
    result = drift.compare(reference(), features(200, seed=2), CFG)
    assert result["verdict"] == "ok" and not result["problems"] and not result["drifted_features"]


def test_rougher_rides_are_drift():
    result = drift.compare(reference(), features(200, seed=2, std_factor=1.4), CFG)
    assert result["verdict"] == "drift" and "ay_std" in result["drifted_features"]


def test_one_shifted_feature_is_only_an_alert():
    current = features(200, seed=2)
    current["rz_mean"] += 0.5
    result = drift.compare(reference(), current, CFG)
    assert result["verdict"] == "alert" and result["drifted_features"] == ["rz_mean"]


def test_accelerometer_in_g_is_an_upstream_bug_not_drift():
    result = drift.compare(reference(), features(200, seed=2, accel_divisor=9.81), CFG)
    assert result["verdict"] == "upstream_bug_suspected"
    assert any("gravity" in p for p in result["problems"])
    assert drift.EXIT_CODES[result["verdict"]] != drift.EXIT_CODES["drift"]


def test_dead_sensor_is_reported_even_with_few_rides():
    current = features(10, seed=2)
    current["ry_std"] = 0.0
    result = drift.compare(reference(), current, CFG)
    assert result["verdict"] == "upstream_bug_suspected" and "ry" in result["problems"][0]


def test_no_verdict_below_the_minimum_number_of_rides():
    result = drift.compare(reference(), features(CFG["min_rides"] - 1, seed=2, std_factor=1.4), CFG)
    assert result["verdict"] == "insufficient_data" and result["max_psi"] is None


def test_prediction_distribution_is_reported():
    ref = pd.concat([reference(), features(150, seed=3).assign(part="heldout", predicted_class="Safe")])
    current = features(200, seed=2).assign(predicted_class=["Safe", "Reckless"] * 100)
    result = drift.compare(ref, current, CFG)
    assert result["prediction"]["pvalue"] < 0.01 and result["verdict"] == "ok"  # reported, never triggers


def test_reference_is_built_from_the_prepared_rides(params):
    prepare(params)
    ref = drift.build_reference(params)
    assert set(drift.FEATURES) <= set(ref.columns) and (ref["part"] == "train").all()


def test_log_window_and_unreadable_lines(tmp_path):
    lines = [{"event": "prediction", "n": 1}, {"event": "feedback", "n": 1}]
    lines += [{"event": "prediction", "n": 2}, {"event": "feedback", "n": 2}]
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(map(json.dumps, lines)) + "\nnot json\n")
    events = read_events(path)
    assert len(events) == 4
    predictions, feedbacks = window(events, last=1)
    assert [p["n"] for p in predictions] == [2] and [f["n"] for f in feedbacks] == [2]
    assert tuple(map(len, window(events))) == (2, 2)
    assert window(events, last=0) == ([], [])


def test_scenarios_change_what_they_claim_to_change():
    ride = np.random.default_rng(0).normal(loc=list(MEANS.values()), scale=list(STDS.values()), size=(500, 6))
    rough = simulate.apply_scenario(ride, "data-drift", vibration=1.4)
    np.testing.assert_allclose(rough.mean(axis=0), ride.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(rough.std(axis=0), 1.4 * ride.std(axis=0))
    in_g = simulate.apply_scenario(ride, "unit-bug")
    np.testing.assert_allclose(in_g[:, :3] * 9.81, ride[:, :3])
    np.testing.assert_array_equal(in_g[:, 3:], ride[:, 3:])
    np.testing.assert_array_equal(simulate.apply_scenario(ride, "concept-drift"), ride)  # inputs unchanged


def test_concept_drift_only_changes_the_labels():
    rng = np.random.default_rng(0)
    labels = [simulate.feedback_label("Safe", "concept-drift", rng) for _ in range(300)]
    assert 0.6 < labels.count("Reckless") / 300 < 0.8
    assert simulate.feedback_label("Safe", "normal", rng) == "Safe"
    assert simulate.feedback_label("Tandem", "concept-drift", rng) == "Tandem"
    assert set(labels) <= set(CLASSES)

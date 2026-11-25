"""Retraining: collection of the labelled rides, leak guard, new data in the train split only, trigger decision."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.api.features import ride_fingerprint
from fraud_detection.api.ride_store import save_ride
from fraud_detection.api.schemas import CHANNELS
from fraud_detection.config import load_params
from fraud_detection.data import load_split, prepare, read_feedback, synthetic_dataset
from fraud_detection.retraining import collect, trigger

CFG = load_params()["retraining"]
PREFIXES = tuple(CFG["excluded_ride_id_prefixes"])


def ride(seed: int, n: int = 150) -> pd.DataFrame:
    x = np.random.default_rng(seed).normal(size=(n, len(CHANNELS)))
    x[3, 0] = np.nan
    return pd.DataFrame(x, columns=CHANNELS)


def scored(frame: pd.DataFrame, store, ride_id: str = "ride-1", label: str | None = "Tandem") -> list[dict]:
    """What the API leaves behind for one ride: the stored signal and the log events."""
    ride_hash = ride_fingerprint(frame.to_numpy())
    save_ride(frame, ride_hash)
    events = [{"event": "prediction", "ride_id": ride_id, "ride_hash": ride_hash}]
    if label:
        events.append({"event": "feedback", "predicted_class": "Safe", "actual_class": label, "ride_hash": ride_hash})
    return events


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "store"
    monkeypatch.setenv("RIDE_STORE_DIR", str(path))
    return path


def test_fingerprint_survives_precision_changes_and_detects_changes():
    x = ride(0).to_numpy()
    assert (
        ride_fingerprint(x)
        == ride_fingerprint(x.astype("float32"))
        == ride_fingerprint(x.astype("float32").astype(float))
    )
    y = x.copy()
    y[10, 2] += 0.01
    assert ride_fingerprint(y) != ride_fingerprint(x)


def test_only_usable_labelled_rides_are_collected(store):
    good, golden, known, simulated, unlabelled, lost = (ride(s) for s in range(6))
    events = (
        scored(good, store)
        + scored(golden, store)
        + scored(known, store)
        + scored(simulated, store, ride_id="sim-1-abc")
        + scored(unlabelled, store, label=None)
        + scored(lost, store)
        + [{"event": "feedback", "predicted_class": "Safe", "actual_class": "Safe", "ride_hash": None}]
        + [{"event": "feedback", "predicted_class": "Safe", "actual_class": "Safe", "ride_hash": "0" * 32}]
    )
    (store / f"{ride_fingerprint(lost.to_numpy())}.parquet").unlink()

    accepted, report = collect.collect(
        events,
        store,
        golden={ride_fingerprint(golden.to_numpy())},
        known={ride_fingerprint(known.to_numpy())},
        excluded_prefixes=PREFIXES,
    )
    assert list(accepted) == [ride_fingerprint(good.to_numpy())]
    assert report == {
        "accepted": 1,
        "in_validation_or_test_set": 1,
        "already_in_training_set": 1,
        "test_traffic": 1,
        "raw_signal_not_stored": 1,
        "feedback_without_ride_hash": 1,
        "feedback_for_an_unknown_ride": 1,
    }


def test_a_ride_also_sent_as_test_traffic_stays_excluded(store):
    frame = ride(0)
    events = scored(frame, store, ride_id="sim-1-x", label=None) + scored(frame, store, ride_id="real-ride")
    accepted, report = collect.collect(events, store, set(), set(), PREFIXES)
    assert not accepted and report["test_traffic"] == 1


def test_dataset_accumulates_and_the_latest_label_wins(store, tmp_path):
    out = tmp_path / "feedback"
    first, _ = collect.collect(scored(ride(0), store, label="Tandem"), store, set(), set(), PREFIXES)
    assert collect.save_dataset(first, out) == 1
    second_events = scored(ride(0), store, label="Reckless") + scored(ride(1), store, label="Safe")
    second, _ = collect.collect(second_events, store, set(), set(), PREFIXES)
    assert collect.save_dataset(second, out) == 2

    samples, labels = read_feedback(out)
    assert samples["ride_id"].nunique() == 2 and samples["ride_id"].str.startswith("fb-").all()
    by_id = dict(zip(labels["RIDE_ID"], labels["LABEL"], strict=True))
    assert by_id[f"fb-{ride_fingerprint(ride(0).to_numpy())}"] == "Reckless"


def test_production_rides_join_the_train_split_only(params, store, tmp_path):
    before = prepare(params)
    frozen = {split: load_split(split, params)[2] for split in ("val", "test")}
    test_ride = pd.DataFrame(load_split("test", params)[0][0], columns=CHANNELS)

    # two new rides, plus a copy of a test ride that must never reach the training data
    new_samples, _ = synthetic_dataset(n_rides=25, min_len=400, max_len=500, seed=99)
    new = [g[CHANNELS] for _, g in new_samples.groupby("ride_id")][:2]
    accepted = {ride_fingerprint(f.to_numpy()): (f, "Reckless") for f in [*new, test_ride]}
    collect.save_dataset(accepted, tmp_path / "feedback")  # bypasses the guard of `collect` on purpose

    after = prepare(params)
    assert after["n_feedback_rides"] == 2 and after["n_feedback_dropped"] == 1
    assert after["n_rides_train"] == before["n_rides_train"] + 2
    assert after["class_counts_train"]["Reckless"] == before["class_counts_train"]["Reckless"] + 2
    assert {split: load_split(split, params)[2] for split in ("val", "test")} == frozen
    assert sum(i.startswith("fb-") for i in load_split("train", params)[2]) == 2
    # and the leak guard of the collector ignores these production rides when run again
    assert collect.split_fingerprints(params, ("train",)).isdisjoint(accepted)


def drift_result(verdict: str, n: int = 200) -> dict:
    return {"verdict": verdict, "n_current": n, "problems": ["gravity"], "drifted_features": ["ay_std"], "max_psi": 1.2}


@pytest.mark.parametrize(
    ("verdict", "n_feedback", "accuracy", "new_rides", "expected"),
    [
        ("ok", 100, 0.55, 50, "no_action"),
        ("alert", 100, 0.55, 50, "no_action"),
        ("insufficient_data", 10, 0.9, 50, "wait"),
        ("drift", 100, 0.55, 50, "retrain"),  # input drift
        ("ok", 100, 0.20, 50, "retrain"),  # concept drift: only the labels see it
        ("insufficient_data", 100, 0.20, 50, "retrain"),
        ("ok", CFG["min_feedback"] - 1, 0.0, 50, "no_action"),  # too few labels to judge
        ("drift", 100, 0.55, 0, "blocked_no_new_data"),
        ("upstream_bug_suspected", 100, 0.10, 500, "investigate"),  # never retrain on a bug
    ],
)
def test_trigger_decision(verdict, n_feedback, accuracy, new_rides, expected):
    decision = trigger.decide(drift_result(verdict), {"n": n_feedback, "accuracy": accuracy}, new_rides, CFG)
    assert decision.action == expected and decision.reason


def test_forcing_without_new_data_never_overrides_an_upstream_bug():
    performance = {"n": 100, "accuracy": 0.5}
    assert trigger.decide(drift_result("drift"), performance, 0, CFG, allow_no_new_data=True).action == "retrain"
    bug = trigger.decide(drift_result("upstream_bug_suspected"), performance, 0, CFG, allow_no_new_data=True)
    assert bug.action == "investigate"


def test_feedback_performance():
    feedbacks = [{"predicted_class": "Safe", "actual_class": "Safe"}] * 3
    feedbacks += [{"predicted_class": "Safe", "actual_class": "Tandem"}]
    assert trigger.feedback_performance(feedbacks) == {"n": 4, "accuracy": 0.75}
    assert trigger.feedback_performance([]) == {"n": 0, "accuracy": None}


def test_training_records_the_production_rides_and_the_reason(params, tmp_path, store, monkeypatch):
    import mlflow
    from mlflow import MlflowClient

    from fraud_detection.train import run

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    try:
        prepare(params)
        client = MlflowClient()
        # an empty registry (e.g. MLFLOW_TRACKING_URI forgotten) is "unknown", not "0 rides seen"
        assert trigger.rides_seen_by_last_training(client, params["mlflow"]["registered_model_name"]) is None

        new_samples, _ = synthetic_dataset(n_rides=25, min_len=400, max_len=500, seed=99)
        new = [g[CHANNELS] for _, g in new_samples.groupby("ride_id")][:3]
        collect.save_dataset({ride_fingerprint(f.to_numpy()): (f, "Safe") for f in new}, tmp_path / "feedback")
        prepare(params)
        params["train"]["epochs"] = 1
        params["quality_gate"] = {"min_test_macro_f1": 0.0}
        monkeypatch.setenv("RETRAIN_REASON", "drift on ay_std")
        summary = run(params)

        assert trigger.rides_seen_by_last_training(client, params["mlflow"]["registered_model_name"]) == 3
        assert client.get_run(summary["run_id"]).data.tags["retrain_reason"] == "drift on ay_std"
    finally:
        mlflow.set_tracking_uri(None)

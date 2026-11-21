import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import FEATURE_COLUMNS, LABELS
from fraud_detection.data import (
    DataValidationError,
    clean,
    load_raw,
    load_split,
    prepare,
    split_rides,
    validate,
)


def _labels(fake_data):
    return fake_data[1].rename(columns={"RIDE_ID": "ride_id", "LABEL": "label"})


def test_synthetic_data_has_gbg500_schema(fake_data):
    samples, labels = fake_data
    assert list(samples.columns) == ["ride_id", "time_index", *FEATURE_COLUMNS]
    assert list(labels.columns) == ["RIDE_ID", "LABEL"]
    assert set(labels["LABEL"]) == set(LABELS)


def test_validate_accepts_valid_data(fake_data):
    validate(fake_data[0], _labels(fake_data))


def test_validate_rejects_missing_column(fake_data):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(fake_data[0].drop(columns=["rz"]), _labels(fake_data))


def test_validate_rejects_unknown_label(fake_data):
    labels = _labels(fake_data)
    labels.loc[0, "label"] = "Drunk"
    with pytest.raises(DataValidationError, match="Unknown labels"):
        validate(fake_data[0], labels)


def test_validate_rejects_unlabelled_ride(fake_data):
    labels = _labels(fake_data).iloc[1:]
    with pytest.raises(DataValidationError, match="no label"):
        validate(fake_data[0], labels)


def test_validate_rejects_fully_missing_channel(fake_data):
    samples = fake_data[0].copy()
    first = samples["ride_id"] == samples["ride_id"].iloc[0]
    samples.loc[first, "ax"] = np.nan
    with pytest.raises(DataValidationError, match="no value at all"):
        validate(samples, _labels(fake_data))


def test_clean_sorts_and_removes_duplicated_timestamps(fake_data):
    samples = fake_data[0]
    messy = samples.iloc[::-1]
    messy = pd.concat([messy, messy.iloc[[0]]])  # one duplicated timestamp
    out = clean(messy)
    assert len(out) == len(samples)
    assert (out.groupby("ride_id")["time_index"].diff().dropna() > 0).all()


def test_split_is_by_ride_stratified_and_reproducible(fake_data):
    labels = _labels(fake_data)
    a = split_rides(labels, 0.2, 0.2, seed=1)
    b = split_rides(labels, 0.2, 0.2, seed=1)
    assert a == b
    assert set(a["train"]).isdisjoint(a["test"]) and set(a["val"]).isdisjoint(a["test"])
    assert sum(len(v) for v in a.values()) == len(labels)


def test_prepare_writes_splits_and_stats(params):
    stats = prepare(params)
    out = Path(params["data"]["processed_dir"])
    for name in ("train", "val", "test"):
        assert (out / f"{name}.parquet").exists()
    assert json.loads((out / "data_stats.json").read_text()) == stats
    assert stats["n_rides_train"] + stats["n_rides_val"] + stats["n_rides_test"] == stats["n_rides"]


def test_load_split_returns_rides(params):
    prepare(params)
    signals, y, ids = load_split("train", params)
    assert len(signals) == len(y) == len(ids)
    assert all(s.shape[1] == len(FEATURE_COLUMNS) for s in signals)
    assert y.min() >= 0 and y.max() < len(LABELS)


def test_load_raw_renames_label_columns(params):
    _, labels = load_raw(params)
    assert {"ride_id", "label"} <= set(labels.columns)

"""Shared fixtures: small synthetic rides with the exact GBG500 schema (no real data needed)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from fraud_detection.config import load_params
from fraud_detection.data import synthetic_dataset


@pytest.fixture
def fake_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(samples, labels) shaped like GBG500.parquet / GBG500_labels.csv."""
    return synthetic_dataset(n_rides=100, min_len=400, max_len=600)


@pytest.fixture
def params(tmp_path, fake_data):
    """Project params pointing to temporary raw files / processed dir, with a fast training config."""
    samples, labels = fake_data
    raw = tmp_path / "raw"
    raw.mkdir()
    samples.to_parquet(raw / "GBG500.parquet", index=False)
    labels.to_csv(raw / "GBG500_labels.csv", index=False)

    p = copy.deepcopy(load_params())
    p["data"]["raw_path"] = str(raw / "GBG500.parquet")
    p["data"]["labels_path"] = str(raw / "GBG500_labels.csv")
    p["data"]["processed_dir"] = str(tmp_path / "processed")
    p["train"]["epochs"] = 15
    p["metrics_path"] = str(tmp_path / "metrics.json")
    return p

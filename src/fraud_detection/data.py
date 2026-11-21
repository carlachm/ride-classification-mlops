"""Data loading, validation and preparation for the GBG500 ride dataset.

The raw data (from Inside) is two files, versioned with DVC:
    data/raw/GBG500.parquet      one row per IMU sample: ride_id, time_index (ms), ax, ay, az, rx, ry, rz
    data/raw/GBG500_labels.csv   one row per ride: RIDE_ID, LABEL (5 classes)

Usage:
    python -m fraud_detection.data prepare    # raw -> train/val/test parquet (split by ride)
    python -m fraud_detection.data sample     # synthetic rides (CI, smoke tests, demos)
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    FEATURE_COLUMNS,
    LABELS,
    PROJECT_ROOT,
    RIDE_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    load_params,
)

logger = logging.getLogger(__name__)

SAMPLING_STEP_MS = 100  # 10 Hz


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the IMU samples and the ride labels."""
    cfg = params["data"]
    samples = pd.read_parquet(PROJECT_ROOT / cfg["raw_path"])
    labels = pd.read_csv(PROJECT_ROOT / cfg["labels_path"])
    labels = labels.rename(columns={"RIDE_ID": RIDE_COLUMN, "LABEL": TARGET_COLUMN})
    return samples, labels


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def validate(samples: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Data quality checks: schema, label values, one label per ride, usable signals."""
    expected = [RIDE_COLUMN, TIME_COLUMN, *FEATURE_COLUMNS]
    missing = set(expected) - set(samples.columns)
    if missing:
        raise DataValidationError(f"Missing columns in samples: {sorted(missing)}")
    if not {RIDE_COLUMN, TARGET_COLUMN} <= set(labels.columns):
        raise DataValidationError("Labels must have RIDE_ID and LABEL columns")

    unknown = set(labels[TARGET_COLUMN]) - set(LABELS)
    if unknown:
        raise DataValidationError(f"Unknown labels: {sorted(unknown)} (expected {LABELS})")
    if labels[RIDE_COLUMN].duplicated().any():
        raise DataValidationError("A ride has several labels")
    unlabelled = set(samples[RIDE_COLUMN]) - set(labels[RIDE_COLUMN])
    if unlabelled:
        raise DataValidationError(f"{len(unlabelled)} rides have no label")

    # A ride where one sensor channel is entirely missing cannot be used
    all_nan = samples.groupby(RIDE_COLUMN)[FEATURE_COLUMNS].apply(lambda d: d.isna().all().any())
    if all_nan.any():
        raise DataValidationError(f"{int(all_nan.sum())} rides have a channel with no value at all")
    if np.isinf(samples[FEATURE_COLUMNS].to_numpy(dtype=float)).any():
        raise DataValidationError("Samples contain infinite values")


# --------------------------------------------------------------------------- #
# Cleaning + preparation
# --------------------------------------------------------------------------- #
def clean(samples: pd.DataFrame) -> pd.DataFrame:
    """Time order inside each ride, duplicated timestamps removed, float32.

    Missing values (sensor dropouts, ~0.7 % of GBG500) are KEPT: they are filled inside the
    model wrapper, exactly as at serving time, so training and inference see the same thing.
    """
    out = samples.sort_values([RIDE_COLUMN, TIME_COLUMN])
    out = out.drop_duplicates(subset=[RIDE_COLUMN, TIME_COLUMN], keep="first")
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype("float32")
    return out.reset_index(drop=True)


def split_rides(labels: pd.DataFrame, test_size: float, val_size: float, seed: int) -> dict[str, list[str]]:
    """Train / validation / test split of the RIDES, stratified on the class.

    Splitting by ride (not by sample or window) prevents leakage: windows of the same
    ride never end up on both sides.
    """
    train_val, test = train_test_split(labels, test_size=test_size, stratify=labels[TARGET_COLUMN], random_state=seed)
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[TARGET_COLUMN],
        random_state=seed,
    )
    return {name: sorted(part[RIDE_COLUMN]) for name, part in {"train": train, "val": val, "test": test}.items()}


def prepare(params: dict) -> dict:
    """Load, validate, clean and split the data; save one parquet per split."""
    out_dir = PROJECT_ROOT / params["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, labels = load_raw(params)
    validate(samples, labels)
    n_before = len(samples)
    samples = clean(samples)
    data = samples.merge(labels[[RIDE_COLUMN, TARGET_COLUMN]], on=RIDE_COLUMN, how="inner")

    splits = split_rides(labels, params["data"]["test_size"], params["data"]["val_size"], params["seed"])
    stats: dict = {
        "n_rides": int(labels[RIDE_COLUMN].nunique()),
        "n_samples": n_before,
        "n_duplicates_removed": n_before - len(samples),
        "missing_ratio": round(float(samples[FEATURE_COLUMNS].isna().to_numpy().mean()), 6),
        "class_counts": labels[TARGET_COLUMN].value_counts().reindex(LABELS, fill_value=0).to_dict(),
    }
    for name, ride_ids in splits.items():
        part = data[data[RIDE_COLUMN].isin(ride_ids)]
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        per_ride = part.drop_duplicates(RIDE_COLUMN)[TARGET_COLUMN]
        stats[f"n_rides_{name}"] = len(ride_ids)
        stats[f"class_counts_{name}"] = per_ride.value_counts().reindex(LABELS, fill_value=0).to_dict()

    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Prepared data: %s", json.dumps(stats))
    return stats


def load_split(name: str, params: dict) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Load one split as a list of rides.

    Returns (signals, labels, ride_ids): ``signals[i]`` has shape (n_samples_i, 6),
    ``labels[i]`` is the class index in ``LABELS``.
    """
    df = pd.read_parquet(PROJECT_ROOT / params["data"]["processed_dir"] / f"{name}.parquet")
    signals, y, ids = [], [], []
    for ride_id, ride in df.groupby(RIDE_COLUMN, sort=True):
        signals.append(ride[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        y.append(LABELS.index(ride[TARGET_COLUMN].iloc[0]))
        ids.append(str(ride_id))
    return signals, np.asarray(y, dtype=np.int64), ids


# --------------------------------------------------------------------------- #
# Synthetic data (tests, CI smoke test, demos without the real files)
# --------------------------------------------------------------------------- #
def synthetic_dataset(
    n_rides: int = 60, min_len: int = 400, max_len: int = 900, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fake rides with the exact GBG500 schema; each class has its own vibration level.

    Returns (samples, labels) shaped like GBG500.parquet / GBG500_labels.csv.
    """
    rng = np.random.default_rng(seed)
    # every class is represented, "Safe" is the majority like in the real data
    classes = [LABELS[i % len(LABELS)] for i in range(len(LABELS) * 4)]
    classes += ["Safe"] * (n_rides - len(classes))
    level = {label: 1.0 + 0.8 * i for i, label in enumerate(LABELS)}
    gravity = np.array([0.0, -9.6, -1.5, 0.0, 0.0, 0.0])

    frames, labels = [], []
    for i, label in enumerate(classes[:n_rides]):
        ride_id = f"synthetic-{i:04d}"
        n = int(rng.integers(min_len, max_len))
        x = gravity + rng.normal(scale=0.5 * level[label], size=(n, len(FEATURE_COLUMNS)))
        x[rng.random(x.shape) < 0.005] = np.nan  # sensor dropouts, like the real data
        ride = pd.DataFrame(x, columns=FEATURE_COLUMNS)
        ride.insert(0, TIME_COLUMN, np.arange(n) * SAMPLING_STEP_MS)
        ride.insert(0, RIDE_COLUMN, ride_id)
        frames.append(ride)
        labels.append({"RIDE_ID": ride_id, "LABEL": label})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(labels)


def write_synthetic(params: dict, n_rides: int) -> None:
    samples, labels = synthetic_dataset(n_rides)
    raw = PROJECT_ROOT / params["data"]["raw_path"]
    raw.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(raw, index=False)
    labels.to_csv(PROJECT_ROOT / params["data"]["labels_path"], index=False)
    logger.info("Synthetic dataset (%d rides) written to %s", n_rides, raw.parent)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "sample"])
    parser.add_argument("--rides", type=int, default=60, help="number of rides for `sample`")
    args = parser.parse_args()

    params = load_params()
    if args.step == "sample":
        write_synthetic(params, args.rides)
    else:
        prepare(params)


if __name__ == "__main__":
    main()

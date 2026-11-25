"""Build the dataset of labelled production rides used to retrain.

    python -m fraud_detection.retraining.collect

prediction log (predictions + feedback) + raw signals (RIDE_STORE_DIR) -> data/feedback/{rides.parquet, labels.csv}
The dataset accumulates from one call to the next; the latest label of a ride wins. Then version it
(docs/retraining.md): ``dvc add data/feedback && dvc push`` and commit ``data/feedback.dvc``.

A ride is rejected when it is test traffic (simulator, web app demo, load test...), when it is already in
the validation or test sets (it would leak into the training), already a GBG500 training ride, or when
its raw signal was not stored.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from fraud_detection.api.features import ride_fingerprint
from fraud_detection.api.schemas import CLASSES
from fraud_detection.config import FEATURE_COLUMNS, PROJECT_ROOT, RIDE_COLUMN, TIME_COLUMN, load_params
from fraud_detection.data import (
    FEEDBACK_LABELS_FILE,
    FEEDBACK_PREFIX,
    FEEDBACK_RIDES_FILE,
    SAMPLING_STEP_MS,
    load_split,
    read_feedback,
)
from fraud_detection.monitoring.logs import read_events

logger = logging.getLogger(__name__)

_HASH = re.compile(r"[0-9a-f]{32}")


def split_fingerprints(params: dict, splits: tuple[str, ...]) -> set[str]:
    """Fingerprints of the GBG500 rides of ``splits`` (production rides added to the train split are ignored)."""
    out = set()
    for split in splits:
        rides, _, ids = load_split(split, params)
        out |= {
            ride_fingerprint(r)
            for r, ride_id in zip(rides, ids, strict=True)
            if not ride_id.startswith(FEEDBACK_PREFIX)
        }
    return out


def collect(
    events: list[dict], store_dir: Path, golden: set[str], known: set[str], excluded_prefixes: tuple[str, ...]
) -> tuple[dict[str, tuple[pd.DataFrame, str]], Counter]:
    """{ride_hash: (signal, label)} of the usable labelled rides, and why the others were rejected."""
    ride_ids: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    report: Counter = Counter()
    for event in events:
        ride_hash = event.get("ride_hash")
        if event.get("event") == "prediction" and ride_hash:
            ride_ids.setdefault(ride_hash, set()).add(str(event.get("ride_id")))
        elif event.get("event") == "feedback":
            if ride_hash and event.get("actual_class") in CLASSES:
                labels[ride_hash] = event["actual_class"]
            else:
                report["feedback_without_ride_hash"] += 1

    accepted = {}
    for ride_hash, label in labels.items():
        path = store_dir / f"{ride_hash}.parquet"
        if not _HASH.fullmatch(ride_hash) or ride_hash not in ride_ids:
            report["feedback_for_an_unknown_ride"] += 1
        elif any(ride_id.startswith(excluded_prefixes) for ride_id in ride_ids[ride_hash]):
            report["test_traffic"] += 1
        elif ride_hash in golden:
            report["in_validation_or_test_set"] += 1
        elif ride_hash in known:
            report["already_in_training_set"] += 1
        elif not path.exists():
            report["raw_signal_not_stored"] += 1
        else:
            frame = pd.read_parquet(path)[FEATURE_COLUMNS]
            if ride_fingerprint(frame.to_numpy()) != ride_hash:
                report["stored_signal_corrupted"] += 1
            else:
                accepted[ride_hash] = (frame, label)
    report["accepted"] = len(accepted)
    return accepted, report


def save_dataset(accepted: dict[str, tuple[pd.DataFrame, str]], directory: Path) -> int:
    """Merge ``accepted`` into the dataset (same schema as the raw GBG500 files). Returns its number of rides."""
    signals: dict[str, pd.DataFrame] = {}
    labels: dict[str, str] = {}
    existing = read_feedback(directory)
    if existing:
        signals = {str(k): v for k, v in existing[0].groupby(RIDE_COLUMN, sort=True)}
        labels = dict(zip(existing[1]["RIDE_ID"], existing[1]["LABEL"], strict=True))
    for ride_hash, (frame, label) in accepted.items():
        ride_id = f"{FEEDBACK_PREFIX}{ride_hash}"
        ride = frame.astype("float32").reset_index(drop=True)
        ride.insert(0, TIME_COLUMN, range(0, len(ride) * SAMPLING_STEP_MS, SAMPLING_STEP_MS))
        ride.insert(0, RIDE_COLUMN, ride_id)
        signals[ride_id], labels[ride_id] = ride, label
    if not labels:
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat([signals[k] for k in sorted(signals)], ignore_index=True).to_parquet(
        directory / FEEDBACK_RIDES_FILE, index=False
    )
    pd.DataFrame({"RIDE_ID": sorted(labels), "LABEL": [labels[k] for k in sorted(labels)]}).to_csv(
        directory / FEEDBACK_LABELS_FILE, index=False
    )
    return len(labels)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    cfg = params["retraining"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "logs" / "predictions.jsonl")
    parser.add_argument("--store", type=Path, default=PROJECT_ROOT / "logs" / "rides")
    args = parser.parse_args()

    # The processed splits are required: without them the leak guard cannot work, so fail loudly
    golden = split_fingerprints(params, ("val", "test"))
    known = split_fingerprints(params, ("train",))
    accepted, report = collect(
        read_events(args.log), args.store, golden, known, tuple(cfg["excluded_ride_id_prefixes"])
    )
    directory = PROJECT_ROOT / params["data"]["feedback_dir"]
    total = save_dataset(accepted, directory)
    for reason, count in sorted(report.items()):
        print(f"{reason:30s} {count}")
    print(f"{total} labelled production rides in {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

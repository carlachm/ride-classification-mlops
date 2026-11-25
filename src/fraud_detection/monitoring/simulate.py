"""Replay "production" traffic against the API, optionally broken in a known way.

    python -m fraud_detection.monitoring.simulate --n 200                            # normal traffic
    python -m fraud_detection.monitoring.simulate --n 200 --scenario data-drift      # rougher rides
    python -m fraud_detection.monitoring.simulate --n 200 --scenario concept-drift   # same rides, other labels
    python -m fraud_detection.monitoring.simulate --n 200 --scenario unit-bug        # accelerometer sent in g

The rides come from the validation and test splits (never used for training), drawn with replacement.
Their ride_id starts with "sim-": the retraining never uses simulated rides.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fraud_detection.api.schemas import CHANNELS, CLASSES
from fraud_detection.config import load_params
from fraud_detection.data import load_split

SCENARIOS = ("normal", "data-drift", "concept-drift", "unit-bug")
GRAVITY = 9.81


def apply_scenario(x: np.ndarray, scenario: str, vibration: float = 1.4) -> np.ndarray:
    """Modified copy of one ride (n_samples, 6)."""
    out = np.array(x, dtype=float)
    if scenario == "data-drift":
        # Rougher road / other scooter: more vibration around the same average position
        mean = np.nanmean(out, axis=0)
        out = mean + (out - mean) * vibration
    elif scenario == "unit-bug":
        out[:, :3] /= GRAVITY  # the sender switched the accelerometer from m/s² to g
    return out


def feedback_label(true_label: str, scenario: str, rng: np.random.Generator, share: float = 0.7) -> str:
    """Label sent back by the reviewers. In concept drift, rides that used to be "Safe" are now judged reckless."""
    if scenario == "concept-drift" and true_label == "Safe" and rng.random() < share:
        return "Reckless"
    return true_label


def to_samples(x: np.ndarray) -> list[dict]:
    return [{c: None if np.isnan(v) else float(v) for c, v in zip(CHANNELS, row, strict=True)} for row in x]


def load_pool(params: dict) -> list[tuple[str, np.ndarray, str]]:
    pool = []
    for split in ("val", "test"):
        rides, y, ids = load_split(split, params)
        pool += [(ride_id, ride, CLASSES[label]) for ride, label, ride_id in zip(rides, y, ids, strict=True)]
    return pool


def simulate(client, pool: list, n: int, scenario: str, feedback_rate: float, seed: int) -> dict:
    """Send ``n`` rides with ``client`` (anything with an httpx-like ``post``) and some feedback."""
    rng = np.random.default_rng(seed)
    stats = {"sent": 0, "errors": 0, "feedback": 0, "correct_feedback": 0}
    for k in range(n):
        ride_id, ride, label = pool[int(rng.integers(len(pool)))]
        x = apply_scenario(ride, scenario)
        response = client.post(
            "/v1/predict", json={"ride_id": f"sim-{seed}-{k}-{ride_id[:12]}", "samples": to_samples(x)}
        )
        if response.status_code != 200:
            stats["errors"] += 1
            continue
        stats["sent"] += 1
        if rng.random() < feedback_rate:
            body = response.json()
            actual = feedback_label(label, scenario, rng)
            sent = client.post(
                "/v1/feedback",
                json={
                    "predicted_class": body["predicted_class"],
                    "actual_class": actual,
                    "ride_hash": body["ride_hash"],
                },
            )
            if sent.status_code == 204:
                stats["feedback"] += 1
                stats["correct_feedback"] += int(actual == body["predicted_class"])
    return stats


def main() -> int:
    import httpx  # development dependency: the simulator never runs inside the API image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--feedback-rate", type=float, default=0.5, help="share of rides whose label comes back")
    parser.add_argument("--seed", type=int, default=int(pd.Timestamp.now().timestamp()))
    args = parser.parse_args()

    pool = load_pool(load_params())
    with httpx.Client(base_url=args.url, timeout=30) as client:
        result = simulate(client, pool, args.n, args.scenario, args.feedback_rate, args.seed)
    accuracy = result["correct_feedback"] / result["feedback"] if result["feedback"] else float("nan")
    print(
        f"[{args.scenario}] {result['sent']} rides scored, {result['errors']} errors, "
        f"{result['feedback']} feedbacks (accuracy on feedback {accuracy:.2f}), seed {args.seed}"
    )
    return 1 if result["errors"] == args.n else 0


if __name__ == "__main__":
    sys.exit(main())

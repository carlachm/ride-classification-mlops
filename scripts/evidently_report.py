# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["evidently>=0.7.23", "pandas>=2.3,<3"]
# ///
"""Evidently HTML report: ride features of the prediction log against the drift reference.

    uv run --script scripts/evidently_report.py --log logs/predictions.jsonl --last 500 --out reports/drift.html

It runs in its OWN environment (versions locked in evidently_report.py.lock): Evidently does not import with
pandas 3, which the project uses. The report is for reading; the verdict and the exit code come from
`fraud_detection.monitoring.drift`. It shows distributions of the real rides: keep it out of Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Same 13 features as fraud_detection.monitoring.drift.FEATURES (checked by tests/test_drift.py)
FEATURES = [f"{c}_{s}" for c in ("ax", "ay", "az", "rx", "ry", "rz") for s in ("mean", "std")] + ["n_samples"]


def current_rides(log: Path, last: int | None) -> pd.DataFrame:
    events = []
    for line in log.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rides = [e for e in events if e.get("event") == "prediction"]
    if last:
        rides = rides[-last:]
    if not rides:
        raise SystemExit(f"no prediction in {log}")
    return pd.DataFrame(rides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=Path("data/monitoring/reference.csv"))
    parser.add_argument("--log", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/drift.html"))
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    reference = reference[reference["part"] == "train"][FEATURES]
    current = current_rides(args.log, args.last)[FEATURES]
    snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.out))
    print(f"Evidently report ({len(current)} rides vs {len(reference)}) -> {args.out}")


if __name__ == "__main__":
    main()

"""Decide whether a retraining is worth starting, and start it only when asked to.

    python -m fraud_detection.retraining.trigger --last 200              # decision only (dry run)
    python -m fraud_detection.retraining.trigger --last 200 --execute    # also runs `gh workflow run train.yml`

Hybrid trigger: input drift OR a drop of the accuracy measured on the feedback. Before starting anything:
an upstream bug is never fixed by retraining, and retraining on the data the last run already saw gives
the same model back.

Exit codes: 0 nothing to do, 1 error, 3 retrain, 4 wait (not enough rides), 5 investigate (upstream bug),
6 signal but no new labelled data.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient

from fraud_detection.config import PROJECT_ROOT, load_params
from fraud_detection.data import read_feedback
from fraud_detection.monitoring import drift
from fraud_detection.monitoring.logs import read_events, window

logger = logging.getLogger(__name__)
EXIT_CODES = {"no_action": 0, "retrain": 3, "wait": 4, "investigate": 5, "blocked_no_new_data": 6}
DATASET_POINTER = "data/feedback.dvc"


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


def feedback_performance(feedbacks: list[dict]) -> dict:
    n = len(feedbacks)
    correct = sum(f.get("predicted_class") == f.get("actual_class") for f in feedbacks)
    return {"n": n, "accuracy": correct / n if n else None}


def decide(
    drift_result: dict, performance: dict, n_new_rides: int, cfg: dict, allow_no_new_data: bool = False
) -> Decision:
    verdict = drift_result["verdict"]
    if verdict == "upstream_bug_suspected":
        return Decision("investigate", "upstream bug suspected: " + "; ".join(drift_result["problems"]))

    signals = []
    if verdict == "drift":
        signals.append(
            f"input drift on {', '.join(drift_result['drifted_features'])} (max PSI {drift_result['max_psi']:.2f})"
        )
    judged = performance["n"] >= cfg["min_feedback"]
    if judged and performance["accuracy"] < cfg["min_feedback_accuracy"]:
        signals.append(f"accuracy {performance['accuracy']:.2f} on {performance['n']} feedbacks")

    if not signals:
        if verdict == "insufficient_data" and not judged:
            return Decision("wait", f"only {drift_result['n_current']} rides and {performance['n']} feedbacks")
        return Decision("no_action", "no input drift and no performance drop")

    reason = " + ".join(signals)
    if n_new_rides < cfg["min_new_rides"] and not allow_no_new_data:
        return Decision(
            "blocked_no_new_data",
            f"{reason}, but only {n_new_rides} labelled rides unseen by the last training "
            f"(minimum {cfg['min_new_rides']}): retraining now would give about the same model",
        )
    return Decision("retrain", f"{reason}; {n_new_rides} new labelled rides")


def rides_seen_by_last_training(client: MlflowClient, model_name: str) -> int | None:
    """Production rides used by the training run of the latest registered version.

    None when the registry has no version at all. Without MLFLOW_TRACKING_URI, MLflow silently opens an
    empty local store: that must not be read as "the last training has seen nothing".
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    return int(client.get_run(latest.run_id).data.params.get("n_train_feedback_rides", 0))


def count_feedback_rides(params: dict) -> int:
    dataset = read_feedback(PROJECT_ROOT / params["data"]["feedback_dir"])
    return 0 if dataset is None else len(dataset[1])


def unpublished_dataset(root: Path = PROJECT_ROOT) -> str | None:
    """The CI runner trains on what is on origin/main + the DVC remote. Returns what is missing, or None."""
    if not (root / DATASET_POINTER).exists():
        return None  # no production dataset at all: nothing local the runner could miss

    def run(*command: str) -> subprocess.CompletedProcess:
        return subprocess.run(command, cwd=root, capture_output=True, text=True)

    if run("git", "fetch", "--quiet", "origin", "main").returncode != 0:
        return "cannot fetch origin/main"
    if run("git", "cat-file", "-e", f"origin/main:{DATASET_POINTER}").returncode != 0:
        return f"{DATASET_POINTER} is not on origin/main yet (commit it and merge a PR)"
    if run("git", "diff", "--quiet", "origin/main", "--", DATASET_POINTER).returncode != 0:
        return f"{DATASET_POINTER} differs from origin/main (commit it and merge a PR)"
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    cfg = params["retraining"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "logs" / "predictions.jsonl")
    parser.add_argument("--last", type=int, default=None, help="only the last N predictions and their feedback")
    parser.add_argument("--execute", action="store_true", help="start train.yml on GitHub when the answer is retrain")
    parser.add_argument(
        "--allow-no-new-data", action="store_true", help="retrain even without new labels (proves the pipeline only)"
    )
    args = parser.parse_args()

    drift_result = drift.run_check(params, args.log, args.last)
    _, feedbacks = window(read_events(args.log), args.last)
    performance = feedback_performance(feedbacks)
    # No registry, no decision: never start a training run blindly (an unreachable server raises too)
    name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()
    seen = rides_seen_by_last_training(client, name)
    if seen is None:
        print(
            f"no version of '{name}' in the MLflow registry at {client.tracking_uri}: "
            "set MLFLOW_TRACKING_URI (`make retrain-check` reads it from .env)",
            file=sys.stderr,
        )
        return 1
    n_new = max(0, count_feedback_rides(params) - seen)

    decision = decide(drift_result, performance, n_new, cfg, args.allow_no_new_data)
    accuracy = "n/a" if performance["accuracy"] is None else f"{performance['accuracy']:.2f}"
    print(f"drift: {drift_result['verdict']} ({drift_result['n_current']} rides)")
    print(f"feedback: {performance['n']} labels, accuracy {accuracy}")
    print(f"new labelled rides since the last training: {n_new}")
    print(f"DECISION: {decision.action.upper()} - {decision.reason}")
    if decision.action != "retrain":
        return EXIT_CODES[decision.action]

    command = ["gh", "workflow", "run", "train.yml", "-f", f"reason={decision.reason[:500]}"]
    if not args.execute:
        print("dry run, nothing started. With --execute this runs: " + " ".join(command[:4]) + " -f reason=...")
        return EXIT_CODES["retrain"]
    missing = unpublished_dataset()
    if missing:
        print(f"NOT started: {missing}", file=sys.stderr)
        return 1
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("train.yml started: gh run watch")
    return EXIT_CODES["retrain"]


if __name__ == "__main__":
    sys.exit(main())

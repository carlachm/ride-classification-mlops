"""Drift detection on the per-ride features logged by the API.

    python -m fraud_detection.monitoring.drift reference          # build the healthy reference (once per dataset)
    python -m fraud_detection.monitoring.drift check --last 200   # compare the last 200 predictions with it

Tests: PSI + Kolmogorov-Smirnov on the 12 channel means / stds and on the ride length, chi-squared on the
predicted classes (reported, never triggers). Before the verdict, a plausibility check looks for upstream
bugs (wrong unit, dead sensor): those are fixed at the source, NOT by retraining.

These tests only see the INPUTS. Concept drift (same signals, other labels) needs the feedback
(monitoring/performance, see docs/monitoring.md).

Exit codes (chain only on 3): 0 ok or alert, 1 crash, 3 data drift, 4 not enough rides,
5 upstream bug suspected.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fraud_detection.api.features import ride_summary
from fraud_detection.api.schemas import CHANNELS, CLASSES
from fraud_detection.config import PROJECT_ROOT, load_params
from fraud_detection.data import load_split
from fraud_detection.monitoring.logs import read_events, window

logger = logging.getLogger(__name__)

FEATURES: list[str] = [f"{c}_{s}" for c in CHANNELS for s in ("mean", "std")] + ["n_samples"]
ACCEL_MEANS = ["ax_mean", "ay_mean", "az_mean"]
EXIT_CODES = {"ok": 0, "alert": 0, "drift": 3, "insufficient_data": 4, "upstream_bug_suspected": 5}
EPS = 1e-4
MIN_RIDES_FOR_PLAUSIBILITY = 5


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def summarise_rides(rides: list[np.ndarray]) -> pd.DataFrame:
    """One row of features per ride, computed exactly like the API does."""
    return pd.DataFrame([ride_summary(np.asarray(r, dtype=float)) | {"n_samples": len(r)} for r in rides])


def build_reference(params: dict, model=None) -> pd.DataFrame:
    """Input reference = training rides. Prediction reference = what ``model`` answers on held-out rides.

    The split is stratified at random BY RIDE, so the three parts describe the same population. Predictions
    are taken on val + test because a model behaves differently on the rides it was trained on.
    """
    train_rides, _, _ = load_split("train", params)
    parts = [summarise_rides(train_rides).assign(part="train")]
    if model is not None:
        held = load_split("val", params)[0] + load_split("test", params)[0]
        predicted = [
            CLASSES[int(np.asarray(model.predict(pd.DataFrame(r, columns=CHANNELS, dtype="float64"))).argmax())]
            for r in held
        ]
        parts.append(summarise_rides(held).assign(part="heldout", predicted_class=predicted))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    """Population Stability Index, bins defined by the reference quantiles, open outer edges."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), EPS, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def plausibility(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> list[str]:
    """Signs of an upstream bug rather than a real change of the rides."""
    problems = []
    # At rest an accelerometer measures gravity: the norm of the mean acceleration stays close to 9.8 m/s²
    gravity = float(np.median(np.sqrt((current[ACCEL_MEANS] ** 2).sum(axis=1))))
    low, high = cfg["gravity_range"]
    if not low <= gravity <= high:
        problems.append(
            f"gravity norm is {gravity:.2f} m/s² (expected {low}-{high}): accelerometer unit, scale or axis problem"
        )
    for channel in CHANNELS:
        ratio = float(current[f"{channel}_std"].median() / reference[f"{channel}_std"].median())
        if ratio > cfg["unit_ratio"] or ratio < 1 / cfg["unit_ratio"]:
            problems.append(f"{channel} varies {ratio:.3g}x as much as in the reference: unit bug or dead sensor")
    return problems


def prediction_test(reference: pd.Series, current: pd.Series) -> dict | None:
    """Chi-squared on the predicted classes. Classes too rare for the test are grouped together."""
    counts = pd.DataFrame(
        {"reference": reference.value_counts(), "current": current.value_counts()}, index=CLASSES
    ).fillna(0)
    rare = counts.sum(axis=1) < 10
    if rare.any():
        counts = pd.concat([counts[~rare], counts[rare].sum().to_frame("other").T])
    counts = counts[counts.sum(axis=1) > 0]
    if len(counts) < 2:
        return None
    return {
        "test": "chi2",
        "pvalue": float(stats.chi2_contingency(counts.T.to_numpy())[1]),
        "reference_share": {c: round(float(v), 3) for c, v in reference.value_counts(normalize=True).items()},
        "current_share": {c: round(float(v), 3) for c, v in current.value_counts(normalize=True).items()},
    }


def compare(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict:
    """Full drift report. ``reference`` comes from ``build_reference``, ``current`` from the prediction log."""
    ref_inputs = reference[reference["part"] == "train"]
    result: dict = {"n_reference": len(ref_inputs), "n_current": len(current), "problems": [], "features": {}}
    if len(current) >= MIN_RIDES_FOR_PLAUSIBILITY:
        result["problems"] = plausibility(ref_inputs, current, cfg)
    if len(current) < cfg["min_rides"]:
        verdict = "upstream_bug_suspected" if result["problems"] else "insufficient_data"
        return result | {"verdict": verdict, "max_psi": None, "drifted_features": []}

    alpha = cfg["ks_alpha"] / len(FEATURES)  # Bonferroni: one test per feature
    drifted = []
    for name in FEATURES:
        ref, cur = ref_inputs[name].to_numpy(float), current[name].to_numpy(float)
        score, pvalue = psi(ref, cur, cfg["bins"]), float(stats.ks_2samp(ref, cur).pvalue)
        if score > cfg["psi_retrain"] and pvalue < alpha:
            level = "drift"
            drifted.append(name)
        else:
            level = "alert" if score > cfg["psi_alert"] else "ok"
        result["features"][name] = {"psi": round(score, 4), "ks_pvalue": pvalue, "level": level}

    held = reference[reference["part"] == "heldout"]
    if len(held) and "predicted_class" in current:
        result["prediction"] = prediction_test(held["predicted_class"], current["predicted_class"])

    if result["problems"]:
        verdict = "upstream_bug_suspected"
    elif len(drifted) >= cfg["min_drifted_features"]:
        verdict = "drift"
    else:
        verdict = "alert" if drifted else "ok"
    return result | {
        "verdict": verdict,
        "max_psi": max(f["psi"] for f in result["features"].values()),
        "drifted_features": drifted,
    }


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def load_reference(params: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / params["drift"]["reference_path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `make drift-reference` first")
    return pd.read_csv(path)


def run_check(params: dict, log_path: Path, last: int | None) -> dict:
    predictions, _ = window(read_events(log_path), last)
    current = pd.DataFrame(predictions, columns=None if predictions else FEATURES)
    return compare(load_reference(params), current, params["drift"])


def format_report(result: dict) -> str:
    lines = []
    for name, f in sorted(result["features"].items(), key=lambda kv: -kv[1]["psi"]):
        lines.append(f"{name:10s} PSI={f['psi']:7.3f}  KS p={f['ks_pvalue']:.2g}  {f['level']}")
    if result.get("prediction"):
        lines.append(f"predicted classes: chi2 p={result['prediction']['pvalue']:.2g}")
    lines += [f"UPSTREAM BUG? {p}" for p in result["problems"]]
    lines.append(
        f"{result['n_current']} rides vs {result['n_reference']} reference rides -> {result['verdict'].upper()}"
        + (f" ({len(result['drifted_features'])} drifted features)" if result["drifted_features"] else "")
    )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference", help="build the reference file from the prepared data")
    ref.add_argument("--no-model", action="store_true", help="skip the predicted-class reference")
    check = sub.add_parser("check", help="compare the prediction log with the reference")
    check.add_argument("--log", type=Path, default=PROJECT_ROOT / "logs" / "predictions.jsonl")
    check.add_argument("--last", type=int, default=None, help="only the last N predictions")
    check.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "drift.json")
    args = parser.parse_args()

    if args.command == "reference":
        model = None
        if not args.no_model:
            from fraud_detection.api import model_store

            model = model_store.load(model_store.default_model_uri()).model
        path = PROJECT_ROOT / params["drift"]["reference_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        reference = build_reference(params, model)
        reference.to_csv(path, index=False)
        print(f"{len(reference)} reference rides written to {path}")
        return 0

    result = run_check(params, args.log, args.last)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(format_report(result))
    print(f"report -> {args.out}")
    return EXIT_CODES[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())

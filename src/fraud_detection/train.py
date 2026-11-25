"""Train the 1D-CNN ride classifier and track everything with MLflow.

Usage:
    python -m fraud_detection.train

Env vars (see .env.example):
    MLFLOW_TRACKING_URI   e.g. http://localhost:5050 (default: local ./mlruns)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_detection.config import FEATURE_COLUMNS, LABELS, PROJECT_ROOT, load_params, set_seed
from fraud_detection.data import load_split
from fraud_detection.evaluate import (
    compute_metrics,
    majority_baseline_macro_f1,
    report_text,
    save_confusion_matrix,
)
from fraud_detection.model import (
    RideClassifierPyfunc,
    RideCNN,
    Standardizer,
    predict_ride_proba,
    ride_to_windows,
    save_artifacts,
)
from fraud_detection.registry import promote_if_better

logger = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).parent


def _flatten(d: dict, prefix: str = "") -> dict:
    """{'train': {'lr': 1}} -> {'train.lr': 1} (for mlflow.log_params)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def class_weights(y_rides: np.ndarray, mode: str) -> torch.Tensor:
    """Class weights for the loss, computed on rides and normalised to mean 1.

    "balanced" = inverse class frequency, "sqrt" = its square root (softer), "none" = all 1.
    """
    if mode == "none":
        return torch.ones(len(LABELS))
    counts = np.bincount(y_rides, minlength=len(LABELS)).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    if mode == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)


def windows_dataset(rides: list[np.ndarray], y: np.ndarray, scaler: Standardizer, wcfg: dict) -> TensorDataset:
    xs, ys = [], []
    for ride, label in zip(rides, y, strict=True):
        w = ride_to_windows(ride, scaler, wcfg["length"], wcfg["stride"])
        xs.append(w)
        ys.append(np.full(len(w), label, dtype=np.int64))
    return TensorDataset(torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)))


def predict_rides(model: RideCNN, rides: list[np.ndarray], scaler: Standardizer, wcfg: dict) -> np.ndarray:
    """(n_rides, 5) probabilities, each ride = mean over its windows."""
    return np.stack(
        [predict_ride_proba(model, ride_to_windows(r, scaler, wcfg["length"], wcfg["stride"])) for r in rides]
    )


def train_model(
    model: RideCNN,
    train_ds: TensorDataset,
    val_rides: list[np.ndarray],
    y_val: np.ndarray,
    scaler: Standardizer,
    params: dict,
    weights: torch.Tensor,
) -> RideCNN:
    """Training loop on windows; early stopping on the RIDE-level validation macro-F1."""
    cfg, wcfg = params["train"], params["windows"]
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(params["seed"])
    )

    best_score, best_state, patience = -1.0, None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        val_pred = predict_rides(model, val_rides, scaler, wcfg).argmax(axis=1)
        val = compute_metrics(y_val, val_pred)
        train_loss = total / len(train_ds)
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_macro_f1_epoch": val["macro_f1"],
                "val_balanced_accuracy_epoch": val["balanced_accuracy"],
            },
            step=epoch,
        )
        logger.info("epoch %02d - loss %.4f - val macro-F1 %.3f", epoch, train_loss, val["macro_f1"])

        if val["macro_f1"] > best_score:
            best_score, patience = val["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    mlflow.log_metric("best_val_macro_f1", best_score)
    return model


def run(params: dict | None = None) -> dict:
    """Full training run: data -> windows -> train -> evaluate -> log -> register -> promote."""
    params = params or load_params()
    set_seed(params["seed"])
    mlf, wcfg = params["mlflow"], params["windows"]
    mlflow.set_experiment(mlf["experiment_name"])

    train_rides, y_train, _ = load_split("train", params)
    val_rides, y_val, _ = load_split("val", params)
    test_rides, y_test, test_ids = load_split("test", params)

    with mlflow.start_run() as active_run:
        mlflow.set_tags({"model_type": "pytorch-1d-cnn", "task": "multiclass-ride-classification"})
        mlflow.log_params(_flatten({k: params[k] for k in ("seed", "data", "windows", "model", "train")}))

        stats_path = PROJECT_ROOT / params["data"]["processed_dir"] / "data_stats.json"
        data_stats = {}
        if stats_path.exists():
            mlflow.log_artifact(str(stats_path), artifact_path="data")
            data_stats = json.loads(stats_path.read_text())
        # How many labelled production rides this run has seen: the retraining trigger reads it
        mlflow.log_param("n_train_feedback_rides", data_stats.get("n_feedback_rides", 0))
        if os.getenv("RETRAIN_REASON"):
            mlflow.set_tag("retrain_reason", os.environ["RETRAIN_REASON"][:500])

        scaler = Standardizer.fit(train_rides)
        train_ds = windows_dataset(train_rides, y_train, scaler, wcfg)
        weights = class_weights(y_train, params["train"]["class_weight"])
        mlflow.log_params(
            {"n_train_rides": len(train_rides), "n_train_windows": len(train_ds)}
            | {f"class_weight.{LABELS[i].lower().replace(' ', '_')}": round(float(w), 3) for i, w in enumerate(weights)}
        )

        m = params["model"]
        model = RideCNN(channels=tuple(m["channels"]), kernel_size=m["kernel_size"], dropout=m["dropout"])
        model = train_model(model, train_ds, val_rides, y_val, scaler, params, weights)

        # Final evaluation on the untouched test rides
        val_metrics = compute_metrics(y_val, predict_rides(model, val_rides, scaler, wcfg).argmax(axis=1))
        test_proba = predict_rides(model, test_rides, scaler, wcfg)
        test_pred = test_proba.argmax(axis=1)
        test_metrics = compute_metrics(y_test, test_pred)
        baseline = majority_baseline_macro_f1(y_test)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("baseline_majority_macro_f1", baseline)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mlflow.log_artifact(str(save_confusion_matrix(y_test, test_pred, tmp, "Test rides")), "plots")
            (tmp / "classification_report.txt").write_text(report_text(y_test, test_pred))
            mlflow.log_artifact(str(tmp / "classification_report.txt"), "reports")
            pd.DataFrame(test_proba, columns=LABELS).assign(
                ride_id=test_ids, true=[LABELS[i] for i in y_test], predicted=[LABELS[i] for i in test_pred]
            ).to_csv(tmp / "test_predictions.csv", index=False)
            mlflow.log_artifact(str(tmp / "test_predictions.csv"), "reports")

            # Log + register the pyfunc that follows docs/api-contract.md
            artifacts = save_artifacts(model, scaler, m, wcfg, tmp / "model_artifacts")
            # float64: the API sends JSON numbers as float64, and MLflow refuses to downcast them to float32
            example = pd.DataFrame(test_rides[0][: wcfg["length"]], columns=FEATURE_COLUMNS).astype("float64")
            wrapper = RideClassifierPyfunc()
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=wrapper,
                artifacts=artifacts,
                # the model ships with the code it needs: loadable by the API image without this repo
                code_paths=[str(PACKAGE_DIR)],
                signature=infer_signature(example, np.zeros((1, len(LABELS)), dtype=np.float64)),
                input_example=example,
                registered_model_name=mlf["registered_model_name"],
            )

        client = MlflowClient()
        version = str(model_info.registered_model_version)
        min_score = params.get("quality_gate", {}).get("min_test_macro_f1", 0.0)
        gate_passed = test_metrics["macro_f1"] >= min_score and test_metrics["macro_f1"] > baseline
        mlflow.set_tag("quality_gate", "passed" if gate_passed else "failed")
        if gate_passed:
            promoted = promote_if_better(
                client, mlf["registered_model_name"], version, test_metrics["macro_f1"], alias=mlf["champion_alias"]
            )
        else:
            logger.warning(
                "Quality gate failed: test macro-F1 %.3f (min %.3f, baseline %.3f)",
                test_metrics["macro_f1"],
                min_score,
                baseline,
            )
            client.set_model_version_tag(mlf["registered_model_name"], version, "promotion", "gate_failed")
            promoted = False

        summary = {
            "run_id": active_run.info.run_id,
            "model_version": version,
            "promoted": promoted,
            "quality_gate_passed": gate_passed,
            "baseline_majority_macro_f1": baseline,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

    # metrics.json is tracked by DVC (dvc metrics show / dvc metrics diff)
    (PROJECT_ROOT / params.get("metrics_path", "metrics.json")).write_text(json.dumps(summary, indent=2))
    logger.info("Test report:\n%s", report_text(y_test, test_pred))
    logger.info("Run summary: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run()
    if not summary["quality_gate_passed"]:
        sys.exit(1)  # makes the CI job / DVC stage fail


if __name__ == "__main__":
    main()

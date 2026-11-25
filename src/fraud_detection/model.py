"""1D-CNN ride classifier and the MLflow pyfunc wrapper served by the API.

Pipeline, identical at training and serving time:
    ride (n_samples, 6) -> fill NaN -> standardise (train statistics) -> windows (length L, stride S)
    -> CNN -> softmax per window -> average over windows -> 5 class probabilities

The pyfunc follows docs/api-contract.md: input = DataFrame (one row per sample, columns ax..rz,
NaN allowed), output = ndarray (1, 5) in the order of ``fraud_detection.api.schemas.CLASSES``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

from fraud_detection.api.schemas import CHANNELS, CLASSES


# --------------------------------------------------------------------------- #
# Pre-processing (pure NumPy, shared by training and serving)
# --------------------------------------------------------------------------- #
def fill_missing(x: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill each channel (sensor dropouts); a fully empty channel becomes 0."""
    df = pd.DataFrame(x)
    return df.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)


def make_windows(x: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Cut a ride (n, C) into windows (n_windows, C, length).

    A ride shorter than ``length`` is padded with zeros (= the channel mean after
    standardisation); the last window is aligned on the end of the ride so no sample is lost.
    """
    n = len(x)
    if n < length:
        x = np.concatenate([x, np.zeros((length - n, x.shape[1]), dtype=x.dtype)])
        n = length
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    return np.stack([x[s : s + length].T for s in starts]).astype(np.float32)


class Standardizer:
    """Per-channel mean / std computed on the TRAINING rides only (no leakage)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.where(np.asarray(std) < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, rides: list[np.ndarray]) -> Standardizer:
        allx = np.concatenate([fill_missing(r) for r in rides])
        return cls(allx.mean(axis=0), allx.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((fill_missing(x) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def ride_to_windows(x: np.ndarray, scaler: Standardizer, length: int, stride: int) -> np.ndarray:
    return make_windows(scaler.transform(x), length, stride)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class RideCNN(nn.Module):
    """Conv1d blocks over time, global average pooling, linear head (logits for 5 classes)."""

    def __init__(
        self,
        n_channels: int = len(CHANNELS),
        n_classes: int = len(CLASSES),
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_channels
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, channels, time)
        return self.head(self.features(x))


@torch.no_grad()
def predict_ride_proba(model: RideCNN, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Class probabilities of ONE ride = mean of the softmax of its windows."""
    model.eval()
    probs = []
    for i in range(0, len(windows), batch_size):
        logits = model(torch.from_numpy(windows[i : i + batch_size]))
        probs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(probs).mean(axis=0)


# --------------------------------------------------------------------------- #
# MLflow pyfunc (what the API loads)
# --------------------------------------------------------------------------- #
WEIGHTS_FILE = "ride_cnn.pt"
CONFIG_FILE = "ride_cnn.json"


def save_artifacts(model: RideCNN, scaler: Standardizer, model_cfg: dict, windows_cfg: dict, out_dir: Path) -> dict:
    """Write the weights + everything needed to rebuild the pre-processing; returns the artifact paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / WEIGHTS_FILE)
    config = {"model": model_cfg, "windows": windows_cfg, "scaler": scaler.to_dict(), "classes": CLASSES}
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    return {"weights": str(out_dir / WEIGHTS_FILE), "config": str(out_dir / CONFIG_FILE)}


class RideClassifierPyfunc(mlflow.pyfunc.PythonModel):
    """Self-contained model: NaN handling, standardisation and windowing live here (API stays model-agnostic)."""

    def load_context(self, context) -> None:
        config = json.loads(Path(context.artifacts["config"]).read_text())
        self.windows_cfg = config["windows"]
        self.scaler = Standardizer(config["scaler"]["mean"], config["scaler"]["std"])
        m = config["model"]
        self.model = RideCNN(channels=tuple(m["channels"]), kernel_size=m["kernel_size"], dropout=m["dropout"])
        self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        x = model_input[CHANNELS].to_numpy(dtype=np.float32)
        windows = ride_to_windows(x, self.scaler, self.windows_cfg["length"], self.windows_cfg["stride"])
        return predict_ride_proba(self.model, windows)[None, :]

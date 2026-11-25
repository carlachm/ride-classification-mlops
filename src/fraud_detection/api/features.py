"""Per-ride summary and fingerprint of the IMU signal, shared by the API log, the monitoring and the retraining."""

from __future__ import annotations

import hashlib

import numpy as np

from fraud_detection.api.schemas import CHANNELS


def ride_fingerprint(x: np.ndarray) -> str:
    """Content hash of one ride (n_samples, 6): the same signal always gives the same 32-character id.

    It links a feedback to the ride that was scored, and lets the retraining recognise a ride that is
    already in the validation / test sets. Values are cast to float32 and rounded first, so that the
    raw file (float64), the processed splits (float32) and a JSON round trip give the same id.
    """
    y = np.round(np.asarray(x, dtype=np.float32).astype("<f8"), 3) + 0.0  # + 0.0 turns -0.0 into 0.0
    y[np.isnan(y)] = np.nan  # one single bit pattern for missing values
    return hashlib.sha256(y.tobytes()).hexdigest()[:32]


def ride_summary(x: np.ndarray) -> dict[str, float]:
    """Mean and standard deviation of each channel; NaN (sensor dropouts) are ignored.

    ``x`` has shape (n_samples, 6), columns in the order of ``CHANNELS``.
    """
    out: dict[str, float] = {}
    for i, name in enumerate(CHANNELS):
        col = x[:, i]
        out[f"{name}_mean"] = float(np.nanmean(col)) if not np.isnan(col).all() else 0.0
        out[f"{name}_std"] = float(np.nanstd(col)) if not np.isnan(col).all() else 0.0
    return out

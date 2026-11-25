"""Optional storage of the raw signals received by the API, so that labelled rides can be used to retrain.

Enabled by ``RIDE_STORE_DIR`` (off by default): one parquet file per ride, named after its fingerprint.
The prediction log only keeps per-ride statistics, which are not enough to train the CNN.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger("fraud_detection.api")


def save_ride(frame: pd.DataFrame, ride_hash: str) -> None:
    """Write the ride once. A storage problem is logged and never makes the prediction fail."""
    directory = os.getenv("RIDE_STORE_DIR")
    if not directory:
        return
    path = Path(directory) / f"{ride_hash}.parquet"
    try:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.astype("float32").to_parquet(tmp, index=False)
        os.replace(tmp, path)  # never leave a half-written ride
    except Exception as exc:
        logger.warning("ride %s not stored: %s", ride_hash, exc)

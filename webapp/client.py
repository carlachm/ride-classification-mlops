"""HTTP client and file parsing of the web app (no Streamlit here, so it can be unit-tested)."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import requests

CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz"]


def read_ride(name: str, content: bytes) -> pd.DataFrame:
    """Read an uploaded CSV / Parquet file into a DataFrame with the 6 IMU channels, in time order.

    Files with several rides (like GBG500.parquet) are accepted: pick one with the ``ride_id`` column.
    """
    if name.lower().endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(CHANNELS)}")
    return df


def demo_ride(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic ride that looks like a calm ride (gravity on ay, small noise)."""
    rng = np.random.default_rng(seed)
    gravity = np.array([-0.1, -9.15, -2.3, 0.0, 0.0, 0.0])
    scale = np.array([0.8, 1.1, 1.0, 5.0, 20.0, 8.0])
    return pd.DataFrame(gravity + rng.normal(size=(n, 6)) * scale, columns=CHANNELS)


def to_payload(df: pd.DataFrame, ride_id: str | None = None) -> dict:
    """API request body; NaN become JSON ``null`` (sensor dropouts are accepted by the API)."""
    if "time_index" in df.columns:
        df = df.sort_values("time_index")
    samples = json.loads(df[CHANNELS].to_json(orient="records"))
    return {"ride_id": ride_id, "samples": samples}


def predict(api_url: str, payload: dict, timeout: float = 30) -> dict:
    """Call POST /v1/predict. Raises ``ApiError`` with a readable message on failure."""
    try:
        r = requests.post(f"{api_url.rstrip('/')}/v1/predict", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"API unreachable at {api_url}: {exc.__class__.__name__}") from exc
    if r.status_code != 200:
        raise ApiError(f"API answered {r.status_code}: {_detail(r)}")
    return r.json()


def health(api_url: str, timeout: float = 5) -> dict:
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise ApiError(f"API unreachable at {api_url}: {exc.__class__.__name__}") from exc


def send_feedback(
    api_url: str, predicted_class: str, actual_class: str, ride_hash: str | None = None, timeout: float = 10
) -> None:
    """``ride_hash`` comes from the prediction: it links the label to the ride (needed to retrain)."""
    r = requests.post(
        f"{api_url.rstrip('/')}/v1/feedback",
        json={"predicted_class": predicted_class, "actual_class": actual_class, "ride_hash": ride_hash},
        timeout=timeout,
    )
    if r.status_code != 204:
        raise ApiError(f"API answered {r.status_code}: {_detail(r)}")


class ApiError(RuntimeError):
    pass


def _detail(r: requests.Response) -> str:
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        return r.text[:200]
    return json.dumps(detail)[:300] if not isinstance(detail, str) else detail

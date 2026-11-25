"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fraud_detection.api.schemas import CHANNELS, CLASSES

# Relative paths in params.yaml are resolved from the directory the commands are
# launched from (the repo root, like DVC does). Override with PROJECT_ROOT if needed.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# Shared with the API (single source of truth: fraud_detection.api.schemas)
FEATURE_COLUMNS: list[str] = CHANNELS
TARGET_COLUMN = "label"
RIDE_COLUMN = "ride_id"
TIME_COLUMN = "time_index"
LABELS: list[str] = CLASSES


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load the project parameters from ``params.yaml``."""
    path = Path(path or os.getenv("PARAMS_PATH", DEFAULT_PARAMS_PATH))
    with path.open() as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Make runs reproducible (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

"""Read the JSON-lines prediction log written by the API (format: docs/api-contract.md)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_events(path: str | Path) -> list[dict]:
    """Prediction and feedback events, in file order. Unreadable lines are skipped and counted."""
    events, bad = [], 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        logger.warning("%d unreadable line(s) skipped in %s", bad, path)
    return events


def window(events: list[dict], last: int | None = None) -> tuple[list[dict], list[dict]]:
    """(predictions, feedbacks) of the last ``last`` predictions (everything when ``last`` is None).

    A feedback belongs to the window if it was received after the first prediction of the window.
    """
    positions = [i for i, e in enumerate(events) if e.get("event") == "prediction"]
    start = 0
    if last is not None:
        positions = positions[-last:] if last > 0 else []
        start = positions[0] if positions else len(events)
    predictions = [events[i] for i in positions]
    feedbacks = [e for e in events[start:] if e.get("event") == "feedback"]
    return predictions, feedbacks

"""MLflow Model Registry helpers.

We use *aliases* (``champion``) rather than the deprecated stages
(Staging/Production). The API will always load ``models:/ride-classifier@champion``,
so promoting or rolling back a model is just moving the alias.
"""

from __future__ import annotations

import logging

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


def get_champion_metric(client: MlflowClient, model_name: str, alias: str, metric: str) -> float | None:
    """Return ``metric`` of the run behind the current champion, or None."""
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None  # no champion yet
    run = client.get_run(version.run_id)
    return run.data.metrics.get(metric)


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    version: str,
    new_score: float,
    alias: str = "champion",
    metric: str = "test_macro_f1",
) -> bool:
    """Set ``alias`` on ``version`` if it beats the current champion on ``metric``."""
    current = get_champion_metric(client, model_name, alias, metric)
    if current is not None and new_score <= current:
        logger.info("Version %s (%s=%.4f) does not beat champion (%.4f)", version, metric, new_score, current)
        client.set_model_version_tag(model_name, version, "promotion", "rejected")
        return False

    client.set_registered_model_alias(model_name, alias, version)
    client.set_model_version_tag(model_name, version, "promotion", "champion")
    logger.info("Version %s promoted to @%s (%s=%.4f, previous=%s)", version, alias, metric, new_score, current)
    return True

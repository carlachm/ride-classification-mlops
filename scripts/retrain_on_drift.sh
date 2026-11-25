#!/usr/bin/env bash
# S7: from the drift verdict to the retraining workflow.
#   CURRENT=logs/baseline.jsonl bash scripts/retrain_on_drift.sh    # normal traffic: must start nothing
#   bash scripts/retrain_on_drift.sh                                # logs/drift.jsonl
#   DRY_RUN=1 bash scripts/retrain_on_drift.sh                      # decision only
# The decision (and its guards: upstream bug, no new labelled ride) lives in
# fraud_detection.retraining.trigger. The workflow starts only on the RETRAIN decision: a crash of the
# check never starts a training run. Extra arguments are passed on (e.g. --allow-no-new-data).
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; . ./.env; set +a; fi   # MLFLOW_TRACKING_URI
CURRENT=${CURRENT:-logs/drift.jsonl}
EXECUTE=--execute
[ "${DRY_RUN:-0}" = "1" ] && EXECUTE=

# Exit codes: 0 nothing to do, 3 retrain, 4 wait, 5 investigate, 6 no new labelled data, 1 error
exec uv run python -m fraud_detection.retraining.trigger --log "$CURRENT" $EXECUTE "$@"

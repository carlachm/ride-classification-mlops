#!/usr/bin/env bash
# Smoke test of the training image: run the WHOLE pipeline inside the container
# on synthetic data with a throw-away MLflow store. If anything crashes
# (missing file, broken import, bad dependency), the script exits non-zero.
#   bash scripts/smoke_test.sh ride-train:ci
set -euo pipefail

IMAGE="${1:?usage: smoke_test.sh <image>}"

docker run --rm \
  -e MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db \
  "$IMAGE" \
  sh -c "python -m fraud_detection.data sample --rides 150 \
      && python -m fraud_detection.data prepare \
      && python -m fraud_detection.train \
      && cat metrics.json"

echo "Smoke test passed for $IMAGE"

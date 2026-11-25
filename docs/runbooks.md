# Runbooks

One section per alert of `infra/prometheus/alerts.yml`. In an incident report, write down when the alert
fired, the decision time and the recovery time.

## API down

Alerts `ApiDown`, `ModelNotLoaded`.

1. `curl -s localhost:8000/health`. No answer: restart the API (`make api`, or check the pods with
   `kubectl -n ride get pods`).
2. `"status": "degraded"`: the model did not load. Read the `model_load_failed` line of the API logs.
   Usual causes: MLflow stack stopped (`make up`), no `champion` alias in the registry.

## Latency

Alert `LatencyP95High`.

1. Grafana, row 1: did the traffic increase at the same time? Then it is capacity: reduce the load, or add
   replicas (`kubectl -n ride scale deploy/ride-api --replicas=4`).
2. Same traffic: look at `inference_ms` in the prediction log and at `input_ride_samples` (long rides cost
   more). New model version? [Roll back](#rollback).

## Errors

Alert `ErrorRateHigh`.

1. `GET /health`, then the API logs.
2. Errors right after a deployment or a model promotion: [roll back](#rollback) first, investigate after.

## Upstream bug

Alerts `InputUnitSuspected`, `RejectedRidesHigh`, `MissingValuesHigh`.

1. `make drift`: the `UPSTREAM BUG?` lines name the channel and the symptom.
2. Compare a recent raw ride with a training ride: units (m/s² vs g, °/s vs rad/s), axis order, a sensor
   stuck at a constant value, a new firmware or phone app version.
3. **Fix at the sender. Do not retrain**: the model would learn the bug.
4. Do not use the feedback of the affected period for retraining (remove those lines from the log before
   `make feedback-data`).
5. Check: after the fix, `make drift` on new rides no longer reports a problem.

## Drift

Alerts `InputShift`, `SafeShareShift`.

1. `make drift`: which features drifted, is the verdict `drift` or only `alert`?
2. Is it an [upstream bug](#upstream-bug)? Rule it out first.
3. Is the sample large enough (at least 100 rides), and is the change explained (season, road works, new
   city, new scooter model)?
4. `make retrain-check`. It only says `RETRAIN` if there are new labelled rides: see
   [retraining.md](retraining.md).

## Performance

Alerts `FeedbackAccuracyDrop`, `RecklessRideMissed`.

1. Check the labels first: who labelled, did the labelling rule change?
2. Inputs stable (`make drift` is `ok`) but accuracy down: likely **concept drift**. Collect the labelled
   rides (`make feedback-data`) and follow [retraining.md](retraining.md).
3. `RecklessRideMissed`: review the ride. The current champion rarely recognises Reckless rides: this is a
   known weakness of the model, to be decided with the model owner.

## Rollback

A model rollback does not need a new image: move the alias, then restart the API (it loads the model at
startup).

```bash
uv run python -c "from mlflow import MlflowClient; MlflowClient().set_registered_model_alias('ride-classifier', 'champion', '<previous version>')"
kubectl -n ride rollout restart deploy/ride-api      # local API: stop and `make api` again
kubectl -n ride rollout status deploy/ride-api
```

(`MLFLOW_TRACKING_URI` must be set: `set -a; . ./.env; set +a`.) A bad **image** is rolled back with
`kubectl -n ride rollout undo deploy/ride-api`.

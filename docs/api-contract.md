# Interface contract (API, model, logs, metrics)

This page is the agreement between the three parts of the project: **training** (Carla), **serving** (Enzo)
and **monitoring / retraining / feature store** (Ugo). If you need to change something here, open an issue first.

## 1. Model → API (Carla → Enzo)

The API loads `models:/ride-classifier@champion` from the MLflow registry (`MODEL_NAME`, `MODEL_ALIAS`, or the
full `MODEL_URI`, which can also be a local directory).

The model is logged with `mlflow.pyfunc.log_model` and must honour this interface:

| | |
|---|---|
| **Input** | a `pandas.DataFrame`, one row per IMU sample, columns `ax, ay, az, rx, ry, rz` (float, may contain `NaN`), rows in time order, 100 ms step |
| **Output** | a `numpy.ndarray` of shape `(1, 5)`: the probability of each class, in the order below, summing to 1 |
| **Classes** | `["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]` (`fraud_detection.api.schemas.CLASSES`) |

The pre-processing (NaN handling, normalisation, padding/cropping to a fixed length) lives **inside** the pyfunc
wrapper, so the API stays model-agnostic. Until the real model exists, `make placeholder` registers a stand-in with the
same interface (`fraud_detection.api.dummy_model`).

## 2. HTTP API

Interactive documentation (Swagger): `http://localhost:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness; `status` is `ok` or `degraded` (no model), always HTTP 200 |
| GET | `/ready` | readiness; 200 only when a model is loaded, else 503 |
| POST | `/v1/predict` | classify one ride |
| POST | `/v1/feedback` | send the true class of a past prediction (204) |
| GET | `/metrics/` | Prometheus metrics |

`POST /v1/predict`

```json
{ "ride_id": "optional-id",
  "samples": [ {"ax": 0.1, "ay": -9.8, "az": -2.3, "rx": 0.0, "ry": 0.1, "rz": 0.0}, "... 100 to 20 000 items" ] }
```

Any value may be `null` (sensor dropout), but a channel that is `null` for the whole ride gives HTTP 422.

```json
{ "ride_id": "optional-id", "predicted_class": "Safe", "confidence": 0.62,
  "probabilities": {"Safe": 0.62, "Safe overall": 0.1, "Bad weather": 0.1, "Tandem": 0.1, "Reckless": 0.08},
  "n_samples": 2349, "model_version": "3", "ride_hash": "9f2c0a...(32 hex characters)" }
```

`ride_hash` is a fingerprint of the signal (same samples, same value).

Errors: 422 invalid input, 503 no model loaded.

`POST /v1/feedback`: `{"predicted_class": "Safe", "actual_class": "Reckless", "ride_hash": "9f2c0a..."}`
(both classes must be known). `ride_hash` is **optional** (older clients still work), but without it the label only
feeds the metrics: it cannot be used for retraining.

When `RIDE_STORE_DIR` is set, the API also keeps the raw signal of each scored ride in
`<RIDE_STORE_DIR>/<ride_hash>.parquet` (needed to retrain; see [retraining.md](retraining.md)). It is off by default.

## 3. Prediction log (API → Ugo)

One JSON line per event in the file given by `PREDICTION_LOG_PATH` (`/app/logs/predictions.jsonl` in the image;
also written to stdout).

```json
{"ts": "2026-11-20T10:00:00+00:00", "event": "prediction", "ride_id": "r-1", "ride_hash": "9f2c0a...",
 "n_samples": 2349,
 "ax_mean": 0.1, "ax_std": 1.2, "ay_mean": -9.1, "ay_std": 0.9, "...": "same for az, rx, ry, rz",
 "predicted_class": "Safe", "probabilities": {"Safe": 0.62, "...": 0.1},
 "model_version": "3", "inference_ms": 12.3}
{"ts": "...", "event": "feedback", "predicted_class": "Safe", "actual_class": "Reckless", "ride_hash": "9f2c0a..."}
```

The `<channel>_mean` / `<channel>_std` columns are the per-ride features used for **drift detection** (compare them to
the training distribution) and a natural candidate for the **feature store** (`fraud_detection.api.features.ride_summary`).

## 4. Prometheus metrics (API → Ugo)

Scraped at `/metrics/` (the pods carry the `prometheus.io/*` annotations).

| Layer | Metric | Type | Labels |
|---|---|---|---|
| Infrastructure | `api_requests_total` | counter | `endpoint`, `status` |
| Infrastructure | `api_request_duration_seconds` | histogram | `endpoint` |
| Input (data drift) | `input_ride_samples` | histogram | |
| Input (data drift) | `input_channel_std` | histogram | `channel` |
| Input (data drift) | `input_missing_ratio` | histogram | |
| Prediction | `prediction_class_total` | counter | `predicted_class` |
| Prediction | `prediction_confidence` | histogram | |
| Business | `feedback_total` | counter | `predicted_class`, `actual_class` |
| Model | `model_info` | gauge | `version` |

## 5. Deployment

Images (never `latest`, the tag is the 7-character commit SHA): `<dockerhub-user>/ride-api`,
`<dockerhub-user>/ride-webapp`. Kubernetes namespace `ride`, Services `ride-api` (NodePort 30080 → host 8080 with kind)
and `ride-webapp` (30501 → host 8501). Inside the cluster the API is `http://ride-api`.
The API reaches MLflow through the `ride-api-config` ConfigMap (`MLFLOW_TRACKING_URI`).

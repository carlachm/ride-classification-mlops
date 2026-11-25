"""Prometheus metrics, one family per monitoring layer (names used by the Grafana dashboards)."""

from prometheus_client import Counter, Gauge, Histogram

from fraud_detection.api.schemas import CHANNELS, CLASSES

# Layer 1: infrastructure
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),
)

# Layer 2: input distribution (data drift)
INPUT_SAMPLES = Histogram(
    "input_ride_samples",
    "Number of IMU samples per ride",
    buckets=(100, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000),
)
INPUT_CHANNEL_STD = Histogram(
    "input_channel_std",
    "Standard deviation of each IMU channel over a ride",
    ["channel"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100),
)
INPUT_MISSING_RATIO = Histogram(
    "input_missing_ratio",
    "Share of null values in a ride",
    buckets=(0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1),
)
MONITORED_CHANNELS = tuple(CHANNELS)

# Layer 3: prediction distribution
PREDICTED_CLASS = Counter("prediction_class_total", "Predictions per class", ["predicted_class"])
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Probability of the predicted class",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

# Layer 4: business outcome (true labels received later)
FEEDBACK = Counter("feedback_total", "True labels received", ["predicted_class", "actual_class"])

MODEL_INFO = Gauge("model_info", "Model version being served", ["version"])

# Export every class combination at 0 from the start: Prometheus' increase() does not count the first
# sample of a series that appears directly with the value 1, so rare labels would be lost.
for _predicted in CLASSES:
    PREDICTED_CLASS.labels(predicted_class=_predicted)
    for _actual in CLASSES:
        FEEDBACK.labels(predicted_class=_predicted, actual_class=_actual)

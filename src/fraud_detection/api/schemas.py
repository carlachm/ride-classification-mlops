"""Request / response contract of the API (also the OpenAPI doc served at /docs).

A request is ONE complete ride: the IMU samples (accelerometer ax/ay/az, gyroscope rx/ry/rz)
in time order, one sample every 100 ms (10 Hz), as in the GBG500 dataset.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Order of the model outputs (and of the labels of GBG500_labels.csv)
CLASSES: list[str] = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
CHANNELS: list[str] = ["ax", "ay", "az", "rx", "ry", "rz"]
MIN_SAMPLES = 100  # 10 s of riding
MAX_SAMPLES = 20_000  # ~33 min; the longest ride of the dataset has ~4 800 samples

_example_sample = {"ax": -0.13, "ay": -9.15, "az": -2.27, "rx": -0.5, "ry": -0.1, "rz": 0.06}


class Sample(BaseModel):
    """One IMU measurement. ``null`` is accepted (sensor dropout) and handled by the model."""

    ax: float | None = Field(..., description="Acceleration x (m/s²)", allow_inf_nan=False)
    ay: float | None = Field(..., description="Acceleration y (m/s²)", allow_inf_nan=False)
    az: float | None = Field(..., description="Acceleration z (m/s²)", allow_inf_nan=False)
    rx: float | None = Field(..., description="Rotation rate x (°/s)", allow_inf_nan=False)
    ry: float | None = Field(..., description="Rotation rate y (°/s)", allow_inf_nan=False)
    rz: float | None = Field(..., description="Rotation rate z (°/s)", allow_inf_nan=False)


class Ride(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"ride_id": "demo-ride", "samples": [_example_sample] * MIN_SAMPLES}}
    )

    ride_id: str | None = Field(None, max_length=128, description="Your own identifier, echoed back")
    samples: list[Sample] = Field(
        ..., min_length=MIN_SAMPLES, max_length=MAX_SAMPLES, description="IMU samples in time order, 10 Hz"
    )


class Prediction(BaseModel):
    ride_id: str | None
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class")
    probabilities: dict[str, float] = Field(..., description="One probability per class, summing to 1")
    n_samples: int
    model_version: str
    ride_hash: str = Field(..., description="Fingerprint of the signal: send it back with the feedback")


class Feedback(BaseModel):
    """True label received after the fact (manual review of the ride)."""

    predicted_class: str
    actual_class: str
    ride_hash: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{32}$",  # also used as a file name by the retraining: nothing else is accepted
        description="`ride_hash` of the prediction. Without it the label cannot be used for retraining",
    )


class Health(BaseModel):
    status: str  # "ok" | "degraded"
    model_loaded: bool
    model_version: str
    classes: list[str]

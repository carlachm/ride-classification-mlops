import numpy as np
import pandas as pd
import torch

from fraud_detection.api.schemas import CHANNELS, CLASSES
from fraud_detection.model import (
    RideCNN,
    Standardizer,
    fill_missing,
    make_windows,
    predict_ride_proba,
    ride_to_windows,
)


def test_fill_missing_forward_then_backward():
    x = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    out = fill_missing(x)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[:, 0], [2.0, 2.0, 2.0])  # leading NaN back-filled
    np.testing.assert_array_equal(out[:, 1], [1.0, 1.0, 1.0])  # trailing NaN forward-filled


def test_fill_missing_empty_channel_becomes_zero():
    x = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=np.float32)
    assert (fill_missing(x)[:, 0] == 0).all()


def test_windows_cover_the_whole_ride():
    x = np.arange(1000 * 6, dtype=np.float32).reshape(1000, 6)
    w = make_windows(x, length=300, stride=150)
    assert w.shape[1:] == (6, 300)
    # last window is aligned on the end of the ride: no sample is lost
    np.testing.assert_array_equal(w[-1][:, -1], x[-1])


def test_short_ride_is_padded_to_one_window():
    w = make_windows(np.ones((120, 6), dtype=np.float32), length=300, stride=150)
    assert w.shape == (1, 6, 300)
    assert (w[0, :, 120:] == 0).all()


def test_standardizer_uses_training_statistics():
    rides = [np.array([[1.0] * 6, [3.0] * 6], dtype=np.float32)]
    scaler = Standardizer.fit(rides)
    np.testing.assert_allclose(scaler.mean, 2.0)
    np.testing.assert_allclose(scaler.transform(np.full((1, 6), 2.0, dtype=np.float32)), 0.0)


def test_cnn_outputs_one_logit_per_class():
    model = RideCNN(channels=(8, 16)).eval()
    out = model(torch.randn(4, len(CHANNELS), 300))
    assert out.shape == (4, len(CLASSES))


def test_ride_probabilities_sum_to_one_for_any_length():
    model = RideCNN(channels=(8, 16))
    scaler = Standardizer(np.zeros(6), np.ones(6))
    for n in (100, 300, 2349):  # API accepts 100 to 20 000 samples
        ride = np.random.default_rng(n).normal(size=(n, 6)).astype(np.float32)
        proba = predict_ride_proba(model, ride_to_windows(ride, scaler, 300, 150))
        assert proba.shape == (len(CLASSES),)
        assert np.isclose(proba.sum(), 1.0, atol=1e-5)


def test_pyfunc_follows_the_api_contract(tmp_path):
    """Input DataFrame (ax..rz, NaN allowed) -> ndarray (1, 5) summing to 1 (docs/api-contract.md)."""
    from types import SimpleNamespace

    from fraud_detection.model import RideClassifierPyfunc, save_artifacts

    model_cfg = {"channels": [8, 16], "kernel_size": 5, "dropout": 0.3}
    windows_cfg = {"length": 300, "stride": 150}
    paths = save_artifacts(
        RideCNN(channels=(8, 16)), Standardizer(np.zeros(6), np.ones(6)), model_cfg, windows_cfg, tmp_path
    )

    wrapper = RideClassifierPyfunc()
    wrapper.load_context(SimpleNamespace(artifacts=paths))
    ride = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 6)), columns=CHANNELS)
    ride.iloc[10:20, 2] = np.nan  # sensor dropout
    out = wrapper.predict(None, ride)
    assert out.shape == (1, len(CLASSES))
    assert np.isclose(out.sum(), 1.0, atol=1e-5)

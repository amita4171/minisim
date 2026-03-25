"""Tests for calibration transformer."""
from src.core.calibration import CalibrationTransformer


def test_unfitted_returns_identity():
    ct = CalibrationTransformer()
    assert ct.transform(0.5) == 0.5
    assert ct.transform(0.3) == 0.3


def test_platt_fit_improves():
    # Systematic bias: predictions are too low
    predictions = [0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8]
    outcomes =    [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    ct = CalibrationTransformer(method="platt")
    ct.fit(predictions, outcomes)

    assert ct.is_fitted
    # After fitting, 0.4 should be corrected upward (actual rate > 0.4)
    corrected = ct.transform(0.4)
    assert corrected > 0.4


def test_temperature_fit():
    predictions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    outcomes =    [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    ct = CalibrationTransformer(method="temperature")
    ct.fit(predictions, outcomes)
    assert ct.is_fitted
    assert ct.temperature != 1.0


def test_ece_computed():
    predictions = [0.2, 0.5, 0.8, 0.3, 0.7]
    outcomes =    [0.0, 1.0, 1.0, 0.0, 1.0]

    ct = CalibrationTransformer()
    ct.fit(predictions, outcomes)
    assert ct.ece is not None
    assert 0 <= ct.ece <= 1


def test_calibration_curve():
    predictions = [0.1, 0.15, 0.5, 0.55, 0.9, 0.95]
    outcomes =    [0.0, 0.0,  1.0, 1.0,  1.0, 1.0]

    ct = CalibrationTransformer()
    ct.fit(predictions, outcomes)

    curve = ct.calibration_curve
    assert len(curve) > 0


def test_transform_bounded():
    ct = CalibrationTransformer(method="platt")
    ct.fit([0.1, 0.5, 0.9] * 5, [0, 1, 1] * 5)

    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        corrected = ct.transform(p)
        assert 0 < corrected < 1, f"Corrected {p} -> {corrected} out of bounds"


def test_save_load(tmp_path):
    ct = CalibrationTransformer(method="platt")
    ct.fit([0.2, 0.5, 0.8] * 5, [0, 1, 1] * 5)

    path = str(tmp_path / "cal.json")
    ct.save(path)

    loaded = CalibrationTransformer.load(path)
    assert loaded.is_fitted
    import pytest
    assert loaded.platt_a == pytest.approx(ct.platt_a, abs=0.01)
    assert loaded.transform(0.5) == pytest.approx(ct.transform(0.5), abs=0.01)

import pandas as pd
import pytest

from decompose_prototype import detect_anomalies, event_components, regularity_index


def test_regularity_index_perfect_signal():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    observed = pd.Series(range(30), index=dates, dtype=float)
    resid = pd.Series(0.0, index=dates)
    assert regularity_index(resid, observed) == 1.0


def test_regularity_index_pure_noise():
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    observed = pd.Series([5.0, 12.0, 3.0, 20.0, 8.0, 15.0, 1.0], index=dates)
    resid = observed - observed.mean()  # même dispersion que l'observé
    assert regularity_index(resid, observed) == pytest.approx(0.0)


def test_regularity_index_clips_at_zero():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    observed = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0], index=dates)
    resid = pd.Series([100.0, -100.0, 100.0, -100.0, 100.0], index=dates)
    assert regularity_index(resid, observed) == 0.0


def test_detect_anomalies_flags_outlier():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    resid = pd.Series([0.0] * 19 + [50.0], index=dates)
    anomalies = detect_anomalies(resid, n_std=2.5)
    assert len(anomalies) == 1
    assert anomalies.index[0] == dates[-1]


def test_detect_anomalies_no_outliers():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    resid = pd.Series([0.1, -0.2, 0.15, -0.1, 0.05, -0.05, 0.2, -0.15, 0.1, -0.1], index=dates)
    assert len(detect_anomalies(resid, n_std=2.5)) == 0


def test_event_components_computes_average_effect():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    resid = pd.Series([10.0, 0.0, 10.0, 0.0, 0.0, 0.0], index=dates)
    flag = pd.Series([1, 0, 1, 0, 0, 0], index=dates)

    combined, effects = event_components(resid, {"ferie": flag})

    assert effects["ferie"] == pytest.approx(10.0)
    assert combined.tolist() == [10.0, 0.0, 10.0, 0.0, 0.0, 0.0]


def test_event_components_empty_flag_returns_zero_effect():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    resid = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
    flag = pd.Series([0, 0, 0, 0], index=dates)

    combined, effects = event_components(resid, {"ferie": flag})

    assert effects["ferie"] == 0.0
    assert (combined == 0).all()


def test_event_components_sums_multiple_flags():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    resid = pd.Series([10.0, 5.0, 0.0, 0.0], index=dates)
    ferie = pd.Series([1, 0, 0, 0], index=dates)
    vacances = pd.Series([0, 1, 0, 0], index=dates)

    combined, effects = event_components(resid, {"ferie": ferie, "vacances": vacances})

    assert effects["ferie"] == 10.0
    assert effects["vacances"] == 5.0
    assert combined.tolist() == [10.0, 5.0, 0.0, 0.0]

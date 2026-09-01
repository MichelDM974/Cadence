import pandas as pd
import pytest
from fastapi import HTTPException

from api import analyze, forecast, parse_csv


def make_csv_bytes(rows):
    lines = ["date,value"] + [f"{d},{v}" for d, v in rows]
    return "\n".join(lines).encode()


def test_parse_csv_builds_daily_series():
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 25)]  # 24 jours
    df = parse_csv(make_csv_bytes(rows), "date", "value")
    assert len(df) == 24
    assert list(df.columns) == ["visitors"]


def test_parse_csv_rejects_unknown_columns():
    content = make_csv_bytes([("2024-01-01", 10)])
    with pytest.raises(HTTPException):
        parse_csv(content, "nope", "value")


def test_parse_csv_detects_semicolon_delimiter():
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 25)]  # 24 jours
    lines = ["date;value"] + [f"{d};{v}" for d, v in rows]
    content = "\n".join(lines).encode()
    df = parse_csv(content, "date", "value")
    assert len(df) == 24
    assert list(df.columns) == ["visitors"]


def test_parse_csv_rejects_short_history():
    rows = [(f"2024-01-{d:02d}", 10) for d in range(1, 10)]  # 9 jours < 21
    with pytest.raises(HTTPException):
        parse_csv(make_csv_bytes(rows), "date", "value")


def _synthetic_series(n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    values = [20 + 5 * (i % 7) for i in range(n_days)]
    return pd.DataFrame({"visitors": values}, index=dates)


def test_analyze_without_flags_returns_expected_shape():
    df = _synthetic_series()
    result = analyze(df)
    assert len(result["dates"]) == len(df)
    assert len(result["trend"]) == len(df)
    assert "regularity_index" in result
    assert result["effects"] == {}
    assert result["weather"] is None


def test_analyze_with_flags_reports_effects():
    df = _synthetic_series()
    flag = pd.Series(0, index=df.index)
    flag.iloc[::10] = 1  # un jour sur dix marqué
    result = analyze(df, flags={"ferie": flag})
    assert "ferie" in result["effects"]


def test_forecast_returns_requested_horizon():
    df = _synthetic_series()
    result = forecast(df, horizon=10)
    assert len(result["dates"]) == 10
    assert len(result["prevision"]) == 10
    assert len(result["ic_bas"]) == 10
    assert len(result["ic_haut"]) == 10

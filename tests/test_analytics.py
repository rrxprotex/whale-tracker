"""Offline tests for the analytics engine (no network required)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.charts import build_charts
from src.analytics.engine import _json_safe
from src.analytics.forecast import forecast_ensemble
from src.analytics.indicators import compute_snapshot, indicator_frame
from src.analytics.patterns import pattern_analysis
from src.analytics.regime import regime_analysis
from src.analytics.report import render_report
from src.analytics.scoring import composite_score
from src.analytics.spectral import spectrum_analysis
from src.analytics.statistics import statistical_analysis
from src.analytics.symbols import resolve_symbol


def synthetic_frame(n: int = 720, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    t = np.arange(n)
    log_price = (
        4.2
        + 0.0006 * t
        + 0.10 * np.sin(2 * math.pi * t / 63.0)
        + 0.05 * np.sin(2 * math.pi * t / 21.0)
        + np.cumsum(rng.normal(0.0, 0.012, n))
    )
    close = np.exp(log_price)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.lognormal(10, 0.4, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": close * volume,
        },
        index=index,
    )


@pytest.fixture(scope="module")
def analysis_bundle():
    df = synthetic_frame()
    bpy = 365.0
    ind = indicator_frame(df, bpy)
    technical = compute_snapshot(df, ind, bpy)
    spectral = spectrum_analysis(df, bpy, 30)
    statistics = statistical_analysis(df, bpy)
    patterns = pattern_analysis(df, ind, bpy, float(ind["atr"].iloc[-1]))
    regime = regime_analysis(df, ind, bpy)
    forecast = forecast_ensemble(df, 30, bpy, "1d", "crypto")
    meta = {
        "symbol": "TESTUSDT",
        "name": "Synthetic",
        "market": "crypto",
        "currency": "USDT",
        "exchange": "test",
        "interval": "1d",
        "source": "synthetic",
        "bars": len(df),
        "last_price": float(df["close"].iloc[-1]),
        "first_timestamp": df.index[0].isoformat(),
        "last_timestamp": df.index[-1].isoformat(),
        "fetched_at": df.index[-1].isoformat(),
        "stale": False,
        "gaps": 0,
        "bars_per_year": bpy,
        "data_age_seconds": 0,
        "forecast_horizon": 30,
    }
    score = composite_score(technical, spectral, statistics, patterns, regime, forecast, meta)
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "meta": meta,
        "quality": {"score": 100, "label": "yüksek", "warnings": []},
        "technical": technical,
        "spectral": spectral,
        "statistics": statistics,
        "patterns": patterns,
        "regime": regime,
        "forecast": forecast,
        "score": score,
    }
    charts = build_charts(df, ind, report)
    return {"df": df, "ind": ind, "report": report, "charts": charts}


def test_indicator_bounds(analysis_bundle):
    ind = analysis_bundle["ind"]
    rsi = ind["rsi"].dropna()
    assert not rsi.empty
    assert rsi.min() >= 0 and rsi.max() <= 100
    assert ind["atr"].dropna().min() > 0
    assert "ema200" in ind.columns


def test_spectrum_detects_sixty_three_bar_cycle(analysis_bundle):
    spectral = analysis_bundle["report"]["spectral"]
    assert spectral["available"] is True
    assert spectral["dominant_cycle"] is not None
    period = spectral["dominant_cycle"]["period_bars"]
    assert 45 <= period <= 85
    assert spectral["harmonic"]
    assert len(spectral["ssa"]["projection_price"]) == 30


def test_forecast_bands_are_ordered(analysis_bundle):
    forecast = analysis_bundle["report"]["forecast"]
    assert forecast["available"] is True
    ensemble = forecast["ensemble"]
    assert ensemble["low95"] <= ensemble["low80"] <= ensemble["high80"] <= ensemble["high95"]
    assert ensemble["low80"] <= ensemble["price"] <= ensemble["high80"]
    assert 0 <= ensemble["up_probability"] <= 100
    assert len(forecast["horizons"]) >= 2
    assert forecast["confidence"]["score"] > 0


def test_report_has_all_charts(analysis_bundle):
    report = analysis_bundle["report"]
    charts = analysis_bundle["charts"]
    assert len(charts) == 10
    for key, svg in charts.items():
        assert "<svg" in svg, key
        assert "Grafik oluşturulamadı" not in svg, key
    html = render_report(report, charts)
    assert html.startswith("<!DOCTYPE html>")
    assert html.count("<svg") == 10
    assert "Yatırım tavsiyesi değildir" in html
    assert "Grafik oluşturulamadı" not in html


def test_json_safe_handles_numpy():
    payload = {
        "a": np.float64(1.5),
        "b": np.int64(2),
        "c": np.bool_(True),
        "d": float("nan"),
        "e": pd.Timestamp("2026-01-01"),
    }
    safe = _json_safe(payload)
    assert safe["a"] == 1.5
    assert safe["b"] == 2
    assert safe["c"] is True
    assert safe["d"] is None
    assert "2026-01-01" in safe["e"]


def test_symbol_resolution():
    assert resolve_symbol("BTCUSDT").market == "crypto"
    assert resolve_symbol("AAPL").market == "us"
    assert resolve_symbol("THYAO").market == "bist"


def test_api_health_and_ui():
    from fastapi.testclient import TestClient

    from src.analytics.api import app

    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    ui = client.get("/")
    assert ui.status_code == 200
    assert "WhaleTracker" in ui.text
    symbols = client.get("/api/symbols?q=btc")
    assert symbols.status_code == 200
    assert symbols.json()["results"]

"""Analysis engine: data -> methods -> score -> narrative -> charts -> report."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .charts import build_charts
from .forecast import forecast_ensemble
from .indicators import compute_snapshot, indicator_frame
from .market_data import BASE_DIR, MarketData, fetch_ohlcv
from .patterns import pattern_analysis
from .regime import regime_analysis
from .report import render_report
from .scoring import composite_score
from .spectral import spectrum_analysis
from .statistics import statistical_analysis

logger = logging.getLogger(__name__)

REPORTS_DIR = BASE_DIR / "reports"

_INTERVAL_SECONDS = {
    "1h": 3600,
    "4h": 4 * 3600,
    "1d": 86400,
    "1w": 7 * 86400,
}


def _json_safe(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "analysis"


def _bar_age_hours(md: MarketData) -> Optional[float]:
    try:
        last = pd.Timestamp(md.last_timestamp)
        now = pd.Timestamp.now(tz="UTC")
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        return max(0.0, (now - last).total_seconds() / 3600.0)
    except Exception:
        return None


def data_quality(md: MarketData) -> dict:
    meta = md.meta()
    warnings = []
    penalty = 0
    bars = meta.get("bars", 0)
    if bars < 120:
        warnings.append("Bar sayısı 120'nin altında; tahmin modelleri devre dışı kalabilir.")
        penalty += 30
    elif bars < 250:
        warnings.append("Bar sayısı 250'nin altında; uzun vadeli göstergeler ve walk-forward zayıf.")
        penalty += 14
    if meta.get("stale"):
        warnings.append("Veri eski önbellekten geldi; canlı fiyatla fark olabilir.")
        penalty += 22
    gaps = int(meta.get("gaps") or 0)
    if gaps > 0:
        ratio = gaps / max(bars - 1, 1) * 100.0
        warnings.append(f"{gaps} bar boşluğu tespit edildi (%{ratio:.2f}).")
        penalty += min(18, int(ratio * 4))
    age = _bar_age_hours(md)
    expected = _INTERVAL_SECONDS.get(md.interval, 86400) / 3600.0
    if age is not None and age > expected * 3.0 + 2.0:
        warnings.append(
            f"Son bar {age:.1f} saat eski (beklenen aralık {expected:.1f} saat); "
            "piyasa kapalı veya veri gecikmeli olabilir."
        )
        penalty += 12
    volume = md.df["volume"].astype(float)
    zero_share = float((volume <= 0).mean()) if len(volume) else 1.0
    if zero_share > 0.20:
        warnings.append(f"Barların %{zero_share * 100:.0f}'inde hacim sıfır; hacim analizleri güvenilmez.")
        penalty += 10
    quality_score = max(0, min(100, 100 - penalty))
    label = (
        "yüksek" if quality_score >= 85 else
        "orta" if quality_score >= 65 else
        "düşük" if quality_score >= 40 else "çok düşük"
    )
    return {
        "score": quality_score,
        "label": label,
        "warnings": warnings,
        "bar_age_hours": round(age, 2) if age is not None else None,
        "zero_volume_share_pct": round(zero_share * 100.0, 2),
    }


@dataclass
class AnalysisResult:
    report: dict
    charts: Dict[str, str]
    html: str
    html_path: Optional[Path] = None
    json_path: Optional[Path] = None

    def save(self, output_dir: Optional[Path] = None) -> "AnalysisResult":
        directory = Path(output_dir) if output_dir else REPORTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        meta = self.report.get("meta", {})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"{_safe_name(meta.get('symbol', 'analysis'))}_{_safe_name(meta.get('interval', '1d'))}_{stamp}"
        html_path = directory / f"{stem}.html"
        json_path = directory / f"{stem}.json"
        html_path.write_text(self.html, encoding="utf-8")
        payload = _json_safe(self.report)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self.html_path = html_path
        self.json_path = json_path
        return self


def analyze(
    symbol: str,
    market: str = "auto",
    interval: str = "1d",
    limit: int = 500,
    horizon: int = 30,
    use_cache: bool = True,
    save: bool = False,
    output_dir: Optional[Path] = None,
) -> AnalysisResult:
    """Run the full analysis pipeline for one symbol.

    Parameters
    ----------
    symbol:
        ``BTCUSDT``, ``ETH/USDT``, ``AAPL``, ``THYAO``, ...
    market:
        ``auto``, ``crypto``, ``us`` or ``bist``.
    interval:
        ``1h``, ``4h``, ``1d``, ``1w``.
    limit:
        Requested bars (80..5000). 500 is a good default.
    horizon:
        Forecast horizon in bars (1..365).
    """
    md = fetch_ohlcv(
        symbol=symbol, market=market, interval=interval,
        limit=limit, use_cache=use_cache,
    )
    df = md.df.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 80:
        raise ValueError("Analiz için en az 80 bar gerekli.")

    bars_per_year = md.bars_per_year
    ind = indicator_frame(df, bars_per_year=bars_per_year)
    technical = compute_snapshot(df, ind, bars_per_year=bars_per_year)
    spectral = spectrum_analysis(df, bars_per_year=bars_per_year, horizon=horizon)
    statistics = statistical_analysis(df, bars_per_year=bars_per_year)
    latest_atr = float(ind["atr"].iloc[-1]) if "atr" in ind and pd.notna(ind["atr"].iloc[-1]) else float(df["close"].iloc[-1]) * 0.02
    patterns = pattern_analysis(df, ind, bars_per_year, latest_atr)
    regime = regime_analysis(df, ind, bars_per_year)
    forecast = forecast_ensemble(
        df, horizon=horizon, bars_per_year=bars_per_year,
        interval=md.interval, market=md.market,
    )

    meta = md.meta()
    meta["forecast_horizon"] = int(horizon)
    quality = data_quality(md)
    meta["quality"] = quality

    score = composite_score(
        technical=technical,
        spectral=spectral,
        statistics=statistics,
        patterns=patterns,
        regime=regime,
        forecast=forecast,
        meta=meta,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "quality": quality,
        "technical": technical,
        "spectral": spectral,
        "statistics": statistics,
        "patterns": patterns,
        "regime": regime,
        "forecast": forecast,
        "score": score,
    }
    charts = build_charts(df, ind, report)
    html = render_report(report, charts)
    result = AnalysisResult(report=report, charts=charts, html=html)
    if save:
        result.save(output_dir)
    return result


def analyze_to_html(
    symbol: str,
    market: str = "auto",
    interval: str = "1d",
    limit: int = 500,
    horizon: int = 30,
    use_cache: bool = True,
    save: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[str, AnalysisResult]:
    result = analyze(
        symbol=symbol, market=market, interval=interval, limit=limit,
        horizon=horizon, use_cache=use_cache, save=save, output_dir=output_dir,
    )
    return result.html, result


def analysis_summary(result_or_report: Any) -> dict:
    """Compact summary suitable for terminals and API lists."""
    report = result_or_report.report if isinstance(result_or_report, AnalysisResult) else result_or_report
    meta = report.get("meta", {})
    score = report.get("score", {})
    forecast = report.get("forecast", {})
    ensemble = forecast.get("ensemble", {})
    return {
        "symbol": meta.get("symbol"),
        "name": meta.get("name"),
        "market": meta.get("market"),
        "interval": meta.get("interval"),
        "price": meta.get("last_price"),
        "currency": meta.get("currency"),
        "score": score.get("score"),
        "label": score.get("label"),
        "tone": score.get("tone"),
        "confidence": score.get("confidence"),
        "risk": score.get("risk", {}).get("level"),
        "forecast_price": ensemble.get("price"),
        "forecast_change_pct": ensemble.get("change_pct"),
        "forecast_low80": ensemble.get("low80"),
        "forecast_high80": ensemble.get("high80"),
        "up_probability_pct": ensemble.get("up_probability"),
        "quality": report.get("quality", {}).get("label"),
        "generated_at": report.get("generated_at"),
    }

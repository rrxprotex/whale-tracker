"""Advanced multi-method market analysis engine.

Modules
-------
market_data : unified OHLCV fetching (Binance / Bybit / Yahoo Finance) + cache
indicators  : technical indicators and snapshot scoring
spectral    : FFT / Welch / harmonic / Hilbert / Haar / SSA analysis
statistics  : long memory, mean reversion, risk and distribution statistics
patterns    : pivots, support/resistance, Fibonacci, volume profile, candles
regime      : trend/volatility regimes, change points, seasonality
forecast    : walk-forward validated multi-model ensemble + Monte Carlo
scoring     : explainable composite score and Turkish narrative
charts      : dependency-free standalone SVG charts
report      : standalone HTML report
engine      : high level orchestration
api         : FastAPI analysis server
"""

from .engine import AnalysisResult, analyze, analysis_summary
from .market_data import fetch_ohlcv
from .report import render_report

__version__ = "1.1.0"

__all__ = [
    "AnalysisResult",
    "analysis_summary",
    "analyze",
    "fetch_ohlcv",
    "render_report",
    "__version__",
]

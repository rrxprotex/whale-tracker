"""Dependency-free SVG charts.

Every chart is a standalone ``<svg>`` string that can be embedded in the HTML
report or returned by the API. No Plotly/Chart.js/CDN is required.
"""

from __future__ import annotations

import html
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


COLORS = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "grid": "#21262d",
    "text": "#c9d1d9",
    "muted": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
    "orange": "#f08833",
    "purple": "#a78bfa",
    "yellow": "#d29922",
    "cyan": "#39c5cf",
    "white": "#f0f6fc",
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _num(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 1:
        return f"{value:,.{digits}f}"
    return f"{value:,.4f}"


def _px(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:,.3f}"
    return f"{value:,.5f}"


class Svg:
    def __init__(self, width: int, height: int, label: str = "chart"):
        self.width = width
        self.height = height
        self.label = label
        self.parts: List[str] = []
        self.defs: List[str] = []

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x, y, w, h, fill, stroke="none", rx=0, opacity=1.0, extra=""):
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0.0, w):.2f}" '
            f'height="{max(0.0, h):.2f}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" opacity="{opacity}" {extra}/>'
        )

    def line(self, x1, y1, x2, y2, stroke, width=1.0, dash=None, opacity=1.0):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'
        )

    def path(self, d, stroke="none", fill="none", width=1.5, dash=None, opacity=1.0):
        if not d:
            return
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" stroke="{stroke}" fill="{fill}" stroke-width="{width}" '
            f'opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )

    def circle(self, x, y, r, fill, stroke="none", width=1.0, opacity=1.0):
        self.add(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
        )

    def text(
        self, x, y, value, size=11, fill=None, anchor="start",
        weight="normal", opacity=1.0, mono=False, extra="",
    ):
        family = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" if mono else FONT
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" '
            f'font-size="{size}" fill="{fill or COLORS["text"]}" '
            f'text-anchor="{anchor}" font-weight="{weight}" opacity="{opacity}" '
            f'style="font-variant-numeric:tabular-nums"{extra}>{_esc(value)}</text>'
        )

    def polygon(self, points: Sequence[Tuple[float, float]], fill, opacity=1.0, stroke="none"):
        if not points:
            return
        values = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(
            f'<polygon points="{values}" fill="{fill}" opacity="{opacity}" stroke="{stroke}"/>'
        )

    def render(self) -> str:
        defs = "<defs>" + "".join(self.defs) + "</defs>" if self.defs else ""
        return (
            f'<svg viewBox="0 0 {self.width} {self.height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="{_esc(self.label)}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="{FONT}">'
            + defs
            + "".join(self.parts)
            + "</svg>"
        )

    def gradient(self, ident: str, color: str, opacity_top=0.35, opacity_bottom=0.0):
        self.defs.append(
            f'<linearGradient id="{ident}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="{opacity_top}"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="{opacity_bottom}"/>'
            f"</linearGradient>"
        )


def _empty(message: str, width: int = 560, height: int = 260) -> str:
    svg = Svg(width, height, message)
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    svg.text(width / 2, height / 2, message, 12, COLORS["muted"], "middle")
    return svg.render()


def _scale(domain_low: float, domain_high: float, range_low: float, range_high: float):
    if not math.isfinite(domain_low) or not math.isfinite(domain_high):
        domain_low, domain_high = 0.0, 1.0
    if abs(domain_high - domain_low) < 1e-12:
        domain_high = domain_low + 1.0
    return lambda value: range_low + (
        (value - domain_low) / (domain_high - domain_low)
    ) * (range_high - range_low)


def _nice_num(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 2.5:
        nice = 2.5
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10 ** exponent)


def _ticks(low: float, high: float, count: int = 5) -> List[float]:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return [low]
    step = _nice_num((high - low) / max(count, 1))
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.01 and len(ticks) < 20:
        ticks.append(value)
        value += step
    return ticks


def _path(xs: Iterable[float], ys: Iterable[Optional[float]]) -> str:
    commands = []
    started = False
    for x, y in zip(xs, ys):
        if y is None or not math.isfinite(y):
            started = False
            continue
        commands.append(f"{'M' if not started else 'L'}{x:.2f},{y:.2f}")
        started = True
    return " ".join(commands)


def _area_path(
    xs: Sequence[float], upper: Sequence[float], lower: Sequence[float]
) -> str:
    upper_path = _path(xs, upper)
    lower_path = _path(list(reversed(xs)), list(reversed(lower)))
    if not upper_path or not lower_path:
        return ""
    # Replace the first M of the lower path with L to connect the polygon.
    lower_path = "L" + lower_path[1:] if lower_path.startswith("M") else lower_path
    return upper_path + " " + lower_path + " Z"


def _date_label(timestamp, intraday: bool = False) -> str:
    try:
        value = pd.Timestamp(timestamp)
    except Exception:
        return ""
    if intraday:
        return value.strftime("%d.%m %H:%M")
    return value.strftime("%d.%m.%y")


def _future_dates(df: pd.DataFrame, interval: str, market: str, h: int) -> List[pd.Timestamp]:
    last = pd.Timestamp(df.index[-1])
    if interval == "1d" and market in ("us", "bist"):
        return list(pd.bdate_range(last + pd.Timedelta(days=1), periods=h))
    if interval == "1d":
        return list(pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D"))
    if interval == "1w":
        return list(pd.date_range(last + pd.Timedelta(days=7), periods=h, freq="7D"))
    if interval == "4h":
        return list(pd.date_range(last + pd.Timedelta(hours=4), periods=h, freq="4h"))
    if interval == "1h":
        return list(pd.date_range(last + pd.Timedelta(hours=1), periods=h, freq="h"))
    return list(pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D"))


def _legend(svg: Svg, items: List[Tuple[str, str, Optional[str]]], x: float, y: float):
    cursor = x
    for label, color, dash in items:
        svg.line(cursor, y - 3, cursor + 14, y - 3, color, 2.0, dash=dash)
        svg.text(cursor + 18, y, label, 10, COLORS["muted"])
        cursor += 18 + len(label) * 5.4 + 14


# ---------------------------------------------------------------------------
# Price + forecast cone
# ---------------------------------------------------------------------------

def price_chart(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    forecast: dict,
    patterns: dict,
    meta: dict,
) -> str:
    if df is None or len(df) < 20:
        return _empty("Fiyat grafiği için yeterli veri yok", 1120, 620)
    window = min(len(df), 210)
    hist = df.iloc[-window:]
    hist_ind = ind.iloc[-window:]
    current = float(df["close"].iloc[-1])
    interval = meta.get("interval", "1d")
    market = meta.get("market", "crypto")
    intraday = interval in ("1h", "4h")
    h = min(int(forecast.get("horizon") or 0), 120) if forecast.get("available") else 0
    bands = forecast.get("bands") or {}
    future_dates = _future_dates(df, interval, market, h) if h else []

    width, height = 1120, 620
    left, right, top = 66, 104, 38
    price_top, price_bottom = 54, 424
    volume_top, volume_bottom = 456, 548
    chart_right = width - right
    total = window + h
    step = (chart_right - left) / max(total, 1)
    xs = [left + (index + 0.5) * step for index in range(total)]
    all_dates = list(hist.index) + future_dates

    hist_low = float(hist["low"].min())
    hist_high = float(hist["high"].max())
    span = max(hist_high - hist_low, current * 0.02)
    candidates = [hist_low, hist_high]
    for key in ("q05", "q95"):
        values = bands.get(key) or []
        if values:
            candidates.extend([float(values[0]), float(values[-1])])
    support = (patterns.get("key_levels") or {}).get("nearest_support") or {}
    resistance = (patterns.get("key_levels") or {}).get("nearest_resistance") or {}
    for level in (support, resistance):
        value = _num(level.get("price"))
        if value and abs(value - current) < span * 2:
            candidates.append(value)
    low = min(candidates)
    high = max(candidates)
    low = max(low, hist_low - span * 1.2)
    high = min(high, hist_high + span * 1.2)
    if high <= low:
        low, high = hist_low, hist_high
    y_price = _scale(low, high, price_bottom, price_top)
    volume_max = float(hist["volume"].max()) if len(hist) else 1.0
    y_volume = _scale(0.0, max(volume_max, 1e-9), volume_bottom, volume_top)

    svg = Svg(width, height, "Fiyat, göstergeler ve tahmin konisi")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=12)

    # Grid + price axis
    for tick in _ticks(low, high, 6):
        y = y_price(tick)
        svg.line(left, y, chart_right, y, COLORS["grid"], 0.8)
        svg.text(left - 10, y + 4, _px(tick), 10, COLORS["muted"], "end", mono=True)
    svg.line(left, price_bottom, chart_right, price_bottom, COLORS["grid"], 1.0)
    svg.line(left, volume_top, chart_right, volume_top, COLORS["grid"], 0.8)

    # Bollinger area
    if "bb_upper" in hist_ind and "bb_lower" in hist_ind:
        upper = [
            y_price(value) if pd.notna(value) else None
            for value in hist_ind["bb_upper"].values
        ]
        lower = [
            y_price(value) if pd.notna(value) else None
            for value in hist_ind["bb_lower"].values
        ]
        area = _area_path(xs[:window], upper, lower)
        if area:
            svg.path(area, fill=COLORS["blue"], opacity=0.055)

    # Forecast cone
    if h and bands:
        q95 = [current] + [float(value) for value in bands.get("q95", [])]
        q05 = [current] + [float(value) for value in bands.get("q05", [])]
        q75 = [current] + [float(value) for value in bands.get("q75", [])]
        q25 = [current] + [float(value) for value in bands.get("q25", [])]
        median = [current] + [float(value) for value in bands.get("q50", [])]
        forecast_xs = xs[window - 1:window + h]
        if len(forecast_xs) == len(median):
            outer = _area_path(
                forecast_xs,
                [y_price(value) for value in q95],
                [y_price(value) for value in q05],
            )
            inner = _area_path(
                forecast_xs,
                [y_price(value) for value in q75],
                [y_price(value) for value in q25],
            )
            if outer:
                svg.path(outer, fill=COLORS["green"], opacity=0.10)
            if inner:
                svg.path(inner, fill=COLORS["green"], opacity=0.16)
            svg.path(
                _path(forecast_xs, [y_price(value) for value in median]),
                stroke=COLORS["green"], width=2.0, dash="5 4",
            )
        split_x = xs[window - 1] if window - 1 < len(xs) else chart_right
        svg.line(split_x, price_top, split_x, volume_bottom, COLORS["muted"], 1.0, dash="4 5", opacity=0.7)
        svg.text(split_x + 6, price_top + 12, "PROJEKSİYON", 9, COLORS["green"], weight="bold")

    # Support / resistance
    for level, color, label in (
        (support, COLORS["green"], "destek"),
        (resistance, COLORS["red"], "direnç"),
    ):
        value = _num(level.get("price"))
        if value and low <= value <= high:
            y = y_price(value)
            svg.line(left, y, chart_right, y, color, 1.0, dash="2 5", opacity=0.55)
            svg.text(chart_right + 6, y + 3, f"{label} {_px(value)}", 9, color, opacity=0.9)

    # Candles
    candle_width = max(1.2, step * 0.62)
    for index, (_, row) in enumerate(hist.iterrows()):
        open_ = float(row["open"])
        close = float(row["close"])
        high = float(row["high"])
        low_value = float(row["low"])
        color = COLORS["green"] if close >= open_ else COLORS["red"]
        x = xs[index]
        svg.line(x, y_price(high), x, y_price(low_value), color, 0.9)
        body_top = y_price(max(open_, close))
        body_bottom = y_price(min(open_, close))
        svg.rect(
            x - candle_width / 2, body_top, candle_width,
            max(0.9, body_bottom - body_top), color,
        )

    # EMA overlays
    for column, color, label, dash in (
        ("ema20", COLORS["blue"], "EMA20", None),
        ("ema50", COLORS["orange"], "EMA50", None),
        ("ema200", COLORS["purple"], "EMA200", "4 3"),
    ):
        if column in hist_ind:
            values = [
                y_price(value) if pd.notna(value) else None
                for value in hist_ind[column].values
            ]
            svg.path(_path(xs[:window], values), stroke=color, width=1.2, opacity=0.95, dash=dash)

    # Volume
    for index, (_, row) in enumerate(hist.iterrows()):
        volume = float(row["volume"])
        color = COLORS["green"] if float(row["close"]) >= float(row["open"]) else COLORS["red"]
        y = y_volume(volume)
        svg.rect(
            xs[index] - candle_width / 2, y, candle_width,
            max(0.6, volume_bottom - y), color, opacity=0.35,
        )

    # X labels
    label_indices = np.linspace(0, total - 1, min(7, total)).astype(int)
    for index in label_indices:
        x = xs[index]
        svg.line(x, volume_bottom, x, volume_bottom + 4, COLORS["grid"], 1.0)
        svg.text(
            x, volume_bottom + 18, _date_label(all_dates[index], intraday),
            9, COLORS["muted"], "middle",
        )

    # Last price badge
    y_last = y_price(current)
    badge_y = min(max(y_last - 11, price_top), price_bottom - 24)
    svg.rect(width - 100, badge_y, 92, 22, COLORS["green"], rx=4)
    svg.text(width - 54, badge_y + 15, _px(current), 11, "#08120a", "middle", "bold", mono=True)

    # Legend
    _legend(
        svg,
        [("EMA20", COLORS["blue"], None), ("EMA50", COLORS["orange"], None),
         ("EMA200", COLORS["purple"], "4 3"), ("Bollinger", COLORS["blue"], None),
         ("Ensemble", COLORS["green"], "5 4")],
        left + 4, 22,
    )
    svg.text(
        left, height - 10,
        f"{meta.get('symbol', '')} · {interval} · {meta.get('source', '')} · "
        f"{len(df)} bar · son { _date_label(df.index[-1], intraday) }",
        9.5, COLORS["muted"],
    )
    svg.text(
        chart_right, height - 10,
        "Fiyat ekseni doğrusal; projeksiyon %80/%95 MC + walk-forward zarfı",
        9.5, COLORS["muted"], "end",
    )
    return svg.render()


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def spectrum_chart(spectral: dict) -> str:
    data = spectral.get("spectrum") or {}
    periods = [value for value in data.get("periods", []) if value and value > 0]
    powers = data.get("relative_power", [])
    if not periods or not powers or len(periods) != len(powers):
        return _empty("Spektrum verisi yok")
    width, height = 560, 330
    left, right, top, bottom = 54, 20, 24, 44
    svg = Svg(width, height, "Fourier güç spektrumu")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)

    log_periods = np.log(periods)
    x_low, x_high = float(log_periods.min()), float(log_periods.max())
    x = _scale(x_low, x_high, left, width - right)
    y = _scale(0.0, 100.0, height - bottom, top)
    for tick in (0, 25, 50, 75, 100):
        yy = y(tick)
        svg.line(left, yy, width - right, yy, COLORS["grid"], 0.8)
        svg.text(left - 8, yy + 4, f"{tick}", 9, COLORS["muted"], "end", mono=True)

    points_x = [x(value) for value in log_periods]
    points_y = [y(value) for value in powers]
    area = _path(points_x, points_y)
    if area:
        first = points_x[0]
        last = points_x[-1]
        svg.path(
            area + f" L{last:.2f},{y(0):.2f} L{first:.2f},{y(0):.2f} Z",
            fill=COLORS["cyan"], opacity=0.12,
        )
        svg.path(area, stroke=COLORS["cyan"], width=1.4)

    top_periods = data.get("top_periods", [])
    for period in top_periods[:5]:
        nearest = min(range(len(periods)), key=lambda index: abs(periods[index] - period))
        px, py = points_x[nearest], points_y[nearest]
        svg.circle(px, py, 3.2, COLORS["green"], COLORS["panel"], 1.0)
        svg.text(px, max(top + 10, py - 7), f"{period:.0f}", 9, COLORS["green"], "middle", "bold")

    ticks = [value for value in (5, 10, 20, 50, 100, 200, 500, 1000)
             if x_low <= math.log(value) <= x_high]
    for value in ticks:
        xx = x(math.log(value))
        svg.line(xx, height - bottom, xx, height - bottom + 4, COLORS["grid"], 1.0)
        svg.text(xx, height - bottom + 17, str(value), 9, COLORS["muted"], "middle")
    svg.text(width / 2, height - 8, "Periyot (bar, log ölçek)", 10, COLORS["muted"], "middle")
    svg.text(left, top - 8, "Göreli güç (%)", 9.5, COLORS["muted"])
    return svg.render()


def harmonic_chart(df: pd.DataFrame, spectral: dict, meta: dict) -> str:
    harmonic = spectral.get("harmonic") or {}
    fitted = harmonic.get("fitted") or []
    projection = harmonic.get("projection") or {}
    if not fitted or not projection.get("price"):
        return _empty("Harmonik model uygun değil")
    width, height = 1120, 300
    left, right, top, bottom = 64, 92, 28, 36
    svg = Svg(width, height, "Harmonik model ve projeksiyon")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    history = df.iloc[-min(len(df), len(fitted)):]
    actual = history["close"].astype(float).values
    fitted_values = [item["price"] for item in fitted[-len(history):]]
    projection_values = projection["price"]
    h = len(projection_values)
    total = len(actual) + h
    step = (width - left - right) / max(total, 1)
    xs = [left + (index + 0.5) * step for index in range(total)]
    future_dates = _future_dates(df, meta.get("interval", "1d"), meta.get("market", "crypto"), h)
    all_dates = list(history.index) + future_dates

    low = min(min(actual), min(fitted_values), min(projection_values))
    high = max(max(actual), max(fitted_values), max(projection_values))
    span = max(high - low, high * 0.02)
    low -= span * 0.08
    high += span * 0.08
    y = _scale(low, high, height - bottom, top)
    for tick in _ticks(low, high, 4):
        yy = y(tick)
        svg.line(left, yy, width - right, yy, COLORS["grid"], 0.8)
        svg.text(left - 8, yy + 4, _px(tick), 9, COLORS["muted"], "end", mono=True)

    split = xs[len(actual) - 1]
    svg.line(split, top, split, height - bottom, COLORS["muted"], 1.0, dash="4 5", opacity=0.7)
    svg.path(_path(xs[:len(actual)], [y(value) for value in actual]), stroke=COLORS["text"], width=1.5)
    svg.path(
        _path(xs[:len(fitted_values)], [y(value) for value in fitted_values]),
        stroke=COLORS["blue"], width=1.3, dash="4 3",
    )
    proj_xs = xs[len(actual) - 1:]
    proj_ys = [actual[-1]] + [float(value) for value in projection_values]
    svg.path(_path(proj_xs, [y(value) for value in proj_ys]), stroke=COLORS["green"], width=1.6, dash="5 4")
    for key, opacity in (("high95", 0.08), ("high80", 0.12)):
        upper = [actual[-1]] + [float(value) for value in projection.get(key, [])]
        lower_key = "low95" if key == "high95" else "low80"
        lower = [actual[-1]] + [float(value) for value in projection.get(lower_key, [])]
        if len(upper) == len(proj_xs) and len(lower) == len(proj_xs):
            area = _area_path(proj_xs, [y(value) for value in upper], [y(value) for value in lower])
            if area:
                svg.path(area, fill=COLORS["green"], opacity=opacity)

    label_indices = np.linspace(0, total - 1, 6).astype(int)
    for index in label_indices:
        svg.text(xs[index], height - 12, _date_label(all_dates[index], meta.get("interval") in ("1h", "4h")), 9, COLORS["muted"], "middle")
    _legend(
        svg,
        [("Gerçek", COLORS["text"], None), ("Harmonik fit", COLORS["blue"], "4 3"),
         ("Projeksiyon", COLORS["green"], "5 4")],
        left + 4, 18,
    )
    svg.text(width - right + 6, 18, f"R² {harmonic.get('r2', '—')}", 10, COLORS["muted"], "start", mono=True)
    return svg.render()


# ---------------------------------------------------------------------------
# Monte Carlo distribution
# ---------------------------------------------------------------------------

def distribution_chart(forecast: dict, meta: dict) -> str:
    histogram = (forecast.get("monte_carlo") or {}).get("histogram") or {}
    prices = histogram.get("prices") or []
    counts = histogram.get("counts_pct") or []
    if not prices or not counts:
        return _empty("Monte Carlo dağılımı yok")
    width, height = 560, 300
    left, right, top, bottom = 52, 22, 26, 40
    svg = Svg(width, height, "Monte Carlo terminal fiyat dağılımı")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    low, high = min(prices), max(prices)
    span = max(high - low, high * 0.01)
    low -= span * 0.02
    high += span * 0.02
    x = _scale(low, high, left, width - right)
    y = _scale(0.0, max(counts) * 1.12, height - bottom, top)
    for tick in _ticks(0.0, max(counts) * 1.12, 4):
        yy = y(tick)
        svg.line(left, yy, width - right, yy, COLORS["grid"], 0.8)
        svg.text(left - 8, yy + 4, f"{tick:.0f}%", 9, COLORS["muted"], "end", mono=True)
    bin_width = (width - left - right) / max(len(prices), 1) * 0.72
    current = forecast.get("current_price")
    median = (forecast.get("monte_carlo") or {}).get("terminal", {}).get("q50")
    for price, count in zip(prices, counts):
        xx = x(price)
        color = COLORS["green"] if median and price >= median else COLORS["red"]
        svg.rect(xx - bin_width / 2, y(count), bin_width, max(0.5, y(0) - y(count)), color, opacity=0.55)
    if current:
        xx = x(current)
        svg.line(xx, top, xx, height - bottom, COLORS["white"], 1.0, dash="3 4")
        svg.text(xx, top - 6, "şimdi", 9, COLORS["white"], "middle")
    if median:
        xx = x(median)
        svg.line(xx, top, xx, height - bottom, COLORS["cyan"], 1.2)
        svg.text(xx, height - bottom + 14, f"medyan {_px(median)}", 9, COLORS["cyan"], "middle")
    svg.text(width / 2, height - 8, "Ufuk sonundaki fiyat dağılımı", 10, COLORS["muted"], "middle")
    return svg.render()


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown_chart(statistics: dict) -> str:
    series = statistics.get("drawdown_series") or []
    if len(series) < 3:
        return _empty("Drawdown serisi yok")
    width, height = 560, 250
    left, right, top, bottom = 48, 20, 22, 38
    svg = Svg(width, height, "Drawdown eğrisi")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    values = [float(item["drawdown_pct"]) for item in series]
    low = min(values + [0.0])
    span = max(abs(low), 1.0)
    x = _scale(0, len(values) - 1, left, width - right)
    y = _scale(low, 0.0, height - bottom, top)
    for tick in _ticks(low, 0, 4):
        yy = y(tick)
        svg.line(left, yy, width - right, yy, COLORS["grid"], 0.8)
        svg.text(left - 8, yy + 4, f"{tick:.0f}%", 9, COLORS["muted"], "end", mono=True)
    xs = [x(index) for index in range(len(values))]
    ys = [y(value) for value in values]
    path = _path(xs, ys)
    if path:
        svg.path(
            path + f" L{xs[-1]:.2f},{y(0):.2f} L{xs[0]:.2f},{y(0):.2f} Z",
            fill=COLORS["red"], opacity=0.18,
        )
        svg.path(path, stroke=COLORS["red"], width=1.3)
    svg.line(left, y(0), width - right, y(0), COLORS["muted"], 1.0)
    svg.text(left, top - 6, f"Maks. düşüş {statistics.get('drawdown', {}).get('max_drawdown_pct', '—')}%", 10, COLORS["red"])
    for index in np.linspace(0, len(series) - 1, 5).astype(int):
        svg.text(xs[index], height - 12, _date_label(series[index]["date"]), 9, COLORS["muted"], "middle")
    return svg.render()


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------

def seasonality_chart(regime: dict) -> str:
    rows = (regime.get("seasonality") or {}).get("monthly") or []
    if not rows:
        return _empty("Aylık sezonluk veri yok")
    width, height = 560, 290
    left, right, top, bottom = 44, 18, 26, 42
    svg = Svg(width, height, "Aylık ortalama getiri")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    values = [float(row["mean_pct"]) for row in rows]
    low = min(values + [0.0])
    high = max(values + [0.0])
    span = max(high - low, 1.0)
    low -= span * 0.12
    high += span * 0.12
    y = _scale(low, high, height - bottom, top)
    for tick in _ticks(low, high, 4):
        yy = y(tick)
        svg.line(left, yy, width - right, yy, COLORS["grid"], 0.8)
        svg.text(left - 8, yy + 4, f"{tick:.1f}%", 9, COLORS["muted"], "end", mono=True)
    zero = y(0)
    svg.line(left, zero, width - right, zero, COLORS["muted"], 1.0)
    slot = (width - left - right) / max(len(rows), 1)
    bar_width = slot * 0.58
    for index, row in enumerate(rows):
        value = float(row["mean_pct"])
        xx = left + slot * (index + 0.5)
        yy = y(value)
        color = COLORS["green"] if value >= 0 else COLORS["red"]
        svg.rect(xx - bar_width / 2, min(yy, zero), bar_width, max(1.0, abs(zero - yy)), color, opacity=0.75)
        svg.text(xx, height - bottom + 15, row["name"][:3], 9, COLORS["muted"], "middle")
        svg.text(xx, yy - 5 if value >= 0 else yy + 12, f"{value:+.1f}", 8.5, color, "middle", mono=True)
    svg.text(width / 2, height - 8, "Aylık ortalama getiri (%, tüm örnek)", 10, COLORS["muted"], "middle")
    return svg.render()


# ---------------------------------------------------------------------------
# Radar + gauge
# ---------------------------------------------------------------------------

def radar_chart(score: dict) -> str:
    contributions = score.get("contributions") or []
    if len(contributions) < 3:
        return _empty("Radar için yeterli bileşen yok", 380, 360)
    width, height = 380, 360
    cx, cy, radius = 190, 180, 118
    svg = Svg(width, height, "Skor bileşenleri radarı")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    n = len(contributions)
    angles = [-math.pi / 2 + index * 2 * math.pi / n for index in range(n)]

    for fraction in (0.25, 0.5, 0.75, 1.0):
        points = [
            (cx + radius * fraction * math.cos(angle), cy + radius * fraction * math.sin(angle))
            for angle in angles
        ]
        svg.polygon(points, "none", stroke=COLORS["grid"], opacity=0.9)
    for angle in angles:
        svg.line(cx, cy, cx + radius * math.cos(angle), cy + radius * math.sin(angle), COLORS["grid"], 0.8)

    points = []
    for angle, item in zip(angles, contributions):
        value = max(-1.0, min(1.0, float(item.get("score") or 0.0)))
        r = radius * (0.5 + 0.5 * value)
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    svg.polygon(points, COLORS["blue"], opacity=0.20, stroke=COLORS["blue"])
    for point in points:
        svg.circle(point[0], point[1], 2.4, COLORS["blue"])

    for angle, item in zip(angles, contributions):
        label = item["label"]
        value = float(item.get("score") or 0.0)
        lx = cx + (radius + 26) * math.cos(angle)
        ly = cy + (radius + 26) * math.sin(angle)
        anchor = "middle"
        if math.cos(angle) > 0.45:
            anchor = "start"
        elif math.cos(angle) < -0.45:
            anchor = "end"
        svg.text(lx, ly, label, 9.5, COLORS["text"], anchor)
        svg.text(lx, ly + 11, f"{value:+.2f}", 9, COLORS["muted"], anchor, mono=True)
    svg.circle(cx, cy, 2.0, COLORS["muted"])
    svg.text(cx, height - 12, "0.50 halkası nötr; dış halka +1", 9, COLORS["muted"], "middle")
    return svg.render()


def gauge_chart(score: dict) -> str:
    width, height = 380, 240
    cx, cy, radius = 190, 184, 128
    value = max(-100.0, min(100.0, float(score.get("score") or 0.0)))
    svg = Svg(width, height, "Bileşik skor göstergesi")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)

    # coloured arc as many small segments
    for step in range(100):
        start = math.pi - step / 100.0 * math.pi
        end = math.pi - (step + 1) / 100.0 * math.pi
        fraction = step / 100.0
        color = (
            COLORS["red"] if fraction < 0.35 else
            COLORS["yellow"] if fraction < 0.52 else
            COLORS["green"]
        )
        x1 = cx + radius * math.cos(start)
        y1 = cy - radius * math.sin(start)
        x2 = cx + (radius - 12) * math.cos(end)
        y2 = cy - (radius - 12) * math.sin(end)
        svg.polygon(
            [
                (x1, y1),
                (cx + (radius - 12) * math.cos(start), cy - (radius - 12) * math.sin(start)),
                (x2, y2),
            ],
            color, opacity=0.85,
        )

    angle = math.pi - (value + 100.0) / 200.0 * math.pi
    nx = cx + (radius - 26) * math.cos(angle)
    ny = cy - (radius - 26) * math.sin(angle)
    svg.line(cx, cy, nx, ny, COLORS["white"], 3.0)
    svg.circle(cx, cy, 6, COLORS["white"])
    svg.circle(cx, cy, 2.5, COLORS["panel"])

    for tick in (-100, -50, 0, 50, 100):
        tick_angle = math.pi - (tick + 100.0) / 200.0 * math.pi
        tx = cx + (radius + 12) * math.cos(tick_angle)
        ty = cy - (radius + 12) * math.sin(tick_angle) + 4
        svg.text(tx, ty, str(tick), 9, COLORS["muted"], "middle", mono=True)

    svg.text(cx, cy - 42, f"{value:+.0f}", 38, COLORS["white"], "middle", "bold")
    svg.text(cx, cy - 20, score.get("label", ""), 11, COLORS["muted"], "middle")
    svg.text(cx, height - 10, f"güven %{score.get('confidence', 0):.0f} · risk {score.get('risk', {}).get('level', '—')}", 9.5, COLORS["muted"], "middle")
    return svg.render()


# ---------------------------------------------------------------------------
# Regime strip + ACF
# ---------------------------------------------------------------------------

_REGIME_COLORS = {
    "güçlü boğa": COLORS["green"],
    "boğa": "#2ea043",
    "yatay": COLORS["yellow"],
    "ayı": "#da3633",
    "güçlü ayı": COLORS["red"],
    "bilinmiyor": COLORS["muted"],
}


def regime_strip(regime: dict, meta: dict) -> str:
    segments = regime.get("segments") or []
    if not segments:
        return _empty("Rejim segmenti yok", 1120, 90)
    width, height = 1120, 92
    left, right, top = 60, 60, 26
    svg = Svg(width, height, "Rejim zaman çizelgesi")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    start = pd.Timestamp(segments[0]["start"])
    end = pd.Timestamp(segments[-1]["end"])
    x = _scale(start.value, max(end.value, start.value + 1), left, width - right)
    for segment in segments:
        x1 = x(pd.Timestamp(segment["start"]).value)
        x2 = x(pd.Timestamp(segment["end"]).value)
        color = _REGIME_COLORS.get(segment["label"], COLORS["muted"])
        svg.rect(x1, top + 8, max(1.0, x2 - x1), 32, color, rx=3, opacity=0.80)
        if x2 - x1 > 92:
            svg.text((x1 + x2) / 2, top + 28, segment["label"], 9.5, "#08120a", "middle", "bold")
    svg.text(left, top, "Trend rejimi (son 260 bar)", 9.5, COLORS["muted"])
    for index in np.linspace(0, len(segments) - 1, 5).astype(int):
        segment = segments[index]
        svg.text(
            x(pd.Timestamp(segment["start"]).value), height - 10,
            _date_label(segment["start"]), 9, COLORS["muted"], "middle",
        )
    return svg.render()


def acf_chart(statistics: dict) -> str:
    mean_reversion = statistics.get("mean_reversion") or {}
    rows = mean_reversion.get("autocorrelation") or []
    if len(rows) < 3:
        return _empty("Otokorelasyon verisi yok")
    width, height = 560, 240
    left, right, top, bottom = 44, 18, 24, 38
    svg = Svg(width, height, "Getiri otokorelasyonu")
    svg.rect(0, 0, width, height, COLORS["panel"], rx=10)
    observations = (statistics.get("return_stats") or {}).get("observations") or 100
    band = 1.96 / math.sqrt(max(observations, 1))
    low = min(-0.25, min(row["rho"] for row in rows) - 0.05)
    high = max(0.25, max(row["rho"] for row in rows) + 0.05)
    y = _scale(low, high, height - bottom, top)
    for tick in _ticks(low, high, 4):
        yy = y(tick)
        svg.line(left, yy, width - right, yy, COLORS["grid"], 0.8)
        svg.text(left - 8, yy + 4, f"{tick:+.2f}", 9, COLORS["muted"], "end", mono=True)
    zero = y(0)
    svg.line(left, zero, width - right, zero, COLORS["muted"], 1.0)
    svg.line(left, y(band), width - right, y(band), COLORS["yellow"], 1.0, dash="3 4")
    svg.line(left, y(-band), width - right, y(-band), COLORS["yellow"], 1.0, dash="3 4")
    slot = (width - left - right) / len(rows)
    bar_width = slot * 0.55
    for index, row in enumerate(rows):
        rho = float(row["rho"])
        xx = left + slot * (index + 0.5)
        color = COLORS["green"] if rho >= 0 else COLORS["red"]
        svg.rect(xx - bar_width / 2, min(zero, y(rho)), bar_width, max(1.0, abs(y(rho) - zero)), color, opacity=0.75)
    svg.text(left, top - 6, "Otokorelasyon ±%95 bant", 9.5, COLORS["muted"])
    svg.text(width / 2, height - 8, "Gecikme (bar)", 10, COLORS["muted"], "middle")
    return svg.render()


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def build_charts(df: pd.DataFrame, ind: pd.DataFrame, report: dict) -> dict:
    charts = {}
    builders = {
        "price": lambda: price_chart(df, ind, report.get("forecast", {}), report.get("patterns", {}), report.get("meta", {})),
        "spectrum": lambda: spectrum_chart(report.get("spectral", {})),
        "harmonic": lambda: harmonic_chart(df, report.get("spectral", {}), report.get("meta", {})),
        "distribution": lambda: distribution_chart(report.get("forecast", {}), report.get("meta", {})),
        "drawdown": lambda: drawdown_chart(report.get("statistics", {})),
        "seasonality": lambda: seasonality_chart(report.get("regime", {})),
        "radar": lambda: radar_chart(report.get("score", {})),
        "gauge": lambda: gauge_chart(report.get("score", {})),
        "regime": lambda: regime_strip(report.get("regime", {}), report.get("meta", {})),
        "acf": lambda: acf_chart(report.get("statistics", {})),
    }
    for key, builder in builders.items():
        try:
            charts[key] = builder()
        except Exception as exc:
            charts[key] = _empty(f"Grafik oluşturulamadı: {exc}")
    return charts

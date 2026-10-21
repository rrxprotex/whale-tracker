"""Market regimes, structural break detection and seasonality."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd


_TURKISH_MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]
_TURKISH_DAYS = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]


def _num(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value, digits: int = 4) -> Optional[float]:
    number = _num(value)
    return None if number is None else round(number, digits)


# ---------------------------------------------------------------------------
# Trend regime
# ---------------------------------------------------------------------------

def _bar_regime(row) -> str:
    close = _num(row.get("close"))
    ema50 = _num(row.get("ema50"))
    ema200 = _num(row.get("ema200"))
    adx_value = _num(row.get("adx"))
    if close is None or ema50 is None or ema200 is None:
        return "bilinmiyor"
    strong = adx_value is not None and adx_value >= 25
    if close > ema200 and ema50 > ema200:
        return "güçlü boğa" if strong else "boğa"
    if close < ema200 and ema50 < ema200:
        return "güçlü ayı" if strong else "ayı"
    return "yatay"


def trend_regime_series(ind: pd.DataFrame) -> pd.Series:
    frame = ind[["close", "ema50", "ema200", "adx"]].copy()
    return frame.apply(_bar_regime, axis=1)


def _compress_regime(labels: pd.Series, max_points: int = 260) -> List[dict]:
    labels = labels.iloc[-max_points:]
    if labels.empty:
        return []
    segments = []
    start = labels.index[0]
    current = labels.iloc[0]
    previous_index = labels.index[0]
    for index, value in labels.iloc[1:].items():
        if value != current:
            segments.append(
                {
                    "start": pd.Timestamp(start).isoformat(),
                    "end": pd.Timestamp(previous_index).isoformat(),
                    "label": str(current),
                }
            )
            start = index
            current = value
        previous_index = index
    segments.append(
        {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(previous_index).isoformat(),
            "label": str(current),
        }
    )
    return segments


# ---------------------------------------------------------------------------
# Change points
# ---------------------------------------------------------------------------

def change_points(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    max_points: int = 4,
    min_size: int = 20,
) -> List[dict]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < min_size * 3:
        return []
    points: List[dict] = []

    def split(start: int, end: int, depth: int) -> None:
        if depth >= max_points or end - start < min_size * 2:
            return
        best_t = 0.0
        best_index = None
        best_means = (0.0, 0.0)
        for index in range(start + min_size, end - min_size + 1):
            left = x[start:index]
            right = x[index:end]
            if len(left) < 5 or len(right) < 5:
                continue
            var_left = float(np.var(left, ddof=1))
            var_right = float(np.var(right, ddof=1))
            denominator = math.sqrt(
                max(1e-16, var_left / len(left) + var_right / len(right))
            )
            t_stat = abs(float(np.mean(left) - np.mean(right))) / denominator
            if t_stat > best_t:
                best_t = t_stat
                best_index = index
                best_means = (float(np.mean(left)), float(np.mean(right)))
        if best_index is None or best_t < 3.2:
            return
        points.append(
            {
                "index": int(best_index),
                "date": pd.Timestamp(dates[best_index]).isoformat(),
                "t_stat": round(best_t, 2),
                "before_mean_pct": round(best_means[0] * 100.0, 3),
                "after_mean_pct": round(best_means[1] * 100.0, 3),
                "direction": "yukarı" if best_means[1] > best_means[0] else "aşağı",
            }
        )
        split(start, best_index, depth + 1)
        split(best_index, end, depth + 1)

    split(0, n, 0)
    points.sort(key=lambda item: item["index"])
    return points


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------

def _resample_month_end(close: pd.Series) -> pd.Series:
    try:
        return close.resample("ME").last()
    except ValueError:
        return close.resample("M").last()


def seasonality(df: pd.DataFrame) -> dict:
    close = df["close"].astype(float)
    result: dict = {"available": True}
    n = len(close)
    median_delta_hours = (
        np.median(np.diff(df.index.values).astype("timedelta64[s]").astype(float))
        / 3600.0
        if n > 1 else 24.0
    )

    monthly = _resample_month_end(close).pct_change().dropna() * 100.0
    if len(monthly) >= 8:
        rows = []
        for month in range(1, 13):
            values = monthly[monthly.index.month == month]
            if len(values) == 0:
                continue
            rows.append(
                {
                    "month": month,
                    "name": _TURKISH_MONTHS[month - 1],
                    "mean_pct": round(float(values.mean()), 2),
                    "median_pct": round(float(values.median()), 2),
                    "win_rate_pct": round(float((values > 0).mean()) * 100.0, 1),
                    "std_pct": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 2),
                    "observations": int(len(values)),
                }
            )
        if rows:
            result["monthly"] = rows
            result["best_month"] = max(rows, key=lambda item: item["mean_pct"])
            result["worst_month"] = min(rows, key=lambda item: item["mean_pct"])
            current_month = int(df.index[-1].month)
            result["current_month"] = next(
                (row for row in rows if row["month"] == current_month), None
            )

    if median_delta_hours >= 18 and n >= 120:
        daily_returns = close.pct_change().dropna() * 100.0
        rows = []
        for weekday in range(7):
            values = daily_returns[daily_returns.index.dayofweek == weekday]
            if len(values) < 3:
                continue
            rows.append(
                {
                    "weekday": weekday,
                    "name": _TURKISH_DAYS[weekday],
                    "mean_pct": round(float(values.mean()), 3),
                    "median_pct": round(float(values.median()), 3),
                    "win_rate_pct": round(float((values > 0).mean()) * 100.0, 1),
                    "observations": int(len(values)),
                    "t_stat": round(
                        float(values.mean() / (values.std(ddof=1) / math.sqrt(len(values))))
                        if values.std(ddof=1) > 0 else 0.0,
                        2,
                    ),
                }
            )
        if rows:
            result["weekday"] = rows
            result["best_weekday"] = max(rows, key=lambda item: item["mean_pct"])
            result["worst_weekday"] = min(rows, key=lambda item: item["mean_pct"])

    if median_delta_hours < 18 and n >= 200:
        returns = close.pct_change().dropna() * 100.0
        rows = []
        for hour in range(24):
            values = returns[returns.index.hour == hour]
            if len(values) < 5:
                continue
            rows.append(
                {
                    "hour": hour,
                    "mean_pct": round(float(values.mean()), 4),
                    "win_rate_pct": round(float((values > 0).mean()) * 100.0, 1),
                    "observations": int(len(values)),
                }
            )
        if rows:
            result["hourly"] = rows
            result["best_hour"] = max(rows, key=lambda item: item["mean_pct"])
            result["worst_hour"] = min(rows, key=lambda item: item["mean_pct"])

    result["note"] = (
        "Sezonluk istatistikler küçük örneklem içerir; "
        "gelecekte aynı desenin süreceği garanti değildir."
    )
    return result


# ---------------------------------------------------------------------------
# Regime summary
# ---------------------------------------------------------------------------

def regime_analysis(
    df: pd.DataFrame, ind: pd.DataFrame, bars_per_year: float
) -> dict:
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    labels = trend_regime_series(ind)
    current_label = str(labels.iloc[-1]) if len(labels) else "bilinmiyor"
    segments = _compress_regime(labels)

    realized_vol = _num(ind["realized_vol"].iloc[-1]) if "realized_vol" in ind else None
    vol_percentile = None
    if realized_vol is not None:
        history = ind["realized_vol"].dropna().tail(252)
        if len(history) > 20:
            vol_percentile = float((history <= realized_vol).mean() * 100.0)
    if vol_percentile is None:
        vol_label = "bilinmiyor"
    elif vol_percentile >= 90:
        vol_label = "aşırı yüksek"
    elif vol_percentile >= 70:
        vol_label = "yüksek"
    elif vol_percentile <= 20:
        vol_label = "düşük"
    else:
        vol_label = "normal"

    volume = df["volume"].astype(float)
    volume_mean = float(volume.rolling(20, min_periods=10).mean().iloc[-1])
    relative_volume = float(volume.iloc[-1] / volume_mean) if volume_mean > 0 else None
    volume_trend = (
        float(volume.tail(20).mean() / volume.tail(60).mean())
        if len(volume) >= 60 and volume.tail(60).mean() > 0 else None
    )

    sma200 = _num(ind["sma200"].iloc[-1]) if "sma200" in ind else None
    sma200_slope = None
    if "sma200" in ind and len(ind) > 40:
        values = ind["sma200"].dropna()
        if len(values) > 25:
            sma200_slope = float(values.iloc[-1] - values.iloc[-21])
    rsi_value = _num(ind["rsi"].iloc[-1]) if "rsi" in ind else None
    macd_hist = _num(ind["macd_hist"].iloc[-1]) if "macd_hist" in ind else None

    if sma200 is None:
        phase = "belirsiz"
        phase_note = "200 barlık ortalama için yeterli veri yok."
    elif last > sma200 and (sma200_slope or 0.0) >= 0 and (rsi_value or 50) >= 50:
        phase = "markup (yükseliş)"
        phase_note = "Fiyat 200 ortalamanın üzerinde, ortalama yükseliyor ve momentum pozitif."
    elif last > sma200 and ((rsi_value is not None and rsi_value < 50) or (macd_hist or 0) < 0):
        phase = "distribution (dağıtım)"
        phase_note = "Fiyat hâlâ 200 ortalama üzerinde fakat momentum zayıflıyor."
    elif last < sma200 and (sma200_slope or 0.0) <= 0:
        phase = "markdown (düşüş)"
        phase_note = "Fiyat 200 ortalamanın altında ve ortalama aşağı eğimli."
    elif last < sma200 and ((rsi_value is not None and rsi_value > 50) or (macd_hist or 0) > 0):
        phase = "accumulation (birikim)"
        phase_note = "Fiyat 200 ortalama altında fakat momentum toparlanıyor."
    else:
        phase = "geçiş / belirsiz"
        phase_note = "Trend ve momentum sinyalleri çelişiyor."

    returns = np.diff(np.log(close.values))
    points = change_points(returns, df.index[1:])

    season = seasonality(df)

    regime_bias = {
        "güçlü boğa": 1.0,
        "boğa": 0.55,
        "yatay": 0.0,
        "ayı": -0.55,
        "güçlü ayı": -1.0,
        "bilinmiyor": 0.0,
    }.get(current_label, 0.0)

    interpretations = [
        f"Trend rejimi: {current_label}; volatilite rejimi: {vol_label}.",
        f"Piyasa fazı (heuristik): {phase} — {phase_note}",
    ]
    if relative_volume is not None:
        interpretations.append(
            f"Son hacim 20-bar ortalamanın {relative_volume:.2f} katı."
        )
    if points:
        latest = points[-1]
        interpretations.append(
            f"Son yapısal kırılma {latest['date'][:10]}: "
            f"getiri ortalaması %{latest['before_mean_pct']:.3f} → "
            f"%{latest['after_mean_pct']:.3f} (t={latest['t_stat']})."
        )
    if season.get("current_month"):
        month = season["current_month"]
        interpretations.append(
            f"{month['name']} ayı tarihsel ortalama %{month['mean_pct']:.2f}, "
            f"kazanma oranı %{month['win_rate_pct']:.0f} (n={month['observations']})."
        )

    return {
        "trend_regime": current_label,
        "trend_regime_bias": round(regime_bias, 2),
        "segments": segments,
        "volatility": {
            "realized_vol_pct": _round(realized_vol, 2),
            "percentile": _round(vol_percentile, 1),
            "label": vol_label,
        },
        "volume": {
            "relative": _round(relative_volume, 3),
            "trend_20_vs_60": _round(volume_trend, 3),
        },
        "market_phase": {
            "phase": phase,
            "note": phase_note,
            "sma200": _round(sma200, 8),
            "sma200_slope_20": _round(sma200_slope, 6),
        },
        "change_points": points,
        "seasonality": season,
        "interpretation": interpretations,
        "method_notes": [
            "Trend rejimi: EMA50/EMA200 ilişkisi + fiyat konumu + ADX.",
            "Kırılma noktaları: ikili bölme + Welch t istatistiği (min 20 bar).",
            "Piyasa fazı Wyckoff'tan esinlenen bir heuristiktir, kesin sınıflandırma değildir.",
        ],
    }

"""Price structure: pivots, support/resistance, Fibonacci, volume profile,
candlestick patterns, divergences, trend channels and double tops/bottoms."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd


def _num(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value, digits: int = 4) -> Optional[float]:
    number = _num(value)
    return None if number is None else round(number, digits)


def _iso(index_value) -> str:
    return pd.Timestamp(index_value).isoformat()


# ---------------------------------------------------------------------------
# Pivots
# ---------------------------------------------------------------------------

def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[dict]:
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    n = len(df)
    pivots: List[dict] = []
    for index in range(left, n - right):
        high_window = high[index - left:index + right + 1]
        low_window = low[index - left:index + right + 1]
        if (
            high[index] == high_window.max()
            and int(np.sum(high_window == high[index])) == 1
        ):
            pivots.append(
                {
                    "index": index,
                    "date": _iso(df.index[index]),
                    "price": float(high[index]),
                    "kind": "high",
                }
            )
        if (
            low[index] == low_window.min()
            and int(np.sum(low_window == low[index])) == 1
        ):
            pivots.append(
                {
                    "index": index,
                    "date": _iso(df.index[index]),
                    "price": float(low[index]),
                    "kind": "low",
                }
            )
    return pivots


# ---------------------------------------------------------------------------
# Support / resistance clustering
# ---------------------------------------------------------------------------

def cluster_levels(
    pivots: List[dict], current_price: float, atr_value: float
) -> List[dict]:
    if not pivots or current_price <= 0:
        return []
    tolerance = max(atr_value * 0.35, current_price * 0.0025)
    ordered = sorted(pivots, key=lambda item: item["price"])
    max_index = max((pivot["index"] for pivot in ordered), default=1) or 1

    clusters: List[dict] = []
    for pivot in ordered:
        if not clusters or abs(pivot["price"] - clusters[-1]["mean"]) > tolerance:
            clusters.append(
                {
                    "mean": pivot["price"],
                    "prices": [pivot["price"]],
                    "indices": [pivot["index"]],
                    "dates": [pivot["date"]],
                    "kinds": {pivot["kind"]},
                }
            )
        else:
            cluster = clusters[-1]
            cluster["prices"].append(pivot["price"])
            cluster["indices"].append(pivot["index"])
            cluster["dates"].append(pivot["date"])
            cluster["kinds"].add(pivot["kind"])
            cluster["mean"] = float(np.mean(cluster["prices"]))

    levels = []
    for cluster in clusters:
        recency = max(cluster["indices"]) / max_index
        touches = len(cluster["prices"])
        weight = touches * (0.5 + 0.5 * recency)
        levels.append(
            {
                "price": float(np.mean(cluster["prices"])),
                "touches": touches,
                "kind": "high" if cluster["kinds"] == {"high"} else (
                    "low" if cluster["kinds"] == {"low"} else "both"
                ),
                "strength_raw": weight,
                "last_date": max(cluster["dates"]),
            }
        )

    if not levels:
        return []
    max_strength = max(level["strength_raw"] for level in levels) or 1.0
    for level in levels:
        level["strength"] = round(level["strength_raw"] / max_strength * 100.0, 1)
        level["distance_pct"] = round(
            (level["price"] / current_price - 1.0) * 100.0, 2
        )
        if level["price"] < current_price * 0.999:
            level["type"] = "destek"
        elif level["price"] > current_price * 1.001:
            level["type"] = "direnç"
        else:
            level["type"] = "mevcut fiyat"
        level.pop("strength_raw", None)
    levels.sort(key=lambda item: item["strength"], reverse=True)
    return levels[:10]


# ---------------------------------------------------------------------------
# Fibonacci
# ---------------------------------------------------------------------------

def fibonacci_levels(df: pd.DataFrame, lookback: int = 252) -> dict:
    window = min(len(df), max(40, lookback))
    subset = df.iloc[-window:]
    high_values = subset["high"].astype(float)
    low_values = subset["low"].astype(float)
    high_index = int(np.argmax(high_values.values))
    low_index = int(np.argmin(low_values.values))
    high_price = float(high_values.iloc[high_index])
    low_price = float(low_values.iloc[low_index])
    if high_price <= low_price:
        return {}
    current = float(df["close"].iloc[-1])
    impulse_up = low_index < high_index
    span = high_price - low_price
    ratios = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

    levels = []
    for ratio in ratios:
        price = (
            high_price - span * ratio if impulse_up
            else low_price + span * ratio
        )
        levels.append(
            {
                "ratio": ratio,
                "label": f"{ratio * 100:.1f}%",
                "price": round(price, 8),
                "distance_pct": round((price / current - 1.0) * 100.0, 2),
                "type": "retracement",
            }
        )
    for extension in (1.272, 1.618):
        price = (
            high_price + span * (extension - 1.0) if impulse_up
            else low_price - span * (extension - 1.0)
        )
        levels.append(
            {
                "ratio": extension,
                "label": f"{extension * 100:.1f}% (uzatma)",
                "price": round(price, 8),
                "distance_pct": round((price / current - 1.0) * 100.0, 2),
                "type": "extension",
            }
        )

    position = (
        (high_price - current) / span if impulse_up
        else (current - low_price) / span
    )
    nearest = min(levels, key=lambda item: abs(item["price"] - current))
    return {
        "impulse": "yukarı" if impulse_up else "aşağı",
        "swing_high": round(high_price, 8),
        "swing_low": round(low_price, 8),
        "swing_high_date": _iso(subset.index[high_index]),
        "swing_low_date": _iso(subset.index[low_index]),
        "current_retracement_pct": round(max(-1.0, min(2.0, position)) * 100.0, 2),
        "nearest": nearest,
        "levels": levels,
        "lookback_bars": window,
    }


# ---------------------------------------------------------------------------
# Volume profile
# ---------------------------------------------------------------------------

def volume_profile(df: pd.DataFrame, lookback: int = 200, bins: int = 30) -> dict:
    window = min(len(df), max(60, lookback))
    subset = df.iloc[-window:]
    low = float(subset["low"].min())
    high = float(subset["high"].max())
    if high <= low:
        return {}
    edges = np.linspace(low, high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    profile = np.zeros(bins, dtype=float)

    for _, row in subset.iterrows():
        bar_low = float(row["low"])
        bar_high = float(row["high"])
        bar_volume = float(row["volume"])
        if bar_high <= bar_low or bar_volume <= 0:
            continue
        bar_bins = np.where((centers >= bar_low) & (centers <= bar_high))[0]
        if len(bar_bins) == 0:
            bar_bins = np.array([int(np.argmin(np.abs(centers - (bar_low + bar_high) / 2)))])
        profile[bar_bins] += bar_volume / len(bar_bins)

    total = float(profile.sum())
    if total <= 0:
        return {}
    poc_index = int(np.argmax(profile))
    poc = float(centers[poc_index])
    value_indices = {poc_index}
    current_volume = profile[poc_index]
    target = total * 0.70
    left = right = poc_index
    while current_volume < target and (left > 0 or right < bins - 1):
        left_volume = profile[left - 1] if left > 0 else -1.0
        right_volume = profile[right + 1] if right < bins - 1 else -1.0
        if right_volume >= left_volume:
            right += 1
            current_volume += profile[right]
            value_indices.add(right)
        else:
            left -= 1
            current_volume += profile[left]
            value_indices.add(left)
    value_low = float(edges[min(value_indices)])
    value_high = float(edges[max(value_indices) + 1])

    order = np.argsort(profile)[::-1]
    hvn = [
        {"price": round(float(centers[index]), 8), "volume_share_pct": round(float(profile[index] / total * 100.0), 2)}
        for index in order[:4]
    ]
    lvn = [
        {"price": round(float(centers[index]), 8), "volume_share_pct": round(float(profile[index] / total * 100.0), 2)}
        for index in np.argsort(profile)[:3]
    ]
    current = float(df["close"].iloc[-1])
    return {
        "lookback_bars": window,
        "poc": round(poc, 8),
        "value_area_low": round(value_low, 8),
        "value_area_high": round(value_high, 8),
        "value_area_pct": round(current_volume / total * 100.0, 2),
        "current_vs_poc_pct": round((current / poc - 1.0) * 100.0, 2) if poc else None,
        "position": (
            "değer alanı üstünde" if current > value_high else
            "değer alanı altında" if current < value_low else "değer alanı içinde"
        ),
        "hvn": hvn,
        "lvn": lvn,
        "bins": [
            {"price": round(float(centers[index]), 8), "volume_share_pct": round(float(profile[index] / total * 100.0), 2)}
            for index in range(bins)
        ],
    }


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------

def _body(open_, close) -> float:
    return abs(close - open_)


def _candle(row) -> dict:
    open_ = float(row["open"])
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    span = max(high - low, 1e-12)
    body = _body(open_, close)
    return {
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "span": span,
        "body": body,
        "body_pct": body / span,
        "upper": high - max(open_, close),
        "lower": min(open_, close) - low,
        "bullish": close > open_,
    }


def candlestick_patterns(df: pd.DataFrame) -> List[dict]:
    if len(df) < 4:
        return []
    candles = [_candle(df.iloc[index]) for index in range(len(df))]
    patterns: List[dict] = []
    start = max(2, len(df) - 4)

    for index in range(start, len(df)):
        current = candles[index]
        previous = candles[index - 1]
        before = candles[index - 2]
        date = _iso(df.index[index])

        if current["body_pct"] <= 0.10:
            patterns.append(
                {"date": date, "name": "Doji", "bias": 0, "strength": 1,
                 "note": "Alıcı-satıcı dengesi; karar için teyit gerekir."}
            )

        if (
            current["lower"] >= 2.0 * max(current["body"], 1e-12)
            and current["upper"] <= 0.30 * current["span"]
            and current["close"] >= current["low"] + current["span"] * 0.55
        ):
            patterns.append(
                {"date": date, "name": "Çekiç", "bias": 1, "strength": 2,
                 "note": "Alt bölgeden alıcı tepkisi; düşüş trendi sonunda anlamlı."}
            )

        if (
            current["upper"] >= 2.0 * max(current["body"], 1e-12)
            and current["lower"] <= 0.30 * current["span"]
            and current["close"] <= current["low"] + current["span"] * 0.45
        ):
            patterns.append(
                {"date": date, "name": "Kayan Yıldız", "bias": -1, "strength": 2,
                 "note": "Üst bölgeden satıcı baskısı; yükseliş trendi sonunda anlamlı."}
            )

        if (
            not previous["bullish"] and current["bullish"]
            and current["body"] > previous["body"] * 1.05
            and current["close"] >= previous["open"]
            and current["open"] <= previous["close"]
        ):
            patterns.append(
                {"date": date, "name": "Yutan Boğa", "bias": 1, "strength": 3,
                 "note": "Önceki düşüş mumunu tamamen yutan alıcı gücü."}
            )

        if (
            previous["bullish"] and not current["bullish"]
            and current["body"] > previous["body"] * 1.05
            and current["open"] >= previous["close"]
            and current["close"] <= previous["open"]
        ):
            patterns.append(
                {"date": date, "name": "Yutan Ayı", "bias": -1, "strength": 3,
                 "note": "Önceki yükseliş mumunu tamamen yutan satıcı gücü."}
            )

        if (
            previous["body"] > current["body"] * 2.0
            and current["open"] > min(previous["open"], previous["close"])
            and current["close"] < max(previous["open"], previous["close"])
            and previous["body_pct"] > 0.5
        ):
            patterns.append(
                {"date": date, "name": "Harami", "bias": 1 if previous["bullish"] else -1,
                 "strength": 1, "note": "Büyük mumun içinde sıkışma; momentum zayıflıyor."}
            )

        if index >= 2:
            first, second, third = before, previous, current
            if (
                not first["bullish"] and first["body_pct"] > 0.5
                and second["body_pct"] < 0.35
                and third["bullish"] and third["close"] > (first["open"] + first["close"]) / 2.0
            ):
                patterns.append(
                    {"date": date, "name": "Sabah Yıldızı", "bias": 1, "strength": 3,
                     "note": "Düşüş sonrası üç mumluk dönüş formasyonu."}
                )
            if (
                first["bullish"] and first["body_pct"] > 0.5
                and second["body_pct"] < 0.35
                and not third["bullish"] and third["close"] < (first["open"] + first["close"]) / 2.0
            ):
                patterns.append(
                    {"date": date, "name": "Akşam Yıldızı", "bias": -1, "strength": 3,
                     "note": "Yükseliş sonrası üç mumluk dönüş formasyonu."}
                )

        if index >= 2:
            last_three = candles[index - 2:index + 1]
            if all(candle["bullish"] for candle in last_three) and all(
                candle["body_pct"] > 0.45 for candle in last_three
            ) and last_three[2]["close"] > last_three[1]["close"] > last_three[0]["close"]:
                patterns.append(
                    {"date": date, "name": "Üç Beyaz Asker", "bias": 1, "strength": 2,
                     "note": "Ardışık güçlü alıcı mumları; trend devam sinyali."}
                )
            if all(not candle["bullish"] for candle in last_three) and all(
                candle["body_pct"] > 0.45 for candle in last_three
            ) and last_three[2]["close"] < last_three[1]["close"] < last_three[0]["close"]:
                patterns.append(
                    {"date": date, "name": "Üç Kara Karga", "bias": -1, "strength": 2,
                     "note": "Ardışık güçlü satıcı mumları; trend devam sinyali."}
                )

    unique = {}
    for pattern in patterns:
        key = (pattern["date"], pattern["name"])
        if key not in unique or pattern["strength"] > unique[key]["strength"]:
            unique[key] = pattern
    return sorted(unique.values(), key=lambda item: item["date"])[-6:]


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def rsi_divergence(
    df: pd.DataFrame, ind: pd.DataFrame, lookback: int = 90
) -> List[dict]:
    if "rsi" not in ind:
        return []
    window = min(len(df), lookback)
    subset = df.iloc[-window:]
    rsi_series = ind["rsi"].iloc[-window:]
    pivots = find_pivots(subset, left=2, right=2)
    highs = [pivot for pivot in pivots if pivot["kind"] == "high"]
    lows = [pivot for pivot in pivots if pivot["kind"] == "low"]
    results = []

    if len(highs) >= 2:
        first, second = highs[-2], highs[-1]
        rsi_first = _num(rsi_series.iloc[first["index"]])
        rsi_second = _num(rsi_series.iloc[second["index"]])
        if rsi_first is not None and rsi_second is not None:
            if second["price"] > first["price"] * 1.002 and rsi_second < rsi_first - 1.5:
                results.append(
                    {
                        "type": "ayı uyumsuzluğu",
                        "bias": -1,
                        "strength": round(min(3.0, (rsi_first - rsi_second) / 5.0), 2),
                        "price_points": [first["price"], second["price"]],
                        "rsi_points": [round(rsi_first, 2), round(rsi_second, 2)],
                        "dates": [first["date"], second["date"]],
                        "note": "Fiyat daha yüksek tepe yaparken RSI zayıflıyor.",
                    }
                )
    if len(lows) >= 2:
        first, second = lows[-2], lows[-1]
        rsi_first = _num(rsi_series.iloc[first["index"]])
        rsi_second = _num(rsi_series.iloc[second["index"]])
        if rsi_first is not None and rsi_second is not None:
            if second["price"] < first["price"] * 0.998 and rsi_second > rsi_first + 1.5:
                results.append(
                    {
                        "type": "boğa uyumsuzluğu",
                        "bias": 1,
                        "strength": round(min(3.0, (rsi_second - rsi_first) / 5.0), 2),
                        "price_points": [first["price"], second["price"]],
                        "rsi_points": [round(rsi_first, 2), round(rsi_second, 2)],
                        "dates": [first["date"], second["date"]],
                        "note": "Fiyat daha düşük dip yaparken RSI toparlanıyor.",
                    }
                )
    return results


# ---------------------------------------------------------------------------
# Trend channel
# ---------------------------------------------------------------------------

def trend_channel(df: pd.DataFrame, bars_per_year: float, lookback: int = 160) -> dict:
    window = min(len(df), max(30, lookback))
    subset = df.iloc[-window:]
    y = np.log(subset["close"].astype(float).values)
    t = np.arange(len(y), dtype=float)
    try:
        slope, intercept = np.polyfit(t, y, 1)
    except np.linalg.LinAlgError:
        return {}
    fitted = slope * t + intercept
    residual = y - fitted
    sigma = float(np.std(residual, ddof=1))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else 0.0
    upper = np.exp(fitted + 2.0 * sigma)
    lower = np.exp(fitted - 2.0 * sigma)
    current = float(df["close"].iloc[-1])
    position = (
        (current - float(lower[-1])) / (float(upper[-1]) - float(lower[-1]))
        if upper[-1] > lower[-1] else 0.5
    )
    breakout = None
    if current > float(upper[-1]):
        breakout = "yukarı"
    elif current < float(lower[-1]):
        breakout = "aşağı"
    annual_slope = (math.exp(float(slope) * bars_per_year) - 1.0) * 100.0
    return {
        "lookback_bars": window,
        "slope_per_bar_pct": round((math.exp(float(slope)) - 1.0) * 100.0, 4),
        "annualized_slope_pct": round(annual_slope, 2),
        "r2": round(r2, 3),
        "channel_position_pct": round(max(-0.5, min(1.5, position)) * 100.0, 2),
        "upper": round(float(upper[-1]), 8),
        "lower": round(float(lower[-1]), 8),
        "mid": round(float(np.exp(fitted[-1])), 8),
        "breakout": breakout,
        "valid": bool(r2 >= 0.45),
        "note": (
            "Kanal istatistiksel olarak anlamlı." if r2 >= 0.45
            else "Kanal zayıf; çizgiler destek/direnç olarak güvenilir değil."
        ),
    }


# ---------------------------------------------------------------------------
# Double top / bottom
# ---------------------------------------------------------------------------

def double_pattern(
    pivots: List[dict], current_price: float, atr_value: float
) -> Optional[dict]:
    if len(pivots) < 4:
        return None
    tolerance = max(current_price * 0.02, atr_value * 1.2)
    highs = [pivot for pivot in pivots if pivot["kind"] == "high"][-4:]
    lows = [pivot for pivot in pivots if pivot["kind"] == "low"][-4:]

    for first, second in zip(highs[-2:], highs[-1:]):
        if abs(first["price"] - second["price"]) <= tolerance and 5 <= second["index"] - first["index"] <= 80:
            between = [
                pivot for pivot in lows
                if first["index"] < pivot["index"] < second["index"]
            ]
            if between:
                neckline = min(pivot["price"] for pivot in between)
                confirmed = current_price < neckline
                return {
                    "type": "ikili tepe",
                    "bias": -1,
                    "status": "teyitli" if confirmed else "oluşuyor",
                    "first_top": round(first["price"], 8),
                    "second_top": round(second["price"], 8),
                    "neckline": round(neckline, 8),
                    "target": round(neckline - (np.mean([first["price"], second["price"]]) - neckline), 8),
                    "note": (
                        "Boyun çizgisi kırıldı; formasyon teyitli."
                        if confirmed else "Boyun çizgisi henüz kırılmadı."
                    ),
                }

    for first, second in zip(lows[-2:], lows[-1:]):
        if abs(first["price"] - second["price"]) <= tolerance and 5 <= second["index"] - first["index"] <= 80:
            between = [
                pivot for pivot in highs
                if first["index"] < pivot["index"] < second["index"]
            ]
            if between:
                neckline = max(pivot["price"] for pivot in between)
                confirmed = current_price > neckline
                return {
                    "type": "ikili dip",
                    "bias": 1,
                    "status": "teyitli" if confirmed else "oluşuyor",
                    "first_bottom": round(first["price"], 8),
                    "second_bottom": round(second["price"], 8),
                    "neckline": round(neckline, 8),
                    "target": round(neckline + (neckline - np.mean([first["price"], second["price"]])), 8),
                    "note": (
                        "Boyun çizgisi yukarı kırıldı; formasyon teyitli."
                        if confirmed else "Boyun çizgisi henüz yukarı kırılmadı."
                    ),
                }
    return None


# ---------------------------------------------------------------------------
# Combined pattern analysis
# ---------------------------------------------------------------------------

def pattern_analysis(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    bars_per_year: float,
    atr_value: float,
) -> dict:
    current = float(df["close"].iloc[-1])
    pivots = find_pivots(df)
    levels = cluster_levels(pivots, current, atr_value)
    fib = fibonacci_levels(df)
    profile = volume_profile(df)
    candles = candlestick_patterns(df)
    divergences = rsi_divergence(df, ind)
    channel = trend_channel(df, bars_per_year)
    double = double_pattern(pivots, current, atr_value)

    supports = sorted(
        [level for level in levels if level["price"] < current * 0.999],
        key=lambda item: item["price"], reverse=True,
    )
    resistances = sorted(
        [level for level in levels if level["price"] > current * 1.001],
        key=lambda item: item["price"],
    )
    nearest_support = supports[0] if supports else None
    nearest_resistance = resistances[0] if resistances else None

    if nearest_resistance and current > nearest_resistance["price"] * 1.002:
        breakout = "direnç yukarı kırıldı"
    elif nearest_support and current < nearest_support["price"] * 0.998:
        breakout = "destek aşağı kırıldı"
    else:
        breakout = "aralık içinde"

    score = 0.0
    for candle in candles:
        score += candle["bias"] * candle["strength"] * 0.10
    for divergence in divergences:
        score += divergence["bias"] * min(0.35, 0.15 + divergence["strength"] * 0.08)
    if channel:
        if channel.get("breakout") == "yukarı":
            score += 0.30
        elif channel.get("breakout") == "aşağı":
            score -= 0.30
        else:
            score += (channel["channel_position_pct"] / 100.0 - 0.5) * 0.30
    if double:
        weight = 0.40 if double["status"] == "teyitli" else 0.20
        score += double["bias"] * weight
    if nearest_support and abs(nearest_support["distance_pct"]) <= 1.5:
        score += 0.12
    if nearest_resistance and abs(nearest_resistance["distance_pct"]) <= 1.5:
        score -= 0.12
    score = float(max(-1.0, min(1.0, score)))

    key_levels = {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "poc": profile.get("poc"),
        "value_area_low": profile.get("value_area_low"),
        "value_area_high": profile.get("value_area_high"),
        "fib_nearest": fib.get("nearest"),
        "swing_high": fib.get("swing_high"),
        "swing_low": fib.get("swing_low"),
        "invalidation_down": (
            round(nearest_support["price"] - atr_value * 0.5, 8)
            if nearest_support else round(current - atr_value * 2.0, 8)
        ),
        "invalidation_up": (
            round(nearest_resistance["price"] + atr_value * 0.5, 8)
            if nearest_resistance else round(current + atr_value * 2.0, 8)
        ),
        "breakout": breakout,
    }

    interpretations = []
    if nearest_support:
        interpretations.append(
            f"En yakın destek {nearest_support['price']:.6g} "
            f"(%{nearest_support['distance_pct']:+.2f}), güç {nearest_support['strength']:.0f}/100."
        )
    if nearest_resistance:
        interpretations.append(
            f"En yakın direnç {nearest_resistance['price']:.6g} "
            f"(%{nearest_resistance['distance_pct']:+.2f}), güç {nearest_resistance['strength']:.0f}/100."
        )
    if profile:
        interpretations.append(
            f"Hacim profili POC {profile['poc']:.6g}; fiyat {profile['position']}."
        )
    if fib:
        interpretations.append(
            f"Fibonacci swing {fib['swing_low']:.6g}–{fib['swing_high']:.6g}, "
            f"mevcut geri çekilme %{fib['current_retracement_pct']:.1f}."
        )
    if divergences:
        interpretations.append(divergences[0]["note"])
    if double:
        interpretations.append(f"{double['type'].title()} formasyonu {double['status']}.")
    if candles:
        interpretations.append(
            "Son mum formasyonları: " + ", ".join(candle["name"] for candle in candles[-3:]) + "."
        )

    return {
        "current_price": round(current, 8),
        "pivots": pivots[-30:],
        "levels": levels,
        "supports": supports[:4],
        "resistances": resistances[:4],
        "key_levels": key_levels,
        "fibonacci": fib,
        "volume_profile": profile,
        "candles": candles,
        "divergences": divergences,
        "channel": channel,
        "double_pattern": double,
        "pattern_score": round(score, 3),
        "interpretation": interpretations,
        "method_notes": [
            "Pivot: 3 bar sol/sağ onaylı fraktal tepeler ve dipler.",
            "S/R kümeleri ATR tabanlı toleransla birleştirildi; güç dokunma ve güncelliğe bağlı.",
            "Hacim profili son 200 barın high-low aralığına hacim dağıtır.",
            "Formasyonlar olasılıksaldır; hacim ve trendle teyit edilmelidir.",
        ],
    }

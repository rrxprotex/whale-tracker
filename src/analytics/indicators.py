"""Technical indicators and an explainable technical snapshot.

All indicators are implemented with pandas/numpy only (no TA-Lib).
The snapshot returns every component with its raw value, normalised score
(-1..+1), weight and a Turkish explanation so the final score is auditable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primitive indicators
# ---------------------------------------------------------------------------

def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    result = result.where(avg_loss != 0.0, 100.0)
    result = result.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return result


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - signal_line
    return {"macd": line, "signal": signal_line, "hist": hist}


def bollinger(
    close: pd.Series, period: int = 20, deviations: float = 2.0
) -> Dict[str, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + deviations * std
    lower = mid - deviations * std
    width = (upper - lower) / mid.replace(0.0, np.nan) * 100.0
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return {
        "mid": mid, "upper": upper, "lower": lower,
        "width": width, "pct_b": pct_b,
    }


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    atr_values = atr(df, period).replace(0.0, np.nan)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_values
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_values
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_values = _wilder(dx, period)
    return {
        "adx": adx_values, "plus_di": plus_di,
        "minus_di": minus_di, "dx": dx,
    }


def stochastic(
    df: pd.DataFrame, period: int = 14, smooth: int = 3
) -> Dict[str, pd.Series]:
    low_n = df["low"].rolling(period, min_periods=period).min()
    high_n = df["high"].rolling(period, min_periods=period).max()
    k = 100.0 * (df["close"] - low_n) / (high_n - low_n).replace(0.0, np.nan)
    d = k.rolling(smooth, min_periods=smooth).mean()
    return {"k": k, "d": d}


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    mean = typical.rolling(period, min_periods=period).mean()
    mad = typical.rolling(period, min_periods=period).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    return (typical - mean) / (0.015 * mad.replace(0.0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_n = df["high"].rolling(period, min_periods=period).max()
    low_n = df["low"].rolling(period, min_periods=period).min()
    return -100.0 * (high_n - df["close"]) / (high_n - low_n).replace(0.0, np.nan)


def roc(close: pd.Series, period: int = 20) -> pd.Series:
    return close.pct_change(period) * 100.0


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    numerator = (typical * df["volume"]).rolling(period, min_periods=period).sum()
    denominator = df["volume"].rolling(period, min_periods=period).sum()
    return numerator / denominator.replace(0.0, np.nan)


def ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    high, low, close = df["high"], df["low"], df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = (
        (high.rolling(52).max() + low.rolling(52).min()) / 2.0
    ).shift(26)
    chikou = close.shift(-26)
    return {
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou,
    }


def indicator_frame(df: pd.DataFrame, bars_per_year: float = 365.0) -> pd.DataFrame:
    """Compute every series used by snapshots, scoring and charts."""
    close, high, low, volume = (
        df["close"].astype(float),
        df["high"].astype(float),
        df["low"].astype(float),
        df["volume"].astype(float),
    )
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["high"] = high
    out["low"] = low
    out["volume"] = volume

    for period in (20, 50, 200):
        out[f"sma{period}"] = sma(close, period)
        out[f"ema{period}"] = ema(close, period)

    bb = bollinger(close)
    out["bb_mid"] = bb["mid"]
    out["bb_upper"] = bb["upper"]
    out["bb_lower"] = bb["lower"]
    out["bb_width"] = bb["width"]
    out["bb_pct_b"] = bb["pct_b"]

    out["rsi"] = rsi(close)
    macd_values = macd(close)
    out["macd"] = macd_values["macd"]
    out["macd_signal"] = macd_values["signal"]
    out["macd_hist"] = macd_values["hist"]

    out["atr"] = atr(df)
    out["atr_pct"] = out["atr"] / close.replace(0.0, np.nan) * 100.0
    adx_values = adx(df)
    out["adx"] = adx_values["adx"]
    out["plus_di"] = adx_values["plus_di"]
    out["minus_di"] = adx_values["minus_di"]

    stoch = stochastic(df)
    out["stoch_k"] = stoch["k"]
    out["stoch_d"] = stoch["d"]

    out["obv"] = obv(df)
    out["cci"] = cci(df)
    out["williams_r"] = williams_r(df)
    out["roc20"] = roc(close, 20)
    out["vwap20"] = rolling_vwap(df)

    ichi = ichimoku(df)
    for key, values in ichi.items():
        out[f"ichi_{key}"] = values

    log_returns = np.log(close).diff()
    out["log_return"] = log_returns
    out["realized_vol"] = (
        log_returns.rolling(20, min_periods=20).std()
        * math.sqrt(max(bars_per_year, 1.0))
        * 100.0
    )
    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std(ddof=0)
    out["volume_mean20"] = vol_mean
    out["volume_z"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _num(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _round(value, digits: int = 4) -> Optional[float]:
    number = _num(value)
    return None if number is None else round(number, digits)


def _tanh(value: float, scale: float) -> float:
    return float(np.tanh(value / scale))


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _signal_label(score: Optional[float]) -> str:
    if score is None:
        return "—"
    if score >= 0.25:
        return "Pozitif"
    if score <= -0.25:
        return "Negatif"
    return "Nötr"


def _rsi_zone(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value >= 70:
        return "aşırı alım"
    if value <= 30:
        return "aşırı satım"
    if value >= 55:
        return "pozitif momentum"
    if value <= 45:
        return "negatif momentum"
    return "nötr"


def compute_snapshot(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    bars_per_year: float = 365.0,
) -> dict:
    """Build a full technical snapshot with per-component contributions."""
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    change_pct = (last / prev - 1.0) * 100.0 if prev else 0.0

    def latest(name: str) -> Optional[float]:
        if name not in ind.columns:
            return None
        return _num(ind[name].iloc[-1])

    def return_over(bars: int) -> Optional[float]:
        if len(close) <= bars:
            return None
        base = float(close.iloc[-1 - bars])
        if base == 0:
            return None
        return (last / base - 1.0) * 100.0

    returns = {}
    for bars in (1, 5, 20, 60, 120, 252):
        value = return_over(bars)
        if value is not None:
            returns[f"{bars}_bar"] = {"bars": bars, "pct": round(value, 2)}

    window = min(len(close), 252)
    high_52w = float(df["high"].iloc[-window:].max())
    low_52w = float(df["low"].iloc[-window:].min())
    range_pos = (
        (last - low_52w) / (high_52w - low_52w)
        if high_52w > low_52w else 0.5
    )

    rsi_value = latest("rsi")
    macd_hist = latest("macd_hist")
    macd_line = latest("macd")
    macd_signal = latest("macd_signal")
    adx_value = latest("adx")
    plus_di = latest("plus_di")
    minus_di = latest("minus_di")
    atr_pct = latest("atr_pct")
    realized_vol = latest("realized_vol")
    bb_pct_b = latest("bb_pct_b")
    bb_width = latest("bb_width")
    stoch_k = latest("stoch_k")
    stoch_d = latest("stoch_d")
    roc20 = latest("roc20")
    obv_series = ind["obv"] if "obv" in ind else None
    volume = float(df["volume"].iloc[-1])
    volume_mean = latest("volume_mean20")
    rel_volume = volume / volume_mean if volume_mean and volume_mean > 0 else None

    sma20, sma50, sma200 = latest("sma20"), latest("sma50"), latest("sma200")
    ema20, ema50, ema200 = latest("ema20"), latest("ema50"), latest("ema200")
    tenkan, kijun = latest("ichi_tenkan"), latest("ichi_kijun")
    senkou_a, senkou_b = latest("ichi_senkou_a"), latest("ichi_senkou_b")

    components: List[dict] = []

    def add(
        key: str,
        label: str,
        group: str,
        score: Optional[float],
        weight: float,
        value_display: str,
        note: str,
    ) -> None:
        if score is None:
            return
        score = _clip(score)
        components.append(
            {
                "key": key,
                "label": label,
                "group": group,
                "score": round(score, 4),
                "weight": weight,
                "contribution": round(score * weight * 100.0, 2),
                "value": value_display,
                "note": note,
                "signal": _signal_label(score),
            }
        )

    # --- Trend -----------------------------------------------------------
    reference = sma200 if sma200 else (sma50 if sma50 else sma20)
    if reference and reference > 0:
        distance = last / reference - 1.0
        add(
            "trend_price", "Fiyat / Uzun Ortalama", "trend",
            _tanh(distance, 0.08), 0.08,
            f"Fiyat referans ortalamanın %{distance * 100:+.2f} üzerinde",
            "Fiyatın uzun vadeli ortalamaya göre konumu.",
        )

    if ema20 and ema50:
        if ema200:
            raw = (ema20 / ema50 - 1.0) + (ema50 / ema200 - 1.0)
            stack_text = (
                "EMA20 > EMA50 > EMA200" if ema20 > ema50 > ema200 else
                "EMA20 < EMA50 < EMA200" if ema20 < ema50 < ema200 else
                "EMA'lar karışık sıralamada"
            )
        else:
            raw = (ema20 / ema50 - 1.0) * 2.0
            stack_text = "EMA20 > EMA50" if ema20 > ema50 else "EMA20 < EMA50"
        add(
            "ema_stack", "EMA Dizilimi", "trend",
            _tanh(raw, 0.05), 0.10, stack_text,
            "Kısa/orta/uzun üstel ortalamaların dizilimi.",
        )

    if adx_value is not None and plus_di is not None and minus_di is not None:
        direction = 1.0 if plus_di >= minus_di else -1.0
        strength = min(adx_value / 40.0, 1.0)
        if adx_value < 18:
            strength *= 0.35
        add(
            "adx", "ADX Trend Gücü", "trend",
            direction * strength, 0.07,
            f"ADX {adx_value:.1f} | +DI {plus_di:.1f} / -DI {minus_di:.1f}",
            "ADX yön değil güç ölçer; yön +DI/-DI karşılaştırmasından gelir.",
        )

    # --- Momentum --------------------------------------------------------
    if rsi_value is not None:
        add(
            "rsi", "RSI(14)", "momentum",
            _tanh(rsi_value - 50.0, 18.0), 0.07,
            f"RSI {rsi_value:.1f} ({_rsi_zone(rsi_value)})",
            "Aşırı bölgelerde momentum yorgunluğu riski artar.",
        )

    if macd_hist is not None and last > 0:
        hist_pct = macd_hist / last * 100.0
        cross = 0.0
        if macd_line is not None and macd_signal is not None:
            cross = 0.15 if macd_line > macd_signal else -0.15
        add(
            "macd", "MACD", "momentum",
            _clip(_tanh(hist_pct, 0.6) + cross), 0.09,
            f"Histogram {macd_hist:+.4g} | MACD {macd_line:+.4g}",
            "MACD histogramının işareti ve büyüklüğü.",
        )

    if roc20 is not None:
        add(
            "roc20", "20-Bar Momentum", "momentum",
            _tanh(roc20, 10.0), 0.05,
            f"ROC(20) %{roc20:+.2f}",
            "Son 20 barın yüzdesel getirisi.",
        )

    # --- Mean reversion / oscillator -------------------------------------
    if bb_pct_b is not None:
        add(
            "bollinger", "Bollinger %B", "mean_reversion",
            -_tanh(bb_pct_b - 0.5, 0.30), 0.05,
            f"%B {bb_pct_b:.2f}" + (f" | genişlik %{bb_width:.1f}" if bb_width else ""),
            "Bandın üst ucuna yakınlık kısa vadede aşırılık olabilir.",
        )

    if stoch_k is not None:
        add(
            "stochastic", "Stochastic", "mean_reversion",
            -_tanh(stoch_k - 50.0, 25.0), 0.04,
            f"%K {stoch_k:.1f} / %D {stoch_d:.1f}" if stoch_d is not None else f"%K {stoch_k:.1f}",
            "Kısa vadeli aşırı alım/satım osilatörü.",
        )

    # --- Volume ----------------------------------------------------------
    if obv_series is not None and len(obv_series) > 21 and volume_mean:
        obv_slope = (
            float(obv_series.iloc[-1]) - float(obv_series.iloc[-21])
        ) / max(volume_mean * 20.0, 1e-9)
        add(
            "obv", "OBV Birikim", "volume",
            _tanh(obv_slope, 0.35), 0.08,
            f"OBV 20-bar eğimi {obv_slope:+.3f}",
            "Hacim destekli birikim/dağıtım baskısı.",
        )

    if rel_volume is not None:
        last_return = last / prev - 1.0 if prev else 0.0
        if rel_volume > 1.2:
            score = math.copysign(min(1.0, (rel_volume - 1.0) / 1.5), last_return or 1.0)
            score *= 1.15
        else:
            score = 0.0
        add(
            "volume_confirmation", "Hacim Onayı", "volume",
            _clip(score), 0.04,
            f"Hacim {rel_volume:.2f}x 20-bar ortalama",
            "Yükseliş/düşüşün hacimle teyit edilip edilmediği.",
        )

    # --- Ichimoku --------------------------------------------------------
    if senkou_a is not None and senkou_b is not None:
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        if last > cloud_top:
            score = 0.75
            position = "Fiyat bulutun üzerinde"
        elif last < cloud_bottom:
            score = -0.75
            position = "Fiyat bulutun altında"
        else:
            score = 0.0
            position = "Fiyat bulutun içinde"
        if tenkan is not None and kijun is not None:
            score += 0.2 if tenkan > kijun else -0.2
            position += " | Tenkan > Kijun" if tenkan > kijun else " | Tenkan < Kijun"
        add(
            "ichimoku", "Ichimoku Bulut", "trend",
            _clip(score), 0.08, position,
            "Bulut konumu ve Tenkan/Kijun kesişimi.",
        )

    # --- Range position --------------------------------------------------
    add(
        "range_position", "52-Bar Aralık Konumu", "position",
        _tanh(range_pos - 0.5, 0.30), 0.05,
        f"Aralık konumu %{range_pos * 100:.0f}",
        "Fiyatın son 252 barın alt/üst bandındaki yeri.",
    )

    total_weight = sum(component["weight"] for component in components)
    technical_score = (
        sum(component["score"] * component["weight"] for component in components)
        / total_weight * 100.0
        if total_weight else 0.0
    )

    # Realised volatility percentile
    vol_percentile = None
    if "realized_vol" in ind and realized_vol is not None:
        history = ind["realized_vol"].dropna().tail(252)
        if len(history) > 20:
            vol_percentile = float((history <= realized_vol).mean() * 100.0)

    trend_score = sum(
        c["score"] * c["weight"] for c in components if c["group"] == "trend"
    ) / max(1e-9, sum(c["weight"] for c in components if c["group"] == "trend"))
    momentum_score = sum(
        c["score"] * c["weight"] for c in components if c["group"] == "momentum"
    ) / max(1e-9, sum(c["weight"] for c in components if c["group"] == "momentum"))

    if trend_score >= 0.45:
        trend_label = "Güçlü yükseliş trendi"
    elif trend_score >= 0.15:
        trend_label = "Yükseliş eğilimi"
    elif trend_score <= -0.45:
        trend_label = "Güçlü düşüş trendi"
    elif trend_score <= -0.15:
        trend_label = "Düşüş eğilimi"
    else:
        trend_label = "Yatay / kararsız"

    if momentum_score >= 0.35:
        momentum_label = "Momentum pozitif"
    elif momentum_score <= -0.35:
        momentum_label = "Momentum negatif"
    else:
        momentum_label = "Momentum zayıf"

    if vol_percentile is None:
        vol_label = "Bilinmiyor"
    elif vol_percentile >= 90:
        vol_label = "Aşırı yüksek volatilite"
    elif vol_percentile >= 70:
        vol_label = "Yüksek volatilite"
    elif vol_percentile <= 20:
        vol_label = "Düşük volatilite"
    else:
        vol_label = "Normal volatilite"

    top_components = sorted(
        components, key=lambda c: abs(c["contribution"]), reverse=True
    )[:3]
    summary = "; ".join(
        f"{c['label']}: {c['signal'].lower()}" for c in top_components
    )
    if summary:
        summary = "Teknik tabloya en çok etki eden bileşenler — " + summary + "."
    else:
        summary = "Yeterli gösterge verisi oluşmadı."

    return {
        "price": round(last, 8),
        "change_pct": round(change_pct, 2),
        "change_abs": round(last - prev, 8),
        "returns": returns,
        "range_52w": {
            "high": round(high_52w, 8),
            "low": round(low_52w, 8),
            "position_pct": round(range_pos * 100.0, 2),
            "distance_to_high_pct": round((last / high_52w - 1.0) * 100.0, 2) if high_52w else None,
            "distance_to_low_pct": round((last / low_52w - 1.0) * 100.0, 2) if low_52w else None,
        },
        "indicators": {
            "rsi": _round(rsi_value, 2),
            "rsi_zone": _rsi_zone(rsi_value),
            "macd": _round(macd_line, 6),
            "macd_signal": _round(macd_signal, 6),
            "macd_hist": _round(macd_hist, 6),
            "adx": _round(adx_value, 2),
            "plus_di": _round(plus_di, 2),
            "minus_di": _round(minus_di, 2),
            "atr_pct": _round(atr_pct, 2),
            "realized_vol_pct": _round(realized_vol, 2),
            "vol_percentile": _round(vol_percentile, 1),
            "bb_pct_b": _round(bb_pct_b, 3),
            "bb_width_pct": _round(bb_width, 2),
            "stoch_k": _round(stoch_k, 2),
            "stoch_d": _round(stoch_d, 2),
            "roc20_pct": _round(roc20, 2),
            "sma20": _round(sma20, 8),
            "sma50": _round(sma50, 8),
            "sma200": _round(sma200, 8),
            "ema20": _round(ema20, 8),
            "ema50": _round(ema50, 8),
            "ema200": _round(ema200, 8),
        },
        "volume": {
            "last": _round(volume, 4),
            "average_20": _round(volume_mean, 4),
            "relative": _round(rel_volume, 3),
        },
        "components": components,
        "technical_score": round(technical_score, 2),
        "trend_score": round(trend_score, 3),
        "momentum_score": round(momentum_score, 3),
        "trend_label": trend_label,
        "momentum_label": momentum_label,
        "volatility_label": vol_label,
        "summary": summary,
    }

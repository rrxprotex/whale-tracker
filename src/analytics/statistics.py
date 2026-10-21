"""Statistical character, long memory, mean reversion and risk metrics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi2_sf(value: float, df: int) -> float:
    """Wilson-Hilferty approximation to the chi-square survival function."""
    if df <= 0:
        return 1.0
    if value <= 0:
        return 1.0
    z = (
        (value / df) ** (1.0 / 3.0)
        - (1.0 - 2.0 / (9.0 * df))
    ) / math.sqrt(2.0 / (9.0 * df))
    return max(0.0, min(1.0, _normal_sf(z)))


def _ols(y: np.ndarray, x: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(y)), x])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        dof = max(1, len(y) - design.shape[1])
        sigma2 = float(residual @ residual / dof)
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(residual @ residual) / ss_tot if ss_tot > 1e-12 else 0.0
        return {
            "beta": beta,
            "se": standard_errors,
            "residual": residual,
            "r2": r2,
            "sigma2": sigma2,
            "dof": dof,
        }
    except np.linalg.LinAlgError:
        return {}


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
# Long memory
# ---------------------------------------------------------------------------

def hurst_rs(y: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    sizes = [size for size in (8, 16, 32, 64, 128, 256) if size <= n // 2]
    if len(sizes) < 3:
        return None
    points = []
    for size in sizes:
        values = []
        for start in range(0, n - size + 1, size):
            segment = y[start:start + size]
            segment = segment - segment.mean()
            cumulative = np.cumsum(segment)
            spread = float(cumulative.max() - cumulative.min())
            std = float(segment.std(ddof=1))
            if std > 1e-12:
                values.append(spread / std)
        if values:
            points.append((size, float(np.mean(values))))
    if len(points) < 3:
        return None
    x = np.log([point[0] for point in points])
    y_values = np.log([max(point[1], 1e-9) for point in points])
    slope = float(np.polyfit(x, y_values, 1)[0])
    return max(0.0, min(1.5, slope))


def hurst_dfa(y: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    sizes = [size for size in (8, 16, 32, 64, 128, 256) if size <= n // 3]
    if len(sizes) < 3:
        return None
    profile = np.cumsum(y - y.mean())
    points = []
    for size in sizes:
        fluctuations = []
        for start in range(0, n - size + 1, size):
            segment = profile[start:start + size]
            t = np.arange(size, dtype=float)
            try:
                slope, intercept = np.polyfit(t, segment, 1)
            except np.linalg.LinAlgError:
                continue
            residual = segment - (slope * t + intercept)
            fluctuations.append(float(np.sqrt(np.mean(residual ** 2))))
        if fluctuations:
            points.append((size, float(np.mean(fluctuations))))
    if len(points) < 3:
        return None
    x = np.log([point[0] for point in points])
    values = np.log([max(point[1], 1e-12) for point in points])
    slope = float(np.polyfit(x, values, 1)[0])
    return max(0.0, min(1.5, slope))


# ---------------------------------------------------------------------------
# Mean reversion / randomness
# ---------------------------------------------------------------------------

def adf_test(y: np.ndarray, max_lag: Optional[int] = None) -> dict:
    """Augmented Dickey-Fuller test (constant, approximate p-value)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 40:
        return {"available": False}
    differences = np.diff(y)
    lag = max_lag or min(8, max(1, int(12.0 * (n / 100.0) ** 0.25)))
    lag = min(lag, max(1, (n - 20) // 4))
    indices = np.arange(lag, len(differences))
    target = differences[indices]
    columns = [np.ones(len(indices)), y[indices]]
    for offset in range(1, lag + 1):
        columns.append(differences[indices - offset])
    fit = _ols(target, np.column_stack(columns))
    if not fit:
        return {"available": False}
    beta = float(fit["beta"][1])
    se = float(fit["se"][1])
    t_stat = beta / se if se > 0 else 0.0
    # MacKinnon critical values for the constant-only case (large sample)
    critical = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    p_approx = float(
        np.interp(
            t_stat,
            [-4.5, -3.43, -2.86, -2.57, -1.62, 0.0, 1.0],
            [0.001, 0.01, 0.05, 0.10, 0.50, 0.95, 0.99],
        )
    )
    return {
        "available": True,
        "t_stat": round(t_stat, 3),
        "beta": round(beta, 6),
        "lag": int(lag),
        "critical_values": critical,
        "p_approx": round(p_approx, 4),
        "stationary_5pct": bool(t_stat < critical["5%"]),
        "interpretation": (
            "Fiyat seviyesinde birim kök reddedilemiyor (durağan değil)."
            if t_stat >= critical["5%"]
            else "Fiyat seviyesinde birim kök reddediliyor (durağanlık kanıtı)."
        ),
    }


def variance_ratio(y: np.ndarray, periods: tuple = (2, 5, 10)) -> List[dict]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    returns = np.diff(y)
    if len(returns) < 12:
        return []
    results = []
    for q in periods:
        if n <= q + 5:
            continue
        q_returns = y[q:] - y[:-q]
        var_1 = float(np.var(returns, ddof=1))
        var_q = float(np.var(q_returns, ddof=1))
        if var_1 <= 1e-16:
            continue
        ratio = var_q / (q * var_1)
        denominator = math.sqrt(
            max(1e-12, 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n))
        )
        z = (ratio - 1.0) / denominator
        p = 2.0 * _normal_sf(abs(z))
        results.append(
            {
                "q": int(q),
                "vr": round(ratio, 4),
                "z": round(z, 3),
                "p": round(p, 4),
                "significant_5pct": bool(abs(z) > 1.96),
                "interpretation": (
                    "momentum" if ratio > 1.05 else
                    "ortalamaya dönüş" if ratio < 0.95 else "rastgele yürüyüş"
                ),
            }
        )
    return results


def half_life(y: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    if len(y) < 30:
        return None
    window = y[-min(len(y), 500):]
    fit = _ols(window[1:], window[:-1])
    if not fit:
        return None
    phi = float(fit["beta"][1])
    if not (0.0 < phi < 1.0):
        return None
    return max(0.1, -math.log(2.0) / math.log(phi))


def permutation_entropy(returns: np.ndarray, order: int = 3) -> Optional[float]:
    returns = np.asarray(returns, dtype=float)
    if len(returns) < order * 20:
        return None
    values = returns[-min(len(returns), 600):]
    patterns: Dict[tuple, int] = {}
    for index in range(len(values) - order + 1):
        window = values[index:index + order]
        pattern = tuple(int(rank) for rank in np.argsort(window, kind="stable"))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    total = sum(patterns.values())
    if total <= 0:
        return None
    probabilities = np.array([count / total for count in patterns.values()])
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    maximum = math.log(math.factorial(order))
    return entropy / maximum if maximum > 0 else None


def runs_test(returns: np.ndarray) -> dict:
    signs = np.sign(np.asarray(returns, dtype=float))
    signs = signs[signs != 0]
    n = len(signs)
    if n < 20:
        return {"available": False}
    positive = int(np.sum(signs > 0))
    negative = int(np.sum(signs < 0))
    if positive == 0 or negative == 0:
        return {"available": False}
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    expected = 2.0 * positive * negative / n + 1.0
    variance = (
        2.0 * positive * negative * (2.0 * positive * negative - n)
        / (n * n * (n - 1.0))
    )
    if variance <= 0:
        return {"available": False}
    z = (runs - expected) / math.sqrt(variance)
    return {
        "available": True,
        "runs": runs,
        "expected_runs": round(expected, 2),
        "z": round(z, 3),
        "p": round(2.0 * _normal_sf(abs(z)), 4),
        "interpretation": (
            "Getiri işaretleri rastgelelikten sapıyor."
            if abs(z) > 1.96 else "Getiri işaretleri rastgele yürüyüşle uyumlu."
        ),
    }


def autocorrelation(returns: np.ndarray, max_lag: int = 20) -> List[dict]:
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    max_lag = min(max_lag, max(1, n // 4))
    if n < 30:
        return []
    centered = returns - returns.mean()
    denominator = float(np.sum(centered ** 2))
    if denominator <= 1e-16:
        return []
    band = 1.96 / math.sqrt(n)
    result = []
    for lag in range(1, max_lag + 1):
        rho = float(np.sum(centered[:-lag] * centered[lag:]) / denominator)
        result.append(
            {
                "lag": lag,
                "rho": round(rho, 4),
                "outside_band": bool(abs(rho) > band),
            }
        )
    return result


def ljung_box(returns: np.ndarray, lags: tuple = (5, 10, 20)) -> List[dict]:
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    max_lag = min(max(lags), max(1, n // 4))
    if n < 40:
        return []
    centered = returns - returns.mean()
    denominator = float(np.sum(centered ** 2))
    if denominator <= 1e-16:
        return []
    rhos = []
    for lag in range(1, max_lag + 1):
        rhos.append(float(np.sum(centered[:-lag] * centered[lag:]) / denominator))
    results = []
    for lag in lags:
        if lag > max_lag:
            continue
        q_stat = n * (n + 2.0) * sum(
            rho ** 2 / (n - index - 1.0)
            for index, rho in enumerate(rhos[:lag])
        )
        results.append(
            {
                "lag": int(lag),
                "q": round(q_stat, 3),
                "p": round(_chi2_sf(q_stat, lag), 4),
                "autocorrelation_5pct": bool(_chi2_sf(q_stat, lag) < 0.05),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Drawdowns / risk
# ---------------------------------------------------------------------------

def drawdown_series(close: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    close = np.asarray(close, dtype=float)
    running_max = np.maximum.accumulate(close)
    drawdown = close / np.maximum(running_max, 1e-12) - 1.0
    max_index = int(np.argmin(drawdown))
    # longest underwater stretch
    longest = 0
    current = 0
    for value in drawdown:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    step = max(1, len(drawdown) // 320)
    series = [
        {
            "date": dates[index].isoformat(),
            "drawdown_pct": round(float(drawdown[index] * 100.0), 3),
        }
        for index in range(0, len(drawdown), step)
    ]
    if (len(drawdown) - 1) % step != 0:
        series.append(
            {
                "date": dates[-1].isoformat(),
                "drawdown_pct": round(float(drawdown[-1] * 100.0), 3),
            }
        )
    return {
        "series": series,
        "max_drawdown_pct": round(float(drawdown.min() * 100.0), 2),
        "max_drawdown_date": dates[max_index].isoformat(),
        "current_drawdown_pct": round(float(drawdown[-1] * 100.0), 2),
        "longest_underwater_bars": int(longest),
        "ulcer_index": round(float(np.sqrt(np.mean(drawdown ** 2)) * 100.0), 2),
    }


def _histogram(returns: np.ndarray, bins: int = 24) -> dict:
    values = returns[np.isfinite(returns)]
    if len(values) < 20:
        return {}
    low, high = np.percentile(values, [0.5, 99.5])
    if high <= low:
        return {}
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = max(int(counts.sum()), 1)
    return {
        "centers_pct": [round(float(value * 100.0), 3) for value in centers],
        "counts_pct": [round(float(value / total * 100.0), 3) for value in counts],
    }


def risk_metrics(
    close: np.ndarray, bars_per_year: float
) -> dict:
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n < 3:
        return {}
    returns = np.diff(np.log(close))
    returns = returns[np.isfinite(returns)]
    if len(returns) < 3:
        return {}
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    bpy = max(bars_per_year, 1.0)

    ann_vol = std * math.sqrt(bpy) * 100.0
    ann_return = mean * bpy * 100.0
    years = max(len(returns) / bpy, 1e-9)
    cagr = ((close[-1] / close[0]) ** (1.0 / years) - 1.0) * 100.0
    sharpe = mean / std * math.sqrt(bpy) if std > 0 else None
    sortino = mean / downside_std * math.sqrt(bpy) if downside_std > 0 else None

    running_max = np.maximum.accumulate(close)
    drawdown = close / np.maximum(running_max, 1e-12) - 1.0
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown * 100.0) if max_drawdown < 0 else None

    q05 = float(np.quantile(returns, 0.05))
    q01 = float(np.quantile(returns, 0.01))
    q95 = float(np.quantile(returns, 0.95))
    var95 = max(0.0, -q05 * 100.0)
    var99 = max(0.0, -q01 * 100.0)
    cvar95 = (
        max(0.0, -float(np.mean(returns[returns <= q05])) * 100.0)
        if np.any(returns <= q05) else None
    )
    cvar99 = (
        max(0.0, -float(np.mean(returns[returns <= q01])) * 100.0)
        if np.any(returns <= q01) else None
    )
    parametric_var95 = max(0.0, -(mean + 1.6449 * std) * 100.0)
    tail_ratio = abs(q95) / abs(q05) if abs(q05) > 1e-12 else None

    skew = float(pd.Series(returns).skew())
    kurtosis = float(pd.Series(returns).kurt())
    jb = len(returns) / 6.0 * (skew ** 2 + (kurtosis ** 2) / 4.0)
    jb_p = math.exp(-jb / 2.0) if jb > 0 else 1.0
    return {
        "total_return_pct": round((close[-1] / close[0] - 1.0) * 100.0, 2),
        "cagr_pct": round(cagr, 2),
        "annualized_return_pct": round(ann_return, 2),
        "annualized_volatility_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "sortino": round(sortino, 3) if sortino is not None else None,
        "calmar": round(calmar, 3) if calmar is not None else None,
        "max_drawdown_pct": round(max_drawdown * 100.0, 2),
        "current_drawdown_pct": round(float(drawdown[-1]) * 100.0, 2),
        "var_95_pct": round(var95, 2),
        "var_99_pct": round(var99, 2),
        "parametric_var_95_pct": round(parametric_var95, 2),
        "cvar_95_pct": round(cvar95, 2) if cvar95 is not None else None,
        "cvar_99_pct": round(cvar99, 2) if cvar99 is not None else None,
        "tail_ratio": round(tail_ratio, 3) if tail_ratio is not None else None,
        "win_rate_pct": round(float(np.mean(returns > 0)) * 100.0, 2),
        "best_bar_pct": round(float(returns.max()) * 100.0, 2),
        "worst_bar_pct": round(float(returns.min()) * 100.0, 2),
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurtosis, 3),
        "jarque_bera": round(jb, 2),
        "jarque_bera_p": round(jb_p, 6),
        "observations": int(len(returns)),
    }


def liquidity_metrics(
    close: np.ndarray, volume: np.ndarray, quote_volume: Optional[np.ndarray] = None
) -> dict:
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if len(close) < 30:
        return {}
    returns = np.diff(np.log(close))
    turnover = (
        np.asarray(quote_volume, dtype=float)
        if quote_volume is not None and len(quote_volume) == len(close)
        else close * volume
    )
    turnover = turnover[1:]
    mask = np.isfinite(returns) & np.isfinite(turnover) & (turnover > 0)
    amihud = None
    if np.any(mask):
        amihud = float(np.mean(np.abs(returns[mask]) / turnover[mask]) * 1e9)
    covariance = float(np.cov(returns[:-1], returns[1:], ddof=1)[0, 1]) if len(returns) > 5 else 0.0
    roll_spread = None
    if covariance < 0:
        roll_spread = float(2.0 * math.sqrt(-covariance) * 100.0)
    return {
        "amihud_illiquidity_1e9": round(amihud, 6) if amihud is not None else None,
        "roll_spread_pct": round(roll_spread, 4) if roll_spread is not None else None,
        "average_turnover": round(
            float(np.nanmean(turnover)) if len(turnover) else 0.0, 2
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def statistical_analysis(df: pd.DataFrame, bars_per_year: float = 365.0) -> dict:
    close = df["close"].astype(float).values
    n = len(close)
    if n < 80:
        return {"available": False, "reason": "İstatistik için en az 80 bar gerekli."}

    log_price = np.log(close)
    returns = np.diff(log_price)
    returns = returns[np.isfinite(returns)]
    dates = df.index

    hurst_rs_raw = hurst_rs(log_price)
    hurst_dfa_raw = hurst_dfa(log_price)
    hurst_values = [
        value for value in (hurst_rs_raw, hurst_dfa_raw) if value is not None
    ]
    hurst_raw_mean = float(np.mean(hurst_values)) if hurst_values else None
    # Standard Hurst interpretation is bounded at 1.0. Keep the raw DFA value
    # too, because DFA > 1 is a useful signal of a strongly non-stationary trend.
    hurst_rs_value = min(1.0, hurst_rs_raw) if hurst_rs_raw is not None else None
    hurst_dfa_value = min(1.0, hurst_dfa_raw) if hurst_dfa_raw is not None else None
    hurst = min(1.0, hurst_raw_mean) if hurst_raw_mean is not None else None

    adf = adf_test(log_price)
    half_life_bars = half_life(log_price)
    vr = variance_ratio(log_price)
    acf = autocorrelation(returns)
    lb = ljung_box(returns)
    pe = permutation_entropy(returns)
    runs = runs_test(returns)
    risk = risk_metrics(close, bars_per_year)
    drawdown = drawdown_series(close, dates)
    liquidity = liquidity_metrics(
        close,
        df["volume"].astype(float).values,
        df["quote_volume"].astype(float).values if "quote_volume" in df else None,
    )
    histogram = _histogram(returns)

    interpretations: List[str] = []
    if hurst is not None:
        if hurst >= 0.58:
            interpretations.append(
                f"Hurst={hurst:.2f}: fiyat serisi kalıcı/trend eğilimli; "
                "momentum stratejileri tarihsel olarak daha uyumlu."
            )
        elif hurst <= 0.42:
            interpretations.append(
                f"Hurst={hurst:.2f}: ortalamaya dönüş eğilimi baskın; "
                "aşırı hareketler sonrası düzeltme olasılığı artar."
            )
        else:
            interpretations.append(
                f"Hurst={hurst:.2f}: rastgele yürüyüşe yakın, belirgin uzun hafıza yok."
            )
    if vr:
        significant = [item for item in vr if item["significant_5pct"]]
        if significant:
            latest = significant[-1]
            interpretations.append(
                f"Varyans oranı q={latest['q']}: VR={latest['vr']:.2f} "
                f"(z={latest['z']:.2f}) → {latest['interpretation']}."
            )
    if adf.get("available"):
        interpretations.append(adf["interpretation"])
    if half_life_bars:
        interpretations.append(
            f"AR(1) şoklarının yarı ömrü ~{half_life_bars:.1f} bar; "
            "kısa yarı ömür ortalamaya dönüşü hızlandırır."
        )
    if pe is not None:
        if pe > 0.98:
            interpretations.append(
                f"Permütasyon entropisi={pe:.3f}: getiri sıralaması yüksek entropi, "
                "rastgeleliğe çok yakın."
            )
        elif pe < 0.90:
            interpretations.append(
                f"Permütasyon entropisi={pe:.3f}: getirilerde tekrarlayan düzenler var."
            )
    if runs.get("available") and abs(runs["z"]) > 1.96:
        interpretations.append(runs["interpretation"])
    if risk.get("jarque_bera_p", 1.0) < 0.05:
        interpretations.append(
            "Getiri dağılımı normalden anlamlı sapıyor "
            f"(çarpıklık={risk['skewness']:.2f}, basıklık={risk['excess_kurtosis']:.2f}); "
            "kuyruk riski normal dağılım varsayımından yüksek."
        )
    if risk.get("annualized_volatility_pct") is not None:
        interpretations.append(
            f"Yıllıklandırılmış volatilite %{risk['annualized_volatility_pct']:.1f}; "
            f"tarihsel VaR(95) tek bar için %{risk['var_95_pct']:.2f}."
        )

    return {
        "available": True,
        "return_stats": {
            "observations": int(len(returns)),
            "mean_bar_pct": round(float(np.mean(returns)) * 100.0, 4),
            "median_bar_pct": round(float(np.median(returns)) * 100.0, 4),
            "std_bar_pct": round(float(np.std(returns, ddof=1)) * 100.0, 4),
            "positive_bars_pct": round(float(np.mean(returns > 0)) * 100.0, 2),
        },
        "risk": risk,
        "distribution": {
            "histogram": histogram,
            "skewness": risk.get("skewness"),
            "excess_kurtosis": risk.get("excess_kurtosis"),
            "jarque_bera_p": risk.get("jarque_bera_p"),
        },
        "long_memory": {
            "hurst_rs": round(hurst_rs_value, 3) if hurst_rs_value is not None else None,
            "hurst_dfa": round(hurst_dfa_value, 3) if hurst_dfa_value is not None else None,
            "hurst_rs_raw": round(hurst_rs_raw, 3) if hurst_rs_raw is not None else None,
            "hurst_dfa_raw": round(hurst_dfa_raw, 3) if hurst_dfa_raw is not None else None,
            "hurst_raw_mean": round(hurst_raw_mean, 3) if hurst_raw_mean is not None else None,
            "nonstationary_dfa": bool(hurst_dfa_raw is not None and hurst_dfa_raw > 1.05),
            "hurst": round(hurst, 3) if hurst is not None else None,
            "regime": (
                "kalıcı/trend" if hurst is not None and hurst >= 0.58 else
                "ortalamaya dönüş" if hurst is not None and hurst <= 0.42 else
                "rastgele yürüyüş" if hurst is not None else "bilinmiyor"
            ),
        },
        "mean_reversion": {
            "adf": adf,
            "half_life_bars": round(half_life_bars, 2) if half_life_bars is not None else None,
            "variance_ratio": vr,
            "autocorrelation": acf,
            "ljung_box": lb,
        },
        "randomness": {
            "permutation_entropy": round(pe, 4) if pe is not None else None,
            "runs_test": runs,
        },
        "liquidity": liquidity,
        "drawdown": {
            key: value for key, value in drawdown.items() if key != "series"
        },
        "drawdown_series": drawdown["series"],
        "interpretations": interpretations,
        "method_notes": [
            "Hurst: R/S ve DFA; 0.5 rastgele yürüyüş, DFA>1 güçlü trend/durağan olmayan seri.",
            "ADF: sabitli model, MacKinnon kritik değerleri (yaklaşık p).",
            "Varyans oranı: Lo-MacKinlay z istatistiği (overlapping getiri).",
            "VaR/CVaR: tarihsel yüzdelikler; normal dağılım varsaymaz.",
        ],
    }

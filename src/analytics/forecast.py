"""Multi-model probabilistic forecasting with walk-forward validation.

Models
------
drift    : random walk with a robust drift
linear   : log-linear trend (capped)
holt     : Holt's linear exponential smoothing
ar       : AR(p) on log returns, AIC selected
ou       : Ornstein-Uhlenbeck mean reversion on log price
harmonic : Fourier harmonic regression + trend
ssa      : recurrent Singular Spectrum Analysis
analog   : k-nearest-neighbour analog forecasting on market-state features

The ensemble weights are learned from out-of-sample walk-forward errors, not
from in-sample fit. Prediction bands combine block-bootstrap Monte Carlo with
the empirical walk-forward error distribution.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import rsi
from .spectral import harmonic_model, harmonic_project, ssa_forecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value, digits: int = 4) -> Optional[float]:
    number = _finite(value)
    return None if number is None else round(number, digits)


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _max_log_move(bars_per_year: float) -> float:
    """Cap extrapolated trends at roughly a 4x move per year."""
    return math.log(4.0) / max(bars_per_year, 1.0)


def _sanitize(path: np.ndarray, last_log: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 1 or len(path) == 0:
        raise ValueError("empty forecast")
    if not np.all(np.isfinite(path)):
        raise ValueError("non-finite forecast")
    return np.clip(path, last_log - math.log(5.0), last_log + math.log(5.0))


def _fallback_drift(y: np.ndarray, h: int, bars_per_year: float) -> np.ndarray:
    window = min(len(y), 180)
    returns = np.diff(y[-window:]) if window > 2 else np.array([0.0])
    sigma = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    clipped = np.clip(returns, -3.0 * sigma, 3.0 * sigma) if sigma > 0 else returns
    mu = float(np.mean(clipped)) if len(clipped) else 0.0
    mu = _clip(mu, -_max_log_move(bars_per_year), _max_log_move(bars_per_year))
    return y[-1] + mu * np.arange(1, h + 1, dtype=float)


# ---------------------------------------------------------------------------
# Individual models
# ---------------------------------------------------------------------------

def _m_drift(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    return _fallback_drift(y, h, ctx["bars_per_year"])


def _m_linear(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    window = min(len(y), 240)
    if window < 30:
        raise ValueError("linear: yetersiz veri")
    values = y[-window:]
    t = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(t, values, 1)
    max_slope = _max_log_move(ctx["bars_per_year"])
    slope = _clip(float(slope), -max_slope, max_slope)
    future_t = np.arange(window, window + h, dtype=float)
    return intercept + slope * future_t


def _m_holt(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    window = min(len(y), 320)
    values = y[-window:]
    if len(values) < 20:
        raise ValueError("holt: yetersiz veri")
    best = None
    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50, 0.70):
        for beta in (0.01, 0.03, 0.06, 0.10, 0.20):
            level = float(values[0])
            trend = float(values[1] - values[0])
            sse = 0.0
            for value in values[1:]:
                previous_level = level
                level = alpha * float(value) + (1.0 - alpha) * (level + trend)
                trend = beta * (level - previous_level) + (1.0 - beta) * trend
                sse += (float(value) - level) ** 2
            if best is None or sse < best[0]:
                best = (sse, level, trend)
    if best is None or not math.isfinite(best[0]):
        raise ValueError("holt: fit başarısız")
    _, level, trend = best
    max_slope = _max_log_move(ctx["bars_per_year"])
    trend = _clip(trend, -max_slope, max_slope)
    damping = 0.995
    steps = np.arange(1, h + 1, dtype=float)
    damped_sum = (1.0 - damping ** steps) / (1.0 - damping)
    return level + trend * damped_sum


def _m_ar(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    window = min(len(y), 600)
    returns = np.diff(y[-window:])
    n = len(returns)
    if n < 40:
        raise ValueError("ar: yetersiz veri")
    max_p = min(10, n // 8)
    best = None
    for p in range(1, max_p + 1):
        rows = n - p
        target = returns[p:]
        columns = [np.ones(rows)]
        for lag in range(1, p + 1):
            columns.append(returns[p - lag:n - lag])
        design = np.column_stack(columns)
        try:
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = target - design @ beta
        sse = float(residual @ residual)
        if sse <= 0:
            continue
        aic = rows * math.log(sse / rows) + 2.0 * (p + 1)
        if best is None or aic < best[0]:
            best = (aic, beta, p)
    if best is None:
        raise ValueError("ar: model seçilemedi")
    _, beta, p = best
    intercept = float(beta[0])
    phi = np.asarray(beta[1:], dtype=float)
    if np.sum(phi) >= 1.0:
        phi = phi * (0.97 / np.sum(phi))
    history = list(returns[-p:])
    predictions = []
    for _ in range(h):
        value = intercept + float(np.dot(phi, np.asarray(history[-p:][::-1])))
        predictions.append(value)
        history.append(value)
    path = y[-1] + np.cumsum(predictions)
    return path


def _m_ou(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    window = min(len(y), 500)
    values = y[-window:]
    if len(values) < 40:
        raise ValueError("ou: yetersiz veri")
    target = values[1:]
    design = np.column_stack([np.ones(len(target)), values[:-1]])
    try:
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("ou: fit başarısız")
    intercept, phi = float(beta[0]), float(beta[1])
    if not (0.0 < phi < 0.999):
        raise ValueError("ou: phi uygun değil")
    path = []
    current = float(values[-1])
    level = intercept / (1.0 - phi)
    for _ in range(h):
        current = intercept + phi * current
        path.append(current)
    # A pure OU forecast converges to the long-run level; if the estimated
    # level is far outside the sample, keep the path bounded.
    path = np.asarray(path)
    if not (np.min(values) - 1.0 <= level <= np.max(values) + 1.0):
        path = values[-1] + (path - values[-1]) * 0.5
    return path


def _m_harmonic(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    model = harmonic_model(y, max_cycles=4, refine=not ctx.get("fast", False))
    if not model or model.get("r2", 0.0) < 0.005:
        raise ValueError("harmonic: anlamlı döngü yok")
    projection = harmonic_project(model, h)
    return np.asarray(projection["log"], dtype=float)


def _m_ssa(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    path = ssa_forecast(y, h)
    if path is None:
        raise ValueError("ssa: kararsız")
    return np.asarray(path, dtype=float)


def _feature_frame(df: pd.DataFrame, bars_per_year: float) -> pd.DataFrame:
    close = df["close"].astype(float)
    log_price = np.log(close)
    returns = log_price.diff()
    features = pd.DataFrame(index=df.index)
    features["_pos"] = np.arange(len(features), dtype=int)
    features["r1"] = returns
    features["r5"] = log_price.diff(5)
    features["r20"] = log_price.diff(20)
    features["vol20"] = returns.rolling(20, min_periods=20).std() * math.sqrt(bars_per_year)
    features["dist_sma20"] = close / close.rolling(20, min_periods=20).mean() - 1.0
    features["dist_sma50"] = close / close.rolling(50, min_periods=50).mean() - 1.0
    features["rsi14"] = rsi(close, 14) / 100.0
    features["drawdown60"] = (
        close / close.rolling(60, min_periods=20).max() - 1.0
    )
    volume = df["volume"].astype(float)
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)
    features["volume_z"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)
    features = features.iloc[60:].replace([np.inf, -np.inf], np.nan).dropna()
    return features


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.median(values))
    cutoff = cumulative[-1] / 2.0
    index = int(np.searchsorted(cumulative, cutoff))
    return float(values[min(index, len(values) - 1)])


def _m_analog(y: np.ndarray, h: int, ctx: dict) -> np.ndarray:
    df = ctx.get("df")
    if df is None or len(df) != len(y) or len(y) < 160:
        raise ValueError("analog: yetersiz veri")
    features = _feature_frame(df, ctx["bars_per_year"])
    if len(features) < 80:
        raise ValueError("analog: özellik yetersiz")
    positions = features["_pos"].to_numpy(dtype=int)
    features = features.drop(columns=["_pos"])
    matrix = features.values.astype(float)
    query = matrix[-1]
    median = np.median(matrix[:-1], axis=0)
    iqr = np.percentile(matrix[:-1], 75, axis=0) - np.percentile(
        matrix[:-1], 25, axis=0
    )
    scale = np.where(iqr > 1e-12, iqr, np.std(matrix[:-1], axis=0) + 1e-12)
    standardized = (matrix - median) / scale
    query_z = standardized[-1]

    candidate_mask = positions <= len(y) - h - 1
    # Avoid trivially close recent states whose forward window overlaps now.
    candidate_mask &= positions <= len(y) - h - 6
    candidate_positions = positions[candidate_mask]
    candidate_vectors = standardized[candidate_mask]
    if len(candidate_positions) < 25:
        raise ValueError("analog: komşu yetersiz")
    distances = np.sqrt(np.sum((candidate_vectors - query_z) ** 2, axis=1))
    k = min(30, len(distances))
    nearest = np.argsort(distances)[:k]
    forward = y[candidate_positions[nearest] + h - 1] - y[candidate_positions[nearest]]
    weights = 1.0 / (distances[nearest] + 0.25)
    analog_return = _weighted_median(forward, weights)
    drift = _fallback_drift(y, h, ctx["bars_per_year"])
    drift_return = float(drift[-1] - y[-1])
    combined = 0.75 * analog_return + 0.25 * drift_return
    steps = np.arange(1, h + 1, dtype=float) / h
    return y[-1] + combined * steps


METHODS = [
    {"key": "drift", "label": "Drift (Rastgele Yürüyüş)", "fn": _m_drift,
     "note": "Geçmiş ortalama getiri; trend yok varsayımına yakın."},
    {"key": "linear", "label": "Doğrusal Trend", "fn": _m_linear,
     "note": "Log fiyat üzerinde OLS trend; aşırı eğimler sınırlandı."},
    {"key": "holt", "label": "Holt Üstel Düzeltme", "fn": _m_holt,
     "note": "Seviye + trend bileşeni; grid search ile alpha/beta."},
    {"key": "ar", "label": "AR(p) Otoregresif", "fn": _m_ar,
     "note": "Log getirilerde AIC ile seçilmiş otoregresif model."},
    {"key": "ou", "label": "OU Ortalamaya Dönüş", "fn": _m_ou,
     "note": "Log fiyat için Ornstein-Uhlenbeck ortalamaya dönüş."},
    {"key": "harmonic", "label": "Fourier Harmonik", "fn": _m_harmonic,
     "note": "Trend + baskın sinüzoidler; döngüsel projeksiyon."},
    {"key": "ssa", "label": "SSA (Tekil Spektrum)", "fn": _m_ssa,
     "note": "Trajectory SVD + recurrent SSA forecast."},
    {"key": "analog", "label": "Analog (k-NN)", "fn": _m_analog,
     "note": "Benzer piyasa durumlarının tarihsel ileri getirileri."},
]


def _run_method(
    method: dict, y: np.ndarray, h: int, ctx: dict
) -> Tuple[Optional[np.ndarray], str]:
    try:
        path = method["fn"](y, h, ctx)
        path = _sanitize(path, float(y[-1]))
        if len(path) != h:
            raise ValueError("path length mismatch")
        return path, "ok"
    except Exception:
        return None, "başarısız"


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward(
    y: np.ndarray, h: int, ctx: dict, max_origins: int = 12
) -> dict:
    n = len(y)
    warmup = 110
    if n - h - 1 <= warmup + 10:
        return {"available": False, "reason": "Walk-forward için yetersiz veri"}

    max_possible = (n - h - 1 - warmup) // 5 + 1
    n_eval = min(max_origins, int(max(0, max_possible)))
    if n_eval < 4:
        return {"available": False, "reason": "En az 4 test penceresi gerekli"}

    step = max(5, (n - h - 1 - warmup) // max(1, n_eval - 1))
    origins = [n - h - 1 - index * step for index in range(n_eval)][::-1]
    origins = [origin for origin in origins if origin >= warmup]
    if len(origins) < 4:
        return {"available": False, "reason": "Yetersiz test penceresi"}

    records: Dict[str, List[dict]] = {method["key"]: [] for method in METHODS}
    baseline_errors = []
    ensemble_records = []

    for origin in origins:
        prefix = y[:origin]
        actual = float(y[origin + h - 1])
        prefix_ctx = {**ctx, "df": ctx["df"].iloc[:origin], "fast": True}
        predictions = {}
        for method in METHODS:
            path, status = _run_method(method, prefix, h, prefix_ctx)
            if path is None:
                continue
            error_log = float(path[-1] - actual)
            percentage = (math.exp(error_log) - 1.0) * 100.0
            start_direction = 1 if path[-1] > prefix[-1] else (-1 if path[-1] < prefix[-1] else 0)
            actual_direction = 1 if actual > prefix[-1] else (-1 if actual < prefix[-1] else 0)
            predictions[method["key"]] = float(path[-1])
            records[method["key"]].append(
                {
                    "error_log": error_log,
                    "error_pct": percentage,
                    "abs_pct": abs(percentage),
                    "direction_ok": start_direction == actual_direction if start_direction and actual_direction else None,
                }
            )
        baseline_error = (math.exp(actual - prefix[-1]) - 1.0) * 100.0
        baseline_errors.append(abs(baseline_error))
        if predictions:
            ensemble_records.append(
                {"predictions": predictions, "actual": actual, "start": float(prefix[-1])}
            )

    stats = {}
    for method in METHODS:
        key = method["key"]
        values = records[key]
        if not values:
            stats[key] = {
                "valid": 0, "mae_pct": None, "rmse_pct": None,
                "bias_pct": None, "hit_rate": None, "status": "başarısız",
            }
            continue
        errors = np.array([item["error_pct"] for item in values])
        directions = [item["direction_ok"] for item in values if item["direction_ok"] is not None]
        stats[key] = {
            "valid": int(len(values)),
            "mae_pct": round(float(np.mean(np.abs(errors))), 3),
            "rmse_pct": round(float(np.sqrt(np.mean(errors ** 2))), 3),
            "bias_pct": round(float(np.mean(errors)), 3),
            "hit_rate": round(float(np.mean(directions)) * 100.0, 1) if directions else None,
            "status": "ok" if len(values) >= max(3, len(origins) - 2) else "kısmi",
        }

    valid_methods = [
        method for method in METHODS
        if stats[method["key"]]["mae_pct"] is not None
        and stats[method["key"]]["valid"] >= max(2, len(origins) // 2)
    ]
    weights = {method["key"]: 0.0 for method in METHODS}
    if valid_methods:
        raw = np.array(
            [1.0 / (stats[method["key"]]["mae_pct"] + 0.35) ** 2 for method in valid_methods]
        )
        data_weights = raw / raw.sum()
        prior = 1.0 / len(valid_methods)
        prior_mix = 0.35 if len(origins) < 8 else 0.20
        blended = (1.0 - prior_mix) * data_weights + prior_mix * prior
        blended = np.minimum(blended, 0.38)
        blended = blended / blended.sum()
        for method, weight in zip(valid_methods, blended):
            weights[method["key"]] = float(weight)

    ensemble_errors = []
    ensemble_directions = []
    for record in ensemble_records:
        weighted = []
        total = 0.0
        for key, prediction in record["predictions"].items():
            weight = weights.get(key, 0.0)
            if weight > 0:
                weighted.append(weight * prediction)
                total += weight
        if total <= 0:
            continue
        prediction = sum(weighted) / total
        error = (math.exp(prediction - record["actual"]) - 1.0) * 100.0
        ensemble_errors.append(error)
        direction_1 = 1 if prediction > record["start"] else (-1 if prediction < record["start"] else 0)
        direction_2 = 1 if record["actual"] > record["start"] else (-1 if record["actual"] < record["start"] else 0)
        if direction_1 and direction_2:
            ensemble_directions.append(direction_1 == direction_2)

    baseline_mape = float(np.mean(baseline_errors)) if baseline_errors else None
    ensemble_mape = float(np.mean(np.abs(ensemble_errors))) if ensemble_errors else None
    ensemble_hit = (
        float(np.mean(ensemble_directions)) * 100.0 if ensemble_directions else None
    )
    return {
        "available": True,
        "origins": len(origins),
        "origin_dates": [pd.Timestamp(ctx["df"].index[origin]).isoformat() for origin in origins],
        "horizon": int(h),
        "method_stats": stats,
        "weights": weights,
        "baseline_mape_pct": round(baseline_mape, 3) if baseline_mape is not None else None,
        "ensemble_mape_pct": round(ensemble_mape, 3) if ensemble_mape is not None else None,
        "ensemble_hit_rate": round(ensemble_hit, 1) if ensemble_hit is not None else None,
        "skill_score": (
            round(1.0 - ensemble_mape / baseline_mape, 3)
            if ensemble_mape is not None and baseline_mape and baseline_mape > 0 else None
        ),
        "ensemble_errors_log": [
            float(value) for value in np.array(ensemble_errors) / 100.0
        ],
    }


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo(
    y: np.ndarray,
    h: int,
    target_log: float,
    n_paths: int = 1800,
    seed: int = 42,
) -> dict:
    n = len(y)
    lookback = min(600, n - 1)
    returns = np.diff(y[-lookback - 1:])
    returns = returns[np.isfinite(returns)]
    if len(returns) < 20:
        returns = np.array([0.0])
    block = max(2, min(20, int(round(math.sqrt(h)))))
    rng = np.random.default_rng(seed)
    blocks_needed = int(math.ceil(h / block))
    max_start = max(1, len(returns) - block)
    starts = rng.integers(0, max_start, size=(n_paths, blocks_needed))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)[:, :h]
    sampled = returns[indices]
    cumulative = np.cumsum(sampled, axis=1)

    # Center the simulated median on the model ensemble target.
    median_final = float(np.median(cumulative[:, -1]))
    adjustment = (target_log - y[-1]) - median_final
    cumulative = cumulative + (adjustment * np.arange(1, h + 1) / h)[None, :]
    prices = np.exp(y[-1] + cumulative)

    quantile_levels = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    terminal = np.quantile(prices[:, -1], quantile_levels)
    terminal_dict = {
        f"q{int(level * 100):02d}": round(float(value), 8)
        for level, value in zip(quantile_levels, terminal)
    }
    step_quantiles = {}
    for level in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        step_quantiles[f"q{int(level * 100):02d}"] = [
            round(float(value), 8)
            for value in np.quantile(prices, level, axis=0)
        ]

    current = float(np.exp(y[-1]))
    up_probability = float(np.mean(prices[:, -1] > current))
    max_drawdown = np.min(prices / current - 1.0, axis=1)
    max_runup = np.max(prices / current - 1.0, axis=1)

    def touch_probability(level: float, above: bool) -> Optional[float]:
        if above:
            touched = np.max(prices, axis=1) >= current * (1.0 + level)
        else:
            touched = np.min(prices, axis=1) <= current * (1.0 - level)
        return float(np.mean(touched))

    def median_first_passage(level: float, above: bool) -> Optional[float]:
        threshold = current * (1.0 + level) if above else current * (1.0 - level)
        hits = []
        for path in prices:
            if above:
                indices_hit = np.where(path >= threshold)[0]
            else:
                indices_hit = np.where(path <= threshold)[0]
            if len(indices_hit):
                hits.append(int(indices_hit[0]) + 1)
        return float(np.median(hits)) if hits else None

    low, high = np.quantile(prices[:, -1], [0.01, 0.99])
    if high <= low:
        low, high = current * 0.8, current * 1.2
    counts, edges = np.histogram(prices[:, -1], bins=32, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = max(int(counts.sum()), 1)
    histogram = {
        "prices": [round(float(value), 8) for value in centers],
        "counts_pct": [round(float(value / total * 100.0), 3) for value in counts],
    }

    return {
        "paths": int(n_paths),
        "block_size": int(block),
        "terminal": terminal_dict,
        "step_quantiles": step_quantiles,
        "up_probability": round(up_probability * 100.0, 2),
        "expected_max_drawdown_pct": round(float(np.median(max_drawdown)) * 100.0, 2),
        "worst_decile_drawdown_pct": round(float(np.quantile(max_drawdown, 0.10)) * 100.0, 2),
        "expected_max_runup_pct": round(float(np.median(max_runup)) * 100.0, 2),
        "touch_up_10pct": round((touch_probability(0.10, True) or 0.0) * 100.0, 2),
        "touch_down_10pct": round((touch_probability(0.10, False) or 0.0) * 100.0, 2),
        "median_bars_to_up_10pct": median_first_passage(0.10, True),
        "median_bars_to_down_10pct": median_first_passage(0.10, False),
        "histogram": histogram,
    }


# ---------------------------------------------------------------------------
# Horizon helpers
# ---------------------------------------------------------------------------

def _horizon_label(h: int, interval: str, market: str) -> str:
    if interval == "1d":
        unit = "işlem günü" if market in ("us", "bist") else "gün"
        return f"{h} {unit}"
    if interval == "1w":
        return f"{h} hafta"
    if interval == "4h":
        return f"{h * 4} saat"
    if interval == "1h":
        return f"{h} saat"
    return f"{h} bar"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def forecast_ensemble(
    df: pd.DataFrame,
    horizon: int = 30,
    bars_per_year: float = 365.0,
    interval: str = "1d",
    market: str = "crypto",
) -> dict:
    close = df["close"].astype(float).values
    n = len(close)
    horizon = int(max(1, min(365, horizon)))
    if n < 120:
        return {"available": False, "reason": "Tahmin için en az 120 bar gerekli."}

    y = np.log(close)
    ctx = {
        "df": df,
        "bars_per_year": bars_per_year,
        "interval": interval,
        "market": market,
    }

    walk = walk_forward(y, horizon, ctx)
    weights = walk.get("weights", {}) if walk.get("available") else {}
    method_stats = walk.get("method_stats", {}) if walk.get("available") else {}

    method_rows = []
    current_paths = {}
    for method in METHODS:
        path, status = _run_method(method, y, horizon, ctx)
        if path is not None:
            current_paths[method["key"]] = path
        stats = method_stats.get(method["key"], {})
        weight = float(weights.get(method["key"], 0.0)) if path is not None else 0.0
        final = float(path[-1]) if path is not None else None
        method_rows.append(
            {
                "key": method["key"],
                "label": method["label"],
                "forecast_price": round(float(np.exp(final)), 8) if final is not None else None,
                "change_pct": round((math.exp(final - y[-1]) - 1.0) * 100.0, 2) if final is not None else None,
                "weight": round(weight, 4),
                "mae_pct": stats.get("mae_pct"),
                "rmse_pct": stats.get("rmse_pct"),
                "bias_pct": stats.get("bias_pct"),
                "hit_rate": stats.get("hit_rate"),
                "status": stats.get("status", status),
                "note": method["note"],
            }
        )

    if not current_paths:
        return {"available": False, "reason": "Hiçbir tahmin modeli çalışmadı."}

    if sum(weights.get(key, 0.0) for key in current_paths) <= 0:
        equal = 1.0 / len(current_paths)
        weights = {key: equal for key in current_paths}
    # Renormalise over models that produced a current forecast.
    total_weight = sum(weights.get(key, 0.0) for key in current_paths)
    weights = {
        key: (weights.get(key, 0.0) / total_weight if total_weight > 0 else 1.0 / len(current_paths))
        for key in current_paths
    }
    for row in method_rows:
        row["weight"] = round(float(weights.get(row["key"], 0.0)), 4)

    ensemble_path = np.zeros(horizon, dtype=float)
    for key, path in current_paths.items():
        ensemble_path += weights[key] * path
    drift_path = _fallback_drift(y, horizon, bars_per_year)
    ensemble_path = 0.90 * ensemble_path + 0.10 * drift_path
    ensemble_path = _sanitize(ensemble_path, float(y[-1]))
    target_log = float(ensemble_path[-1])
    target_price = float(np.exp(target_log))
    change_pct = (target_price / close[-1] - 1.0) * 100.0

    # Monte Carlo fan
    mc = monte_carlo(y, horizon, target_log)

    # Walk-forward error quantiles for calibrated bands
    calibrated = {}
    errors = walk.get("ensemble_errors_log", []) if walk.get("available") else []
    if len(errors) >= 8:
        errors = np.asarray(errors, dtype=float)
        errors = errors - np.median(errors)
        calibrated = {
            "low80": float(np.exp(target_log + np.quantile(errors, 0.10))),
            "high80": float(np.exp(target_log + np.quantile(errors, 0.90))),
            "low95": float(np.exp(target_log + np.quantile(errors, 0.05))),
            "high95": float(np.exp(target_log + np.quantile(errors, 0.95))),
            "samples": int(len(errors)),
        }

    mc_low80 = float(mc["step_quantiles"]["q10"][-1])
    mc_high80 = float(mc["step_quantiles"]["q90"][-1])
    mc_low95 = float(mc["step_quantiles"]["q05"][-1])
    mc_high95 = float(mc["step_quantiles"]["q95"][-1])
    low80 = min(mc_low80, calibrated.get("low80", mc_low80))
    high80 = max(mc_high80, calibrated.get("high80", mc_high80))
    low95 = min(mc_low95, calibrated.get("low95", mc_low95))
    high95 = max(mc_high95, calibrated.get("high95", mc_high95))

    # Horizon table
    horizon_values = sorted(
        {
            value for value in (1, 5, 10, 20, 30, 60, 90)
            if value <= horizon
        } | {horizon}
    )
    horizons = []
    scale = math.sqrt(max(horizon, 1))
    for value in horizon_values:
        if value == horizon:
            path_final = target_log
        else:
            weighted = []
            total = 0.0
            for key, path in current_paths.items():
                # Re-run each model for the shorter horizon so non-linear
                # models (OU, analog) are not approximated by truncation.
                method = next(item for item in METHODS if item["key"] == key)
                sub_path, status = _run_method(method, y, value, ctx)
                if sub_path is None:
                    continue
                weighted.append(weights[key] * sub_path[-1])
                total += weights[key]
            if total <= 0:
                path_final = y[-1]
            else:
                path_final = 0.90 * (sum(weighted) / total) + 0.10 * _fallback_drift(
                    y, value, bars_per_year
                )[-1]
        factor = math.sqrt(value / max(horizon, 1))
        low80_value = math.exp(path_final + math.log(low80 / target_price) * factor)
        high80_value = math.exp(path_final + math.log(high80 / target_price) * factor)
        horizons.append(
            {
                "bars": int(value),
                "label": _horizon_label(value, interval, market),
                "price": round(float(math.exp(path_final)), 8),
                "change_pct": round((math.exp(path_final - y[-1]) - 1.0) * 100.0, 2),
                "low80": round(float(low80_value), 8),
                "high80": round(float(high80_value), 8),
                "direction": "yukarı" if path_final > y[-1] else ("aşağı" if path_final < y[-1] else "yatay"),
            }
        )

    # Method agreement
    final_changes = np.array(
        [float(path[-1] - y[-1]) for path in current_paths.values()], dtype=float
    )
    agreement = 1.0 - min(
        1.0, float(np.std(final_changes)) / (abs(float(np.mean(final_changes))) + 0.03)
    )
    if len(final_changes) <= 2:
        agreement *= 0.7

    baseline_mape = walk.get("baseline_mape_pct") if walk.get("available") else None
    ensemble_mape = walk.get("ensemble_mape_pct") if walk.get("available") else None
    hit_rate = walk.get("ensemble_hit_rate") if walk.get("available") else None
    if ensemble_mape is not None and baseline_mape and baseline_mape > 0:
        skill_component = _clip(0.5 + 0.5 * (1.0 - ensemble_mape / baseline_mape), 0.0, 1.0)
    else:
        skill_component = 0.5
    hit_component = (hit_rate / 100.0) if hit_rate is not None else 0.5
    data_component = min(1.0, n / 350.0)
    confidence_score = 100.0 * (
        0.30 * max(0.0, agreement)
        + 0.30 * skill_component
        + 0.25 * hit_component
        + 0.15 * data_component
    )
    confidence_label = (
        "yüksek" if confidence_score >= 70 else
        "orta" if confidence_score >= 50 else
        "düşük" if confidence_score >= 32 else "çok düşük"
    )

    interval_note = (
        "Aralıklar Monte Carlo (blok bootstrap) + walk-forward hata dağılımının "
        "dış zarfıdır; koşullu bir güven aralığı değil, belirsizlik aralığıdır."
    )

    return {
        "available": True,
        "horizon": int(horizon),
        "current_price": round(float(close[-1]), 8),
        "ensemble": {
            "price": round(target_price, 8),
            "change_pct": round(change_pct, 2),
            "low80": round(float(low80), 8),
            "high80": round(float(high80), 8),
            "low95": round(float(low95), 8),
            "high95": round(float(high95), 8),
            "up_probability": mc["up_probability"],
            "median_price": mc["terminal"]["q50"],
        },
        "bands": {
            "q05": mc["step_quantiles"]["q05"],
            "q10": mc["step_quantiles"]["q10"],
            "q25": mc["step_quantiles"]["q25"],
            "q50": mc["step_quantiles"]["q50"],
            "q75": mc["step_quantiles"]["q75"],
            "q90": mc["step_quantiles"]["q90"],
            "q95": mc["step_quantiles"]["q95"],
        },
        "horizons": horizons,
        "methods": method_rows,
        "walk_forward": {
            "available": walk.get("available", False),
            "origins": walk.get("origins"),
            "origin_dates": walk.get("origin_dates", []),
            "baseline_mape_pct": baseline_mape,
            "ensemble_mape_pct": ensemble_mape,
            "skill_score": walk.get("skill_score"),
            "hit_rate_pct": hit_rate,
            "weight_prior": "Walk-forward MAE tabanlı ters-kare ağırlık + eşit ağırlık önselleri.",
        },
        "monte_carlo": {
            key: value for key, value in mc.items()
            if key not in ("step_quantiles",)
        },
        "calibration": {
            "walk_forward_samples": calibrated.get("samples", 0),
            "note": interval_note,
        },
        "confidence": {
            "score": round(confidence_score, 1),
            "label": confidence_label,
            "components": {
                "model_agreement": round(max(0.0, agreement) * 100.0, 1),
                "walk_forward_skill": round(skill_component * 100.0, 1),
                "direction_hit_rate": round(hit_component * 100.0, 1),
                "data_quality": round(data_component * 100.0, 1),
            },
        },
        "scenarios": {
            "bearish": {
                "price": mc["terminal"]["q10"],
                "change_pct": round((mc["terminal"]["q10"] / close[-1] - 1.0) * 100.0, 2),
                "label": "Kötümser (%10 yüzdelik)",
            },
            "base": {
                "price": round(target_price, 8),
                "change_pct": round(change_pct, 2),
                "label": "Baz senaryo (ensemble)",
            },
            "bullish": {
                "price": mc["terminal"]["q90"],
                "change_pct": round((mc["terminal"]["q90"] / close[-1] - 1.0) * 100.0, 2),
                "label": "İyimser (%90 yüzdelik)",
            },
        },
        "caveats": [
            "Bu çıktı olasılıksal bir modellemedir; yatırım tavsiyesi değildir.",
            "Walk-forward pencereleri geçmişe aittir ve piyasa rejimi değişince model performansı düşebilir.",
            "Kripto varlıklarda 7/24 işlem ve likidite şokları belirsizliği artırır.",
            "Uzun ufuklarda belirsizlik doğrusal olmayan biçimde büyür; nokta tahmini tek başına kullanılmamalıdır.",
            "Bileşik skor mevcut teknik yapıyı, tahmin ise olasılıklı gelecek dağılımını ölçer; ikisi farklı yönde olabilir.",
        ],
        "method_notes": [
            "Tüm modeller log fiyat uzayında çalışır; yüzde değişimler üstel dönüşümle alınır.",
            "Walk-forward: son 12 pencere, her pencerede model sadece geçmiş veriyle yeniden fit edilir.",
            "Ensemble ağırlıkları hata büyüklüğüne göre ters-kare; maksimum ağırlık %38.",
            "Monte Carlo: blok bootstrap (blok≈√ufuk), 1800 patika, medyanı ensemble hedefe hizalanır.",
        ],
    }

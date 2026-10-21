"""Spectral and cycle analysis.

Methods
-------
* Welch PSD + local peak detection (dominant cycles)
* Least-squares harmonic regression (cos/sin + linear trend) with projection
* Hilbert transform instantaneous phase on a band-passed dominant cycle
* Haar wavelet energy distribution (multi-scale)
* Singular Spectrum Analysis (SSA): decomposition + recurrent forecast
* Spectral entropy / periodicity score

These methods describe *cyclic structure in the sample*. They do not prove
that a cycle will persist; every output is accompanied by reliability hints.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Small utilities
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


def _linfit(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return np.zeros_like(y), np.array([0.0, float(y[0]) if len(y) else 0.0])
    slope, intercept = np.polyfit(t, y, 1)
    return np.polyval([slope, intercept], t), np.array([slope, intercept])


def _detrend(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    trend, coef = _linfit(y)
    return y - trend, coef


def _fit_sinusoid(y: np.ndarray, period: float) -> Optional[dict]:
    n = len(y)
    if period <= 2.0 or period >= n * 0.75 or n < 12:
        return None
    t = np.arange(n, dtype=float)
    omega = 2.0 * math.pi / period
    design = np.column_stack(
        [np.ones(n), np.cos(omega * t), np.sin(omega * t)]
    )
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    residual = y - design @ beta
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    amplitude = float(math.hypot(beta[1], beta[2]))
    phase_angle = math.atan2(beta[2], beta[1])
    current_phase = (omega * (n - 1) - phase_angle) % (2.0 * math.pi)
    direction = -1.0 if math.sin(current_phase) > 0 else 1.0
    return {
        "period": float(period),
        "amplitude": amplitude,
        "phase_angle": phase_angle,
        "current_phase": current_phase,
        "direction": direction,
        "r2": r2,
        "beta": beta,
    }


def _refine_cycle(y: np.ndarray, period: float) -> Optional[dict]:
    low = max(4.0, period * 0.80)
    high = min(len(y) * 0.55, period * 1.20)
    if high <= low:
        return _fit_sinusoid(y, period)
    best = None
    for candidate in np.geomspace(low, high, 110):
        fit = _fit_sinusoid(y, float(candidate))
        if fit is None:
            continue
        if best is None or fit["r2"] > best["r2"]:
            best = fit
    return best or _fit_sinusoid(y, period)


def welch_psd(
    x: np.ndarray, nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density with a Hann window."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 24:
        return np.array([]), np.array([])
    if nperseg is None:
        # Use a segment long enough to resolve slow cycles (up to 128 bars)
        # while keeping several overlapping windows for stability.
        target = min(128, max(64, n // 4))
        nperseg = 1 << int(math.ceil(math.log2(target)))
        nperseg = min(128, nperseg)
    nperseg = int(min(max(32, nperseg), n))
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    window_power = float(np.sum(window ** 2))
    accumulator = np.zeros(nperseg // 2 + 1, dtype=float)
    segments = 0
    for start in range(0, n - nperseg + 1, step):
        segment = x[start:start + nperseg]
        segment = segment - segment.mean()
        spectrum = np.fft.rfft(segment * window)
        accumulator += (np.abs(spectrum) ** 2) / max(window_power, 1e-12)
        segments += 1
    if segments == 0:
        return np.array([]), np.array([])
    accumulator /= segments
    frequencies = np.fft.rfftfreq(nperseg, d=1.0)
    return frequencies[1:], accumulator[1:]


def _candidate_periods(
    periods: np.ndarray,
    power: np.ndarray,
    min_period: float = 5.0,
    max_period: Optional[float] = None,
    max_candidates: int = 8,
) -> List[float]:
    if len(periods) == 0:
        return []
    max_period = max_period or float(np.max(periods))
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return []
    p = power.copy()
    p[~mask] = 0.0
    indices = list(np.where((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]))[0] + 1)
    # The global maximum often sits on the first/last in-band bin (e.g. a
    # slow trend component), so include boundaries and the strongest bins
    # explicitly. Without this, a monotonically decaying spectrum can hide
    # the dominant cycle at the left edge.
    boundaries = [index for index in (0, len(p) - 1) if mask[index] and p[index] > 0]
    indices = list(dict.fromkeys(boundaries + indices))
    top_count = min(len(p), max(8, max_candidates * 2))
    for index in np.argsort(p)[-top_count:]:
        if p[index] > 0 and int(index) not in indices:
            indices.append(int(index))

    candidates = sorted(
        [(float(periods[i]), float(power[i])) for i in indices],
        key=lambda item: item[1],
        reverse=True,
    )

    keep: List[Tuple[float, float]] = []
    for period, score in candidates:
        if score <= 0:
            continue
        duplicate = False
        for kept_period, _ in keep:
            # Merge only nearly identical peaks; harmonic multiples (2x, 3x)
            # are legitimate regressors and must not be discarded.
            if abs(math.log(max(period / kept_period, 1e-9))) < 0.10:
                duplicate = True
                break
        if not duplicate:
            keep.append((period, score))
        if len(keep) >= max_candidates:
            break
    return [period for period, _ in keep]


# ---------------------------------------------------------------------------
# Harmonic regression
# ---------------------------------------------------------------------------

def harmonic_model(
    y: np.ndarray,
    max_cycles: int = 5,
    refine: bool = True,
    min_period: float = 5.0,
) -> Optional[dict]:
    """Fit ``y = a + b*t + sum(cos/sin)`` using dominant FFT cycles."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 60:
        return None
    detrended, _ = _detrend(y)
    frequencies, power = welch_psd(detrended)
    if len(frequencies) == 0:
        return None
    periods = 1.0 / np.maximum(frequencies, 1e-12)
    candidates = _candidate_periods(
        periods, power, min_period=min_period,
        max_period=max(min_period * 2.0, n * 0.55),
        max_candidates=max(max_cycles * 2, 8),
    )
    if not candidates:
        return None

    fits = []
    for period in candidates[: max(max_cycles * 3, 10)]:
        fit = _refine_cycle(detrended, period) if refine else _fit_sinusoid(detrended, period)
        if fit and fit["r2"] > 0.005:
            fits.append(fit)
    if not fits:
        return None
    fits.sort(key=lambda item: item["r2"], reverse=True)
    unique_fits = []
    for fit in fits:
        if any(
            abs(fit["period"] / previous["period"] - 1.0) < 0.02
            for previous in unique_fits
        ):
            continue
        unique_fits.append(fit)
    selected = unique_fits[:max_cycles]

    t = np.arange(n, dtype=float)
    columns = [np.ones(n), t / max(n, 1)]
    for fit in selected:
        omega = 2.0 * math.pi / fit["period"]
        columns.append(np.cos(omega * t))
        columns.append(np.sin(omega * t))
    design = np.column_stack(columns)
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    fitted = design @ beta
    residual = y - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    residual_std = float(np.std(residual, ddof=1)) if n > 2 else 0.0

    return {
        "beta": [float(value) for value in beta],
        "periods": [float(fit["period"]) for fit in selected],
        "r2": float(r2),
        "residual_std": residual_std,
        "n": n,
        "fitted": fitted,
        "residual": residual,
    }


def harmonic_project(model: dict, h: int) -> dict:
    """Project a harmonic model ``h`` bars forward with heuristic bands."""
    if not model:
        return {}
    n = int(model["n"])
    beta = np.asarray(model["beta"], dtype=float)
    periods = model["periods"]
    h = max(1, int(h))
    future_t = np.arange(n, n + h, dtype=float)
    columns = [np.ones(h), future_t / max(n, 1)]
    for period in periods:
        omega = 2.0 * math.pi / period
        columns.append(np.cos(omega * future_t))
        columns.append(np.sin(omega * future_t))
    design = np.column_stack(columns)
    predicted = design @ beta
    residual_std = float(model.get("residual_std") or 0.0)
    steps = np.arange(1, h + 1, dtype=float)
    widening = np.sqrt(1.0 + steps / max(n, 1))
    band80 = 1.2816 * residual_std * widening
    band95 = 1.9600 * residual_std * widening
    return {
        "log": predicted,
        "log_low80": predicted - band80,
        "log_high80": predicted + band80,
        "log_low95": predicted - band95,
        "log_high95": predicted + band95,
    }


# ---------------------------------------------------------------------------
# Hilbert transform
# ---------------------------------------------------------------------------

def _analytic_signal(x: np.ndarray) -> np.ndarray:
    n = len(x)
    spectrum = np.fft.fft(x)
    multiplier = np.zeros(n)
    if n % 2 == 0:
        multiplier[0] = 1.0
        multiplier[n // 2] = 1.0
        multiplier[1:n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier)


def _bandpass(x: np.ndarray, period: float, width: float = 0.28) -> np.ndarray:
    n = len(x)
    spectrum = np.fft.rfft(x)
    frequencies = np.fft.rfftfreq(n, d=1.0)
    center = 1.0 / max(period, 1e-9)
    mask = (frequencies >= center * (1.0 - width)) & (
        frequencies <= center * (1.0 + width)
    )
    return np.fft.irfft(spectrum * mask, n=n)


def hilbert_cycle(y: np.ndarray, period: float) -> Optional[dict]:
    if period <= 2 or len(y) < 40:
        return None
    filtered = _bandpass(y, period)
    analytic = _analytic_signal(filtered)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    phase_mod = float(phase[-1] % (2.0 * math.pi))
    instantaneous = np.diff(phase) / (2.0 * math.pi)
    recent = instantaneous[-30:]
    recent = recent[np.isfinite(recent) & (recent > 0)]
    if len(recent) < 5:
        return None
    frequency = float(np.median(recent))
    if frequency <= 0:
        return None
    inst_period = 1.0 / frequency
    to_peak = (2.0 * math.pi - phase_mod) / (2.0 * math.pi * frequency)
    if phase_mod < math.pi:
        to_trough = (math.pi - phase_mod) / (2.0 * math.pi * frequency)
    else:
        to_trough = (3.0 * math.pi - phase_mod) / (2.0 * math.pi * frequency)
    return {
        "instantaneous_period_bars": round(inst_period, 2),
        "phase_position_pct": round(phase_mod / (2.0 * math.pi) * 100.0, 2),
        "bars_to_next_peak": round(max(0.0, to_peak), 2),
        "bars_to_next_trough": round(max(0.0, to_trough), 2),
        "direction": "yukarı" if math.sin(phase_mod) < 0 else "aşağı",
        "amplitude_now": float(amplitude[-1]),
        "amplitude_median": float(np.median(amplitude[-60:])),
    }


# ---------------------------------------------------------------------------
# Haar wavelet energy
# ---------------------------------------------------------------------------

def haar_energy(y: np.ndarray) -> dict:
    current = np.asarray(y, dtype=float)
    details: List[np.ndarray] = []
    while len(current) >= 4:
        even = current[0::2]
        odd = current[1::2]
        size = min(len(even), len(odd))
        detail = (even[:size] - odd[:size]) / math.sqrt(2.0)
        approximation = (even[:size] + odd[:size]) / math.sqrt(2.0)
        details.append(detail)
        current = approximation
    if not details:
        return {}
    energies = [float(np.sum(detail ** 2)) for detail in details]
    total = sum(energies)
    if total <= 0:
        return {}
    shares = [energy / total * 100.0 for energy in energies]
    dominant = int(np.argmax(shares))
    trend_slope = 0.0
    if len(current) >= 3:
        trend_slope = float(np.polyfit(np.arange(len(current)), current, 1)[0])
    return {
        "levels": [
            {
                "scale_bars": 2 ** (index + 1),
                "energy_share_pct": round(share, 2),
            }
            for index, share in enumerate(shares)
        ],
        "dominant_scale_bars": 2 ** (dominant + 1),
        "dominant_energy_pct": round(shares[dominant], 2),
        "trend_slope_per_bar": round(trend_slope, 6),
    }


# ---------------------------------------------------------------------------
# SSA
# ---------------------------------------------------------------------------

def _diagonal_average(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    output = np.zeros(rows + cols - 1, dtype=float)
    counts = np.zeros(rows + cols - 1, dtype=float)
    for row in range(rows):
        output[row:row + cols] += matrix[row, :]
        counts[row:row + cols] += 1.0
    return output / np.maximum(counts, 1.0)


def ssa_decompose(
    y: np.ndarray, length: Optional[int] = None, max_components: int = 12
) -> Optional[dict]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 80:
        return None
    length = int(length or min(50, max(20, n // 8)))
    length = min(length, n // 2)
    cols = n - length + 1
    if cols < length:
        length = max(10, n // 2)
        cols = n - length + 1
    trajectory = np.column_stack(
        [y[start:start + length] for start in range(cols)]
    )
    try:
        u, singular, vt = np.linalg.svd(trajectory, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    variance = singular ** 2
    total = float(np.sum(variance))
    if total <= 0:
        return None
    shares = variance / total
    components = []
    count = min(max_components, len(singular))
    reconstructed: List[np.ndarray] = []
    for index in range(count):
        matrix = singular[index] * np.outer(u[:, index], vt[index, :])
        component = _diagonal_average(matrix)
        reconstructed.append(component)
        detrended, _ = _detrend(component)
        spectrum = np.abs(np.fft.rfft(detrended))
        frequencies = np.fft.rfftfreq(len(detrended), d=1.0)
        period = None
        if len(spectrum) > 2:
            peak = int(np.argmax(spectrum[1:])) + 1
            if frequencies[peak] > 0 and spectrum[peak] > 0:
                candidate = 1.0 / frequencies[peak]
                if 3 < candidate < n / 1.5:
                    period = float(candidate)
        components.append(
            {
                "index": index,
                "variance_share_pct": round(float(shares[index] * 100.0), 2),
                "period_bars": round(period, 2) if period else None,
            }
        )
    return {
        "length": int(length),
        "components": components,
        "reconstructed": reconstructed,
        "singular": [float(value) for value in singular],
        "variance_shares": [float(value) for value in shares],
    }


def ssa_forecast(
    y: np.ndarray, h: int, length: Optional[int] = None, n_components: Optional[int] = None
) -> Optional[np.ndarray]:
    """Recurrent SSA forecast (Golyandina style)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 80:
        return None
    length = int(length or min(50, max(20, n // 8)))
    length = min(length, n // 2)
    cols = n - length + 1
    if cols < length:
        length = max(10, n // 2)
        cols = n - length + 1
    trajectory = np.column_stack(
        [y[start:start + length] for start in range(cols)]
    )
    try:
        u, singular, _ = np.linalg.svd(trajectory, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if n_components is None:
        cumulative = np.cumsum(singular ** 2) / max(np.sum(singular ** 2), 1e-12)
        n_components = int(np.searchsorted(cumulative, 0.85) + 1)
        n_components = max(2, min(10, n_components))
    n_components = max(1, min(int(n_components), len(singular) - 1, 15))
    selected = u[:, :n_components]
    pi = selected[-1, :]
    nu2 = float(np.sum(pi ** 2))
    if nu2 >= 0.995 or nu2 <= 1e-9:
        return None
    recurrence = (selected[:-1, :] @ pi) / (1.0 - nu2)
    window = list(y[-length:])
    predictions = []
    recurrence_length = len(recurrence)
    for _ in range(int(h)):
        value = float(np.dot(recurrence, np.asarray(window[-recurrence_length:])))
        if not math.isfinite(value):
            return None
        # keep forecasts within a sane 5x range
        value = float(np.clip(value, y[-1] - math.log(5.0), y[-1] + math.log(5.0)))
        predictions.append(value)
        window.append(value)
    return np.asarray(predictions)


# ---------------------------------------------------------------------------
# Spectrum compression for charts
# ---------------------------------------------------------------------------

def _compress_spectrum(
    periods: np.ndarray, power: np.ndarray, max_points: int = 150
) -> Tuple[List[float], List[float]]:
    if len(periods) <= max_points:
        return [round(float(p), 3) for p in periods], [
            round(float(v), 6) for v in power
        ]
    log_periods = np.log(np.maximum(periods, 1e-9))
    bins = np.linspace(log_periods.min(), log_periods.max(), max_points + 1)
    indices = np.digitize(log_periods, bins) - 1
    out_periods, out_power = [], []
    for index in range(max_points):
        mask = indices == index
        if not np.any(mask):
            continue
        local = np.where(mask)[0]
        best = local[int(np.argmax(power[local]))]
        out_periods.append(float(periods[best]))
        out_power.append(float(power[best]))
    return (
        [round(value, 3) for value in out_periods],
        [round(value, 6) for value in out_power],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def spectrum_analysis(
    df: pd.DataFrame,
    bars_per_year: float = 365.0,
    horizon: int = 30,
) -> dict:
    close = df["close"].astype(float).values
    n = len(close)
    if n < 80:
        return {"available": False, "reason": "Spektral analiz için en az 80 bar gerekli."}

    y = np.log(close)
    detrended, trend_coef = _detrend(y)
    frequencies, power = welch_psd(detrended)
    if len(frequencies) == 0:
        return {"available": False, "reason": "Spektrum hesaplanamadı."}

    periods = 1.0 / np.maximum(frequencies, 1e-12)
    total_power = float(np.sum(power))
    candidates = _candidate_periods(
        periods, power, min_period=5.0,
        max_period=max(10.0, n * 0.55), max_candidates=8,
    )

    bar_seconds = (
        (df.index[-1] - df.index[0]).total_seconds() / max(n - 1, 1)
    )
    bar_days = bar_seconds / 86400.0

    cycles = []
    for candidate in candidates:
        fit = _refine_cycle(detrended, candidate)
        if fit is None:
            continue
        period = fit["period"]
        omega = 2.0 * math.pi / period
        next_peak = (fit["phase_angle"] + 2.0 * math.pi) / omega
        while next_peak <= n - 1:
            next_peak += period
        next_trough = (fit["phase_angle"] + math.pi) / omega
        while next_trough <= n - 1:
            next_trough += period
        # nearest peak/trough in the future
        peak_offset = next_peak - (n - 1)
        trough_offset = next_trough - (n - 1)
        if peak_offset > period:
            peak_offset -= period * math.floor(peak_offset / period)
        if trough_offset > period:
            trough_offset -= period * math.floor(trough_offset / period)

        nearest_power_index = int(np.argmin(np.abs(periods - period)))
        power_share = (
            float(power[nearest_power_index] / total_power * 100.0)
            if total_power > 0 else 0.0
        )
        # A slow component sitting on the PSD edge is usually trend/random-walk
        # leakage, not a verified cycle. Flag it instead of calling it strong.
        at_spectrum_edge = period >= 0.90 * float(np.max(periods))
        if at_spectrum_edge or period > n * 0.35:
            reliability = "sınırda"
        elif fit["r2"] > 0.15 and period <= n / 3.0:
            reliability = "yüksek"
        elif fit["r2"] > 0.05 and period <= n / 2.0:
            reliability = "orta"
        else:
            reliability = "düşük"
        cycles.append(
            {
                "period_bars": round(period, 2),
                "period_days": round(period * bar_days, 2),
                "power_share_pct": round(power_share, 2),
                "variance_explained_pct": round(fit["r2"] * 100.0, 2),
                "amplitude_pct": round((math.exp(fit["amplitude"]) - 1.0) * 100.0, 2),
                "phase_deg": round(math.degrees(fit["current_phase"]), 1),
                "cycle_position_pct": round(
                    fit["current_phase"] / (2.0 * math.pi) * 100.0, 1
                ),
                "direction": "yukarı" if fit["direction"] > 0 else "aşağı",
                "next_peak_bars": round(max(0.0, peak_offset), 1),
                "next_trough_bars": round(max(0.0, trough_offset), 1),
                "reliability": reliability,
            }
        )

    unique_cycles = []
    for cycle in cycles:
        if any(
            abs(cycle["period_bars"] / previous["period_bars"] - 1.0) < 0.02
            for previous in unique_cycles
        ):
            continue
        unique_cycles.append(cycle)
    cycles = unique_cycles
    cycles.sort(key=lambda item: item["power_share_pct"], reverse=True)
    dominant = cycles[0] if cycles else None

    # Harmonic model on the full sample
    model = harmonic_model(y, max_cycles=5, refine=True)
    harmonic = {}
    if model:
        projection = harmonic_project(model, horizon)
        fitted = model["fitted"]
        last_hist = min(n, 280)
        hist_dates = [timestamp.isoformat() for timestamp in df.index[-last_hist:]]
        harmonic = {
            "r2": round(model["r2"], 4),
            "residual_std_pct": round(
                (math.exp(model["residual_std"]) - 1.0) * 100.0, 3
            ),
            "periods_bars": [round(period, 2) for period in model["periods"]],
            "fitted": [
                {
                    "date": hist_dates[index],
                    "price": round(float(np.exp(fitted[-last_hist:][index])), 8),
                }
                for index in range(last_hist)
            ],
            "projection": {
                "log": [round(float(value), 6) for value in projection.get("log", [])],
                "price": [
                    round(float(np.exp(value)), 8)
                    for value in projection.get("log", [])
                ],
                "low80": [
                    round(float(np.exp(value)), 8)
                    for value in projection.get("log_low80", [])
                ],
                "high80": [
                    round(float(np.exp(value)), 8)
                    for value in projection.get("log_high80", [])
                ],
                "low95": [
                    round(float(np.exp(value)), 8)
                    for value in projection.get("log_low95", [])
                ],
                "high95": [
                    round(float(np.exp(value)), 8)
                    for value in projection.get("log_high95", [])
                ],
            },
        }

    # Spectral entropy / periodicity
    band_mask = (periods >= 5.0) & (periods <= n * 0.55)
    entropy = None
    periodicity = None
    if np.any(band_mask):
        values = power[band_mask]
        total = float(np.sum(values))
        if total > 0:
            probabilities = values / total
            entropy_value = -float(
                np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
            )
            entropy = entropy_value / math.log(len(probabilities))
            periodicity = 1.0 - entropy

    # Wavelet
    wavelet = haar_energy(detrended)

    # Hilbert phase of the dominant cycle
    hilbert = None
    if dominant:
        hilbert = hilbert_cycle(detrended, dominant["period_bars"])

    # SSA
    ssa = ssa_decompose(y, max_components=10)
    ssa_summary = {}
    if ssa:
        ssa_prediction = ssa_forecast(y, horizon)
        ssa_summary = {
            "length": ssa["length"],
            "components": ssa["components"],
            "dominant_period_bars": next(
                (
                    component["period_bars"]
                    for component in ssa["components"]
                    if component["period_bars"]
                ),
                None,
            ),
            "projection_price": (
                [round(float(np.exp(value)), 8) for value in ssa_prediction]
                if ssa_prediction is not None else []
            ),
            "variance_first_component_pct": ssa["components"][0]["variance_share_pct"]
            if ssa["components"] else None,
        }

    compressed_periods, compressed_power = _compress_spectrum(periods, power)
    max_power = max(compressed_power) if compressed_power else 1.0
    relative_power = [
        round(value / max_power * 100.0, 3) for value in compressed_power
    ]

    interpretations: List[str] = []
    if periodicity is not None:
        if periodicity >= 0.55:
            interpretations.append(
                "Spektral entropi düşük: fiyatta belirgin döngüsel/tonal yapı var."
            )
        elif periodicity >= 0.30:
            interpretations.append(
                "Orta düzeyde döngüsel yapı; trend ve gürültü baskın olabilir."
            )
        else:
            interpretations.append(
                "Spektrum gürültüye yakın; döngü avcılığı tek başına güvenilir değil."
            )
    if dominant:
        interpretations.append(
            f"En güçlü döngü ~{dominant['period_bars']:.0f} bar "
            f"(~{dominant['period_days']:.1f} gün), varyansın "
            f"%{dominant['variance_explained_pct']:.1f}'ini açıklıyor."
        )
        interpretations.append(
            f"Döngü şu anda {dominant['cycle_position_pct']:.0f}% konumunda ve "
            f"{dominant['direction']} yönlü; sonraki tepe ~{dominant['next_peak_bars']:.0f} bar."
        )
    if model:
        interpretations.append(
            f"Harmonik model R²={model['r2']:.2f}; açıklanmayan kısım "
            f"%{(math.exp(model['residual_std']) - 1) * 100:.2f}."
        )
    interpretations.append(
        "Uyarı: Fourier/SSA geçmiş örnekteki periyodikleri tarif eder; "
        "rejim değişiminde döngü kayabilir veya kaybolabilir."
    )

    cycle_sign = 0.0
    if dominant:
        base_strength = min(1.0, (periodicity or 0.0) * 1.4)
        fit_strength = min(1.0, dominant["variance_explained_pct"] / 20.0)
        cycle_sign = (1.0 if dominant["direction"] == "yukarı" else -1.0) * max(
            0.2, base_strength * (0.4 + 0.6 * fit_strength)
        )

    return {
        "available": True,
        "dominant_cycle": dominant,
        "cycles": cycles[:6],
        "harmonic": harmonic,
        "spectrum": {
            "periods": compressed_periods,
            "relative_power": relative_power,
            "top_periods": [cycle["period_bars"] for cycle in cycles[:5]],
        },
        "entropy": {
            "spectral_entropy": round(entropy, 4) if entropy is not None else None,
            "periodicity_score": round(periodicity, 4) if periodicity is not None else None,
            "top5_power_share_pct": round(
                min(100.0, sum(cycle["power_share_pct"] for cycle in cycles[:5])), 2
            ),
        },
        "wavelet": wavelet,
        "hilbert": hilbert,
        "ssa": ssa_summary,
        "cycle_score": round(float(cycle_sign), 3),
        "trend_log_slope_per_bar": round(float(trend_coef[0]), 8),
        "interpretation": interpretations,
        "method_notes": [
            "Welch PSD: Hann penceresi, %50 örtüşme, 128 bara kadar segment.",
            "Döngü periyodu OLS sinüzoid uyumu ile rafine edildi.",
            "Harmonik model: doğrusal trend + en iyi 5 sinüzoid.",
            "SSA: trajectory matrix SVD + recurrent forecast.",
        ],
    }

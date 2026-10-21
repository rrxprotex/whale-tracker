"""Explainable composite scoring and Turkish narrative generation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def _num(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tr_num(value, digits: int = 2) -> str:
    number = _num(value)
    if number is None:
        return "—"
    formatted = f"{number:,.{digits}f}"
    return formatted.replace(",", "§").replace(".", ",").replace("§", ".")


def _pct(value, digits: int = 2, signed: bool = True) -> str:
    number = _num(value)
    if number is None:
        return "—"
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{_tr_num(number, digits)}%"


def _price(value, currency: str = "USD") -> str:
    number = _num(value)
    if number is None:
        return "—"
    absolute = abs(number)
    digits = 2 if absolute >= 100 else (4 if absolute >= 1 else 6)
    formatted = _tr_num(number, digits)
    if currency == "TRY":
        return f"₺{formatted}"
    if currency in ("USD", "USDT", "USDC"):
        return f"${formatted}"
    return f"{formatted} {currency}"


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _group_score(technical: dict, group: str) -> Optional[float]:
    components = [
        component for component in technical.get("components", [])
        if component.get("group") == group
    ]
    if not components:
        return None
    total_weight = sum(component["weight"] for component in components)
    if total_weight <= 0:
        return None
    return sum(
        component["score"] * component["weight"] for component in components
    ) / total_weight


def _statistical_score(statistics: dict, technical: dict) -> Optional[float]:
    long_memory = statistics.get("long_memory", {})
    hurst = _num(long_memory.get("hurst"))
    mean_reversion = statistics.get("mean_reversion", {})
    vr_rows = mean_reversion.get("variance_ratio") or []
    returns = technical.get("returns", {})
    ret60 = _num(returns.get("60_bar", {}).get("pct"))
    if ret60 is None:
        ret60 = _num(returns.get("20_bar", {}).get("pct")) or 0.0

    score = 0.0
    if hurst is not None:
        if hurst >= 0.5:
            score += _clip((hurst - 0.5) * 5.0) * (1.0 if ret60 >= 0 else -1.0)
        else:
            sma20 = _num(technical.get("indicators", {}).get("sma20"))
            price = _num(technical.get("price"))
            if sma20 and price:
                deviation = (price / sma20 - 1.0) * 100.0
                score += _clip((0.5 - hurst) * 5.0) * (-1.0 if deviation > 0 else 1.0)

    significant = [row for row in vr_rows if row.get("significant_5pct")]
    if significant:
        row = significant[-1]
        direction = 1.0 if ret60 >= 0 else -1.0
        if row.get("interpretation") == "momentum":
            score += 0.30 * direction
        elif row.get("interpretation") == "ortalamaya dönüş":
            score -= 0.30 * direction
    return _clip(score)


def _forecast_score(forecast: dict, statistics: dict, technical: dict) -> Optional[float]:
    ensemble = forecast.get("ensemble") or {}
    change = _num(ensemble.get("change_pct"))
    if change is None:
        return None
    up_probability = _num(ensemble.get("up_probability"))
    up_probability = 0.5 if up_probability is None else up_probability / 100.0
    risk = statistics.get("risk", {})
    annual_vol = _num(risk.get("annualized_volatility_pct")) or 60.0
    horizon = max(1, int(forecast.get("horizon") or 30))
    horizon_vol = annual_vol * math.sqrt(horizon / 365.0)
    directional = math.tanh(change / max(horizon_vol * 1.2, 1.0))
    probability_component = (up_probability - 0.5) * 2.0
    confidence = _num(forecast.get("confidence", {}).get("score")) or 50.0
    shrink = 0.55 + 0.45 * confidence / 100.0
    return _clip((0.65 * directional + 0.35 * probability_component) * shrink)


def composite_score(
    technical: dict,
    spectral: dict,
    statistics: dict,
    patterns: dict,
    regime: dict,
    forecast: dict,
    meta: dict,
) -> dict:
    trend = _group_score(technical, "trend")
    momentum = _group_score(technical, "momentum")
    mean_reversion = _group_score(technical, "mean_reversion")
    volume = _group_score(technical, "volume")
    position = _group_score(technical, "position")
    cycle = _num(spectral.get("cycle_score"))
    statistical = _statistical_score(statistics, technical)
    pattern = _num(patterns.get("pattern_score"))
    forecast_value = _forecast_score(forecast, statistics, technical)

    adx = _num(technical.get("indicators", {}).get("adx")) or 0.0
    trend_score = _num(technical.get("trend_score")) or 0.0
    if adx >= 25 and abs(trend_score) >= 0.25:
        regime_type = "trend"
        weights = {
            "trend": 0.24, "momentum": 0.17, "mean_reversion": 0.05,
            "volume": 0.09, "cycle": 0.08, "statistics": 0.08,
            "pattern": 0.09, "forecast": 0.16, "position": 0.04,
        }
    elif adx < 18:
        regime_type = "range"
        weights = {
            "trend": 0.10, "momentum": 0.10, "mean_reversion": 0.18,
            "volume": 0.08, "cycle": 0.16, "statistics": 0.13,
            "pattern": 0.11, "forecast": 0.10, "position": 0.04,
        }
    else:
        regime_type = "mixed"
        weights = {
            "trend": 0.18, "momentum": 0.15, "mean_reversion": 0.09,
            "volume": 0.09, "cycle": 0.12, "statistics": 0.10,
            "pattern": 0.10, "forecast": 0.13, "position": 0.04,
        }

    values = {
        "trend": trend,
        "momentum": momentum,
        "mean_reversion": mean_reversion,
        "volume": volume,
        "cycle": cycle,
        "statistics": statistical,
        "pattern": pattern,
        "forecast": forecast_value,
        "position": position,
    }
    labels = {
        "trend": "Trend yapısı",
        "momentum": "Momentum",
        "mean_reversion": "Ortalamaya dönüş",
        "volume": "Hacim / birikim",
        "cycle": "Döngü fazı",
        "statistics": "İstatistiksel karakter",
        "pattern": "Fiyat formasyonları",
        "forecast": "Tahmin ensemble",
        "position": "Aralık konumu",
    }
    explanations = {
        "trend": "EMA dizilimi, fiyatın uzun ortalamaya mesafesi, ADX ve Ichimoku.",
        "momentum": "RSI, MACD histogramı ve 20 bar momentumu.",
        "mean_reversion": "Bollinger %B ve Stochastic aşırılıkları (kontraryen).",
        "volume": "OBV birikimi ve hacim teyidi.",
        "cycle": "Baskın döngünün yönü ve spektral periyodiklik.",
        "statistics": "Hurst, varyans oranı ve ortalamaya dönüş kanıtı.",
        "pattern": "Mum formasyonları, uyumsuzluklar, kanal ve ikili tepe/dip.",
        "forecast": "Walk-forward ensemble yönü ve Monte Carlo yukarı olasılığı.",
        "position": "52 bar aralığındaki konum.",
    }

    contributions = []
    composite = 0.0
    for key, weight in weights.items():
        value = values.get(key)
        if value is None:
            continue
        weighted = value * weight * 100.0
        composite += weighted
        contributions.append(
            {
                "key": key,
                "label": labels[key],
                "score": round(value, 3),
                "weight": weight,
                "contribution": round(weighted, 2),
                "explanation": explanations[key],
            }
        )
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    if composite >= 55:
        label, stance, tone = "Güçlü Boğa Eğilimi", "long", "bull"
    elif composite >= 20:
        label, stance, tone = "Boğa Eğilimi", "long", "bull"
    elif composite > -20:
        label, stance, tone = "Nötr / Kararsız", "neutral", "neutral"
    elif composite > -55:
        label, stance, tone = "Ayı Eğilimi", "short", "bear"
    else:
        label, stance, tone = "Güçlü Ayı Eğilimi", "short", "bear"

    sub_scores = [value for value in values.values() if value is not None]
    agreement = (
        1.0 - min(1.0, float(np.std(sub_scores)) / 0.75)
        if len(sub_scores) >= 3 else 0.4
    )
    forecast_confidence = _num(forecast.get("confidence", {}).get("score")) or 45.0
    bars_value = _num(meta.get("bars"))
    data_quality = min(1.0, (bars_value or 0.0) / 350.0) if bars_value is not None else 0.7
    quality_score = _num(_safe_get(meta, "quality", "score"))
    if quality_score is not None:
        data_quality *= max(0.0, min(1.0, quality_score / 100.0))
    confidence_value = 100.0 * (
        0.35 * max(0.0, agreement)
        + 0.40 * forecast_confidence / 100.0
        + 0.25 * data_quality
    )
    confidence_label = (
        "yüksek" if confidence_value >= 70 else
        "orta" if confidence_value >= 50 else
        "düşük" if confidence_value >= 32 else "çok düşük"
    )

    risk = statistics.get("risk", {})
    ensemble = forecast.get("ensemble", {})
    current_price = _num(meta.get("last_price")) or _num(technical.get("price")) or 1.0
    band_width = None
    if ensemble.get("low95") and ensemble.get("high95") and current_price:
        band_width = (
            float(ensemble["high95"]) - float(ensemble["low95"])
        ) / current_price * 100.0

    risk_points = 0
    drivers = []
    vol_percentile = _num(technical.get("indicators", {}).get("vol_percentile"))
    atr_pct = _num(technical.get("indicators", {}).get("atr_pct"))
    var95 = _num(risk.get("var_95_pct"))
    max_dd = _num(risk.get("max_drawdown_pct"))
    annual_vol = _num(risk.get("annualized_volatility_pct"))
    if annual_vol is not None and annual_vol >= 60.0:
        risk_points += 2
        drivers.append(f"Yıllık volatilite %{annual_vol:.0f}")
    elif annual_vol is not None and annual_vol >= 35.0:
        risk_points += 1
        drivers.append(f"Yıllık volatilite %{annual_vol:.0f}")
    if vol_percentile is not None and vol_percentile >= 85:
        risk_points += 1
        drivers.append(f"Volatilite yüzdelik dilimi %{vol_percentile:.0f}")
    if atr_pct is not None and atr_pct >= 4.0:
        risk_points += 1
        drivers.append(f"ATR %{atr_pct:.1f}")
    if var95 is not None and var95 >= 4.0:
        risk_points += 1
        drivers.append(f"Günlük VaR(95) %{var95:.1f}")
    if max_dd is not None and max_dd <= -40.0:
        risk_points += 2
        drivers.append(f"Maksimum düşüş %{max_dd:.0f}")
    elif max_dd is not None and max_dd <= -20.0:
        risk_points += 1
        drivers.append(f"Maksimum düşüş %{max_dd:.0f}")
    if band_width is not None and band_width >= 40.0:
        risk_points += 2
        drivers.append(f"Tahmin %95 bandı %{band_width:.0f}")
    elif band_width is not None and band_width >= 25.0:
        risk_points += 1
        drivers.append(f"Tahmin %95 bandı %{band_width:.0f}")
    if risk_points >= 8:
        risk_level = "Çok Yüksek"
    elif risk_points >= 5:
        risk_level = "Yüksek"
    elif risk_points >= 2:
        risk_level = "Orta"
    else:
        risk_level = "Düşük"

    composite_display = "0" if abs(composite) < 0.5 else f"{composite:+.0f}"
    headline = f"{label} — skor {composite_display}/100, güven {confidence_label} (%{confidence_value:.0f})"
    top = contributions[:2]
    summary = (
        f"{headline}. Baskın etkenler: "
        + ", ".join(f"{item['label']} ({item['contribution']:+.0f})" for item in top)
        + ". Skor mevcut teknik yapıyı, tahmin paneli ise olasılıklı geleceği ölçer."
    )

    result = {
        "score": round(float(composite), 1),
        "label": label,
        "stance": stance,
        "tone": tone,
        "confidence": round(float(confidence_value), 1),
        "confidence_label": confidence_label,
        "regime_type": regime_type,
        "weights": weights,
        "contributions": contributions,
        "risk": {
            "level": risk_level,
            "points": int(risk_points),
            "drivers": drivers,
            "forecast_band_width_pct": round(float(band_width), 2) if band_width is not None else None,
        },
        "summary": summary,
    }
    result["narrative"] = build_narrative(
        result, technical, spectral, statistics, patterns, regime, forecast, meta
    )
    return result


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------

def _safe_get(data: dict, *path, default=None):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def build_narrative(
    score: dict,
    technical: dict,
    spectral: dict,
    statistics: dict,
    patterns: dict,
    regime: dict,
    forecast: dict,
    meta: dict,
) -> dict:
    currency = meta.get("currency", "USD")
    price = _num(technical.get("price")) or _num(meta.get("last_price"))
    indicators = technical.get("indicators", {})
    risk = statistics.get("risk", {})
    ensemble = forecast.get("ensemble", {})
    dominant_cycle = spectral.get("dominant_cycle") or {}
    levels = patterns.get("key_levels", {})
    support = levels.get("nearest_support") or {}
    resistance = levels.get("nearest_resistance") or {}
    fib = patterns.get("fibonacci") or {}
    profile = patterns.get("volume_profile") or {}
    confidence = forecast.get("confidence", {})

    trend_bullets = [
        f"Trend rejimi: **{technical.get('trend_label', '—')}**; momentum: **{technical.get('momentum_label', '—')}**.",
        f"Fiyat {_price(price, currency)}; 200 barlık ortalamaya göre "
        f"{_pct((price / indicators['sma200'] - 1.0) * 100.0) if indicators.get('sma200') and price else '—'}.",
        f"RSI {_tr_num(indicators.get('rsi'), 1)} ({indicators.get('rsi_zone', '—')}), "
        f"ADX {_tr_num(indicators.get('adx'), 1)}, MACD histogram {_tr_num(indicators.get('macd_hist'), 4)}.",
        f"Son hacim 20-bar ortalamanın {_tr_num(_safe_get(technical, 'volume', 'relative'), 2)} katı.",
    ]

    cycle_bullets = []
    if dominant_cycle:
        cycle_bullets.append(
            f"Baskın döngü ~{_tr_num(dominant_cycle.get('period_bars'), 1)} bar "
            f"(~{_tr_num(dominant_cycle.get('period_days'), 1)} gün); "
            f"güvenilirlik **{dominant_cycle.get('reliability', '—')}**."
        )
        cycle_bullets.append(
            f"Döngü konumu %{_tr_num(dominant_cycle.get('cycle_position_pct'), 0)} ve yön **{dominant_cycle.get('direction')}**; "
            f"sonraki tepe ~{_tr_num(dominant_cycle.get('next_peak_bars'), 0)} bar, "
            f"sonraki dip ~{_tr_num(dominant_cycle.get('next_trough_bars'), 0)} bar."
        )
    else:
        cycle_bullets.append("Baskın döngü tespit edilemedi.")
    harmonic = spectral.get("harmonic") or {}
    entropy = spectral.get("entropy") or {}
    if harmonic:
        cycle_bullets.append(
            f"Harmonik model R²={_tr_num(harmonic.get('r2'), 3)}; "
            f"periyodiklik skoru {_tr_num(entropy.get('periodicity_score'), 3)}."
        )
    cycle_bullets.append(
        "Fourier/SSA geçmiş örnekteki düzenleri tarif eder; döngünün kalıcılığı garanti değildir."
    )

    long_memory = statistics.get("long_memory", {})
    mean_reversion = statistics.get("mean_reversion", {})
    hurst_rs_display = long_memory.get("hurst_rs_raw", long_memory.get("hurst_rs"))
    hurst_dfa_display = long_memory.get("hurst_dfa_raw", long_memory.get("hurst_dfa"))
    dfa_note = " (DFA>1: güçlü trend/durağan değil)" if long_memory.get("nonstationary_dfa") else ""
    statistical_bullets = [
        f"Hurst R/S {_tr_num(hurst_rs_display, 3)}, DFA {_tr_num(hurst_dfa_display, 3)}{dfa_note} "
        f"→ **{long_memory.get('regime', '—')}**; yarı ömür "
        f"{_tr_num(mean_reversion.get('half_life_bars'), 1)} bar.",
        f"ADF t={_tr_num(_safe_get(mean_reversion, 'adf', 't_stat'), 2)}; "
        f"durağanlık 5%: {'evet' if _safe_get(mean_reversion, 'adf', 'stationary_5pct') else 'hayır'}.",
        f"Yıllık volatilite {_pct(risk.get('annualized_volatility_pct'), 1, signed=False)}; "
        f"tarihsel VaR(95) {_pct(risk.get('var_95_pct'), 2, signed=False)}; maksimum düşüş {_pct(risk.get('max_drawdown_pct'))}.",
    ]
    if mean_reversion.get("variance_ratio"):
        latest_vr = mean_reversion["variance_ratio"][-1]
        statistical_bullets.append(
            f"Varyans oranı q={latest_vr.get('q')}: {_tr_num(latest_vr.get('vr'), 2)} → **{latest_vr.get('interpretation')}**."
        )

    level_bullets = []
    if support:
        level_bullets.append(
            f"Destek **{_price(support.get('price'), currency)}** "
            f"({_pct(support.get('distance_pct'))}, güç {_tr_num(support.get('strength'), 0)}/100)."
        )
    if resistance:
        level_bullets.append(
            f"Direnç **{_price(resistance.get('price'), currency)}** "
            f"({_pct(resistance.get('distance_pct'))}, güç {_tr_num(resistance.get('strength'), 0)}/100)."
        )
    if profile.get("poc"):
        level_bullets.append(
            f"Hacim profili POC {_price(profile.get('poc'), currency)}; fiyat {profile.get('position')}."
        )
    if fib:
        level_bullets.append(
            f"Fibonacci swing {_price(fib.get('swing_low'), currency)}–{_price(fib.get('swing_high'), currency)}; "
            f"mevcut geri çekilme %{_tr_num(fib.get('current_retracement_pct'), 1)}."
        )
    if levels.get("invalidation_down") and levels.get("invalidation_up"):
        level_bullets.append(
            f"Yukarı senaryo invalidation: {_price(levels.get('invalidation_down'), currency)} altı; "
            f"aşağı senaryo invalidation: {_price(levels.get('invalidation_up'), currency)} üstü."
        )

    forecast_bullets = [
        f"{forecast.get('horizon', '—')} bar ensemble hedefi **{_price(ensemble.get('price'), currency)}** "
        f"({_pct(ensemble.get('change_pct'))}).",
        f"%80 belirsizlik bandı {_price(ensemble.get('low80'), currency)} – {_price(ensemble.get('high80'), currency)}; "
        f"%95 bandı {_price(ensemble.get('low95'), currency)} – {_price(ensemble.get('high95'), currency)}.",
        f"Monte Carlo yukarı olasılığı **{_pct(ensemble.get('up_probability'), 0, signed=False)}**; "
        f"tahmin güveni **{confidence.get('label', '—')}** (%{_tr_num(confidence.get('score'), 0)}).",
    ]
    walk = forecast.get("walk_forward", {})
    if walk.get("available"):
        forecast_bullets.append(
            f"Walk-forward: {walk.get('origins')} pencere, ensemble MAPE {_pct(walk.get('ensemble_mape_pct'))}, "
            f"yön isabeti {_pct(walk.get('hit_rate_pct'), 1, signed=False)}, "
            f"baseline MAPE {_pct(walk.get('baseline_mape_pct'))}."
        )

    scenarios = forecast.get("scenarios", {})
    risk_bullets = [
        f"Risk seviyesi **{score['risk']['level']}**; sürücüler: "
        + (", ".join(score["risk"]["drivers"]) if score["risk"]["drivers"] else "belirgin tek faktör yok") + ".",
        f"Tahmin %95 bandı genişliği %{_tr_num(score['risk'].get('forecast_band_width_pct'), 1)}; "
        f"beklenen maksimum düşüş (MC medyan) {_pct(_safe_get(forecast, 'monte_carlo', 'expected_max_drawdown_pct'))}.",
    ]
    if scenarios:
        risk_bullets.append(
            f"Senaryolar — kötümser {_price(_safe_get(scenarios, 'bearish', 'price'), currency)} "
            f"({_pct(_safe_get(scenarios, 'bearish', 'change_pct'))}), "
            f"baz {_price(_safe_get(scenarios, 'base', 'price'), currency)}, "
            f"iyimser {_price(_safe_get(scenarios, 'bullish', 'price'), currency)} "
            f"({_pct(_safe_get(scenarios, 'bullish', 'change_pct'))})."
        )
    risk_bullets.append(
        "Senaryo olasılıkları Monte Carlo dağılımından türetilmiştir; gerçek sonuç bu aralığın dışına çıkabilir."
    )

    return {
        "headline": score["summary"],
        "executive_summary": " ".join(trend_bullets[:2] + forecast_bullets[:2]),
        "sections": [
            {"key": "trend", "title": "Trend ve Momentum", "bullets": trend_bullets},
            {"key": "cycle", "title": "Döngü ve Spektrum", "bullets": cycle_bullets},
            {"key": "statistics", "title": "İstatistiksel Karakter", "bullets": statistical_bullets},
            {"key": "levels", "title": "Kritik Seviyeler", "bullets": level_bullets},
            {"key": "forecast", "title": "Gelecek Projeksiyonu", "bullets": forecast_bullets},
            {"key": "risk", "title": "Risk ve Senaryolar", "bullets": risk_bullets},
        ],
        "disclaimer": (
            "Bu rapor yalnızca bilgilendirme amaçlıdır ve yatırım tavsiyesi değildir. "
            "Geçmiş performans geleceği garanti etmez; tahminler olasılıksaldır. "
            "Pozisyon almadan önce kendi risk toleransınızı ve bağımsız doğrulamayı kullanın."
        ),
    }

"""Standalone HTML report renderer.

The report is a single self-contained file: CSS, JavaScript (only for the
JSON download / print helpers) and every chart as inline SVG. It works offline
and can be e-mailed or opened directly from disk.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any, Dict, List, Optional

from .scoring import _price, _tr_num, _pct


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _get(data: Any, *path, default=None):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def _num(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cls(value) -> str:
    number = _num(value)
    if number is None:
        return "neutral"
    if number > 0:
        return "pos"
    if number < 0:
        return "neg"
    return "neutral"


def _badge(label: str, tone: str = "neutral") -> str:
    return f'<span class="badge {tone}">{_e(label)}</span>'


def _metric(label: str, value: str, note: str = "", tone: str = "") -> str:
    return (
        f'<div class="metric {tone}">'
        f'<div class="metric-label">{_e(label)}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{_e(note)}</div>'
        f"</div>"
    )


def _card(title: str, body: str, subtitle: str = "", extra: str = "") -> str:
    sub = f'<div class="card-sub">{_e(subtitle)}</div>' if subtitle else ""
    return (
        f'<section class="card {extra}">'
        f'<div class="card-head"><h2>{_e(title)}</h2>{sub}</div>'
        f"{body}</section>"
    )


def _empty(message: str) -> str:
    return f'<div class="empty">{_e(message)}</div>'


def _table(headers: List[str], rows: List[List[str]], extra: str = "") -> str:
    if not rows:
        return _empty("Veri yok.")
    head = "".join(f"<th>{_e(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="{extra}">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _bullets(items: List[str]) -> str:
    if not items:
        return _empty("Bu bölüm için veri yok.")
    return "<ul class=\"bullets\">" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def _score_bar(value: Optional[float]) -> str:
    number = _num(value) or 0.0
    width = min(50.0, abs(number) * 50.0)
    left = 50.0 if number >= 0 else 50.0 - width
    color = "pos" if number >= 0 else "neg"
    return (
        '<div class="scorebar">'
        f'<span class="bar {color}" style="left:{left:.1f}%;width:{width:.1f}%"></span>'
        '<span class="mid"></span></div>'
    )


# ---------------------------------------------------------------------------
# Subsections
# ---------------------------------------------------------------------------

def _hero(report: dict, charts: dict) -> str:
    meta = report.get("meta", {})
    technical = report.get("technical", {})
    score = report.get("score", {})
    forecast = report.get("forecast", {})
    stats = report.get("statistics", {})
    ensemble = forecast.get("ensemble", {})
    risk = stats.get("risk", {})
    currency = meta.get("currency", "USD")
    price = technical.get("price") or meta.get("last_price")
    change = technical.get("change_pct")
    tone = score.get("tone", "neutral")
    forecast_horizon = forecast.get("horizon", meta.get("forecast_horizon", 30))
    confidence = forecast.get("confidence", {})
    score_value = _num(score.get("score")) or 0.0
    score_text = "0" if abs(score_value) < 0.5 else f"{score_value:+.0f}"

    market_names = {"crypto": "Kripto", "us": "ABD", "bist": "BIST"}
    metrics = "".join(
        [
            _metric(
                "Bileşik Skor", f"{score_text}<small>/100</small>",
                score.get("label", ""), tone,
            ),
            _metric(
                "Skor Güveni", f"%{_tr_num(score.get('confidence'), 0)}",
                score.get("confidence_label", ""),
            ),
            _metric(
                "Risk", _e(score.get("risk", {}).get("level", "—")),
                " · ".join(score.get("risk", {}).get("drivers", [])[:2]) or "volatilite + kuyruk + bant",
                "neg" if score.get("risk", {}).get("level") in ("Yüksek", "Çok Yüksek") else "",
            ),
            _metric(
                "Tahmin Hedefi", _price(ensemble.get("price"), currency),
                f"{forecast_horizon} bar · {_pct(ensemble.get('change_pct'))}",
                _cls(ensemble.get("change_pct")),
            ),
            _metric(
                "Yukarı Olasılığı", _pct(ensemble.get("up_probability"), 0, signed=False),
                "Monte Carlo terminal",
            ),
            _metric(
                "Yıllık Volatilite", _pct(risk.get("annualized_volatility_pct"), 1, signed=False),
                f"VaR95 {_pct(risk.get('var_95_pct'))}",
            ),
        ]
    )

    band = ""
    if ensemble:
        band = (
            f'<div class="band-line">'
            f'<span>%80: {_price(ensemble.get("low80"), currency)} – {_price(ensemble.get("high80"), currency)}</span>'
            f'<span>%95: {_price(ensemble.get("low95"), currency)} – {_price(ensemble.get("high95"), currency)}</span>'
            f"</div>"
        )
    quality = report.get("quality", {})
    quality_html = ""
    if quality.get("warnings"):
        quality_html = (
            '<div class="callout warn"><strong>Veri kalitesi uyarısı</strong>'
            f'<span>{_e(" · ".join(quality.get("warnings", [])[:3]))}</span></div>'
        )

    return f"""
    <header class="hero">
      <div class="hero-top">
        <div>
          <div class="eyebrow">{_badge(market_names.get(meta.get("market"), meta.get("market") or "—"), "market")}
            {_badge(meta.get("interval") or "—", "neutral")}
            {_badge(meta.get("source") or "—", "muted")}</div>
          <h1>{_e(meta.get("symbol"))} <span>{_e(meta.get("name"))}</span></h1>
          <div class="price-line">
            <strong>{_price(price, currency)}</strong>
            <span class="{_cls(change)}">{_pct(change)} bugün</span>
            <span class="muted">Son bar: {_e((meta.get("last_timestamp") or "")[:16].replace("T", " "))}</span>
          </div>
        </div>
        <div class="hero-actions">
          <button onclick="window.print()" class="btn ghost">Yazdır / PDF</button>
          <button onclick="downloadJson()" class="btn">JSON indir</button>
        </div>
      </div>
      <div class="hero-grid">
        <div class="hero-gauge">{charts.get("gauge", "")}</div>
        <div class="hero-summary">
          <div class="verdict {tone}">{_e(score.get("label"))}</div>
          <p>{_e(score.get("summary"))}</p>
          {band}
          {quality_html}
          <div class="hero-meta">
            <span>Veri: {_e(meta.get("bars"))} bar</span>
            <span>Kaynak: {_e(meta.get("source"))}</span>
            <span>Üretim: {_e((report.get("generated_at") or "")[:19].replace("T", " "))} UTC</span>
          </div>
        </div>
      </div>
      <div class="metric-grid">{metrics}</div>
    </header>
    """


def _forecast_panel(report: dict) -> str:
    forecast = report.get("forecast", {})
    meta = report.get("meta", {})
    currency = meta.get("currency", "USD")
    if not forecast.get("available"):
        return _card("Gelecek Projeksiyonu", _empty(forecast.get("reason", "Tahmin üretilemedi.")))

    ensemble = forecast.get("ensemble", {})
    confidence = forecast.get("confidence", {})
    scenarios = forecast.get("scenarios", {})
    walk = forecast.get("walk_forward", {})
    mc = forecast.get("monte_carlo", {})

    scenario_rows = []
    for key, label in (("bearish", "Kötümser"), ("base", "Baz"), ("bullish", "İyimser")):
        item = scenarios.get(key, {})
        scenario_rows.append(
            [
                _e(label),
                _price(item.get("price"), currency),
                f'<span class="{_cls(item.get("change_pct"))}">{_pct(item.get("change_pct"))}</span>',
                _e(item.get("label", "—")),
            ]
        )
    scenario_html = _table(["Senaryo", "Fiyat", "Değişim", "Tanım"], scenario_rows)

    horizon_rows = []
    for item in forecast.get("horizons", []):
        horizon_rows.append(
            [
                _e(item.get("label")),
                _price(item.get("price"), currency),
                f'<span class="{_cls(item.get("change_pct"))}">{_pct(item.get("change_pct"))}</span>',
                f'{_price(item.get("low80"), currency)} – {_price(item.get("high80"), currency)}',
                _e(item.get("direction")),
            ]
        )

    method_rows = []
    for item in sorted(forecast.get("methods", []), key=lambda row: row.get("weight") or 0, reverse=True):
        method_rows.append(
            [
                _e(item.get("label")),
                _price(item.get("forecast_price"), currency),
                f'<span class="{_cls(item.get("change_pct"))}">{_pct(item.get("change_pct"))}</span>',
                f"{_tr_num((item.get('weight') or 0) * 100, 1)}%",
                _pct(item.get("mae_pct")),
                _pct(item.get("hit_rate"), 1, signed=False),
                _badge(item.get("status", "—"), "muted"),
            ]
        )

    walk_html = ""
    if walk.get("available"):
        walk_html = f"""
        <div class="callout">
          <strong>Walk-forward doğrulama</strong>
          <span>{_e(walk.get("origins"))} pencere · ensemble MAPE {_pct(walk.get("ensemble_mape_pct"))}
          · baseline {_pct(walk.get("baseline_mape_pct"))} · yön isabeti {_pct(walk.get("hit_rate_pct"), 1, signed=False)}
          · skill {_tr_num(walk.get("skill_score"), 3)}</span>
        </div>
        """
    else:
        walk_html = f'<div class="callout warn">{_e(walk.get("reason", "Walk-forward doğrulaması yapılamadı."))}</div>'

    components = confidence.get("components", {})
    confidence_html = "".join(
        f'<div class="conf-row"><span>{_e(label)}</span><div class="track"><span style="width:{min(100, max(0, _num(value) or 0)):.0f}%"></span></div><b>{_tr_num(value, 0)}</b></div>'
        for label, value in (
            ("Model uyumu", components.get("model_agreement")),
            ("Walk-forward becerisi", components.get("walk_forward_skill")),
            ("Yön isabeti", components.get("direction_hit_rate")),
            ("Veri kalitesi", components.get("data_quality")),
        )
    )

    return _card(
        "Gelecek Projeksiyonu",
        f"""
        <div class="forecast-head">
          <div>
            <div class="big-number">{_price(ensemble.get("price"), currency)}</div>
            <div class="{_cls(ensemble.get("change_pct"))} big-change">{_pct(ensemble.get("change_pct"))} · {_e(forecast.get("horizon"))} bar</div>
            <div class="muted">%80 bant: {_price(ensemble.get("low80"), currency)} – {_price(ensemble.get("high80"), currency)}</div>
            <div class="muted">%95 bant: {_price(ensemble.get("low95"), currency)} – {_price(ensemble.get("high95"), currency)}</div>
          </div>
          <div class="forecast-side">
            <div class="pill">Yukarı olasılığı <b>{_pct(ensemble.get("up_probability"), 0, signed=False)}</b></div>
            <div class="pill">Tahmin güveni <b>{_e(confidence.get("label"))} (%{_tr_num(confidence.get("score"), 0)})</b></div>
            <div class="pill">MC medyan <b>{_price(mc.get("terminal", {}).get("q50"), currency)}</b></div>
            <div class="pill">Beklenen max DD <b>{_pct(mc.get("expected_max_drawdown_pct"))}</b></div>
          </div>
        </div>
        {walk_html}
        <p class="fine">Bileşik skor mevcut teknik yapıyı, bu panel ise olasılıklı gelecek dağılımını ölçer; yönler farklılaşabilir.</p>
        <div class="two-col">
          <div>
            <h3>Senaryolar</h3>
            {scenario_html}
          </div>
          <div>
            <h3>Güven bileşenleri</h3>
            <div class="conf-list">{confidence_html}</div>
            <p class="fine">{_e(forecast.get("calibration", {}).get("note", ""))}</p>
          </div>
        </div>
        <h3>Ufuk tablosu</h3>
        {_table(["Ufuk", "Merkez", "Değişim", "%80 bant", "Yön"], horizon_rows)}
        <h3>Modeller ve walk-forward hataları</h3>
        {_table(["Model", "Tahmin", "Değişim", "Ağırlık", "MAPE", "İsabet", "Durum"], method_rows)}
        """,
        subtitle="Çoklu model ensemble · blok bootstrap Monte Carlo · walk-forward kalibrasyon",
        extra="wide",
    )


def _narrative(report: dict) -> str:
    narrative = _get(report, "score", "narrative", default={}) or {}
    sections = narrative.get("sections", [])
    if not sections:
        return ""
    cards = []
    for section in sections:
        cards.append(
            f'<section class="card narrative-card">'
            f'<h2>{_e(section.get("title"))}</h2>'
            f'{_bullets(section.get("bullets", []))}</section>'
        )
    return f"""
    <div class="section-title">
      <h2>Yönetici Özeti</h2>
      <p>{_e(narrative.get("executive_summary", ""))}</p>
    </div>
    <div class="narrative-grid">{''.join(cards)}</div>
    """


def _charts(report: dict, charts: dict) -> str:
    price = charts.get("price", "")
    harmonic = charts.get("harmonic", "")
    regime = charts.get("regime", "")
    return f"""
    <div class="section-title"><h2>Grafikler</h2><p>Fiyat, projeksiyon konisi, spektrum ve dağılım.</p></div>
    <section class="card chart-card wide">{price}</section>
    <section class="card chart-card wide">{harmonic}</section>
    <div class="chart-grid">
      <section class="card chart-card">{charts.get("spectrum", "")}</section>
      <section class="card chart-card">{charts.get("distribution", "")}</section>
    </div>
    <section class="card chart-card wide">{regime}</section>
    <div class="chart-grid">
      <section class="card chart-card">{charts.get("drawdown", "")}</section>
      <section class="card chart-card">{charts.get("seasonality", "")}</section>
      <section class="card chart-card">{charts.get("radar", "")}</section>
      <section class="card chart-card">{charts.get("acf", "")}</section>
    </div>
    """


def _technical(report: dict) -> str:
    technical = report.get("technical", {})
    score = report.get("score", {})
    meta = report.get("meta", {})
    currency = meta.get("currency", "USD")
    indicators = technical.get("indicators", {})
    components = technical.get("components", [])

    snapshot = "".join(
        [
            _metric("Fiyat", _price(technical.get("price"), currency), f"RSI {_tr_num(indicators.get('rsi'), 1)}"),
            _metric("Trend", _e(technical.get("trend_label", "—")), f"trend skoru {_tr_num(technical.get('trend_score'), 2)}"),
            _metric("Momentum", _e(technical.get("momentum_label", "—")), f"momentum skoru {_tr_num(technical.get('momentum_score'), 2)}"),
            _metric("Volatilite", _e(technical.get("volatility_label", "—")), f"ATR {_pct(indicators.get('atr_pct'), 1, signed=False)}"),
            _metric("ADX", _tr_num(indicators.get("adx"), 1), f"+DI {_tr_num(indicators.get('plus_di'), 1)} / -DI {_tr_num(indicators.get('minus_di'), 1)}"),
            _metric("Hacim", f"{_tr_num(_get(technical, 'volume', 'relative'), 2)}x", "20-bar ortalamaya göre"),
        ]
    )

    rows = []
    for item in sorted(components, key=lambda row: abs(row.get("contribution") or 0), reverse=True):
        rows.append(
            [
                _e(item.get("label")),
                _badge(item.get("signal", "—"), _cls(item.get("score"))),
                _tr_num(item.get("score"), 2),
                _tr_num((item.get("weight") or 0) * 100, 1) + "%",
                f'<span class="{_cls(item.get("contribution"))}">{_tr_num(item.get("contribution"), 1)}</span>',
                _e(item.get("value")),
                _e(item.get("note")),
            ]
        )

    contrib_rows = []
    for item in score.get("contributions", []):
        contrib_rows.append(
            [
                _e(item.get("label")),
                _tr_num(item.get("score"), 2),
                _tr_num((item.get("weight") or 0) * 100, 1) + "%",
                f'<span class="{_cls(item.get("contribution"))}">{_tr_num(item.get("contribution"), 1)}</span>',
                _score_bar(item.get("score")),
                _e(item.get("explanation")),
            ]
        )

    return _card(
        "Teknik Tablo ve Skor Ayrışımı",
        f"""
        <div class="metric-grid compact">{snapshot}</div>
        <h3>Bileşik skora katkılar ({_e(score.get("regime_type"))} rejim ağırlıkları)</h3>
        {_table(["Bileşen", "Skor", "Ağırlık", "Katkı", "", "Açıklama"], contrib_rows)}
        <h3>Teknik göstergeler</h3>
        {_table(["Gösterge", "Sinyal", "Skor", "Ağırlık", "Katkı", "Değer", "Not"], rows)}
        """,
        subtitle="Her skor bileşeni ayrı ayrı denetlenebilir; toplam ağırlıklar normalize edilir.",
        extra="wide",
    )


def _spectral(report: dict) -> str:
    spectral = report.get("spectral", {})
    if not spectral.get("available"):
        return _card("Döngü ve Spektrum", _empty(spectral.get("reason", "Spektral analiz yok.")))
    cycles = spectral.get("cycles", [])
    rows = []
    for item in cycles:
        rows.append(
            [
                f"{_tr_num(item.get('period_bars'), 1)} bar",
                f"{_tr_num(item.get('period_days'), 1)} gün",
                _pct(item.get("power_share_pct"), 2, signed=False),
                _pct(item.get("variance_explained_pct"), 2, signed=False),
                _pct(item.get("amplitude_pct"), 2, signed=False),
                f"%{_tr_num(item.get('cycle_position_pct'), 0)}",
                _e(item.get("direction")),
                f"{_tr_num(item.get('next_peak_bars'), 0)} / {_tr_num(item.get('next_trough_bars'), 0)}",
                _badge(
                    item.get("reliability", "—"),
                    "pos" if item.get("reliability") == "yüksek"
                    else "warn" if item.get("reliability") == "sınırda"
                    else "neutral" if item.get("reliability") == "orta"
                    else "muted",
                ),
            ]
        )
    harmonic = spectral.get("harmonic", {})
    entropy = spectral.get("entropy", {})
    wavelet = spectral.get("wavelet", {})
    hilbert = spectral.get("hilbert") or {}
    ssa = spectral.get("ssa", {})
    wavelet_rows = [
        [_e(level.get("scale_bars")), _pct(level.get("energy_share_pct"), 2, signed=False)]
        for level in wavelet.get("levels", [])
    ]
    ssa_rows = [
        [
            _e(item.get("index")),
            _pct(item.get("variance_share_pct"), 2, signed=False),
            _tr_num(item.get("period_bars"), 1),
        ]
        for item in ssa.get("components", [])[:8]
    ]
    notes = _bullets(spectral.get("interpretation", []))
    return _card(
        "Döngü ve Spektrum",
        f"""
        <div class="metric-grid compact">
          {_metric("Periyodiklik", _tr_num(entropy.get("periodicity_score"), 3), "0 = gürültü, 1 = tonal")}
          {_metric("Spektral entropi", _tr_num(entropy.get("spectral_entropy"), 3), "normalize")}
          {_metric("Harmonik R²", _tr_num(harmonic.get("r2"), 3), f"artık {_pct(harmonic.get('residual_std_pct'), 2, signed=False)}")}
          {_metric("SSA 1. bileşen", _pct(ssa.get("variance_first_component_pct"), 1, signed=False), "varyans payı")}
          {_metric("Hilbert fazı", f"%{_tr_num(hilbert.get('phase_position_pct'), 0)}", f"{_e(hilbert.get('direction', '—'))} yönlü")}
          {_metric("Dalga baskın ölçek", f"{_tr_num(wavelet.get('dominant_scale_bars'), 0)} bar", _pct(wavelet.get("dominant_energy_pct"), 1, signed=False))}
        </div>
        {notes}
        <h3>Dominant döngüler</h3>
        {_table(["Periyot", "Gün", "Güç payı", "Varyans", "Genlik", "Faz", "Yön", "Tepe/Dip (bar)", "Güven"], rows)}
        <div class="two-col">
          <div><h3>Haar dalgacık enerjisi</h3>{_table(["Ölçek (bar)", "Enerji"], wavelet_rows)}</div>
          <div><h3>SSA bileşenleri</h3>{_table(["#", "Varyans", "Periyot"], ssa_rows)}</div>
        </div>
        """,
        subtitle="Welch PSD + harmonik regresyon + Hilbert + Haar + SSA",
        extra="wide",
    )


def _statistics(report: dict) -> str:
    stats = report.get("statistics", {})
    if not stats.get("available"):
        return _card("İstatistiksel Karakter", _empty(stats.get("reason", "İstatistik yok.")))
    risk = stats.get("risk", {})
    long_memory = stats.get("long_memory", {})
    mean_reversion = stats.get("mean_reversion", {})
    randomness = stats.get("randomness", {})
    liquidity = stats.get("liquidity", {})
    risk_rows = [
        ["Toplam getiri", _pct(risk.get("total_return_pct"))],
        ["CAGR", _pct(risk.get("cagr_pct"))],
        ["Yıllık volatilite", _pct(risk.get("annualized_volatility_pct"), 1, signed=False)],
        ["Sharpe", _tr_num(risk.get("sharpe"), 2)],
        ["Sortino", _tr_num(risk.get("sortino"), 2)],
        ["Calmar", _tr_num(risk.get("calmar"), 2)],
        ["Maksimum düşüş", _pct(risk.get("max_drawdown_pct"))],
        ["VaR 95 / 99", f"{_pct(risk.get('var_95_pct'))} / {_pct(risk.get('var_99_pct'))}"],
        ["CVaR 95", _pct(risk.get("cvar_95_pct"))],
        ["Kuyruk oranı", _tr_num(risk.get("tail_ratio"), 2)],
        ["İsabet oranı", _pct(risk.get("win_rate_pct"), 1, signed=False)],
        ["Çarpıklık / basıklık", f"{_tr_num(risk.get('skewness'), 2)} / {_tr_num(risk.get('excess_kurtosis'), 2)}"],
        ["Amihud (1e9)", _tr_num(liquidity.get("amihud_illiquidity_1e9"), 4)],
        ["Roll spread", _pct(liquidity.get("roll_spread_pct"), 3, signed=False)],
    ]
    hurst_rows = [
        ["Hurst (R/S)", _tr_num(long_memory.get("hurst_rs_raw", long_memory.get("hurst_rs")), 3)],
        ["Hurst (DFA)", _tr_num(long_memory.get("hurst_dfa_raw", long_memory.get("hurst_dfa")), 3)],
        ["Ortalama Hurst (0-1)", _tr_num(long_memory.get("hurst"), 3)],
        ["DFA notu", ">1: güçlü trend / durağan değil" if long_memory.get("nonstationary_dfa") else "normal aralık"],
        ["Rejim", _e(long_memory.get("regime"))],
        ["Yarı ömür (bar)", _tr_num(mean_reversion.get("half_life_bars"), 1)],
        ["Permütasyon entropisi", _tr_num(randomness.get("permutation_entropy"), 3)],
        ["ADF t", _tr_num(_get(mean_reversion, "adf", "t_stat"), 2)],
        ["ADF p (yaklaşık)", _tr_num(_get(mean_reversion, "adf", "p_approx"), 3)],
    ]
    vr_rows = [
        [
            _e(item.get("q")),
            _tr_num(item.get("vr"), 3),
            _tr_num(item.get("z"), 2),
            _tr_num(item.get("p"), 3),
            _badge(item.get("interpretation"), "muted"),
        ]
        for item in mean_reversion.get("variance_ratio", [])
    ]
    return _card(
        "İstatistiksel Karakter ve Risk",
        f"""
        {_bullets(stats.get("interpretations", []))}
        <div class="two-col">
          <div><h3>Risk metrikleri</h3>{_table(["Metrik", "Değer"], risk_rows)}</div>
          <div><h3>Uzun hafıza / rastgelelik</h3>{_table(["Metrik", "Değer"], hurst_rows)}
          <h3>Varyans oranı</h3>{_table(["q", "VR", "z", "p", "Yorum"], vr_rows)}</div>
        </div>
        """,
        subtitle="Hurst, ADF, varyans oranı, entropi, VaR/CVaR ve likidite",
        extra="wide",
    )


def _patterns(report: dict) -> str:
    patterns = report.get("patterns", {})
    meta = report.get("meta", {})
    currency = meta.get("currency", "USD")
    levels = patterns.get("levels", [])
    level_rows = [
        [
            _price(item.get("price"), currency),
            _badge(item.get("type"), "pos" if item.get("type") == "destek" else "neg" if item.get("type") == "direnç" else "muted"),
            _tr_num(item.get("touches"), 0),
            _tr_num(item.get("strength"), 0),
            _pct(item.get("distance_pct")),
            _e((item.get("last_date") or "")[:10]),
        ]
        for item in levels
    ]
    candle_rows = [
        [
            _e((item.get("date") or "")[:10]),
            _e(item.get("name")),
            _badge("Boğa" if item.get("bias", 0) > 0 else "Ayı" if item.get("bias", 0) < 0 else "Nötr", _cls(item.get("bias"))),
            _tr_num(item.get("strength"), 0),
            _e(item.get("note")),
        ]
        for item in patterns.get("candles", [])
    ]
    fib = patterns.get("fibonacci", {})
    fib_rows = [
        [
            _e(item.get("label")),
            _price(item.get("price"), currency),
            _pct(item.get("distance_pct")),
            _e(item.get("type")),
        ]
        for item in fib.get("levels", [])
    ]
    profile = patterns.get("volume_profile", {})
    profile_rows = [
        [_price(item.get("price"), currency), _pct(item.get("volume_share_pct"), 2, signed=False)]
        for item in profile.get("hvn", [])
    ]
    key = patterns.get("key_levels", {})
    key_rows = [
        ["En yakın destek", _price(_get(key, "nearest_support", "price"), currency)],
        ["En yakın direnç", _price(_get(key, "nearest_resistance", "price"), currency)],
        ["POC", _price(key.get("poc"), currency)],
        ["Değer alanı", f"{_price(key.get('value_area_low'), currency)} – {_price(key.get('value_area_high'), currency)}"],
        ["Swing yüksek / düşük", f"{_price(key.get('swing_high'), currency)} / {_price(key.get('swing_low'), currency)}"],
        ["Aşağı invalidation", _price(key.get("invalidation_down"), currency)],
        ["Yukarı invalidation", _price(key.get("invalidation_up"), currency)],
        ["Kırılım", _e(key.get("breakout"))],
    ]
    return _card(
        "Seviyeler, Formasyonlar ve Hacim Profili",
        f"""
        {_bullets(patterns.get("interpretation", []))}
        <div class="two-col">
          <div><h3>Kritik seviyeler</h3>{_table(["Seviye", "Değer"], key_rows)}
          <h3>Fibonacci</h3>{_table(["Oran", "Fiyat", "Mesafe", "Tür"], fib_rows)}</div>
          <div><h3>Destek / direnç kümeleri</h3>{_table(["Fiyat", "Tür", "Dokunma", "Güç", "Mesafe", "Son"], level_rows)}
          <h3>Hacim profili HVN</h3>{_table(["Fiyat", "Hacim payı"], profile_rows)}</div>
        </div>
        <h3>Mum formasyonları</h3>
        {_table(["Tarih", "Formasyon", "Yön", "Güç", "Not"], candle_rows)}
        """,
        subtitle="Pivot kümeleri, Fibonacci, hacim profili, mumlar, uyumsuzluklar, kanal",
        extra="wide",
    )


def _regime(report: dict) -> str:
    regime = report.get("regime", {})
    season = regime.get("seasonality", {})
    month_rows = [
        [
            _e(item.get("name")),
            _pct(item.get("mean_pct")),
            _pct(item.get("median_pct")),
            _pct(item.get("win_rate_pct"), 1, signed=False),
            _pct(item.get("std_pct"), 1, signed=False),
            _e(item.get("observations")),
        ]
        for item in season.get("monthly", [])
    ]
    weekday_rows = [
        [
            _e(item.get("name")),
            _pct(item.get("mean_pct"), 3),
            _pct(item.get("win_rate_pct"), 1, signed=False),
            _tr_num(item.get("t_stat"), 2),
            _e(item.get("observations")),
        ]
        for item in season.get("weekday", [])
    ]
    change_rows = [
        [
            _e((item.get("date") or "")[:10]),
            _tr_num(item.get("t_stat"), 2),
            _pct(item.get("before_mean_pct"), 3),
            _pct(item.get("after_mean_pct"), 3),
            _e(item.get("direction")),
        ]
        for item in regime.get("change_points", [])
    ]
    volatility = regime.get("volatility", {})
    phase_raw = _get(regime, "market_phase", "phase")
    phase_map = {
        "markup (yükseliş)": "Yükseliş",
        "markdown (düşüş)": "Düşüş",
        "accumulation (birikim)": "Birikim",
        "distribution (dağılım)": "Dağılım",
    }
    phase_label = phase_map.get(str(phase_raw).lower(), phase_raw)
    return _card(
        "Rejim, Kırılmalar ve Sezonluk",
        f"""
        {_bullets(regime.get("interpretation", []))}
        <div class="metric-grid compact">
          {_metric("Trend rejimi", _e(regime.get("trend_regime")), f"bias {_tr_num(regime.get('trend_regime_bias'), 2)}")}
          {_metric("Volatilite rejimi", _e(volatility.get("label")), f"yüzdelik {_tr_num(volatility.get('percentile'), 0)}")}
          {_metric("Piyasa fazı", _e(phase_label), "heuristik")}
          {_metric("Hacim trendi", f"{_tr_num(_get(regime, 'volume', 'trend_20_vs_60'), 2)}x", "20g / 60g")}
        </div>
        <div class="two-col">
          <div><h3>Aylık sezonluk</h3>{_table(["Ay", "Ort.", "Medyan", "Kazanma", "Std", "n"], month_rows)}
          <p class="fine">n &lt; 5 olan aylar zayıf kanıttır; 500 günlük veri yalnızca 1-2 yıl kapsar.</p></div>
          <div><h3>Haftanın günleri</h3>{_table(["Gün", "Ort.", "Kazanma", "t", "n"], weekday_rows)}
          <h3>Yapısal kırılmalar</h3>{_table(["Tarih", "t", "Önce", "Sonra", "Yön"], change_rows)}</div>
        </div>
        """,
        subtitle="EMA/ADX rejimi, Welch t kırılmaları, aylık ve günlük sezonluk",
        extra="wide",
    )


def _methodology(report: dict) -> str:
    meta = report.get("meta", {})
    forecast = report.get("forecast", {})
    spectral = report.get("spectral", {})
    statistics = report.get("statistics", {})
    patterns = report.get("patterns", {})
    notes = []
    notes.extend(forecast.get("method_notes", []))
    notes.extend(spectral.get("method_notes", []))
    notes.extend(statistics.get("method_notes", []))
    notes.extend(patterns.get("method_notes", []))
    notes.extend((report.get("regime") or {}).get("method_notes", []))
    caveats = forecast.get("caveats", [])
    narrative = _get(report, "score", "narrative", default={}) or {}
    data_rows = [
        ["Sembol", _e(meta.get("symbol"))],
        ["Ad", _e(meta.get("name"))],
        ["Piyasa", _e(meta.get("market"))],
        ["Borsa / kaynak", f"{_e(meta.get('exchange'))} / {_e(meta.get('source'))}"],
        ["Aralık", _e(meta.get("interval"))],
        ["Bar sayısı", _e(meta.get("bars"))],
        ["İlk bar", _e((meta.get("first_timestamp") or "")[:19].replace("T", " "))],
        ["Son bar", _e((meta.get("last_timestamp") or "")[:19].replace("T", " "))],
        ["Çekilme", _e((meta.get("fetched_at") or "")[:19].replace("T", " "))],
        ["Önbellek eski mi", "evet" if meta.get("stale") else "hayır"],
        ["Veri boşlukları", _e(meta.get("gaps"))],
        ["Veri kalitesi", f"{_e((report.get('quality') or {}).get('label'))} (%{_tr_num((report.get('quality') or {}).get('score'), 0)})"],
        ["Son bar yaşı", f"{_tr_num((report.get('quality') or {}).get('bar_age_hours'), 1)} saat"],
    ]
    return _card(
        "Veri, Yöntem ve Uyarılar",
        f"""
        <div class="two-col">
          <div><h3>Veri provenance</h3>{_table(["Alan", "Değer"], data_rows)}</div>
          <div><h3>Yöntem notları</h3>{_bullets(notes)}</div>
        </div>
        <h3>Uyarılar</h3>
        {_bullets(caveats)}
        <p class="disclaimer">{_e(narrative.get("disclaimer", "Bu rapor yatırım tavsiyesi değildir."))}</p>
        """,
        subtitle="Şeffaflık için veri kaynağı, pencere ve model sınırlamaları",
        extra="wide",
    )


# ---------------------------------------------------------------------------
# CSS / page shell
# ---------------------------------------------------------------------------

CSS = """
:root{
  --bg:#080b10; --panel:#11161d; --panel2:#151c25; --line:#222c38;
  --text:#dce6f2; --muted:#8b9bb0; --green:#3fb950; --red:#f85149;
  --blue:#58a6ff; --orange:#f08833; --purple:#a78bfa; --yellow:#d29922; --cyan:#39c5cf;
  --accent:#58a6ff; --radius:16px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:radial-gradient(1200px 700px at 15% -10%,#172234 0%,var(--bg) 55%) fixed;
  color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased}
a{color:var(--blue);text-decoration:none}
.wrap{max-width:1240px;margin:0 auto;padding:28px 22px 70px}
.hero{background:linear-gradient(150deg,rgba(30,41,59,.86),rgba(17,22,29,.96));
  border:1px solid var(--line);border-radius:24px;padding:26px 28px 22px;
  box-shadow:0 24px 70px rgba(0,0,0,.35);margin-bottom:24px;position:relative;overflow:hidden}
.hero:before{content:"";position:absolute;inset:-40% -20% auto auto;width:520px;height:520px;
  background:radial-gradient(circle,rgba(88,166,255,.18),transparent 65%);pointer-events:none}
.hero-top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;position:relative;z-index:1}
.eyebrow{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
h1{font-size:34px;line-height:1.15;margin:0 0 8px;letter-spacing:-.02em}
h1 span{font-size:15px;font-weight:500;color:var(--muted);letter-spacing:0;margin-left:8px}
h2{font-size:18px;margin:0 0 4px;letter-spacing:-.01em}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  margin:22px 0 8px;font-weight:600}
.price-line{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
.price-line strong{font-size:27px;font-variant-numeric:tabular-nums}
.muted{color:var(--muted)}
.hero-actions{display:flex;gap:8px;flex-wrap:wrap}
.btn{border:0;border-radius:9px;padding:9px 14px;background:var(--blue);color:#04101f;
  font-weight:700;font-size:12px;cursor:pointer;font-family:inherit}
.btn.ghost{background:transparent;color:var(--text);border:1px solid var(--line)}
.btn:hover{filter:brightness(1.08)}
.hero-grid{display:grid;grid-template-columns:minmax(280px,380px) 1fr;gap:22px;align-items:center;
  margin-top:18px;position:relative;z-index:1}
.hero-gauge svg{width:100%;height:auto;display:block}
.hero-summary .verdict{font-size:21px;font-weight:800;margin-bottom:6px}
.verdict.bull{color:var(--green)} .verdict.bear{color:var(--red)} .verdict.neutral{color:var(--yellow)}
.hero-summary p{margin:0 0 10px;color:#b9c6d6;max-width:720px}
.band-line{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;
  padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:rgba(8,11,16,.35)}
.hero-meta{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:11.5px;margin-top:10px}
.metric-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-top:20px;position:relative;z-index:1}
.metric-grid.compact{grid-template-columns:repeat(6,minmax(0,1fr));margin:4px 0 8px}
.metric{background:rgba(8,11,16,.40);border:1px solid var(--line);border-radius:12px;padding:12px 13px;min-width:0}
.metric-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:4px}
.metric-value{font-size:19px;font-weight:750;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metric-value small{font-size:11px;color:var(--muted);font-weight:500}
.metric-note{font-size:10.5px;color:var(--muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metric.pos .metric-value{color:var(--green)} .metric.neg .metric-value{color:var(--red)}
.card{background:linear-gradient(180deg,rgba(21,28,37,.94),rgba(17,22,29,.96));
  border:1px solid var(--line);border-radius:var(--radius);padding:20px 20px 18px;margin:0 0 18px;
  box-shadow:0 14px 40px rgba(0,0,0,.18);min-width:0}
.card.wide{grid-column:1/-1}
.card-head{margin-bottom:12px}
.card-sub{color:var(--muted);font-size:12px;margin-top:2px}
.chart-card{padding:10px;overflow:hidden}
.chart-card svg{display:block;width:100%;height:auto;border-radius:10px}
.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-bottom:18px}
.chart-grid .card{margin:0}
.section-title{margin:32px 0 14px}
.section-title h2{font-size:24px} .section-title p{color:var(--muted);margin:0}
.narrative-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:20px}
.narrative-card{margin:0;padding:17px 18px}
.narrative-card h2{font-size:14px;text-transform:uppercase;letter-spacing:.07em;color:var(--blue);margin-bottom:9px}
.bullets{margin:0;padding-left:17px;color:#b9c6d6}
.bullets li{margin:0 0 7px}
.bullets strong{color:var(--text)}
.badge{display:inline-flex;align-items:center;border-radius:999px;padding:3px 8px;font-size:10px;
  font-weight:700;letter-spacing:.03em;border:1px solid transparent;white-space:nowrap}
.badge.pos{background:rgba(63,185,80,.14);color:var(--green);border-color:rgba(63,185,80,.25)}
.badge.neg{background:rgba(248,81,73,.14);color:var(--red);border-color:rgba(248,81,73,.25)}
.badge.neutral,.badge.muted{background:rgba(139,155,176,.12);color:var(--muted);border-color:var(--line)}
.badge.warn{background:rgba(210,153,34,.14);color:var(--yellow);border-color:rgba(210,153,34,.28)}
.badge.market{background:rgba(88,166,255,.14);color:var(--blue);border-color:rgba(88,166,255,.25)}
.pos{color:var(--green)} .neg{color:var(--red)} .neutral{color:var(--muted)}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:11px;margin:6px 0 4px}
table{width:100%;border-collapse:collapse;font-size:12px;min-width:460px}
thead th{text-align:left;padding:9px 10px;background:rgba(8,11,16,.55);color:var(--muted);
  font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-weight:600;white-space:nowrap}
tbody td{padding:9px 10px;border-top:1px solid var(--line);vertical-align:top}
tbody tr:hover{background:rgba(88,166,255,.04)}
td{font-variant-numeric:tabular-nums}
.empty{color:var(--muted);padding:18px;text-align:center;border:1px dashed var(--line);border-radius:10px}
.forecast-head{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;align-items:flex-start}
.big-number{font-size:42px;font-weight:850;letter-spacing:-.03em;line-height:1.05}
.big-change{font-size:15px;font-weight:700;margin:4px 0 8px}
.forecast-side{display:flex;flex-direction:column;gap:7px;min-width:230px}
.pill{display:flex;justify-content:space-between;gap:16px;background:rgba(8,11,16,.45);
  border:1px solid var(--line);border-radius:9px;padding:8px 11px;font-size:12px;color:var(--muted)}
.pill b{color:var(--text)}
.callout{display:flex;gap:10px;flex-wrap:wrap;background:rgba(63,185,80,.07);border:1px solid rgba(63,185,80,.24);
  border-radius:10px;padding:10px 12px;margin:14px 0;font-size:12px;color:#b9c6d6}
.callout strong{color:var(--green)}
.callout.warn{background:rgba(210,153,34,.07);border-color:rgba(210,153,34,.28)}
.callout.warn strong{color:var(--yellow)}
.two-col{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}
.conf-list{display:flex;flex-direction:column;gap:9px;margin-top:8px}
.conf-row{display:grid;grid-template-columns:150px 1fr 40px;gap:9px;align-items:center;font-size:11.5px;color:var(--muted)}
.conf-row b{color:var(--text);text-align:right;font-variant-numeric:tabular-nums}
.track{height:7px;border-radius:99px;background:#0a0f15;border:1px solid var(--line);overflow:hidden}
.track span{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:99px}
.scorebar{position:relative;height:7px;width:100px;background:#0a0f15;border:1px solid var(--line);border-radius:99px;overflow:hidden;min-width:70px}
.scorebar .bar{position:absolute;top:0;bottom:0;border-radius:99px}
.scorebar .bar.pos{background:var(--green)} .scorebar .bar.neg{background:var(--red)}
.scorebar .mid{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted);opacity:.5}
.fine{font-size:11px;color:var(--muted);margin:8px 0 0}
.disclaimer{border-left:3px solid var(--yellow);background:rgba(210,153,34,.06);padding:11px 14px;
  color:#c4b48d;font-size:12px;border-radius:0 9px 9px 0;margin-top:14px}
.foot{color:var(--muted);font-size:11px;text-align:center;margin-top:34px;line-height:1.7}
@media (max-width:1050px){
  .metric-grid,.metric-grid.compact{grid-template-columns:repeat(3,minmax(0,1fr))}
  .narrative-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width:820px){
  .hero-grid,.two-col,.chart-grid{grid-template-columns:1fr}
  .hero-top{flex-direction:column}
  h1{font-size:27px}
  .metric-grid,.metric-grid.compact{grid-template-columns:repeat(2,minmax(0,1fr))}
  .narrative-grid{grid-template-columns:1fr}
  .wrap{padding:14px 12px 50px}.hero{padding:20px 17px}.card{padding:16px}
}
@media print{
  body{background:#fff;color:#111}
  .hero,.card{box-shadow:none;background:#fff;color:#111;border-color:#ddd}
  .muted,.card-sub,.metric-note,.hero-meta{color:#555}
  .hero-summary p,.bullets{color:#222}
  .btn,.hero-actions{display:none}
  .card,.hero{break-inside:avoid}
  .chart-card svg{background:#0d1117}
}
"""

JS = """
function downloadJson(){
  const node=document.getElementById('analysis-data');
  const blob=new Blob([node.textContent],{type:'application/json;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=(document.body.dataset.symbol||'analysis')+'-analiz.json';
  document.body.appendChild(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
"""


def render_report(report: dict, charts: Dict[str, str]) -> str:
    meta = report.get("meta", {})
    symbol = meta.get("symbol", "analiz")
    tone = _get(report, "score", "tone", default="neutral")
    json_payload = {key: value for key, value in report.items() if key != "charts"}
    serialized = json.dumps(json_payload, ensure_ascii=False, default=str)
    serialized = serialized.replace("</", "<\\/")

    body = (
        _hero(report, charts)
        + _narrative(report)
        + _forecast_panel(report)
        + _charts(report, charts)
        + _technical(report)
        + _spectral(report)
        + _statistics(report)
        + _patterns(report)
        + _regime(report)
        + _methodology(report)
    )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(symbol)} · Piyasa Analiz Raporu</title>
<style>{CSS}</style>
</head>
<body data-symbol="{_e(symbol)}" class="{tone}">
<div class="wrap">
{body}
<div class="foot">
  Bu rapor <b>WhaleTracker Analytics</b> tarafından otomatik üretildi.<br>
  Yatırım tavsiyesi değildir. Geçmiş performans geleceği garanti etmez.
</div>
</div>
<script id="analysis-data" type="application/json">{serialized}</script>
<script>{JS}</script>
</body>
</html>
"""

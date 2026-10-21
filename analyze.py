#!/usr/bin/env python
"""WhaleTracker advanced analysis CLI.

Examples
--------
    python analyze.py BTCUSDT
    python analyze.py BTCUSDT --interval 4h --horizon 42
    python analyze.py AAPL --market us
    python analyze.py THYAO --market bist --limit 400
    python analyze.py BTCUSDT ETHUSDT SOLUSDT --no-save

The command writes a standalone HTML report and a JSON file to ``reports/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Windows consoles often default to cp1254/cp1252; never crash on an arrow.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.analytics.engine import analyze, analysis_summary  # noqa: E402

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH = True
except Exception:  # pragma: no cover - rich is optional
    RICH = False


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(number) >= 1000:
        text = f"{number:,.0f}"
    elif abs(number) >= 1:
        text = f"{number:,.{digits}f}"
    else:
        text = f"{number:,.4f}"
    return text + suffix


def _pct(value, digits: int = 2) -> str:
    try:
        return f"{float(value):+.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _score_text(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "0" if abs(number) < 0.5 else f"{number:+.0f}"


def _print_plain(report: dict, paths: dict) -> None:
    meta = report["meta"]
    score = report["score"]
    forecast = report["forecast"]
    ensemble = forecast.get("ensemble", {})
    print("=" * 78)
    print(f"{meta.get('symbol')} · {meta.get('name')} · {meta.get('market')} · {meta.get('interval')}")
    print(f"Fiyat: {_fmt(meta.get('last_price'))} {meta.get('currency')} | Skor: {_score_text(score.get('score'))} | {score.get('label')}")
    print(f"Güven: %{score.get('confidence'):.0f} ({score.get('confidence_label')}) | Risk: {score.get('risk', {}).get('level')}")
    print(f"Tahmin: {_fmt(ensemble.get('price'))} ({_pct(ensemble.get('change_pct'))}) "
          f"%80: {_fmt(ensemble.get('low80'))}–{_fmt(ensemble.get('high80'))} | "
          f"yukarı olasılık: %{ensemble.get('up_probability', 0):.0f}")
    print("-" * 78)
    for item in score.get("contributions", [])[:5]:
        print(f"  {item['label']:<24} skor {item['score']:+.2f}  ağırlık {item['weight']*100:>4.1f}%  katkı {item['contribution']:+.1f}")
    print("-" * 78)
    for section in score.get("narrative", {}).get("sections", [])[:3]:
        print(f"[{section['title']}]")
        for bullet in section.get("bullets", [])[:3]:
            print("  - " + _strip_markdown(bullet))
    print("-" * 78)
    for key, value in paths.items():
        if value:
            print(f"{key}: {value}")
    print("=" * 78)


def _strip_markdown(value: str) -> str:
    return (
        value.replace("**", "")
        .replace("`", "")
        .replace("→", "->")
        .replace("←", "<-")
        .replace("•", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("≥", ">=")
        .replace("≤", "<=")
    )


def _print_rich(report: dict, paths: dict) -> None:
    console = Console()
    meta = report["meta"]
    score = report["score"]
    forecast = report["forecast"]
    ensemble = forecast.get("ensemble", {})
    tone = score.get("tone", "neutral")
    color = {"bull": "green", "bear": "red", "neutral": "yellow"}.get(tone, "white")

    header = Text()
    header.append(f"{meta.get('symbol')}", style=f"bold {color}")
    header.append(f"  {meta.get('name')}  ·  {meta.get('market')} / {meta.get('interval')}\n", style="dim")
    header.append(f"{_fmt(meta.get('last_price'))} {meta.get('currency')}", style="bold")
    header.append(f"   skor {_score_text(score.get('score'))}  ·  {score.get('label')}  ·  güven %{score.get('confidence'):.0f}\n", style=color)
    header.append(
        f"Risk: {score.get('risk', {}).get('level')}   "
        f"Trend: {report.get('technical', {}).get('trend_label')}   "
        f"Momentum: {report.get('technical', {}).get('momentum_label')}",
        style="dim",
    )
    console.print(Panel(header, border_style=color, padding=(1, 2)))

    forecast_table = Table(title="Tahmin ufukları", box=None, header_style="bold cyan")
    for column in ("Ufuk", "Merkez", "Değişim", "%80 alt", "%80 üst", "Yön"):
        forecast_table.add_column(column, justify="right" if column != "Ufuk" else "left")
    for item in forecast.get("horizons", []):
        forecast_table.add_row(
            str(item.get("label")),
            _fmt(item.get("price")),
            _pct(item.get("change_pct")),
            _fmt(item.get("low80")),
            _fmt(item.get("high80")),
            str(item.get("direction")),
        )
    console.print(forecast_table)

    score_table = Table(title="Skor ayrışımı", box=None, header_style="bold cyan")
    score_table.add_column("Bileşen")
    score_table.add_column("Skor", justify="right")
    score_table.add_column("Ağırlık", justify="right")
    score_table.add_column("Katkı", justify="right")
    for item in score.get("contributions", []):
        score_table.add_row(
            item["label"],
            f"{item['score']:+.2f}",
            f"{item['weight'] * 100:.1f}%",
            f"{item['contribution']:+.1f}",
        )
    console.print(score_table)

    methods = Table(title="Modeller / walk-forward", box=None, header_style="bold cyan")
    for column in ("Model", "Ağırlık", "Tahmin", "Değişim", "MAPE", "İsabet"):
        methods.add_column(column, justify="right" if column != "Model" else "left")
    for item in sorted(forecast.get("methods", []), key=lambda row: row.get("weight") or 0, reverse=True):
        methods.add_row(
            str(item.get("label")),
            f"{(item.get('weight') or 0) * 100:.1f}%",
            _fmt(item.get("forecast_price")),
            _pct(item.get("change_pct")),
            _pct(item.get("mae_pct")),
            _pct(item.get("hit_rate"), 1),
        )
    console.print(methods)

    narrative = score.get("narrative", {})
    for section in narrative.get("sections", [])[:4]:
        body = "\n".join("• " + _strip_markdown(bullet) for bullet in section.get("bullets", [])[:4])
        console.print(Panel(body, title=section.get("title"), border_style="dim", title_align="left"))

    if paths:
        console.print("[bold]Çıktılar[/bold]")
        for key, value in paths.items():
            if value:
                console.print(f"  [cyan]{key}[/cyan]: {value}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Çoklu metodlu piyasa analizi ve gelecek projeksiyonu."
    )
    parser.add_argument("symbols", nargs="+", help="Sembol(ler): BTCUSDT, AAPL, THYAO ...")
    parser.add_argument("--market", default="auto", choices=("auto", "crypto", "us", "bist"))
    parser.add_argument("--interval", default="1d", choices=("1h", "4h", "1d", "1w"))
    parser.add_argument("--limit", type=int, default=500, help="Bar sayısı (80-5000)")
    parser.add_argument("--horizon", type=int, default=30, help="Tahmin ufku (bar)")
    parser.add_argument("--no-cache", action="store_true", help="Disk önbelleğini okuma")
    parser.add_argument("--no-save", action="store_true", help="HTML/JSON yazma")
    parser.add_argument("--output-dir", default=None, help="Rapor dizini")
    parser.add_argument("--open", action="store_true", help="Raporu tarayıcıda aç")
    parser.add_argument("--json", action="store_true", help="Özet JSON yazdır")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else None
    failures = 0
    summaries = []

    for symbol in args.symbols:
        try:
            result = analyze(
                symbol=symbol,
                market=args.market,
                interval=args.interval,
                limit=args.limit,
                horizon=args.horizon,
                use_cache=not args.no_cache,
                save=not args.no_save,
                output_dir=output_dir,
            )
        except Exception as exc:
            failures += 1
            if RICH:
                Console().print(f"[red]{symbol} analiz edilemedi:[/red] {exc}")
            else:
                print(f"{symbol} analiz edilemedi: {exc}", file=sys.stderr)
            continue

        summaries.append(analysis_summary(result))
        paths = {
            "HTML": str(result.html_path) if result.html_path else None,
            "JSON": str(result.json_path) if result.json_path else None,
        }
        if args.json:
            print(json.dumps(analysis_summary(result), ensure_ascii=False, indent=2))
        elif RICH:
            _print_rich(result.report, paths)
        else:
            _print_plain(result.report, paths)

        if args.open and result.html_path:
            webbrowser.open(result.html_path.resolve().as_uri())

    if args.json and len(summaries) > 1:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

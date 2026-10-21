"""Unified OHLCV data layer.

Providers
---------
* Binance public REST (crypto, primary)
* Bybit public REST (crypto, fallback)
* Yahoo Finance chart API (US, BIST and crypto fallback)

No API key is required. Responses are cached on disk to keep repeated
analyses fast and to allow a stale-cache fallback when a provider is down.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as url_quote

import httpx
import numpy as np
import pandas as pd

from .symbols import (
    ResolvedSymbol,
    interval_ms,
    resolve_symbol,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data" / "cache"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class DataError(RuntimeError):
    """Raised when market data cannot be retrieved or is unusable."""


@dataclass
class MarketData:
    """Validated OHLCV frame plus provenance metadata."""

    symbol: str
    name: str
    market: str
    currency: str
    exchange: str
    interval: str
    source: str
    df: pd.DataFrame
    fetched_at: datetime
    resolved: ResolvedSymbol
    stale: bool = False

    @property
    def bars(self) -> int:
        return int(len(self.df))

    @property
    def last_price(self) -> float:
        return float(self.df["close"].iloc[-1])

    @property
    def bars_per_year(self) -> float:
        return bars_per_year(self.market, self.interval)

    @property
    def first_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.df.index[0])

    @property
    def last_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.df.index[-1])

    def meta(self) -> dict:
        df = self.df
        deltas = pd.Series(df.index).diff().dropna()
        median_delta = (
            float(deltas.median().total_seconds()) if len(deltas) else 0.0
        )
        gaps = 0
        if self.market == "crypto" and median_delta > 0:
            gaps = int((deltas.dt.total_seconds() > median_delta * 2.5).sum())

        return {
            "symbol": self.symbol,
            "name": self.name,
            "market": self.market,
            "currency": self.currency,
            "exchange": self.exchange,
            "interval": self.interval,
            "source": self.source,
            "bars": self.bars,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "stale": bool(self.stale),
            "gaps": gaps,
            "last_price": round(self.last_price, 8),
            "bars_per_year": round(self.bars_per_year, 2),
            "data_age_seconds": round(
                max(0.0, (datetime.now(timezone.utc) - self.fetched_at).total_seconds()),
                1,
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bars_per_year(market: str, interval: str) -> float:
    if interval == "1w":
        return 52.0
    seconds = interval_ms(interval) / 1000.0
    if market == "crypto":
        return 365.0 * 24.0 * 3600.0 / seconds
    hours_per_session = 6.5 if market == "us" else 7.0
    if interval == "1d":
        return 252.0
    return 252.0 * hours_per_session * 3600.0 / seconds


def _cache_key(resolved: ResolvedSymbol, interval: str, limit: int) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in resolved.symbol)
    return f"{resolved.market}_{safe}_{interval}_{int(limit)}"


def _cache_paths(key: str) -> Tuple[Path, Path]:
    return CACHE_DIR / f"{key}.csv", CACHE_DIR / f"{key}.json"


def _ttl_seconds(interval: str) -> int:
    if interval == "1d":
        return 6 * 3600
    if interval == "1w":
        return 12 * 3600
    return 15 * 60


def _http_json(
    client: httpx.Client,
    url: str,
    params: Optional[dict] = None,
    retries: int = 3,
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = client.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (418, 429):
                time.sleep(1.2 * (attempt + 1))
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request,
                    response=response,
                )
                continue
            if 500 <= response.status_code < 600:
                time.sleep(0.6 * (attempt + 1))
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request,
                    response=response,
                )
                continue
            raise DataError(
                f"HTTP {response.status_code}: {response.text[:180]}"
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    raise DataError(f"İstek başarısız: {url} ({last_error})")


def _finalize_frame(
    df: pd.DataFrame,
    *,
    market: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataError("Sağlayıcı boş veri döndürdü")

    df = df.copy()
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    if df.empty:
        raise DataError("Geçerli fiyat satırı bulunamadı")

    # Repair partially missing OHLC values
    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df[["open", "close"]].max(axis=1))
    df["low"] = df["low"].fillna(df[["open", "close"]].min(axis=1))
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    if df["quote_volume"].isna().all() or (df["quote_volume"] <= 0).all():
        df["quote_volume"] = df["close"] * df["volume"]
    df["quote_volume"] = df["quote_volume"].fillna(df["close"] * df["volume"])

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.tail(int(limit))
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    if len(df) < 80:
        raise DataError(
            f"Yetersiz veri: {len(df)} bar bulundu, en az 80 bar gerekli. "
            "Daha uzun bir zaman aralığı deneyin."
        )
    return df[["open", "high", "low", "close", "volume", "quote_volume"]]


# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------

def _fetch_binance(
    client: httpx.Client, resolved: ResolvedSymbol, interval: str, limit: int
) -> Tuple[pd.DataFrame, str, str]:
    step = interval_ms(interval)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - step * (limit + 8)
    rows: List[list] = []
    seen = set()
    bases = [
        "https://api.binance.com",
        "https://data-api.binance.vision",
    ]
    base = bases[0]
    errors: List[str] = []

    while len(rows) < limit + 5:
        remaining = limit + 5 - len(rows)
        params = {
            "symbol": resolved.data_symbol,
            "interval": interval,
            "limit": min(1000, remaining),
            "startTime": start_ms,
            "endTime": now_ms,
        }
        try:
            data = _http_json(client, f"{base}/api/v3/klines", params)
        except DataError as exc:
            errors.append(str(exc))
            if base != bases[-1]:
                base = bases[-1]
                continue
            raise DataError("Binance: " + "; ".join(errors[-2:]))

        if not isinstance(data, list):
            raise DataError("Binance: beklenmeyen yanıt")
        if not data:
            break
        added = 0
        for row in data:
            if row and row[0] not in seen:
                seen.add(row[0])
                rows.append(row)
                added += 1
        if added == 0 or len(data) < 1000:
            break
        start_ms = int(data[-1][6]) + 1
        if start_ms >= now_ms:
            break

    if not rows:
        raise DataError(f"Binance: {resolved.data_symbol} için veri yok")

    rows.sort(key=lambda row: row[0])
    records = []
    for row in rows[-limit:]:
        records.append(
            {
                "timestamp": pd.to_datetime(row[0], unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "quote_volume": float(row[7]),
            }
        )
    frame = pd.DataFrame(records).set_index("timestamp")
    return frame, "Binance", "Binance Spot"


# ---------------------------------------------------------------------------
# Bybit
# ---------------------------------------------------------------------------

_BYBIT_INTERVALS = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1d": "D", "1w": "W",
}


def _fetch_bybit(
    client: httpx.Client, resolved: ResolvedSymbol, interval: str, limit: int
) -> Tuple[pd.DataFrame, str, str]:
    bybit_interval = _BYBIT_INTERVALS.get(interval)
    if not bybit_interval:
        raise DataError(f"Bybit {interval} aralığını desteklemiyor")
    if resolved.quote not in ("USDT", "USDC"):
        raise DataError("Bybit yalnızca USDT/USDC paritelerini destekliyor")

    end_ms = int(time.time() * 1000)
    rows: List[list] = []
    seen = set()

    while len(rows) < limit:
        params = {
            "category": "spot",
            "symbol": resolved.data_symbol,
            "interval": bybit_interval,
            "limit": min(1000, limit - len(rows) + 5),
            "end": end_ms,
        }
        payload = _http_json(
            client, "https://api.bybit.com/v5/market/kline", params
        )
        if payload.get("retCode") != 0:
            raise DataError(f"Bybit: {payload.get('retMsg', 'bilinmeyen hata')}")
        batch = (payload.get("result") or {}).get("list") or []
        if not batch:
            break
        added = 0
        for row in batch:
            if row and row[0] not in seen:
                seen.add(row[0])
                rows.append(row)
                added += 1
        if added == 0 or len(batch) < 1000:
            break
        end_ms = min(int(row[0]) for row in batch) - 1

    if not rows:
        raise DataError(f"Bybit: {resolved.data_symbol} için veri yok")

    rows.sort(key=lambda row: int(row[0]))
    records = []
    for row in rows[-limit:]:
        records.append(
            {
                "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "quote_volume": float(row[6]) if len(row) > 6 else np.nan,
            }
        )
    frame = pd.DataFrame(records).set_index("timestamp")
    return frame, "Bybit", "Bybit Spot"


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

_YAHOO_INTERVALS = {
    "1h": ("60m", None),
    "4h": ("60m", "4h"),
    "1d": ("1d", None),
    "1w": ("1wk", None),
}


def _yahoo_symbol(resolved: ResolvedSymbol) -> str:
    if resolved.market != "crypto":
        return resolved.data_symbol
    if resolved.quote in ("USDT", "USDC", "FDUSD", "BUSD", "USD"):
        return f"{resolved.base}-USD"
    return f"{resolved.base}-{resolved.quote}"


def _fetch_yahoo(
    client: httpx.Client, resolved: ResolvedSymbol, interval: str, limit: int
) -> Tuple[pd.DataFrame, str, str]:
    mapping = _YAHOO_INTERVALS.get(interval)
    if not mapping:
        raise DataError(f"Yahoo Finance {interval} aralığını desteklemiyor")
    yahoo_interval, resample_rule = mapping
    if resample_rule and resolved.market != "crypto":
        raise DataError(
            "Hisse senetleri için 4 saatlik veri desteklenmiyor; 1h veya 1d kullanın."
        )

    step_seconds = interval_ms(interval) / 1000.0
    if interval in ("1d", "1w"):
        factor = 2.2 if resolved.market != "crypto" else 1.35
    else:
        factor = 6.5 if resolved.market != "crypto" else 1.5

    symbol = _yahoo_symbol(resolved)
    encoded = url_quote(symbol, safe="")
    now = datetime.now(timezone.utc)
    best: Optional[pd.DataFrame] = None
    best_meta: Dict[str, Any] = {}

    for multiplier in (1.0, 2.0, 4.0):
        period1 = int(now.timestamp() - step_seconds * (limit + 10) * factor * multiplier)
        period2 = int(now.timestamp()) + 60
        params = {
            "period1": period1,
            "period2": period2,
            "interval": yahoo_interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
        errors = []
        payload = None
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            try:
                payload = _http_json(
                    client,
                    f"https://{host}/v8/finance/chart/{encoded}",
                    params,
                )
                break
            except DataError as exc:
                errors.append(str(exc))
        if payload is None:
            raise DataError("Yahoo: " + "; ".join(errors[-2:]))

        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise DataError(f"Yahoo: {chart['error'].get('description', 'hata')}")
        results = chart.get("result") or []
        if not results:
            raise DataError(f"Yahoo: {symbol} için sonuç yok")

        result = results[0]
        timestamps = result.get("timestamp") or []
        quote_data = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adj_data = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
        adjclose = adj_data.get("adjclose") or []
        if not timestamps:
            raise DataError(f"Yahoo: {symbol} için zaman damgası yok")

        records = []
        for i, ts in enumerate(timestamps):
            close = _at(quote_data.get("close"), i)
            if close is None:
                continue
            open_ = _at(quote_data.get("open"), i) or close
            high = _at(quote_data.get("high"), i) or max(open_, close)
            low = _at(quote_data.get("low"), i) or min(open_, close)
            volume = _at(quote_data.get("volume"), i) or 0.0
            adj = _at(adjclose, i) if adjclose else None
            factor_adj = 1.0
            if adj is not None and close > 0:
                candidate = adj / close
                if 0.05 <= candidate <= 20.0:
                    factor_adj = candidate
            records.append(
                {
                    "timestamp": pd.to_datetime(ts, unit="s", utc=True),
                    "open": float(open_) * factor_adj,
                    "high": float(high) * factor_adj,
                    "low": float(low) * factor_adj,
                    "close": float(close) * factor_adj,
                    "volume": float(volume),
                    "quote_volume": float(close) * factor_adj * float(volume),
                }
            )

        if not records:
            raise DataError(f"Yahoo: {symbol} için geçerli satır yok")

        frame = pd.DataFrame(records).set_index("timestamp")
        if resample_rule:
            frame = (
                frame.resample(resample_rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                        "quote_volume": "sum",
                    }
                )
                .dropna(subset=["close"])
            )
        best = frame
        best_meta = result.get("meta") or {}
        if len(frame) >= limit * 0.9:
            break

    if best is None or best.empty:
        raise DataError(f"Yahoo: {symbol} için veri yok")

    meta = best_meta
    exchange = (
        meta.get("fullExchangeName")
        or meta.get("exchangeName")
        or "Yahoo Finance"
    )
    return best, "Yahoo Finance", str(exchange)


def _at(values: Optional[list], index: int) -> Optional[float]:
    if not values or index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _read_cache(
    key: str, interval: str, limit: int
) -> Optional[Tuple[pd.DataFrame, dict]]:
    csv_path, meta_path = _cache_paths(key)
    if not csv_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = float(meta.get("fetched_at_epoch", 0))
        age = time.time() - fetched_at
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        for column in ("open", "high", "low", "close", "volume", "quote_volume"):
            if column not in df.columns:
                df[column] = np.nan
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["close"]).sort_index().tail(limit)
        return df, {**meta, "age_seconds": age}
    except Exception as exc:  # corrupted cache must never break analysis
        logger.debug("Cache read error %s: %s", key, exc)
        return None


def _write_cache(
    key: str,
    df: pd.DataFrame,
    resolved: ResolvedSymbol,
    interval: str,
    source: str,
    exchange: str,
) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        csv_path, meta_path = _cache_paths(key)
        tmp_csv = csv_path.with_suffix(".tmp")
        df.to_csv(tmp_csv)
        os.replace(tmp_csv, csv_path)
        meta = {
            "symbol": resolved.symbol,
            "market": resolved.market,
            "interval": interval,
            "source": source,
            "exchange": exchange,
            "rows": int(len(df)),
            "fetched_at_epoch": time.time(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.debug("Cache write error %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    market: str = "auto",
    interval: str = "1d",
    limit: int = 500,
    use_cache: bool = True,
) -> MarketData:
    """Fetch validated OHLCV data for a symbol.

    Parameters
    ----------
    symbol:
        Ticker or pair, e.g. ``BTCUSDT``, ``ETH/USDT``, ``AAPL``, ``THYAO``.
    market:
        ``auto``, ``crypto``, ``us`` or ``bist``.
    interval:
        ``1h``, ``4h`` (crypto only), ``1d`` or ``1w``.
    limit:
        Number of bars requested (80..5000).
    use_cache:
        When false the cache is bypassed for reading (it is still refreshed).
    """
    resolved = resolve_symbol(symbol, market)
    interval = (interval or "1d").lower().strip()
    interval_ms(interval)  # validation
    limit = int(max(80, min(5000, limit)))
    if resolved.market != "crypto" and interval == "4h":
        raise DataError(
            "Hisse senetleri için 4 saatlik veri desteklenmiyor; 1h veya 1d kullanın."
        )

    key = _cache_key(resolved, interval, limit)
    cached = _read_cache(key, interval, limit) if use_cache else None
    if cached:
        df, meta = cached
        if meta.get("age_seconds", 1e9) <= _ttl_seconds(interval):
            return MarketData(
                symbol=resolved.symbol,
                name=resolved.name,
                market=resolved.market,
                currency=resolved.currency,
                exchange=meta.get("exchange", ""),
                interval=interval,
                source=str(meta.get("source", "cache")),
                df=df,
                fetched_at=datetime.now(timezone.utc)
                - pd.Timedelta(seconds=float(meta.get("age_seconds", 0))),
                resolved=resolved,
                stale=False,
            )

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    }
    errors: List[str] = []
    frame: Optional[pd.DataFrame] = None
    source = exchange = ""

    with httpx.Client(
        headers=headers, timeout=25.0, follow_redirects=True
    ) as client:
        providers = []
        if resolved.market == "crypto":
            providers = [
                ("Binance", _fetch_binance),
                ("Bybit", _fetch_bybit),
                ("Yahoo", _fetch_yahoo),
            ]
        else:
            providers = [("Yahoo", _fetch_yahoo)]

        for provider_name, provider in providers:
            try:
                frame, source, exchange = provider(client, resolved, interval, limit)
                if frame is not None and len(frame) >= 80:
                    break
                errors.append(f"{provider_name}: yetersiz veri")
                frame = None
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")
                logger.debug("Provider %s failed for %s: %s", provider_name, symbol, exc)
                frame = None

    if frame is None or frame.empty:
        # Stale cache is better than no analysis at all.
        if cached:
            df, meta = cached
            return MarketData(
                symbol=resolved.symbol,
                name=resolved.name,
                market=resolved.market,
                currency=resolved.currency,
                exchange=meta.get("exchange", ""),
                interval=interval,
                source=str(meta.get("source", "cache")) + " (eski önbellek)",
                df=df,
                fetched_at=datetime.now(timezone.utc)
                - pd.Timedelta(seconds=float(meta.get("age_seconds", 0))),
                resolved=resolved,
                stale=True,
            )
        raise DataError(
            f"{resolved.symbol} verisi alınamadı. " + " | ".join(errors[-3:])
        )

    frame = _finalize_frame(
        frame, market=resolved.market, interval=interval, limit=limit
    )
    _write_cache(key, frame, resolved, interval, source, exchange)

    name = resolved.name
    if source == "Yahoo Finance" and name == resolved.symbol:
        name = resolved.name

    return MarketData(
        symbol=resolved.symbol,
        name=name,
        market=resolved.market,
        currency=resolved.currency,
        exchange=exchange,
        interval=interval,
        source=source,
        df=frame,
        fetched_at=datetime.now(timezone.utc),
        resolved=resolved,
        stale=False,
    )


def clear_cache(symbol: Optional[str] = None) -> int:
    """Delete cached files. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    pattern = "*.csv" if not symbol else f"*{symbol.upper()}*.csv"
    for path in CACHE_DIR.glob(pattern):
        try:
            path.unlink()
            meta = path.with_suffix(".json")
            if meta.exists():
                meta.unlink()
            removed += 1
        except OSError:
            pass
    return removed

"""Crypto Market data source (Binance, Coinbase, etc.)."""

import asyncio
import json
from datetime import datetime
from typing import List, Optional

import ccxt.async_support as ccxt
import websocket
from loguru import logger

from src.data_sources.base import BaseDataSource
from config.settings import settings


class CryptoMarketDataSource(BaseDataSource):
    """Cryptocurrency market data source.

    Uses:
    - ccxt library for exchange data (Binance, Coinbase, etc.)
    - Binance WebSocket for real-time trade feed
    - whale alert style detection from trade size
    """

    # High-volume crypto pairs to monitor
    DEFAULT_SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
        "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "FIL/USDT",
    ]

    # Major exchange WebSocket URLs
    BINANCE_WS = "wss://stream.binance.com:9443/ws"
    COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, exchanges: List[str] = None):
        super().__init__(name="CryptoMarket", market="crypto")
        self._exchanges = exchanges or ["binance"]
        self._exchange_clients = {}
        self._ws_connections = {}

    async def start(self):
        await super().start()
        # Initialize ccxt exchange clients
        for exchange_id in self._exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                client = exchange_class({
                    "apiKey": getattr(settings, f"{exchange_id}_api_key", ""),
                    "secret": getattr(settings, f"{exchange_id}_secret_key", ""),
                    "sandbox": False,
                    "enableRateLimit": True,
                })
                self._exchange_clients[exchange_id] = client
                logger.info(f"[CryptoMarket] Connected to {exchange_id}")
            except Exception as e:
                logger.error(f"[CryptoMarket] Failed to connect to {exchange_id}: {e}")

    async def stop(self):
        await super().stop()
        for name, client in self._exchange_clients.items():
            try:
                await client.close()
            except Exception:
                pass
        for ws in self._ws_connections.values():
            try:
                ws.close()
            except Exception:
                pass

    async def fetch_realtime(self, symbols: List[str] = None) -> List[dict]:
        """Fetch current prices for crypto pairs."""
        symbols = symbols or self.DEFAULT_SYMBOLS
        results = []

        for exchange_id, client in self._exchange_clients.items():
            try:
                # Fetch tickers in batch
                tickers = await client.fetch_tickers(symbols)

                for symbol, ticker in tickers.items():
                    results.append({
                        "symbol": symbol.replace("/", ""),
                        "market": "crypto",
                        "price": float(ticker.get("last", 0)),
                        "bid": float(ticker.get("bid", 0)),
                        "ask": float(ticker.get("ask", 0)),
                        "volume_24h": float(ticker.get("quoteVolume", 0)),
                        "price_change_24h_pct": float(
                            ticker.get("percentage", 0) or 0
                        ),
                        "high_24h": float(ticker.get("high", 0)),
                        "low_24h": float(ticker.get("low", 0)),
                        "exchange": exchange_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            except Exception as e:
                logger.error(f"[CryptoMarket] Realtime fetch error ({exchange_id}): {e}")

        return results

    async def fetch_historical(
        self, symbol: str, period: str = "1d", interval: str = "1h"
    ) -> List[dict]:
        """Fetch historical OHLCV data for crypto pair."""
        # Convert period to ccxt timeframe
        timeframe_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
        }
        timeframe = timeframe_map.get(interval, "1h")

        # Calculate limit from period
        limit_map = {
            "1d": 24, "5d": 120, "1wk": 168,
            "1mo": 720, "3mo": 2160,
        }
        limit = limit_map.get(period, 100)

        results = []

        for exchange_id, client in self._exchange_clients.items():
            try:
                # Normalize symbol for exchange
                ccxt_symbol = symbol if "/" in symbol else f"{symbol[:3]}/{symbol[3:]}"

                ohlcv = await client.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)

                for candle in ohlcv:
                    results.append({
                        "symbol": symbol.replace("/", ""),
                        "market": "crypto",
                        "timestamp": datetime.utcfromtimestamp(candle[0] / 1000).isoformat(),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "exchange": exchange_id,
                    })

                break  # Use first successful exchange

            except Exception as e:
                logger.error(f"[CryptoMarket] Historical error ({exchange_id}): {e}")

        return results

    async def fetch_large_trades(
        self, symbols: List[str] = None, min_value: float = None
    ) -> List[dict]:
        """Fetch recent large trades from exchanges.

        Uses exchange-specific endpoints for recent trade history,
        then filters by value threshold.
        """
        symbols = symbols or self.DEFAULT_SYMBOLS
        min_value = min_value or settings.whale_threshold_crypto
        results = []

        for exchange_id, client in self._exchange_clients.items():
            try:
                for symbol in symbols:
                    ccxt_symbol = symbol if "/" in symbol else f"{symbol[:3]}/{symbol[3:]}"

                    # Fetch recent trades (limit: last 1000 trades)
                    trades = await client.fetch_trades(ccxt_symbol, limit=500)

                    for trade in trades:
                        price = float(trade.get("price", 0))
                        amount = float(trade.get("amount", 0))
                        total_value = price * amount

                        if total_value >= min_value:
                            results.append({
                                "symbol": symbol.replace("/", ""),
                                "market": "crypto",
                                "side": trade.get("side", "unknown"),
                                "price": price,
                                "quantity": amount,
                                "total_value_usd": total_value,
                                "trade_type": "whale" if total_value >= min_value * 5 else "large",
                                "exchange": exchange_id,
                                "trade_id": trade.get("id"),
                                "timestamp": trade.get("datetime", datetime.utcnow().isoformat()),
                                "details": {
                                    "taker_or_maker": trade.get("takerOrMaker"),
                                    "fee": trade.get("fee"),
                                }
                            })

                    await asyncio.sleep(0.5)  # Rate limit between symbols

            except Exception as e:
                logger.error(f"[CryptoMarket] Large trades error ({exchange_id}): {e}")

        # Sort by value descending
        results.sort(key=lambda x: x["total_value_usd"], reverse=True)
        return results

    async def subscribe_realtime(self, symbols: List[str] = None):
        """Subscribe to Binance WebSocket for real-time trade data."""
        symbols = symbols or self.DEFAULT_SYMBOLS

        # Build Binance WebSocket stream names
        stream_names = []
        for symbol in symbols:
            stream_symbol = symbol.lower().replace("/", "")
            stream_names.append(f"{stream_symbol}@trade")

        streams = "/".join(stream_names)
        ws_url = f"{self.BINANCE_WS}/streams={streams}"

        logger.info(f"[CryptoMarket] Connecting to Binance WebSocket...")

        def on_message(ws, message):
            try:
                data = json.loads(message)
                trade = data.get("data", data)

                price = float(trade.get("p", 0))
                quantity = float(trade.get("q", 0))
                total_value = price * quantity

                # Only emit if significant
                if total_value >= settings.whale_threshold_crypto * 0.1:
                    asyncio.create_task(self._emit({
                        "symbol": trade.get("s", ""),
                        "market": "crypto",
                        "side": "buy" if trade.get("m") is False else "sell",
                        "price": price,
                        "quantity": quantity,
                        "total_value_usd": total_value,
                        "trade_type": "realtime",
                        "exchange": "binance",
                        "trade_id": trade.get("t"),
                        "timestamp": datetime.utcfromtimestamp(
                            trade.get("T", 0) / 1000
                        ).isoformat(),
                    }))

            except Exception as e:
                logger.debug(f"[CryptoMarket] WS message error: {e}")

        def on_error(ws, error):
            logger.error(f"[CryptoMarket] WS error: {error}")

        def on_close(ws, close_status, close_msg):
            logger.warning("[CryptoMarket] WS connection closed")

        def on_open(ws):
            logger.info("[CryptoMarket] WebSocket connected to Binance")

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )

        self._ws_connections["binance"] = ws

        # Run in background thread (websocket-client is sync)
        import threading
        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()

        # Keep the async loop alive
        while self._running:
            await asyncio.sleep(1)

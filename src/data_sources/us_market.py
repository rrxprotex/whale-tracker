"""US Market data source (NYSE, NASDAQ)."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import yfinance as yf
import httpx
from loguru import logger

from src.data_sources.base import BaseDataSource
from config.settings import settings


class USMarketDataSource(BaseDataSource):
    """US stock market data source.

    Uses:
    - yfinance for historical data and basic real-time quotes
    - Alpha Vantage API for detailed volume/flow data
    - Polling-based real-time simulation
    """

    # High-volume symbols to monitor for whale activity
    DEFAULT_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        "JPM", "BAC", "GS", "MS", "V", "MA", "WMT", "JNJ",
        "SPY", "QQQ", "IWM", "DIA",  # ETFs
    ]

    def __init__(self):
        super().__init__(name="USMarket", market="us")
        self._alpha_vantage_key = settings.alpha_vantage_api_key
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self):
        await super().start()
        self._http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("[USMarket] HTTP client initialized")

    async def stop(self):
        await super().stop()
        if self._http_client:
            await self._http_client.aclose()

    async def fetch_realtime(self, symbols: List[str] = None) -> List[dict]:
        """Fetch current prices for US stocks."""
        symbols = symbols or self.DEFAULT_SYMBOLS
        results = []

        try:
            # yfinance batch download (faster than individual queries)
            tickers = yf.Tickers(" ".join(symbols))

            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.fast_info

                    last_price = info.get("lastPrice", 0)
                    prev_close = info.get("previousClose", 0)
                    volume = info.get("lastVolume", 0)

                    price_change_pct = 0.0
                    if prev_close > 0:
                        price_change_pct = ((last_price - prev_close) / prev_close) * 100

                    results.append({
                        "symbol": symbol,
                        "market": "us",
                        "price": float(last_price),
                        "prev_close": float(prev_close),
                        "volume": float(volume),
                        "price_change_pct": round(price_change_pct, 2),
                        "market_cap": float(info.get("marketCap", 0)),
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                except Exception as e:
                    logger.debug(f"[USMarket] Error fetching {symbol}: {e}")

        except Exception as e:
            logger.error(f"[USMarket] Batch fetch error: {e}")

        return results

    async def fetch_historical(
        self, symbol: str, period: str = "1d", interval: str = "1h"
    ) -> List[dict]:
        """Fetch historical OHLCV data for US stock.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"
            interval: "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            records = []
            for idx, row in df.iterrows():
                records.append({
                    "symbol": symbol,
                    "market": "us",
                    "timestamp": idx.isoformat(),
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": float(row.get("Volume", 0)),
                })

            return records

        except Exception as e:
            logger.error(f"[USMarket] Historical fetch error for {symbol}: {e}")
            return []

    async def fetch_large_trades(
        self, symbols: List[str] = None, min_value: float = None
    ) -> List[dict]:
        """Fetch large/block trades using Alpha Vantage or yfinance volume analysis.

        Since real-time tick data requires paid API, this estimates large trades
        from volume anomalies and unusual options activity.
        """
        symbols = symbols or self.DEFAULT_SYMBOLS
        min_value = min_value or settings.whale_threshold_us
        large_trades = []

        try:
            # Use Alpha Vantage for TIME_SERIES_DAILY with volume
            if self._alpha_vantage_key:
                large_trades = await self._fetch_alpha_vantage_flow(symbols, min_value)
            else:
                # Fallback: Estimate from volume spikes via yfinance
                large_trades = await self._estimate_large_trades(symbols, min_value)

        except Exception as e:
            logger.error(f"[USMarket] Large trades fetch error: {e}")

        return large_trades

    async def _fetch_alpha_vantage_flow(
        self, symbols: List[str], min_value: float
    ) -> List[dict]:
        """Fetch volume and trade data from Alpha Vantage API."""
        results = []

        for symbol in symbols[:5]:  # Rate limit: 5 requests per minute
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "apikey": self._alpha_vantage_key,
                    "outputsize": "compact",
                }

                response = await self._http_client.get(url, params=params)
                data = response.json()

                time_series = data.get("Time Series (Daily)", {})
                if time_series:
                    latest_date = list(time_series.keys())[0]
                    latest = time_series[latest_date]

                    volume = int(latest.get("5. volume", 0))
                    close = float(latest.get("4. close", 0))
                    total_value = volume * close

                    if total_value >= min_value:
                        results.append({
                            "symbol": symbol,
                            "market": "us",
                            "side": "buy",  # Default, would need more data
                            "price": close,
                            "quantity": volume,
                            "total_value_usd": total_value,
                            "trade_type": "block" if total_value >= min_value * 10 else "large",
                            "exchange": "NYSE/NASDAQ",
                            "timestamp": datetime.utcnow().isoformat(),
                            "details": {
                                "open": float(latest.get("1. open", 0)),
                                "high": float(latest.get("2. high", 0)),
                                "low": float(latest.get("3. low", 0)),
                                "volume": volume,
                            }
                        })

                # Rate limit: Alpha Vantage free tier
                await asyncio.sleep(12)  # 5 requests/min = 12s between requests

            except Exception as e:
                logger.debug(f"[USMarket] Alpha Vantage error for {symbol}: {e}")

        return results

    async def _estimate_large_trades(
        self, symbols: List[str], min_value: float
    ) -> List[dict]:
        """Estimate large trades from volume analysis when no API key available."""
        results = []

        try:
            tickers = yf.Tickers(" ".join(symbols))

            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    hist = ticker.history(period="1mo", interval="1d")

                    if hist.empty:
                        continue

                    avg_volume = hist["Volume"].mean()
                    latest_volume = hist["Volume"].iloc[-1]
                    latest_close = hist["Close"].iloc[-1]

                    # Detect volume spike (potential whale activity)
                    if avg_volume > 0 and latest_volume > avg_volume * settings.volume_spike_multiplier:
                        total_value = latest_volume * latest_close

                        # Estimate buy vs sell from price action
                        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest_close
                        side = "buy" if latest_close > prev_close else "sell"

                        results.append({
                            "symbol": symbol,
                            "market": "us",
                            "side": side,
                            "price": float(latest_close),
                            "quantity": float(latest_volume),
                            "total_value_usd": float(total_value),
                            "trade_type": "volume_spike",
                            "exchange": "NYSE/NASDAQ",
                            "timestamp": datetime.utcnow().isoformat(),
                            "details": {
                                "avg_volume_30d": float(avg_volume),
                                "volume_multiplier": round(latest_volume / avg_volume, 2),
                                "price_change_pct": round(
                                    ((latest_close - prev_close) / prev_close) * 100, 2
                                ),
                            }
                        })

                except Exception as e:
                    logger.debug(f"[USMarket] Analysis error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"[USMarket] Estimate error: {e}")

        return results

    async def subscribe_realtime(self, symbols: List[str] = None):
        """Poll-based real-time subscription (yfinance doesn't support WebSocket)."""
        symbols = symbols or self.DEFAULT_SYMBOLS
        logger.info(f"[USMarket] Starting polling for {len(symbols)} symbols...")

        while self._running:
            try:
                data = await self.fetch_realtime(symbols)
                for item in data:
                    await self._emit(item)
            except Exception as e:
                logger.error(f"[USMarket] Polling error: {e}")

            await asyncio.sleep(30)  # Poll every 30 seconds

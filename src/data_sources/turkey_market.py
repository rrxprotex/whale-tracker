"""Turkish Market (Borsa Istanbul / BIST) data source."""

import asyncio
from datetime import datetime
from typing import List, Optional

import yfinance as yf
import httpx
from loguru import logger

from src.data_sources.base import BaseDataSource
from config.settings import settings


class TurkeyMarketDataSource(BaseDataSource):
    """Turkish stock market data source (BIST).

    Uses:
    - yfinance with BIST symbols (e.g., THYAO.IS)
    - KAP (Kamuyu Aydınlatma Platformu) for announcements
    - Midas API (if key available) for real-time data

    BIST Symbols format: SYMBOL.IS (e.g., THYAO.IS, GARAN.IS)
    """

    # Popular BIST-100 symbols
    DEFAULT_SYMBOLS = [
        "THYAO", "GARAN", "ASELS", "EREGL", "BIMAS",
        "AKBNK", "SISE", "KOZAL", "SAHOL", "TUPRS",
        "KCHOL", "HALKB", "VAKBN", "TCELL", "PGSUS",
        "PETKM", "TOASO", "FROTO", "DOHOL", "TAVHL",
    ]

    def __init__(self):
        super().__init__(name="TurkeyMarket", market="turkey")
        self._http_client: Optional[httpx.AsyncClient] = None
        self._midas_api_key = settings.midas_api_key

    async def start(self):
        await super().start()
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def stop(self):
        await super().stop()
        if self._http_client:
            await self._http_client.aclose()

    @staticmethod
    def _to_yfinance_symbol(symbol: str) -> str:
        """Convert BIST symbol to yfinance format (THYAO -> THYAO.IS)."""
        if symbol.endswith(".IS"):
            return symbol
        return f"{symbol}.IS"

    @staticmethod
    def _to_bist_symbol(yf_symbol: str) -> str:
        """Convert yfinance format back to BIST symbol."""
        return yf_symbol.replace(".IS", "")

    async def fetch_realtime(self, symbols: List[str] = None) -> List[dict]:
        """Fetch current prices for BIST stocks."""
        symbols = symbols or self.DEFAULT_SYMBOLS
        yf_symbols = [self._to_yfinance_symbol(s) for s in symbols]
        results = []

        try:
            tickers = yf.Tickers(" ".join(yf_symbols))

            for symbol, yf_symbol in zip(symbols, yf_symbols):
                try:
                    ticker = tickers.tickers[yf_symbol]
                    info = ticker.fast_info

                    last_price = info.get("lastPrice", 0)
                    prev_close = info.get("previousClose", 0)
                    volume = info.get("lastVolume", 0)

                    price_change_pct = 0.0
                    if prev_close > 0:
                        price_change_pct = ((last_price - prev_close) / prev_close) * 100

                    # Convert TL volume to USD approximation
                    # (USD/TRY rate needed for accurate conversion)
                    approx_usd_rate = 35.0  # Approximate, should be fetched dynamically

                    results.append({
                        "symbol": symbol,
                        "market": "turkey",
                        "price": float(last_price),
                        "currency": "TRY",
                        "prev_close": float(prev_close),
                        "volume": float(volume),
                        "price_change_pct": round(price_change_pct, 2),
                        "total_value_try": float(last_price * volume),
                        "total_value_usd_approx": float(last_price * volume / approx_usd_rate),
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                except Exception as e:
                    logger.debug(f"[TurkeyMarket] Error fetching {symbol}: {e}")

        except Exception as e:
            logger.error(f"[TurkeyMarket] Batch fetch error: {e}")

        return results

    async def fetch_historical(
        self, symbol: str, period: str = "1d", interval: str = "1h"
    ) -> List[dict]:
        """Fetch historical OHLCV data for BIST stock."""
        yf_symbol = self._to_yfinance_symbol(symbol)

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            records = []
            for idx, row in df.iterrows():
                records.append({
                    "symbol": self._to_bist_symbol(yf_symbol),
                    "market": "turkey",
                    "timestamp": idx.isoformat(),
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": float(row.get("Volume", 0)),
                    "currency": "TRY",
                })

            return records

        except Exception as e:
            logger.error(f"[TurkeyMarket] Historical error for {symbol}: {e}")
            return []

    async def fetch_large_trades(
        self, symbols: List[str] = None, min_value: float = None
    ) -> List[dict]:
        """Fetch large/block trades for BIST.

        BIST provides block trade (特殊 işlem) data through KAP.
        Volume analysis is used as fallback.
        """
        symbols = symbols or self.DEFAULT_SYMBOLS
        min_value_tl = min_value or settings.whale_threshold_turkey
        results = []

        yf_symbols = [self._to_yfinance_symbol(s) for s in symbols]

        try:
            tickers = yf.Tickers(" ".join(yf_symbols))

            for symbol, yf_symbol in zip(symbols, yf_symbols):
                try:
                    ticker = tickers.tickers[yf_symbol]
                    hist = ticker.history(period="1mo", interval="1d")

                    if hist.empty:
                        continue

                    avg_volume = hist["Volume"].mean()
                    latest_volume = hist["Volume"].iloc[-1]
                    latest_close = hist["Close"].iloc[-1]

                    total_value_tl = latest_volume * latest_close

                    # Detect volume spike
                    if avg_volume > 0 and latest_volume > avg_volume * settings.volume_spike_multiplier:
                        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest_close
                        side = "buy" if latest_close > prev_close else "sell"

                        results.append({
                            "symbol": symbol,
                            "market": "turkey",
                            "side": side,
                            "price": float(latest_close),
                            "quantity": float(latest_volume),
                            "total_value_tl": float(total_value_tl),
                            "total_value_usd_approx": float(total_value_tl / 35.0),
                            "trade_type": "volume_spike",
                            "exchange": "BIST",
                            "timestamp": datetime.utcnow().isoformat(),
                            "details": {
                                "avg_volume_30d": float(avg_volume),
                                "volume_multiplier": round(latest_volume / avg_volume, 2),
                                "price_change_pct": round(
                                    ((latest_close - prev_close) / prev_close) * 100, 2
                                ),
                                "currency": "TRY",
                            }
                        })

                except Exception as e:
                    logger.debug(f"[TurkeyMarket] Analysis error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"[TurkeyMarket] Large trades error: {e}")

        return results

    async def fetch_kap_announcements(self, symbol: str = None) -> List[dict]:
        """Fetch latest announcements from KAP (Kamuyu Aydınlatma Platformu).

        This provides institutional activity, insider trading, and block trade info.
        """
        try:
            url = "https://www.kap.org.tr/tr/api/market-data"
            # KAP API structure (may require specific headers)
            response = await self._http_client.get(
                url,
                headers={"User-Agent": "WhaleTracker/1.0"}
            )

            if response.status_code == 200:
                return response.json()[:20]  # Latest 20 announcements

        except Exception as e:
            logger.debug(f"[TurkeyMarket] KAP fetch error: {e}")

        return []

    async def fetch_usdtry_rate(self) -> float:
        """Fetch current USD/TRY exchange rate."""
        try:
            ticker = yf.Ticker("USDTRY=X")
            info = ticker.fast_info
            return float(info.get("lastPrice", 35.0))
        except Exception:
            return 35.0  # Fallback

    async def subscribe_realtime(self, symbols: List[str] = None):
        """Poll-based real-time subscription for BIST."""
        symbols = symbols or self.DEFAULT_SYMBOLS
        logger.info(f"[TurkeyMarket] Starting polling for {len(symbols)} symbols...")

        while self._running:
            try:
                data = await self.fetch_realtime(symbols)
                for item in data:
                    await self._emit(item)
            except Exception as e:
                logger.error(f"[TurkeyMarket] Polling error: {e}")

            # BIST trades between 10:00-18:00 Turkey time
            await asyncio.sleep(30)

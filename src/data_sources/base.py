"""Base class for all market data sources."""

import abc
from datetime import datetime
from typing import List, Optional, Callable
from loguru import logger


class BaseDataSource(abc.ABC):
    """Abstract base class for all market data sources.

    Each data source must implement:
    - fetch_realtime(): Get current/live price data
    - fetch_historical(): Get historical OHLCV data
    - fetch_large_trades(): Get large/block trades
    - subscribe_realtime(): Subscribe to real-time WebSocket feed
    """

    def __init__(self, name: str, market: str):
        self.name = name
        self.market = market
        self._callbacks: List[Callable] = []
        self._running = False
        logger.info(f"[{self.name}] Initialized for market: {self.market}")

    def on_data(self, callback: Callable):
        """Register callback for incoming data."""
        self._callbacks.append(callback)

    async def _emit(self, data: dict):
        """Emit data to all registered callbacks."""
        for cb in self._callbacks:
            try:
                await cb(data)
            except Exception as e:
                logger.error(f"[{self.name}] Callback error: {e}")

    @abc.abstractmethod
    async def fetch_realtime(self, symbols: List[str]) -> List[dict]:
        """Fetch current/live price data for given symbols."""
        ...

    @abc.abstractmethod
    async def fetch_historical(
        self, symbol: str, period: str = "1d", interval: str = "1h"
    ) -> List[dict]:
        """Fetch historical OHLCV data.

        Args:
            symbol: Stock/crypto symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk)
        """
        ...

    @abc.abstractmethod
    async def fetch_large_trades(
        self, symbols: List[str], min_value: float
    ) -> List[dict]:
        """Fetch large/block trades above a minimum value threshold."""
        ...

    async def subscribe_realtime(self, symbols: List[str]):
        """Subscribe to real-time data feed (WebSocket or polling)."""
        logger.warning(f"[{self.name}] subscribe_realtime not implemented")

    async def start(self):
        """Start the data source."""
        self._running = True
        logger.info(f"[{self.name}] Started")

    async def stop(self):
        """Stop the data source."""
        self._running = False
        logger.info(f"[{self.name}] Stopped")

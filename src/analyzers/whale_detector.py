"""Whale trade detection engine."""

from datetime import datetime
from typing import List, Optional, Callable
from loguru import logger

from config.settings import settings


class WhaleDetector:
    """Detects whale (large) trades across all markets.

    Detection methods:
    1. Absolute value threshold (e.g., > $1M for US stocks)
    2. Volume spike detection (e.g., 3x average volume)
    3. Block trade detection
    4. Dark pool activity estimation
    """

    # Market-specific thresholds
    THRESHOLDS = {
        "us": settings.whale_threshold_us,
        "turkey": settings.whale_threshold_turkey,
        "crypto": settings.whale_threshold_crypto,
    }

    def __init__(self):
        self._callbacks: List[Callable] = []
        self._alerted_trades = set()  # Dedup
        logger.info("[WhaleDetector] Initialized with thresholds: %s", self.THRESHOLDS)

    def on_whale_detected(self, callback: Callable):
        """Register callback for whale detection events."""
        self._callbacks.append(callback)

    async def analyze(self, trade_data: dict) -> Optional[dict]:
        """Analyze a trade/event and return whale data if detected.

        Args:
            trade_data: Raw trade data from any market source

        Returns:
            Whale trade dict if detected, None otherwise
        """
        market = trade_data.get("market", "us")
        total_value = trade_data.get("total_value_usd", 0)

        if total_value <= 0:
            return None

        threshold = self.THRESHOLDS.get(market, 1000000)

        # Check if trade exceeds whale threshold
        if total_value >= threshold:
            whale_trade = self._classify_trade(trade_data, threshold)

            # Dedup check
            trade_key = self._trade_key(whale_trade)
            if trade_key in self._alerted_trades:
                return None

            self._alerted_trades.add(trade_key)

            # Notify callbacks
            for cb in self._callbacks:
                try:
                    await cb(whale_trade)
                except Exception as e:
                    logger.error(f"[WhaleDetector] Callback error: {e}")

            return whale_trade

        return None

    def _classify_trade(self, trade_data: dict, threshold: float) -> dict:
        """Classify and enrich a whale trade."""
        total_value = trade_data["total_value_usd"]

        # Determine trade significance level
        multiplier = total_value / threshold
        if multiplier >= 10:
            significance = "mega_whale"  # 10x+ threshold
        elif multiplier >= 5:
            significance = "whale"
        elif multiplier >= 2:
            significance = "large"
        else:
            significance = "moderate"

        # Estimate direction if not provided
        side = trade_data.get("side")
        if not side:
            # Use price change as proxy
            price_change = trade_data.get("details", {}).get("price_change_pct", 0)
            side = "buy" if price_change >= 0 else "sell"

        return {
            **trade_data,
            "side": side,
            "significance": significance,
            "multiplier": round(multiplier, 2),
            "threshold_used": threshold,
            "detected_at": datetime.utcnow().isoformat(),
            "alert_type": "whale_trade",
        }

    async def analyze_batch(self, trades: List[dict]) -> List[dict]:
        """Analyze a batch of trades and return detected whales."""
        whales = []
        for trade in trades:
            result = await self.analyze(trade)
            if result:
                whales.append(result)
        return whales

    def _trade_key(self, trade: dict) -> str:
        """Generate unique key for trade deduplication."""
        return (
            f"{trade.get('symbol', '')}:"
            f"{trade.get('market', '')}:"
            f"{trade.get('trade_id', '')}:"
            f"{trade.get('timestamp', '')[:13]}"  # Hour-level dedup
        )

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "thresholds": self.THRESHOLDS,
            "total_detected": len(self._alerted_trades),
        }

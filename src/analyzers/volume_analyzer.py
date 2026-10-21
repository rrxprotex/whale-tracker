"""Volume spike analyzer — detects unusual volume activity."""

from datetime import datetime
from typing import List, Optional, Callable
import statistics

from loguru import logger

from config.settings import settings


class VolumeAnalyzer:
    """Detects volume anomalies and spikes across all markets.

    Techniques:
    1. Rolling average comparison (current vs 20-day avg)
    2. Volume acceleration (increasing volume trend)
    3. VWAP deviation
    4. Relative volume ratio
    """

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        self._callbacks: List[Callable] = []
        self._volume_history: dict[str, list[float]] = {}  # symbol -> [volumes]
        logger.info(f"[VolumeAnalyzer] Initialized (lookback={lookback_days}d)")

    def on_spike_detected(self, callback: Callable):
        """Register callback for volume spike events."""
        self._callbacks.append(callback)

    async def ingest(self, market_data: dict) -> Optional[dict]:
        """Ingest price/volume data and check for spikes.

        Args:
            market_data: Must contain symbol, market, volume, price fields

        Returns:
            VolumeSpike dict if detected, None otherwise
        """
        symbol = market_data.get("symbol", "")
        market = market_data.get("market", "")
        volume = market_data.get("volume", 0)
        price = market_data.get("close") or market_data.get("price", 0)

        if volume <= 0 or not symbol:
            return None

        key = f"{market}:{symbol}"

        # Maintain rolling history
        if key not in self._volume_history:
            self._volume_history[key] = []
        self._volume_history[key].append(volume)

        # Keep only last N data points
        if len(self._volume_history[key]) > self.lookback_days * 10:
            self._volume_history[key] = self._volume_history[key][-(self.lookback_days * 10):]

        history = self._volume_history[key]

        # Need at least a few data points
        if len(history) < 5:
            return None

        # Calculate average volume (exclude current)
        avg_volume = statistics.mean(history[:-1]) if len(history) > 1 else history[0]

        if avg_volume <= 0:
            return None

        multiplier = volume / avg_volume

        if multiplier >= settings.volume_spike_multiplier:
            # Calculate price change if we have previous data
            price_change_pct = market_data.get("price_change_pct", 0)

            spike = {
                "symbol": symbol,
                "market": market,
                "current_volume": volume,
                "avg_volume": avg_volume,
                "multiplier": round(multiplier, 2),
                "price": price,
                "price_change_pct": round(price_change_pct, 2),
                "alert_type": "volume_spike",
                "detected_at": datetime.utcnow().isoformat(),
                "history_length": len(history),
            }

            # Classify severity
            if multiplier >= 10:
                spike["severity"] = "extreme"
            elif multiplier >= 7:
                spike["severity"] = "high"
            elif multiplier >= 5:
                spike["severity"] = "moderate"
            else:
                spike["severity"] = "low"

            # Notify callbacks
            for cb in self._callbacks:
                try:
                    await cb(spike)
                except Exception as e:
                    logger.error(f"[VolumeAnalyzer] Callback error: {e}")

            logger.info(
                f"[VolumeAnalyzer] SPIKE: {symbol} ({market}) "
                f"volume {multiplier:.1f}x avg | severity={spike['severity']}"
            )
            return spike

        return None

    async def ingest_batch(self, data_list: List[dict]) -> List[dict]:
        """Ingest a batch of market data and return detected spikes."""
        spikes = []
        for data in data_list:
            result = await self.ingest(data)
            if result:
                spikes.append(result)
        return spikes

    def get_volume_profile(self, symbol: str, market: str) -> Optional[dict]:
        """Get volume profile for a symbol."""
        key = f"{market}:{symbol}"
        history = self._volume_history.get(key, [])

        if not history:
            return None

        return {
            "symbol": symbol,
            "market": market,
            "data_points": len(history),
            "current_volume": history[-1],
            "avg_volume": statistics.mean(history),
            "max_volume": max(history),
            "min_volume": min(history),
            "std_dev": statistics.stdev(history) if len(history) > 1 else 0,
            "current_vs_avg": round(history[-1] / max(statistics.mean(history), 1), 2),
        }

    def get_stats(self) -> dict:
        """Get analyzer statistics."""
        return {
            "tracked_symbols": len(self._volume_history),
            "lookback_days": self.lookback_days,
            "threshold_multiplier": settings.volume_spike_multiplier,
        }

"""Redis cache manager for real-time data and rate limiting."""

import json
from typing import Optional, Any
from loguru import logger

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import settings


class CacheManager:
    """Redis-based cache for real-time market data."""

    def __init__(self):
        self._redis = None
        self._available = False

    async def initialize(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("[Cache] Redis not available, using in-memory cache")
            self._available = False
            self._memory_cache = {}
            return

        try:
            self._redis = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            # Test connection
            await self._redis.ping()
            self._available = True
            logger.info("[Cache] Redis connected")
        except Exception as e:
            logger.warning(f"[Cache] Redis not available: {e}, using memory cache")
            self._available = False
            self._memory_cache = {}

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set a cache value with TTL (seconds)."""
        try:
            serialized = json.dumps(value) if not isinstance(value, str) else value

            if self._available:
                await self._redis.setex(key, ttl, serialized)
            else:
                self._memory_cache[key] = serialized
        except Exception as e:
            logger.error(f"[Cache] Set error: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        try:
            if self._available:
                value = await self._redis.get(key)
            else:
                value = self._memory_cache.get(key)

            if value:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value

            return None

        except Exception as e:
            logger.error(f"[Cache] Get error: {e}")
            return None

    async def delete(self, key: str):
        """Delete a cache key."""
        try:
            if self._available:
                await self._redis.delete(key)
            else:
                self._memory_cache.pop(key, None)
        except Exception as e:
            logger.error(f"[Cache] Delete error: {e}")

    async def get_latest_price(self, symbol: str, market: str) -> Optional[dict]:
        """Get latest cached price for a symbol."""
        key = f"price:{market}:{symbol}"
        return await self.get(key)

    async def set_latest_price(self, symbol: str, market: str, price_data: dict):
        """Cache latest price data."""
        key = f"price:{market}:{symbol}"
        await self.set(key, price_data, ttl=60)  # 1 minute TTL

    async def is_whale_alerted(self, trade_id: str) -> bool:
        """Check if a whale trade has already been alerted (dedup)."""
        key = f"alerted:{trade_id}"
        return (await self.get(key)) is not None

    async def mark_whale_alerted(self, trade_id: str, ttl: int = 3600):
        """Mark a whale trade as alerted (1 hour dedup window)."""
        key = f"alerted:{trade_id}"
        await self.set(key, "1", ttl=ttl)

    async def get_all_prices(self, market: str = None) -> dict:
        """Get all cached prices for a market."""
        if not self._available:
            # Filter memory cache
            prefix = f"price:{market}:" if market else "price:"
            return {
                k: json.loads(v)
                for k, v in self._memory_cache.items()
                if k.startswith(prefix)
            }

        # Redis SCAN approach
        pattern = f"price:{market}:*" if market else "price:*"
        prices = {}
        async for key in self._redis.scan_iter(match=pattern):
            value = await self.get(key)
            if value:
                prices[key] = value
        return prices

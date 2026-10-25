"""Regression tests for the Redis fallback cache."""

from __future__ import annotations

import asyncio

import pytest

try:
    from src.storage.cache import CacheManager
except ModuleNotFoundError as exc:  # pragma: no cover - local minimal env only
    if exc.name == "loguru":
        pytest.skip("WhaleTracker runtime dependencies are not installed", allow_module_level=True)
    raise


def test_memory_cache_honors_ttl_and_removes_expired_prices():
    async def scenario():
        cache = CacheManager()
        cache._available = False
        cache._memory_cache = {}

        await cache.set("price:crypto:BTCUSDT", {"price": 100}, ttl=0.02)
        assert await cache.get("price:crypto:BTCUSDT") == {"price": 100}
        assert await cache.get_all_prices("crypto") == {
            "price:crypto:BTCUSDT": {"price": 100}
        }

        await asyncio.sleep(0.05)

        assert await cache.get("price:crypto:BTCUSDT") is None
        assert await cache.get_all_prices("crypto") == {}

    asyncio.run(scenario())

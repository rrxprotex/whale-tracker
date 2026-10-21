"""Main orchestrator — coordinates all components."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings
from src.data_sources import USMarketDataSource, TurkeyMarketDataSource, CryptoMarketDataSource
from src.analyzers import WhaleDetector, VolumeAnalyzer
from src.storage import DatabaseManager, CacheManager
from src.alerts import TelegramAlert, WebhookAlert


class Orchestrator:
    """Central orchestrator that:

    1. Starts all data sources (US, Turkey, Crypto)
    2. Routes data through analyzers (whale detection, volume spikes)
    3. Stores results in database
    4. Sends alerts via Telegram / webhooks
    """

    def __init__(self):
        # Data sources
        self.us_source = USMarketDataSource()
        self.turkey_source = TurkeyMarketDataSource()
        self.crypto_source = CryptoMarketDataSource()

        # Analyzers
        self.whale_detector = WhaleDetector()
        self.volume_analyzer = VolumeAnalyzer()

        # Storage
        self.db = DatabaseManager()
        self.cache = CacheManager()

        # Alerts
        self.telegram = TelegramAlert()
        self.webhook = WebhookAlert()

        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("🐋 WhaleTracker Orchestrator — Initializing...")
        logger.info("=" * 60)

        # Database
        await self.db.initialize()

        # Cache
        await self.cache.initialize()

        # Webhooks
        await self.webhook.start()

        # Wire up callbacks
        self._wire_callbacks()

        logger.info("[Orchestrator] Initialization complete")

    def _wire_callbacks(self):
        """Wire up data flow: sources → analyzers → storage → alerts."""

        # US Market data → analyzers
        self.us_source.on_data(self._on_market_data)
        self.turkey_source.on_data(self._on_market_data)
        self.crypto_source.on_data(self._on_market_data)

        # Whale detection → alerts
        self.whale_detector.on_whale_detected(self._on_whale_detected)

        # Volume spike → alerts
        self.volume_analyzer.on_spike_detected(self._on_spike_detected)

    async def start(self):
        """Start all data sources and processing."""
        self._running = True
        logger.info("[Orchestrator] Starting all data sources...")

        # Start data sources
        await self.us_source.start()
        await self.turkey_source.start()
        await self.crypto_source.start()

        # Launch polling tasks
        self._tasks.append(
            asyncio.create_task(self._poll_us_market())
        )
        self._tasks.append(
            asyncio.create_task(self._poll_turkey_market())
        )
        self._tasks.append(
            asyncio.create_task(self._poll_crypto_market())
        )

        # Launch crypto WebSocket (real-time)
        self._tasks.append(
            asyncio.create_task(self._start_crypto_websocket())
        )

        # Daily summary scheduler
        self._tasks.append(
            asyncio.create_task(self._daily_summary_scheduler())
        )

        logger.info(f"[Orchestrator] Started with {len(self._tasks)} tasks")

    async def stop(self):
        """Gracefully stop all components."""
        self._running = False
        logger.info("[Orchestrator] Stopping...")

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop data sources
        await self.us_source.stop()
        await self.turkey_source.stop()
        await self.crypto_source.stop()

        # Close connections
        await self.webhook.stop()
        await self.cache.close()

        logger.info("[Orchestrator] Stopped")

    # ============================================================
    # Polling Tasks
    # ============================================================

    async def _poll_us_market(self):
        """Periodically poll US market data."""
        while self._running:
            try:
                logger.debug("[Orchestrator] Polling US market...")
                data = await self.us_source.fetch_realtime()

                for item in data:
                    await self._on_market_data(item)
                    await self.cache.set_latest_price(
                        item["symbol"], "us", item
                    )

                # Also check for large trades
                large = await self.us_source.fetch_large_trades()
                whales = await self.whale_detector.analyze_batch(large)
                for w in whales:
                    await self._on_whale_detected(w)

            except Exception as e:
                logger.error(f"[Orchestrator] US poll error: {e}")

            await asyncio.sleep(60)  # Poll every 60 seconds

    async def _poll_turkey_market(self):
        """Periodically poll Turkish market data."""
        while self._running:
            try:
                logger.debug("[Orchestrator] Polling Turkey market...")
                data = await self.turkey_source.fetch_realtime()

                for item in data:
                    await self._on_market_data(item)
                    await self.cache.set_latest_price(
                        item["symbol"], "turkey", item
                    )

                large = await self.turkey_source.fetch_large_trades()
                whales = await self.whale_detector.analyze_batch(large)
                for w in whales:
                    await self._on_whale_detected(w)

            except Exception as e:
                logger.error(f"[Orchestrator] Turkey poll error: {e}")

            await asyncio.sleep(60)

    async def _poll_crypto_market(self):
        """Periodically poll crypto market for large trades."""
        while self._running:
            try:
                logger.debug("[Orchestrator] Polling crypto market...")
                data = await self.crypto_source.fetch_realtime()

                for item in data:
                    await self._on_market_data(item)
                    await self.cache.set_latest_price(
                        item["symbol"], "crypto", item
                    )

                large = await self.crypto_source.fetch_large_trades()
                whales = await self.whale_detector.analyze_batch(large)
                for w in whales:
                    await self._on_whale_detected(w)

            except Exception as e:
                logger.error(f"[Orchestrator] Crypto poll error: {e}")

            await asyncio.sleep(30)  # Crypto: poll more frequently

    async def _start_crypto_websocket(self):
        """Start real-time WebSocket for crypto."""
        try:
            await self.crypto_source.subscribe_realtime()
        except Exception as e:
            logger.error(f"[Orchestrator] Crypto WebSocket error: {e}")

    async def _daily_summary_scheduler(self):
        """Send daily summary every 24 hours."""
        while self._running:
            await asyncio.sleep(86400)  # 24 hours
            try:
                stats = await self.db.get_whale_stats(hours=24)
                await self.telegram.send_daily_summary(stats)
            except Exception as e:
                logger.error(f"[Orchestrator] Daily summary error: {e}")

    # ============================================================
    # Data Handlers
    # ============================================================

    async def _on_market_data(self, data: dict):
        """Handle incoming market data from any source."""
        try:
            # Analyze for volume spikes
            await self.volume_analyzer.ingest(data)

            # Store in database
            if "close" in data or "price" in data:
                await self.db.save_price_data({
                    "symbol": data.get("symbol", ""),
                    "market": data.get("market", ""),
                    "open": data.get("open", data.get("price", 0)),
                    "high": data.get("high", data.get("price", 0)),
                    "low": data.get("low", data.get("price", 0)),
                    "close": data.get("close", data.get("price", 0)),
                    "volume": data.get("volume", 0),
                    "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
                })

        except Exception as e:
            logger.error(f"[Orchestrator] Market data handler error: {e}")

    async def _on_whale_detected(self, whale: dict):
        """Handle detected whale trade."""
        try:
            logger.warning(
                f"🐋 WHALE: {whale['symbol']} ({whale['market'].upper()}) "
                f"${whale.get('total_value_usd', 0):,.0f} "
                f"[{whale.get('significance', 'unknown')}]"
            )

            # Save to database
            await self.db.save_whale_trade(whale)

            # Send alerts
            await self.telegram.send_whale_alert(whale)
            await self.webhook.send_to_all(whale)

        except Exception as e:
            logger.error(f"[Orchestrator] Whale handler error: {e}")

    async def _on_spike_detected(self, spike: dict):
        """Handle detected volume spike."""
        try:
            logger.warning(
                f"📊 VOLUME SPIKE: {spike['symbol']} ({spike['market'].upper()}) "
                f"{spike.get('multiplier', 0)}x avg "
                f"[{spike.get('severity', 'unknown')}]"
            )

            # Save to database
            await self.db.save_volume_spike(spike)

            # Send alert
            await self.telegram.send_volume_spike_alert(spike)

        except Exception as e:
            logger.error(f"[Orchestrator] Spike handler error: {e}")

    # ============================================================
    # Public API (for dashboard / REST)
    # ============================================================

    async def get_live_prices(self, market: str = None) -> dict:
        """Get all cached live prices."""
        return await self.cache.get_all_prices(market)

    async def get_whale_trades(
        self, market: str = None, hours: int = 24, min_value: float = None
    ) -> List[dict]:
        """Get recent whale trades."""
        return await self.db.get_recent_whale_trades(market, hours, min_value)

    async def get_stats(self, hours: int = 24) -> dict:
        """Get whale trading statistics."""
        return await self.db.get_whale_stats(hours)

#!/usr/bin/env python3
"""
🐋 WhaleTracker — Real-time Whale Trade Detection System

Monitors US, Turkey (BIST), and Crypto markets for:
- Large/block trades (whale activity)
- Volume spikes
- Dark pool activity

Usage:
    python main.py              # Run orchestrator + dashboard
    python main.py --no-api     # Run orchestrator only (no web server)
    python main.py --port 8080  # Custom port
"""

import asyncio
import signal
import sys
import argparse

import uvicorn
from loguru import logger

from config.settings import settings
from src.orchestrator import Orchestrator
from src.dashboard.app import create_app


# ============================================================
# Log Configuration
# ============================================================

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)
logger.add(
    "logs/whale_tracker_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    compression="zip",
)


# ============================================================
# Application Lifecycle
# ============================================================

async def run_app(port: int = 8000, host: str = "0.0.0.0", api_only: bool = False):
    """Run the full application."""

    # Create orchestrator
    orchestrator = Orchestrator()
    await orchestrator.initialize()

    # Create FastAPI app
    app = create_app(orchestrator)

    # Setup shutdown handler
    shutdown_event = asyncio.Event()

    def handle_shutdown(*_):
        logger.info("Shutdown signal received...")
        shutdown_event.set()

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, handle_shutdown)

    # Start orchestrator
    await orchestrator.start()

    logger.info("=" * 60)
    logger.info("🐋 WhaleTracker is LIVE!")
    logger.info(f"   Dashboard: http://{host}:{port}")
    logger.info(f"   API Docs:  http://{host}:{port}/docs")
    logger.info(f"   Markets:   US, Turkey (BIST), Crypto")
    logger.info("=" * 60)

    if not api_only:
        # Run FastAPI server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
        server = uvicorn.Server(config)

        # Run server and wait for shutdown
        async def run_server():
            await server.serve()

        server_task = asyncio.create_task(run_server())

        # Wait for shutdown signal
        await shutdown_event.wait()

        # Graceful shutdown
        server.should_exit = True
        await server_task
    else:
        # Run without web server
        logger.info("Running in --no-api mode (orchestrator only)")
        await shutdown_event.wait()

    # Cleanup
    await orchestrator.stop()
    logger.info("🐋 WhaleTracker stopped.")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="🐋 WhaleTracker — Real-time Whale Trade Detection"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Dashboard port (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Dashboard host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--no-api", action="store_true",
        help="Run without web server (orchestrator only)"
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)"
    )

    args = parser.parse_args()

    if args.log_level:
        settings.log_level = args.log_level

    logger.info("Starting WhaleTracker...")

    try:
        asyncio.run(
            run_app(
                port=args.port,
                host=args.host,
                api_only=args.no_api,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()

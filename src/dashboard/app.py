"""FastAPI dashboard and REST API for WhaleTracker."""

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.orchestrator import Orchestrator


def create_app(orchestrator: Orchestrator) -> FastAPI:
    """Create FastAPI application."""

    app = FastAPI(
        title="🐋 WhaleTracker",
        description="Real-time whale trade detection across US, Turkey & Crypto markets",
        version="1.0.0",
    )

    # Store orchestrator reference
    app.state.orchestrator = orchestrator

    # ============================================================
    # Dashboard Routes
    # ============================================================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Main dashboard page."""
        return DASHBOARD_HTML

    # ============================================================
    # REST API Routes
    # ============================================================

    @app.get("/api/live-prices")
    async def api_live_prices(market: Optional[str] = Query(None)):
        """Get live cached prices for all symbols."""
        orch = app.state.orchestrator
        prices = await orch.get_live_prices(market)
        return {"data": prices, "count": len(prices)}

    @app.get("/api/whale-trades")
    async def api_whale_trades(
        market: Optional[str] = Query(None),
        hours: int = Query(24, ge=1, le=168),
        min_value: Optional[float] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Get recent whale trades."""
        orch = app.state.orchestrator
        trades = await orch.get_whale_trades(market, hours, min_value)
        return {
            "data": trades[:limit],
            "count": len(trades),
            "filters": {
                "market": market,
                "hours": hours,
                "min_value": min_value,
            }
        }

    @app.get("/api/stats")
    async def api_stats(hours: int = Query(24, ge=1, le=168)):
        """Get whale trading statistics."""
        orch = app.state.orchestrator
        stats = await orch.get_stats(hours)
        return {"data": stats}

    @app.get("/api/stats/detailed")
    async def api_stats_detailed():
        """Get detailed statistics from all components."""
        orch = app.state.orchestrator
        return {
            "whale_detector": orch.whale_detector.get_stats(),
            "volume_analyzer": orch.volume_analyzer.get_stats(),
            "markets": {
                "us": {"status": "active" if orch.us_source._running else "stopped"},
                "turkey": {"status": "active" if orch.turkey_source._running else "stopped"},
                "crypto": {"status": "active" if orch.crypto_source._running else "stopped"},
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.get("/api/symbols")
    async def api_symbols():
        """Get all monitored symbols."""
        return {
            "us": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
                "JPM", "BAC", "GS", "MS", "V", "MA", "WMT", "JNJ",
                "SPY", "QQQ", "IWM", "DIA",
            ],
            "turkey": [
                "THYAO", "GARAN", "ASELS", "EREGL", "BIMAS",
                "AKBNK", "SISE", "KOZAL", "SAHOL", "TUPRS",
                "KCHOL", "HALKB", "VAKBN", "TCELL", "PGSUS",
            ],
            "crypto": [
                "BTC", "ETH", "BNB", "SOL", "XRP",
                "DOGE", "ADA", "AVAX", "DOT", "LINK",
                "MATIC", "UNI", "ATOM", "LTC", "FIL",
            ],
        }

    @app.get("/api/whales/top")
    async def api_top_whales(hours: int = Query(24)):
        """Get top whale symbols ranked by activity."""
        orch = app.state.orchestrator
        stats = await orch.get_stats(hours)
        return {"data": stats.get("top_symbols", [])}

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }

    return app


# ============================================================
# Dashboard HTML
# ============================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐋 WhaleTracker Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e17;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1f2e, #0d1117);
            padding: 20px 30px;
            border-bottom: 1px solid #21262d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 24px; }
        .header .status { color: #3fb950; font-size: 14px; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 20px;
        }
        .card h3 { color: #58a6ff; margin-bottom: 12px; font-size: 16px; }
        .stat-value { font-size: 32px; font-weight: 700; }
        .stat-label { color: #8b949e; font-size: 13px; margin-top: 4px; }
        .whale-list { list-style: none; }
        .whale-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #21262d;
        }
        .whale-item:last-child { border-bottom: none; }
        .symbol { font-weight: 600; font-size: 16px; }
        .market-tag {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .market-us { background: #1f6feb33; color: #58a6ff; }
        .market-turkey { background: #f0883333; color: #f08833; }
        .market-crypto { background: #8b5cf633; color: #a78bfa; }
        .value { font-weight: 600; color: #3fb950; }
        .side-buy { color: #3fb950; }
        .side-sell { color: #f85149; }
        .refresh-btn {
            background: #238636;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        .refresh-btn:hover { background: #2ea043; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #21262d; }
        th { color: #8b949e; font-weight: 600; font-size: 12px; text-transform: uppercase; }
        .live-dot {
            width: 8px; height: 8px;
            background: #3fb950;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .section-title {
            font-size: 18px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #8b949e;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐋 WhaleTracker</h1>
        <div>
            <span class="live-dot"></span>
            <span class="status">LIVE</span>
            <button class="refresh-btn" onclick="refreshAll()">↻ Refresh</button>
        </div>
    </div>

    <div class="container">
        <!-- Stats Cards -->
        <div class="grid" id="stats-grid">
            <div class="card">
                <h3>🐋 Whale Trades (24h)</h3>
                <div class="stat-value" id="stat-trades">—</div>
                <div class="stat-label">detected whale trades</div>
            </div>
            <div class="card">
                <h3>💰 Total Value (24h)</h3>
                <div class="stat-value" id="stat-value">—</div>
                <div class="stat-label">USD traded by whales</div>
            </div>
            <div class="card">
                <h3>🟢 Buy Ratio</h3>
                <div class="stat-value" id="stat-buy-ratio">—</div>
                <div class="stat-label">whale buy percentage</div>
            </div>
            <div class="card">
                <h3>📊 Markets Active</h3>
                <div class="stat-value">3</div>
                <div class="stat-label">US, Turkey, Crypto</div>
            </div>
        </div>

        <!-- Recent Whale Trades -->
        <div class="card">
            <div class="section-title">🐋 Recent Whale Trades</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Market</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Value (USD)</th>
                        <th>Type</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody id="whale-table-body">
                    <tr><td colspan="7" class="empty-state">Loading whale trades...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        async function refreshAll() {
            try {
                // Stats
                const statsRes = await fetch('/api/stats?hours=24');
                const statsData = await statsRes.json();
                const stats = statsData.data;

                document.getElementById('stat-trades').textContent =
                    (stats.total_trades || 0).toLocaleString();
                document.getElementById('stat-value').textContent =
                    '$' + (stats.total_value_usd || 0).toLocaleString(undefined,
                    {maximumFractionDigits: 0});
                document.getElementById('stat-buy-ratio').textContent =
                    (stats.buy_ratio || 0) + '%';

                // Whale trades
                const tradesRes = await fetch('/api/whale-trades?hours=24&limit=50');
                const tradesData = await tradesRes.json();

                const tbody = document.getElementById('whale-table-body');
                if (tradesData.data && tradesData.data.length > 0) {
                    tbody.innerHTML = tradesData.data.map(t => `
                        <tr>
                            <td><span class="symbol">${t.symbol}</span></td>
                            <td><span class="market-tag market-${t.market}">${t.market.toUpperCase()}</span></td>
                            <td class="side-${t.side}">${t.side.toUpperCase()}</td>
                            <td>$${Number(t.price).toLocaleString(undefined,
                                {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
                            <td class="value">$${Number(t.total_value_usd).toLocaleString()}</td>
                            <td>${t.trade_type}</td>
                            <td>${new Date(t.timestamp).toLocaleString()}</td>
                        </tr>
                    `).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">' +
                        '🐋 No whale trades detected yet. Monitoring markets...</td></tr>';
                }
            } catch (err) {
                console.error('Refresh error:', err);
            }
        }

        // Auto-refresh every 30 seconds
        refreshAll();
        setInterval(refreshAll, 30000);
    </script>
</body>
</html>
"""

# 🐋 WhaleTracker

Real-time whale trade detection system that monitors **US stocks**, **Turkey (BIST)**, and **Crypto** markets for large/block trades and volume spikes.

## Features

- 🇺🇸 **US Market** — NYSE, NASDAQ whale detection
- 🇹🇷 **Turkey (BIST)** — Borsa İstanbul large trades
- ₿ **Crypto** — Binance, Coinbase whale monitoring (WebSocket real-time)
- 📊 **Volume Spike Detection** — 3x+ average volume alerts
- 🐋 **Whale Classification** — Mega whale / Whale / Large / Moderate
- 📱 **Telegram Alerts** — Instant notifications
- 🔔 **Webhook Support** — Slack, Discord, custom endpoints
- 🌐 **Web Dashboard** — Real-time monitoring UI
- 💾 **Database Storage** — PostgreSQL (prod) / SQLite (dev)
- ⚡ **Redis Cache** — Fast real-time data access

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  US Market  │   │   BIST      │   │   Crypto    │
│  (yfinance) │   │  (yfinance) │   │  (ccxt+ws)  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                  │                  │
       └────────┬─────────┴─────────┬───────┘
                │                   │
         ┌──────▼──────────────────▼──────┐
         │        Data Orchestrator        │
         │   (polling + WebSocket feeds)   │
         └──────┬──────────────┬──────────┘
                │              │
    ┌───────────▼───┐  ┌──────▼──────────┐
    │ Whale Detector│  │ Volume Analyzer  │
    │ (> $1M trades)│  │ (3x avg volume)  │
    └───────┬───────┘  └──────┬──────────┘
            │                  │
            └────────┬─────────┘
                     │
         ┌───────────▼────────────┐
         │    Storage + Alerts     │
         │  DB │ Cache │ Telegram  │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │    Web Dashboard       │
         │   FastAPI + HTML UI    │
         └────────────────────────┘
```

## Quick Start

### 1. Clone & Install

```bash
cd whale-tracker
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (optional for basic functionality)
```

### 3. Run

```bash
# Run with dashboard
python main.py

# Run without web server (CLI only)
python main.py --no-api

# Custom port
python main.py --port 3000

# Debug mode
python main.py --log-level DEBUG
```

### 4. Open Dashboard

Navigate to: **http://localhost:8000**

## Docker Deployment

```bash
# Full stack (app + PostgreSQL + Redis)
docker-compose up -d

# Check logs
docker-compose logs -f whale-tracker
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web Dashboard |
| GET | `/api/live-prices` | Live cached prices |
| GET | `/api/whale-trades` | Recent whale trades |
| GET | `/api/stats` | 24h whale statistics |
| GET | `/api/whales/top` | Top whale symbols |
| GET | `/api/symbols` | All monitored symbols |
| GET | `/api/health` | Health check |

### Query Parameters

```
GET /api/whale-trades?market=us&hours=24&min_value=5000000&limit=50
GET /api/live-prices?market=crypto
GET /api/stats?hours=48
```

## Whale Detection Thresholds

| Market | Default Threshold | Currency |
|--------|-------------------|----------|
| US | $1,000,000 | USD |
| Turkey | ₺5,000,000 | TRY |
| Crypto | $100,000 | USD |

Customize in `.env`:
```bash
WHALE_THRESHOLD_US=1000000
WHALE_THRESHOLD_TURKEY=5000000
WHALE_THRESHOLD_CRYPTO=100000
VOLUME_SPIKE_MULTIPLIER=3.0
```

## Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Add to `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=your_chat_id
```

## API Keys (Optional)

Basic functionality works **without API keys** (uses yfinance for data).

For enhanced data:
- **Alpha Vantage** (free: 25 req/day) — US market detailed volume
- **Polygon.io** — Real-time US market data
- **Binance/Coinbase** — Crypto exchange data
- **Midas API** — Turkish market data

## Project Structure

```
whale-tracker/
├── main.py                      # Entry point
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration
├── src/
│   ├── __init__.py
│   ├── models.py                # Database models + Pydantic schemas
│   ├── orchestrator.py          # Main coordination engine
│   ├── data_sources/
│   │   ├── base.py              # Abstract data source
│   │   ├── us_market.py         # US stocks (yfinance)
│   │   ├── turkey_market.py     # BIST stocks (yfinance)
│   │   └── crypto_market.py     # Crypto (ccxt + WebSocket)
│   ├── analyzers/
│   │   ├── whale_detector.py    # Large trade detection
│   │   └── volume_analyzer.py   # Volume spike detection
│   ├── storage/
│   │   ├── database.py          # PostgreSQL/SQLite
│   │   └── cache.py             # Redis cache
│   ├── alerts/
│   │   ├── telegram_alert.py    # Telegram notifications
│   │   └── webhook_alert.py     # Webhook integrations
│   └── dashboard/
│       └── app.py               # FastAPI + HTML dashboard
└── logs/                        # Log files
```

## License

MIT

# 🐋 WhaleTracker

**Public repo:** https://github.com/rrxprotex/whale-tracker

Real-time whale trade detection system that monitors **US stocks**, **Turkey (BIST)**, and **Crypto** markets for large/block trades and volume spikes.

It also includes an independent **advanced analytics engine** that produces Turkish, offline HTML research reports with technical, spectral, statistical, pattern, regime and probabilistic forecast sections. The analytics output is **not investment advice**; every projection is a scenario with explicit uncertainty.

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

## Advanced Analytics Engine

`src/analytics/` is independent from the whale-detection orchestrator. It only needs `numpy`, `pandas`, `httpx` and (for the web UI) `fastapi`. No API key is required. Charts are generated as **inline SVG**, so a report works offline without Plotly or a CDN.

### CLI

```bash
# Crypto (Binance public REST, Bybit fallback)
python analyze.py BTCUSDT

# US stock (Yahoo Finance chart API)
python analyze.py AAPL --market us

# BIST stock
python analyze.py THYAO --market bist --limit 500

# Multiple symbols, custom horizon, no files written
python analyze.py BTCUSDT ETHUSDT SOLUSDT --interval 4h --horizon 42 --no-save

# Open the generated HTML report in the browser
python analyze.py BTCUSDT --open
```

Reports and JSON snapshots are written to `reports/` (ignored by git). The command prints a compact terminal summary and writes a standalone HTML file plus the full JSON data.

### Web UI and API

```bash
uvicorn src.analytics.api:app --reload --port 8010
```

Open **http://localhost:8010** for the Turkish analysis UI, or use the JSON/HTML endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service and supported-interval status |
| GET | `/api/catalog` | Curated crypto/US/BIST symbol catalog |
| GET | `/api/symbols?q=btc` | Symbol search |
| GET | `/api/summary/{symbol}` | Compact JSON summary |
| GET | `/api/analyze/{symbol}` | Full report JSON + SVG charts |
| GET | `/api/report/{symbol}` | Standalone HTML report |
| POST | `/api/analyze` | JSON body variant |

Common query parameters: `market=auto|crypto|us|bist`, `interval=1h|4h|1d|1w`, `limit=80..5000`, `horizon=1..365`, `use_cache=true|false`, `save=true|false`.

Examples:

```bash
curl "http://localhost:8010/api/summary/BTCUSDT?interval=1d&horizon=30"
curl "http://localhost:8010/api/report/AAPL?market=us" -o AAPL.html
```

### Methods

| Layer | What is computed |
|-------|------------------|
| Technical | EMA/SMA, RSI, MACD, ADX/DI, ATR, Bollinger, stochastic, OBV, volume profile inputs; explainable weighted technical score |
| Spectral | Welch PSD, dominant cycles, harmonic regression, Hilbert phase, Haar wavelet energy, SSA components and SSA projection |
| Statistical | Hurst (R/S and DFA), ADF, variance ratio, permutation entropy, Sharpe/Sortino/Calmar, VaR/CVaR, drawdown, tail ratio, Amihud and Roll spread |
| Patterns | Pivot-based support/resistance clusters, Fibonacci retracements, volume profile (POC/value area/HVN), candlestick patterns, RSI divergence, channel and double top/bottom checks |
| Regime | Trend and volatility regimes, market phase, Welch change points, monthly and weekday seasonality |
| Forecast | 8-model ensemble: drift, linear trend, Holt, AR, Ornstein–Uhlenbeck, harmonic, SSA and k-NN analog; walk-forward weighting; block-bootstrap Monte Carlo; calibrated 80%/95% bands |
| Scoring | Regime-aware composite score with component weights and contributions, confidence decomposition and Turkish narrative |

Forecast bands are **empirical scenario ranges**, not guarantees. The engine reports walk-forward error, direction hit rate, model weights and data-quality warnings so the user can judge how much to trust a projection.

### Data sources and cache

- Crypto: Binance public REST (primary), Bybit public REST (fallback), Yahoo Finance (fallback).
- US/BIST: Yahoo Finance chart API.
- Responses are cached under `data/cache/` with a TTL; stale cache is used only as a fallback when a provider is unavailable and is flagged in the report.
- No private keys, exchange accounts or paid data subscriptions are required.

### Tests

```bash
python -m pytest tests/ -q
```

The tests use synthetic OHLCV data and do not require network access.

### Limitations

- The engine analyzes price/volume history only; it does not know news, order-book depth, funding, macro events or company fundamentals.
- Fourier/SSA/harmonic methods assume recurring structure that may disappear; regime changes invalidate historical cycles.
- Walk-forward validation reduces, but does not eliminate, overfitting and look-ahead risk.
- Thinly traded BIST names and stale Yahoo bars can produce unreliable volume/pattern statistics; check the data-quality warning in the report.
- **This project is for research and education. It is not investment advice, a recommendation, or a guarantee of future prices.**

## Docker Deployment

```bash
# Full stack (app + PostgreSQL + Redis)
docker-compose up -d

# Check logs
docker-compose logs -f whale-tracker
```

## Whale Monitoring API Endpoints

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
├── main.py                      # Whale-tracker entry point
├── analyze.py                   # Advanced analytics CLI
├── requirements.txt
├── LICENSE
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
│   ├── analytics/               # Independent multi-method analysis package
│   │   ├── market_data.py       # Binance/Bybit/Yahoo + disk cache
│   │   ├── symbols.py           # Crypto/US/BIST catalog and resolution
│   │   ├── indicators.py        # Technical indicators + snapshot
│   │   ├── spectral.py          # Welch, harmonic, Hilbert, Haar, SSA
│   │   ├── statistics.py        # Hurst, ADF, variance ratio, risk
│   │   ├── patterns.py          # Levels, Fibonacci, profile, candles
│   │   ├── regime.py            # Regimes, change points, seasonality
│   │   ├── forecast.py          # 8-model ensemble + Monte Carlo
│   │   ├── scoring.py           # Composite score + Turkish narrative
│   │   ├── charts.py            # Dependency-free inline SVG charts
│   │   ├── report.py            # Standalone Turkish HTML report
│   │   ├── engine.py            # Pipeline orchestration
│   │   └── api.py               # FastAPI analytics UI/endpoints
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
├── tests/
│   └── test_analytics.py        # Offline synthetic-data tests
├── data/cache/                  # Runtime market-data cache (gitignored)
├── reports/                     # Generated HTML/JSON reports (gitignored)
└── logs/                        # Log files (gitignored)
```

## License

MIT

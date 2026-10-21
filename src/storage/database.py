"""Database manager for persistent storage."""

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models import (
    WhaleTrade, VolumeSpike, PriceData, Alert,
    Base, init_db, get_session,
)


class DatabaseManager:
    """Async database operations for whale tracker."""

    def __init__(self):
        self._initialized = False

    async def initialize(self, database_url: str = None):
        """Initialize database and create tables."""
        await init_db(database_url)
        self._initialized = True
        logger.info("[Database] Initialized successfully")

    async def save_whale_trade(self, trade: dict):
        """Save a detected whale trade."""
        async for session in get_session():
            try:
                db_trade = WhaleTrade(
                    symbol=trade["symbol"],
                    market=trade["market"],
                    side=trade.get("side", "buy"),
                    price=trade["price"],
                    quantity=trade["quantity"],
                    total_value_usd=trade["total_value_usd"],
                    trade_type=trade.get("trade_type", "regular"),
                    exchange=trade.get("exchange"),
                    details=str(trade.get("details", {})),
                )
                session.add(db_trade)
                await session.commit()
                logger.debug(f"[Database] Saved whale trade: {trade['symbol']} ${trade['total_value_usd']:,.0f}")
            except Exception as e:
                await session.rollback()
                logger.error(f"[Database] Save whale trade error: {e}")

    async def save_volume_spike(self, spike: dict):
        """Save a volume spike detection."""
        async for session in get_session():
            try:
                db_spike = VolumeSpike(
                    symbol=spike["symbol"],
                    market=spike["market"],
                    current_volume=spike["current_volume"],
                    avg_volume=spike["avg_volume"],
                    multiplier=spike["multiplier"],
                    price=spike["price"],
                    price_change_pct=spike.get("price_change_pct", 0),
                )
                session.add(db_spike)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"[Database] Save volume spike error: {e}")

    async def save_price_data(self, data: dict):
        """Save price data for historical analysis."""
        async for session in get_session():
            try:
                db_data = PriceData(
                    symbol=data["symbol"],
                    market=data["market"],
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    volume=data["volume"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                )
                session.add(db_data)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"[Database] Save price data error: {e}")

    async def save_alert(self, alert: dict):
        """Save a sent alert."""
        async for session in get_session():
            try:
                db_alert = Alert(
                    alert_type=alert["alert_type"],
                    symbol=alert["symbol"],
                    market=alert["market"],
                    message=alert["message"],
                    sent=True,
                    channel=alert.get("channel"),
                )
                session.add(db_alert)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"[Database] Save alert error: {e}")

    async def get_recent_whale_trades(
        self,
        market: str = None,
        hours: int = 24,
        min_value: float = None,
        limit: int = 100,
    ) -> List[dict]:
        """Get recent whale trades from database."""
        async for session in get_session():
            try:
                since = datetime.utcnow() - timedelta(hours=hours)

                query = select(WhaleTrade).where(
                    WhaleTrade.timestamp >= since
                )

                if market:
                    query = query.where(WhaleTrade.market == market)
                if min_value:
                    query = query.where(WhaleTrade.total_value_usd >= min_value)

                query = query.order_by(WhaleTrade.total_value_usd.desc()).limit(limit)

                result = await session.execute(query)
                trades = result.scalars().all()

                return [
                    {
                        "id": t.id,
                        "symbol": t.symbol,
                        "market": t.market,
                        "side": t.side,
                        "price": t.price,
                        "quantity": t.quantity,
                        "total_value_usd": t.total_value_usd,
                        "trade_type": t.trade_type,
                        "exchange": t.exchange,
                        "timestamp": t.timestamp.isoformat(),
                    }
                    for t in trades
                ]

            except Exception as e:
                logger.error(f"[Database] Get whale trades error: {e}")
                return []

    async def get_whale_stats(self, hours: int = 24) -> dict:
        """Get whale trading statistics for the last N hours."""
        async for session in get_session():
            try:
                since = datetime.utcnow() - timedelta(hours=hours)

                # Total whale trades count
                count_query = select(func.count(WhaleTrade.id)).where(
                    WhaleTrade.timestamp >= since
                )
                count_result = await session.execute(count_query)
                total_count = count_result.scalar() or 0

                # Total value traded
                value_query = select(func.sum(WhaleTrade.total_value_usd)).where(
                    WhaleTrade.timestamp >= since
                )
                value_result = await session.execute(value_query)
                total_value = value_result.scalar() or 0

                # Top symbols by whale activity
                top_symbols_query = (
                    select(
                        WhaleTrade.symbol,
                        func.count(WhaleTrade.id).label("count"),
                        func.sum(WhaleTrade.total_value_usd).label("total_value"),
                    )
                    .where(WhaleTrade.timestamp >= since)
                    .group_by(WhaleTrade.symbol)
                    .order_by(func.sum(WhaleTrade.total_value_usd).desc())
                    .limit(10)
                )
                top_result = await session.execute(top_symbols_query)
                top_symbols = [
                    {"symbol": row[0], "count": row[1], "total_value": row[2]}
                    for row in top_result.all()
                ]

                # Buy vs Sell ratio
                buy_query = select(func.count(WhaleTrade.id)).where(
                    and_(
                        WhaleTrade.timestamp >= since,
                        WhaleTrade.side == "buy"
                    )
                )
                buy_result = await session.execute(buy_query)
                buy_count = buy_result.scalar() or 0

                return {
                    "period_hours": hours,
                    "total_trades": total_count,
                    "total_value_usd": float(total_value),
                    "buy_count": buy_count,
                    "sell_count": total_count - buy_count,
                    "buy_ratio": round(buy_count / max(total_count, 1) * 100, 1),
                    "top_symbols": top_symbols,
                }

            except Exception as e:
                logger.error(f"[Database] Stats error: {e}")
                return {}

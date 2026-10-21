"""Database models and data schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Enum as SAEnum,
    Boolean, Text, Index
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


# ============================================================
# Enums
# ============================================================

class MarketType(str, Enum):
    US = "us"
    TURKEY = "turkey"
    CRYPTO = "crypto"


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class AlertType(str, Enum):
    WHALE_TRADE = "whale_trade"
    VOLUME_SPIKE = "volume_spike"
    LARGE_BLOCK = "large_block"
    DARK_POOL = "dark_pool"


# ============================================================
# SQLAlchemy ORM Models
# ============================================================

class Base(DeclarativeBase):
    pass


class WhaleTrade(Base):
    """Detected whale trades / large block trades."""
    __tablename__ = "whale_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    market = Column(String(10), nullable=False, index=True)  # MarketType
    side = Column(String(4), nullable=False)  # TradeSide
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    total_value_usd = Column(Float, nullable=False, index=True)
    trade_type = Column(String(20), default="regular")  # regular, block, dark_pool, options
    exchange = Column(String(20))
    details = Column(Text)  # JSON string with extra details
    processed = Column(Boolean, default=False)

    __table_args__ = (
        Index("idx_whale_market_time", "market", "timestamp"),
        Index("idx_whale_value", "total_value_usd"),
    )


class VolumeSpike(Base):
    """Volume spike detections."""
    __tablename__ = "volume_spikes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    market = Column(String(10), nullable=False)
    current_volume = Column(Float, nullable=False)
    avg_volume = Column(Float, nullable=False)
    multiplier = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    price_change_pct = Column(Float, default=0.0)


class PriceData(Base):
    """Stored price data for analysis."""
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    market = Column(String(10), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    turnover = Column(Float)

    __table_args__ = (
        Index("idx_price_symbol_time", "symbol", "timestamp"),
    )


class Alert(Base):
    """Sent alerts log."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    alert_type = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    market = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    sent = Column(Boolean, default=False)
    channel = Column(String(20))  # telegram, webhook, etc.


# ============================================================
# Pydantic Schemas (for API / internal use)
# ============================================================

class WhaleTradeSchema(BaseModel):
    symbol: str
    market: MarketType
    side: TradeSide
    price: float
    quantity: float
    total_value_usd: float
    trade_type: str = "regular"
    exchange: Optional[str] = None
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class VolumeSpikeSchema(BaseModel):
    symbol: str
    market: MarketType
    current_volume: float
    avg_volume: float
    multiplier: float
    price: float
    price_change_pct: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AlertSchema(BaseModel):
    alert_type: AlertType
    symbol: str
    market: MarketType
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Database Engine
# ============================================================

_engine = None
_session_factory = None


async def init_db(database_url: str):
    """Initialize async database engine and create tables."""
    global _engine, _session_factory
    from config.settings import settings

    url = database_url or settings.database_url
    _engine = create_async_engine(url, echo=settings.app_debug)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return _engine


async def get_session() -> AsyncSession:
    """Get an async database session."""
    global _session_factory
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        yield session


async def close_db():
    """Close database connections."""
    global _engine
    if _engine:
        await _engine.dispose()

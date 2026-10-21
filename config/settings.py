"""Application configuration settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === App ===
    app_name: str = "WhaleTracker"
    app_env: str = "development"
    app_debug: bool = True
    log_level: str = "INFO"

    # === Database ===
    database_url: str = "sqlite+aiosqlite:///./whale_tracker.db"

    # === Redis ===
    redis_url: str = "redis://localhost:6379/0"

    # === Telegram ===
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # === US Market ===
    alpha_vantage_api_key: str = ""
    polygon_api_key: str = ""

    # === Turkish Market ===
    midas_api_key: str = ""

    # === Crypto ===
    binance_api_key: str = ""
    binance_secret_key: str = ""
    coinbase_api_key: str = ""
    coinbase_secret_key: str = ""

    # === Whale Thresholds ===
    whale_threshold_us: float = 1_000_000  # USD
    whale_threshold_turkey: float = 5_000_000  # TL
    whale_threshold_crypto: float = 100_000  # USD equivalent

    # === Volume ===
    volume_spike_multiplier: float = 3.0

    @property
    def is_prod(self) -> bool:
        return self.app_env == "production"


# Singleton
settings = Settings()

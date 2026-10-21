"""Telegram alert system for whale detections."""

import asyncio
from typing import Optional
from loguru import logger

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from config.settings import settings


class TelegramAlert:
    """Sends whale detection alerts via Telegram bot."""

    # Emoji mappings
    MARKET_EMOJI = {
        "us": "🇺🇸",
        "turkey": "🇹🇷",
        "crypto": "₿",
    }
    SIDE_EMOJI = {
        "buy": "🟢",
        "sell": "🔴",
    }
    SIGNIFICANCE_EMOJI = {
        "mega_whale": "🐋🐋🐋",
        "whale": "🐋🐋",
        "large": "🐋",
        "moderate": "🐟",
    }
    SEVERITY_EMOJI = {
        "extreme": "🚨🚨🚨",
        "high": "🚨🚨",
        "moderate": "⚠️",
        "low": "📊",
    }

    def __init__(self):
        self._bot: Optional[Bot] = None
        self._chat_id: str = settings.telegram_chat_id
        self._enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)

        if self._enabled and TELEGRAM_AVAILABLE:
            self._bot = Bot(token=settings.telegram_bot_token)
            logger.info("[TelegramAlert] Initialized")
        else:
            if not TELEGRAM_AVAILABLE:
                logger.warning("[TelegramAlert] python-telegram-bot not installed")
            else:
                logger.warning("[TelegramAlert] Bot token or chat ID not configured")

    async def send_whale_alert(self, whale_data: dict):
        """Send a whale trade alert."""
        if not self._enabled:
            self._log_alert(whale_data)
            return

        try:
            message = self._format_whale_message(whale_data)
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
            )
            logger.info(
                f"[TelegramAlert] Whale alert sent: "
                f"{whale_data['symbol']} ${whale_data.get('total_value_usd', 0):,.0f}"
            )
        except Exception as e:
            logger.error(f"[TelegramAlert] Send error: {e}")

    async def send_volume_spike_alert(self, spike_data: dict):
        """Send a volume spike alert."""
        if not self._enabled:
            self._log_alert(spike_data)
            return

        try:
            message = self._format_spike_message(spike_data)
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
            )
            logger.info(f"[TelegramAlert] Volume spike sent: {spike_data['symbol']}")
        except Exception as e:
            logger.error(f"[TelegramAlert] Send error: {e}")

    async def send_daily_summary(self, stats: dict):
        """Send daily whale activity summary."""
        if not self._enabled:
            logger.info(f"[TelegramAlert] Daily summary (no Telegram): {stats}")
            return

        try:
            message = self._format_daily_summary(stats)
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
            )
            logger.info("[TelegramAlert] Daily summary sent")
        except Exception as e:
            logger.error(f"[TelegramAlert] Summary send error: {e}")

    def _format_whale_message(self, data: dict) -> str:
        """Format whale trade as Telegram message."""
        market = data.get("market", "us")
        side = data.get("side", "buy")
        significance = data.get("significance", "whale")
        total_value = data.get("total_value_usd", 0)

        emoji_m = self.MARKET_EMOJI.get(market, "📊")
        emoji_s = self.SIDE_EMOJI.get(side, "❓")
        emoji_sig = self.SIGNIFICANCE_EMOJI.get(significance, "🐋")

        lines = [
            f"{emoji_sig} <b>WHALE TRADE DETECTED</b> {emoji_sig}",
            "",
            f"{emoji_m} <b>{data.get('symbol', '?')}</b> | {market.upper()}",
            f"{emoji_s} <b>Side:</b> {side.upper()}",
            f"💰 <b>Value:</b> ${total_value:,.0f}",
            f"💵 <b>Price:</b> ${data.get('price', 0):,.2f}",
            f"📦 <b>Quantity:</b> {data.get('quantity', 0):,.0f}",
            f"📈 <b>Type:</b> {data.get('trade_type', 'regular')}",
            f"🏛️ <b>Exchange:</b> {data.get('exchange', 'N/A')}",
            f"⏰ <b>Time:</b> {data.get('detected_at', data.get('timestamp', 'N/A'))}",
        ]

        if data.get("multiplier"):
            lines.append(f"📏 <b>Threshold:</b> {data['multiplier']}x normal")

        details = data.get("details", {})
        if details:
            if details.get("volume_multiplier"):
                lines.append(
                    f"📊 <b>Volume:</b> {details['volume_multiplier']}x avg"
                )
            if details.get("price_change_pct"):
                pct = details["price_change_pct"]
                arrow = "📈" if pct > 0 else "📉"
                lines.append(f"{arrow} <b>Change:</b> {pct:+.2f}%")

        lines.append("")
        lines.append("🤖 <i>WhaleTracker</i>")

        return "\n".join(lines)

    def _format_spike_message(self, data: dict) -> str:
        """Format volume spike as Telegram message."""
        severity = data.get("severity", "low")
        emoji_sev = self.SEVERITY_EMOJI.get(severity, "📊")

        lines = [
            f"{emoji_sev} <b>VOLUME SPIKE</b> {emoji_sev}",
            "",
            f"📊 <b>{data.get('symbol', '?')}</b> | {data.get('market', '').upper()}",
            f"📈 <b>Volume:</b> {data.get('multiplier', 0)}x average",
            f"💵 <b>Price:</b> ${data.get('price', 0):,.2f}",
            f"📉 <b>Change:</b> {data.get('price_change_pct', 0):+.2f}%",
            f"⚠️ <b>Severity:</b> {severity.upper()}",
            f"⏰ <b>Time:</b> {data.get('detected_at', 'N/A')}",
            "",
            "🤖 <i>WhaleTracker</i>",
        ]

        return "\n".join(lines)

    def _format_daily_summary(self, stats: dict) -> str:
        """Format daily summary as Telegram message."""
        top = stats.get("top_symbols", [])

        lines = [
            "📊 <b>DAILY WHALE SUMMARY</b>",
            "",
            f"🔢 <b>Total whale trades:</b> {stats.get('total_trades', 0)}",
            f"💰 <b>Total value:</b> ${stats.get('total_value_usd', 0):,.0f}",
            f"🟢 <b>Buy count:</b> {stats.get('buy_count', 0)}",
            f"🔴 <b>Sell count:</b> {stats.get('sell_count', 0)}",
            f"📊 <b>Buy ratio:</b> {stats.get('buy_ratio', 0)}%",
            "",
            "🏆 <b>Top Symbols:</b>",
        ]

        for i, s in enumerate(top[:5], 1):
            lines.append(
                f"  {i}. <b>{s['symbol']}</b> — "
                f"${s['total_value']:,.0f} ({s['count']} trades)"
            )

        lines.extend(["", "🤖 <i>WhaleTracker</i>"])

        return "\n".join(lines)

    def _log_alert(self, data: dict):
        """Log alert when Telegram is not configured."""
        alert_type = data.get("alert_type", "whale_trade")
        symbol = data.get("symbol", "?")
        value = data.get("total_value_usd", 0)
        logger.info(
            f"[Alert] {alert_type}: {symbol} ${value:,.0f} "
            f"(Telegram not configured, logged only)"
        )

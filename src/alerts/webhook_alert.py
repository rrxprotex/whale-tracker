"""Generic webhook alert system for external integrations."""

import json
from typing import List, Optional, Dict
from loguru import logger

import httpx


class WebhookAlert:
    """Sends alerts to external webhook endpoints (Slack, Discord, custom).

    Usage:
        webhook = WebhookAlert()
        webhook.add_endpoint("slack", "https://hooks.slack.com/services/...")
        webhook.add_endpoint("discord", "https://discord.com/api/webhooks/...")
        webhook.add_endpoint("custom", "https://my-api.com/alerts")
        await webhook.send(whale_data, endpoint_name="slack")
    """

    def __init__(self):
        self._endpoints: Dict[str, str] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self):
        self._http_client = httpx.AsyncClient(timeout=10.0)

    async def stop(self):
        if self._http_client:
            await self._http_client.aclose()

    def add_endpoint(self, name: str, url: str):
        """Register a webhook endpoint."""
        self._endpoints[name] = url
        logger.info(f"[WebhookAlert] Added endpoint: {name}")

    async def send(
        self,
        data: dict,
        endpoint_name: str = None,
        endpoints: List[str] = None,
    ):
        """Send alert to webhook(s).

        Args:
            data: Alert data (whale trade, volume spike, etc.)
            endpoint_name: Single endpoint to send to
            endpoints: List of endpoint names to send to
        """
        if not self._http_client:
            await self.start()

        targets = []
        if endpoint_name:
            targets = [endpoint_name]
        elif endpoints:
            targets = endpoints
        else:
            targets = list(self._endpoints.keys())

        for name in targets:
            url = self._endpoints.get(name)
            if not url:
                logger.warning(f"[WebhookAlert] Unknown endpoint: {name}")
                continue

            try:
                payload = self._build_payload(name, data)

                response = await self._http_client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in (200, 201, 204):
                    logger.info(f"[WebhookAlert] Sent to {name}: {response.status_code}")
                else:
                    logger.warning(
                        f"[WebhookAlert] {name} returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )

            except Exception as e:
                logger.error(f"[WebhookAlert] Error sending to {name}: {e}")

    async def send_to_all(self, data: dict):
        """Send alert to all registered endpoints."""
        await self.send(data, endpoints=list(self._endpoints.keys()))

    def _build_payload(self, endpoint_type: str, data: dict) -> dict:
        """Build endpoint-specific payload format."""
        if endpoint_type == "slack":
            return self._slack_payload(data)
        elif endpoint_type == "discord":
            return self._discord_payload(data)
        else:
            return self._generic_payload(data)

    def _slack_payload(self, data: dict) -> dict:
        """Build Slack-formatted payload."""
        alert_type = data.get("alert_type", "whale_trade")
        symbol = data.get("symbol", "?")
        market = data.get("market", "").upper()
        value = data.get("total_value_usd", 0)
        side = data.get("side", "?")

        emoji = "🐋" if alert_type == "whale_trade" else "📊"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Whale Alert: {symbol} ({market})"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Symbol:*\n{symbol}"},
                    {"type": "mrkdwn", "text": f"*Market:*\n{market}"},
                    {"type": "mrkdwn", "text": f"*Value:*\n${value:,.0f}"},
                    {"type": "mrkdwn", "text": f"*Side:*\n{side.upper()}"},
                    {"type": "mrkdwn", "text": f"*Price:*\n${data.get('price', 0):,.2f}"},
                    {"type": "mrkdwn", "text": f"*Type:*\n{data.get('trade_type', 'regular')}"},
                ]
            }
        ]

        if data.get("multiplier"):
            blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"📏 {data['multiplier']}x normal threshold"}
                ]
            })

        return {"blocks": blocks}

    def _discord_payload(self, data: dict) -> dict:
        """Build Discord-formatted embed payload."""
        alert_type = data.get("alert_type", "whale_trade")
        symbol = data.get("symbol", "?")
        market = data.get("market", "").upper()
        value = data.get("total_value_usd", 0)

        color = 0xFF0000 if data.get("side") == "sell" else 0x00FF00

        embed = {
            "title": f"🐋 Whale Alert: {symbol}",
            "description": f"Large {alert_type.replace('_', ' ')} detected on {market}",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Market", "value": market, "inline": True},
                {"name": "Value", "value": f"${value:,.0f}", "inline": True},
                {"name": "Side", "value": data.get("side", "?").upper(), "inline": True},
                {"name": "Price", "value": f"${data.get('price', 0):,.2f}", "inline": True},
                {"name": "Type", "value": data.get("trade_type", "regular"), "inline": True},
            ],
            "footer": {"text": "WhaleTracker 🤖"},
        }

        return {"embeds": [embed]}

    def _generic_payload(self, data: dict) -> dict:
        """Build generic JSON payload."""
        return {
            "event": data.get("alert_type", "unknown"),
            "data": data,
            "source": "whale-tracker",
        }

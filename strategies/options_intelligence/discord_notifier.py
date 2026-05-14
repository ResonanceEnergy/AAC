"""
Gasket Discord Notifier — Trade Confirmation Flow
=====================================================
Sends Options Intelligence trade recommendations to Discord via webhook
and polls for user confirmation via reaction before executing.

Flow:
    1. Pipeline generates scored trade recommendations
    2. Notifier formats and posts to Gasket Discord channel
    3. Adds ✅ and ❌ reactions to the message
    4. Polls for user reaction (up to CONFIRM_TIMEOUT_MINUTES)
    5. Returns CONFIRMED / REJECTED / TIMED_OUT

Requires:
    DISCORD_WEBHOOK_URL — Webhook URL for the Gasket channel
    DISCORD_BOT_TOKEN   — Bot token to read reactions
    GASKET_CHANNEL_ID   — Channel ID where the webhook posts

All communication uses stdlib urllib (no external deps).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIRM_TIMEOUT_MINUTES = int(os.environ.get("TRADE_CONFIRM_TIMEOUT_MINUTES", "15"))
POLL_INTERVAL_SECONDS = 10


class ConfirmationStatus(Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    ERROR = "error"


@dataclass
class TradeRecommendation:
    """A single trade recommendation for display."""
    symbol: str
    strike: float
    expiry: str
    contracts: int
    max_price: float
    score: int
    crisis_vectors: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ConfirmationResult:
    """Result of the Discord confirmation flow."""
    status: ConfirmationStatus
    message_id: Optional[str] = None
    confirmed_by: Optional[str] = None
    elapsed_seconds: float = 0.0


class GasketDiscordNotifier:
    """
    Discord webhook notifier with reaction-based trade confirmation.

    Usage:
        notifier = GasketDiscordNotifier()
        recommendations = [TradeRecommendation(...), ...]
        result = notifier.send_and_confirm(recommendations)
        if result.status == ConfirmationStatus.CONFIRMED:
            # execute trades
    """

    DISCORD_API = "https://discord.com/api/v10"

    def __init__(
        self,
        webhook_url: str = "",
        bot_token: str = "",
        channel_id: str = "",
    ):
        self._webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        self._bot_token = bot_token or os.environ.get("DISCORD_BOT_TOKEN", "")
        self._channel_id = channel_id or os.environ.get("GASKET_CHANNEL_ID", "")
        self._last_message_id: Optional[str] = None

    @property
    def configured(self) -> bool:
        """Check if all required Discord config is present."""
        return bool(self._webhook_url and self._bot_token and self._channel_id)

    def health_check(self) -> Dict[str, Any]:
        """Return health status for the notifier."""
        return {
            "component": "discord_notifier",
            "configured": self.configured,
            "webhook_set": bool(self._webhook_url),
            "bot_token_set": bool(self._bot_token),
            "channel_id_set": bool(self._channel_id),
            "last_message_id": self._last_message_id,
        }

    # ── Send & Confirm Flow ─────────────────────────────────────────────

    def send_and_confirm(
        self,
        recommendations: List[TradeRecommendation],
        timeout_minutes: int = 0,
        pipeline_summary: str = "",
    ) -> ConfirmationResult:
        """
        Post trade recommendations and wait for confirmation reaction.

        Args:
            recommendations: List of trade recommendations to display
            timeout_minutes: Override default timeout (0 = use env default)
            pipeline_summary: Optional summary text from the pipeline

        Returns:
            ConfirmationResult with status and metadata
        """
        if not self.configured:
            logger.error("Discord notifier not configured — missing keys")
            return ConfirmationResult(status=ConfirmationStatus.ERROR)

        if not recommendations:
            logger.info("No trade recommendations to send")
            return ConfirmationResult(status=ConfirmationStatus.ERROR)

        timeout = timeout_minutes or CONFIRM_TIMEOUT_MINUTES

        # Build and send the embed
        embed = self._build_embed(recommendations, pipeline_summary, timeout)
        message_id = self._send_webhook(embed)

        if not message_id:
            logger.error("Failed to send Discord message")
            return ConfirmationResult(status=ConfirmationStatus.ERROR)

        self._last_message_id = message_id
        logger.info("Trade plan posted to Discord (msg=%s)", message_id)

        # Add reaction buttons
        self._add_reaction(message_id, "\u2705")  # ✅
        self._add_reaction(message_id, "\u274c")  # ❌

        # Poll for confirmation
        return self._poll_reactions(message_id, timeout)

    def send_execution_result(
        self,
        summary_text: str,
        success: bool = True,
    ) -> Optional[str]:
        """Post execution results back to Discord after trades complete."""
        if not self._webhook_url:
            return None

        color = 0x00FF00 if success else 0xFF0000
        title = "✅ Trades Executed" if success else "❌ Execution Failed"

        embed = {
            "embeds": [{
                "title": title,
                "description": summary_text[:4000],
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "AAC Options Intelligence"},
            }]
        }
        return self._send_webhook(embed)

    def send_status_update(self, text: str) -> Optional[str]:
        """Send a plain status message to Discord."""
        if not self._webhook_url:
            return None
        payload = {
            "content": text[:2000],
            "username": "AAC Options Intelligence",
        }
        return self._send_webhook(payload)

    # ── Embed Builder ───────────────────────────────────────────────────

    def _build_embed(
        self,
        recommendations: List[TradeRecommendation],
        summary: str,
        timeout_minutes: int,
    ) -> Dict[str, Any]:
        """Build a rich Discord embed with trade recommendations."""
        now = datetime.utcnow()

        # Build the order table
        lines = []
        total_premium = 0.0
        for i, rec in enumerate(recommendations, 1):
            cost = rec.contracts * rec.max_price * 100
            total_premium += cost
            score_bar = self._score_bar(rec.score)
            lines.append(
                f"**{i}. {rec.symbol}** ${rec.strike:.0f}P {rec.expiry}\n"
                f"   {rec.contracts}x @ ${rec.max_price:.2f} "
                f"(${cost:.0f}) | Score: {rec.score} {score_bar}"
            )
            if rec.crisis_vectors:
                lines.append(f"   Vectors: {', '.join(rec.crisis_vectors[:3])}")

        order_text = "\n".join(lines)

        description_parts = []
        if summary:
            description_parts.append(summary)
        description_parts.append(f"**{len(recommendations)} trade(s) | "
                                 f"Total premium: ${total_premium:,.0f}**")
        description_parts.append("")
        description_parts.append(order_text)
        description_parts.append("")
        description_parts.append(
            f"React ✅ to **CONFIRM** or ❌ to **CANCEL**\n"
            f"Auto-cancels in {timeout_minutes} minutes if no response."
        )

        return {
            "username": "AAC Options Intelligence",
            "embeds": [{
                "title": "🔔 Pre-Market Trade Plan — Confirmation Required",
                "description": "\n".join(description_parts)[:4000],
                "color": 0xFFAA00,  # Amber = pending
                "timestamp": now.isoformat(),
                "footer": {
                    "text": (
                        f"AAC Options Intelligence | "
                        f"{now.strftime('%A %B %d, %Y')} | "
                        f"Timeout: {timeout_minutes}min"
                    ),
                },
                "fields": [
                    {
                        "name": "Total Premium",
                        "value": f"${total_premium:,.0f}",
                        "inline": True,
                    },
                    {
                        "name": "Orders",
                        "value": str(len(recommendations)),
                        "inline": True,
                    },
                    {
                        "name": "Deadline",
                        "value": f"9:30 AM ET (market open)",
                        "inline": True,
                    },
                ],
            }],
        }

    @staticmethod
    def _score_bar(score: int) -> str:
        filled = score // 10
        return "🟩" * filled + "⬜" * (10 - filled)

    # ── Discord API Calls ───────────────────────────────────────────────

    def _send_webhook(self, payload: Dict[str, Any]) -> Optional[str]:
        """Send a message via Discord webhook. Returns message ID."""
        url = self._webhook_url
        if "?" in url:
            url += "&wait=true"
        else:
            url += "?wait=true"

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "AAC-OptionsIntelligence/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("id")
        except Exception as e:
            logger.error("Discord webhook send failed: %s", e)
            return None

    def _add_reaction(self, message_id: str, emoji: str) -> bool:
        """Add a reaction to a message via Discord bot API."""
        encoded = urllib.request.quote(emoji)
        url = (
            f"{self.DISCORD_API}/channels/{self._channel_id}"
            f"/messages/{message_id}/reactions/{encoded}/@me"
        )
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bot {self._bot_token}",
                    "Content-Length": "0",
                    "User-Agent": "AAC-OptionsIntelligence/1.0",
                },
                method="PUT",
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            logger.warning("Failed to add reaction %s: %s", emoji, e)
            return False

    def _get_reactions(self, message_id: str, emoji: str) -> List[Dict[str, Any]]:
        """Get users who reacted with a specific emoji."""
        encoded = urllib.request.quote(emoji)
        url = (
            f"{self.DISCORD_API}/channels/{self._channel_id}"
            f"/messages/{message_id}/reactions/{encoded}"
        )
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bot {self._bot_token}",
                    "User-Agent": "AAC-OptionsIntelligence/1.0",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug("Failed to fetch reactions: %s", e)
            return []

    def _poll_reactions(
        self,
        message_id: str,
        timeout_minutes: int,
    ) -> ConfirmationResult:
        """Poll for user reaction on the confirmation message."""
        deadline = time.monotonic() + (timeout_minutes * 60)
        start = time.monotonic()

        logger.info(
            "Waiting for Discord confirmation (timeout=%d min)...",
            timeout_minutes,
        )

        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_SECONDS)

            # Check ✅ reactions (exclude bot's own)
            confirms = self._get_reactions(message_id, "\u2705")
            human_confirms = [u for u in confirms if not u.get("bot", False)]
            if human_confirms:
                elapsed = time.monotonic() - start
                user = human_confirms[0].get("username", "unknown")
                logger.info(
                    "Trade plan CONFIRMED by %s (%.0fs)", user, elapsed
                )
                self._update_message_status(message_id, "CONFIRMED", user)
                return ConfirmationResult(
                    status=ConfirmationStatus.CONFIRMED,
                    message_id=message_id,
                    confirmed_by=user,
                    elapsed_seconds=elapsed,
                )

            # Check ❌ reactions
            rejects = self._get_reactions(message_id, "\u274c")
            human_rejects = [u for u in rejects if not u.get("bot", False)]
            if human_rejects:
                elapsed = time.monotonic() - start
                user = human_rejects[0].get("username", "unknown")
                logger.info(
                    "Trade plan REJECTED by %s (%.0fs)", user, elapsed
                )
                self._update_message_status(message_id, "REJECTED", user)
                return ConfirmationResult(
                    status=ConfirmationStatus.REJECTED,
                    message_id=message_id,
                    confirmed_by=user,
                    elapsed_seconds=elapsed,
                )

        # Timeout
        elapsed = time.monotonic() - start
        logger.warning("Trade confirmation timed out after %d min", timeout_minutes)
        self._update_message_status(message_id, "TIMED OUT")
        return ConfirmationResult(
            status=ConfirmationStatus.TIMED_OUT,
            message_id=message_id,
            elapsed_seconds=elapsed,
        )

    def _update_message_status(
        self,
        message_id: str,
        status: str,
        user: str = "",
    ) -> None:
        """Update the original message to reflect confirmation status."""
        colors = {
            "CONFIRMED": 0x00FF00,
            "REJECTED": 0xFF0000,
            "TIMED OUT": 0x808080,
        }
        icons = {
            "CONFIRMED": "✅",
            "REJECTED": "❌",
            "TIMED OUT": "⏰",
        }

        # Post a follow-up via webhook (can't edit webhook messages without message ID)
        payload = {
            "content": (
                f"{icons.get(status, '❓')} **Trade plan {status}**"
                + (f" by {user}" if user else "")
                + f" at {datetime.utcnow().strftime('%H:%M:%S UTC')}"
            ),
            "username": "AAC Options Intelligence",
        }
        self._send_webhook(payload)

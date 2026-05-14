from __future__ import annotations

"""shared/alerter.py — Sprint 21: Real-Time Alerter.

Provides a single :class:`Alerter` that sends event notifications to Telegram.

Design goals
------------
* **Fails-open** — an unconfigured or broken alerter never blocks trading.
  Every method returns ``bool`` and swallows all exceptions.
* **Sync API** — callers on the trading hot-path don't need to manage an event
  loop.  ``send()`` wraps the async Telegram HTTP call with ``asyncio.run()``.
* **Windows DNS fix** — uses ``aiohttp.resolver.ThreadedResolver()`` with a
  ``TCPConnector`` so c-ares DNS failures don't silently drop messages.
* **Zero new hard dependencies** — only ``aiohttp`` (already installed).

Usage::

    from shared.alerter import Alerter

    alerter = Alerter()                     # reads TELEGRAM_BOT_TOKEN / CHAT_ID from .env
    alerter.send("DRAWDOWN_TRIPPED",        # event_type
                 "Drawdown 11.2% — halted") # human message

    # Or inject pre-configured:
    alerter = Alerter(bot_token="...", chat_id="123456")
"""

import asyncio
import os

import structlog

_log = structlog.get_logger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_ENV_TOKEN = "TELEGRAM_BOT_TOKEN"
_ENV_CHAT_ID = "TELEGRAM_CHAT_ID"

# ── message formatting ─────────────────────────────────────────────────────

_EVENT_ICONS: dict[str, str] = {
    "DRAWDOWN_TRIPPED": "🔴",
    "DAILY_LOSS_TRIPPED": "🔴",
    "STOP_CLOSE": "🛑",
    "TAKE_PROFIT": "✅",
    "ROLL_CLOSE": "🔄",
    "EOD_BRIEF": "📊",
    "SYSTEM": "⚙️",
}


def format_alert(event_type: str, message: str) -> str:
    """Return a Telegram-ready message string for the given event.

    Format::

        🔴 DRAWDOWN_TRIPPED
        Drawdown 11.2% — halted

    HTML entities are escaped so the message can be sent as plain text
    (``parse_mode`` defaults to an empty string to avoid accidental HTML).
    """
    icon = _EVENT_ICONS.get(event_type, "📢")
    return f"{icon} <b>{event_type}</b>\n{message}"


# ── Alerter ────────────────────────────────────────────────────────────────


class Alerter:
    """Send real-time notifications to Telegram.

    Parameters
    ----------
    bot_token:
        Telegram bot token.  Defaults to ``TELEGRAM_BOT_TOKEN`` env var.
    chat_id:
        Telegram chat / user ID.  Defaults to ``TELEGRAM_CHAT_ID`` env var.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        # Use explicit arg when not None; fall back to env only when arg is None.
        self._bot_token: str = (
            bot_token if bot_token is not None else (os.environ.get(_ENV_TOKEN) or "")
        )
        self._chat_id: str = (
            chat_id if chat_id is not None else (os.environ.get(_ENV_CHAT_ID) or "")
        )
        self._enabled: bool = bool(self._bot_token and self._chat_id)

        if not self._enabled:
            _log.debug("alerter_disabled_missing_config",
                       has_token=bool(self._bot_token), has_chat=bool(self._chat_id))

    # ── public API ────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True when both bot token and chat ID are configured."""
        return self._enabled

    def send(self, event_type: str, message: str) -> bool:
        """Send a Telegram notification.  Never raises.

        Args:
            event_type: Short label for the event (``"DRAWDOWN_TRIPPED"`` etc.).
            message:    Human-readable detail text.

        Returns:
            ``True`` on confirmed delivery; ``False`` on any failure or when
            the alerter is unconfigured.
        """
        if not self._enabled:
            _log.debug("alerter_send_skipped_disabled", event_type=event_type)
            return False
        try:
            return asyncio.run(self._send_async(event_type, message))
        except Exception as exc:
            _log.warning("alerter_send_failed", event_type=event_type, error=str(exc))
            return False

    # ── internal ──────────────────────────────────────────────────────────

    async def _send_async(self, event_type: str, message: str) -> bool:
        """Async Telegram HTTP POST with Windows-safe DNS resolver."""
        try:
            import aiohttp  # noqa: PLC0415
            import aiohttp.resolver  # noqa: PLC0415
        except ImportError:
            _log.warning("alerter_aiohttp_unavailable")
            return False

        text = format_alert(event_type, message)
        url = _TELEGRAM_API.format(token=self._bot_token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        try:
            resolver = aiohttp.resolver.ThreadedResolver()
            connector = aiohttp.TCPConnector(resolver=resolver)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        _log.info("alerter_sent", event_type=event_type, status=resp.status)
                        return True
                    body = await resp.text()
                    _log.warning("alerter_api_error", event_type=event_type,
                                 status=resp.status, body=body[:200])
                    return False
        except Exception as exc:
            _log.warning("alerter_http_failed", event_type=event_type, error=str(exc))
            return False

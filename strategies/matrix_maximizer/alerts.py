"""
MATRIX MAXIMIZER — Alert System
==================================
Multi-channel alerting:
  - Telegram (via @barrenwuffet069bot)
  - Email (SMTP)
  - Watchdog (circuit breaker changes, mandate shifts, new picks)
  - Alert dedup + throttling
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    TRADE = "trade"


class AlertChannel(Enum):
    TELEGRAM = "telegram"
    EMAIL = "email"
    LOG = "log"


@dataclass
class Alert:
    """Single alert message."""
    level: AlertLevel
    title: str
    body: str
    channels: List[AlertChannel]
    timestamp: str = ""
    ticker: str = ""
    dedup_key: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.dedup_key:
            self.dedup_key = f"{self.title}_{self.ticker}"


class AlertManager:
    """Central alert hub for MATRIX MAXIMIZER.

    Features:
        - Multi-channel dispatch (Telegram, Email, log)
        - Rate limiting (max N alerts per channel per hour)
        - Deduplication (same alert not sent twice within window)
        - Batch mode (collect alerts, send summary)

    Usage:
        mgr = AlertManager()
        mgr.send(Alert(AlertLevel.TRADE, "New Pick", "BUY SPY $540P", [AlertChannel.TELEGRAM]))
        mgr.send_trade_alert("SPY", 540, "2026-04-18", 3.50, 2)
        mgr.send_circuit_breaker_alert("GREEN", "YELLOW", "VIX spike to 28")
    """

    MAX_PER_HOUR = 20  # Max alerts per channel per hour

    def __init__(self, enable_telegram: bool = True,
                 enable_email: bool = True) -> None:
        self._telegram_enabled = enable_telegram
        self._email_enabled = enable_email
        self._sent: Dict[str, datetime] = {}    # dedup_key → last sent
        self._counts: Dict[str, int] = defaultdict(int)  # channel → count this hour
        self._hour_start = datetime.utcnow()
        self._batch: List[Alert] = []
        self._lock = threading.Lock()

    def send(self, alert: Alert) -> bool:
        """Send an alert through configured channels."""
        with self._lock:
            # Rate limit reset
            now = datetime.utcnow()
            if (now - self._hour_start).total_seconds() > 3600:
                self._counts.clear()
                self._hour_start = now

            # Dedup check (30 min window)
            if alert.dedup_key in self._sent:
                last = self._sent[alert.dedup_key]
                if (now - last).total_seconds() < 1800:
                    logger.debug("Alert deduped: %s", alert.dedup_key)
                    return False

            self._sent[alert.dedup_key] = now
            sent = False

            for channel in alert.channels:
                if self._counts[channel.value] >= self.MAX_PER_HOUR:
                    logger.warning("Rate limit hit for %s", channel.value)
                    continue

                if channel == AlertChannel.TELEGRAM:
                    sent = self._send_telegram(alert) or sent
                elif channel == AlertChannel.EMAIL:
                    sent = self._send_email(alert) or sent
                elif channel == AlertChannel.LOG:
                    logger.info("[ALERT:%s] %s — %s", alert.level.value, alert.title, alert.body)
                    sent = True

                if sent:
                    self._counts[channel.value] += 1

            return sent

    # ═══════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def send_trade_alert(self, ticker: str, strike: float, expiry: str,
                         premium: float, contracts: int,
                         action: str = "BUY") -> bool:
        """Send a trade execution alert."""
        cost = contracts * premium * 100
        body = (
            f"{'🟢' if action == 'BUY' else '🔴'} {action} {contracts}x "
            f"{ticker} ${strike:.0f}P {expiry}\n"
            f"Premium: ${premium:.2f} | Total: ${cost:.0f}"
        )
        return self.send(Alert(
            level=AlertLevel.TRADE,
            title=f"MM {action}: {ticker} ${strike:.0f}P",
            body=body,
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            ticker=ticker,
        ))

    def send_circuit_breaker_alert(self, old_state: str, new_state: str,
                                   reason: str = "") -> bool:
        """Alert on circuit breaker state change."""
        body = f"Circuit breaker: {old_state} → {new_state}"
        if reason:
            body += f"\nReason: {reason}"

        level = AlertLevel.WARNING
        if new_state in ("RED", "BLACK"):
            level = AlertLevel.CRITICAL

        return self.send(Alert(
            level=level,
            title=f"⚡ Circuit Breaker → {new_state}",
            body=body,
            channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL, AlertChannel.LOG],
            dedup_key=f"cb_{new_state}",
        ))

    def send_mandate_alert(self, mandate: str, conviction: float) -> bool:
        """Alert on mandate change."""
        return self.send(Alert(
            level=AlertLevel.WARNING,
            title=f"📋 Mandate: {mandate}",
            body=f"System mandate changed to {mandate} (conviction: {conviction:.0%})",
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            dedup_key=f"mandate_{mandate}",
        ))

    def send_roll_alert(self, ticker: str, action: str, reason: str) -> bool:
        """Alert on roll recommendation."""
        return self.send(Alert(
            level=AlertLevel.INFO,
            title=f"🔄 Roll: {ticker} — {action}",
            body=f"Roll signal for {ticker}: {action}\n{reason}",
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            ticker=ticker,
        ))

    def send_risk_alert(self, risk_flag: str, details: str) -> bool:
        """Alert on risk threshold breach."""
        return self.send(Alert(
            level=AlertLevel.CRITICAL,
            title=f"🚨 Risk: {risk_flag}",
            body=details,
            channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL, AlertChannel.LOG],
            dedup_key=f"risk_{risk_flag}",
        ))

    def send_pnl_alert(self, unrealized: float, realized: float,
                       positions: int) -> bool:
        """Daily P&L summary."""
        return self.send(Alert(
            level=AlertLevel.INFO,
            title="💰 Daily P&L",
            body=(
                f"Unrealized: ${unrealized:+.2f}\n"
                f"Realized: ${realized:+.2f}\n"
                f"Positions: {positions}"
            ),
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            dedup_key="daily_pnl",
        ))

    def send_morning_briefing(self, cycle_summary: str) -> bool:
        """Send morning briefing from cycle output."""
        return self.send(Alert(
            level=AlertLevel.INFO,
            title="🌅 Matrix Maximizer Morning Brief",
            body=cycle_summary[:4000],  # Telegram max ~4096
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            dedup_key="morning_brief",
        ))

    # ═══════════════════════════════════════════════════════════════════════
    # BATCH MODE
    # ═══════════════════════════════════════════════════════════════════════

    def batch_add(self, alert: Alert) -> None:
        """Add alert to batch (sent later with flush_batch)."""
        self._batch.append(alert)

    def flush_batch(self) -> int:
        """Send all batched alerts as a single summary."""
        if not self._batch:
            return 0

        lines = []
        for a in self._batch:
            lines.append(f"[{a.level.value.upper()}] {a.title}\n{a.body}")

        combined = "\n\n---\n\n".join(lines)
        sent = self.send(Alert(
            level=AlertLevel.INFO,
            title=f"📦 Batch Alert ({len(self._batch)} items)",
            body=combined[:4000],
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            dedup_key=f"batch_{datetime.utcnow().strftime('%Y%m%d_%H')}",
        ))

        count = len(self._batch)
        self._batch.clear()
        return count if sent else 0

    # ═══════════════════════════════════════════════════════════════════════
    # CHANNEL IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════

    def _send_telegram(self, alert: Alert) -> bool:
        """Send via Telegram bot."""
        if not self._telegram_enabled:
            return False

        try:
            from integrations.barren_wuffet_telegram_bot import send_alert
            send_alert(f"*{alert.title}*\n\n{alert.body}")
            logger.info("Telegram alert sent: %s", alert.title)
            return True
        except ImportError:
            logger.debug("Telegram bot not available")
            return False
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    def _send_email(self, alert: Alert) -> bool:
        """Send via SMTP email."""
        if not self._email_enabled:
            return False

        host = os.getenv("SMTP_HOST", "")
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASSWORD", "")
        to_addr = os.getenv("ALERT_EMAIL_TO", "")

        if not all([host, user, password, to_addr]):
            logger.debug("Email not configured — skipping")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = user
            msg["To"] = to_addr
            msg["Subject"] = f"[MATRIX MAXIMIZER] {alert.title}"
            msg.attach(MIMEText(alert.body, "plain"))

            with smtplib.SMTP(host, port, timeout=10) as server:
                server.starttls()
                server.login(user, password)
                server.send_message(msg)

            logger.info("Email alert sent: %s", alert.title)
            return True
        except Exception as exc:
            logger.warning("Email send failed: %s", exc)
            return False


class Watchdog:
    """Monitors system state and triggers alerts on changes.

    Tracks:
      - Circuit breaker transitions
      - Mandate changes
      - New picks in scanner output
      - Position P&L thresholds
      - Risk layer breaches

    Usage:
        dog = Watchdog(alert_manager)
        dog.check_circuit_breaker(old_state, new_state)
        dog.check_pnl(positions)
    """

    def __init__(self, alerts: AlertManager) -> None:
        self.alerts = alerts
        self._last_cb_state = "GREEN"
        self._last_mandate = ""
        self._known_picks: set = set()
        self._pnl_alert_thresholds = [-100, -200, -500]  # Alert at these P&L levels
        self._pnl_alerts_sent: set = set()

    def check_circuit_breaker(self, new_state: str, reason: str = "") -> None:
        """Check if circuit breaker changed."""
        if new_state != self._last_cb_state:
            self.alerts.send_circuit_breaker_alert(self._last_cb_state, new_state, reason)
            self._last_cb_state = new_state

    def check_mandate(self, mandate: str, conviction: float) -> None:
        """Check if mandate changed."""
        if mandate != self._last_mandate:
            self.alerts.send_mandate_alert(mandate, conviction)
            self._last_mandate = mandate

    def check_new_picks(self, picks: List[Dict[str, Any]]) -> None:
        """Check for new scanner picks."""
        for pick in picks:
            key = f"{pick.get('ticker', '')}_{pick.get('strike', 0)}"
            if key not in self._known_picks:
                self._known_picks.add(key)
                self.alerts.send_trade_alert(
                    pick.get("ticker", ""),
                    pick.get("strike", 0),
                    pick.get("expiry", ""),
                    pick.get("premium", 0),
                    1,
                    "PICK",
                )

    def check_pnl(self, unrealized: float) -> None:
        """Check if P&L crossed alert thresholds."""
        for threshold in self._pnl_alert_thresholds:
            key = f"pnl_{threshold}"
            if unrealized <= threshold and key not in self._pnl_alerts_sent:
                self._pnl_alerts_sent.add(key)
                self.alerts.send_risk_alert(
                    f"P&L below ${threshold}",
                    f"Unrealized P&L: ${unrealized:.2f} — crossed ${threshold} threshold",
                )

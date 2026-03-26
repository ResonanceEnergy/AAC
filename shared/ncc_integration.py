#!/usr/bin/env python3
from __future__ import annotations

"""
NCC Integration — AAC ↔ NCC Command & Control Bridge
=====================================================
Connects AAC (BANK pillar) to NCC governance layer.

Features:
    1. HEARTBEAT publisher   — 60s interval, proves AAC is alive
    2. STATUS_REPORT publisher — 5min interval, full system snapshot
    3. COMMAND consumer       — accepts HALT / SAFE_MODE from NCC
    4. Portfolio & trade event publishing to relay
    5. Outbox flush for offline resilience

Architecture:
    AAC ──relay_client──► NCC Relay (8787) ──► NCC Supreme Monitor
    AAC ◄──poll_commands── NCC Command API (8765)

Usage:
    from shared.ncc_integration import get_ncc_bridge, NCCCommand

    bridge = get_ncc_bridge()
    bridge.start()           # begin heartbeat + status loops
    bridge.publish_portfolio_snapshot()
    bridge.stop()

CLI:
    python -m shared.ncc_integration --status    # Show bridge status
    python -m shared.ncc_integration --snapshot  # Publish portfolio once
    python -m shared.ncc_integration --start     # Run heartbeat daemon
"""

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.ncc_relay_client import NCCRelayClient, get_relay_client  # noqa: E402

# ── Constants ───────────────────────────────────────────────────

NCC_COMMAND_URL = os.environ.get("NCC_COMMAND_URL", "http://127.0.0.1:8765")
HEARTBEAT_INTERVAL = int(os.environ.get("NCC_HEARTBEAT_INTERVAL", "60"))
STATUS_INTERVAL = int(os.environ.get("NCC_STATUS_INTERVAL", "300"))


class NCCCommand(Enum):
    """Commands that NCC can issue to AAC."""

    HALT = "HALT"
    SAFE_MODE = "SAFE_MODE"
    RESUME = "RESUME"
    STATUS_REQUEST = "STATUS_REQUEST"


class PillarStatus(Enum):
    """AAC operational status reported to NCC."""

    ONLINE = "online"
    DEGRADED = "degraded"
    SAFE_MODE = "safe_mode"
    HALTED = "halted"
    OFFLINE = "offline"
    BOOTSTRAPPING = "bootstrapping"


# ── Main Bridge ─────────────────────────────────────────────────


class NCC_AAC_Bridge:
    """Full integration bridge between AAC and NCC governance."""

    def __init__(
        self,
        relay_client: NCCRelayClient | None = None,
        command_url: str = NCC_COMMAND_URL,
    ):
        self.relay = relay_client or get_relay_client()
        self.command_url = command_url.rstrip("/")

        # State
        self._status = PillarStatus.BOOTSTRAPPING
        self._halted = False
        self._safe_mode = False
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._status_thread: threading.Thread | None = None
        self._command_thread: threading.Thread | None = None
        self._last_heartbeat: float = 0.0
        self._last_status_report: float = 0.0
        self._commands_received: list[dict] = []

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Start heartbeat, status report, and command polling loops."""
        if self._running:
            return
        self._running = True
        self._status = PillarStatus.ONLINE
        logger.info("NCC-AAC bridge starting")

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="ncc-heartbeat"
        )
        self._heartbeat_thread.start()

        self._status_thread = threading.Thread(
            target=self._status_loop, daemon=True, name="ncc-status"
        )
        self._status_thread.start()

        self._command_thread = threading.Thread(
            target=self._command_poll_loop, daemon=True, name="ncc-command"
        )
        self._command_thread.start()

    def stop(self) -> None:
        """Stop all background loops."""
        self._running = False
        self._status = PillarStatus.OFFLINE
        # Send one last heartbeat marking offline
        self._publish_heartbeat()
        logger.info("NCC-AAC bridge stopped")

    # ── Heartbeat (60s) ────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        while self._running:
            self._publish_heartbeat()
            time.sleep(HEARTBEAT_INTERVAL)

    def _publish_heartbeat(self) -> bool:
        data = {
            "pillar_id": "aac",
            "status": self._status.value,
            "halted": self._halted,
            "safe_mode": self._safe_mode,
            "uptime_seconds": round(time.monotonic(), 1),
            "relay_stats": self.relay.stats,
        }
        ok = self.relay.publish("ncl.sync.v1.bank.heartbeat", data)
        if ok:
            self._last_heartbeat = time.monotonic()
        return ok

    # ── Status Report (5min) ────────────────────────────────────

    def _status_loop(self) -> None:
        while self._running:
            self._publish_status_report()
            time.sleep(STATUS_INTERVAL)

    def _publish_status_report(self) -> bool:
        report = self._collect_status()
        ok = self.relay.publish("ncl.sync.v1.bank.status_report", report)
        if ok:
            self._last_status_report = time.monotonic()
        return ok

    def _collect_status(self) -> dict:
        """Collect comprehensive AAC status for NCC."""
        report: dict = {
            "pillar_id": "aac",
            "status": self._status.value,
            "halted": self._halted,
            "safe_mode": self._safe_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Config & environment
        try:
            from shared.config_loader import get_config

            config = get_config()
            validation = config.validate()
            report["config"] = {
                "valid": validation.get("valid", False),
                "exchanges": validation.get("exchanges_configured", []),
                "dry_run": validation.get("dry_run", True),
                "issues": validation.get("issues", []),
            }
        except Exception as exc:
            report["config"] = {"error": str(exc)}

        # Portfolio state
        account = self._read_paper_account()
        if account:
            report["portfolio"] = {
                "account_id": account.get("account_id", "paper_1"),
                "balance": account.get("balance", 0),
                "equity": account.get("equity", 0),
                "daily_pnl": account.get("daily_pnl", 0),
                "total_pnl": account.get("total_pnl", 0),
                "position_count": len(account.get("positions", [])),
                "open_orders": len(account.get("orders", [])),
            }

        # Relay connectivity
        report["relay"] = self.relay.stats

        # Command history
        report["commands_received"] = len(self._commands_received)

        return report

    # ── Portfolio Publishing ────────────────────────────────────

    def publish_portfolio_snapshot(self) -> bool:
        """Read AAC account state, publish to NCC relay."""
        account = self._read_paper_account()
        if not account:
            return False
        positions = account.get("positions", [])
        return self.relay.publish(
            "ncl.sync.v1.bank.portfolio_snapshot",
            {
                "account_id": account.get("account_id", "paper_1"),
                "balance": account.get("balance", 0),
                "equity": account.get("equity", 0),
                "daily_pnl": account.get("daily_pnl", 0),
                "total_pnl": account.get("total_pnl", 0),
                "margin_used": account.get("margin_used", 0),
                "margin_available": account.get("margin_available", 0),
                "position_count": len(positions) if isinstance(positions, list) else 0,
                "open_orders": len(account.get("orders", [])),
                "updated_at": account.get("updated_at", ""),
            },
        )

    def publish_trade_event(self, trade: dict) -> bool:
        """Publish a trade execution event to NCC relay."""
        return self.relay.publish(
            "ncl.sync.v1.bank.trade_executed",
            {
                "order_id": trade.get("order_id", ""),
                "symbol": trade.get("symbol", ""),
                "side": trade.get("side", ""),
                "quantity": trade.get("quantity", 0),
                "price": trade.get("price", 0),
                "status": trade.get("status", ""),
                "exchange": trade.get("exchange", ""),
                "timestamp": trade.get("timestamp", ""),
            },
        )

    # ── COMMAND Consumer ────────────────────────────────────────

    def _command_poll_loop(self) -> None:
        """Poll NCC Command API for pending commands addressed to AAC."""
        while self._running:
            try:
                self._poll_commands()
            except Exception:
                logger.debug("Command poll failed (NCC Command API unreachable)")
            time.sleep(30)

    def _poll_commands(self) -> None:
        """Fetch pending commands from NCC Command API."""
        try:
            req = urllib.request.Request(
                f"{self.command_url}/commands/aac",
                method="GET",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    data = json.loads(resp.read())
                    commands = data.get("commands", [])
                    for cmd in commands:
                        self._handle_command(cmd)
        except (urllib.error.URLError, OSError, ValueError):
            pass

    def _handle_command(self, cmd: dict) -> None:
        """Execute an NCC command."""
        command_type = cmd.get("command", "").upper()
        logger.info("Received NCC command: %s", command_type)
        self._commands_received.append(
            {
                "command": command_type,
                "received_at": datetime.now(timezone.utc).isoformat(),
                "payload": cmd,
            }
        )

        if command_type == NCCCommand.HALT.value:
            self._execute_halt(cmd)
        elif command_type == NCCCommand.SAFE_MODE.value:
            self._execute_safe_mode(cmd)
        elif command_type == NCCCommand.RESUME.value:
            self._execute_resume(cmd)
        elif command_type == NCCCommand.STATUS_REQUEST.value:
            self._publish_status_report()
        else:
            logger.warning("Unknown NCC command: %s", command_type)

    def _execute_halt(self, cmd: dict) -> None:
        """HALT — stop all trading immediately."""
        self._halted = True
        self._status = PillarStatus.HALTED
        logger.critical("NCC HALT command received — trading halted")

        # Write halt flag to disk for other AAC processes to detect
        halt_file = PROJECT_ROOT / "data" / "ncc_halt.flag"
        halt_file.parent.mkdir(parents=True, exist_ok=True)
        halt_file.write_text(
            json.dumps(
                {
                    "command": "HALT",
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                    "reason": cmd.get("reason", "NCC directive"),
                    "source": "ncc",
                }
            ),
            encoding="utf-8",
        )

        # Notify relay
        self.relay.publish(
            "ncl.sync.v1.bank.command_ack",
            {
                "command": "HALT",
                "status": "executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _execute_safe_mode(self, cmd: dict) -> None:
        """SAFE_MODE — close-only, no new positions."""
        self._safe_mode = True
        self._status = PillarStatus.SAFE_MODE
        logger.warning("NCC SAFE_MODE command received — close-only mode")

        safe_file = PROJECT_ROOT / "data" / "ncc_safe_mode.flag"
        safe_file.parent.mkdir(parents=True, exist_ok=True)
        safe_file.write_text(
            json.dumps(
                {
                    "command": "SAFE_MODE",
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                    "reason": cmd.get("reason", "NCC directive"),
                    "source": "ncc",
                }
            ),
            encoding="utf-8",
        )

        self.relay.publish(
            "ncl.sync.v1.bank.command_ack",
            {
                "command": "SAFE_MODE",
                "status": "executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _execute_resume(self, cmd: dict) -> None:
        """RESUME — return to normal trading operations."""
        self._halted = False
        self._safe_mode = False
        self._status = PillarStatus.ONLINE
        logger.info("NCC RESUME command received — normal operations")

        # Remove flag files
        for flag in ("ncc_halt.flag", "ncc_safe_mode.flag"):
            flag_path = PROJECT_ROOT / "data" / flag
            if flag_path.exists():
                flag_path.unlink()

        self.relay.publish(
            "ncl.sync.v1.bank.command_ack",
            {
                "command": "RESUME",
                "status": "executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ── Data Readers ────────────────────────────────────────────

    def _read_paper_account(self) -> dict | None:
        account_file = PROJECT_ROOT / "data" / "paper_trading" / "paper_account_1.json"
        if account_file.exists():
            try:
                return json.loads(account_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return None

    # ── Status Queries ──────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        """Check if NCC has issued a HALT command."""
        if self._halted:
            return True
        halt_file = PROJECT_ROOT / "data" / "ncc_halt.flag"
        return halt_file.exists()

    @property
    def is_safe_mode(self) -> bool:
        """Check if NCC has issued a SAFE_MODE command."""
        if self._safe_mode:
            return True
        safe_file = PROJECT_ROOT / "data" / "ncc_safe_mode.flag"
        return safe_file.exists()

    @property
    def platform_status(self) -> dict:
        """Full platform status for the /platform_status endpoint."""
        status = self._collect_status()
        status["ncc_relay"] = self.relay.relay_health()
        status["is_halted"] = self.is_halted
        status["is_safe_mode"] = self.is_safe_mode
        status["heartbeat_age_s"] = (
            round(time.monotonic() - self._last_heartbeat, 1)
            if self._last_heartbeat
            else None
        )
        status["status_report_age_s"] = (
            round(time.monotonic() - self._last_status_report, 1)
            if self._last_status_report
            else None
        )
        return status


# ── Module-level singleton ──────────────────────────────────────

_bridge: NCC_AAC_Bridge | None = None
_bridge_lock = threading.Lock()


def get_ncc_bridge() -> NCC_AAC_Bridge:
    """Get or create the module-level NCC bridge singleton."""
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                _bridge = NCC_AAC_Bridge()
    return _bridge


# ── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="AAC ↔ NCC Integration Bridge")
    parser.add_argument("--status", action="store_true", help="Show bridge status")
    parser.add_argument("--snapshot", action="store_true", help="Publish portfolio snapshot")
    parser.add_argument("--start", action="store_true", help="Run heartbeat daemon")
    parser.add_argument("--flush", action="store_true", help="Flush outbox")
    args = parser.parse_args()

    bridge = get_ncc_bridge()

    if args.status:
        import pprint

        pprint.pprint(bridge.platform_status)
    elif args.snapshot:
        ok = bridge.publish_portfolio_snapshot()
        print(f"Portfolio snapshot: {'sent' if ok else 'queued'}")
    elif args.flush:
        result = bridge.relay.flush()
        print(f"Flush: {result}")
    elif args.start:
        bridge.start()
        print("NCC-AAC bridge running (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            bridge.stop()
    else:
        parser.print_help()

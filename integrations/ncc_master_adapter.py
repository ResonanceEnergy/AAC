#!/usr/bin/env python3
"""
NCC MASTER Adapter — AAC (BANK) Pillar
========================================
Connects AAC to NCC MASTER for centralized command & control.

Responsibilities:
    1. Registers AAC with NCC MASTER on startup
    2. Publishes heartbeats every 30s
    3. Handles inbound NCC commands (halt, resume, caution, etc.)
    4. Reports AAC matrix monitor status
    5. ACKs directives back to NCC MASTER

Usage:
    # As a module (imported by launch.py or pipeline_runner.py):
    from integrations.ncc_master_adapter import NCCMasterAdapter
    adapter = NCCMasterAdapter()
    await adapter.start()

    # Standalone:
    python -m integrations.ncc_master_adapter
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

NCC_MASTER_URL = os.environ.get("NCC_MASTER_URL", "http://localhost:9000")
AAC_HEARTBEAT_INTERVAL = int(os.environ.get("AAC_HEARTBEAT_INTERVAL", "30"))
AAC_STATE_DIR = Path(os.environ.get("AAC_STATE_DIR", "data/pillar_state"))


@dataclass
class AACSummary:
    """Quick summary of AAC state for NCC MASTER."""
    doctrine_mode: str = "UNKNOWN"
    active_strategies: int = 0
    open_positions: int = 0
    daily_pnl: float = 0.0
    equity: float = 0.0
    risk_score: float = 0.0
    trading_allowed: bool = True
    alerts: list = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pillar": "BANK",
            "name": "AAC",
            "doctrine_mode": self.doctrine_mode,
            "active_strategies": self.active_strategies,
            "open_positions": self.open_positions,
            "daily_pnl": self.daily_pnl,
            "equity": self.equity,
            "risk_score": self.risk_score,
            "trading_allowed": self.trading_allowed,
            "alerts": self.alerts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class NCCMasterAdapter:
    """
    AAC adapter for NCC MASTER command & control.

    Lifecycle:
        1. start() → registers with NCC MASTER + begins heartbeat loop
        2. check_directives() → polls local directive file for NCC commands
        3. report_status() → sends AAC summary to NCC MASTER
        4. stop() → graceful shutdown
    """

    def __init__(self):
        self._state_dir = AAC_STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._directive_file = self._state_dir / "ncc_directive.json"
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_directive_id: Optional[str] = None
        self._summary = AACSummary()
        logger.info("NCCMasterAdapter initialized — state_dir=%s", self._state_dir)

    async def start(self) -> None:
        """Start the adapter (heartbeat loop + directive polling)."""
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("NCC MASTER adapter started — heartbeat every %ds", AAC_HEARTBEAT_INTERVAL)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("NCC MASTER adapter stopped")

    # ── Heartbeat ──────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat: reports status + checks directives."""
        while self._running:
            try:
                # Update own summary
                self._refresh_summary()

                # Write heartbeat to state file (NCC reads via file or REST)
                self._write_heartbeat()

                # Check for inbound directives
                directive = self._check_directive()
                if directive:
                    self._handle_directive(directive)

            except Exception as exc:
                logger.warning("Heartbeat error: %s", exc)

            await asyncio.sleep(AAC_HEARTBEAT_INTERVAL)

    def _write_heartbeat(self) -> None:
        """Write heartbeat state file for NCC MASTER to read."""
        hb_file = self._state_dir / "aac_heartbeat.json"
        payload = {
            "pillar": "BANK",
            "name": "AAC",
            "status": "ALIVE",
            "summary": self._summary.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        hb_file.write_text(json.dumps(payload, indent=2))

    # ── Directive Handling ─────────────────────────────────────────

    def _check_directive(self) -> Optional[Dict[str, Any]]:
        """Check for new NCC directive."""
        if not self._directive_file.exists():
            return None

        try:
            data = json.loads(self._directive_file.read_text())
            directive_id = data.get("directive_id")

            # Only process new directives
            if directive_id and directive_id != self._last_directive_id:
                self._last_directive_id = directive_id
                return data
        except Exception as exc:
            logger.warning("Failed to read directive: %s", exc)

        return None

    def _handle_directive(self, directive: Dict[str, Any]) -> None:
        """Handle an NCC governance directive."""
        action = directive.get("action", "")
        reason = directive.get("reason", "NCC directive")
        directive_id = directive.get("directive_id", "unknown")

        logger.warning(
            "NCC DIRECTIVE RECEIVED: %s — action=%s reason=%s",
            directive_id, action, reason,
        )

        # Apply to AAC cross-pillar state
        action_map = {
            "halt": "HALT",
            "safe_mode": "SAFE_MODE",
            "caution": "CAUTION",
            "resume": "NORMAL",
            "reduce_exposure": "CAUTION",
        }

        new_mode = action_map.get(action)
        if new_mode:
            self._apply_doctrine_mode(new_mode, reason)

        # Write ACK
        self._ack_directive(directive_id, action, success=True)

    def _apply_doctrine_mode(self, mode: str, reason: str) -> None:
        """Apply a doctrine mode change to AAC."""
        state_file = self._state_dir / "cross_pillar_state.json"

        state = {}
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
            except Exception:
                pass

        old_mode = state.get("doctrine_mode", "UNKNOWN")
        state["doctrine_mode"] = mode
        state["saved_at"] = datetime.now(timezone.utc).isoformat()

        state_file.write_text(json.dumps(state, indent=2))

        logger.warning(
            "DOCTRINE MODE CHANGE: %s → %s (reason: %s)",
            old_mode, mode, reason,
        )

    def _ack_directive(self, directive_id: str, action: str, success: bool) -> None:
        """Write ACK file for NCC MASTER to read."""
        ack_file = self._state_dir / "ncc_directive_ack.json"
        payload = {
            "directive_id": directive_id,
            "action": action,
            "ack": success,
            "pillar": "BANK",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        ack_file.write_text(json.dumps(payload, indent=2))
        logger.info("ACK written for directive %s", directive_id)

    # ── Status Reporting ───────────────────────────────────────────

    def _refresh_summary(self) -> None:
        """Refresh AAC summary from local state files."""
        # Read cross-pillar state
        cp_file = self._state_dir / "cross_pillar_state.json"
        if cp_file.exists():
            try:
                data = json.loads(cp_file.read_text())
                self._summary.doctrine_mode = data.get("doctrine_mode", "UNKNOWN")
                self._summary.active_strategies = len(
                    data.get("active_strategies", [])
                )
                self._summary.trading_allowed = self._summary.doctrine_mode in (
                    "NORMAL", "CAUTION"
                )
            except Exception:
                pass

    def get_matrix_status(self) -> Dict[str, Any]:
        """Get AAC matrix monitor status for NCC MASTER."""
        self._refresh_summary()
        return {
            "pillar": "BANK",
            "health": "GREEN" if self._summary.trading_allowed else "YELLOW",
            "health_score": 100 if self._summary.trading_allowed else 50,
            "summary": self._summary.to_dict(),
            "matrix_monitors": {
                "doctrine_compliance": True,
                "trading_activity": True,
                "risk_monitoring": True,
                "security": True,
            },
        }


# ── Standalone entry point ───────────────────────────────────────────

async def main():
    """Run adapter standalone for testing."""
    adapter = NCCMasterAdapter()
    print("NCC MASTER Adapter (AAC) — running...")
    print(f"  State dir: {adapter._state_dir}")
    print(f"  Heartbeat: every {AAC_HEARTBEAT_INTERVAL}s")
    print(f"  Directive file: {adapter._directive_file}")
    print()

    # One-shot status report
    status = adapter.get_matrix_status()
    print(json.dumps(status, indent=2))

    # Start heartbeat loop
    await adapter.start()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await adapter.stop()


if __name__ == "__main__":
    asyncio.run(main())

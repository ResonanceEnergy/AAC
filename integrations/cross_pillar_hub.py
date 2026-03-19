#!/usr/bin/env python3
"""
Cross-Pillar Integration Hub
==============================
Connects AAC (BANK) with NCC (Command), NCL (BRAIN), and BRS (Bravo).

Integration Points:
    - NCC: Governance directives, doctrine state, halt/resume commands
    - NCL: Market intelligence, forecasting data, strategy recommendations
    - BRS: Trading education signals, pattern recognition, methodology scoring

Communication: HTTP REST + file-based state sharing for local dev.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PillarStatus:
    """Health status of a connected pillar."""
    name: str
    connected: bool = False
    last_heartbeat: Optional[str] = None
    version: str = "unknown"
    mode: str = "offline"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "connected": self.connected,
            "last_heartbeat": self.last_heartbeat,
            "version": self.version,
            "mode": self.mode,
            "error": self.error,
        }


@dataclass
class GovernanceDirective:
    """A directive from NCC governance."""
    directive_id: str
    action: str  # halt, resume, reduce_exposure, increase_caution
    reason: str
    timestamp: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossPillarState:
    """Shared state across all pillars."""
    doctrine_mode: str = "NORMAL"  # NORMAL, CAUTION, SAFE_MODE, HALT
    ncc: PillarStatus = field(default_factory=lambda: PillarStatus("NCC"))
    ncl: PillarStatus = field(default_factory=lambda: PillarStatus("NCL"))
    brs: PillarStatus = field(default_factory=lambda: PillarStatus("BRS"))
    last_directive: Optional[GovernanceDirective] = None
    active_strategies: List[str] = field(default_factory=list)


class CrossPillarHub:
    """
    Central integration hub for cross-pillar communication.

    Modes:
        - local: File-based state sharing (development)
        - rest: HTTP REST API calls (production)
    """

    def __init__(self):
        self.state = CrossPillarState()
        self._state_dir = Path(os.environ.get("AAC_STATE_DIR", "data/pillar_state"))
        self._state_dir.mkdir(parents=True, exist_ok=True)

        # NCC config
        self._ncc_endpoint = os.environ.get("NCC_COORDINATOR_ENDPOINT", "http://localhost:8000/api/v1")
        self._ncc_token = os.environ.get("NCC_AUTH_TOKEN", "")
        self._ncc_relay = os.environ.get("NCC_RELAY_URL", "http://127.0.0.1:8787")

        # NCL config
        self._ncl_path = Path(os.environ.get("NCL_DATA_PATH", ""))

        # Load persisted state
        self._load_state()
        logger.info("CrossPillarHub initialized — doctrine_mode=%s", self.state.doctrine_mode)

    # ── NCC Integration ────────────────────────────────────────────────

    async def check_ncc_governance(self) -> Dict[str, Any]:
        """Check NCC for governance directives."""
        # Try local state file first (works without NCC server running)
        directive_file = self._state_dir / "ncc_directive.json"
        if directive_file.exists():
            try:
                data = json.loads(directive_file.read_text())
                directive = GovernanceDirective(**data)
                self._apply_directive(directive)
                return {"source": "local", "directive": data}
            except Exception as exc:
                logger.warning("Failed to read NCC directive file: %s", exc)

        # Try REST endpoint if token configured
        if self._ncc_token:
            try:
                import aiohttp
                headers = {"Authorization": f"Bearer {self._ncc_token}"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self._ncc_endpoint}/governance/directive",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.state.ncc.connected = True
                            self.state.ncc.last_heartbeat = datetime.now().isoformat()
                            if "action" in data:
                                directive = GovernanceDirective(
                                    directive_id=data.get("id", "ncc-live"),
                                    action=data["action"],
                                    reason=data.get("reason", "NCC governance"),
                                    timestamp=datetime.now().isoformat(),
                                    params=data.get("params", {}),
                                )
                                self._apply_directive(directive)
                            return {"source": "rest", "data": data}
            except Exception as exc:
                self.state.ncc.error = str(exc)
                logger.debug("NCC REST unavailable: %s", exc)

        self.state.ncc.mode = "offline"
        return {"source": "none", "doctrine_mode": self.state.doctrine_mode}

    def _apply_directive(self, directive: GovernanceDirective) -> None:
        """Apply a governance directive to trading state."""
        self.state.last_directive = directive
        action_map = {
            "halt": "HALT",
            "safe_mode": "SAFE_MODE",
            "caution": "CAUTION",
            "resume": "NORMAL",
            "reduce_exposure": "CAUTION",
        }
        new_mode = action_map.get(directive.action, self.state.doctrine_mode)
        if new_mode != self.state.doctrine_mode:
            logger.warning(
                "DOCTRINE MODE CHANGE: %s -> %s (reason: %s)",
                self.state.doctrine_mode, new_mode, directive.reason,
            )
            self.state.doctrine_mode = new_mode
        self._save_state()

    def should_trade(self) -> bool:
        """Check if trading is allowed under current governance."""
        return self.state.doctrine_mode not in ("HALT", "SAFE_MODE")

    def get_risk_multiplier(self) -> float:
        """Get position size multiplier based on doctrine mode."""
        return {
            "NORMAL": 1.0,
            "CAUTION": 0.5,
            "SAFE_MODE": 0.0,
            "HALT": 0.0,
        }.get(self.state.doctrine_mode, 0.0)

    # ── NCL Integration ────────────────────────────────────────────────

    async def get_ncl_intelligence(self) -> Dict[str, Any]:
        """Get market intelligence from NCL (BRAIN pillar)."""
        result: Dict[str, Any] = {"source": "none", "signals": []}

        # Check NCL data path for shared intelligence files
        if self._ncl_path.exists():
            intel_file = self._ncl_path / "data" / "market_intelligence.json"
            if intel_file.exists():
                try:
                    data = json.loads(intel_file.read_text())
                    self.state.ncl.connected = True
                    self.state.ncl.last_heartbeat = datetime.now().isoformat()
                    self.state.ncl.mode = "file-sync"
                    result = {"source": "file", "data": data}
                except Exception as exc:
                    logger.warning("Failed to read NCL intelligence: %s", exc)

        # Check for NCL forecast outputs
        if self._ncl_path.exists():
            forecast_dir = self._ncl_path / "data" / "forecasts"
            if forecast_dir.exists():
                forecasts = []
                for f in sorted(forecast_dir.glob("*.json"))[-5:]:
                    try:
                        forecasts.append(json.loads(f.read_text()))
                    except Exception:
                        pass
                if forecasts:
                    result["forecasts"] = forecasts

        return result

    async def push_intelligence_to_ncl(self, data: Dict[str, Any]) -> bool:
        """
        Push AAC intelligence snapshot to NCL (BRAIN pillar) for bi-directional sync.
        Writes to NCL_DATA_PATH/data/aac_intelligence.json.
        NCL reads this file to incorporate AAC market signals into its BRAIN layer.
        """
        if not self._ncl_path.exists():
            return False
        try:
            out_dir = self._ncl_path / "data"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "source": "AAC",
                "timestamp": datetime.now().isoformat(),
                **data,
            }
            (out_dir / "aac_intelligence.json").write_text(
                json.dumps(payload, indent=2)
            )
            logger.debug("Pushed AAC intelligence to NCL (%d keys)", len(payload))
            return True
        except Exception as exc:
            logger.warning("push_intelligence_to_ncl failed: %s", exc)
            return False

    # ── BRS Integration ────────────────────────────────────────────────

    async def get_brs_signals(self) -> Dict[str, Any]:
        """Get trading education/pattern signals from BRS (Bravo)."""
        result: Dict[str, Any] = {"source": "none", "patterns": []}

        # BRS shares pattern recognition via state files
        pattern_file = self._state_dir / "brs_patterns.json"
        if pattern_file.exists():
            try:
                data = json.loads(pattern_file.read_text())
                self.state.brs.connected = True
                self.state.brs.last_heartbeat = datetime.now().isoformat()
                result = {"source": "file", "patterns": data.get("patterns", [])}
            except Exception as exc:
                logger.warning("Failed to read BRS patterns: %s", exc)

        return result

    # ── Status & Reporting ─────────────────────────────────────────────

    def get_full_status(self) -> Dict[str, Any]:
        """Get full cross-pillar status report."""
        return {
            "doctrine_mode": self.state.doctrine_mode,
            "should_trade": self.should_trade(),
            "risk_multiplier": self.get_risk_multiplier(),
            "pillars": {
                "ncc": self.state.ncc.to_dict(),
                "ncl": self.state.ncl.to_dict(),
                "brs": self.state.brs.to_dict(),
            },
            "last_directive": (
                {
                    "action": self.state.last_directive.action,
                    "reason": self.state.last_directive.reason,
                    "timestamp": self.state.last_directive.timestamp,
                }
                if self.state.last_directive
                else None
            ),
            "active_strategies": self.state.active_strategies,
            "timestamp": datetime.now().isoformat(),
        }

    # ── Persistence ────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist cross-pillar state to disk."""
        state_file = self._state_dir / "cross_pillar_state.json"
        state_data = {
            "doctrine_mode": self.state.doctrine_mode,
            "active_strategies": self.state.active_strategies,
            "last_directive": (
                {
                    "directive_id": self.state.last_directive.directive_id,
                    "action": self.state.last_directive.action,
                    "reason": self.state.last_directive.reason,
                    "timestamp": self.state.last_directive.timestamp,
                }
                if self.state.last_directive
                else None
            ),
            "saved_at": datetime.now().isoformat(),
        }
        state_file.write_text(json.dumps(state_data, indent=2))

    def _load_state(self) -> None:
        """Load persisted cross-pillar state."""
        state_file = self._state_dir / "cross_pillar_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.state.doctrine_mode = data.get("doctrine_mode", "NORMAL")
                self.state.active_strategies = data.get("active_strategies", [])
            except Exception as exc:
                logger.warning("Failed to load cross-pillar state: %s", exc)


# ── Singleton accessor ─────────────────────────────────────────────────

_hub: Optional[CrossPillarHub] = None


def get_cross_pillar_hub() -> CrossPillarHub:
    """Get or create the cross-pillar integration hub."""
    global _hub
    if _hub is None:
        _hub = CrossPillarHub()
    return _hub

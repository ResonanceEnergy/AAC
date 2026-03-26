#!/usr/bin/env python3
from __future__ import annotations

"""
Strategy Relay Bridge — Wire All 7 Strategies + Advisor to NCL BRAIN
=====================================================================
"ALL INTELLIGENCE GOES TO NCL BRAIN FOR PROCESSING THROUGH RELAY"

This module publishes strategy outputs to the NCC Relay (port 8787)
using the envelope format: ncl.sync.v1.bank.<category>

Envelope categories:
    ncl.sync.v1.bank.strategy.war_room
    ncl.sync.v1.bank.strategy.storm_lifeboat
    ncl.sync.v1.bank.strategy.capital_engine
    ncl.sync.v1.bank.strategy.matrix_maximizer
    ncl.sync.v1.bank.strategy.exploitation_matrix
    ncl.sync.v1.bank.strategy.polymarket
    ncl.sync.v1.bank.strategy.blackswan_authority
    ncl.sync.v1.bank.advisor.leaderboard
    ncl.sync.v1.bank.advisor.recommendation
    ncl.sync.v1.bank.doctrine.state_change
    ncl.sync.v1.bank.doctrine.manual_override
    ncl.sync.v1.bank.paper_proof.performance

Usage:
    from shared.strategy_relay_bridge import get_strategy_relay

    relay = get_strategy_relay()
    relay.publish_strategy("war_room", {"mandate": "DEFENSIVE", ...})
    relay.publish_advisor_leaderboard([...])
    relay.publish_doctrine_state({...})
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("StrategyRelayBridge")


class StrategyRelayBridge:
    """
    Centralised relay bridge for all strategy-to-NCL communication.
    Wraps NCCRelayClient with typed publish methods for each category.
    """

    STRATEGY_KEYS = (
        "war_room", "storm_lifeboat", "capital_engine",
        "matrix_maximizer", "exploitation_matrix",
        "polymarket", "blackswan_authority",
    )

    def __init__(self):
        self._relay = None
        self._publish_count = 0

    def _get_relay(self):
        """Lazy-load NCCRelayClient to avoid circular imports."""
        if self._relay is None:
            try:
                from shared.ncc_relay_client import get_relay_client
                self._relay = get_relay_client()
            except Exception as exc:
                logger.warning("NCCRelayClient not available: %s", exc)
        return self._relay

    # ── Strategy Output Publishing ──────────────────────────────────

    def publish_strategy(self, strategy_key: str, data: Dict[str, Any]) -> bool:
        """
        Publish a strategy's output to NCL BRAIN.
        event_type: ncl.sync.v1.bank.strategy.<key>
        """
        if strategy_key not in self.STRATEGY_KEYS:
            logger.warning("Unknown strategy key: %s", strategy_key)
            return False

        relay = self._get_relay()
        if relay is None:
            return False

        payload = {
            "strategy": strategy_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        event_type = f"ncl.sync.v1.bank.strategy.{strategy_key}"
        result = relay.publish(event_type, payload)
        if result:
            self._publish_count += 1
        return result

    def publish_all_strategies(self, outputs: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Publish outputs from multiple strategies in one call."""
        results = {}
        for key, data in outputs.items():
            results[key] = self.publish_strategy(key, data)
        return results

    # ── Advisor Publishing ──────────────────────────────────────────

    def publish_advisor_leaderboard(self, leaderboard: List[Dict[str, Any]]) -> bool:
        """Publish strategy advisor leaderboard to NCL BRAIN."""
        relay = self._get_relay()
        if relay is None:
            return False

        payload = {
            "leaderboard": leaderboard,
            "count": len(leaderboard),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return relay.publish("ncl.sync.v1.bank.advisor.leaderboard", payload)

    def publish_advisor_recommendations(self, recommendations: List[Dict[str, Any]]) -> bool:
        """Publish advisor recommendations to NCL BRAIN."""
        relay = self._get_relay()
        if relay is None:
            return False

        payload = {
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return relay.publish("ncl.sync.v1.bank.advisor.recommendation", payload)

    # ── Doctrine Publishing ─────────────────────────────────────────

    def publish_doctrine_state(self, state: Dict[str, Any]) -> bool:
        """Publish doctrine state change to NCL BRAIN."""
        relay = self._get_relay()
        if relay is None:
            return False

        payload = {
            "doctrine_state": state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return relay.publish("ncl.sync.v1.bank.doctrine.state_change", payload)

    def publish_manual_override(self, strategy_name: str, override: Dict[str, Any]) -> bool:
        """Publish manual override event to NCL BRAIN."""
        relay = self._get_relay()
        if relay is None:
            return False

        payload = {
            "strategy": strategy_name,
            "override": override,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return relay.publish("ncl.sync.v1.bank.doctrine.manual_override", payload)

    # ── Paper Proof Publishing ──────────────────────────────────────

    def publish_paper_proof_performance(self, performance: Dict[str, Any]) -> bool:
        """Publish paper-proof performance data to NCL BRAIN."""
        relay = self._get_relay()
        if relay is None:
            return False

        payload = {
            "paper_proof": performance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return relay.publish("ncl.sync.v1.bank.paper_proof.performance", payload)

    # ── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        relay = self._get_relay()
        base = {"bridge_publishes": self._publish_count}
        if relay and hasattr(relay, "stats"):
            base.update(relay.stats)
        return base


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL ACCESSOR
# ═══════════════════════════════════════════════════════════════════════════

_strategy_relay: Optional[StrategyRelayBridge] = None


def get_strategy_relay() -> StrategyRelayBridge:
    """Get or create the singleton StrategyRelayBridge."""
    global _strategy_relay
    if _strategy_relay is None:
        _strategy_relay = StrategyRelayBridge()
    return _strategy_relay

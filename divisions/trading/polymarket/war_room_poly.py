"""
War Room Poly — Geopolitical Thesis-Aligned Polymarket Bets
============================================================
AAC Polymarket Division — Strategy #1

Bridges the War Room thesis engine to Polymarket execution:
  - Uses the Black Swan Pressure Cooker thesis chain
  - Maps crisis escalation indicators to Polymarket markets
  - Deep OTM bets on geopolitical tail-risk outcomes
  - Integrates with polymarket_blackswan_scanner for scanning
  - Forward Monte Carlo from war_room_engine for probability estimation

Thesis Chain:
  Iran escalation -> US ME withdrawal -> Gulf adopts yuan ->
  Gold reprices -> USD collapses -> Crypto contagion

This is a WRAPPER that unifies the existing war_room_engine thesis
with the polymarket_blackswan_scanner execution layer.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# Pressure cooker thesis stages and their Polymarket keyword mappings
THESIS_STAGES = {
    "stage_1_iran_escalation": {
        "description": "Iran-Israel conflict escalation, Hormuz threat",
        "pressure_weight": 0.25,
        "keywords": ["iran", "hormuz", "irgc", "persian gulf", "hezbollah", "israeli strike"],
        "polymarket_bias": "YES",  # We bet YES on escalation
    },
    "stage_2_us_withdrawal": {
        "description": "US military pullback from Middle East",
        "pressure_weight": 0.20,
        "keywords": ["us troops", "centcom", "us military", "middle east withdrawal"],
        "polymarket_bias": "YES",
    },
    "stage_3_gulf_shift": {
        "description": "Gulf states pivot to BRICS/yuan",
        "pressure_weight": 0.20,
        "keywords": ["saudi", "brics", "petrodollar", "yuan", "petro-yuan", "de-dollarization"],
        "polymarket_bias": "YES",
    },
    "stage_4_gold_reprice": {
        "description": "Gold reprices above $3000-$5000",
        "pressure_weight": 0.15,
        "keywords": ["gold price", "gold above", "gold 3000", "gold 4000", "gold 5000"],
        "polymarket_bias": "YES",
    },
    "stage_5_usd_collapse": {
        "description": "USD index crashes, reserve currency shift",
        "pressure_weight": 0.10,
        "keywords": ["dollar crash", "dxy below", "dollar index", "reserve currency"],
        "polymarket_bias": "YES",
    },
    "stage_6_crypto_contagion": {
        "description": "Crypto selloff from risk-off cascade",
        "pressure_weight": 0.10,
        "keywords": ["bitcoin below", "crypto crash", "ethereum crash", "bitcoin crash"],
        "polymarket_bias": "YES",
    },
}


@dataclass
class ThesisMarketMatch:
    """A Polymarket market matching a War Room thesis stage."""
    stage: str
    stage_description: str
    market_question: str
    condition_id: str
    outcome: str
    market_price: float
    thesis_multiplier: float
    thesis_probability: float
    edge: float
    token_id: str = ""
    volume_24h: float = 0.0
    liquidity: float = 0.0


class WarRoomPoly:
    """
    War Room Polymarket adapter.

    Scans Polymarket for markets that match our geopolitical thesis chain,
    applies War Room probability adjustments, and identifies high-edge bets.
    """

    def __init__(self):
        self.matches: List[ThesisMarketMatch] = []
        self.pressure_level: float = 0.44  # Current pressure cooker level

    def set_pressure_level(self, level: float):
        """Update the pressure cooker level (0.0 to 1.0)."""
        self.pressure_level = max(0.0, min(1.0, level))

    def classify_market(self, question: str) -> Optional[str]:
        """Classify a market question into a thesis stage."""
        q_lower = question.lower()
        for stage, info in THESIS_STAGES.items():
            for kw in info["keywords"]:
                if kw in q_lower:
                    return stage
        return None

    def thesis_probability(self, stage: str, market_price: float) -> float:
        """
        Calculate thesis-adjusted probability for a market.

        POST-MORTEM FIX (2026-04-03): The old multiplier system (2x-5x)
        created fantasy probabilities (50% for 10-cent markets). This caused
        $452 in losses. Markets are efficient — our edge is 2-5%, not 40%.

        New approach: bounded deviation from market consensus.
        Max 2x multiplier with strong evidence, typical 1.2-1.5x.
        """
        info = THESIS_STAGES.get(stage)
        if not info:
            return market_price

        # Cap the multiplier: max 2x for highest-conviction thesis bets
        # Previously was 2x-5x which produced insane probabilities
        if market_price < 0.05:
            base_mult = 1.5  # Was 5.0 — markets below 5c are usually correct
        elif market_price < 0.10:
            base_mult = 1.4  # Was 4.0
        elif market_price < 0.20:
            base_mult = 1.3  # Was 3.0
        elif market_price < 0.30:
            base_mult = 1.2  # Was 2.5
        else:
            base_mult = 1.1  # Was 2.0

        # Pressure adjustment: mild boost (was 1.0 + pressure*0.5 = up to 1.5x)
        pressure_boost = 1.0 + (self.pressure_level * 0.15)  # Max 1.15x

        # Stage weight: slight preference for confirmed early stages
        stage_boost = 1.0 + (info["pressure_weight"] * 0.3)  # Max ~1.075x

        adjusted = market_price * base_mult * pressure_boost * stage_boost
        # Hard cap: never claim more than 2x market price OR 50% probability
        upper_bound = min(market_price * 2.0, 0.50)
        return min(adjusted, upper_bound)

    def scan_results(self, scanner_opportunities: list) -> List[ThesisMarketMatch]:
        """
        Take BlackSwanOpportunity results and classify them into thesis stages.
        Returns matches sorted by edge.
        """
        self.matches = []

        for opp in scanner_opportunities:
            question = getattr(opp, "market_question", "")
            stage = self.classify_market(question)
            if not stage:
                continue

            market_price = getattr(opp, "market_price", 0)
            thesis_prob = self.thesis_probability(stage, market_price)
            edge = thesis_prob - market_price

            if edge <= 0:
                continue

            self.matches.append(ThesisMarketMatch(
                stage=stage,
                stage_description=THESIS_STAGES[stage]["description"],
                market_question=question,
                condition_id=getattr(opp, "condition_id", ""),
                outcome=getattr(opp, "outcome", "YES"),
                market_price=market_price,
                thesis_multiplier=thesis_prob / max(market_price, 0.001),
                thesis_probability=thesis_prob,
                edge=edge,
                token_id=getattr(opp, "token_id", ""),
                volume_24h=getattr(opp, "volume_24h", 0),
                liquidity=getattr(opp, "liquidity", 0),
            ))

        self.matches.sort(key=lambda x: x.edge, reverse=True)

        # TurboQuant: record thesis-market edge snapshot
        try:
            from strategies.turboquant_integrations import IntegrationHub
            _tq_hub = IntegrationHub()
            _tq_hub.record_polymarket(
                [m.__dict__ if hasattr(m, '__dict__') else m for m in self.matches]
            )
            _tq_hub.save_all()
        except Exception:
            pass

        return self.matches

    def generate_report(self) -> str:
        """Generate War Room Polymarket report."""
        lines = []
        lines.append("=" * 120)
        lines.append("  WAR ROOM POLY -- GEOPOLITICAL THESIS ALIGNED BETS")
        lines.append(f"  Pressure Cooker: {self.pressure_level:.0%} | "
                      f"Matches: {len(self.matches)} | "
                      f"Thesis Chain: 6 stages")
        lines.append("=" * 120)
        lines.append("")

        # Stage summary
        lines.append("  THESIS CHAIN STATUS:")
        for stage, info in THESIS_STAGES.items():
            stage_matches = [m for m in self.matches if m.stage == stage]
            lines.append(
                f"    {stage}: {info['description'][:50]:<52} "
                f"weight={info['pressure_weight']:.0%}  "
                f"markets={len(stage_matches)}"
            )
        lines.append("")

        if self.matches:
            lines.append(f"  {'#':<4} {'Stage':<25} {'Market':<40} {'Price':>7} "
                          f"{'Thesis':>7} {'Edge':>7} {'Mult':>6} {'Vol24h':>10}")
            lines.append("  " + "-" * 108)

            for i, m in enumerate(self.matches[:20], 1):
                stage_short = m.stage.replace("stage_", "S").replace("_", " ")[:23]
                q = m.market_question[:38] if len(m.market_question) > 38 else m.market_question
                lines.append(
                    f"  {i:<4} {stage_short:<25} {q:<40} ${m.market_price:.3f} "
                    f"{m.thesis_probability:.1%} {m.edge:>+.3f} "
                    f"{m.thesis_multiplier:>5.1f}x ${m.volume_24h:>8,.0f}"
                )

        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict."""
        return {
            "pressure_level": self.pressure_level,
            "n_matches": len(self.matches),
            "stages": {
                stage: {
                    "description": info["description"],
                    "weight": info["pressure_weight"],
                    "matches": len([m for m in self.matches if m.stage == stage]),
                }
                for stage, info in THESIS_STAGES.items()
            },
            "top_matches": [
                {
                    "stage": m.stage,
                    "question": m.market_question,
                    "price": m.market_price,
                    "thesis_prob": m.thesis_probability,
                    "edge": m.edge,
                    "multiplier": m.thesis_multiplier,
                }
                for m in self.matches[:10]
            ],
        }

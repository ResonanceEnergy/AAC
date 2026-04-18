from __future__ import annotations

"""Polymarket Council Division — autonomous prediction market scanner.

Periodically scrapes Polymarket via Gamma API, analyzes market
movements and arbitrage opportunities, publishes INTEL_UPDATE signals.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from divisions.division_protocol import DivisionProtocol, Signal, SignalType

from councils.polymarket.models import CouncilEntry
from councils.polymarket.pipeline import run_polymarket_council

_log = structlog.get_logger(__name__)

_SEEN_DB = Path(__file__).parent / "output" / "polymarket_seen.json"


class PolymarketCouncilDivision(DivisionProtocol):
    """Autonomous Polymarket intelligence scraper.

    Each scan cycle:
    1. Fetches trending markets from Gamma API
    2. Detects arbitrage opportunities (YES+NO < 1.0)
    3. Analyzes volume leaders, category breakdown, sentiment
    4. Publishes INTEL_UPDATE signals with key market insights
    """

    def __init__(
        self,
        limit: int = 50,
        keywords: list[str] | None = None,
        min_edge_pct: float = 0.5,
    ) -> None:
        super().__init__(division_name="polymarket_council")
        self._limit = limit
        self._keywords = keywords or [
            "crypto", "bitcoin", "oil", "iran", "fed", "tariff",
        ]
        self._min_edge_pct = min_edge_pct
        self._total_scans: int = 0
        self._total_arbs_found: int = 0
        self._last_entry: CouncilEntry | None = None
        self._seen_conditions: set[str] = set()
        self._load_seen()

    def _load_seen(self) -> None:
        if _SEEN_DB.exists():
            try:
                data = json.loads(_SEEN_DB.read_text(encoding="utf-8"))
                self._seen_conditions = set(data.get("seen_ids", []))
            except (json.JSONDecodeError, KeyError):
                self._seen_conditions = set()

    def _save_seen(self) -> None:
        _SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
        # Keep only the last 500 condition IDs to avoid unbounded growth
        trimmed = sorted(self._seen_conditions)[-500:]
        _SEEN_DB.write_text(
            json.dumps({
                "seen_ids": trimmed,
                "updated": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )

    async def scan(self) -> list[Signal]:
        """Scrape Polymarket, analyze, return intel signals."""
        signals: list[Signal] = []

        try:
            entry = await run_polymarket_council(
                limit=self._limit,
                keywords=self._keywords,
                min_edge_pct=self._min_edge_pct,
            )
        except Exception as exc:
            _log.warning("polymarket_council.scan_failed", error=str(exc))
            return []

        if not entry:
            return []

        self._total_scans += 1
        self._last_entry = entry
        self._total_arbs_found += len(entry.insights.arb_opportunities)

        # Track new markets
        new_conditions = {
            m.condition_id for m in entry.markets
        } - self._seen_conditions
        self._seen_conditions.update(m.condition_id for m in entry.markets)
        self._save_seen()

        # Main intel signal — summary of this scan cycle
        trust_overall = entry.insights.trust_score.get("overall", 0.7)
        signals.append(Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division=self.division_name,
            timestamp=datetime.now(timezone.utc),
            data={
                "type": "polymarket_scan",
                "total_markets": entry.insights.total_markets,
                "total_volume": entry.insights.total_volume,
                "new_markets": len(new_conditions),
                "sentiment": entry.insights.sentiment,
                "summary": entry.insights.summary[:400],
                "top_volume": entry.insights.top_by_volume[:3],
                "categories": dict(list(entry.insights.category_breakdown.items())[:5]),
                "markdown_path": entry.markdown_path,
                "trust_score": entry.insights.trust_score,
            },
            confidence=trust_overall,
            urgency=0,
        ))

        # Separate signal for arbitrage opportunities
        if entry.insights.arb_opportunities:
            signals.append(Signal(
                signal_type=SignalType.INTEL_UPDATE,
                source_division=self.division_name,
                timestamp=datetime.now(timezone.utc),
                data={
                    "type": "polymarket_arb_alert",
                    "arb_count": len(entry.insights.arb_opportunities),
                    "best_edge": entry.insights.arb_opportunities[0].edge_pct,
                    "best_market": entry.insights.arb_opportunities[0].question[:100],
                    "opportunities": [
                        {
                            "question": a.question[:80],
                            "edge_pct": a.edge_pct,
                            "side": a.side,
                            "yes": a.yes_price,
                            "no": a.no_price,
                        }
                        for a in entry.insights.arb_opportunities[:5]
                    ],
                },
                confidence=min(trust_overall + 0.1, 1.0),
                urgency=1,
            ))

        return signals

    async def report(self) -> dict[str, Any]:
        return {
            "division": self.division_name,
            "total_scans": self._total_scans,
            "total_arbs_found": self._total_arbs_found,
            "seen_conditions": len(self._seen_conditions),
            "keywords": self._keywords,
            "last_scan_markets": self._last_entry.insights.total_markets if self._last_entry else 0,
            "last_scan_sentiment": self._last_entry.insights.sentiment if self._last_entry else "n/a",
        }

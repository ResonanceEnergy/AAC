from __future__ import annotations

"""Polymarket Council data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MarketSnapshot:
    """A snapshot of a single Polymarket binary-outcome market."""

    condition_id: str
    question: str
    slug: str = ""
    yes_price: float = 0.5
    no_price: float = 0.5
    volume: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    tags: list[str] = field(default_factory=list)
    active: bool = True
    end_date: str = ""

    @property
    def overround(self) -> float:
        """YES + NO price. <1.0 implies pure arb."""
        return self.yes_price + self.no_price

    @property
    def implied_prob(self) -> float:
        """Market-implied probability (YES price)."""
        return self.yes_price


@dataclass
class ArbitrageOpp:
    """An arbitrage or edge opportunity detected on Polymarket."""

    question: str
    condition_id: str
    yes_price: float
    no_price: float
    edge_pct: float
    side: str = "YES"  # which side the edge favours


@dataclass
class MarketInsight:
    """Aggregated analysis across a batch of scraped markets."""

    total_markets: int = 0
    total_volume: float = 0.0
    top_by_volume: list[dict[str, Any]] = field(default_factory=list)
    top_by_liquidity: list[dict[str, Any]] = field(default_factory=list)
    arb_opportunities: list[ArbitrageOpp] = field(default_factory=list)
    category_breakdown: dict[str, int] = field(default_factory=dict)
    high_conviction: list[dict[str, Any]] = field(default_factory=list)
    low_conviction: list[dict[str, Any]] = field(default_factory=list)
    sentiment: str = "neutral"
    summary: str = ""
    trust_score: dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilEntry:
    """Result of a single Polymarket Council scan cycle."""

    markets: list[MarketSnapshot]
    insights: MarketInsight
    markdown_path: str = ""
    processed_at: str = ""

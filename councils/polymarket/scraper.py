from __future__ import annotations

"""Polymarket Council scraper — fetch markets from Gamma API.

Uses the existing PolymarketAgent for all API access (Gamma + CLOB).
Does NOT place orders — this is pure intelligence gathering.
"""

import asyncio
from typing import Any

import structlog

from councils.polymarket.models import ArbitrageOpp, MarketSnapshot

_log = structlog.get_logger(__name__)

# Gamma API base — same constant as the agent
GAMMA_API = "https://gamma-api.polymarket.com"


async def scrape_trending_markets(
    limit: int = 50,
) -> list[MarketSnapshot]:
    """Fetch top Polymarket markets by volume via Gamma API.

    Uses the PolymarketAgent internally so we get retry/backoff for free.
    """
    from agents.polymarket_agent import PolymarketAgent

    snapshots: list[MarketSnapshot] = []
    try:
        async with PolymarketAgent() as agent:
            events = await agent.get_trending_events(limit=limit)
            for ev in events:
                for mkt_data in ev.markets:
                    from agents.polymarket_agent import PolymarketMarket as PM

                    mkt = PM.from_api(mkt_data)
                    snapshots.append(MarketSnapshot(
                        condition_id=mkt.condition_id,
                        question=mkt.question,
                        slug=mkt.slug,
                        yes_price=mkt.yes_price,
                        no_price=mkt.no_price,
                        volume=mkt.volume,
                        liquidity=mkt.liquidity,
                        tags=ev.tags,
                        active=mkt.active,
                        end_date=ev.end_date,
                        volume_24h=ev.volume_24hr,
                    ))
    except Exception as exc:
        _log.warning("polymarket.scrape_trending_failed", error=str(exc))

    _log.info("polymarket.scraped_trending", count=len(snapshots))
    return snapshots


async def scrape_active_markets(
    limit: int = 100,
) -> list[MarketSnapshot]:
    """Fetch active markets directly (not nested under events)."""
    from agents.polymarket_agent import PolymarketAgent

    snapshots: list[MarketSnapshot] = []
    try:
        async with PolymarketAgent() as agent:
            markets = await agent.get_active_markets(limit=limit)
            for mkt in markets:
                snapshots.append(MarketSnapshot(
                    condition_id=mkt.condition_id,
                    question=mkt.question,
                    slug=mkt.slug,
                    yes_price=mkt.yes_price,
                    no_price=mkt.no_price,
                    volume=mkt.volume,
                    liquidity=mkt.liquidity,
                    active=mkt.active,
                ))
    except Exception as exc:
        _log.warning("polymarket.scrape_active_failed", error=str(exc))

    _log.info("polymarket.scraped_active", count=len(snapshots))
    return snapshots


async def scrape_search(
    keywords: list[str],
    limit_per_keyword: int = 20,
) -> list[MarketSnapshot]:
    """Search Gamma API for markets matching keywords."""
    from agents.polymarket_agent import PolymarketAgent

    seen: set[str] = set()
    snapshots: list[MarketSnapshot] = []
    try:
        async with PolymarketAgent() as agent:
            for kw in keywords:
                markets = await agent.search_markets(kw, limit=limit_per_keyword)
                for mkt in markets:
                    if mkt.condition_id in seen:
                        continue
                    seen.add(mkt.condition_id)
                    snapshots.append(MarketSnapshot(
                        condition_id=mkt.condition_id,
                        question=mkt.question,
                        slug=mkt.slug,
                        yes_price=mkt.yes_price,
                        no_price=mkt.no_price,
                        volume=mkt.volume,
                        liquidity=mkt.liquidity,
                        active=mkt.active,
                    ))
    except Exception as exc:
        _log.warning("polymarket.scrape_search_failed", error=str(exc))

    _log.info("polymarket.scraped_search", keywords=keywords, count=len(snapshots))
    return snapshots


def detect_arbitrage(
    markets: list[MarketSnapshot],
    min_edge_pct: float = 0.5,
) -> list[ArbitrageOpp]:
    """Find markets where YES+NO < 1.0 (pure overround arb)."""
    opps: list[ArbitrageOpp] = []
    for m in markets:
        overround = m.yes_price + m.no_price
        if overround < (1.0 - min_edge_pct / 100):
            edge = (1.0 - overround) * 100
            side = "YES" if m.yes_price <= m.no_price else "NO"
            opps.append(ArbitrageOpp(
                question=m.question,
                condition_id=m.condition_id,
                yes_price=m.yes_price,
                no_price=m.no_price,
                edge_pct=round(edge, 2),
                side=side,
            ))
    opps.sort(key=lambda o: o.edge_pct, reverse=True)
    return opps

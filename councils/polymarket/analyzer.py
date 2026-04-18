from __future__ import annotations

"""Polymarket Council analyzer — extract insights from scraped markets."""

from collections import Counter
from typing import Any

import structlog

from councils.polymarket.models import ArbitrageOpp, MarketInsight, MarketSnapshot

_log = structlog.get_logger(__name__)


def _compute_poly_trust(
    markets: list[MarketSnapshot],
) -> dict[str, float]:
    """Compute trust score for Polymarket analysis."""
    from councils.trust import TrustScore, polymarket_source_trust, evidence_score

    avg_liq = sum(m.liquidity for m in markets) / max(len(markets), 1)
    avg_vol = sum(m.volume for m in markets) / max(len(markets), 1)
    src = polymarket_source_trust(
        market_count=len(markets),
        avg_liquidity=avg_liq,
        avg_volume=avg_vol,
    )
    ev = evidence_score(len(markets), target=100)

    ts = TrustScore(
        source_reliability=src,
        data_freshness=0.9,  # Gamma API returns near-real-time data
        evidence_volume=ev,
    )
    return ts.to_dict()


def analyze_markets(
    markets: list[MarketSnapshot],
    arb_opps: list[ArbitrageOpp] | None = None,
) -> MarketInsight:
    """Produce aggregated insights from a batch of market snapshots."""
    if not markets:
        return MarketInsight(summary="No markets to analyze.")

    total_volume = sum(m.volume for m in markets)

    # Top by volume
    by_vol = sorted(markets, key=lambda m: m.volume, reverse=True)
    top_by_volume = [
        {"question": m.question, "volume": m.volume, "yes": m.yes_price}
        for m in by_vol[:10]
    ]

    # Top by liquidity
    by_liq = sorted(markets, key=lambda m: m.liquidity, reverse=True)
    top_by_liquidity = [
        {"question": m.question, "liquidity": m.liquidity, "yes": m.yes_price}
        for m in by_liq[:10]
    ]

    # Category breakdown from tags
    tag_counter: Counter[str] = Counter()
    for m in markets:
        for tag in m.tags:
            tag_counter[tag] += 1
    category_breakdown = dict(tag_counter.most_common(15))

    # High conviction (YES > 85% or < 15%)
    high_conviction = [
        {"question": m.question, "yes": m.yes_price, "volume": m.volume}
        for m in markets
        if m.yes_price > 0.85 or m.yes_price < 0.15
    ]
    high_conviction.sort(key=lambda x: x["volume"], reverse=True)

    # Low conviction (YES 40-60% — toss-ups, most interesting)
    low_conviction = [
        {"question": m.question, "yes": m.yes_price, "volume": m.volume}
        for m in markets
        if 0.40 <= m.yes_price <= 0.60
    ]
    low_conviction.sort(key=lambda x: x["volume"], reverse=True)

    # Sentiment: are most markets priced bullish or bearish?
    bullish = sum(1 for m in markets if m.yes_price > 0.6)
    bearish = sum(1 for m in markets if m.yes_price < 0.4)
    if bullish > bearish * 1.5:
        sentiment = "bullish"
    elif bearish > bullish * 1.5:
        sentiment = "bearish"
    else:
        sentiment = "mixed"

    arbs = arb_opps or []

    summary_parts = [
        f"Scanned {len(markets)} markets, ${total_volume:,.0f} total volume.",
    ]
    if arbs:
        summary_parts.append(f"{len(arbs)} arbitrage opportunities detected (best edge: {arbs[0].edge_pct:.1f}%).")
    if high_conviction:
        summary_parts.append(f"{len(high_conviction)} high-conviction markets (>85% or <15%).")
    if low_conviction:
        summary_parts.append(f"{len(low_conviction)} toss-up markets (40-60%).")
    summary_parts.append(f"Overall sentiment: {sentiment}.")

    trust = _compute_poly_trust(markets)

    return MarketInsight(
        total_markets=len(markets),
        total_volume=total_volume,
        top_by_volume=top_by_volume[:10],
        top_by_liquidity=top_by_liquidity[:10],
        arb_opportunities=arbs,
        category_breakdown=category_breakdown,
        high_conviction=high_conviction[:10],
        low_conviction=low_conviction[:10],
        sentiment=sentiment,
        summary=" ".join(summary_parts),
        trust_score=trust,
    )

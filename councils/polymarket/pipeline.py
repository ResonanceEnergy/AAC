from __future__ import annotations

"""Polymarket Council pipeline — end-to-end scan → analyze → save."""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import structlog

from councils.polymarket.analyzer import analyze_markets
from councils.polymarket.formatter import format_to_markdown
from councils.polymarket.models import CouncilEntry
from councils.polymarket.scraper import (
    detect_arbitrage,
    scrape_active_markets,
    scrape_search,
    scrape_trending_markets,
)

_log = structlog.get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


async def run_polymarket_council(
    limit: int = 50,
    keywords: list[str] | None = None,
    min_edge_pct: float = 0.5,
) -> CouncilEntry | None:
    """Run a full Polymarket Council cycle.

    1. Scrape trending markets from Gamma API
    2. Optionally search for keyword-specific markets
    3. Detect arbitrage opportunities
    4. Analyze and produce insights
    5. Save Markdown + JSON reports
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1 — Scrape
    _log.info("polymarket_council.starting", limit=limit, keywords=keywords)
    markets = await scrape_trending_markets(limit=limit)

    if keywords:
        extra = await scrape_search(keywords)
        seen = {m.condition_id for m in markets}
        for m in extra:
            if m.condition_id not in seen:
                markets.append(m)
                seen.add(m.condition_id)

    if not markets:
        _log.warning("polymarket_council.no_markets_found")
        return None

    # 2 — Detect arb
    arbs = detect_arbitrage(markets, min_edge_pct=min_edge_pct)

    # 3 — Analyze
    insights = analyze_markets(markets, arb_opps=arbs)

    # 4 — Format
    md_content = format_to_markdown(markets, insights)

    # 5 — Save
    ts = datetime.now(timezone.utc)
    filename = f"polymarket_{ts.strftime('%Y%m%d_%H%M')}.md"
    md_path = OUTPUT_DIR / filename
    md_path.write_text(md_content, encoding="utf-8")

    json_path = OUTPUT_DIR / filename.replace(".md", ".json")
    json_path.write_text(
        json.dumps(
            {
                "timestamp": ts.isoformat(),
                "total_markets": len(markets),
                "arb_opportunities": len(arbs),
                "summary": insights.summary,
                "sentiment": insights.sentiment,
                "top_volume": insights.top_by_volume[:5],
                "categories": insights.category_breakdown,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entry = CouncilEntry(
        markets=markets,
        insights=insights,
        markdown_path=str(md_path),
        processed_at=ts.isoformat(),
    )

    _log.info(
        "polymarket_council.complete",
        markets=len(markets),
        arbs=len(arbs),
        md_path=str(md_path),
    )
    return entry


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Council — prediction market scanner")
    parser.add_argument("--limit", type=int, default=50, help="Markets to scrape")
    parser.add_argument("--keywords", nargs="*", help="Search keywords")
    parser.add_argument("--min-edge", type=float, default=0.5, help="Min arb edge %%")
    args = parser.parse_args()

    entry = asyncio.run(run_polymarket_council(
        limit=args.limit,
        keywords=args.keywords,
        min_edge_pct=args.min_edge,
    ))

    if entry:
        print(f"Polymarket Council complete: {len(entry.markets)} markets")
        print(f"  Arb opportunities: {len(entry.insights.arb_opportunities)}")
        print(f"  Sentiment: {entry.insights.sentiment}")
        print(f"  Report: {entry.markdown_path}")
    else:
        print("No markets found.")


if __name__ == "__main__":
    main()

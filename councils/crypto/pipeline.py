from __future__ import annotations

"""Crypto Council pipeline — end-to-end scrape → analyze → save."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

from councils.crypto.analyzer import analyze_crypto_market
from councils.crypto.formatter import format_to_markdown
from councils.crypto.models import CouncilEntry
from councils.crypto.scraper import DEFAULT_COINS, scrape_global, scrape_prices, scrape_trending

_log = structlog.get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


async def run_crypto_council(
    coin_ids: list[str] | None = None,
    vs_currency: str = "usd",
) -> CouncilEntry | None:
    """Run a full Crypto Council cycle.

    1. Scrape prices for watchlist coins via CoinGecko
    2. Scrape trending coins
    3. Scrape global market data
    4. Analyze and produce insights
    5. Save Markdown + JSON reports
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = coin_ids or list(DEFAULT_COINS)

    _log.info("crypto_council.starting", coin_count=len(ids), vs_currency=vs_currency)

    # 1 — Scrape
    coins = await scrape_prices(coin_ids=ids, vs_currency=vs_currency)
    trending = await scrape_trending()
    global_data = await scrape_global()

    if not coins:
        _log.warning("crypto_council.no_coin_data")
        return None

    # 2 — Analyze
    insights = analyze_crypto_market(coins, trending, global_data)

    # 3 — Format
    md_content = format_to_markdown(coins, insights)

    # 4 — Save
    ts = datetime.now(timezone.utc)
    filename = f"crypto_{ts.strftime('%Y%m%d_%H%M')}.md"
    md_path = OUTPUT_DIR / filename
    md_path.write_text(md_content, encoding="utf-8")

    json_path = OUTPUT_DIR / filename.replace(".md", ".json")
    json_path.write_text(
        json.dumps(
            {
                "timestamp": ts.isoformat(),
                "total_coins": len(coins),
                "trending_count": len(trending),
                "btc_price": insights.btc_price,
                "eth_price": insights.eth_price,
                "sentiment": insights.sentiment,
                "summary": insights.summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _log.info(
        "crypto_council.complete",
        coins=len(coins),
        trending=len(trending),
        sentiment=insights.sentiment,
        btc=insights.btc_price,
    )

    return CouncilEntry(
        coins=coins,
        trending_coins=trending,
        insights=insights,
        markdown_path=str(md_path),
        processed_at=ts.isoformat(),
    )


def main() -> None:
    """CLI entry-point."""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Council — intelligence scraper")
    parser.add_argument("--coins", nargs="*", default=None, help="CoinGecko coin IDs to track")
    parser.add_argument("--currency", default="usd", help="vs_currency (default: usd)")
    args = parser.parse_args()

    result = asyncio.run(run_crypto_council(coin_ids=args.coins, vs_currency=args.currency))
    if result:
        print(f"Report saved to {result.markdown_path}")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Unusual Whales integration using the configured API key."""

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_env_file
from integrations.unusual_whales_client import UnusualWhalesClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validate_unusual_whales")


async def main() -> int:
    """Run a lightweight connectivity and parsing check against Unusual Whales."""
    load_env_file()

    client = UnusualWhalesClient()
    if not client.endpoint.enabled:
        logger.error("UNUSUAL_WHALES_API_KEY is not configured")
        logger.info("Set UNUSUAL_WHALES_API_KEY in .env, then rerun this validator")
        return 1

    async with client:
        summary = await client.get_market_flow_summary()
        flow = await client.get_flow(limit=5, min_premium=0)
        dark_pool = await client.get_dark_pool(limit=5)
        congress = await client.get_congress_trades(limit=5)

    logger.info("Unusual Whales integration check complete")
    logger.info("Market summary keys: %s", sorted(summary.keys())[:10] if isinstance(summary, dict) else [])
    logger.info("Options flow rows: %d", len(flow))
    logger.info("Dark pool rows: %d", len(dark_pool))
    logger.info("Congress trade rows: %d", len(congress))

    if not summary and not flow and not dark_pool and not congress:
        logger.warning("The API responded with no usable data. Check plan entitlements and endpoint availability.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
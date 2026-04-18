from __future__ import annotations

"""Crypto Council scraper — fetch market data from CoinGecko.

Uses the existing CoinGeckoClient from shared/data_sources.py for all
API access. Rate-limit aware (free tier: 10 req/min).
"""

import asyncio
from typing import Any

import structlog

from councils.crypto.models import CoinSnapshot, GlobalData, TrendingCoin

_log = structlog.get_logger(__name__)

# Default watchlist — major coins that matter for AAC trading
DEFAULT_COINS = [
    "bitcoin", "ethereum", "solana", "ripple", "cardano",
    "dogecoin", "polkadot", "avalanche-2", "chainlink", "polygon",
    "litecoin", "uniswap", "aave", "maker", "arbitrum",
]


async def scrape_prices(
    coin_ids: list[str] | None = None,
    vs_currency: str = "usd",
) -> list[CoinSnapshot]:
    """Fetch current prices for a list of coins via CoinGecko."""
    from shared.data_sources import CoinGeckoClient

    ids = coin_ids or DEFAULT_COINS
    snapshots: list[CoinSnapshot] = []

    client = CoinGeckoClient()
    try:
        await client.connect()
        ticks = await client.get_prices_batch(ids, vs_currency=vs_currency)
        for tick in ticks:
            # tick.symbol is e.g. "BITCOIN/USD" — extract coin name
            coin_id = tick.symbol.split("/")[0].lower()
            snapshots.append(CoinSnapshot(
                coin_id=coin_id,
                symbol=tick.symbol,
                price=tick.price,
                volume_24h=tick.volume_24h,
                change_24h=tick.change_24h,
            ))
    except Exception as exc:
        _log.warning("crypto.scrape_prices_failed", error=str(exc))
    finally:
        await client.disconnect()

    _log.info("crypto.scraped_prices", count=len(snapshots))
    return snapshots


async def scrape_trending() -> list[TrendingCoin]:
    """Fetch trending coins from CoinGecko."""
    from shared.data_sources import CoinGeckoClient

    trending: list[TrendingCoin] = []

    client = CoinGeckoClient()
    try:
        await client.connect()
        raw = await client.get_trending()
        for item in raw:
            coin = item.get("item", item)
            trending.append(TrendingCoin(
                coin_id=coin.get("id", ""),
                name=coin.get("name", ""),
                symbol=coin.get("symbol", ""),
                market_cap_rank=coin.get("market_cap_rank", 0) or 0,
                thumb=coin.get("thumb", ""),
                score=coin.get("score", 0) or 0,
            ))
    except Exception as exc:
        _log.warning("crypto.scrape_trending_failed", error=str(exc))
    finally:
        await client.disconnect()

    _log.info("crypto.scraped_trending", count=len(trending))
    return trending


async def scrape_global() -> GlobalData:
    """Fetch global crypto market data from CoinGecko."""
    from shared.data_sources import CoinGeckoClient

    client = CoinGeckoClient()
    try:
        await client.connect()
        data = await client.get_global_data()
        if not data:
            return GlobalData()

        total_mcap = data.get("total_market_cap", {})
        total_vol = data.get("total_volume", {})
        mcap_pct = data.get("market_cap_percentage", {})

        return GlobalData(
            total_market_cap_usd=total_mcap.get("usd", 0),
            total_volume_24h_usd=total_vol.get("usd", 0),
            btc_dominance=mcap_pct.get("btc", 0),
            eth_dominance=mcap_pct.get("eth", 0),
            active_cryptocurrencies=data.get("active_cryptocurrencies", 0),
            market_cap_change_24h_pct=data.get("market_cap_change_percentage_24h_usd", 0),
        )
    except Exception as exc:
        _log.warning("crypto.scrape_global_failed", error=str(exc))
        return GlobalData()
    finally:
        await client.disconnect()

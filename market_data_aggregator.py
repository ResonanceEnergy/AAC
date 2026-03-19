#!/usr/bin/env python3
"""
Market Data Aggregator — Facade Module
========================================
Wraps ``shared.data_sources.DataAggregator`` and ``CoinGeckoClient``
so that legacy imports (``from market_data_aggregator import ...``)
continue to work and now return real data.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shared.data_sources import (
    CoinGeckoClient,
    MarketTick,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMarketData:
    """Aggregated market data snapshot."""
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    source: str = "coingecko"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tick(cls, tick: MarketTick) -> "AggregatedMarketData":
        return cls(
            symbol=tick.symbol,
            price=tick.price,
            volume=tick.volume_24h,
            bid=tick.bid or tick.price,
            ask=tick.ask or tick.price,
            spread=(tick.ask - tick.bid) if tick.bid and tick.ask else 0.0,
            source=tick.source,
        )


# Map common symbols → CoinGecko coin IDs
_SYMBOL_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "XRP": "ripple", "ADA": "cardano", "AVAX": "avalanche-2",
    "DOT": "polkadot", "MATIC": "matic-network", "LINK": "chainlink",
    "DOGE": "dogecoin",
}


class MarketDataAggregator:
    """Aggregates market data from CoinGecko (and future sources).

    Uses the real ``CoinGeckoClient`` from ``shared.data_sources``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.sources: List[str] = kwargs.get("sources", ["coingecko"])
        self._client = CoinGeckoClient()
        self._cache: Dict[str, AggregatedMarketData] = {}
        self._started = False

    # ── Sync helpers (for callers that aren't async) ──────────────────

    def _run(self, coro):
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop — create a task and return immediately
            # This path is used when called from async code that forgot 'await'
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    # ── Public API ────────────────────────────────────────────────────

    def get_aggregated_data(self, symbol: str) -> AggregatedMarketData:
        """Get aggregated data for a symbol (sync wrapper)."""
        cached = self._cache.get(symbol)
        if cached:
            return cached

        try:
            tick = self._run(self._fetch_tick(symbol))
            if tick and tick.price > 0:
                agg = AggregatedMarketData.from_tick(tick)
                self._cache[symbol] = agg
                return agg
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", symbol, exc)

        return AggregatedMarketData(symbol=symbol)

    def get_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        data = self.get_aggregated_data(symbol)
        return data.price

    def get_available_sources(self) -> List[str]:
        """Get available data sources."""
        return list(self.sources)

    def start(self) -> None:
        """Start the aggregator."""
        self._started = True
        logger.info("MarketDataAggregator started (sources=%s)", self.sources)

    def stop(self) -> None:
        """Stop the aggregator and clear cache."""
        self._cache.clear()
        self._started = False
        logger.info("MarketDataAggregator stopped")

    # ── Internal ──────────────────────────────────────────────────────

    async def _fetch_tick(self, symbol: str) -> Optional[MarketTick]:
        """Fetch a single tick from CoinGecko."""
        base = symbol.split("/")[0].upper()
        coin_id = _SYMBOL_MAP.get(base, base.lower())

        try:
            await self._client.connect()
            tick = await self._client.get_price(coin_id)
            return tick
        finally:
            try:
                await self._client.disconnect()
            except Exception:
                pass


def get_market_data_aggregator(**kwargs: Any) -> MarketDataAggregator:
    """Factory function for MarketDataAggregator."""
    return MarketDataAggregator(**kwargs)

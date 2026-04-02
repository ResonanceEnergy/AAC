#!/usr/bin/env python3
"""
Foreign Exchange Data Source
=============================
BaseDataSource implementation for live FX rate feeds.
Integrates with the Knightsbridge FX client to provide
real-time forex MarketTick data into AAC's DataAggregator.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from integrations.knightsbridge_fx_client import (
    CAD_PAIRS,
    CROSS_PAIRS,
    MAJOR_PAIRS,
    UYU_PAIRS,
    KnightsbridgeFXClient,
)
from shared.data_sources import BaseDataSource, MarketTick

logger = logging.getLogger(__name__)


class ForexDataSource(BaseDataSource):
    """
    Live FX rate data source backed by the Knightsbridge FX client.

    Polls ExchangeRate-API for spot mid-rates, applies institutional
    spread, and emits MarketTick objects for each currency pair.
    """

    def __init__(self):
        super().__init__("forex")
        self._client: Optional[KnightsbridgeFXClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        # Read spread config from env or default
        api_key = getattr(self.config, 'fx_api_key', '')
        spread = getattr(self.config, 'fx_spread_bps', 50.0)
        self._client = KnightsbridgeFXClient(
            api_key=api_key,
            spread_bps=spread,
        )

    async def connect(self):
        """Connect the FX client."""
        await self._client.connect()
        self.is_connected = True
        self.logger.info("Forex data source connected")

    async def disconnect(self):
        """Disconnect and cancel polling."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        await self._client.disconnect()
        self.is_connected = False

    async def get_fx_tick(self, from_ccy: str, to_ccy: str) -> Optional[MarketTick]:
        """Fetch a single FX pair and return as MarketTick."""
        rate = await self._client.get_pair(from_ccy, to_ccy)
        if rate is None:
            return None
        tick = MarketTick(
            symbol=rate.pair,
            price=rate.mid,
            volume_24h=0.0,  # FX spot — no volume from rate APIs
            change_24h=0.0,
            bid=rate.bid,
            ask=rate.ask,
            source=f"forex_{rate.source}",
        )
        await self._notify(tick)
        return tick

    async def get_all_rates(self, base: str = "USD") -> Dict[str, MarketTick]:
        """Fetch all rates for a base currency and return as MarketTick dict."""
        rates = await self._client.get_rates(base)
        ticks: Dict[str, MarketTick] = {}
        for pair_str, rate in rates.items():
            tick = MarketTick(
                symbol=rate.pair,
                price=rate.mid,
                volume_24h=0.0,
                change_24h=0.0,
                bid=rate.bid,
                ask=rate.ask,
                source=f"forex_{rate.source}",
            )
            ticks[pair_str] = tick
            await self._notify(tick)
        return ticks

    async def poll_rates(
        self,
        bases: Optional[List[str]] = None,
        interval: float = 60.0,
    ):
        """
        Background polling loop — fetches rates at the given interval.

        Default polls USD, CAD, EUR every 60 seconds.
        """
        if bases is None:
            bases = ["USD", "CAD", "EUR"]

        while True:
            try:
                for base in bases:
                    await self.get_all_rates(base)
                    await asyncio.sleep(1)  # stagger base requests
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"FX poll error: {exc}")
            await asyncio.sleep(interval)

    def start_polling(self, interval: float = 60.0):
        """Launch the polling loop as a background task."""
        self._poll_task = asyncio.ensure_future(self.poll_rates(interval=interval))

    async def find_triangular_arb(self, base: str = "USD", min_profit_bps: float = 5.0):
        """Proxy to the underlying client's triangular arb scanner."""
        return await self._client.find_triangular_arb(base, min_profit_bps)

    async def compare_spreads(self, pairs: Optional[List[str]] = None):
        """Proxy to the underlying client's spread comparison."""
        return await self._client.compare_spreads(pairs)

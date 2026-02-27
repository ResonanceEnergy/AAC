#!/usr/bin/env python3
"""
Market Data Aggregator — Stub Module
=====================================
Original module was lost during 2026-02-17 security scrub.
This stub provides the public API so dependent modules can import without error.
Real implementation should be restored from external backup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_STUB_WARNING = (
    "market_data_aggregator is a stub — real implementation pending restore"
)


@dataclass
class AggregatedMarketData:
    """Aggregated market data snapshot."""
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    source: str = "stub"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketDataAggregator:
    """Aggregates market data from multiple sources.

    Stub implementation — logs a warning on first use.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.sources: List[str] = kwargs.get("sources", [])
        self._warned = False

    def _warn_once(self) -> None:
        if not self._warned:
            logger.warning(_STUB_WARNING)
            self._warned = True

    def get_aggregated_data(self, symbol: str) -> AggregatedMarketData:
        self._warn_once()
        return AggregatedMarketData(symbol=symbol)

    def get_price(self, symbol: str) -> float:
        self._warn_once()
        return 0.0

    def get_available_sources(self) -> List[str]:
        return self.sources

    def start(self) -> None:
        self._warn_once()

    def stop(self) -> None:
        pass


def get_market_data_aggregator(**kwargs: Any) -> MarketDataAggregator:
    """Factory function for MarketDataAggregator."""
    return MarketDataAggregator(**kwargs)

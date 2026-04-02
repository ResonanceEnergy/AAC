"""Operational Unusual Whales snapshot service with cache and graceful fallback."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from integrations.unusual_whales_client import DarkPoolTrade, OptionsFlow, UnusualWhalesClient

logger = logging.getLogger("UnusualWhalesService")


@dataclass
class UnusualWhalesSnapshot:
    status: str
    as_of: str
    source: str = "unusual_whales"
    configured: bool = False
    put_call_ratio: float = 0.0
    market_tone: str = "unknown"
    options_flow_signal_count: int = 0
    total_options_premium: float = 0.0
    dark_pool_trade_count: int = 0
    dark_pool_notional: float = 0.0
    congress_trade_count: int = 0
    top_flow_tickers: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnusualWhalesSnapshotService:
    """Cached market-intelligence snapshot service for Unusual Whales."""

    def __init__(self, refresh_ttl_seconds: int = 300):
        self.refresh_ttl = timedelta(seconds=refresh_ttl_seconds)
        self._client = UnusualWhalesClient()
        self._snapshot = UnusualWhalesSnapshot(
            status="unconfigured" if not self._client.endpoint.enabled else "stale",
            as_of=datetime.now().isoformat(),
            configured=self._client.endpoint.enabled,
        )
        self._last_refresh: Optional[datetime] = None
        self._lock = asyncio.Lock()

    @property
    def client(self) -> UnusualWhalesClient:
        return self._client

    def get_cached_snapshot(self) -> Dict[str, Any]:
        return self._snapshot.to_dict()

    async def get_snapshot(self, force_refresh: bool = False) -> Dict[str, Any]:
        async with self._lock:
            if not self._client.endpoint.enabled:
                self._snapshot = UnusualWhalesSnapshot(
                    status="unconfigured",
                    as_of=datetime.now().isoformat(),
                    configured=False,
                    error="UNUSUAL_WHALES_API_KEY is not configured",
                )
                return self._snapshot.to_dict()

            now = datetime.now()
            if (
                not force_refresh
                and self._last_refresh is not None
                and now - self._last_refresh < self.refresh_ttl
            ):
                return self._snapshot.to_dict()

            self._snapshot = await self._fetch_snapshot()
            self._last_refresh = now
            return self._snapshot.to_dict()

    async def _fetch_snapshot(self) -> UnusualWhalesSnapshot:
        try:
            summary = await self._client.get_market_flow_summary()
            flow = await self._client.get_flow(limit=25, min_premium=0)
            dark_pool = await self._client.get_dark_pool(limit=25)
            congress = await self._client.get_congress_trades(limit=25)

            return self._build_snapshot(summary, flow, dark_pool, congress)
        except Exception as exc:
            logger.error("Unusual Whales snapshot refresh failed: %s", exc)
            return UnusualWhalesSnapshot(
                status="error",
                as_of=datetime.now().isoformat(),
                configured=self._client.endpoint.enabled,
                error=str(exc),
            )

    def _build_snapshot(
        self,
        summary: Dict[str, Any],
        flow: List[OptionsFlow],
        dark_pool: List[DarkPoolTrade],
        congress: List[Dict[str, Any]],
    ) -> UnusualWhalesSnapshot:
        top_flow_tickers = sorted(
            {item.ticker for item in flow if item.ticker},
            key=lambda ticker: sum(entry.premium for entry in flow if entry.ticker == ticker),
            reverse=True,
        )[:5]

        put_count = sum(1 for item in flow if item.option_type == "put")
        call_count = sum(1 for item in flow if item.option_type == "call")
        put_call_ratio = put_count / call_count if call_count else float(put_count)

        market_tone = "neutral"
        if put_call_ratio > 1.2:
            market_tone = "bearish"
        elif call_count > put_count and call_count > 0:
            market_tone = "bullish"

        summary_ratio = self._extract_float(summary, ["put_call_ratio", "putCallRatio", "pc_ratio"])
        if summary_ratio > 0:
            put_call_ratio = summary_ratio
            if put_call_ratio > 1.2:
                market_tone = "bearish"
            elif put_call_ratio < 0.9:
                market_tone = "bullish"

        return UnusualWhalesSnapshot(
            status="healthy",
            as_of=datetime.now().isoformat(),
            configured=True,
            put_call_ratio=put_call_ratio,
            market_tone=market_tone,
            options_flow_signal_count=len(flow),
            total_options_premium=sum(item.premium for item in flow),
            dark_pool_trade_count=len(dark_pool),
            dark_pool_notional=sum(item.notional for item in dark_pool),
            congress_trade_count=len(congress),
            top_flow_tickers=top_flow_tickers,
        )

    @staticmethod
    def _extract_float(data: Dict[str, Any], keys: List[str]) -> float:
        for key in keys:
            value = data.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0


_snapshot_service: Optional[UnusualWhalesSnapshotService] = None


def get_unusual_whales_snapshot_service() -> UnusualWhalesSnapshotService:
    """Get or create the shared Unusual Whales snapshot service."""
    global _snapshot_service
    if _snapshot_service is None:
        _snapshot_service = UnusualWhalesSnapshotService()
    return _snapshot_service

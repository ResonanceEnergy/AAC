"""Verify Unusual Whales Tier-1 fields propagate from snapshot to LiveFeedResult."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from strategies.war_room_live_feeds import LiveFeedResult, fetch_unusual_whales_data


@pytest.mark.asyncio
async def test_uw_tier1_fields_copied_to_live_feed_result() -> None:
    snapshot = {
        "status": "healthy",
        "put_call_ratio": 0.85,
        "market_tone": "neutral",
        "options_flow_signal_count": 12,
        "total_options_premium": 1_234_567.0,
        "dark_pool_trade_count": 3,
        "dark_pool_notional": 9_999.0,
        "congress_trade_count": 1,
        "top_flow_tickers": ["SPY", "QQQ"],
        "market_tide_latest": {"timestamp": "2026-05-14T12:00:00Z", "net_call_premium": 5.0},
        "market_tide_net_call_premium": 5.0,
        "market_tide_net_put_premium": 3.5,
        "iv_ranks": {"SPY": 42.1, "QQQ": 55.3},
        "gex_walls": {"SPY": [{"strike": 500.0, "gex": 1_000_000.0}]},
    }

    fake_service = type("FakeService", (), {"get_snapshot": AsyncMock(return_value=snapshot)})()

    result = LiveFeedResult()
    with patch(
        "integrations.unusual_whales_service.get_unusual_whales_snapshot_service",
        return_value=fake_service,
    ):
        await fetch_unusual_whales_data(result)

    assert result.put_call_ratio == 0.85
    assert result.market_tide_latest == snapshot["market_tide_latest"]
    assert result.market_tide_net_call_premium == 5.0
    assert result.market_tide_net_put_premium == 3.5
    assert result.iv_ranks == {"SPY": 42.1, "QQQ": 55.3}
    assert result.gex_walls == {"SPY": [{"strike": 500.0, "gex": 1_000_000.0}]}
    assert result.errors == []


@pytest.mark.asyncio
async def test_uw_tier1_defaults_when_snapshot_unconfigured() -> None:
    snapshot = {"status": "unconfigured"}
    fake_service = type("FakeService", (), {"get_snapshot": AsyncMock(return_value=snapshot)})()

    result = LiveFeedResult()
    with patch(
        "integrations.unusual_whales_service.get_unusual_whales_snapshot_service",
        return_value=fake_service,
    ):
        await fetch_unusual_whales_data(result)

    assert result.market_tide_latest is None
    assert result.market_tide_net_call_premium == 0.0
    assert result.market_tide_net_put_premium == 0.0
    assert result.iv_ranks == {}
    assert result.gex_walls == {}

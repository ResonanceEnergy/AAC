"""Tests for the Unusual Whales snapshot service and FFD integration hooks."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aac.doctrine.ffd.ffd_engine import FFDEngine
from integrations.unusual_whales_service import UnusualWhalesSnapshotService


@pytest.mark.asyncio
async def test_snapshot_service_unconfigured_returns_safe_status():
    service = UnusualWhalesSnapshotService()
    service.client.endpoint.enabled = False

    snapshot = await service.get_snapshot(force_refresh=True)

    assert snapshot["status"] == "unconfigured"
    assert snapshot["configured"] is False


@pytest.mark.asyncio
async def test_ffd_applies_market_intelligence_snapshot():
    engine = FFDEngine()
    engine._apply_market_intelligence(
        {
            "status": "healthy",
            "market_tone": "bearish",
            "put_call_ratio": 1.8,
            "options_flow_signal_count": 120,
        }
    )

    assert engine.metrics.options_flow_signal_count == 120
    assert engine.metrics.capital_flight_signal >= 55.0


def test_ffd_recommended_actions_reflect_state():
    engine = FFDEngine()
    engine.metrics.regulatory_shock_score = 65.0
    engine.metrics.capital_flight_signal = 52.0
    engine.metrics.options_flow_signal_count = 150

    actions = engine.get_recommended_actions()
    action_names = {item["action"] for item in actions}

    assert "enter_safe_mode" in action_names
    assert "throttle_risk" in action_names
    assert "increase_market_surveillance" in action_names


# ---------------------------------------------------------------------------
# Tier-1 field plumbing: market_tide, iv_ranks, gex_walls
# ---------------------------------------------------------------------------


def _make_service_with_mock_client(watchlist: list[str]) -> UnusualWhalesSnapshotService:
    """Build a configured service with an AsyncMock client and given watchlist."""
    service = UnusualWhalesSnapshotService()
    service._watchlist = list(watchlist)
    service._client.endpoint.enabled = True

    # Base endpoints: empty but well-typed so _build_snapshot does not crash.
    service._client.get_market_flow_summary = AsyncMock(return_value={})
    service._client.get_flow = AsyncMock(return_value=[])
    service._client.get_dark_pool = AsyncMock(return_value=[])
    service._client.get_congress_trades = AsyncMock(return_value=[])
    return service


@pytest.mark.asyncio
async def test_snapshot_populates_tier1_market_tide_and_iv_and_gex():
    service = _make_service_with_mock_client(["SPY"])

    service._client.get_market_tide = AsyncMock(
        return_value=[
            {"net_call_premium": 100.0, "net_put_premium": 50.0, "timestamp": "t0"},
            {"net_call_premium": 1500.0, "net_put_premium": 600.0, "timestamp": "t1"},
        ]
    )
    service._client.get_interpolated_iv = AsyncMock(return_value={"iv_rank": 42.5})
    service._client.get_spot_gex = AsyncMock(
        return_value=[
            {"strike": 500, "gamma": 1_000_000, "total_open_interest": 10},
            {"strike": 510, "gamma": -5_000_000, "total_open_interest": 50},
            {"strike": 520, "gamma": 2_000_000, "total_open_interest": 20},
            {"strike": 530, "gamma": 200, "total_open_interest": 5},
        ]
    )

    snap = await service.get_snapshot(force_refresh=True)

    assert snap["status"] == "healthy"
    assert snap["market_tide_net_call_premium"] == pytest.approx(1500.0)
    assert snap["market_tide_net_put_premium"] == pytest.approx(600.0)
    assert snap["market_tide_latest"]["timestamp"] == "t1"
    assert snap["iv_ranks"]["SPY"] == pytest.approx(42.5)
    walls = snap["gex_walls"]["SPY"]
    assert len(walls) == 3
    # Top wall must be the largest absolute gamma (-5,000,000).
    assert walls[0]["strike"] == 510


@pytest.mark.asyncio
async def test_snapshot_survives_tier1_endpoint_failures():
    service = _make_service_with_mock_client(["SPY", "QQQ"])

    service._client.get_market_tide = AsyncMock(
        side_effect=aiohttp.ClientError("tide down")
    )
    service._client.get_interpolated_iv = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )
    service._client.get_spot_gex = AsyncMock(
        side_effect=aiohttp.ClientError("gex down")
    )

    snap = await service.get_snapshot(force_refresh=True)

    assert snap["status"] == "healthy"
    assert snap["market_tide_latest"] is None
    assert snap["market_tide_net_call_premium"] == 0.0
    assert snap["market_tide_net_put_premium"] == 0.0
    assert snap["iv_ranks"] == {}
    assert snap["gex_walls"] == {}


@pytest.mark.asyncio
async def test_snapshot_respects_watchlist_cap_and_per_ticker_isolation():
    service = _make_service_with_mock_client(["AAA", "BBB", "CCC"])

    service._client.get_market_tide = AsyncMock(return_value=[])

    iv_calls: list[str] = []

    async def iv_side_effect(ticker: str):
        iv_calls.append(ticker)
        if ticker == "BBB":
            raise aiohttp.ClientError("iv failure for BBB")
        return {"iv_rank": 30.0 + len(iv_calls)}

    service._client.get_interpolated_iv = AsyncMock(side_effect=iv_side_effect)
    service._client.get_spot_gex = AsyncMock(return_value=[])

    snap = await service.get_snapshot(force_refresh=True)

    # All three watchlist entries attempted; failures isolated per ticker.
    assert iv_calls == ["AAA", "BBB", "CCC"]
    assert "AAA" in snap["iv_ranks"]
    assert "BBB" not in snap["iv_ranks"]
    assert "CCC" in snap["iv_ranks"]


def test_load_watchlist_returns_capped_list_or_default(tmp_path, monkeypatch):
    from integrations import unusual_whales_service as svc_mod

    # File missing → default.
    monkeypatch.setattr(svc_mod, "_WATCHLIST_PATH", tmp_path / "missing.yaml")
    assert svc_mod._load_watchlist() == ["SPY", "QQQ", "IWM"]

    # File present with > 5 tickers → capped at _MAX_WATCHLIST_TICKERS.
    yaml_path = tmp_path / "watchlist.yaml"
    yaml_path.write_text(
        "vol_premium:\n"
        "  - SPY\n  - QQQ\n  - IWM\n  - HYG\n  - JNK\n  - XLF\n  - KRE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(svc_mod, "_WATCHLIST_PATH", yaml_path)
    loaded = svc_mod._load_watchlist()
    assert len(loaded) == svc_mod._MAX_WATCHLIST_TICKERS
    assert loaded[0] == "SPY"

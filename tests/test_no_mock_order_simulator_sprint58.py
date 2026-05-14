"""Sprint 58 — OrderSimulator._get_simulated_price no longer uses hardcoded prices.

Source-level proof that the stale price table and random.uniform variation are
gone from core/aac_automation_engine.py.  Behavioural proof that the method
wires through MarketDataFeed (or raises RuntimeError when the feed has no data).
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SRC_ENGINE = Path("core/aac_automation_engine.py").read_text(encoding="utf-8")
_SRC_SIMULATOR = Path("PaperTradingDivision/order_simulator.py").read_text(encoding="utf-8")

# Combined source for shared assertions
_SRC = _SRC_ENGINE + _SRC_SIMULATOR

# ---------------------------------------------------------------------------
# Source-level assertions
# ---------------------------------------------------------------------------

def test_no_hardcoded_spy_price():
    """Old stale SPY:450.0 price table must be gone from the simulator source."""
    assert "'SPY': 450" not in _SRC_SIMULATOR, "Stale hardcoded SPY price still present in simulator"


def test_no_hardcoded_qqq_price():
    """Old stale QQQ:380 table must be gone from the simulator source."""
    assert "'QQQ': 380" not in _SRC_SIMULATOR, "Stale hardcoded QQQ price still present in simulator"


def test_no_random_uniform_variation():
    """random.uniform price variation must not exist in _fetch_price."""
    method_block = re.search(
        r"async def _fetch_price.*?(?=\n    @|\n    async def |\n    def |\Z)",
        _SRC_SIMULATOR,
        re.DOTALL,
    )
    assert method_block, "_fetch_price method not found in order_simulator.py"
    assert "random.uniform" not in method_block.group(), \
        "random.uniform still present in _fetch_price"


def test_get_market_data_feed_import_present():
    """_fetch_price must reference get_market_data_feed."""
    method_block = re.search(
        r"async def _fetch_price.*?(?=\n    @|\n    async def |\n    def |\Z)",
        _SRC_SIMULATOR,
        re.DOTALL,
    )
    assert method_block, "_fetch_price method not found in order_simulator.py"
    assert "get_market_data_feed" in method_block.group(), \
        "get_market_data_feed not referenced in _fetch_price"


def test_sprint58_marker_present():
    """Sprint 58 label must appear in order_simulator.py."""
    assert "Sprint 58" in _SRC_SIMULATOR


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_price_returns_real_price():
    """When the feed has data, _fetch_price returns the feed price."""
    from PaperTradingDivision.order_simulator import OrderSimulator

    mock_data = MagicMock()
    mock_data.price = 123.45

    mock_feed = MagicMock()
    mock_feed.get_latest_price = AsyncMock(return_value=mock_data)

    with patch(
        "shared.market_data_feeds.get_market_data_feed",
        new=AsyncMock(return_value=mock_feed),
    ):
        sim = OrderSimulator()
        price = await sim._fetch_price("SPY")

    assert price == pytest.approx(123.45)


@pytest.mark.asyncio
async def test_fetch_price_falls_back_to_base_prices_when_feed_unavailable():
    """When the feed raises, _fetch_price falls back to _BASE_PRICES (not random)."""
    from PaperTradingDivision.order_simulator import OrderSimulator, _BASE_PRICES

    with patch(
        "shared.market_data_feeds.get_market_data_feed",
        side_effect=RuntimeError("feed down"),
    ):
        sim = OrderSimulator()
        price = await sim._fetch_price("SPY")

    assert price == pytest.approx(_BASE_PRICES["SPY"])


@pytest.mark.asyncio
async def test_fetch_price_falls_back_for_unknown_symbol():
    """Unknown symbol with unavailable feed returns 100.0 fallback, not random."""
    from PaperTradingDivision.order_simulator import OrderSimulator

    with patch(
        "shared.market_data_feeds.get_market_data_feed",
        side_effect=RuntimeError("feed down"),
    ):
        sim = OrderSimulator()
        price = await sim._fetch_price("ZZZZ_NOTREAL")

    assert price == pytest.approx(100.0)

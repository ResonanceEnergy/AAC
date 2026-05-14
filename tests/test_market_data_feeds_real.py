"""Real-call tests for ``shared/market_data_feeds.py`` (Sprint 52).

NO MOCKS.  These tests hit live yfinance.  Marked ``api`` so they can be
deselected from the fast suite via ``-m "not api"``.

Validates the post-rewrite contract:
* ``get_latest_price`` returns a populated ``MarketData`` for SPY.
* ``get_order_book`` returns ``None`` (no L2 source available).
* ``get_historical_prices`` returns a non-empty pandas DataFrame.
* ``get_market_sentiment`` returns real momentum/volatility numbers.
* ``get_options_data`` returns a non-empty calls/puts dict for SPY.
* ``get_market_data_feed`` singleton round-trips.
* Bad symbol returns ``None`` (no fabricated price).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from shared import market_data_feeds as mdf


@pytest.fixture(autouse=True)
def _reset_singleton():
    mdf._reset_singleton_for_tests()
    yield
    mdf._reset_singleton_for_tests()


@pytest.mark.api
class TestRealPrice:
    async def test_get_latest_price_spy_returns_real_data(self):
        feed = mdf.MarketDataFeed()
        data = await feed.get_latest_price("SPY")
        assert data is not None, "yfinance returned no data for SPY"
        assert isinstance(data, mdf.MarketData)
        assert data.symbol == "SPY"
        assert data.price > 0
        assert data.bid > 0
        assert data.ask >= data.bid
        assert data.metadata.get("source") == "yfinance"
        assert isinstance(data.timestamp, datetime)

    async def test_cache_returns_same_instance_within_ttl(self):
        feed = mdf.MarketDataFeed()
        first = await feed.get_latest_price("SPY")
        second = await feed.get_latest_price("SPY")
        assert first is second, "second call within TTL should hit cache"

    async def test_unknown_symbol_returns_none_not_simulated(self):
        feed = mdf.MarketDataFeed()
        data = await feed.get_latest_price("ZZZZNONEXISTENT123XYZ")
        assert data is None, "unknown symbol must NOT return fabricated data"


@pytest.mark.api
class TestRealOrderBook:
    async def test_order_book_returns_none(self):
        """yfinance has no L2 -- must honestly return None, never fabricated."""
        feed = mdf.MarketDataFeed()
        book = await feed.get_order_book("SPY")
        assert book is None


@pytest.mark.api
class TestRealHistorical:
    async def test_historical_returns_real_dataframe(self):
        feed = mdf.MarketDataFeed()
        df = await feed.get_historical_prices("SPY", days=7)
        assert df is not None, "yfinance returned no history for SPY"
        assert len(df) > 0
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"missing {col}"
        assert (df["close"] > 0).all()


@pytest.mark.api
class TestRealSentiment:
    async def test_sentiment_uses_real_returns(self):
        feed = mdf.MarketDataFeed()
        sent = await feed.get_market_sentiment("SPY")
        assert sent is not None
        for key in ("momentum_1d", "momentum_5d", "volatility_5d", "volume_trend"):
            assert key in sent
            assert isinstance(sent[key], float)
        # volatility must be non-negative
        assert sent["volatility_5d"] >= 0.0


@pytest.mark.api
class TestRealOptions:
    async def test_options_data_real_chain(self):
        feed = mdf.MarketDataFeed()
        chain = await feed.get_options_data("SPY")
        assert chain is not None
        assert "calls" in chain and "puts" in chain
        assert len(chain["calls"]) > 0
        assert len(chain["puts"]) > 0
        assert chain.get("expiry"), "expiry must be a real date string"
        # spot may be None if quote fetch fails -- acceptable, but if present must be > 0
        spot = chain.get("spot_price")
        if spot is not None:
            assert spot > 0


@pytest.mark.api
class TestSingleton:
    async def test_singleton_creates_and_initializes(self):
        feed = await mdf.get_market_data_feed()
        try:
            assert isinstance(feed, mdf.MarketDataFeed)
            again = await mdf.get_market_data_feed()
            assert feed is again
        finally:
            await feed.close()


@pytest.mark.api
class TestSubscriptions:
    async def test_subscribe_unsubscribe_roundtrip(self):
        feed = mdf.MarketDataFeed()
        received: list = []

        async def cb(data):
            received.append(data)

        await feed.subscribe_to_price_updates("SPY", cb)
        assert "SPY" in feed.subscriptions
        await feed.unsubscribe_from_price_updates("SPY", cb)
        assert "SPY" not in feed.subscriptions


class TestNoMockArtifacts:
    """Pure (non-network) check that the rewrite removed all simulated paths."""

    def test_module_has_no_simulated_methods(self):
        for name in (
            "_initialize_simulated_data",
            "_get_simulated_price_data",
            "simulated_prices",
            "_run_mock_simulation",
        ):
            assert not hasattr(mdf.MarketDataFeed, name), (
                f"MarketDataFeed must not expose {name} after Sprint 52 rewrite"
            )

    def test_module_does_not_import_random(self):
        import importlib
        import sys

        importlib.reload(mdf)
        # Module must not pull `random` -- proxy for "no simulation".
        # (The rewritten file does not import random at all.)
        src = open(mdf.__file__, encoding="utf-8").read()
        assert "import random" not in src
        assert "random." not in src


def test_smoke_can_construct_feed_synchronously():
    """Construct without initializing -- no event loop required."""
    feed = mdf.MarketDataFeed()
    assert feed.cache_ttl_seconds == 5
    assert feed.subscriptions == {}

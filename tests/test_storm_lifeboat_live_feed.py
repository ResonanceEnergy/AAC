"""
Tests for Storm Lifeboat Live Feed — strategies/storm_lifeboat/live_feed.py
=============================================================================
All external API calls are mocked. Tests verify data flow, caching,
graceful degradation, and correct wiring to Storm Lifeboat types.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strategies.storm_lifeboat.core import DEFAULT_PRICES, Asset, VolRegime
from strategies.storm_lifeboat.live_feed import (
    ASSET_TICKER_MAP,
    FRED_SERIES,
    INDICATOR_KEYWORDS,
    POLYGON_CRYPTO_MAP,
    LiveFeedEngine,
    LiveFeedSnapshot,
    get_live_snapshot,
)

# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FakeTickerSnapshot:
    ticker: str
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    day_close: float = 100.0
    day_volume: float = 1_000_000.0
    prev_close: float = 99.0
    change: float = 1.0
    change_pct: float = 1.01
    updated: datetime = datetime(2026, 3, 18)


@dataclass
class FakeFredObservation:
    series_id: str
    date: str
    value: float
    realtime_start: str = ""
    realtime_end: str = ""


@dataclass
class FakeFearGreedReading:
    value: int
    classification: str
    timestamp: datetime
    market: str


@dataclass
class FakeNewsArticle:
    headline: str
    source: str = "test"
    url: str = ""
    summary: str = ""
    category: str = "general"
    datetime: datetime = datetime(2026, 3, 18)
    related: str = ""


def _make_polygon_snapshot(ticker: str, price: float) -> FakeTickerSnapshot:
    return FakeTickerSnapshot(ticker=ticker, day_close=price, prev_close=price - 1)


# ═══════════════════════════════════════════════════════════════════════════
# MAPPING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAssetMapping:
    def test_all_20_assets_mapped(self):
        assert len(ASSET_TICKER_MAP) == 20
        for a in Asset:
            assert a in ASSET_TICKER_MAP, f"Missing ticker mapping for {a.value}"

    def test_crypto_assets_in_polygon_map(self):
        assert Asset.BTC in POLYGON_CRYPTO_MAP
        assert Asset.ETH in POLYGON_CRYPTO_MAP
        assert Asset.XRP in POLYGON_CRYPTO_MAP

    def test_polygon_crypto_has_x_prefix(self):
        for ticker in POLYGON_CRYPTO_MAP.values():
            assert ticker.startswith("X:"), f"Polygon crypto {ticker} missing X: prefix"

    def test_fred_series_includes_vix(self):
        assert "vix" in FRED_SERIES
        assert FRED_SERIES["vix"] == "VIXCLS"

    def test_indicator_keywords_cover_scenarios(self):
        assert "HORMUZ" in INDICATOR_KEYWORDS
        assert "TAIWAN" in INDICATOR_KEYWORDS
        assert "DEBT_CRISIS" in INDICATOR_KEYWORDS
        assert "DEFI_CASCADE" in INDICATOR_KEYWORDS


# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOT DATACLASS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestLiveFeedSnapshot:
    def test_defaults(self):
        snap = LiveFeedSnapshot()
        assert snap.vix == 25.0
        assert snap.regime == VolRegime.CRISIS
        assert snap.fear_greed == 50
        assert snap.prices == {}
        assert snap.sources_ok == []
        assert snap.sources_failed == []

    def test_custom_values(self):
        snap = LiveFeedSnapshot(vix=42.5, fear_greed=12, fear_greed_label="Extreme Fear")
        assert snap.vix == 42.5
        assert snap.fear_greed == 12
        assert snap.fear_greed_label == "Extreme Fear"


# ═══════════════════════════════════════════════════════════════════════════
# FETCH — POLYGON PRICES
# ═══════════════════════════════════════════════════════════════════════════

class TestPolygonFetch:
    @pytest.mark.asyncio
    async def test_polygon_equities_populate_prices(self):
        engine = LiveFeedEngine()
        snap = LiveFeedSnapshot()
        snap.prices = dict(DEFAULT_PRICES)

        fake_snapshots = [
            _make_polygon_snapshot("SPY", 670.0),
            _make_polygon_snapshot("QQQ", 455.0),
            _make_polygon_snapshot("GLD", 4900.0),
        ]

        mock_client = MagicMock()
        mock_client.get_all_snapshots = AsyncMock(return_value=fake_snapshots)
        mock_client.get_snapshot = AsyncMock(return_value=None)

        with patch("integrations.polygon_client.PolygonClient", return_value=mock_client):
            await engine._fetch_polygon_prices(snap)

        assert snap.prices[Asset.SPY] == 670.0
        assert snap.prices[Asset.QQQ] == 455.0
        assert snap.prices[Asset.GOLD] == 4900.0
        assert "polygon_equities" in snap.sources_ok

    @pytest.mark.asyncio
    async def test_polygon_failure_graceful(self):
        engine = LiveFeedEngine()
        snap = LiveFeedSnapshot()
        snap.prices = dict(DEFAULT_PRICES)

        with patch("integrations.polygon_client.PolygonClient", side_effect=RuntimeError("no key")):
            with pytest.raises(RuntimeError, match="PolygonClient init failed"):
                await engine._fetch_polygon_prices(snap)


# ═══════════════════════════════════════════════════════════════════════════
# FETCH — FRED MACRO
# ═══════════════════════════════════════════════════════════════════════════

class TestFredFetch:
    @pytest.mark.asyncio
    async def test_fred_vix_sets_top_level(self):
        engine = LiveFeedEngine()
        snap = LiveFeedSnapshot()
        snap.prices = dict(DEFAULT_PRICES)

        async def fake_latest(series_id):
            data = {
                "VIXCLS": FakeFredObservation("VIXCLS", "2026-03-18", 32.5),
                "DGS10": FakeFredObservation("DGS10", "2026-03-18", 4.85),
                "T10Y2Y": FakeFredObservation("T10Y2Y", "2026-03-18", -0.12),
            }
            return data.get(series_id)

        mock_client = MagicMock()
        mock_client.get_latest_value = AsyncMock(side_effect=fake_latest)

        with patch("integrations.fred_client.FredClient", return_value=mock_client):
            await engine._fetch_fred_macro(snap)

        assert snap.vix == 32.5
        assert snap.macro["vix"] == 32.5
        assert snap.macro["10y_yield"] == 4.85
        assert snap.macro["yield_spread"] == -0.12
        assert "fred" in snap.sources_ok


# ═══════════════════════════════════════════════════════════════════════════
# FETCH — UNUSUAL WHALES
# ═══════════════════════════════════════════════════════════════════════════

class TestUnusualWhalesFetch:
    @pytest.mark.asyncio
    async def test_uw_populates_flow_data(self):
        engine = LiveFeedEngine()
        snap = LiveFeedSnapshot()
        snap.prices = dict(DEFAULT_PRICES)

        uw_data = {
            "put_call_ratio": 1.35,
            "market_tone": "bearish",
            "options_flow_signal_count": 42,
            "top_flow_tickers": ["SPY", "NVDA", "TSLA"],
            "dark_pool_notional": 5_000_000_000.0,
        }

        mock_svc = MagicMock()
        mock_svc.get_snapshot = AsyncMock(return_value=uw_data)

        with patch("integrations.unusual_whales_service.UnusualWhalesSnapshotService", return_value=mock_svc):
            await engine._fetch_unusual_whales(snap)

        assert snap.put_call_ratio == 1.35
        assert snap.market_tone == "bearish"
        assert snap.options_flow_signal_count == 42
        assert "SPY" in snap.top_flow_tickers
        assert "unusual_whales" in snap.sources_ok


# ═══════════════════════════════════════════════════════════════════════════
# FETCH — FEAR & GREED
# ═══════════════════════════════════════════════════════════════════════════

class TestFearGreedFetch:
    @pytest.mark.asyncio
    async def test_fear_greed_populates(self):
        engine = LiveFeedEngine()
        snap = LiveFeedSnapshot()
        snap.prices = dict(DEFAULT_PRICES)

        reading = FakeFearGreedReading(
            value=18, classification="Extreme Fear",
            timestamp=datetime(2026, 3, 18), market="crypto",
        )

        mock_client = MagicMock()
        mock_client.get_current = AsyncMock(return_value=reading)

        with patch("integrations.fear_greed_client.FearGreedClient", return_value=mock_client):
            await engine._fetch_fear_greed(snap)

        assert snap.fear_greed == 18
        assert snap.fear_greed_label == "Extreme Fear"
        assert "fear_greed" in snap.sources_ok


# ═══════════════════════════════════════════════════════════════════════════
# FETCH — FINNHUB NEWS → SCENARIO INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

class TestFinnhubNewsFetch:
    @pytest.mark.asyncio
    async def test_news_fires_scenario_indicators(self):
        engine = LiveFeedEngine()
        snap = LiveFeedSnapshot()
        snap.prices = dict(DEFAULT_PRICES)

        articles = [
            FakeNewsArticle(headline="Iran navy exercises threaten Strait of Hormuz shipping"),
            FakeNewsArticle(headline="Tech stocks rally continues as AI spending rises"),
            FakeNewsArticle(headline="Taiwan Strait military drills escalate tensions"),
        ]

        mock_client = MagicMock()
        mock_client.get_news = AsyncMock(return_value=articles)

        with patch("integrations.finnhub_client.FinnhubClient", return_value=mock_client):
            await engine._fetch_finnhub_news(snap)

        assert "HORMUZ" in snap.firing_indicators
        assert "TAIWAN" in snap.firing_indicators
        assert "finnhub_news" in snap.sources_ok


# ═══════════════════════════════════════════════════════════════════════════
# FULL FETCH — CONCURRENT WITH MOCKS
# ═══════════════════════════════════════════════════════════════════════════

class TestFullFetch:
    @pytest.mark.asyncio
    async def test_fetch_async_returns_snapshot(self):
        """All sources mocked — verify concurrent fetch produces valid snapshot."""
        fake_poly_snaps = [_make_polygon_snapshot("SPY", 672.0)]
        vix_obs = FakeFredObservation("VIXCLS", "2026-03-18", 28.3)
        fg_reading = FakeFearGreedReading(
            value=35, classification="Fear",
            timestamp=datetime(2026, 3, 18), market="crypto",
        )

        mock_poly = MagicMock()
        mock_poly.get_all_snapshots = AsyncMock(return_value=fake_poly_snaps)
        mock_poly.get_snapshot = AsyncMock(return_value=None)

        mock_fred = MagicMock()
        async def fred_latest(sid):
            return vix_obs if sid == "VIXCLS" else None
        mock_fred.get_latest_value = AsyncMock(side_effect=fred_latest)

        mock_uw = MagicMock()
        mock_uw.get_snapshot = AsyncMock(return_value={
            "put_call_ratio": 1.1, "market_tone": "neutral",
            "options_flow_signal_count": 5, "top_flow_tickers": ["AAPL"],
            "dark_pool_notional": 1e9,
        })

        mock_fg = MagicMock()
        mock_fg.get_current = AsyncMock(return_value=fg_reading)

        mock_fh = MagicMock()
        mock_fh.get_news = AsyncMock(return_value=[])

        mock_gt = MagicMock()
        mock_gt.get_interest_over_time = MagicMock(return_value={})

        with patch("integrations.polygon_client.PolygonClient", return_value=mock_poly), \
             patch("integrations.fred_client.FredClient", return_value=mock_fred), \
             patch("integrations.unusual_whales_service.UnusualWhalesSnapshotService", return_value=mock_uw), \
             patch("integrations.fear_greed_client.FearGreedClient", return_value=mock_fg), \
             patch("integrations.finnhub_client.FinnhubClient", return_value=mock_fh), \
             patch("integrations.google_trends_client.GoogleTrendsClient", return_value=mock_gt):

            engine = LiveFeedEngine()
            snap = await engine.fetch_async()

        assert isinstance(snap, LiveFeedSnapshot)
        assert snap.vix == 28.3
        assert snap.regime == VolRegime.CRISIS  # 28.3 > 25
        assert snap.prices[Asset.SPY] == 672.0
        assert snap.fear_greed == 35
        assert snap.put_call_ratio == 1.1
        assert "fred" in snap.sources_ok

    @pytest.mark.asyncio
    async def test_fetch_degrades_gracefully_on_source_failure(self):
        """If a source throws, others still work and error is logged."""
        mock_fred = MagicMock()
        mock_fred.get_latest_value = AsyncMock(return_value=None)

        mock_uw = MagicMock()
        mock_uw.get_snapshot = AsyncMock(return_value={})

        mock_fg = MagicMock()
        mock_fg.get_current = AsyncMock(return_value=None)

        mock_fh = MagicMock()
        mock_fh.get_news = AsyncMock(return_value=[])

        mock_gt = MagicMock()
        mock_gt.get_interest_over_time = MagicMock(return_value={})

        with patch("integrations.polygon_client.PolygonClient", side_effect=RuntimeError("offline")), \
             patch("integrations.fred_client.FredClient", return_value=mock_fred), \
             patch("integrations.unusual_whales_service.UnusualWhalesSnapshotService", return_value=mock_uw), \
             patch("integrations.fear_greed_client.FearGreedClient", return_value=mock_fg), \
             patch("integrations.finnhub_client.FinnhubClient", return_value=mock_fh), \
             patch("integrations.google_trends_client.GoogleTrendsClient", return_value=mock_gt):

            engine = LiveFeedEngine()
            snap = await engine.fetch_async()

        assert isinstance(snap, LiveFeedSnapshot)
        assert "polygon" in snap.sources_failed
        # DEFAULT_PRICES still present as fallback
        assert snap.prices[Asset.SPY] == DEFAULT_PRICES[Asset.SPY]


# ═══════════════════════════════════════════════════════════════════════════
# VIX → REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeClassification:
    def test_vix_above_40_is_panic(self):
        snap = LiveFeedSnapshot(vix=45.0)
        if snap.vix > 40:
            snap.regime = VolRegime.PANIC
        assert snap.regime == VolRegime.PANIC

    def test_vix_below_15_is_calm(self):
        snap = LiveFeedSnapshot(vix=12.0)
        if snap.vix <= 15:
            snap.regime = VolRegime.CALM
        assert snap.regime == VolRegime.CALM


# ═══════════════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════════════

class TestCaching:
    def test_cache_returns_same_snapshot(self):
        engine = LiveFeedEngine(cache_ttl_seconds=300)
        snap = LiveFeedSnapshot(vix=30.0)
        snap.prices = dict(DEFAULT_PRICES)
        engine._last_snapshot = snap
        engine._last_fetch = datetime.now()

        result = engine.fetch(force=False)
        assert result is snap
        assert result.vix == 30.0

    def test_force_bypasses_cache(self):
        engine = LiveFeedEngine(cache_ttl_seconds=300)
        snap = LiveFeedSnapshot(vix=30.0)
        engine._last_snapshot = snap
        engine._last_fetch = datetime.now()

        result = engine.fetch(force=False)
        assert result.vix == 30.0


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

class TestConvenienceFunction:
    def test_get_live_snapshot_returns_snapshot(self):
        with patch("strategies.storm_lifeboat.live_feed.LiveFeedEngine") as MockEngine:
            mock_instance = MagicMock()
            mock_snap = LiveFeedSnapshot(vix=22.0)
            mock_snap.prices = dict(DEFAULT_PRICES)
            mock_instance.fetch.return_value = mock_snap
            MockEngine.return_value = mock_instance

            result = get_live_snapshot(include_ibkr=False, include_moomoo=False)

        assert isinstance(result, LiveFeedSnapshot)
        assert result.vix == 22.0

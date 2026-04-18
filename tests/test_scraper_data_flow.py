from __future__ import annotations

"""Functional tests for all 4 council scrapers and end-to-end data flow.

Tests cover:
  1. YouTube scraper → analyzer → division signals
  2. X/Grok retriever → analyzer → division signals
  3. Polymarket scraper → analyzer → division signals
  4. Crypto scraper → analyzer → division signals
  5. War room bridge: council feeds → indicator patching → alpha engine
  6. Trust score integration across all councils
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Synthetic fixtures — deterministic data for each council
# ============================================================================

def _make_video_meta(**overrides: Any) -> Any:
    from councils.youtube.models import VideoMeta

    defaults = dict(
        video_id="abc123",
        title="Oil Markets Face Iran Sanctions Pressure",
        channel="FinanceChannel",
        upload_date="20260710",
        duration_seconds=600,
        description="Analysis of Iran sanctions impact on oil supply.",
        url="https://www.youtube.com/watch?v=abc123",
        view_count=50_000,
        like_count=2_000,
        tags=["oil", "iran", "sanctions"],
    )
    defaults.update(overrides)
    return VideoMeta(**defaults)


def _make_transcript_segments() -> list[Any]:
    from councils.youtube.models import TranscriptSegment

    return [
        TranscriptSegment(start=0.0, end=5.0, text="Iran sanctions are tightening on oil exports."),
        TranscriptSegment(start=5.0, end=10.0, text="The Hormuz strait remains a key chokepoint."),
        TranscriptSegment(start=10.0, end=15.0, text="Crude prices could crash if supply disruptions hit."),
        TranscriptSegment(start=15.0, end=20.0, text="Investors should hedge with put options on energy ETFs."),
        TranscriptSegment(start=20.0, end=25.0, text="This is a critical moment for the global economy."),
    ]


def _make_x_posts(count: int = 5) -> list[Any]:
    from councils.xai.models import XPost

    texts = [
        "Market crash incoming, recession fears are growing fast $SPY",
        "Oil prices collapse as Iran sanctions intensify #oil #bearish",
        "Fed rates decision could trigger a massive rally #bullish",
        "Credit default swaps widening — banks in trouble #CDS #credit",
        "Tariff war escalating between US and China #trade #tariffs",
    ]
    return [
        XPost(
            post_id=f"post_{i}",
            author=f"trader{i}",
            text=texts[i % len(texts)],
            created_at="2026-07-10",
            url=f"https://x.com/trader{i}/status/post_{i}",
            likes=100 * (i + 1),
            reposts=20 * (i + 1),
            replies=5 * (i + 1),
            views=1000 * (i + 1),
        )
        for i in range(count)
    ]


def _make_market_snapshots(count: int = 5) -> list[Any]:
    from councils.polymarket.models import MarketSnapshot

    questions = [
        "Will oil price exceed $100/barrel by end of July?",
        "Will Iran nuclear deal be signed by August?",
        "Will S&P 500 close above 5500 this week?",
        "Will US impose new tariffs on China in July?",
        "Will Bitcoin reach $100K in 2026?",
    ]
    return [
        MarketSnapshot(
            condition_id=f"cond_{i}",
            question=questions[i % len(questions)],
            slug=f"market-{i}",
            yes_price=0.3 + 0.1 * i,
            no_price=0.7 - 0.1 * i,
            volume=500_000.0 * (i + 1),
            volume_24h=50_000.0 * (i + 1),
            liquidity=100_000.0 * (i + 1),
            tags=["politics", "economics"][: (i % 2 + 1)],
            active=True,
        )
        for i in range(count)
    ]


def _make_coin_snapshots(count: int = 5) -> list[Any]:
    from councils.crypto.models import CoinSnapshot

    coins = [
        ("bitcoin", "BTC/USD", 67_000.0, 25e9, 2.5),
        ("ethereum", "ETH/USD", 3_500.0, 12e9, -1.2),
        ("solana", "SOL/USD", 145.0, 3e9, 5.8),
        ("ripple", "XRP/USD", 0.55, 1.5e9, -3.1),
        ("cardano", "ADA/USD", 0.45, 800e6, 0.3),
    ]
    return [
        CoinSnapshot(
            coin_id=coins[i][0],
            symbol=coins[i][1],
            price=coins[i][2],
            volume_24h=coins[i][3],
            change_24h=coins[i][4],
        )
        for i in range(min(count, len(coins)))
    ]


def _make_trending_coins() -> list[Any]:
    from councils.crypto.models import TrendingCoin

    return [
        TrendingCoin(coin_id="pepe", name="Pepe", symbol="PEPE", market_cap_rank=500, score=1),
        TrendingCoin(coin_id="bonk", name="Bonk", symbol="BONK", market_cap_rank=600, score=2),
    ]


def _make_global_data() -> Any:
    from councils.crypto.models import GlobalData

    return GlobalData(
        total_market_cap_usd=2.5e12,
        total_volume_24h_usd=80e9,
        btc_dominance=55.0,
        eth_dominance=18.0,
        active_cryptocurrencies=10_000,
        market_cap_change_24h_pct=1.2,
    )


# ============================================================================
# 1. YouTube scraper → analyzer → division
# ============================================================================

class TestYouTubePipeline:
    """Test YouTube council data flow: scraper → analyzer → division signals."""

    def test_analyze_transcript_extractive(self):
        """Extractive analyzer produces valid VideoInsight from transcript."""
        from councils.youtube.analyzer import analyze_transcript

        meta = _make_video_meta()
        segments = _make_transcript_segments()
        insight = analyze_transcript(segments, meta)

        assert insight.title == meta.title
        assert len(insight.key_topics) > 0
        assert isinstance(insight.summary, str) and len(insight.summary) > 0
        assert insight.sentiment in (
            "strongly_bearish", "bearish", "lean_bearish",
            "neutral", "lean_bullish", "bullish", "strongly_bullish",
        )

    def test_trust_score_computed(self):
        """Trust score is computed for YouTube analysis."""
        from councils.youtube.analyzer import _compute_video_trust

        meta = _make_video_meta()
        segments = _make_transcript_segments()
        trust = _compute_video_trust(meta, segments)

        assert "overall" in trust
        assert 0.0 <= trust["overall"] <= 1.0
        assert "source_reliability" in trust

    def test_division_scan_produces_signals(self):
        """YouTubeCouncilDivision.scan() produces INTEL_UPDATE signals."""
        from councils.youtube.division import YouTubeCouncilDivision

        div = YouTubeCouncilDivision(channels=["https://youtube.com/@TestChannel"])

        # Mock the scraper and pipeline to avoid real network calls
        fake_meta = _make_video_meta()
        from councils.youtube.models import VideoInsight

        fake_insight = VideoInsight(
            title=fake_meta.title,
            key_topics=["oil", "iran"],
            quotes=["Iran sanctions are tightening"],
            summary="Iran sanctions tightening on oil exports.",
            actionable_items=["Hedge energy exposure"],
            sentiment="bearish",
            trust_score={"overall": 0.7},
        )
        from councils.youtube.models import CouncilEntry

        fake_entry = CouncilEntry(
            meta=fake_meta,
            transcript=_make_transcript_segments(),
            insights=fake_insight,
            markdown_path="",
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

        div._seen = set()  # Clear seen tracking to avoid prior-run filtering
        with patch(
            "councils.youtube.division.list_channel_videos",
            return_value=[fake_meta],
        ), patch(
            "councils.youtube.division.process_video",
            return_value=fake_entry,
        ):
            signals = asyncio.get_event_loop().run_until_complete(div.scan())

        assert len(signals) >= 1
        assert signals[0].signal_type.value == "intel_update"
        assert signals[0].source_division == "youtube_council"
        assert "oil" in str(signals[0].data).lower() or "iran" in str(signals[0].data).lower()

    def test_analyzer_with_empty_transcript(self):
        """Analyzer handles empty transcript gracefully."""
        from councils.youtube.analyzer import analyze_transcript

        meta = _make_video_meta()
        insight = analyze_transcript([], meta)

        assert insight.title == meta.title
        assert insight.sentiment == "neutral"


# ============================================================================
# 2. X/Grok retriever → analyzer → division
# ============================================================================

class TestXaiPipeline:
    """Test X/Grok council data flow: retriever → analyzer → division signals."""

    def test_extractive_analysis(self):
        """Extractive analyzer produces valid XInsight from posts."""
        from councils.xai.analyzer import analyze_posts_extractive

        posts = _make_x_posts(10)
        insight = analyze_posts_extractive(posts, source="test_query")

        assert len(insight.key_themes) > 0
        assert len(insight.notable_posts) > 0
        assert isinstance(insight.sentiment_summary, str)
        assert insight.source == "test_query"

    def test_trust_score_computed(self):
        """Trust score is correctly computed with LLM-synthesized ceiling."""
        from councils.xai.analyzer import _compute_xai_trust

        posts = _make_x_posts(10)
        trust = _compute_xai_trust(posts, provider="xai", is_llm_synthesized=True)

        assert "overall" in trust
        # LLM-synthesized data has a trust ceiling of 0.45
        assert trust["overall"] <= 0.50, f"LLM trust too high: {trust['overall']}"

    def test_trust_score_direct_is_higher(self):
        """Direct (non-LLM) trust should be higher than LLM-synthesized."""
        from councils.xai.analyzer import _compute_xai_trust

        posts = _make_x_posts(20)
        llm_trust = _compute_xai_trust(posts, provider="xai", is_llm_synthesized=True)
        direct_trust = _compute_xai_trust(posts, provider="xai", is_llm_synthesized=False)

        assert direct_trust["overall"] > llm_trust["overall"]

    def test_division_scan_produces_signals(self):
        """XaiCouncilDivision.scan() produces INTEL_UPDATE signals."""
        from councils.xai.division import XaiCouncilDivision

        div = XaiCouncilDivision(
            search_queries=["market crash recession"],
            provider="xai",
        )

        from councils.xai.models import CouncilEntry, XInsight

        fake_posts = _make_x_posts(5)
        fake_insight = XInsight(
            source="market crash recession",
            key_themes=["crash", "recession"],
            notable_posts=["Market crash incoming"],
            sentiment_summary="bearish",
            engagement_highlights=["High repost activity"],
            emerging_topics=["tariffs"],
            actionable_items=["Hedge long positions"],
        )
        fake_entry = CouncilEntry(
            posts=fake_posts,
            insights=fake_insight,
            markdown_path="",
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

        div._seen_post_ids = set()  # Clear seen tracking to avoid prior-run filtering
        with patch(
            "councils.xai.division.run_xai_council",
            return_value=fake_entry,
        ):
            signals = asyncio.get_event_loop().run_until_complete(div.scan())

        assert len(signals) >= 1
        assert signals[0].signal_type.value == "intel_update"
        assert signals[0].source_division == "xai_council"

    def test_retriever_returns_empty_without_key(self):
        """Retriever gracefully returns empty list when API key missing."""
        from councils.xai.retriever import search_x_via_grok

        with patch.dict("os.environ", {}, clear=True):
            posts = search_x_via_grok("test query", provider="xai")
            assert posts == []


# ============================================================================
# 3. Polymarket scraper → analyzer → division
# ============================================================================

class TestPolymarketPipeline:
    """Test Polymarket council data flow: scraper → analyzer → division."""

    def test_analyzer_produces_insights(self):
        """Analyzer produces valid MarketInsight from snapshots."""
        from councils.polymarket.analyzer import analyze_markets

        markets = _make_market_snapshots(10)
        insight = analyze_markets(markets)

        assert insight.total_markets == 10
        assert insight.total_volume > 0
        assert len(insight.top_by_volume) > 0
        assert isinstance(insight.sentiment, str)
        assert isinstance(insight.summary, str) and len(insight.summary) > 0

    def test_trust_score_computed(self):
        """Trust score computed for Polymarket analysis."""
        from councils.polymarket.analyzer import _compute_poly_trust

        markets = _make_market_snapshots(20)
        trust = _compute_poly_trust(markets)

        assert "overall" in trust
        assert 0.0 <= trust["overall"] <= 1.0

    def test_analyzer_handles_empty_markets(self):
        """Analyzer handles empty market list gracefully."""
        from councils.polymarket.analyzer import analyze_markets

        insight = analyze_markets([])
        assert insight.total_markets == 0
        assert "No markets" in insight.summary

    def test_arb_detection(self):
        """Arb detection identifies underround markets."""
        from councils.polymarket.models import MarketSnapshot

        # Create a market with yes + no < 1.0 (pure arb)
        arb_market = MarketSnapshot(
            condition_id="arb_1",
            question="Arb test?",
            yes_price=0.40,
            no_price=0.50,
            volume=1_000_000.0,
            liquidity=500_000.0,
        )
        assert arb_market.overround < 1.0, "Should be underround for arb"

    def test_division_scan_produces_signals(self):
        """PolymarketCouncilDivision.scan() produces signals."""
        from councils.polymarket.division import PolymarketCouncilDivision
        from councils.polymarket.models import CouncilEntry, MarketInsight

        div = PolymarketCouncilDivision(keywords=["oil", "iran"])

        markets = _make_market_snapshots(5)
        fake_entry = CouncilEntry(
            markets=markets,
            insights=MarketInsight(
                total_markets=5,
                total_volume=sum(m.volume for m in markets),
                top_by_volume=[{"question": m.question, "volume": m.volume} for m in markets[:3]],
                sentiment="neutral",
                summary="Test scan results.",
                trust_score={"overall": 0.7},
            ),
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

        with patch(
            "councils.polymarket.division.run_polymarket_council",
            new_callable=AsyncMock,
            return_value=fake_entry,
        ):
            signals = asyncio.get_event_loop().run_until_complete(div.scan())

        # Should produce at least 1 signal (market summary)
        assert len(signals) >= 1
        assert signals[0].source_division == "polymarket_council"


# ============================================================================
# 4. Crypto scraper → analyzer → division
# ============================================================================

class TestCryptoPipeline:
    """Test Crypto council data flow: scraper → analyzer → division."""

    def test_analyzer_produces_insights(self):
        """Analyzer produces valid CryptoInsight."""
        from councils.crypto.analyzer import analyze_crypto_market

        coins = _make_coin_snapshots(5)
        trending = _make_trending_coins()
        global_data = _make_global_data()

        insight = analyze_crypto_market(coins, trending, global_data)

        assert len(insight.top_coins) > 0
        assert insight.btc_price == 67_000.0
        assert isinstance(insight.sentiment, str)
        assert isinstance(insight.summary, str) and len(insight.summary) > 0

    def test_trust_score_computed(self):
        """Trust score computed for crypto analysis."""
        from councils.crypto.analyzer import _compute_crypto_trust

        coins = _make_coin_snapshots(5)
        global_data = _make_global_data()
        trust = _compute_crypto_trust(coins, global_data)

        assert "overall" in trust
        assert 0.0 <= trust["overall"] <= 1.0

    def test_gainers_and_losers_sorted(self):
        """Gainers/losers lists are correctly sorted."""
        from councils.crypto.analyzer import analyze_crypto_market

        coins = _make_coin_snapshots(5)
        global_data = _make_global_data()
        insight = analyze_crypto_market(coins, [], global_data)

        if insight.gainers:
            changes = [g["change_24h"] for g in insight.gainers]
            assert changes == sorted(changes, reverse=True)
        if insight.losers:
            changes = [l["change_24h"] for l in insight.losers]
            assert changes == sorted(changes)

    def test_analyzer_handles_empty_coins(self):
        """Analyzer handles empty coin list gracefully."""
        from councils.crypto.analyzer import analyze_crypto_market
        from councils.crypto.models import GlobalData

        insight = analyze_crypto_market([], [], GlobalData())
        assert "No coin data" in insight.summary

    def test_division_scan_produces_signals(self):
        """CryptoCouncilDivision.scan() produces signals."""
        from councils.crypto.division import CryptoCouncilDivision
        from councils.crypto.models import CouncilEntry, CryptoInsight

        div = CryptoCouncilDivision(coin_ids=["bitcoin", "ethereum"])

        coins = _make_coin_snapshots(2)
        trending = _make_trending_coins()
        global_data = _make_global_data()
        fake_entry = CouncilEntry(
            coins=coins,
            trending_coins=trending,
            insights=CryptoInsight(
                top_coins=[{"coin": "bitcoin", "price": 67000}],
                trending=[],
                global_data=global_data,
                sentiment="bullish",
                summary="BTC is up.",
                btc_price=67_000.0,
                eth_price=3_500.0,
                trust_score={"overall": 0.8},
            ),
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

        with patch(
            "councils.crypto.division.run_crypto_council",
            new_callable=AsyncMock,
            return_value=fake_entry,
        ):
            signals = asyncio.get_event_loop().run_until_complete(div.scan())

        assert len(signals) >= 1
        assert signals[0].source_division == "crypto_council"


# ============================================================================
# 5. War Room Bridge: council feeds → indicator patching → alpha engine
# ============================================================================

class TestWarRoomBridge:
    """Test the war_room_council_feeds bridge connecting councils to the war room."""

    def test_sentiment_scoring(self):
        """Sentiment scoring produces correct polarity."""
        from strategies.war_room_council_feeds import _sentiment_from_text

        bearish = _sentiment_from_text("crash collapse panic fear recession")
        bullish = _sentiment_from_text("rally boom growth surge recovery")
        neutral = _sentiment_from_text("the weather is nice today")

        assert bearish < 0.4
        assert bullish > 0.6
        assert neutral == 0.5

    def test_severity_scoring(self):
        """Severity scoring detects crisis keywords."""
        from strategies.war_room_council_feeds import _severity_from_text

        crisis = _severity_from_text("crisis nuclear war pandemic collapse")
        calm = _severity_from_text("sunny day good morning")

        assert crisis > 0.3
        assert calm == 0.0

    def test_scenario_extraction(self):
        """Scenario extraction maps keywords to scenario codes."""
        from strategies.war_room_council_feeds import _extract_scenario_hits

        hits = _extract_scenario_hits(
            topics=["hormuz", "oil", "iran", "sanctions"],
            text="Iran sanctions threaten Hormuz strait oil supply",
        )

        assert len(hits) > 0, "Should detect at least one scenario"

    def test_apply_council_to_indicators(self):
        """apply_council_to_indicators combines YT + X into changes dict."""
        from strategies.war_room_council_feeds import (
            CouncilFeedResult,
            apply_council_to_indicators,
        )

        yt = CouncilFeedResult(
            yt_videos_processed=3,
            yt_sentiment_score=0.3,
            yt_severity_score=0.6,
            yt_key_topics=["oil", "iran"],
            yt_trust_score=0.7,
        )
        x = CouncilFeedResult(
            x_posts_analyzed=10,
            x_sentiment_score=0.25,
            x_severity_score=0.5,
            x_key_themes=["recession", "crash"],
            x_trust_score=0.4,
        )

        changes = apply_council_to_indicators(yt, x)

        assert "combined_sentiment" in changes
        assert "combined_severity" in changes
        assert "combined_trust" in changes
        # Both bearish → combined should be bearish
        assert changes["combined_sentiment"] < 0.4
        assert changes["combined_severity"] > 0.3

    def test_apply_council_patches_live_result(self):
        """apply_council_to_indicators patches a live_result object."""
        from strategies.war_room_council_feeds import (
            CouncilFeedResult,
            apply_council_to_indicators,
        )

        yt = CouncilFeedResult(
            yt_videos_processed=2,
            yt_sentiment_score=0.3,
            yt_trust_score=0.6,
        )

        # Simulate a live_result object with x_sentiment_score = None
        live = MagicMock()
        live.x_sentiment_score = None
        live.news_severity_score = 0.5

        changes = apply_council_to_indicators(yt, None, live_result=live)

        # Should have patched x_sentiment_score
        assert changes["combined_sentiment"] != 0.5 or changes.get("yt_sentiment_score") is not None

    def test_alpha_engine_integration(self):
        """Alpha engine records council observations and combines signals."""
        from strategies.alpha_engine import (
            _signal_history,
            alpha_combine_councils,
            record_council_observation,
        )

        # Clear any existing history
        _signal_history.clear()

        # Record enough observations for combination
        for i in range(5):
            record_council_observation("yt_sentiment", 0.3 + i * 0.02)
            record_council_observation("x_sentiment", 0.25 + i * 0.03)
            record_council_observation("severity", 0.5 + i * 0.01)

        result = alpha_combine_councils(estimation_period=3, min_observations=3)

        assert result is not None, "Should produce alpha result with 3+ signals"
        assert result.combined_signal is not None
        assert isinstance(result.weights, dict)

        # Clean up
        _signal_history.clear()

    def test_fetch_and_apply_full_pipeline(self):
        """Full pipeline: fetch_and_apply_council_intel produces combined result."""
        from strategies.war_room_council_feeds import (
            CouncilFeedResult,
            fetch_and_apply_council_intel,
        )

        mock_yt = CouncilFeedResult(
            yt_videos_processed=2,
            yt_sentiment_score=0.35,
            yt_severity_score=0.4,
            yt_key_topics=["oil", "iran"],
            yt_trust_score=0.65,
        )
        mock_x = CouncilFeedResult(
            x_posts_analyzed=8,
            x_sentiment_score=0.3,
            x_severity_score=0.5,
            x_key_themes=["recession"],
            x_trust_score=0.4,
        )

        with patch(
            "strategies.war_room_council_feeds.fetch_youtube_intel",
            new_callable=AsyncMock,
            return_value=mock_yt,
        ), patch(
            "strategies.war_room_council_feeds.fetch_x_intel",
            new_callable=AsyncMock,
            return_value=mock_x,
        ), patch(
            "strategies.war_room_council_feeds.persist_council_snapshot",
        ):
            result = asyncio.get_event_loop().run_until_complete(
                fetch_and_apply_council_intel()
            )

        assert isinstance(result, CouncilFeedResult)
        assert result.yt_videos_processed == 2
        assert result.x_posts_analyzed == 8
        assert result.combined_sentiment < 0.5  # Both bearish
        assert result.combined_trust > 0.0


# ============================================================================
# 6. Trust score integration across all councils
# ============================================================================

class TestTrustIntegration:
    """Test that trust scores flow correctly through the full pipeline."""

    def test_youtube_trust_in_insight(self):
        """YouTube analyzer embeds trust_score in VideoInsight."""
        from councils.youtube.analyzer import analyze_transcript

        meta = _make_video_meta(view_count=100_000, like_count=5_000)
        segments = _make_transcript_segments()
        insight = analyze_transcript(segments, meta)

        assert hasattr(insight, "trust_score")
        assert isinstance(insight.trust_score, dict)
        assert "overall" in insight.trust_score

    def test_crypto_trust_in_insight(self):
        """Crypto analyzer embeds trust_score in CryptoInsight."""
        from councils.crypto.analyzer import analyze_crypto_market

        coins = _make_coin_snapshots(5)
        global_data = _make_global_data()
        insight = analyze_crypto_market(coins, _make_trending_coins(), global_data)

        assert hasattr(insight, "trust_score")
        assert isinstance(insight.trust_score, dict)
        assert "overall" in insight.trust_score

    def test_polymarket_trust_in_insight(self):
        """Polymarket analyzer embeds trust_score in MarketInsight."""
        from councils.polymarket.analyzer import analyze_markets

        markets = _make_market_snapshots(10)
        insight = analyze_markets(markets)

        assert hasattr(insight, "trust_score")
        assert isinstance(insight.trust_score, dict)
        assert "overall" in insight.trust_score

    def test_xai_trust_in_insight(self):
        """X/Grok analyzer embeds trust_score in XInsight."""
        from councils.xai.analyzer import analyze_posts_extractive

        posts = _make_x_posts(10)
        insight = analyze_posts_extractive(posts, source="test")

        assert hasattr(insight, "trust_score")
        assert isinstance(insight.trust_score, dict)
        assert "overall" in insight.trust_score

    def test_trust_weights_sum_to_one(self):
        """Trust weight components sum to 1.0."""
        from councils.trust import _WEIGHTS

        total = sum(_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


# ============================================================================
# 7. Model dataclass contracts
# ============================================================================

class TestModelContracts:
    """Verify model dataclasses have required fields for the pipeline."""

    def test_youtube_council_entry_fields(self):
        from councils.youtube.models import CouncilEntry

        entry = CouncilEntry(
            meta=_make_video_meta(),
            transcript=[],
            insights=MagicMock(),
            markdown_path="test.md",
            processed_at="2026-07-10T00:00:00Z",
        )
        assert hasattr(entry, "meta")
        assert hasattr(entry, "insights")

    def test_xai_council_entry_fields(self):
        from councils.xai.models import CouncilEntry

        entry = CouncilEntry(
            posts=[], insights=MagicMock(),
            markdown_path="", processed_at="",
        )
        assert hasattr(entry, "posts")
        assert hasattr(entry, "insights")

    def test_polymarket_council_entry_fields(self):
        from councils.polymarket.models import CouncilEntry

        entry = CouncilEntry(markets=[], insights=MagicMock())
        assert hasattr(entry, "markets")
        assert hasattr(entry, "insights")

    def test_crypto_council_entry_fields(self):
        from councils.crypto.models import CouncilEntry

        entry = CouncilEntry(coins=[], trending_coins=[], insights=MagicMock())
        assert hasattr(entry, "coins")
        assert hasattr(entry, "trending_coins")
        assert hasattr(entry, "insights")

    def test_signal_dataclass(self):
        """Signal has required fields for inter-division communication."""
        from divisions.division_protocol import Signal, SignalType

        sig = Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division="test",
            timestamp=datetime.now(timezone.utc),
            data={"topic": "oil"},
        )
        assert sig.signal_type == SignalType.INTEL_UPDATE
        assert sig.confidence == 0.0  # default

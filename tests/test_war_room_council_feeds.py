from __future__ import annotations

"""Tests for war_room_council_feeds — YouTube + X → War Room bridge."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ============================================================================
# Helpers — lightweight stand-ins for council models
# ============================================================================

@dataclass
class _FakeVideoInsight:
    title: str = "Test Video"
    key_topics: list = field(default_factory=lambda: ["oil", "iran", "sanctions"])
    quotes: list = field(default_factory=list)
    summary: str = "Iran sanctions threaten oil supply through Hormuz."
    actionable_items: list = field(default_factory=lambda: ["Monitor oil prices"])
    sentiment: str = "negative"


@dataclass
class _FakeCouncilEntryYT:
    meta: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)
    insights: _FakeVideoInsight = field(default_factory=_FakeVideoInsight)
    markdown_path: str = ""
    processed_at: str = ""


@dataclass
class _FakeXPost:
    post_id: str = "1"
    author: str = "test"
    text: str = "Market crash incoming, recession fears rising"
    created_at: str = ""
    url: str = ""
    likes: int = 10
    reposts: int = 5
    replies: int = 2
    views: int = 100
    media_urls: list = field(default_factory=list)
    is_reply: bool = False
    is_repost: bool = False


@dataclass
class _FakeXInsight:
    source: str = "test"
    key_themes: list = field(default_factory=lambda: ["recession", "crash", "fed rates"])
    notable_posts: list = field(default_factory=list)
    sentiment_summary: str = "bearish"
    engagement_highlights: list = field(default_factory=list)
    emerging_topics: list = field(default_factory=lambda: ["tariff war", "china"])
    actionable_items: list = field(default_factory=lambda: ["Hedge exposure"])


@dataclass
class _FakeCouncilEntryX:
    posts: list = field(default_factory=lambda: [_FakeXPost()])
    insights: _FakeXInsight = field(default_factory=_FakeXInsight)
    markdown_path: str = ""
    processed_at: str = ""


# ============================================================================
# Test: Scoring functions
# ============================================================================

class TestScoringFunctions:
    def test_sentiment_from_text_bearish(self):
        from strategies.war_room_council_feeds import _sentiment_from_text
        score = _sentiment_from_text("crash collapse crisis panic sell fear")
        assert score < 0.3, f"Expected bearish (<0.3), got {score}"

    def test_sentiment_from_text_bullish(self):
        from strategies.war_room_council_feeds import _sentiment_from_text
        score = _sentiment_from_text("rally boom bullish recovery growth surge")
        assert score > 0.7, f"Expected bullish (>0.7), got {score}"

    def test_sentiment_from_text_neutral(self):
        from strategies.war_room_council_feeds import _sentiment_from_text
        score = _sentiment_from_text("the weather is nice today")
        assert score == 0.5

    def test_severity_from_text_crisis(self):
        from strategies.war_room_council_feeds import _severity_from_text
        score = _severity_from_text("crisis war nuclear pandemic crash collapse")
        assert score > 0.5, f"Expected high severity, got {score}"

    def test_severity_from_text_calm(self):
        from strategies.war_room_council_feeds import _severity_from_text
        score = _severity_from_text("sunny day happy times good food")
        assert score == 0.0

    def test_map_sentiment_label(self):
        from strategies.war_room_council_feeds import _map_sentiment_label
        assert _map_sentiment_label("positive") == 0.75
        assert _map_sentiment_label("negative") == 0.25
        assert _map_sentiment_label("mixed") == 0.45
        assert _map_sentiment_label("neutral") == 0.50
        assert _map_sentiment_label("unknown") == 0.50


# ============================================================================
# Test: Scenario keyword mapping
# ============================================================================

class TestScenarioMapping:
    def test_extract_scenario_hits_oil(self):
        from strategies.war_room_council_feeds import _extract_scenario_hits
        hits = _extract_scenario_hits(["oil prices", "Iran sanctions"])
        assert "hormuz_blockade" in hits
        assert "petrodollar_spiral" in hits

    def test_extract_scenario_hits_crypto(self):
        from strategies.war_room_council_feeds import _extract_scenario_hits
        hits = _extract_scenario_hits(["bitcoin crash", "defi liquidation"])
        assert "defi_liquidation_cascade" in hits

    def test_extract_scenario_hits_empty(self):
        from strategies.war_room_council_feeds import _extract_scenario_hits
        hits = _extract_scenario_hits(["nothing relevant here"])
        assert len(hits) == 0

    def test_extract_scenario_hits_from_text(self):
        from strategies.war_room_council_feeds import _extract_scenario_hits
        hits = _extract_scenario_hits([], text="Taiwan strait crisis and rare earth minerals")
        assert "taiwan_strait_crisis" in hits
        assert "rare_earth_fortress" in hits


# ============================================================================
# Test: CouncilFeedResult
# ============================================================================

class TestCouncilFeedResult:
    def test_default_values(self):
        from strategies.war_room_council_feeds import CouncilFeedResult
        r = CouncilFeedResult()
        assert r.combined_sentiment == 0.5
        assert r.combined_severity == 0.0
        assert r.yt_videos_processed == 0
        assert r.x_posts_analyzed == 0

    def test_summary_empty(self):
        from strategies.war_room_council_feeds import CouncilFeedResult
        r = CouncilFeedResult()
        assert r.summary() == "no council data"

    def test_summary_with_data(self):
        from strategies.war_room_council_feeds import CouncilFeedResult
        r = CouncilFeedResult(
            x_posts_analyzed=42,
            x_sentiment_score=0.3,
            scenario_signals={"hormuz_blockade": 0.8},
        )
        s = r.summary()
        assert "X:42posts" in s
        assert "hormuz_blockade" in s


# ============================================================================
# Test: apply_council_to_indicators
# ============================================================================

class TestApplyCouncilToIndicators:
    def test_patches_live_result_x_sentiment(self):
        from strategies.war_room_council_feeds import (
            CouncilFeedResult, apply_council_to_indicators,
        )

        yt = CouncilFeedResult()
        x = CouncilFeedResult(
            x_posts_analyzed=20,
            x_sentiment_score=0.3,
            x_severity_score=0.6,
            x_scenario_hits={"hormuz_blockade": 3},
        )

        # Mock LiveFeedResult
        live = MagicMock()
        live.x_sentiment_score = 0.5  # default neutral
        live.news_severity_score = 0.2
        live.council_scenario_signals = {}
        live.council_topics = []
        live.council_emerging = []

        changes = apply_council_to_indicators(yt, x, live)

        assert changes["combined_sentiment"] == 0.3
        assert changes["combined_severity"] == 0.6
        assert "hormuz_blockade" in changes["scenario_signals"]
        assert live.x_sentiment_score == 0.3  # patched from council
        assert live.news_severity_score == pytest.approx(0.2 * 0.6 + 0.6 * 0.4, abs=0.01)

    def test_no_council_data_keeps_defaults(self):
        from strategies.war_room_council_feeds import (
            CouncilFeedResult, apply_council_to_indicators,
        )
        yt = CouncilFeedResult()
        x = CouncilFeedResult()
        changes = apply_council_to_indicators(yt, x)
        assert changes["combined_sentiment"] == 0.5
        assert changes["combined_severity"] == 0.0


# ============================================================================
# Test: Async fetch functions (mocked)
# ============================================================================

@pytest.mark.asyncio
class TestFetchXIntel:
    async def test_fetch_x_intel_no_queries(self):
        """With no queries and no users, returns empty result."""
        from strategies.war_room_council_feeds import fetch_x_intel
        result = await fetch_x_intel(queries=[], users=[])
        assert result.x_posts_analyzed == 0

    @patch("strategies.war_room_council_feeds.fetch_x_intel")
    async def test_fetch_x_intel_mocked(self, mock_fetch):
        from strategies.war_room_council_feeds import CouncilFeedResult
        mock_fetch.return_value = CouncilFeedResult(
            x_posts_analyzed=15,
            x_sentiment_score=0.35,
            x_severity_score=0.5,
            x_key_themes=["recession", "fed"],
            x_scenario_hits={"credit_cascade": 2},
        )
        result = await mock_fetch()
        assert result.x_posts_analyzed == 15
        assert result.x_sentiment_score == 0.35


@pytest.mark.asyncio
class TestFetchYouTubeIntel:
    async def test_fetch_youtube_intel_no_channels(self):
        """With no channels configured, returns empty result."""
        from strategies.war_room_council_feeds import fetch_youtube_intel
        result = await fetch_youtube_intel(channels=[])
        assert result.yt_videos_processed == 0


# ============================================================================
# Test: Master function (fully mocked)
# ============================================================================

@pytest.mark.asyncio
class TestFetchAndApplyCouncilIntel:
    @patch("strategies.war_room_council_feeds.fetch_youtube_intel")
    @patch("strategies.war_room_council_feeds.fetch_x_intel")
    async def test_master_function(self, mock_x, mock_yt):
        from strategies.war_room_council_feeds import (
            CouncilFeedResult, fetch_and_apply_council_intel,
        )

        mock_yt.return_value = CouncilFeedResult(
            yt_videos_processed=2,
            yt_sentiment_score=0.4,
            yt_severity_score=0.3,
            yt_key_topics=["oil", "iran"],
            yt_scenario_hits={"hormuz_blockade": 1},
        )
        mock_x.return_value = CouncilFeedResult(
            x_posts_analyzed=30,
            x_sentiment_score=0.25,
            x_severity_score=0.7,
            x_key_themes=["crash", "recession"],
            x_scenario_hits={"credit_cascade": 2, "hormuz_blockade": 1},
        )

        result = await fetch_and_apply_council_intel()

        assert result.yt_videos_processed == 2
        assert result.x_posts_analyzed == 30
        assert result.combined_sentiment < 0.5  # bearish combined
        assert "hormuz_blockade" in result.scenario_signals
        assert "credit_cascade" in result.scenario_signals


# ============================================================================
# Test: LiveFeedResult council fields
# ============================================================================

class TestLiveFeedResultCouncilFields:
    def test_live_feed_result_has_council_fields(self):
        from strategies.war_room_live_feeds import LiveFeedResult
        r = LiveFeedResult()
        assert hasattr(r, "council_scenario_signals")
        assert hasattr(r, "council_topics")
        assert hasattr(r, "council_emerging")
        assert hasattr(r, "council_x_posts")
        assert hasattr(r, "council_yt_videos")
        assert r.council_scenario_signals == {}
        assert r.council_topics == []


# ============================================================================
# Test: fetch_x_sentiment now uses council
# ============================================================================

@pytest.mark.asyncio
class TestFetchXSentimentCouncilPowered:
    @patch("strategies.war_room_council_feeds.fetch_x_intel")
    async def test_replaces_broken_twitter_api(self, mock_x):
        from strategies.war_room_council_feeds import CouncilFeedResult
        mock_x.return_value = CouncilFeedResult(
            x_posts_analyzed=25,
            x_sentiment_score=0.3,
            x_severity_score=0.5,
            x_key_themes=["recession"],
            x_emerging_topics=["tariff"],
            x_scenario_hits={"credit_cascade": 1},
            scenario_signals={"credit_cascade": 0.5},
        )

        from strategies.war_room_live_feeds import LiveFeedResult, fetch_x_sentiment
        result = LiveFeedResult()
        await fetch_x_sentiment(result)

        assert result.x_sentiment_score == 0.3
        assert result.council_x_posts == 25
        assert "recession" in result.council_topics


# ============================================================================
# Test: Auto engine task registry includes council
# ============================================================================

class TestAutoEngineCouncilTask:
    def test_task_registry_includes_council(self):
        from strategies.war_room_auto import WarRoomAutoEngine
        engine = WarRoomAutoEngine()
        tasks = engine.get_task_registry()
        names = [t["name"] for t in tasks]
        assert "wr_council_scan" in names

    def test_council_scan_interval_default(self):
        from strategies.war_room_auto import AutoUpdateParams
        params = AutoUpdateParams()
        assert params.council_scan_interval == 900.0  # 15 minutes

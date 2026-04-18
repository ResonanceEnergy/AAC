from __future__ import annotations

"""Tests for the council trust scoring system."""

import pytest

from councils.trust import (
    TrustScore,
    compute_agreement,
    crypto_source_trust,
    evidence_score,
    freshness_score,
    polymarket_source_trust,
    xai_source_trust,
    youtube_source_trust,
)


# ============================================================================
# TrustScore dataclass
# ============================================================================

class TestTrustScore:
    def test_default_overall(self):
        ts = TrustScore()
        assert 0.0 <= ts.overall <= 1.0

    def test_overall_weighted_sum(self):
        ts = TrustScore(
            source_reliability=1.0,
            data_freshness=1.0,
            evidence_volume=1.0,
            cross_source_agreement=1.0,
        )
        assert ts.overall == 1.0

    def test_overall_zero(self):
        ts = TrustScore(
            source_reliability=0.0,
            data_freshness=0.0,
            evidence_volume=0.0,
            cross_source_agreement=0.0,
        )
        assert ts.overall == 0.0

    def test_overall_auto_computes(self):
        ts = TrustScore(source_reliability=0.8, data_freshness=0.6)
        assert ts.overall > 0.0

    def test_recalculate(self):
        ts = TrustScore(source_reliability=0.3)
        old = ts.overall
        ts.source_reliability = 0.9
        ts.recalculate()
        assert ts.overall > old

    def test_to_dict_keys(self):
        ts = TrustScore()
        d = ts.to_dict()
        assert "source_reliability" in d
        assert "data_freshness" in d
        assert "evidence_volume" in d
        assert "cross_source_agreement" in d
        assert "overall" in d
        assert "details" in d

    def test_clamped_to_0_1(self):
        ts = TrustScore(source_reliability=2.0, data_freshness=2.0,
                        evidence_volume=2.0, cross_source_agreement=2.0)
        assert ts.overall <= 1.0


# ============================================================================
# Freshness score
# ============================================================================

class TestFreshnessScore:
    def test_zero_age_is_1(self):
        assert freshness_score(0) == 1.0

    def test_negative_age_is_1(self):
        assert freshness_score(-100) == 1.0

    def test_half_life_is_half(self):
        score = freshness_score(3600, half_life=3600)
        assert abs(score - 0.5) < 0.01

    def test_two_half_lives_is_quarter(self):
        score = freshness_score(7200, half_life=3600)
        assert abs(score - 0.25) < 0.01

    def test_very_old_approaches_zero(self):
        score = freshness_score(86400 * 30, half_life=3600)
        assert score < 0.01


# ============================================================================
# Evidence score
# ============================================================================

class TestEvidenceScore:
    def test_zero_count(self):
        assert evidence_score(0) == 0.0

    def test_negative_count(self):
        assert evidence_score(-5) == 0.0

    def test_at_target(self):
        score = evidence_score(20, target=20)
        assert score >= 0.49  # 1 - 0.5^1 = 0.5

    def test_above_target(self):
        score = evidence_score(100, target=20)
        assert score > 0.95

    def test_one_item(self):
        score = evidence_score(1, target=20)
        assert 0.0 < score < 0.5

    def test_monotonically_increasing(self):
        scores = [evidence_score(i, target=20) for i in range(1, 50)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]


# ============================================================================
# YouTube source trust
# ============================================================================

class TestYoutubeSourceTrust:
    def test_minimal_video(self):
        score = youtube_source_trust()
        assert score >= 0.2  # base with short duration penalty

    def test_high_view_count(self):
        score = youtube_source_trust(view_count=1_000_000)
        assert score > 0.4

    def test_with_transcript(self):
        score = youtube_source_trust(has_transcript=True)
        assert score > 0.3

    def test_sweet_spot_duration(self):
        s1 = youtube_source_trust(duration_seconds=600)   # 10min
        s2 = youtube_source_trust(duration_seconds=30)    # 30 sec
        assert s1 > s2

    def test_known_finance_channel(self):
        score = youtube_source_trust(channel="ThePlainBagel")
        assert score > 0.3

    def test_max_capped_at_1(self):
        score = youtube_source_trust(
            view_count=100_000_000,
            like_count=5_000_000,
            has_transcript=True,
            duration_seconds=600,
            channel="ThePlainBagel",
        )
        assert score <= 1.0


# ============================================================================
# X/Grok source trust
# ============================================================================

class TestXaiSourceTrust:
    def test_synthesized_ceiling(self):
        score = xai_source_trust(post_count=100, provider="xai", is_llm_synthesized=True)
        assert score <= 0.45

    def test_real_posts_higher_ceiling(self):
        score = xai_source_trust(post_count=50, provider="xai", is_llm_synthesized=False)
        assert score <= 0.85

    def test_real_vs_synthesized(self):
        real = xai_source_trust(post_count=20, is_llm_synthesized=False)
        synth = xai_source_trust(post_count=20, is_llm_synthesized=True)
        assert real >= synth

    def test_xai_provider_bonus(self):
        xai = xai_source_trust(post_count=10, provider="xai")
        openai = xai_source_trust(post_count=10, provider="openai")
        assert xai >= openai


# ============================================================================
# Crypto source trust
# ============================================================================

class TestCryptoSourceTrust:
    def test_base_high(self):
        score = crypto_source_trust()
        assert score >= 0.6

    def test_pro_tier_boost(self):
        pro = crypto_source_trust(api_tier="pro")
        free = crypto_source_trust(api_tier="free")
        assert pro > free

    def test_with_global_data(self):
        with_g = crypto_source_trust(has_global_data=True)
        without = crypto_source_trust(has_global_data=False)
        assert with_g > without

    def test_many_coins(self):
        many = crypto_source_trust(coin_count=20)
        few = crypto_source_trust(coin_count=2)
        assert many > few


# ============================================================================
# Polymarket source trust
# ============================================================================

class TestPolymarketSourceTrust:
    def test_base(self):
        score = polymarket_source_trust()
        assert score >= 0.5

    def test_high_liquidity(self):
        high = polymarket_source_trust(avg_liquidity=600_000)
        low = polymarket_source_trust(avg_liquidity=100)
        assert high > low

    def test_high_volume(self):
        high = polymarket_source_trust(avg_volume=2_000_000)
        low = polymarket_source_trust(avg_volume=100)
        assert high > low

    def test_max_capped(self):
        score = polymarket_source_trust(
            market_count=500,
            avg_liquidity=10_000_000,
            avg_volume=10_000_000,
        )
        assert score <= 1.0


# ============================================================================
# Cross-source agreement
# ============================================================================

class TestComputeAgreement:
    def test_single_source(self):
        assert compute_agreement([("yt", 0.7)]) == 0.5

    def test_perfect_agreement_bullish(self):
        score = compute_agreement([("yt", 0.8), ("x", 0.9), ("crypto", 0.85)])
        assert score > 0.8

    def test_perfect_agreement_bearish(self):
        score = compute_agreement([("yt", 0.2), ("x", 0.1)])
        assert score > 0.8

    def test_disagreement(self):
        score = compute_agreement([("yt", 0.1), ("x", 0.9)])
        assert score < 0.6

    def test_mixed_signals(self):
        score = compute_agreement([("yt", 0.6), ("x", 0.4), ("crypto", 0.5)])
        # All near 0.5 = low direction agreement, but close in magnitude
        assert 0.3 < score < 0.8


# ============================================================================
# Integration: analyzer trust wiring
# ============================================================================

class TestAnalyzerTrustWiring:
    """Verify analyzers produce trust_score on output models."""

    def test_youtube_analyzer_adds_trust(self):
        from councils.youtube.analyzer import analyze_transcript
        from councils.youtube.models import TranscriptSegment, VideoMeta

        meta = VideoMeta(
            video_id="test",
            title="Stock Market Analysis",
            channel="TestChannel",
            upload_date="20260401",
            duration_seconds=600,
            description="Test video about stock market",
            url="https://youtube.com/watch?v=test",
            view_count=10000,
            like_count=500,
        )
        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="The stock market is showing signs of recovery after the crash."),
            TranscriptSegment(start=10.0, end=20.0, text="Investors should consider defensive positions in this environment."),
        ]
        result = analyze_transcript(segments, meta)
        assert result.trust_score, "trust_score should be populated"
        assert "overall" in result.trust_score
        assert 0.0 < result.trust_score["overall"] <= 1.0

    def test_xai_analyzer_adds_trust(self):
        from councils.xai.analyzer import analyze_posts_extractive
        from councils.xai.models import XPost

        posts = [
            XPost(post_id="1", author="user1", text="Bearish on markets, recession incoming",
                  created_at="", url="", likes=10, reposts=5, replies=2, views=100),
            XPost(post_id="2", author="user2", text="Fed rate decision is key this week",
                  created_at="", url="", likes=20, reposts=10, replies=5, views=500),
        ]
        result = analyze_posts_extractive(posts, source="test_query")
        assert result.trust_score, "trust_score should be populated"
        assert "overall" in result.trust_score
        assert result.trust_score["overall"] <= 0.45  # LLM-synthesized ceiling

    def test_crypto_analyzer_adds_trust(self):
        from councils.crypto.analyzer import analyze_crypto_market
        from councils.crypto.models import CoinSnapshot, GlobalData, TrendingCoin

        coins = [
            CoinSnapshot(coin_id="bitcoin", symbol="btc",
                         price=70000, volume_24h=30e9, market_cap=1.3e12, change_24h=2.5),
            CoinSnapshot(coin_id="ethereum", symbol="eth",
                         price=2200, volume_24h=15e9, market_cap=260e9, change_24h=-1.2),
        ]
        global_data = GlobalData(
            total_market_cap_usd=2.5e12,
            total_volume_24h_usd=100e9,
            btc_dominance=52.0,
            market_cap_change_24h_pct=1.5,
            active_cryptocurrencies=10000,
        )
        result = analyze_crypto_market(coins, [], global_data)
        assert result.trust_score, "trust_score should be populated"
        assert "overall" in result.trust_score
        assert result.trust_score["overall"] > 0.5  # real market data = higher trust

    def test_polymarket_analyzer_adds_trust(self):
        from councils.polymarket.analyzer import analyze_markets
        from councils.polymarket.models import MarketSnapshot

        markets = [
            MarketSnapshot(
                condition_id="abc123",
                question="Will BTC exceed $100K by Dec 2026?",
                yes_price=0.35,
                no_price=0.65,
                volume=500_000,
                liquidity=200_000,
                end_date="2026-12-31",
                tags=["crypto", "bitcoin"],
            ),
        ]
        result = analyze_markets(markets)
        assert result.trust_score, "trust_score should be populated"
        assert "overall" in result.trust_score
        assert result.trust_score["overall"] > 0.4


# ============================================================================
# Integration: CouncilFeedResult trust fields
# ============================================================================

class TestCouncilFeedResultTrust:
    def test_trust_fields_exist(self):
        from strategies.war_room_council_feeds import CouncilFeedResult
        r = CouncilFeedResult()
        assert hasattr(r, "yt_trust_score")
        assert hasattr(r, "x_trust_score")
        assert hasattr(r, "combined_trust")

    def test_summary_includes_trust(self):
        from strategies.war_room_council_feeds import CouncilFeedResult
        r = CouncilFeedResult(
            yt_videos_processed=3,
            yt_sentiment_score=0.6,
            yt_trust_score=0.72,
        )
        summary = r.summary()
        assert "trust=" in summary

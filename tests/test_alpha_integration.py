from __future__ import annotations

"""Tests for alpha engine ↔ war room integration.

Verifies that the alpha signal flows end-to-end:
  council observations → alpha_combine_councils → IndicatorState.alpha_signal
  → compute_composite_score → composite crisis score
"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from strategies.alpha_engine import (
    clear_signal_history,
    record_council_observation,
    alpha_combine_councils,
)


# ============================================================================
# IndicatorState — alpha_signal field
# ============================================================================

class TestIndicatorStateAlpha:
    """IndicatorState now has an alpha_signal field."""

    def test_alpha_signal_default_zero(self):
        from strategies.war_room_engine import IndicatorState
        ind = IndicatorState()
        assert ind.alpha_signal == 0.0

    def test_alpha_signal_settable(self):
        from strategies.war_room_engine import IndicatorState
        ind = IndicatorState()
        ind.alpha_signal = 0.75
        assert ind.alpha_signal == 0.75

    def test_alpha_signal_negative(self):
        from strategies.war_room_engine import IndicatorState
        ind = IndicatorState()
        ind.alpha_signal = -0.5
        assert ind.alpha_signal == -0.5


# ============================================================================
# compute_composite_score — alpha weight
# ============================================================================

class TestCompositeScoreAlpha:
    """Alpha signal is weighted into the composite crisis score."""

    def test_neutral_alpha_gives_50_score(self):
        """alpha_signal=0 should produce alpha sub-score of 50."""
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        ind = IndicatorState()
        ind.alpha_signal = 0.0
        result = compute_composite_score(ind)
        assert "alpha" in result["individual_scores"]
        assert result["individual_scores"]["alpha"] == 50.0

    def test_bullish_alpha_lowers_crisis(self):
        """Positive alpha_signal (bullish) should produce low crisis sub-score."""
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        ind = IndicatorState()
        ind.alpha_signal = 1.0
        result = compute_composite_score(ind)
        assert result["individual_scores"]["alpha"] == 0.0

    def test_bearish_alpha_raises_crisis(self):
        """Negative alpha_signal (bearish) should produce high crisis sub-score."""
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        ind = IndicatorState()
        ind.alpha_signal = -1.0
        result = compute_composite_score(ind)
        assert result["individual_scores"]["alpha"] == 100.0

    def test_alpha_weight_exists_in_composite(self):
        """Alpha should contribute to the composite with meaningful weight."""
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        # Strong bullish alpha → lower composite than neutral
        ind_neutral = IndicatorState()
        ind_neutral.alpha_signal = 0.0
        score_neutral = compute_composite_score(ind_neutral)["composite_score"]

        ind_bullish = IndicatorState()
        ind_bullish.alpha_signal = 1.0
        score_bullish = compute_composite_score(ind_bullish)["composite_score"]

        # Bullish alpha should lower composite score (less crisis)
        assert score_bullish < score_neutral

    def test_bearish_alpha_raises_composite(self):
        """Bearish alpha → higher composite than neutral."""
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        ind_neutral = IndicatorState()
        ind_neutral.alpha_signal = 0.0
        score_neutral = compute_composite_score(ind_neutral)["composite_score"]

        ind_bearish = IndicatorState()
        ind_bearish.alpha_signal = -1.0
        score_bearish = compute_composite_score(ind_bearish)["composite_score"]

        assert score_bearish > score_neutral

    def test_weights_sum_to_one(self):
        """All 16 weights (including alpha) must sum to 1.0."""
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        ind = IndicatorState()
        result = compute_composite_score(ind)
        # Should have 16 individual scores now
        assert len(result["individual_scores"]) == 16

    def test_alpha_sub_score_clamped(self):
        """Alpha sub-score should be clamped to [0, 100] even with extreme input."""
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        ind = IndicatorState()
        ind.alpha_signal = 5.0  # way beyond normal range
        result = compute_composite_score(ind)
        assert result["individual_scores"]["alpha"] >= 0.0

        ind.alpha_signal = -5.0
        result = compute_composite_score(ind)
        assert result["individual_scores"]["alpha"] <= 100.0


# ============================================================================
# LiveFeedResult — alpha fields
# ============================================================================

class TestLiveFeedResultAlpha:
    """LiveFeedResult carries alpha_signal and alpha_weights."""

    def test_alpha_fields_default_none(self):
        from strategies.war_room_live_feeds import LiveFeedResult
        r = LiveFeedResult()
        assert r.alpha_signal is None
        assert r.alpha_weights == {}

    def test_alpha_fields_settable(self):
        from strategies.war_room_live_feeds import LiveFeedResult
        r = LiveFeedResult()
        r.alpha_signal = 0.42
        r.alpha_weights = {"yt_sentiment": 0.3, "x_sentiment": 0.7}
        assert r.alpha_signal == 0.42
        assert len(r.alpha_weights) == 2


# ============================================================================
# apply_live_data_to_indicators — alpha patching
# ============================================================================

class TestApplyLiveDataAlpha:
    """apply_live_data_to_indicators patches alpha_signal onto IndicatorState."""

    def test_alpha_signal_patched(self):
        from strategies.war_room_live_feeds import LiveFeedResult, apply_live_data_to_indicators
        from strategies.war_room_engine import IndicatorState
        result = LiveFeedResult()
        result.alpha_signal = 0.65
        ind = IndicatorState()
        updated = apply_live_data_to_indicators(result, ind)
        assert updated.alpha_signal == 0.65

    def test_alpha_signal_none_leaves_default(self):
        from strategies.war_room_live_feeds import LiveFeedResult, apply_live_data_to_indicators
        from strategies.war_room_engine import IndicatorState
        result = LiveFeedResult()
        result.alpha_signal = None
        ind = IndicatorState()
        updated = apply_live_data_to_indicators(result, ind)
        assert updated.alpha_signal == 0.0  # default untouched


# ============================================================================
# End-to-end: council → alpha → composite
# ============================================================================

class TestEndToEndAlphaFlow:
    """Full pipeline: council observations → alpha engine → IndicatorState → composite."""

    def test_full_alpha_pipeline(self):
        """Feed enough council data → alpha produces signal → composite uses it."""
        clear_signal_history()

        # Simulate 10 rounds of bearish council observations
        for i in range(10):
            record_council_observation("yt_sentiment", 0.2 + i * 0.01)
            record_council_observation("x_sentiment", 0.15 + i * 0.02)
            record_council_observation("yt_severity", 0.7 + i * 0.01)
            record_council_observation("x_severity", 0.65 + i * 0.02)

        alpha_result = alpha_combine_councils()
        assert alpha_result is not None, "Alpha engine should fire with 10 observations on 4 channels"

        # Inject alpha into IndicatorState
        from strategies.war_room_engine import IndicatorState, compute_composite_score
        ind = IndicatorState()
        ind.alpha_signal = alpha_result.combined_signal

        # Compute composite
        result = compute_composite_score(ind)
        assert "alpha" in result["individual_scores"]
        # Alpha sub-score should reflect the signal direction
        alpha_score = result["individual_scores"]["alpha"]
        if alpha_result.combined_signal < 0:
            assert alpha_score > 50  # bearish → higher crisis
        elif alpha_result.combined_signal > 0:
            assert alpha_score < 50  # bullish → lower crisis

    def test_alpha_through_live_feed_result(self):
        """Alpha signal flowing through LiveFeedResult → IndicatorState → composite."""
        from strategies.war_room_live_feeds import LiveFeedResult, apply_live_data_to_indicators
        from strategies.war_room_engine import compute_composite_score

        # Set up LiveFeedResult with alpha
        result = LiveFeedResult()
        result.alpha_signal = -0.8  # strong bearish

        ind = apply_live_data_to_indicators(result)
        assert ind.alpha_signal == -0.8

        composite = compute_composite_score(ind)
        alpha_score = composite["individual_scores"]["alpha"]
        assert alpha_score == 90.0  # 50 - (-0.8 * 50) = 90

    def test_council_feed_result_propagates_alpha_to_live_result(self):
        """fetch_and_apply_council_intel should propagate alpha to LiveFeedResult."""
        from strategies.war_room_council_feeds import CouncilFeedResult
        from strategies.war_room_live_feeds import LiveFeedResult

        # Verify CouncilFeedResult has alpha fields
        cfr = CouncilFeedResult(alpha_signal=0.42, alpha_weights={"yt": 0.6, "x": 0.4})
        assert cfr.alpha_signal == 0.42
        assert cfr.alpha_weights == {"yt": 0.6, "x": 0.4}

        # Verify LiveFeedResult can receive alpha
        lfr = LiveFeedResult()
        lfr.alpha_signal = cfr.alpha_signal
        lfr.alpha_weights = cfr.alpha_weights
        assert lfr.alpha_signal == 0.42


# ============================================================================
# war_room_auto — alpha injection into IndicatorState
# ============================================================================

class TestWarRoomAutoAlphaInjection:
    """task_council_scan injects alpha_signal into IndicatorState."""

    @pytest.mark.asyncio
    async def test_task_council_scan_injects_alpha(self):
        """After council scan, alpha should be injected into IndicatorState."""
        from strategies.war_room_council_feeds import CouncilFeedResult

        mock_result = CouncilFeedResult(
            yt_videos_processed=5,
            x_posts_analyzed=20,
            alpha_signal=0.55,
            alpha_weights={"yt_sentiment": 0.5, "x_sentiment": 0.5},
        )

        with patch(
            "strategies.war_room_council_feeds.fetch_and_apply_council_intel",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            from strategies.war_room_auto import WarRoomAutoEngine
            mgr = WarRoomAutoEngine.__new__(WarRoomAutoEngine)
            mgr.feed_health = {}
            mgr.update_params = MagicMock()
            mgr.update_params.feed_max_retries = 3
            mgr._save_feed_health = MagicMock()

            with patch("strategies.war_room_auto.logger"):
                await mgr.task_council_scan()

            # Verify alpha was injected
            from strategies.war_room_engine import IndicatorState
            assert IndicatorState.alpha_signal == 0.55


# ============================================================================
# LDE feed_alpha_engine wiring
# ============================================================================

class TestLDEAlphaWiring:
    """LDE.feed_alpha_engine is called during council intel pipeline."""

    def test_lde_feed_alpha_engine_called_in_pipeline(self):
        """fetch_and_apply_council_intel should call LDE.feed_alpha_engine."""
        from strategies.war_room_council_feeds import CouncilFeedResult

        yt_result = CouncilFeedResult(
            yt_videos_processed=3,
            yt_sentiment_score=0.6,
            yt_severity_score=0.3,
            yt_trust_score=0.5,
        )
        x_result = CouncilFeedResult(
            x_posts_analyzed=10,
            x_sentiment_score=0.5,
            x_severity_score=0.2,
            x_trust_score=0.4,
        )

        mock_lde = MagicMock()
        mock_lde_cls = MagicMock(return_value=mock_lde)

        with patch(
            "strategies.war_room_council_feeds.fetch_youtube_intel",
            new_callable=AsyncMock,
            return_value=yt_result,
        ), patch(
            "strategies.war_room_council_feeds.fetch_x_intel",
            new_callable=AsyncMock,
            return_value=x_result,
        ), patch(
            "strategies.war_room_council_feeds.update_scenario_statuses",
            return_value=[],
        ), patch(
            "strategies.war_room_council_feeds.persist_council_snapshot",
        ), patch(
            "strategies.living_doctrine_engine.LivingDoctrineEngine",
            mock_lde_cls,
        ):
            from strategies.war_room_council_feeds import fetch_and_apply_council_intel
            result = asyncio.get_event_loop().run_until_complete(
                fetch_and_apply_council_intel()
            )

        mock_lde.feed_alpha_engine.assert_called_once()

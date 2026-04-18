"""Tests for the Alpha Extraction & Signal Combination Engine.

Validates every stage of the pipeline from the working paper:
  Eqs 1-12: demean → variance → normalise → truncate → cross-sectional
  demean → expected return → vol-normalise → orthogonalise → weight →
  combine.
"""

from __future__ import annotations

import math

import pytest

from strategies.alpha_engine import (
    AlphaResult,
    SignalSeries,
    _cross_sectional_demean,
    _compute_weights,
    _demean,
    _expected_return,
    _normalise,
    _orthogonalise,
    _truncate,
    _variance,
    _combined_signal,
    alpha_combine_councils,
    clear_signal_history,
    extract_alpha,
    get_signal_history,
    record_council_observation,
)


# ============================================================================
# Eq 1 — Demeaning
# ============================================================================

class TestDemean:
    def test_empty(self):
        assert _demean([]) == []

    def test_constant_series(self):
        result = _demean([5.0, 5.0, 5.0])
        assert all(abs(v) < 1e-12 for v in result)

    def test_simple_values(self):
        # mean of [1,2,3] = 2.0
        result = _demean([1.0, 2.0, 3.0])
        assert result == pytest.approx([-1.0, 0.0, 1.0])

    def test_sum_is_zero(self):
        result = _demean([10, -4, 7, 3, -1])
        assert sum(result) == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Eq 2 — Variance
# ============================================================================

class TestVariance:
    def test_empty(self):
        assert _variance([]) == 0.0

    def test_zero_demeaned(self):
        assert _variance([0.0, 0.0, 0.0]) == 0.0

    def test_simple(self):
        # demeaned = [-1, 0, 1] → var = (1+0+1)/3 = 2/3
        assert _variance([-1.0, 0.0, 1.0]) == pytest.approx(2.0 / 3.0)

    def test_single_value(self):
        # single demeaned value → variance = v²
        assert _variance([3.0]) == pytest.approx(9.0)


# ============================================================================
# Eq 3 — Normalisation
# ============================================================================

class TestNormalise:
    def test_zero_sigma_returns_zeros(self):
        result = _normalise([1.0, 2.0, 3.0], sigma=0.0)
        assert result == [0.0, 0.0, 0.0]

    def test_unit_sigma(self):
        result = _normalise([2.0, -3.0], sigma=1.0)
        assert result == pytest.approx([2.0, -3.0])

    def test_scaling(self):
        result = _normalise([4.0, -8.0], sigma=2.0)
        assert result == pytest.approx([2.0, -4.0])


# ============================================================================
# Eq 4 — Truncation
# ============================================================================

class TestTruncate:
    def test_keep_less_than_length(self):
        assert _truncate([1, 2, 3, 4, 5], keep=3) == [3, 4, 5]

    def test_keep_more_than_length(self):
        assert _truncate([1, 2], keep=5) == [1, 2]

    def test_keep_zero(self):
        assert _truncate([1, 2, 3], keep=0) == []

    def test_negative_keep(self):
        assert _truncate([1, 2, 3], keep=-1) == []


# ============================================================================
# Eq 5 — Cross-Sectional Demeaning
# ============================================================================

class TestCrossSectionalDemean:
    def test_empty(self):
        assert _cross_sectional_demean({}) == {}

    def test_single_signal_becomes_zero(self):
        result = _cross_sectional_demean({"a": [1.0, 2.0, 3.0]})
        assert all(abs(v) < 1e-12 for v in result["a"])

    def test_two_signals_symmetric(self):
        result = _cross_sectional_demean({
            "a": [1.0, 3.0],
            "b": [3.0, 1.0],
        })
        # Cross-mean at t=0: (1+3)/2=2, at t=1: (3+1)/2=2
        assert result["a"] == pytest.approx([-1.0, 1.0])
        assert result["b"] == pytest.approx([1.0, -1.0])

    def test_cross_mean_is_zero_each_step(self):
        result = _cross_sectional_demean({
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "c": [5.0, 6.0],
        })
        for s in range(2):
            cross_sum = sum(result[n][s] for n in result)
            assert cross_sum == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Eq 7 — Expected Return
# ============================================================================

class TestExpectedReturn:
    def test_empty(self):
        assert _expected_return([], 5) == 0.0

    def test_zero_d(self):
        assert _expected_return([1, 2, 3], 0) == 0.0

    def test_negative_d(self):
        assert _expected_return([1, 2, 3], -1) == 0.0

    def test_full_window(self):
        # last 3 of [1,2,3] → mean = 2
        assert _expected_return([1.0, 2.0, 3.0], 3) == pytest.approx(2.0)

    def test_partial_window(self):
        # last 2 of [1,2,3,4] → mean = 3.5
        assert _expected_return([1.0, 2.0, 3.0, 4.0], 2) == pytest.approx(3.5)

    def test_d_larger_than_series(self):
        # d=10 but only 3 values → uses all 3
        assert _expected_return([2.0, 4.0, 6.0], 10) == pytest.approx(4.0)


# ============================================================================
# Eq 9 — Orthogonalisation (Residual Extraction)
# ============================================================================

class TestOrthogonalise:
    def test_empty(self):
        assert _orthogonalise({}, {}) == {}

    def test_zero_factor_variance(self):
        # When factors are all zero, residuals = e_norm unchanged
        e = {"a": 1.5, "b": -0.5}
        f = {"a": [0.0, 0.0], "b": [0.0, 0.0]}
        result = _orthogonalise(e, f)
        assert result["a"] == pytest.approx(1.5)
        assert result["b"] == pytest.approx(-0.5)

    def test_residual_removes_common_factor(self):
        # Two perfectly correlated signals → after projection,
        # residuals should differ from raw e_norm
        e = {"a": 1.0, "b": 1.0}
        f = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}
        result = _orthogonalise(e, f)
        # Residuals must still exist (not crash) and sum differently
        assert "a" in result and "b" in result
        # With identical factors the market projection is non-trivial
        assert abs(result["a"]) <= abs(e["a"]) + 1e-9


# ============================================================================
# Eq 10-11 — Weight Computation
# ============================================================================

class TestComputeWeights:
    def test_equal_residuals_equal_vol(self):
        residuals = {"a": 1.0, "b": 1.0}
        vols = {"a": 1.0, "b": 1.0}
        w = _compute_weights(residuals, vols)
        assert w["a"] == pytest.approx(0.5)
        assert w["b"] == pytest.approx(0.5)

    def test_weights_sum_to_one(self):
        residuals = {"a": 0.3, "b": -0.7, "c": 0.1}
        vols = {"a": 0.5, "b": 1.0, "c": 0.8}
        w = _compute_weights(residuals, vols)
        assert sum(abs(v) for v in w.values()) == pytest.approx(1.0)

    def test_zero_residuals_equal_weight(self):
        residuals = {"a": 0.0, "b": 0.0}
        vols = {"a": 1.0, "b": 2.0}
        w = _compute_weights(residuals, vols)
        assert w["a"] == pytest.approx(0.5)
        assert w["b"] == pytest.approx(0.5)

    def test_low_vol_gets_more_weight(self):
        # Same residual, but 'a' has lower vol → higher weight
        residuals = {"a": 1.0, "b": 1.0}
        vols = {"a": 0.5, "b": 2.0}
        w = _compute_weights(residuals, vols)
        assert abs(w["a"]) > abs(w["b"])


# ============================================================================
# Eq 12 — Combined Signal
# ============================================================================

class TestCombinedSignal:
    def test_simple(self):
        w = {"a": 0.6, "b": 0.4}
        latest = {"a": 1.0, "b": 0.5}
        assert _combined_signal(w, latest) == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)

    def test_missing_latest(self):
        w = {"a": 0.5, "b": 0.5}
        latest = {"a": 1.0}  # 'b' missing → treated as 0
        assert _combined_signal(w, latest) == pytest.approx(0.5)

    def test_zero_weights(self):
        w = {"a": 0.0, "b": 0.0}
        latest = {"a": 100.0, "b": -50.0}
        assert _combined_signal(w, latest) == pytest.approx(0.0)


# ============================================================================
# Full Pipeline — extract_alpha
# ============================================================================

class TestExtractAlpha:
    def test_no_signals(self):
        result = extract_alpha([])
        assert result.combined_signal == 0.0
        assert "error" in result.details

    def test_all_empty_signals(self):
        result = extract_alpha([SignalSeries("x", [])])
        assert "error" in result.details

    def test_single_signal_returns_weight_one(self):
        result = extract_alpha([
            SignalSeries("only", [0.5, 0.6, 0.7, 0.8]),
        ])
        assert "only" in result.weights
        # Single signal gets full weight
        assert abs(result.weights["only"]) == pytest.approx(1.0)

    def test_two_identical_signals(self):
        values = [0.5, 0.6, 0.55, 0.65, 0.7]
        result = extract_alpha([
            SignalSeries("a", list(values)),
            SignalSeries("b", list(values)),
        ])
        assert sum(abs(v) for v in result.weights.values()) == pytest.approx(1.0)
        assert result.combined_signal != 0.0

    def test_divergent_signals_get_different_weights(self):
        result = extract_alpha([
            SignalSeries("trending_up", [0.1, 0.2, 0.3, 0.4, 0.5]),
            SignalSeries("stable", [0.5, 0.5, 0.5, 0.5, 0.5]),
        ])
        assert result.weights
        # trending signal has non-zero residual; stable has zero variance
        assert result.volatilities["trending_up"] > result.volatilities.get("stable", 0)

    def test_estimation_period(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = extract_alpha(
            [SignalSeries("x", vals), SignalSeries("y", [v * 0.5 for v in vals])],
            estimation_period=3,
        )
        # E(x) over last 3 = mean(4,5,6) = 5
        assert result.expected_returns["x"] == pytest.approx(5.0)

    def test_truncation_window(self):
        vals = list(range(20))
        result = extract_alpha(
            [SignalSeries("a", vals), SignalSeries("b", [v * 0.1 for v in vals])],
            truncation_window=5,
        )
        assert result.details["n_signals"] == 2

    def test_result_has_all_fields(self):
        result = extract_alpha([
            SignalSeries("a", [0.5, 0.6, 0.4]),
            SignalSeries("b", [0.7, 0.3, 0.5]),
        ])
        assert isinstance(result, AlphaResult)
        assert isinstance(result.weights, dict)
        assert isinstance(result.residuals, dict)
        assert isinstance(result.volatilities, dict)
        assert isinstance(result.expected_returns, dict)
        assert isinstance(result.details, dict)

    def test_weights_sum_to_one(self):
        result = extract_alpha([
            SignalSeries("sent_yt", [0.4, 0.5, 0.6, 0.55, 0.65]),
            SignalSeries("sent_x", [0.3, 0.4, 0.35, 0.45, 0.5]),
            SignalSeries("severity", [0.1, 0.2, 0.15, 0.25, 0.3]),
        ])
        total = sum(abs(v) for v in result.weights.values())
        assert total == pytest.approx(1.0)


# ============================================================================
# Council Adapter — history recording
# ============================================================================

class TestSignalHistory:
    def setup_method(self):
        clear_signal_history()

    def test_record_and_retrieve(self):
        record_council_observation("test_sig", 0.5)
        record_council_observation("test_sig", 0.6)
        assert get_signal_history("test_sig") == [0.5, 0.6]

    def test_unknown_signal_empty(self):
        assert get_signal_history("nonexistent") == []

    def test_max_history_cap(self):
        for i in range(250):
            record_council_observation("big", float(i))
        history = get_signal_history("big")
        assert len(history) == 200  # _MAX_HISTORY
        assert history[0] == 50.0  # oldest kept

    def test_clear(self):
        record_council_observation("x", 1.0)
        clear_signal_history()
        assert get_signal_history("x") == []


class TestAlphaCombineCouncils:
    def setup_method(self):
        clear_signal_history()

    def test_insufficient_history_returns_none(self):
        record_council_observation("a", 0.5)
        assert alpha_combine_councils() is None

    def test_insufficient_signals_returns_none(self):
        # Only 1 signal with enough data
        for _ in range(5):
            record_council_observation("a", 0.5)
        assert alpha_combine_councils() is None

    def test_sufficient_data_returns_result(self):
        for i in range(10):
            record_council_observation("yt_sent", 0.4 + i * 0.02)
            record_council_observation("x_sent", 0.5 - i * 0.01)
        result = alpha_combine_councils()
        assert result is not None
        assert isinstance(result, AlphaResult)
        assert sum(abs(v) for v in result.weights.values()) == pytest.approx(1.0)

    def test_custom_estimation_period(self):
        for i in range(10):
            record_council_observation("a", float(i))
            record_council_observation("b", float(i) * 0.5)
        result = alpha_combine_councils(estimation_period=3)
        assert result is not None

    def test_min_observations_filter(self):
        for i in range(2):
            record_council_observation("short", float(i))
        for i in range(10):
            record_council_observation("long", float(i))
        # short has only 2, min_observations=3 default → only 1 signal → None
        assert alpha_combine_councils() is None

    def test_four_council_signals(self):
        """Simulate all four councils feeding the alpha engine."""
        for i in range(15):
            record_council_observation("yt_sentiment", 0.5 + 0.02 * math.sin(i))
            record_council_observation("x_sentiment", 0.45 + 0.03 * math.cos(i))
            record_council_observation("yt_severity", 0.1 + 0.01 * i)
            record_council_observation("x_severity", 0.15 + 0.005 * i)
        result = alpha_combine_councils()
        assert result is not None
        assert len(result.weights) == 4
        assert sum(abs(v) for v in result.weights.values()) == pytest.approx(1.0)


# ============================================================================
# Integration — council feed result carries alpha data
# ============================================================================

class TestCouncilFeedResultAlpha:
    def test_alpha_fields_exist(self):
        from strategies.war_room_council_feeds import CouncilFeedResult
        r = CouncilFeedResult()
        assert r.alpha_signal is None
        assert r.alpha_weights == {}

    def test_summary_includes_alpha(self):
        from strategies.war_room_council_feeds import CouncilFeedResult
        r = CouncilFeedResult(alpha_signal=0.123, yt_videos_processed=1,
                              yt_sentiment_score=0.6, yt_trust_score=0.5)
        s = r.summary()
        assert "alpha=0.123" in s

    def test_apply_records_observations(self):
        from strategies.war_room_council_feeds import (
            CouncilFeedResult,
            apply_council_to_indicators,
        )
        clear_signal_history()
        yt = CouncilFeedResult(yt_videos_processed=3, yt_sentiment_score=0.7,
                               yt_severity_score=0.2, yt_trust_score=0.5)
        x = CouncilFeedResult(x_posts_analyzed=10, x_sentiment_score=0.6,
                              x_severity_score=0.3, x_trust_score=0.4)
        apply_council_to_indicators(yt, x)
        assert len(get_signal_history("yt_sentiment")) == 1
        assert len(get_signal_history("x_sentiment")) == 1
        assert get_signal_history("yt_sentiment")[0] == 0.7

    def test_apply_with_enough_history_populates_alpha(self):
        from strategies.war_room_council_feeds import (
            CouncilFeedResult,
            apply_council_to_indicators,
        )
        clear_signal_history()
        # Feed enough history so alpha engine activates
        for i in range(10):
            record_council_observation("yt_sentiment", 0.5 + i * 0.01)
            record_council_observation("x_sentiment", 0.4 + i * 0.02)
            record_council_observation("yt_severity", 0.1 + i * 0.005)
            record_council_observation("x_severity", 0.15 + i * 0.01)

        yt = CouncilFeedResult(yt_videos_processed=3, yt_sentiment_score=0.6,
                               yt_severity_score=0.15, yt_trust_score=0.5)
        x = CouncilFeedResult(x_posts_analyzed=10, x_sentiment_score=0.6,
                              x_severity_score=0.25, x_trust_score=0.4)
        changes = apply_council_to_indicators(yt, x)
        assert "alpha_signal" in changes
        assert "alpha_weights" in changes
        assert isinstance(changes["alpha_weights"], dict)

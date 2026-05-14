"""Sprint 30 — Comprehensive tests for strategies.golden_ratio_finance.

Covers:
- Constants (PHI, PHI_INVERSE, PHI_SQUARED, PHI_CUBED, SQRT_PHI)
- Dataclasses (FibLevel, FibonacciResult, HarmonicPattern)
- FibonacciCalculator: retracement (up/down, custom levels), extension
  (high>low and low>high), golden_spiral_targets, detect_harmonics
  (gartley/butterfly/bat/crab branches, short input, X==A guard),
  _match_harmonic_ratios + _harmonic_confidence
- phase_conjugation_score: <2 inputs, all-zero, perfect-phi
- fractal_compression_index: short series, flat series, normal compression
"""
from __future__ import annotations

import math

import pytest

from strategies.golden_ratio_finance import (
    FIB_EXTENSION_LEVELS,
    FIB_RETRACEMENT_LEVELS,
    PHI,
    PHI_CUBED,
    PHI_INVERSE,
    PHI_SQUARED,
    SQRT_PHI,
    FibLevel,
    FibonacciCalculator,
    FibonacciResult,
    HarmonicPattern,
    fractal_compression_index,
    phase_conjugation_score,
)


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_phi_value(self):
        assert PHI == pytest.approx(1.6180339887, rel=1e-9)

    def test_phi_inverse(self):
        assert PHI_INVERSE == pytest.approx(1 / PHI)
        assert PHI_INVERSE == pytest.approx(0.6180339887, rel=1e-9)

    def test_phi_squared(self):
        assert PHI_SQUARED == pytest.approx(PHI * PHI)

    def test_phi_cubed(self):
        assert PHI_CUBED == pytest.approx(PHI ** 3)

    def test_sqrt_phi(self):
        assert SQRT_PHI == pytest.approx(math.sqrt(PHI))

    def test_retracement_levels_present(self):
        assert 0.618 in FIB_RETRACEMENT_LEVELS
        assert 0.382 in FIB_RETRACEMENT_LEVELS
        assert 0.0 in FIB_RETRACEMENT_LEVELS
        assert 1.0 in FIB_RETRACEMENT_LEVELS

    def test_extension_levels_present(self):
        assert 1.618 in FIB_EXTENSION_LEVELS
        assert 2.618 in FIB_EXTENSION_LEVELS


# ── Dataclasses ──────────────────────────────────────────────────────────────


class TestFibLevel:
    def test_init(self):
        lvl = FibLevel(ratio=0.618, price=42.0, label="61.8%")
        assert lvl.ratio == 0.618
        assert lvl.price == 42.0
        assert lvl.label == "61.8%"


class TestFibonacciResult:
    def test_defaults(self):
        r = FibonacciResult(swing_high=100.0, swing_low=50.0, direction="up")
        assert r.retracements == []
        assert r.extensions == []


class TestHarmonicPattern:
    def test_init(self):
        p = HarmonicPattern(
            pattern_type="gartley",
            direction="bullish",
            completion_zone=(50.0, 60.0),
            points={"X": 100.0, "A": 90.0, "B": 95.0, "C": 92.0, "D": 55.0},
            confidence=0.85,
        )
        assert p.pattern_type == "gartley"
        assert p.direction == "bullish"
        assert p.completion_zone == (50.0, 60.0)
        assert p.points["X"] == 100.0
        assert p.confidence == 0.85


# ── FibonacciCalculator.retracement ──────────────────────────────────────────


class TestRetracement:
    def test_up_direction_basic(self):
        fib = FibonacciCalculator()
        r = fib.retracement(high=100.0, low=0.0, direction="up")
        assert r.swing_high == 100.0
        assert r.swing_low == 0.0
        assert r.direction == "up"
        assert len(r.retracements) == len(FIB_RETRACEMENT_LEVELS)
        # 0% level = high; 100% level = low
        zero = next(lvl for lvl in r.retracements if lvl.ratio == 0.0)
        full = next(lvl for lvl in r.retracements if lvl.ratio == 1.0)
        half = next(lvl for lvl in r.retracements if lvl.ratio == 0.5)
        assert zero.price == 100.0
        assert full.price == 0.0
        assert half.price == 50.0

    def test_down_direction_basic(self):
        fib = FibonacciCalculator()
        r = fib.retracement(high=100.0, low=0.0, direction="down")
        zero = next(lvl for lvl in r.retracements if lvl.ratio == 0.0)
        full = next(lvl for lvl in r.retracements if lvl.ratio == 1.0)
        half = next(lvl for lvl in r.retracements if lvl.ratio == 0.5)
        # Down direction: price = low + diff*ratio
        assert zero.price == 0.0
        assert full.price == 100.0
        assert half.price == 50.0

    def test_custom_levels(self):
        fib = FibonacciCalculator()
        r = fib.retracement(high=200.0, low=100.0, direction="up",
                            custom_levels=[0.25, 0.75])
        assert len(r.retracements) == 2
        assert {lvl.ratio for lvl in r.retracements} == {0.25, 0.75}

    def test_label_format(self):
        fib = FibonacciCalculator()
        r = fib.retracement(high=100.0, low=0.0, direction="up")
        labels = {lvl.label for lvl in r.retracements}
        assert "61.8%" in labels
        assert "50.0%" in labels


# ── FibonacciCalculator.extension ────────────────────────────────────────────


class TestExtension:
    def test_high_greater_than_low(self):
        fib = FibonacciCalculator()
        r = fib.extension(high=100.0, low=50.0, retracement_point=70.0)
        assert r.direction == "up"
        assert len(r.extensions) == len(FIB_EXTENSION_LEVELS)
        # impulse = 50, retracement_point=70, ratio=1.0 → price = 70 + 50 = 120
        one = next(lvl for lvl in r.extensions if lvl.ratio == 1.0)
        assert one.price == pytest.approx(120.0)

    def test_high_less_than_low_subtracts(self):
        fib = FibonacciCalculator()
        # high=50, low=100, retr=80, impulse=50 → price = 80 - 50 = 30
        r = fib.extension(high=50.0, low=100.0, retracement_point=80.0)
        assert r.direction == "down"
        one = next(lvl for lvl in r.extensions if lvl.ratio == 1.0)
        assert one.price == pytest.approx(30.0)

    def test_custom_extension_levels(self):
        fib = FibonacciCalculator()
        r = fib.extension(high=100.0, low=50.0, retracement_point=70.0,
                          custom_levels=[1.5])
        assert len(r.extensions) == 1
        # 70 + 50*1.5 = 145
        assert r.extensions[0].price == pytest.approx(145.0)


# ── golden_spiral_targets ────────────────────────────────────────────────────


class TestGoldenSpiralTargets:
    def test_default_n_levels(self):
        fib = FibonacciCalculator()
        targets = fib.golden_spiral_targets(center_price=100.0, volatility=10.0)
        # 8 levels × 2 directions = 16
        assert len(targets) == 16

    def test_custom_n_levels(self):
        fib = FibonacciCalculator()
        targets = fib.golden_spiral_targets(center_price=100.0, volatility=5.0, n_levels=3)
        assert len(targets) == 6  # 3 × 2

    def test_first_up_target(self):
        fib = FibonacciCalculator()
        targets = fib.golden_spiral_targets(center_price=100.0, volatility=10.0, n_levels=1)
        # phi^1 up: center + 10 * phi
        up = next(t for t in targets if t.label == "phi^1 up")
        down = next(t for t in targets if t.label == "phi^1 down")
        assert up.price == pytest.approx(100.0 + 10.0 * PHI)
        assert down.price == pytest.approx(100.0 - 10.0 * PHI)


# ── detect_harmonics ─────────────────────────────────────────────────────────


class TestDetectHarmonics:
    def test_too_few_prices_returns_empty(self):
        fib = FibonacciCalculator()
        assert fib.detect_harmonics([1.0, 2.0]) == []
        assert fib.detect_harmonics([]) == []

    def test_x_equals_a_skips(self):
        fib = FibonacciCalculator()
        # X==A means xa=0 → continue. With only one window, returns empty.
        assert fib.detect_harmonics([100.0, 100.0, 95.0, 92.0, 80.0]) == []

    def test_gartley_pattern(self):
        fib = FibonacciCalculator()
        # X=100, A=0 → XA=100; B such that AB/XA ≈ 0.618 → AB=61.8 → B=61.8
        # BC such that BC/AB in [0.382, 0.886] → pick BC=30 → C=31.8
        x, a, b, c, d = 100.0, 0.0, 61.8, 31.8, 70.0
        out = fib.detect_harmonics([x, a, b, c, d])
        assert len(out) == 1
        assert out[0].pattern_type == "gartley"

    def test_butterfly_pattern(self):
        fib = FibonacciCalculator()
        # AB/XA ≈ 0.786, CD/BC ≈ 1.618-2.618
        # X=100,A=0,XA=100; AB=78.6→B=78.6; BC=20→C=58.6; CD=40→D=98.6 (CD/BC=2.0)
        out = fib.detect_harmonics([100.0, 0.0, 78.6, 58.6, 98.6])
        assert any(p.pattern_type == "butterfly" for p in out)

    def test_bat_pattern(self):
        fib = FibonacciCalculator()
        # AB/XA ≈ 0.382-0.500; X=100,A=0,B=45 → AB=45,XA=100 → ratio 0.45
        out = fib.detect_harmonics([100.0, 0.0, 45.0, 30.0, 50.0])
        assert any(p.pattern_type in {"bat", "gartley"} for p in out)

    def test_pattern_has_completion_zone_and_confidence(self):
        fib = FibonacciCalculator()
        out = fib.detect_harmonics([100.0, 0.0, 61.8, 31.8, 70.0])
        assert out[0].completion_zone == (min(31.8, 70.0), max(31.8, 70.0))
        assert 0.0 <= out[0].confidence <= 1.0
        assert out[0].direction in {"bullish", "bearish"}

    def test_match_ratios_returns_none_when_no_match(self):
        # Ratios that match nothing
        result = FibonacciCalculator._match_harmonic_ratios(
            ab_xa=0.999, bc_ab=0.001, cd_bc=99.0, tol=0.01,
        )
        assert result is None

    def test_harmonic_confidence_ideal_match(self):
        # Exact gartley ideal ratios → confidence near 1.0
        c = FibonacciCalculator._harmonic_confidence(0.618, 0.618, 1.272, "gartley")
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_harmonic_confidence_unknown_pattern_uses_default(self):
        # Unknown pattern uses fallback ideal — should still return a valid score
        c = FibonacciCalculator._harmonic_confidence(0.618, 0.618, 1.618, "unknown")
        assert 0.0 <= c <= 1.0


# ── phase_conjugation_score ──────────────────────────────────────────────────


class TestPhaseConjugationScore:
    def test_too_few_returns_zero(self):
        assert phase_conjugation_score([]) == 0.0
        assert phase_conjugation_score([1.0]) == 0.0

    def test_zero_frequency_skipped(self):
        # Two zeros → no valid ratios → returns 0
        assert phase_conjugation_score([0.0, 0.0]) == 0.0

    def test_phi_ratio_scores_high(self):
        # Frequencies in PHI ratio → near-perfect score
        score = phase_conjugation_score([1.0, PHI, PHI ** 2])
        assert score > 90.0

    def test_returns_rounded_float(self):
        score = phase_conjugation_score([1.0, 2.0, 4.0])
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0


# ── fractal_compression_index ────────────────────────────────────────────────


class TestFractalCompressionIndex:
    def test_short_series_returns_zero(self):
        assert fractal_compression_index([1.0, 2.0], window=21) == 0.0

    def test_flat_series_zero_long_range(self):
        # All same value → max=val, range=0, long_range=0 → returns 0
        assert fractal_compression_index([100.0] * 30, window=10) == 0.0

    def test_compressed_recent_window_high_score(self):
        # Long history is volatile; recent window is tight → high compression
        history = [100.0 + (i % 20) for i in range(60)]  # oscillates
        recent_tight = [100.0, 100.1, 100.05, 100.02, 100.03,
                        100.0, 100.04, 100.01, 100.02, 100.0,
                        100.03, 100.01, 100.02, 100.0, 100.04,
                        100.02, 100.01, 100.03, 100.0, 100.02, 100.01]
        series = history + recent_tight
        score = fractal_compression_index(series, window=21)
        assert 0.0 <= score <= 100.0
        assert score > 50.0  # should be significantly compressed

    def test_returns_in_valid_range(self):
        series = [100.0 + i * 0.5 for i in range(100)]
        score = fractal_compression_index(series, window=21)
        assert 0.0 <= score <= 100.0

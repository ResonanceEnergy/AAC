"""Sprint 29 — Comprehensive tests for strategies.gamma_exposure_tracker.

Covers every public class and the key private helpers:
- VolatilityRegime enum values
- OptionOI dataclass + total_oi / put_call_ratio properties
- GEXLevel dataclass + is_positive property
- GEXProfile dataclass + to_dict
- DealerExposure dataclass defaults
- GammaExposureEngine: add_oi_data, compute_gex_at_strike,
  compute_gex_profile, _find_flip_level, _find_put_wall,
  _find_call_wall, _determine_regime
- DealerHedgingTracker: estimate_hedging_flow,
  get_support_resistance, opex_week_analysis,
  _opex_recommendation
- VolatilitySurfaceAnalyzer: compute_skew, detect_iv_crush_opportunity
- PutCallAnalyzer.analyze (every sentiment branch)
"""
from __future__ import annotations

import math

import pytest

from strategies.gamma_exposure_tracker import (
    DealerExposure,
    DealerHedgingTracker,
    GammaExposureEngine,
    GEXLevel,
    GEXProfile,
    OptionOI,
    PutCallAnalyzer,
    VolatilityRegime,
    VolatilitySurfaceAnalyzer,
)


# ── VolatilityRegime ─────────────────────────────────────────────────────────


class TestVolatilityRegime:
    def test_suppressed_value(self):
        assert VolatilityRegime.SUPPRESSED.value == "suppressed"

    def test_amplified_value(self):
        assert VolatilityRegime.AMPLIFIED.value == "amplified"

    def test_transitional_value(self):
        assert VolatilityRegime.TRANSITIONAL.value == "transitional"


# ── OptionOI ─────────────────────────────────────────────────────────────────


class TestOptionOI:
    def test_defaults(self):
        oi = OptionOI(strike=100.0, expiry="2026-05-15")
        assert oi.strike == 100.0
        assert oi.call_oi == 0
        assert oi.put_oi == 0
        assert oi.call_volume == 0

    def test_total_oi_sum(self):
        oi = OptionOI(strike=100.0, expiry="2026-05-15", call_oi=500, put_oi=300)
        assert oi.total_oi == 800

    def test_put_call_ratio_normal(self):
        oi = OptionOI(strike=100.0, expiry="2026-05-15", call_oi=200, put_oi=100)
        assert oi.put_call_ratio == pytest.approx(0.5)

    def test_put_call_ratio_zero_calls_returns_inf(self):
        oi = OptionOI(strike=100.0, expiry="2026-05-15", call_oi=0, put_oi=10)
        assert oi.put_call_ratio == float("inf")


# ── GEXLevel ─────────────────────────────────────────────────────────────────


class TestGEXLevel:
    def test_defaults(self):
        lvl = GEXLevel(strike=100.0)
        assert lvl.call_gex == 0.0
        assert lvl.put_gex == 0.0
        assert lvl.net_gex == 0.0
        assert lvl.is_positive is False

    def test_is_positive_true(self):
        lvl = GEXLevel(strike=100.0, net_gex=5000.0)
        assert lvl.is_positive is True

    def test_is_positive_false_at_zero(self):
        lvl = GEXLevel(strike=100.0, net_gex=0.0)
        assert lvl.is_positive is False


# ── GEXProfile ───────────────────────────────────────────────────────────────


class TestGEXProfile:
    def test_defaults(self):
        p = GEXProfile(symbol="SPY", underlying_price=520.0, timestamp="t")
        assert p.symbol == "SPY"
        assert p.levels == []
        assert p.regime == VolatilityRegime.TRANSITIONAL

    def test_to_dict_keys(self):
        p = GEXProfile(symbol="SPY", underlying_price=520.0, timestamp="t",
                       total_gex=1000.0, flip_level=520.0, max_gamma_strike=520.0,
                       put_wall=510.0, call_wall=530.0)
        d = p.to_dict()
        assert set(d.keys()) == {
            "symbol", "underlying_price", "total_gex", "flip_level",
            "max_gamma_strike", "regime", "put_wall", "call_wall",
            "num_levels", "timestamp",
        }
        assert d["regime"] == "transitional"
        assert d["num_levels"] == 0


# ── DealerExposure ───────────────────────────────────────────────────────────


class TestDealerExposure:
    def test_defaults(self):
        de = DealerExposure()
        assert de.net_delta == 0.0
        assert de.net_gamma == 0.0
        assert de.hedging_pressure == ""
        assert de.estimated_shares == 0


# ── GammaExposureEngine ──────────────────────────────────────────────────────


def _build_demo_oi(spot: float = 520.0) -> list[OptionOI]:
    """Synthetic OI chain peaked at ATM (mirrors module's __main__ block)."""
    chain: list[OptionOI] = []
    for strike in range(int(spot) - 30, int(spot) + 35, 5):
        dist = abs(strike - spot)
        oi_factor = max(1, 30 - dist) * 1000
        gamma_val = 0.003 * math.exp(-(dist ** 2) / 200.0)
        chain.append(OptionOI(
            strike=float(strike),
            expiry="2026-07-18",
            call_oi=int(oi_factor * 0.7),
            put_oi=int(oi_factor * 1.1),
            call_gamma=gamma_val,
            put_gamma=gamma_val * 0.9,
        ))
    return chain


class TestGammaExposureEngine:
    def test_init(self):
        eng = GammaExposureEngine("SPY", 520.0)
        assert eng.symbol == "SPY"
        assert eng.S == 520.0
        assert eng.oi_data == []

    def test_add_oi_data(self):
        eng = GammaExposureEngine("SPY", 520.0)
        chain = _build_demo_oi()
        eng.add_oi_data(chain)
        assert eng.oi_data == chain

    def test_compute_gex_at_strike(self):
        eng = GammaExposureEngine("SPY", 520.0)
        oi = OptionOI(strike=520.0, expiry="x", call_oi=1000, put_oi=1000,
                      call_gamma=0.003, put_gamma=0.003)
        lvl = eng.compute_gex_at_strike(oi)
        # Both gex values are negative by convention
        assert lvl.call_gex < 0
        assert lvl.put_gex < 0
        assert lvl.net_gex == lvl.call_gex + lvl.put_gex
        assert lvl.strike == 520.0

    def test_compute_gex_at_strike_zero_oi(self):
        eng = GammaExposureEngine("SPY", 520.0)
        oi = OptionOI(strike=520.0, expiry="x")
        lvl = eng.compute_gex_at_strike(oi)
        assert lvl.net_gex == 0.0

    def test_compute_gex_profile_basic(self):
        eng = GammaExposureEngine("SPY", 520.0)
        eng.add_oi_data(_build_demo_oi())
        profile = eng.compute_gex_profile()
        assert profile.symbol == "SPY"
        assert profile.underlying_price == 520.0
        assert len(profile.levels) > 0
        # Strikes are sorted ascending
        strikes = [lvl.strike for lvl in profile.levels]
        assert strikes == sorted(strikes)
        # Regime is one of the enum values
        assert isinstance(profile.regime, VolatilityRegime)

    def test_compute_gex_profile_empty_chain(self):
        eng = GammaExposureEngine("SPY", 520.0)
        profile = eng.compute_gex_profile()
        assert profile.levels == []
        assert profile.total_gex == 0.0
        # Empty chain => max_gamma_strike defaults to 0
        assert profile.max_gamma_strike == 0

    def test_find_flip_level_no_crossover_returns_spot(self):
        eng = GammaExposureEngine("SPY", 520.0)
        # All levels positive — no flip
        levels = [GEXLevel(strike=510.0, net_gex=100.0),
                  GEXLevel(strike=515.0, net_gex=200.0),
                  GEXLevel(strike=520.0, net_gex=300.0)]
        assert eng._find_flip_level(levels) == 520.0

    def test_find_flip_level_with_crossover(self):
        eng = GammaExposureEngine("SPY", 520.0)
        levels = [GEXLevel(strike=510.0, net_gex=100.0),
                  GEXLevel(strike=520.0, net_gex=-100.0)]
        flip = eng._find_flip_level(levels)
        # Linear interpolation puts the flip exactly at midpoint
        assert flip == pytest.approx(515.0)

    def test_find_put_wall_returns_strike_below_spot(self):
        eng = GammaExposureEngine("SPY", 520.0)
        levels = [GEXLevel(strike=510.0, put_gex=-5000.0, net_gex=-5000.0),
                  GEXLevel(strike=515.0, put_gex=-1000.0, net_gex=-1000.0),
                  GEXLevel(strike=525.0, put_gex=-9000.0, net_gex=-9000.0)]
        # 525 is above spot, should be excluded → put_wall is 510 (most negative below spot)
        assert eng._find_put_wall(levels) == 510.0

    def test_find_put_wall_no_negative_returns_zero(self):
        eng = GammaExposureEngine("SPY", 520.0)
        levels = [GEXLevel(strike=510.0, put_gex=100.0)]
        assert eng._find_put_wall(levels) == 0.0

    def test_find_call_wall_returns_strike_above_spot(self):
        eng = GammaExposureEngine("SPY", 520.0)
        levels = [GEXLevel(strike=525.0, call_gex=-3000.0, net_gex=-3000.0),
                  GEXLevel(strike=530.0, call_gex=-7000.0, net_gex=-7000.0)]
        # Most negative above spot
        assert eng._find_call_wall(levels) == 530.0

    def test_find_call_wall_no_negative_returns_zero(self):
        eng = GammaExposureEngine("SPY", 520.0)
        assert eng._find_call_wall([GEXLevel(strike=530.0, call_gex=100.0)]) == 0.0

    def test_determine_regime_transitional_near_flip(self):
        eng = GammaExposureEngine("SPY", 520.0)
        # Within 0.5% of flip
        assert eng._determine_regime(0.0, 520.5) == VolatilityRegime.TRANSITIONAL

    def test_determine_regime_suppressed_when_total_positive(self):
        eng = GammaExposureEngine("SPY", 520.0)
        # Far from flip, total_gex > 0
        assert eng._determine_regime(1000.0, 600.0) == VolatilityRegime.SUPPRESSED

    def test_determine_regime_amplified_when_negative_above_flip(self):
        eng = GammaExposureEngine("SPY", 520.0)
        # Far above flip, total_gex < 0
        assert eng._determine_regime(-1000.0, 400.0) == VolatilityRegime.AMPLIFIED


# ── DealerHedgingTracker ─────────────────────────────────────────────────────


def _profile_with_total(total_gex: float, levels: list[GEXLevel] | None = None,
                        spot: float = 520.0) -> GEXProfile:
    return GEXProfile(
        symbol="SPY",
        underlying_price=spot,
        timestamp="t",
        levels=levels or [],
        total_gex=total_gex,
        flip_level=spot,
        max_gamma_strike=spot,
        regime=VolatilityRegime.TRANSITIONAL,
        put_wall=spot - 10.0,
        call_wall=spot + 10.0,
    )


class TestDealerHedgingTracker:
    def test_estimate_hedging_flow_positive_gamma_rally_sells(self):
        tracker = DealerHedgingTracker(_profile_with_total(1000.0))
        out = tracker.estimate_hedging_flow(price_change_pct=1.0)
        assert isinstance(out, DealerExposure)
        assert out.hedging_pressure == "selling"
        assert out.estimated_shares >= 0

    def test_estimate_hedging_flow_positive_gamma_dip_buys(self):
        tracker = DealerHedgingTracker(_profile_with_total(1000.0))
        out = tracker.estimate_hedging_flow(price_change_pct=-1.0)
        assert out.hedging_pressure == "buying"

    def test_estimate_hedging_flow_negative_gamma_rally_buys(self):
        tracker = DealerHedgingTracker(_profile_with_total(-1000.0))
        out = tracker.estimate_hedging_flow(price_change_pct=1.0)
        assert out.hedging_pressure == "buying"

    def test_estimate_hedging_flow_negative_gamma_dip_sells(self):
        tracker = DealerHedgingTracker(_profile_with_total(-1000.0))
        out = tracker.estimate_hedging_flow(price_change_pct=-1.0)
        assert out.hedging_pressure == "selling"

    def test_get_support_resistance_partitions_by_spot(self):
        levels = [
            GEXLevel(strike=510.0, net_gex=-1000.0),  # below spot, neg → support
            GEXLevel(strike=515.0, net_gex=-500.0),
            GEXLevel(strike=525.0, net_gex=2000.0),   # above spot, pos → resistance
            GEXLevel(strike=530.0, net_gex=3000.0),
        ]
        tracker = DealerHedgingTracker(_profile_with_total(0.0, levels=levels))
        sr = tracker.get_support_resistance()
        assert 510.0 in sr["support"]
        assert 515.0 in sr["support"]
        assert 525.0 in sr["resistance"]
        assert 530.0 in sr["resistance"]
        assert "put_wall" in sr
        assert "call_wall" in sr
        assert "flip_level" in sr

    def test_get_support_resistance_top_5_only(self):
        # Build 10 support and 10 resistance levels
        levels = []
        for i in range(10):
            levels.append(GEXLevel(strike=500.0 - i, net_gex=-(i + 1) * 100.0))
            levels.append(GEXLevel(strike=540.0 + i, net_gex=(i + 1) * 100.0))
        tracker = DealerHedgingTracker(_profile_with_total(0.0, levels=levels))
        sr = tracker.get_support_resistance()
        assert len(sr["support"]) == 5
        assert len(sr["resistance"]) == 5

    def test_opex_week_analysis_critical_2_dte(self):
        tracker = DealerHedgingTracker(_profile_with_total(0.0))
        out = tracker.opex_week_analysis(days_to_opex=1)
        assert out["urgency"] == "CRITICAL"
        assert out["expected_pinning"] is True
        assert out["gamma_amplification"] > 1.0

    def test_opex_week_analysis_high_5_dte(self):
        tracker = DealerHedgingTracker(_profile_with_total(0.0))
        out = tracker.opex_week_analysis(days_to_opex=5)
        assert out["urgency"] == "HIGH"

    def test_opex_week_analysis_moderate_10_dte(self):
        tracker = DealerHedgingTracker(_profile_with_total(0.0))
        out = tracker.opex_week_analysis(days_to_opex=10)
        assert out["urgency"] == "MODERATE"

    def test_opex_week_analysis_low_far(self):
        tracker = DealerHedgingTracker(_profile_with_total(0.0))
        out = tracker.opex_week_analysis(days_to_opex=30)
        assert out["urgency"] == "LOW"
        assert out["expected_pinning"] is False
        assert out["gamma_amplification"] == pytest.approx(1.0)

    def test_opex_recommendation_branches(self):
        tracker = DealerHedgingTracker(_profile_with_total(0.0))
        assert "0-1 DTE" in tracker._opex_recommendation(0)
        assert "0-1 DTE" in tracker._opex_recommendation(1)
        assert "2-3 DTE" in tracker._opex_recommendation(3)
        assert "4-7 DTE" in tracker._opex_recommendation(7)
        assert "7+" in tracker._opex_recommendation(10)


# ── VolatilitySurfaceAnalyzer ────────────────────────────────────────────────


class TestVolatilitySurfaceAnalyzer:
    def test_compute_skew_basic(self):
        chain = [
            OptionOI(strike=100.0, expiry="x", call_iv=0.20, put_iv=0.22),
            OptionOI(strike=110.0, expiry="x", call_iv=0.18, put_iv=0.0),
            OptionOI(strike=90.0,  expiry="x", call_iv=0.0,  put_iv=0.30),
        ]
        out = VolatilitySurfaceAnalyzer.compute_skew(chain, atm_strike=100.0)
        assert "atm_iv" in out
        assert "risk_reversal" in out
        assert "butterfly" in out
        assert out["put_skew"] in {"steep", "flat"}
        assert isinstance(out["tail_risk_priced"], bool)

    def test_compute_skew_steep_when_puts_expensive(self):
        chain = [
            OptionOI(strike=100.0, expiry="x", call_iv=0.20, put_iv=0.20),
            OptionOI(strike=110.0, expiry="x", call_iv=0.18, put_iv=0.0),  # OTM call low IV
            OptionOI(strike=90.0,  expiry="x", call_iv=0.0,  put_iv=0.40),  # OTM put high IV
        ]
        out = VolatilitySurfaceAnalyzer.compute_skew(chain, atm_strike=100.0)
        # Risk reversal = call_iv - put_iv = 0.18 - 0.40 = -0.22 → steep
        assert out["put_skew"] == "steep"

    def test_compute_skew_empty_chain(self):
        out = VolatilitySurfaceAnalyzer.compute_skew([], atm_strike=100.0)
        assert out["atm_iv"] == 0.0
        assert out["risk_reversal"] == 0.0
        assert out["butterfly"] == 0.0

    def test_detect_iv_crush_event_passed(self):
        out = VolatilitySurfaceAnalyzer.detect_iv_crush_opportunity(
            current_iv=0.50, historical_iv_mean=0.30, event_date_dte=0,
        )
        assert out["opportunity"] is False

    def test_detect_iv_crush_high_iv_recommends_short_vol(self):
        out = VolatilitySurfaceAnalyzer.detect_iv_crush_opportunity(
            current_iv=0.50, historical_iv_mean=0.30, event_date_dte=3,
        )
        assert out["crush_expected"] is True
        assert "Short" in out["recommended_strategy"]
        assert out["iv_premium_pct"] > 0
        assert out["days_to_event"] == 3

    def test_detect_iv_crush_low_iv_recommends_calendar(self):
        out = VolatilitySurfaceAnalyzer.detect_iv_crush_opportunity(
            current_iv=0.20, historical_iv_mean=0.25, event_date_dte=5,
        )
        assert out["crush_expected"] is False
        assert "Calendar" in out["recommended_strategy"]

    def test_detect_iv_crush_oversized_extreme_iv(self):
        out = VolatilitySurfaceAnalyzer.detect_iv_crush_opportunity(
            current_iv=2.0, historical_iv_mean=1.0, event_date_dte=2,
        )
        # 100% premium → oversized
        assert "OVERSIZED" in out["sizing"]


# ── PutCallAnalyzer ──────────────────────────────────────────────────────────


class TestPutCallAnalyzer:
    def test_extreme_fear(self):
        out = PutCallAnalyzer.analyze(
            put_volume=1000, call_volume=1000,  # vol_pc = 1.0
            put_oi=500, call_oi=500,
        )
        assert out["sentiment"] == "EXTREME_FEAR"
        assert out["contrarian_signal"] == "BULLISH"

    def test_fear(self):
        out = PutCallAnalyzer.analyze(
            put_volume=700, call_volume=1000,  # 0.7
            put_oi=500, call_oi=500,
        )
        assert out["sentiment"] == "FEAR"
        assert out["contrarian_signal"] == "MILDLY_BULLISH"

    def test_extreme_greed(self):
        out = PutCallAnalyzer.analyze(
            put_volume=300, call_volume=1000,  # 0.3
            put_oi=500, call_oi=500,
        )
        assert out["sentiment"] == "EXTREME_GREED"
        assert out["contrarian_signal"] == "BEARISH"

    def test_greed(self):
        out = PutCallAnalyzer.analyze(
            put_volume=450, call_volume=1000,  # 0.45
            put_oi=500, call_oi=500,
        )
        assert out["sentiment"] == "GREED"
        assert out["contrarian_signal"] == "MILDLY_BEARISH"

    def test_neutral(self):
        out = PutCallAnalyzer.analyze(
            put_volume=550, call_volume=1000,  # 0.55
            put_oi=500, call_oi=500,
        )
        assert out["sentiment"] == "NEUTRAL"
        assert out["contrarian_signal"] == "NEUTRAL"

    def test_zero_call_volume_safe(self):
        out = PutCallAnalyzer.analyze(
            put_volume=100, call_volume=0,
            put_oi=100, call_oi=0,
        )
        # Doesn't crash; volume_pc is 0 → NEUTRAL
        assert "sentiment" in out
        assert out["volume_pc_ratio"] == 0
        assert out["oi_pc_ratio"] == 0

    def test_volume_oi_divergence_flagged(self):
        out = PutCallAnalyzer.analyze(
            put_volume=900, call_volume=1000,  # vol_pc = 0.9
            put_oi=200, call_oi=1000,           # oi_pc  = 0.2 → diff > 0.15
        )
        assert out["volume_oi_divergence"] is True
        assert "diverging" in out["divergence_note"]

    def test_no_divergence(self):
        out = PutCallAnalyzer.analyze(
            put_volume=550, call_volume=1000,  # vol_pc = 0.55
            put_oi=550, call_oi=1000,           # oi_pc  = 0.55 → diff = 0
        )
        assert out["volume_oi_divergence"] is False

    def test_keys_present(self):
        out = PutCallAnalyzer.analyze(1, 1, 1, 1)
        assert set(out.keys()) == {
            "volume_pc_ratio", "oi_pc_ratio", "historical_avg",
            "sentiment", "contrarian_signal", "volume_oi_divergence",
            "divergence_note",
        }

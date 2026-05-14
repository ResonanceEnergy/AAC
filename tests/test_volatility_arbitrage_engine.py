from __future__ import annotations

import math

import pytest

from strategies.volatility_arbitrage_engine import (
    SkewAnalyzer,
    SkewRegime,
    TermStructure,
    TermStructureAnalyzer,
    TermStructureSignal,
    VarianceRiskPremiumEngine,
    VolRegime,
    VolRegimeDetector,
    VolSurface,
    VRPSignal,
)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class TestEnums:
    def test_vol_regime_values(self):
        assert VolRegime.LOW.value == "low_vol"
        assert VolRegime.NORMAL.value == "normal_vol"
        assert VolRegime.ELEVATED.value == "elevated_vol"
        assert VolRegime.HIGH.value == "high_vol"
        assert VolRegime.EXTREME.value == "extreme_vol"

    def test_term_structure_values(self):
        assert TermStructure.CONTANGO.value == "contango"
        assert TermStructure.FLAT.value == "flat"
        assert TermStructure.BACKWARDATION.value == "backwardation"

    def test_skew_regime_values(self):
        assert SkewRegime.NORMAL.value == "normal"
        assert SkewRegime.STEEP.value == "steep"
        assert SkewRegime.FLAT.value == "flat"
        assert SkewRegime.INVERTED.value == "inverted"


# ═══════════════════════════════════════════════════════════════════════════
# VolSurface
# ═══════════════════════════════════════════════════════════════════════════

class TestVolSurface:
    def test_defaults(self):
        s = VolSurface(symbol="SPY", underlying_price=500)
        assert s.symbol == "SPY"
        assert s.underlying_price == 500
        assert s.timestamp == ""
        assert s.atm_term_structure == {}
        assert s.skew_by_expiry == {}
        assert s.rv_5d == 0.0
        assert s.rv_60d == 0.0

    def test_front_iv_empty(self):
        s = VolSurface(symbol="SPY", underlying_price=500)
        assert s.front_iv == 0

    def test_back_iv_empty(self):
        s = VolSurface(symbol="SPY", underlying_price=500)
        assert s.back_iv == 0

    def test_front_iv_picks_min_dte(self):
        s = VolSurface(
            symbol="SPY", underlying_price=500,
            atm_term_structure={30: 0.20, 7: 0.18, 60: 0.21},
        )
        assert s.front_iv == 0.18

    def test_back_iv_picks_max_dte(self):
        s = VolSurface(
            symbol="SPY", underlying_price=500,
            atm_term_structure={30: 0.20, 7: 0.18, 60: 0.21},
        )
        assert s.back_iv == 0.21


# ═══════════════════════════════════════════════════════════════════════════
# VRPSignal / TermStructureSignal dataclasses
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalDataclasses:
    def test_vrp_signal_construction(self):
        sig = VRPSignal(
            iv_current=0.20, rv_current=0.15, vrp=5.0, vrp_percentile=80,
            z_score=1.5, signal="SELL_VOL", confidence=75,
            recommended_strategy="x",
        )
        assert sig.signal == "SELL_VOL"
        assert sig.confidence == 75

    def test_term_structure_signal_construction(self):
        sig = TermStructureSignal(
            front_dte=7, front_iv=0.25, back_dte=60, back_iv=0.20,
            spread=-0.05, spread_pct=-20, structure=TermStructure.BACKWARDATION,
            signal="SELL_FRONT_BUY_BACK", recommended_strategy="x",
        )
        assert sig.structure == TermStructure.BACKWARDATION


# ═══════════════════════════════════════════════════════════════════════════
# VarianceRiskPremiumEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestVarianceRiskPremiumEngine:
    def test_init_no_history(self):
        e = VarianceRiskPremiumEngine()
        assert e.vrp_history == []

    def test_init_with_history(self):
        e = VarianceRiskPremiumEngine(vrp_history=[1.0, 2.0])
        assert e.vrp_history == [1.0, 2.0]

    def test_no_history_neutral_percentile(self):
        e = VarianceRiskPremiumEngine()
        sig = e.compute_vrp(iv_30d=0.20, rv_20d=0.18, rv_5d=0.17)
        assert sig.vrp_percentile == 50
        assert sig.z_score == 0

    def test_strong_sell_vol_high_vrp_high_percentile(self):
        # vrp = 5.0, history all below → percentile=100
        e = VarianceRiskPremiumEngine(vrp_history=[0.0, 0.5, 1.0, 1.5, 2.0])
        sig = e.compute_vrp(iv_30d=6.0, rv_20d=1.0, rv_5d=0.5)
        assert sig.signal == "SELL_VOL"
        assert sig.vrp == 5.0
        assert sig.vrp_percentile == 100.0
        # confidence formula: min(95, 60 + 100*0.3) = 90
        assert sig.confidence == 90.0
        assert "Short Straddle" in sig.recommended_strategy

    def test_moderate_sell_vol(self):
        # vrp ~ 3.0, percentile high enough but vrp not > 4
        e = VarianceRiskPremiumEngine(vrp_history=[0.0, 1.0, 2.0])
        sig = e.compute_vrp(iv_30d=4.0, rv_20d=1.0, rv_5d=0.5)
        # vrp = 3.0, all history below → percentile=100, but vrp not > 4
        assert sig.signal == "SELL_VOL"
        assert sig.vrp == 3.0
        # confidence = min(80, 50 + 100*0.2) = 70
        assert sig.confidence == 70.0
        assert "Iron Condor" in sig.recommended_strategy

    def test_buy_vol_negative_vrp(self):
        e = VarianceRiskPremiumEngine(vrp_history=[1.0, 2.0])
        sig = e.compute_vrp(iv_30d=0.10, rv_20d=0.15, rv_5d=0.05)
        # vrp = -0.05, but rv_5d (0.05) NOT > iv_30d * 1.2 (0.12) → no override
        assert sig.signal == "BUY_VOL"
        assert sig.vrp == -0.05
        # confidence = min(85, 60 + 0.05*5) = 60.25
        assert sig.confidence == pytest.approx(60.25)

    def test_neutral_low_vrp(self):
        e = VarianceRiskPremiumEngine()
        sig = e.compute_vrp(iv_30d=0.20, rv_20d=0.195, rv_5d=0.19)
        # vrp=0.005 → < 1, > 0 → NEUTRAL "stand aside"
        assert sig.signal == "NEUTRAL"
        assert sig.confidence == 40

    def test_neutral_marginal_vrp(self):
        e = VarianceRiskPremiumEngine()
        # vrp = 1.5, percentile=50 (no history) → falls past first two branches
        sig = e.compute_vrp(iv_30d=2.5, rv_20d=1.0, rv_5d=0.9)
        assert sig.signal == "NEUTRAL"
        assert sig.confidence == 50
        assert "Small position" in sig.recommended_strategy

    def test_rv_spike_override(self):
        # Even with healthy vrp, if rv_5d > iv_30d*1.2 → flip to BUY_VOL
        e = VarianceRiskPremiumEngine()
        sig = e.compute_vrp(iv_30d=0.20, rv_20d=0.10, rv_5d=0.30)
        # vrp = 0.10, rv_5d=0.30 > 0.20*1.2=0.24 → override
        assert sig.signal == "BUY_VOL"
        assert sig.confidence >= 70
        assert "RV spiking" in sig.recommended_strategy

    def test_z_score_computation(self):
        e = VarianceRiskPremiumEngine(vrp_history=[1.0, 2.0, 3.0])
        sig = e.compute_vrp(iv_30d=4.0, rv_20d=2.0, rv_5d=1.0)
        # vrp=2.0, mean=2.0 → z=0
        assert sig.z_score == 0.0

    def test_z_score_zero_std(self):
        # All-equal history → std=0 → z=0 (no division by zero)
        e = VarianceRiskPremiumEngine(vrp_history=[2.0, 2.0, 2.0])
        sig = e.compute_vrp(iv_30d=5.0, rv_20d=2.0, rv_5d=1.0)
        assert sig.z_score == 0


# ═══════════════════════════════════════════════════════════════════════════
# TermStructureAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

def _surface_with_term(term: dict) -> VolSurface:
    return VolSurface(symbol="SPY", underlying_price=500, atm_term_structure=term)


class TestTermStructureAnalyzer:
    def test_insufficient_data(self):
        s = _surface_with_term({30: 0.20})
        a = TermStructureAnalyzer(s)
        assert a.analyze_structure() == {"structure": "insufficient_data"}

    def test_empty_term_structure(self):
        s = _surface_with_term({})
        a = TermStructureAnalyzer(s)
        assert a.analyze_structure() == {"structure": "insufficient_data"}

    def test_contango(self):
        # back > front by > 5%
        s = _surface_with_term({7: 0.18, 60: 0.22})
        a = TermStructureAnalyzer(s)
        result = a.analyze_structure()
        assert result["structure"] == "contango"
        assert result["front_dte"] == 7
        assert result["back_dte"] == 60
        assert result["spread"] == 0.04
        assert result["spread_pct"] == pytest.approx(22.22, rel=0.01)
        assert result["has_hump"] is False
        assert result["num_tenors"] == 2

    def test_backwardation(self):
        # front > back by > 5%
        s = _surface_with_term({7: 0.30, 60: 0.20})
        a = TermStructureAnalyzer(s)
        result = a.analyze_structure()
        assert result["structure"] == "backwardation"
        assert result["spread_pct"] == pytest.approx(-33.33, rel=0.01)

    def test_flat(self):
        # spread within ±5%
        s = _surface_with_term({7: 0.20, 60: 0.205})
        a = TermStructureAnalyzer(s)
        result = a.analyze_structure()
        assert result["structure"] == "flat"

    def test_has_hump_true(self):
        # belly higher than both ends
        s = _surface_with_term({7: 0.18, 30: 0.30, 60: 0.20})
        a = TermStructureAnalyzer(s)
        result = a.analyze_structure()
        assert result["has_hump"] is True

    def test_has_hump_false_with_3_tenors(self):
        s = _surface_with_term({7: 0.18, 30: 0.19, 60: 0.20})
        a = TermStructureAnalyzer(s)
        result = a.analyze_structure()
        assert result["has_hump"] is False

    def test_signal_insufficient_data(self):
        s = _surface_with_term({30: 0.20})
        sig = TermStructureAnalyzer(s).generate_signal()
        assert sig.signal == "NO_SIGNAL"
        assert sig.recommended_strategy == "Insufficient data"

    def test_signal_deep_backwardation(self):
        # spread_pct < -10
        s = _surface_with_term({7: 0.30, 60: 0.20})
        sig = TermStructureAnalyzer(s).generate_signal()
        assert sig.signal == "SELL_FRONT_BUY_BACK"
        assert "Calendar Spread" in sig.recommended_strategy

    def test_signal_mild_backwardation(self):
        # backwardation but not < -10
        s = _surface_with_term({7: 0.21, 60: 0.20})
        sig = TermStructureAnalyzer(s).generate_signal()
        # spread_pct ≈ -4.76 → flat, not backwardation
        # construct one within (-10, -5)
        s2 = _surface_with_term({7: 0.215, 60: 0.20})
        sig2 = TermStructureAnalyzer(s2).generate_signal()
        # spread_pct ≈ -6.97 → backwardation, not deep
        assert sig2.signal == "MILD_SELL_FRONT"

    def test_signal_steep_contango(self):
        # spread_pct > 15
        s = _surface_with_term({7: 0.18, 60: 0.25})
        sig = TermStructureAnalyzer(s).generate_signal()
        assert sig.signal == "BUY_FRONT_SELL_BACK"
        assert "Reverse Calendar" in sig.recommended_strategy

    def test_signal_neutral_normal_contango(self):
        # contango but not > 15%
        s = _surface_with_term({7: 0.20, 60: 0.21})
        sig = TermStructureAnalyzer(s).generate_signal()
        # spread_pct = 5 → exactly flat boundary; need strictly > 5 for contango
        assert sig.signal == "NEUTRAL"

    def test_spread_pct_zero_when_front_zero(self):
        s = _surface_with_term({7: 0.0, 60: 0.20})
        result = TermStructureAnalyzer(s).analyze_structure()
        assert result["spread_pct"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# SkewAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

def _surface_with_skew(skew: dict) -> VolSurface:
    return VolSurface(
        symbol="SPY", underlying_price=500,
        skew_by_expiry={30: skew},
    )


class TestSkewAnalyzer:
    def test_init_no_history(self):
        s = _surface_with_skew({})
        a = SkewAnalyzer(s)
        assert a.skew_history == []

    def test_init_with_history(self):
        s = _surface_with_skew({})
        a = SkewAnalyzer(s, skew_history=[-0.02, -0.03])
        assert a.skew_history == [-0.02, -0.03]

    def test_no_skew_data(self):
        s = _surface_with_skew({})
        result = SkewAnalyzer(s).compute_risk_reversal(30)
        assert result == {"risk_reversal": 0, "regime": "no_data"}

    def test_no_skew_data_missing_expiry(self):
        s = _surface_with_skew({-0.25: 0.30})
        result = SkewAnalyzer(s).compute_risk_reversal(60)
        assert result["regime"] == "no_data"

    def test_steep_skew(self):
        # call_25d - put_25d < -0.05
        skew = {-0.25: 0.30, 0.25: 0.20, 0.50: 0.22}
        s = _surface_with_skew(skew)
        result = SkewAnalyzer(s).compute_risk_reversal(30)
        assert result["risk_reversal"] == -0.10
        assert result["regime"] == "steep"
        assert result["put_25d_iv"] == 0.30
        assert result["call_25d_iv"] == 0.20
        assert result["atm_iv"] == 0.22

    def test_inverted_skew(self):
        # call > put by > 0.02
        skew = {-0.25: 0.18, 0.25: 0.25, 0.50: 0.20}
        result = SkewAnalyzer(_surface_with_skew(skew)).compute_risk_reversal(30)
        assert result["regime"] == "inverted"

    def test_flat_skew(self):
        # |rr| < 0.01
        skew = {-0.25: 0.20, 0.25: 0.205, 0.50: 0.20}
        result = SkewAnalyzer(_surface_with_skew(skew)).compute_risk_reversal(30)
        assert result["regime"] == "flat"

    def test_normal_skew(self):
        # rr in [-0.05, -0.01]
        skew = {-0.25: 0.22, 0.25: 0.20, 0.50: 0.21}
        result = SkewAnalyzer(_surface_with_skew(skew)).compute_risk_reversal(30)
        assert result["regime"] == "normal"

    def test_butterfly_zero_when_atm_zero(self):
        skew = {-0.25: 0.20, 0.25: 0.18}  # no atm
        result = SkewAnalyzer(_surface_with_skew(skew)).compute_risk_reversal(30)
        assert result["butterfly"] == 0
        assert result["atm_iv"] == 0

    def test_butterfly_computed(self):
        skew = {-0.25: 0.30, 0.25: 0.30, 0.50: 0.20}
        result = SkewAnalyzer(_surface_with_skew(skew)).compute_risk_reversal(30)
        # bf = (0.30 + 0.30)/2 - 0.20 = 0.10
        assert result["butterfly"] == 0.10

    def test_fallback_deltas(self):
        # Uses ±0.20 when ±0.25 missing, 0.45 when 0.50 missing
        skew = {-0.20: 0.25, 0.20: 0.22, 0.45: 0.23}
        result = SkewAnalyzer(_surface_with_skew(skew)).compute_risk_reversal(30)
        assert result["put_25d_iv"] == 0.25
        assert result["call_25d_iv"] == 0.22
        assert result["atm_iv"] == 0.23

    def test_z_score_with_history(self):
        skew = {-0.25: 0.22, 0.25: 0.20, 0.50: 0.21}
        # rr=-0.02; history mean=-0.02 → z=0
        a = SkewAnalyzer(_surface_with_skew(skew), skew_history=[-0.01, -0.03])
        result = a.compute_risk_reversal(30)
        assert result["z_score"] == 0.0

    def test_z_score_zero_std(self):
        skew = {-0.25: 0.22, 0.25: 0.20, 0.50: 0.21}
        a = SkewAnalyzer(_surface_with_skew(skew), skew_history=[-0.05, -0.05])
        result = a.compute_risk_reversal(30)
        assert result["z_score"] == 0

    def test_signal_extreme_negative_z_sell_skew(self):
        skew = {-0.25: 0.30, 0.25: 0.20, 0.50: 0.22}  # rr=-0.10
        # history mean=0, std small → very negative z
        a = SkewAnalyzer(_surface_with_skew(skew), skew_history=[0.0, 0.01, -0.01])
        sig = a.skew_trade_signal(30)
        assert sig["signal"] == "SELL_PUT_SKEW"
        assert sig["confidence"] == 80

    def test_signal_extreme_positive_z_buy_skew(self):
        skew = {-0.25: 0.18, 0.25: 0.25, 0.50: 0.20}  # rr=+0.07
        a = SkewAnalyzer(_surface_with_skew(skew), skew_history=[-0.05, -0.04, -0.06])
        sig = a.skew_trade_signal(30)
        assert sig["signal"] == "BUY_PUT_SKEW"
        assert sig["confidence"] == 75

    def test_signal_mild_sell_steep_regime(self):
        # steep regime, z between -2 and -1
        skew = {-0.25: 0.27, 0.25: 0.20, 0.50: 0.22}  # rr=-0.07 → steep
        a = SkewAnalyzer(_surface_with_skew(skew), skew_history=[-0.05, -0.06, -0.04])
        sig = a.skew_trade_signal(30)
        # rr=-0.07, mean=-0.05, std≈0.008 → z≈-2.45 → SELL_PUT_SKEW
        # Need rr where z is between -2 and -1; build directly
        skew2 = {-0.25: 0.255, 0.25: 0.20, 0.50: 0.22}  # rr=-0.055 → steep
        a2 = SkewAnalyzer(_surface_with_skew(skew2), skew_history=[-0.05, -0.045, -0.055])
        sig2 = a2.skew_trade_signal(30)
        # rr=-0.055, mean=-0.05, std≈0.004 → z≈-1.22 → MILD_SELL
        assert sig2["signal"] == "MILD_SELL_SKEW"
        assert sig2["confidence"] == 60

    def test_signal_neutral(self):
        skew = {-0.25: 0.22, 0.25: 0.205, 0.50: 0.21}  # rr=-0.015 → normal
        sig = SkewAnalyzer(_surface_with_skew(skew)).skew_trade_signal(30)
        assert sig["signal"] == "NEUTRAL"
        assert sig["confidence"] == 40

    def test_signal_no_data(self):
        sig = SkewAnalyzer(_surface_with_skew({})).skew_trade_signal(30)
        assert sig["signal"] == "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════════
# VolRegimeDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestVolRegimeDetector:
    def test_low_vol(self):
        r = VolRegimeDetector.classify(vix=10)
        assert r["regime"] == "low_vol"
        assert "Low vol" in r["note"]
        assert r["sizing_adjustment"] == "standard_or_reduced"
        assert len(r["recommended_strategies"]) == 3
        assert r["vvix"] is None

    def test_normal_vol(self):
        r = VolRegimeDetector.classify(vix=15)
        assert r["regime"] == "normal_vol"
        assert r["sizing_adjustment"] == "standard"

    def test_elevated_vol(self):
        r = VolRegimeDetector.classify(vix=25)
        assert r["regime"] == "elevated_vol"
        assert r["sizing_adjustment"] == "increase_25pct"

    def test_high_vol(self):
        r = VolRegimeDetector.classify(vix=35)
        assert r["regime"] == "high_vol"
        assert r["sizing_adjustment"] == "reduce_50pct"

    def test_extreme_vol(self):
        r = VolRegimeDetector.classify(vix=50)
        assert r["regime"] == "extreme_vol"
        assert r["sizing_adjustment"] == "minimal_or_zero"
        assert "Crisis" in r["note"] or "crisis" in r["note"]

    def test_vvix_low(self):
        r = VolRegimeDetector.classify(vix=15, vvix=70)
        assert "low" in r["vvix_note"]
        assert r["vvix"] == 70
        assert r["vix_vvix_ratio"] == round(15/70, 3)

    def test_vvix_normal(self):
        r = VolRegimeDetector.classify(vix=15, vvix=90)
        assert "normal" in r["vvix_note"]

    def test_vvix_elevated(self):
        r = VolRegimeDetector.classify(vix=15, vvix=110)
        assert "elevated" in r["vvix_note"]

    def test_vvix_extreme(self):
        r = VolRegimeDetector.classify(vix=15, vvix=130)
        assert "extreme" in r["vvix_note"]

    def test_no_vvix_no_ratio(self):
        r = VolRegimeDetector.classify(vix=15, vvix=0)
        assert r["vix_vvix_ratio"] is None
        assert r["vvix_note"] == ""

    def test_recommended_strategies_present_for_each_regime(self):
        for vix in (10, 15, 25, 35, 50):
            r = VolRegimeDetector.classify(vix=vix)
            assert len(r["recommended_strategies"]) == 3

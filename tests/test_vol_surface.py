"""Tests for strategies/vol_surface.py — Volatility Surface & Term Structure."""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_chains():
    """Synthetic options chain data."""
    chains = []
    spot = 450.0
    for dte in [14, 30, 60, 90]:
        for strike_pct in [0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10]:
            strike = spot * strike_pct
            # IV increases for OTM puts (skew) and with DTE
            base_iv = 0.18 + (1.0 - strike_pct) * 0.3 + dte * 0.0001
            chains.append({
                "strike": strike,
                "dte": dte,
                "impliedVolatility": base_iv,
                "optionType": "put" if strike_pct < 1.0 else "call",
                "delta": -(1.0 - strike_pct) if strike_pct < 1.0 else strike_pct - 1.0,
            })
    return chains


class TestVolPoint:
    def test_dataclass(self):
        from strategies.vol_surface import VolPoint
        vp = VolPoint(
            strike=450.0,
            expiry_days=30,
            implied_vol=0.20,
            option_type="call",
            moneyness=1.0,
        )
        assert vp.implied_vol == 0.20


class TestSkewMetrics:
    def test_dataclass(self):
        from strategies.vol_surface import SkewMetrics
        sm = SkewMetrics(
            expiry_days=30,
            atm_vol=0.20,
            rr_25d=-0.03,
            bf_25d=0.01,
            skew_slope=-0.5,
            smile_curvature=0.8,
        )
        assert sm.rr_25d == -0.03
        assert sm.bf_25d == 0.01


class TestVolSurfaceBuilder:
    def test_build_from_chains(self, sample_chains):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains(
            ticker="SPY",
            spot_price=450.0,
            chains=sample_chains,
            timestamp="2026-04-08T12:00:00Z",
        )
        assert snap.ticker == "SPY"
        assert snap.spot_price == 450.0
        assert len(snap.points) > 0
        assert len(snap.term_structure) > 0

    def test_skew_by_expiry(self, sample_chains):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains("SPY", 450.0, sample_chains)
        assert len(snap.skew_by_expiry) > 0
        for skew in snap.skew_by_expiry:
            assert skew.atm_vol > 0
            assert skew.expiry_days > 0

    def test_term_structure(self, sample_chains):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains("SPY", 450.0, sample_chains)
        dtes = [t.expiry_days for t in snap.term_structure]
        # Should be in ascending order
        assert dtes == sorted(dtes)
        # All ATM vols should be positive
        assert all(t.atm_vol > 0 for t in snap.term_structure)

    def test_vol_regime_normal(self, sample_chains):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains("SPY", 450.0, sample_chains)
        assert snap.vol_regime in ("low", "normal", "elevated", "crisis")

    def test_vol_regime_crisis(self):
        from strategies.vol_surface import VolSurfaceBuilder
        # High IV chains
        chains = []
        for dte in [14, 30]:
            for strike_pct in [0.95, 1.00, 1.05]:
                chains.append({
                    "strike": 450 * strike_pct,
                    "dte": dte,
                    "impliedVolatility": 0.50,  # crisis-level vol
                    "optionType": "call",
                })
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains("SPY", 450.0, chains)
        assert snap.vol_regime == "crisis"

    def test_moneyness_filter(self, sample_chains):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder(moneyness_range=(0.95, 1.05))
        snap = builder.build_from_chains("SPY", 450.0, sample_chains)
        # Should filter out far OTM options
        for p in snap.points:
            assert 0.95 <= p.moneyness <= 1.05

    def test_dte_filter(self, sample_chains):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder(min_dte=20, max_dte=70)
        snap = builder.build_from_chains("SPY", 450.0, sample_chains)
        for p in snap.points:
            assert 20 <= p.expiry_days <= 70

    def test_to_dict(self, sample_chains):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains("SPY", 450.0, sample_chains)
        d = snap.to_dict()
        assert d["ticker"] == "SPY"
        assert "vol_regime" in d
        assert "term_structure" in d

    def test_empty_chains(self):
        from strategies.vol_surface import VolSurfaceBuilder
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains("SPY", 450.0, [])
        assert len(snap.points) == 0
        assert snap.vol_regime == "normal"

    def test_zero_iv_filtered(self):
        from strategies.vol_surface import VolSurfaceBuilder
        chains = [{"strike": 450.0, "dte": 30, "impliedVolatility": 0.0, "optionType": "call"}]
        builder = VolSurfaceBuilder()
        snap = builder.build_from_chains("SPY", 450.0, chains)
        assert len(snap.points) == 0

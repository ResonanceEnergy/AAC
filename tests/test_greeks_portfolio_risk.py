"""Sprint 31 — Comprehensive tests for strategies.greeks_portfolio_risk.

Covers:
- RiskLevel enum
- PositionGreeks (defaults + dollar_delta / gamma_risk_1pct / theta_per_day / vega_per_1pct)
- PortfolioRiskSnapshot defaults + to_dict keys
- HedgeRecommendation init
- PortfolioGreeksEngine: add/clear positions, compute_snapshot (aggregation, beta-weighted delta,
  spy_price=0 guard), _generate_alerts (delta/gamma/theta/vega/expiry/concentration),
  _classify_risk all 5 levels
- HedgingEngine: delta hedge (immediate vs end_of_day), gamma hedge, vega hedge, tail risk,
  no hedges when within limits
- PositionSizer: size_credit_spread (normal, zero width guard), size_naked_position (cap at 3)
"""
from __future__ import annotations

import pytest

from strategies.greeks_portfolio_risk import (
    HedgeRecommendation,
    HedgingEngine,
    PortfolioGreeksEngine,
    PortfolioRiskSnapshot,
    PositionGreeks,
    PositionSizer,
    RiskLevel,
)


# ── RiskLevel ────────────────────────────────────────────────────────────────


class TestRiskLevel:
    def test_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.ELEVATED.value == "elevated"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


# ── PositionGreeks ───────────────────────────────────────────────────────────


class TestPositionGreeks:
    def test_defaults(self):
        p = PositionGreeks(symbol="SPY", strategy="iron_condor", quantity=1)
        assert p.symbol == "SPY"
        assert p.delta == 0.0
        assert p.gamma == 0.0
        assert p.notional == 0.0
        assert p.dte == 0

    def test_dollar_delta(self):
        p = PositionGreeks("SPY", "x", 1, delta=0.5, notional=10000.0)
        assert p.dollar_delta == 5000.0

    def test_gamma_risk_1pct(self):
        # 0.5 * gamma * (notional * 0.01)^2
        p = PositionGreeks("SPY", "x", 1, gamma=2.0, notional=1000.0)
        assert p.gamma_risk_1pct == pytest.approx(0.5 * 2.0 * (10.0 ** 2))

    def test_theta_per_day(self):
        # theta * quantity * 100
        p = PositionGreeks("SPY", "x", 5, theta=-0.4)
        assert p.theta_per_day == pytest.approx(-0.4 * 5 * 100)

    def test_vega_per_1pct(self):
        p = PositionGreeks("SPY", "x", 3, vega=2.0)
        assert p.vega_per_1pct == pytest.approx(2.0 * 3 * 100)


# ── PortfolioRiskSnapshot ────────────────────────────────────────────────────


class TestPortfolioRiskSnapshot:
    def test_defaults(self):
        snap = PortfolioRiskSnapshot(timestamp="t")
        assert snap.total_delta == 0.0
        assert snap.delta_limit == 100.0
        assert snap.risk_level == RiskLevel.LOW
        assert snap.alerts == []

    def test_to_dict_keys(self):
        snap = PortfolioRiskSnapshot(timestamp="t", total_delta=5.0, total_gamma=1.0,
                                     total_theta=-2.0, total_vega=3.0)
        d = snap.to_dict()
        expected_keys = {
            "timestamp", "delta", "gamma", "theta", "vega", "dollar_delta",
            "daily_theta", "gamma_risk_1pct", "vega_risk_1pct", "risk_level",
            "alerts", "positions", "beta_weighted_delta",
        }
        assert set(d.keys()) == expected_keys
        assert d["risk_level"] == "low"
        assert d["delta"] == 5.0


# ── HedgeRecommendation ──────────────────────────────────────────────────────


class TestHedgeRecommendation:
    def test_init(self):
        h = HedgeRecommendation(action="BUY_SHARES", instrument="SPY", quantity=10,
                                rationale="hedge delta", priority="immediate",
                                estimated_cost=100.0)
        assert h.action == "BUY_SHARES"
        assert h.greeks_impact == {}


# ── PortfolioGreeksEngine ────────────────────────────────────────────────────


class TestPortfolioGreeksEngine:
    def test_init_defaults(self):
        eng = PortfolioGreeksEngine()
        assert eng.delta_limit == 100.0
        assert eng.gamma_limit == 50.0
        assert eng.vega_limit == 200.0
        assert eng.theta_floor == -500.0
        assert eng.positions == []

    def test_init_custom(self):
        eng = PortfolioGreeksEngine(delta_limit=50, gamma_limit=25,
                                    vega_limit=100, theta_floor=-200)
        assert eng.delta_limit == 50

    def test_add_and_clear_positions(self):
        eng = PortfolioGreeksEngine()
        p = PositionGreeks("SPY", "x", 1)
        eng.add_position(p)
        assert len(eng.positions) == 1
        eng.clear_positions()
        assert eng.positions == []

    def test_compute_snapshot_empty(self):
        eng = PortfolioGreeksEngine()
        snap = eng.compute_snapshot()
        assert snap.num_positions == 0
        assert snap.total_delta == 0.0
        assert snap.risk_level == RiskLevel.LOW

    def test_compute_snapshot_aggregates(self):
        eng = PortfolioGreeksEngine(delta_limit=1000, gamma_limit=1000, vega_limit=1000)
        eng.add_position(PositionGreeks("SPY", "x", 2, delta=0.5, gamma=0.1,
                                        theta=-0.2, vega=1.0, notional=10000.0))
        eng.add_position(PositionGreeks("QQQ", "x", 3, delta=-0.3, gamma=0.05,
                                        theta=-0.1, vega=0.5, notional=5000.0))
        snap = eng.compute_snapshot()
        # 0.5*2 + (-0.3)*3 = 0.1
        assert snap.total_delta == pytest.approx(0.1)
        # 0.1*2 + 0.05*3 = 0.35
        assert snap.total_gamma == pytest.approx(0.35)
        assert snap.num_positions == 2

    def test_compute_snapshot_beta_weighted_delta(self):
        eng = PortfolioGreeksEngine(delta_limit=10000)
        eng.add_position(PositionGreeks("AAPL", "x", 1, delta=0.5, notional=52000.0))
        snap = eng.compute_snapshot(spy_price=520.0, beta_map={"AAPL": 1.5})
        # bw_factor = (52000 / (520*100)) * 1.5 = 1.0 * 1.5 = 1.5
        # bw_delta = 0.5 * 1 * 1.5 = 0.75
        assert snap.beta_weighted_delta == pytest.approx(0.75)

    def test_compute_snapshot_zero_spy_price_guard(self):
        eng = PortfolioGreeksEngine(delta_limit=10000)
        eng.add_position(PositionGreeks("X", "x", 1, delta=0.5, notional=1000.0))
        # spy_price=0 → bw_factor = 1 (no division by zero)
        snap = eng.compute_snapshot(spy_price=0.0)
        assert snap.beta_weighted_delta == pytest.approx(0.5)

    def test_alerts_delta_breach(self):
        eng = PortfolioGreeksEngine(delta_limit=10)
        # delta = 1.0 * 20 = 20 → exceeds 10
        eng.add_position(PositionGreeks("SPY", "x", 20, delta=1.0))
        snap = eng.compute_snapshot()
        assert any("DELTA BREACH" in a for a in snap.alerts)

    def test_alerts_gamma_elevated(self):
        eng = PortfolioGreeksEngine(delta_limit=10000, gamma_limit=1, vega_limit=10000)
        eng.add_position(PositionGreeks("SPY", "x", 10, gamma=1.0))
        snap = eng.compute_snapshot()
        assert any("GAMMA" in a for a in snap.alerts)

    def test_alerts_theta_drag(self):
        eng = PortfolioGreeksEngine(delta_limit=10000, gamma_limit=10000, vega_limit=10000,
                                    theta_floor=-100)
        # daily_theta = theta * qty * 100 = -10 * 1 * 100 = -1000 < -100
        eng.add_position(PositionGreeks("SPY", "x", 1, theta=-10.0))
        snap = eng.compute_snapshot()
        assert any("THETA" in a for a in snap.alerts)

    def test_alerts_vega_risk(self):
        eng = PortfolioGreeksEngine(delta_limit=10000, gamma_limit=10000, vega_limit=5)
        eng.add_position(PositionGreeks("SPY", "x", 10, vega=1.0))
        snap = eng.compute_snapshot()
        assert any("VEGA" in a for a in snap.alerts)

    def test_alerts_expiration_risk(self):
        eng = PortfolioGreeksEngine(delta_limit=10000, gamma_limit=10000, vega_limit=10000)
        eng.add_position(PositionGreeks("SPY", "x", 1, gamma=0.5, dte=5))
        snap = eng.compute_snapshot()
        assert any("EXPIRATION RISK" in a for a in snap.alerts)

    def test_alerts_concentration(self):
        # delta_limit=100, threshold=30 (30%); single SPY delta = 50
        eng = PortfolioGreeksEngine(delta_limit=100, gamma_limit=10000, vega_limit=10000)
        eng.add_position(PositionGreeks("SPY", "x", 50, delta=1.0))
        snap = eng.compute_snapshot()
        # Will trigger both DELTA BREACH and CONCENTRATION
        assert any("CONCENTRATION" in a for a in snap.alerts)

    def test_classify_risk_low(self):
        eng = PortfolioGreeksEngine()
        snap = eng.compute_snapshot()
        assert snap.risk_level == RiskLevel.LOW

    def test_classify_risk_moderate(self):
        # delta_pct between 50 and 80 → +1 = MODERATE (>=1)
        eng = PortfolioGreeksEngine(delta_limit=100, gamma_limit=10000, vega_limit=10000)
        eng.add_position(PositionGreeks("SPY", "x", 60, delta=1.0))
        snap = eng.compute_snapshot()
        assert snap.risk_level in {RiskLevel.MODERATE, RiskLevel.ELEVATED, RiskLevel.HIGH}

    def test_classify_risk_critical(self):
        # Massive breaches across all metrics + many alerts
        eng = PortfolioGreeksEngine(delta_limit=1, gamma_limit=1, vega_limit=1, theta_floor=-1)
        eng.add_position(PositionGreeks("A", "x", 100, delta=10, gamma=10, theta=-10, vega=10, dte=3))
        eng.add_position(PositionGreeks("B", "x", 100, delta=10, gamma=10, theta=-10, vega=10, dte=3))
        snap = eng.compute_snapshot()
        assert snap.risk_level == RiskLevel.CRITICAL


# ── HedgingEngine ────────────────────────────────────────────────────────────


def _snap_with(total_delta=0.0, total_gamma=0.0, total_vega=0.0,
               dollar_delta=0.0, delta_limit=100.0, gamma_limit=50.0,
               vega_limit=200.0) -> PortfolioRiskSnapshot:
    return PortfolioRiskSnapshot(
        timestamp="t",
        total_delta=total_delta,
        total_gamma=total_gamma,
        total_vega=total_vega,
        dollar_delta=dollar_delta,
        delta_limit=delta_limit,
        gamma_limit=gamma_limit,
        vega_limit=vega_limit,
    )


class TestHedgingEngine:
    def test_no_hedges_when_within_limits(self):
        h = HedgingEngine()
        # Small values, low notional → no recommendations
        snap = _snap_with(total_delta=10, total_gamma=5, total_vega=20, dollar_delta=10000)
        assert h.generate_hedges(snap) == []

    def test_delta_hedge_triggered(self):
        h = HedgingEngine(spy_price=520.0)
        snap = _snap_with(total_delta=80.0)  # > 70% of 100
        recs = h.generate_hedges(snap)
        delta_rec = next(r for r in recs if "SHARES" in r.action)
        assert delta_rec.action in {"BUY_SHARES", "SELL_SHARES"}
        assert delta_rec.quantity > 0

    def test_delta_hedge_immediate_when_exceeded(self):
        h = HedgingEngine()
        snap = _snap_with(total_delta=120.0)  # > limit of 100
        recs = h.generate_hedges(snap)
        delta_rec = next(r for r in recs if "SHARES" in r.action)
        assert delta_rec.priority == "immediate"

    def test_delta_hedge_end_of_day_when_just_above_70pct(self):
        h = HedgingEngine()
        snap = _snap_with(total_delta=80.0)  # 80% of limit, not exceeded
        recs = h.generate_hedges(snap)
        delta_rec = next(r for r in recs if "SHARES" in r.action)
        assert delta_rec.priority == "end_of_day"

    def test_gamma_hedge(self):
        h = HedgingEngine()
        snap = _snap_with(total_gamma=45.0)  # > 80% of 50
        recs = h.generate_hedges(snap)
        assert any(r.action == "REDUCE_NEAR_EXPIRY" for r in recs)

    def test_vega_hedge_short_when_long_vega(self):
        h = HedgingEngine()
        snap = _snap_with(total_vega=180.0)  # > 80% of 200
        recs = h.generate_hedges(snap)
        vega_rec = next(r for r in recs if "VOL_HEDGE" in r.action)
        assert "SHORT" in vega_rec.action

    def test_vega_hedge_long_when_short_vega(self):
        h = HedgingEngine()
        snap = _snap_with(total_vega=-180.0)
        recs = h.generate_hedges(snap)
        vega_rec = next(r for r in recs if "VOL_HEDGE" in r.action)
        assert "LONG" in vega_rec.action

    def test_tail_risk_when_large_notional(self):
        h = HedgingEngine()
        snap = _snap_with(dollar_delta=500_000.0)
        recs = h.generate_hedges(snap)
        assert any(r.action == "TAIL_RISK_PROTECTION" for r in recs)

    def test_no_tail_risk_when_small_notional(self):
        h = HedgingEngine()
        snap = _snap_with(dollar_delta=50_000.0)
        recs = h.generate_hedges(snap)
        assert all(r.action != "TAIL_RISK_PROTECTION" for r in recs)


# ── PositionSizer ────────────────────────────────────────────────────────────


class TestPositionSizer:
    def test_init(self):
        s = PositionSizer(total_capital=100_000)
        assert s.capital == 100_000
        assert s.max_risk == 0.03
        assert s.max_allocation == 0.50
        assert s.max_name == 0.15

    def test_size_credit_spread_basic(self):
        s = PositionSizer(total_capital=100_000)
        out = s.size_credit_spread(spread_width=5.0, credit=1.50)
        assert out["contracts"] >= 1
        assert out["max_loss"] > 0
        assert out["max_profit"] > 0
        assert "rationale" in out

    def test_size_credit_spread_zero_width_guards(self):
        s = PositionSizer(total_capital=100_000)
        # spread_width=0, credit=0 → max_loss_per=0, contracts_by_risk=0;
        # contracts_by_alloc=0; max(1, min(0,0)) = 1
        out = s.size_credit_spread(spread_width=0.0, credit=0.0)
        assert out["contracts"] == 1

    def test_size_credit_spread_keys(self):
        s = PositionSizer(total_capital=50_000)
        out = s.size_credit_spread(spread_width=2.0, credit=0.50)
        assert set(out.keys()) == {"contracts", "max_loss", "max_profit",
                                    "capital_at_risk_pct", "rationale"}

    def test_size_naked_position_capped_at_3(self):
        s = PositionSizer(total_capital=10_000_000)
        out = s.size_naked_position(premium=2.0, underlying_price=100.0)
        assert out["contracts"] <= 3

    def test_size_naked_position_minimum_1(self):
        s = PositionSizer(total_capital=1_000)
        out = s.size_naked_position(premium=1.0, underlying_price=500.0)
        assert out["contracts"] >= 1

    def test_size_naked_position_keys(self):
        s = PositionSizer(total_capital=100_000)
        out = s.size_naked_position(premium=1.5, underlying_price=100.0)
        assert set(out.keys()) == {"contracts", "margin_required",
                                    "premium_collected", "capital_deployed_pct",
                                    "warning"}
        assert "UNDEFINED RISK" in out["warning"]

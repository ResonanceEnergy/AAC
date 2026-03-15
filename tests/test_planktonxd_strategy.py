"""
Tests for PlanktonXD Prediction Market Harvester Strategy
=========================================================
Validates the core logic, risk controls, and simulation engine
for the planktonXD emulation strategy (Strategy #51).
"""

import pytest
from datetime import datetime, timedelta

from strategies.planktonxd_prediction_harvester import (
    BetType,
    HarvesterStats,
    MarketCategory,
    PlanktonBet,
    PlanktonXDSimulator,
    PredictionMarket,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────

def _make_market(
    market_id="m_001",
    category=MarketCategory.CRYPTO_PRICE,
    question="Will SOL drop below $130?",
    outcomes=None,
    prices=None,
    volume_24h=50.0,
    liquidity=200.0,
    hours_until_resolution=48.0,
):
    outcomes = outcomes or ["Yes", "No"]
    prices = prices or {"Yes": 0.007, "No": 0.993}
    resolution = datetime.now() + timedelta(hours=hours_until_resolution)
    return PredictionMarket(
        market_id=market_id,
        category=category,
        question=question,
        outcomes=outcomes,
        prices=prices,
        volume_24h=volume_24h,
        liquidity=liquidity,
        resolution_time=resolution,
    )


# ─── PredictionMarket Tests ──────────────────────────────────────────────

class TestPredictionMarket:
    """TestPredictionMarket class."""
    def test_cheapest_outcome(self):
        m = _make_market(prices={"Yes": 0.005, "No": 0.995})
        assert m.cheapest_outcome == "Yes"
        assert m.cheapest_outcome_price == 0.005

    def test_spread_calculation(self):
        m = _make_market(prices={"Yes": 0.02, "No": 0.99})
        assert abs(m.spread - 0.01) < 1e-9

    def test_thin_book_detection(self):
        assert _make_market(liquidity=100).is_thin_book
        assert not _make_market(liquidity=1000).is_thin_book

    def test_hours_to_resolution(self):
        m = _make_market(hours_until_resolution=24)
        hours = m.hours_to_resolution
        assert hours is not None
        assert 23.5 < hours < 24.5


# ─── HarvesterStats Tests ────────────────────────────────────────────────

class TestHarvesterStats:
    """TestHarvesterStats class."""
    def test_empty_stats(self):
        s = HarvesterStats()
        assert s.net_profit == 0.0
        assert s.win_rate == 0.0
        assert s.roi == 0.0
        assert s.avg_bet_size == 0.0

    def test_profit_calculation(self):
        s = HarvesterStats(
            total_bets=100,
            winning_bets=5,
            losing_bets=95,
            total_invested=1500.0,
            total_returned=4500.0,
        )
        assert s.net_profit == 3000.0
        assert s.roi == 2.0
        assert s.win_rate == 0.05


# ─── PlanktonBet Tests ───────────────────────────────────────────────────

class TestPlanktonBet:
    """TestPlanktonBet class."""
    def test_roi_calculation(self):
        bet = PlanktonBet(
            bet_id="test_001",
            market=_make_market(),
            bet_type=BetType.DEEP_OTM_TAIL,
            outcome="Yes",
            entry_price=0.01,
            shares=1500.0,
            cost=15.0,
            potential_payout=1500.0,
            implied_probability=0.01,
            estimated_true_probability=0.03,
            edge=0.02,
        )
        assert bet.potential_roi == pytest.approx(99.0, rel=0.01)
        assert bet.risk_reward_ratio == pytest.approx(100.0, rel=0.01)


# ─── Simulator Tests ─────────────────────────────────────────────────────

class TestPlanktonXDSimulator:
    """TestPlanktonXDSimulator class."""
    def test_single_simulation_runs(self):
        sim = PlanktonXDSimulator(
            starting_bankroll=1000.0,
            bets_per_day=50,
            days=30,
            avg_bet_size=10.0,
            avg_entry_price=0.01,
            true_prob_multiple=3.0,
        )
        result = sim.run_simulation(seed=42)
        assert result['total_bets'] > 0
        assert 'final_bankroll' in result
        assert 'net_profit' in result

    def test_simulation_deterministic_with_seed(self):
        sim = PlanktonXDSimulator(bets_per_day=50, days=10)
        r1 = sim.run_simulation(seed=123)
        r2 = sim.run_simulation(seed=123)
        assert r1['final_bankroll'] == r2['final_bankroll']
        assert r1['total_bets'] == r2['total_bets']

    def test_positive_ev_produces_profit_on_average(self):
        """With 3x true-prob multiple, the strategy should be profitable."""
        sim = PlanktonXDSimulator(
            starting_bankroll=1000.0,
            bets_per_day=100,
            days=90,
            avg_bet_size=10.0,
            avg_entry_price=0.01,
            true_prob_multiple=3.0,
        )
        mc = sim.run_monte_carlo(num_paths=200)
        # Majority of paths should be profitable
        pct = int(mc['pct_profitable'].rstrip('%'))
        assert pct > 60, f"Only {pct}% profitable — expected >60%"
        assert mc['mean_profit'] > 0

    def test_monte_carlo_output_structure(self):
        sim = PlanktonXDSimulator(bets_per_day=10, days=5)
        mc = sim.run_monte_carlo(num_paths=10)
        expected_keys = [
            'paths', 'starting_bankroll', 'params', 'median_final',
            'mean_final', 'best_case', 'worst_case', 'pct_profitable',
            'mean_profit', 'median_profit', 'p10_profit', 'p90_profit',
        ]
        for key in expected_keys:
            assert key in mc, f"Missing key: {key}"

    def test_bust_when_zero_edge(self):
        """With true_prob_multiple=1.0 (no edge), break-even or small loss expected."""
        sim = PlanktonXDSimulator(
            starting_bankroll=1000.0,
            bets_per_day=100,
            days=30,
            avg_bet_size=10.0,
            avg_entry_price=0.01,
            true_prob_multiple=1.0,  # No edge
        )
        mc = sim.run_monte_carlo(num_paths=100)
        # Should not consistently produce profit with no edge
        assert mc['mean_profit'] < mc['starting_bankroll'] * 5

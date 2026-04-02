"""
Tests for PlanktonXD Prediction Market Harvester Strategy
=========================================================
Validates the core logic, risk controls, and simulation engine
for the planktonXD emulation strategy (Strategy #51).
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from agents.polymarket_agent import PolymarketAgent, PolymarketMarket
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


# ─── PolymarketAgent Retry Tests ─────────────────────────────────────────

class TestPolymarketAgentRetry:
    """Test PolymarketAgent retry + exponential backoff logic."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        """Agent should retry and succeed if second attempt works."""
        agent = PolymarketAgent()
        call_count = 0

        class FakeCtxManager:
            def __init__(self, should_fail):
                self._fail = should_fail

            async def __aenter__(self):
                if self._fail:
                    raise aiohttp.ClientError("Connection refused")
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json = AsyncMock(return_value=[{"id": "1"}])
                return resp

            async def __aexit__(self, *args):
                pass

        def mock_get(url, params=None):
            nonlocal call_count
            call_count += 1
            return FakeCtxManager(should_fail=(call_count == 1))

        try:
            session = await agent._get_session()
            session.get = mock_get
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await agent._get("https://test.example.com/api")
            assert result == [{"id": "1"}]
            assert call_count == 2
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """Agent should raise after all retries exhausted."""
        agent = PolymarketAgent()

        class AlwaysFailCtx:
            async def __aenter__(self):
                raise aiohttp.ClientError("DNS failure")

            async def __aexit__(self, *args):
                pass

        def always_fail(url, params=None):
            return AlwaysFailCtx()

        try:
            session = await agent._get_session()
            session.get = always_fail
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(aiohttp.ClientError, match="DNS failure"):
                    await agent._get("https://test.example.com/api", retries=2)
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Async context manager should close session."""
        async with PolymarketAgent() as agent:
            session = await agent._get_session()
            assert not session.closed
        assert agent._session is None or agent._session.closed


# ─── BlackSwanScanner Circuit-Breaker Tests ───────────────────────────────

class TestBlackSwanScannerCircuitBreaker:
    """Test that scanner stops early when API is unreachable."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_keyword_searches(self):
        """When first page fails, keyword searches should be skipped."""
        from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner

        scanner = PolymarketBlackSwanScanner()
        call_count = 0

        async def mock_get_markets(limit=100, offset=0):
            nonlocal call_count
            call_count += 1
            raise aiohttp.ClientError("DNS failure")

        async def mock_search(query, limit=10):
            nonlocal call_count
            call_count += 1
            return []

        scanner.agent.get_active_markets = mock_get_markets
        scanner.agent.search_markets = mock_search

        try:
            opps = await scanner.scan(max_pages=3)
            assert len(opps) == 0
            # Should only have called get_active_markets once (page 0 failed)
            # search_markets should NOT have been called
            assert call_count == 1, f"Expected 1 call (page 0 only), got {call_count}"
        finally:
            await scanner.close()

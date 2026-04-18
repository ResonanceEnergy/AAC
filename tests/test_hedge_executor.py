"""Tests for strategies/hedge_executor.py — Hedge Auto-Execution."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_rec(**kwargs):
    """Create a HedgeRecommendation-like object with attribute access."""
    defaults = {
        "action": "buy",
        "instrument": "SPY",
        "quantity": 10,
        "priority": "immediate",
        "rationale": "Delta hedge needed",
        "estimated_cost": 300.0,
        "greeks_impact": {"delta": -5.0},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def hedge_recommendations():
    return [
        _make_rec(action="buy", instrument="SPY", quantity=10,
                  priority="immediate", estimated_cost=300.0),
        _make_rec(action="sell", instrument="TLT", quantity=5,
                  priority="end_of_day", estimated_cost=150.0),
        _make_rec(action="buy", instrument="VIX", quantity=3,
                  priority="optional", estimated_cost=100.0),
    ]


class TestHedgeExecutor:
    def test_dry_run_default(self):
        from strategies.hedge_executor import HedgeExecutor
        executor = HedgeExecutor()
        assert executor.dry_run is True

    @pytest.mark.asyncio
    async def test_process_recommendations_dry_run(self, hedge_recommendations):
        from strategies.hedge_executor import HedgeExecutor, HedgeAction
        executor = HedgeExecutor(dry_run=True)
        result = await executor.process_recommendations(hedge_recommendations)
        assert result.recommendations_received == 3
        # In dry run, nothing should be EXECUTED
        assert result.executed == 0
        assert len(result.orders) == 3

    @pytest.mark.asyncio
    async def test_cost_limit_rejection(self):
        from strategies.hedge_executor import HedgeExecutor, HedgeAction
        expensive = _make_rec(estimated_cost=10000.0, priority="immediate")
        executor = HedgeExecutor(dry_run=True, max_hedge_cost=500.0)
        result = await executor.process_recommendations([expensive])
        assert len(result.orders) == 1
        assert result.orders[0].status == HedgeAction.REJECTED

    @pytest.mark.asyncio
    async def test_priority_filtering(self, hedge_recommendations):
        from strategies.hedge_executor import HedgeExecutor, HedgeAction
        executor = HedgeExecutor(
            dry_run=True,
            auto_execute_priorities=["immediate"],
        )
        result = await executor.process_recommendations(hedge_recommendations)
        approved = [o for o in result.orders if o.status == HedgeAction.APPROVED]
        # Only "immediate" priority should be approved
        for o in approved:
            assert o.priority == "immediate"

    @pytest.mark.asyncio
    async def test_daily_budget(self):
        from strategies.hedge_executor import HedgeExecutor, HedgeAction
        recs = [
            _make_rec(estimated_cost=1500.0, priority="immediate"),
            _make_rec(estimated_cost=800.0, priority="immediate"),
        ]
        executor = HedgeExecutor(
            dry_run=True,
            max_daily_hedge_cost=2000.0,
            max_hedge_cost=2000.0,
        )
        result = await executor.process_recommendations(recs)
        # First should be approved, second may be deferred due to budget
        assert result.orders[0].status == HedgeAction.APPROVED

    @pytest.mark.asyncio
    async def test_hedge_order_translation(self):
        from strategies.hedge_executor import HedgeExecutor
        rec = _make_rec(instrument="SPY", quantity=10, action="buy",
                        priority="immediate", estimated_cost=300.0)
        executor = HedgeExecutor(dry_run=True)
        result = await executor.process_recommendations([rec])
        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.instrument == "SPY"
        assert order.quantity == 10

    @pytest.mark.asyncio
    async def test_result_to_dict(self):
        from strategies.hedge_executor import HedgeExecutor
        rec = _make_rec(priority="immediate", estimated_cost=200.0)
        executor = HedgeExecutor(dry_run=True)
        result = await executor.process_recommendations([rec])
        d = result.to_dict()
        assert "received" in d
        assert "approved" in d

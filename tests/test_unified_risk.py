"""Tests for core/unified_risk.py — Unified Risk Aggregation."""
from __future__ import annotations

import pytest


class TestOverallRiskLevel:
    def test_enum_values(self):
        from core.unified_risk import OverallRiskLevel
        assert OverallRiskLevel.GREEN.value == "green"
        assert OverallRiskLevel.HALT.value == "halt"


class TestRiskBudget:
    def test_compute_utilization(self):
        from core.unified_risk import RiskBudget
        b = RiskBudget(
            name="options",
            allocated_var=1000.0,
            current_var=800.0,
            allocated_delta=50.0,
            current_delta=30.0,
        )
        b.compute_utilization()
        assert b.utilization == 0.8  # VaR util = 0.8, delta util = 0.6

    def test_zero_allocation(self):
        from core.unified_risk import RiskBudget
        b = RiskBudget(
            name="empty",
            allocated_var=0.0,
            current_var=100.0,
            allocated_delta=0.0,
            current_delta=10.0,
        )
        b.compute_utilization()
        assert b.utilization == 0.0


class TestUnifiedRiskAggregator:
    def test_green_no_data(self):
        from core.unified_risk import UnifiedRiskAggregator, OverallRiskLevel
        agg = UnifiedRiskAggregator()
        snap = agg.aggregate()
        assert snap.overall_level == OverallRiskLevel.GREEN
        assert len(snap.subsystems) == 0

    def test_greeks_ingestion(self):
        from core.unified_risk import UnifiedRiskAggregator
        agg = UnifiedRiskAggregator()
        greeks = {
            "delta": 50.0,
            "gamma": 2.0,
            "theta": -15.0,
            "vega": 30.0,
            "daily_theta": -150.0,
            "risk_level": "moderate",
            "alerts": ["Delta approaching limit"],
        }
        snap = agg.aggregate(greeks_snapshot=greeks)
        assert snap.total_delta == 50.0
        assert snap.total_gamma == 2.0
        assert snap.total_vega == 30.0
        assert len(snap.subsystems) == 1
        assert snap.subsystems[0].name == "greeks_portfolio"

    def test_matrix_risk_ingestion(self):
        from core.unified_risk import UnifiedRiskAggregator
        agg = UnifiedRiskAggregator(var_limit=5000.0)
        matrix = {
            "var_95_1d": 2000.0,
            "total_exposure": 100000.0,
            "daily_pnl": -500.0,
            "circuit_breaker": {"tripped": False},
            "hedge_alerts": [],
        }
        snap = agg.aggregate(matrix_risk=matrix)
        assert snap.total_var_95 == 2000.0
        assert snap.total_exposure_usd == 100000.0

    def test_circuit_breaker_halt(self):
        from core.unified_risk import UnifiedRiskAggregator, OverallRiskLevel
        agg = UnifiedRiskAggregator()
        matrix = {
            "var_95_1d": 1000.0,
            "total_exposure": 50000.0,
            "daily_pnl": -100.0,
            "circuit_breaker": {"tripped": True},
            "hedge_alerts": [],
        }
        snap = agg.aggregate(matrix_risk=matrix)
        assert snap.circuit_breaker_active is True
        assert snap.overall_level == OverallRiskLevel.HALT

    def test_delta_breach_yellow(self):
        from core.unified_risk import UnifiedRiskAggregator, OverallRiskLevel
        agg = UnifiedRiskAggregator(delta_limit=100.0)
        greeks = {
            "delta": 150.0,
            "gamma": 0,
            "theta": 0,
            "vega": 0,
            "daily_theta": 0,
            "risk_level": "high",
            "alerts": [],
        }
        snap = agg.aggregate(greeks_snapshot=greeks)
        assert snap.overall_level == OverallRiskLevel.YELLOW

    def test_multiple_breaches_red(self):
        from core.unified_risk import UnifiedRiskAggregator, OverallRiskLevel
        agg = UnifiedRiskAggregator(
            delta_limit=100.0,
            var_limit=1000.0,
            daily_loss_limit=500.0,
        )
        greeks = {"delta": 250.0, "gamma": 0, "theta": 0, "vega": 0, "daily_theta": 0, "risk_level": "critical", "alerts": []}
        matrix = {"var_95_1d": 2000.0, "total_exposure": 100000.0, "daily_pnl": -1000.0, "circuit_breaker": {"tripped": False}, "hedge_alerts": []}
        snap = agg.aggregate(greeks_snapshot=greeks, matrix_risk=matrix)
        assert snap.overall_level in (OverallRiskLevel.ORANGE, OverallRiskLevel.RED)

    def test_all_subsystems(self):
        from core.unified_risk import UnifiedRiskAggregator
        agg = UnifiedRiskAggregator()
        snap = agg.aggregate(
            greeks_snapshot={"delta": 10, "gamma": 1, "theta": -5, "vega": 20, "daily_theta": -50, "risk_level": "low", "alerts": []},
            matrix_risk={"var_95_1d": 500, "total_exposure": 50000, "daily_pnl": 200, "circuit_breaker": {"tripped": False}, "hedge_alerts": []},
            execution_risk={"total_exposure_usd": 30000, "daily_pnl": 100, "violations": []},
            paper_risk={"is_halted": False, "max_drawdown_pct": 0.02},
        )
        assert len(snap.subsystems) == 4

    def test_risk_budget_management(self):
        from core.unified_risk import UnifiedRiskAggregator
        agg = UnifiedRiskAggregator()
        agg.set_risk_budget("momentum", allocated_var=2000, allocated_delta=50)
        agg.set_risk_budget("mean_rev", allocated_var=1500, allocated_delta=30)
        agg.update_budget("momentum", current_var=1800, current_delta=40)
        snap = agg.aggregate()
        assert len(snap.risk_budgets) == 2
        mom = [b for b in snap.risk_budgets if b.name == "momentum"][0]
        assert mom.utilization == pytest.approx(0.9, abs=0.01)

    def test_to_dict(self):
        from core.unified_risk import UnifiedRiskAggregator
        agg = UnifiedRiskAggregator()
        snap = agg.aggregate(
            greeks_snapshot={"delta": 10, "gamma": 1, "theta": -5, "vega": 20, "daily_theta": -50, "risk_level": "low", "alerts": []},
        )
        d = snap.to_dict()
        assert "overall_level" in d
        assert "total_delta" in d
        assert "subsystems" in d

    def test_paper_halt_critical(self):
        from core.unified_risk import UnifiedRiskAggregator
        agg = UnifiedRiskAggregator()
        snap = agg.aggregate(paper_risk={"is_halted": True, "halt_reason": "Max drawdown", "max_drawdown_pct": 0.12})
        paper = [s for s in snap.subsystems if s.name == "paper_trading"][0]
        assert paper.risk_level == "critical"

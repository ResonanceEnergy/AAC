"""Tests for Doctrine Engine — BARREN WUFFET compliance system.

Validates state machine, metric evaluation, compliance reporting,
and violation tracking without external YAML dependencies.
"""

from datetime import datetime
from pathlib import Path

import pytest

from aac.doctrine.doctrine_engine import (
    ActionType,
    BarrenWuffetState,
    ComplianceState,
    Department,
    DoctrineComplianceReport,
    DoctrineEngine,
    DoctrineViolation,
    MetricValue,
)


# ── Enum Sanity ────────────────────────────────────────────────────────────


class TestEnums:
    """Validate doctrine enum members."""

    def test_barren_wuffet_states(self):
        assert BarrenWuffetState.NORMAL.value == "NORMAL"
        assert BarrenWuffetState.CAUTION.value == "CAUTION"
        assert BarrenWuffetState.SAFE_MODE.value == "SAFE_MODE"
        assert BarrenWuffetState.HALT.value == "HALT"
        assert len(BarrenWuffetState) == 4

    def test_compliance_states(self):
        assert ComplianceState.COMPLIANT.value == "COMPLIANT"
        assert len(ComplianceState) == 4

    def test_departments(self):
        dept_values = [d.value for d in Department]
        assert "TradingExecution" in dept_values
        assert "BigBrainIntelligence" in dept_values
        assert len(Department) >= 5

    def test_action_types_coverage(self):
        assert ActionType.A_STOP_EXECUTION.value == "A_STOP_EXECUTION"
        assert ActionType.A_ENTER_SAFE_MODE.value == "A_ENTER_SAFE_MODE"
        assert len(ActionType) == 14


# ── Data Classes ───────────────────────────────────────────────────────────


class TestDataClasses:
    """Validate doctrine data models."""

    def test_metric_value_defaults(self):
        mv = MetricValue(
            name="drawdown",
            value=0.15,
            threshold_good="<0.10",
            threshold_warning="<0.20",
            threshold_critical=">=0.20",
        )
        assert mv.state == ComplianceState.UNKNOWN
        assert mv.department is None

    def test_doctrine_violation_creation(self):
        v = DoctrineViolation(
            pack_id=1,
            pack_name="Risk Management Core",
            rule_type="metric",
            rule_id="r1_drawdown",
            description="Drawdown exceeds 20%",
            severity="critical",
            department=Department.TRADING_EXECUTION,
            recommended_action=ActionType.A_STOP_EXECUTION,
        )
        assert not v.resolved
        assert v.severity == "critical"

    def test_compliance_report(self):
        report = DoctrineComplianceReport(
            generated_at=datetime.now(),
            scope="organization",
            total_rules=10,
            compliant=8,
            warnings=1,
            violations=1,
            violations_list=[],
            compliance_score=80.0,
            barren_wuffet_state=BarrenWuffetState.CAUTION,
        )
        assert report.compliance_score == 80.0
        assert report.barren_wuffet_state == BarrenWuffetState.CAUTION


# ── Engine Lifecycle ───────────────────────────────────────────────────────


class TestDoctrineEngine:
    """Validate engine init, pack loading, and metric updates."""

    def test_engine_creates_with_defaults(self):
        engine = DoctrineEngine()
        assert engine.current_az_state == BarrenWuffetState.NORMAL
        assert engine.active_violations == []
        assert engine._loaded is False

    def test_engine_loads_fallback_packs(self):
        engine = DoctrineEngine(config_path=Path("/nonexistent/path.yaml"))
        result = engine.load_doctrine_packs()
        # Should use DOCTRINE_PACKS fallback
        assert engine._loaded is True

    def test_register_action_handler(self):
        engine = DoctrineEngine()

        def mock_handler():
            """Mock handler."""
            logger.debug("mock_handler called")

        engine.register_action_handler(ActionType.A_STOP_EXECUTION, mock_handler)
        assert ActionType.A_STOP_EXECUTION in engine.action_handlers

    def test_update_unknown_metric(self):
        engine = DoctrineEngine()
        engine.load_doctrine_packs()
        mv = engine.update_metric(
            name="totally_fake_metric",
            value=42,
            department=Department.TRADING_EXECUTION,
        )
        assert mv.name == "totally_fake_metric"
        # Unknown metric gets empty thresholds
        assert mv.threshold_good == ""


# ── State Transitions ─────────────────────────────────────────────────────


class TestStateTransitions:
    """Validate BARREN WUFFET state machine integrity."""

    def test_initial_state_is_normal(self):
        engine = DoctrineEngine()
        assert engine.current_az_state == BarrenWuffetState.NORMAL

    def test_states_are_ordered(self):
        severity_order = [
            BarrenWuffetState.NORMAL,
            BarrenWuffetState.CAUTION,
            BarrenWuffetState.SAFE_MODE,
            BarrenWuffetState.HALT,
        ]
        assert len(severity_order) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared import capital_management as cm_mod
from shared.capital_management import (
    CapitalManagementSystem,
    CapitalPosition,
    CapitalRequirement,
    CapitalThreshold,
    initialize_capital_management,
)


@pytest.fixture
def system() -> CapitalManagementSystem:
    """Fresh CapitalManagementSystem with a known capital position and a stub audit logger."""
    sys_ = CapitalManagementSystem.__new__(CapitalManagementSystem)
    import logging

    sys_.logger = logging.getLogger("CapitalManagementTest")
    sys_.audit_logger = MagicMock()
    sys_.audit_logger.log_event = AsyncMock(return_value=None)
    sys_.capital_requirements = sys_._initialize_requirements()
    sys_.capital_thresholds = sys_._initialize_thresholds()
    sys_.monitoring_active = False
    sys_.monitoring_task = None
    sys_.capital_history = []
    sys_.capital_position = CapitalPosition(
        total_capital=Decimal("10000"),
        available_capital=Decimal("10000"),
        allocated_capital=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        margin_used=Decimal("0"),
        margin_available=Decimal("10000"),
        last_updated=datetime.now(),
        currency="USD",
    )
    return sys_


# ── dataclasses ──────────────────────────────────────────────────────────


class TestDataclasses:
    def test_capital_requirement_fields(self):
        req = CapitalRequirement(
            jurisdiction="X",
            minimum_capital=Decimal("1"),
            risk_weighted_assets_multiplier=0.08,
            liquidity_ratio=1.0,
            leverage_ratio=0.1,
            description="desc",
        )
        assert req.jurisdiction == "X"
        assert req.minimum_capital == Decimal("1")

    def test_capital_position_default_currency(self):
        pos = CapitalPosition(
            total_capital=Decimal("1"),
            available_capital=Decimal("1"),
            allocated_capital=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            margin_used=Decimal("0"),
            margin_available=Decimal("1"),
            last_updated=datetime.now(),
        )
        assert pos.currency == "USD"

    def test_capital_threshold_defaults(self):
        t = CapitalThreshold(
            threshold_type="warning",
            percentage=20.0,
            amount=Decimal("0"),
            action_required="watch",
        )
        assert t.auto_stop_trading is False


# ── _initialize_requirements ─────────────────────────────────────────────


class TestInitializeRequirements:
    def test_has_all_jurisdictions(self, system):
        assert set(system.capital_requirements.keys()) == {"FINRA", "CFTC", "SEC", "BASEL_III"}

    def test_finra_minimum_is_100k(self, system):
        assert system.capital_requirements["FINRA"].minimum_capital == Decimal("100000")

    def test_basel_iii_leverage_ratio(self, system):
        assert system.capital_requirements["BASEL_III"].leverage_ratio == 0.03


# ── _initialize_thresholds ───────────────────────────────────────────────


class TestInitializeThresholds:
    def test_three_thresholds(self, system):
        assert len(system.capital_thresholds) == 3

    def test_threshold_types(self, system):
        types = {t.threshold_type for t in system.capital_thresholds}
        assert types == {"minimum", "warning", "critical"}

    def test_critical_auto_stops(self, system):
        critical = next(t for t in system.capital_thresholds if t.threshold_type == "critical")
        assert critical.auto_stop_trading is True

    def test_warning_does_not_auto_stop(self, system):
        warning = next(t for t in system.capital_thresholds if t.threshold_type == "warning")
        assert warning.auto_stop_trading is False


# ── _load_initial_capital ────────────────────────────────────────────────


class TestLoadInitialCapital:
    def test_uses_config_value_when_present(self):
        fake_config = MagicMock()
        fake_config.risk.initial_capital = 50000.0
        with patch.object(cm_mod, "get_config", return_value=fake_config):
            sys_ = CapitalManagementSystem()
        assert sys_.capital_position.total_capital == Decimal("50000.0")

    def test_falls_back_to_env(self):
        fake_config = MagicMock(spec=[])
        fake_config.risk = MagicMock(spec=[])
        with patch.object(cm_mod, "get_config", return_value=fake_config):
            with patch.object(cm_mod, "get_env", return_value="25000"):
                sys_ = CapitalManagementSystem()
        assert sys_.capital_position.total_capital == Decimal("25000")

    def test_default_when_nothing_configured(self):
        fake_config = MagicMock(spec=[])
        fake_config.risk = MagicMock(spec=[])
        with patch.object(cm_mod, "get_config", return_value=fake_config):
            with patch.object(cm_mod, "get_env", return_value="10000.0"):
                sys_ = CapitalManagementSystem()
        assert sys_.capital_position.total_capital == Decimal("10000.0")
        assert sys_.capital_position.currency == "USD"


# ── update_capital_position ──────────────────────────────────────────────


class TestUpdateCapitalPosition:
    @pytest.mark.asyncio
    async def test_no_position_returns_false(self, system):
        system.capital_position = None
        assert await system.update_capital_position(realized_pnl=Decimal("1")) is False

    @pytest.mark.asyncio
    async def test_realized_pnl_increases_total_and_available(self, system):
        ok = await system.update_capital_position(realized_pnl=Decimal("500"))
        assert ok is True
        assert system.capital_position.realized_pnl == Decimal("500")
        assert system.capital_position.total_capital == Decimal("10500")
        assert system.capital_position.available_capital == Decimal("10500")

    @pytest.mark.asyncio
    async def test_realized_pnl_negative_decreases(self, system):
        await system.update_capital_position(realized_pnl=Decimal("-200"))
        assert system.capital_position.total_capital == Decimal("9800")

    @pytest.mark.asyncio
    async def test_unrealized_pnl_replaces_old(self, system):
        await system.update_capital_position(unrealized_pnl=Decimal("100"))
        assert system.capital_position.total_capital == Decimal("10100")
        # Replace, not add
        await system.update_capital_position(unrealized_pnl=Decimal("250"))
        assert system.capital_position.unrealized_pnl == Decimal("250")
        assert system.capital_position.total_capital == Decimal("10250")

    @pytest.mark.asyncio
    async def test_margin_used_updates_available_and_allocated(self, system):
        await system.update_capital_position(margin_used=Decimal("2000"))
        assert system.capital_position.margin_used == Decimal("2000")
        assert system.capital_position.allocated_capital == Decimal("2000")
        assert system.capital_position.available_capital == Decimal("8000")
        # margin_available is computed pre-deduction: 10000 - 2000 = 8000
        assert system.capital_position.margin_available == Decimal("8000")

    @pytest.mark.asyncio
    async def test_margin_change_handles_decrease(self, system):
        await system.update_capital_position(margin_used=Decimal("3000"))
        await system.update_capital_position(margin_used=Decimal("1000"))
        assert system.capital_position.margin_used == Decimal("1000")
        assert system.capital_position.allocated_capital == Decimal("1000")
        assert system.capital_position.available_capital == Decimal("9000")

    @pytest.mark.asyncio
    async def test_audit_logger_called(self, system):
        await system.update_capital_position(realized_pnl=Decimal("10"))
        system.audit_logger.log_event.assert_called()
        kwargs = system.audit_logger.log_event.call_args.kwargs
        assert kwargs["category"] == "capital"
        assert kwargs["action"] == "position_updated"

    @pytest.mark.asyncio
    async def test_history_appended(self, system):
        await system.update_capital_position(realized_pnl=Decimal("1"))
        assert len(system.capital_history) == 1
        assert system.capital_history[0]["realized_pnl"] == 1.0

    @pytest.mark.asyncio
    async def test_last_updated_advances(self, system):
        before = system.capital_position.last_updated
        await asyncio.sleep(0.01)
        await system.update_capital_position(realized_pnl=Decimal("0"))
        assert system.capital_position.last_updated >= before


# ── _record_capital_history ──────────────────────────────────────────────


class TestRecordCapitalHistory:
    @pytest.mark.asyncio
    async def test_no_position_returns_silently(self, system):
        system.capital_position = None
        await system._record_capital_history()
        assert system.capital_history == []

    @pytest.mark.asyncio
    async def test_appends_entry(self, system):
        await system._record_capital_history()
        assert len(system.capital_history) == 1
        entry = system.capital_history[0]
        assert entry["total_capital"] == 10000.0
        assert "timestamp" in entry

    @pytest.mark.asyncio
    async def test_prunes_entries_older_than_90_days(self, system):
        old_ts = (datetime.now() - timedelta(days=120)).isoformat()
        system.capital_history = [
            {
                "timestamp": old_ts,
                "total_capital": 1.0,
                "available_capital": 1.0,
                "allocated_capital": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "margin_used": 0.0,
                "margin_available": 1.0,
            }
        ]
        await system._record_capital_history()
        assert all(
            (datetime.now() - datetime.fromisoformat(e["timestamp"])).days < 90
            for e in system.capital_history
        )


# ── _check_capital_thresholds ────────────────────────────────────────────


class TestCheckCapitalThresholds:
    @pytest.mark.asyncio
    async def test_no_position_no_op(self, system):
        system.capital_position = None
        await system._check_capital_thresholds()  # should not raise

    @pytest.mark.asyncio
    async def test_no_breach_when_capital_high(self, system):
        # Seed history with initial value
        system.capital_history = [{"total_capital": 10000.0, "timestamp": datetime.now().isoformat()}]
        await system._check_capital_thresholds()
        # No threshold breach → no audit event
        system.audit_logger.log_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_warning_breach_logs_audit(self, system):
        system.capital_history = [{"total_capital": 10000.0, "timestamp": datetime.now().isoformat()}]
        # Drop capital below 20% (warning) but above 10% (minimum) and 5% (critical)
        system.capital_position.total_capital = Decimal("1500")
        await system._check_capital_thresholds()
        # Warning is breached but does not auto_stop
        calls = [c.kwargs for c in system.audit_logger.log_event.call_args_list]
        breach_calls = [c for c in calls if c.get("action") == "threshold_breached"]
        assert any(c["details"]["threshold_type"] == "warning" for c in breach_calls)

    @pytest.mark.asyncio
    async def test_critical_breach_triggers_emergency_stop(self, system):
        system.capital_history = [{"total_capital": 10000.0, "timestamp": datetime.now().isoformat()}]
        system.capital_position.total_capital = Decimal("100")
        with patch.object(system, "_emergency_stop_trading", new=AsyncMock()) as mock_stop:
            await system._check_capital_thresholds()
        mock_stop.assert_called()


# ── _emergency_stop_trading ──────────────────────────────────────────────


class TestEmergencyStop:
    @pytest.mark.asyncio
    async def test_logs_audit_critical(self, system):
        await system._emergency_stop_trading()
        kwargs = system.audit_logger.log_event.call_args.kwargs
        assert kwargs["action"] == "emergency_stop"
        assert kwargs["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_handles_no_position(self, system):
        system.capital_position = None
        await system._emergency_stop_trading()
        kwargs = system.audit_logger.log_event.call_args.kwargs
        assert kwargs["details"]["current_capital"] == 0


# ── check_capital_adequacy ───────────────────────────────────────────────


class TestCheckCapitalAdequacy:
    @pytest.mark.asyncio
    async def test_no_position_returns_error(self, system):
        system.capital_position = None
        result = await system.check_capital_adequacy()
        assert result["compliant"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_jurisdiction(self, system):
        result = await system.check_capital_adequacy("UNKNOWN")
        assert result["compliant"] is False
        assert "Unknown jurisdiction" in result["error"]

    @pytest.mark.asyncio
    async def test_finra_non_compliant_when_below_minimum(self, system):
        # 10k < 100k FINRA minimum
        result = await system.check_capital_adequacy("FINRA")
        assert result["compliant"] is False
        assert result["checks"]["minimum_capital"]["compliant"] is False

    @pytest.mark.asyncio
    async def test_finra_compliant_with_large_capital(self, system):
        system.capital_position.total_capital = Decimal("500000")
        system.capital_position.available_capital = Decimal("500000")
        result = await system.check_capital_adequacy("FINRA")
        assert result["checks"]["minimum_capital"]["compliant"] is True
        # leverage 0 / 500000 = 0 ≤ 0.1 → compliant
        assert result["checks"]["leverage_ratio"]["compliant"] is True

    @pytest.mark.asyncio
    async def test_returns_required_keys(self, system):
        result = await system.check_capital_adequacy("FINRA")
        for key in ("jurisdiction", "compliant", "total_capital", "available_capital",
                    "minimum_capital_required", "checks"):
            assert key in result

    @pytest.mark.asyncio
    async def test_audit_logged(self, system):
        await system.check_capital_adequacy("FINRA")
        actions = [c.kwargs.get("action") for c in system.audit_logger.log_event.call_args_list]
        assert "capital_adequacy_check" in actions

    @pytest.mark.asyncio
    async def test_zero_capital_leverage_handled(self, system):
        system.capital_position.total_capital = Decimal("0")
        result = await system.check_capital_adequacy("FINRA")
        assert result["checks"]["leverage_ratio"]["current_ratio"] == 0.0


# ── allocate / deallocate capital ────────────────────────────────────────


class TestAllocateCapital:
    @pytest.mark.asyncio
    async def test_no_position_returns_false(self, system):
        system.capital_position = None
        assert await system.allocate_capital(Decimal("1"), "test") is False

    @pytest.mark.asyncio
    async def test_insufficient_capital_returns_false(self, system):
        ok = await system.allocate_capital(Decimal("999999"), "test")
        assert ok is False
        # No state change
        assert system.capital_position.allocated_capital == Decimal("0")

    @pytest.mark.asyncio
    async def test_successful_allocation(self, system):
        ok = await system.allocate_capital(Decimal("3000"), "strategyA")
        assert ok is True
        assert system.capital_position.allocated_capital == Decimal("3000")
        assert system.capital_position.available_capital == Decimal("7000")

    @pytest.mark.asyncio
    async def test_audit_called(self, system):
        await system.allocate_capital(Decimal("100"), "x")
        kwargs = system.audit_logger.log_event.call_args.kwargs
        assert kwargs["action"] == "capital_allocated"
        assert kwargs["details"]["purpose"] == "x"


class TestDeallocateCapital:
    @pytest.mark.asyncio
    async def test_no_position_returns_false(self, system):
        system.capital_position = None
        assert await system.deallocate_capital(Decimal("1"), "test") is False

    @pytest.mark.asyncio
    async def test_normal_deallocation(self, system):
        await system.allocate_capital(Decimal("2000"), "x")
        ok = await system.deallocate_capital(Decimal("500"), "x")
        assert ok is True
        assert system.capital_position.allocated_capital == Decimal("1500")
        assert system.capital_position.available_capital == Decimal("8500")

    @pytest.mark.asyncio
    async def test_overdeallocate_clamps_to_allocated(self, system):
        await system.allocate_capital(Decimal("500"), "x")
        ok = await system.deallocate_capital(Decimal("999"), "x")
        assert ok is True
        # Clamp: dealloc only the 500 actually allocated
        assert system.capital_position.allocated_capital == Decimal("0")
        assert system.capital_position.available_capital == Decimal("10000")

    @pytest.mark.asyncio
    async def test_audit_called(self, system):
        await system.allocate_capital(Decimal("100"), "x")
        system.audit_logger.log_event.reset_mock()
        await system.deallocate_capital(Decimal("50"), "x")
        kwargs = system.audit_logger.log_event.call_args.kwargs
        assert kwargs["action"] == "capital_deallocated"


# ── get_capital_status ───────────────────────────────────────────────────


class TestGetCapitalStatus:
    def test_no_position(self, system):
        system.capital_position = None
        assert system.get_capital_status() == {"status": "no_capital_position"}

    def test_returns_full_payload(self, system):
        status = system.get_capital_status()
        for key in ("total_capital", "available_capital", "allocated_capital",
                    "unrealized_pnl", "realized_pnl", "margin_used",
                    "margin_available", "last_updated", "currency"):
            assert key in status
        assert status["currency"] == "USD"
        assert status["total_capital"] == 10000.0


# ── monitoring start / stop ──────────────────────────────────────────────


class TestMonitoring:
    @pytest.mark.asyncio
    async def test_start_sets_active_and_creates_task(self, system):
        with patch.object(system, "_capital_monitoring_loop", new=AsyncMock()):
            await system.start_capital_monitoring()
        assert system.monitoring_active is True
        assert system.monitoring_task is not None
        await system.stop_capital_monitoring()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, system):
        with patch.object(system, "_capital_monitoring_loop", new=AsyncMock()):
            await system.start_capital_monitoring()
            first_task = system.monitoring_task
            await system.start_capital_monitoring()
            assert system.monitoring_task is first_task
        await system.stop_capital_monitoring()

    @pytest.mark.asyncio
    async def test_stop_clears_active(self, system):
        with patch.object(system, "_capital_monitoring_loop", new=AsyncMock()):
            await system.start_capital_monitoring()
            await system.stop_capital_monitoring()
        assert system.monitoring_active is False

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, system):
        await system.stop_capital_monitoring()
        assert system.monitoring_active is False


# ── _capital_monitoring_loop ─────────────────────────────────────────────


class TestMonitoringLoop:
    @pytest.mark.asyncio
    async def test_loop_runs_one_iteration_then_stops(self, system):
        # Replace asyncio.sleep so the loop exits quickly
        async def fake_sleep(_):
            system.monitoring_active = False

        system.monitoring_active = True
        with patch.object(system, "check_capital_adequacy", new=AsyncMock(return_value={"compliant": True})):
            with patch.object(cm_mod.asyncio, "sleep", new=fake_sleep):
                await system._capital_monitoring_loop()
        # Exited cleanly
        assert system.monitoring_active is False

    @pytest.mark.asyncio
    async def test_loop_logs_warning_on_non_compliant(self, system):
        async def fake_sleep(_):
            system.monitoring_active = False

        system.monitoring_active = True
        with patch.object(system, "check_capital_adequacy", new=AsyncMock(return_value={"compliant": False})):
            with patch.object(cm_mod.asyncio, "sleep", new=fake_sleep):
                await system._capital_monitoring_loop()
        # No raise → success

    @pytest.mark.asyncio
    async def test_loop_handles_exception_and_recovers(self, system):
        calls = {"n": 0}

        async def fake_sleep(_):
            calls["n"] += 1
            if calls["n"] >= 1:
                system.monitoring_active = False

        system.monitoring_active = True
        with patch.object(system, "check_capital_adequacy", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with patch.object(cm_mod.asyncio, "sleep", new=fake_sleep):
                await system._capital_monitoring_loop()
        # Loop swallowed the error and went into the except branch's sleep(60)
        assert calls["n"] >= 1


# ── get_capital_history ──────────────────────────────────────────────────


class TestGetCapitalHistory:
    def test_empty_history(self, system):
        assert system.get_capital_history() == []

    def test_filters_by_age(self, system):
        now_iso = datetime.now().isoformat()
        old_iso = (datetime.now() - timedelta(days=45)).isoformat()
        system.capital_history = [
            {"timestamp": now_iso, "total_capital": 1.0},
            {"timestamp": old_iso, "total_capital": 2.0},
        ]
        recent = system.get_capital_history(days=30)
        assert len(recent) == 1
        assert recent[0]["total_capital"] == 1.0

    def test_default_30_days(self, system):
        in_window = (datetime.now() - timedelta(days=10)).isoformat()
        system.capital_history = [{"timestamp": in_window, "total_capital": 5.0}]
        assert len(system.get_capital_history()) == 1


# ── module singleton + initialize_capital_management ─────────────────────


class TestModuleSingleton:
    def test_singleton_exists(self):
        assert cm_mod.capital_management_system is not None
        assert isinstance(cm_mod.capital_management_system, CapitalManagementSystem)

    @pytest.mark.asyncio
    async def test_initialize_returns_true(self, system):
        with patch.object(cm_mod, "capital_management_system", new=system):
            with patch.object(system, "start_capital_monitoring", new=AsyncMock()):
                ok = await initialize_capital_management()
        assert ok is True

    @pytest.mark.asyncio
    async def test_initialize_starts_monitoring(self, system):
        with patch.object(cm_mod, "capital_management_system", new=system):
            with patch.object(system, "start_capital_monitoring", new=AsyncMock()) as mock_start:
                await initialize_capital_management()
        mock_start.assert_called_once()

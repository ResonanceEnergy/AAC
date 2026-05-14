"""tests/test_risk_roll.py — Sprint 3.5 end-to-end risk & roll tests.

Tests:
  * ExposureCalculator — per-position risk, portfolio totals, stop/TP, alerts
  * ExposureCalculator.check_new_position — pre-trade gate
  * RollManager — all roll actions (HOLD, ROLL, CLOSE, DEAD_PUT, EXPIRED, NOT_OPTION)
  * days_to_expiry utility
  * PositionSnapshot expiry/strike/right fields (from position_tracker update)

No real IBKR connections — all positions constructed inline.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _snap(
    symbol: str = "SPY",
    sec_type: str = "OPT",
    quantity: float = -10,
    avg_cost: float = 1.50,
    market_price: float = 1.20,
    market_value: float = -120.0,
    unrealized_pnl: float = -30.0,
    expiry: str | None = None,
    strike: float | None = 510.0,
    right: str | None = "P",
    multiplier: int = 100,
):
    """Build a lightweight PositionSnapshot mock."""
    snap = MagicMock()
    snap.symbol = symbol
    snap.sec_type = sec_type
    snap.quantity = quantity
    snap.avg_cost = avg_cost
    snap.market_price = market_price
    snap.market_value = market_value
    snap.unrealized_pnl = unrealized_pnl
    snap.realized_pnl = 0.0
    snap.expiry = expiry
    snap.strike = strike
    snap.right = right
    snap.multiplier = multiplier
    snap.is_long = quantity > 0
    return snap


def _expiry(dte: int) -> str:
    """Return YYYYMMDD expiry that is ``dte`` days from today (UTC)."""
    d = date.today() + timedelta(days=dte)
    return d.strftime("%Y%m%d")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: days_to_expiry utility
# ═══════════════════════════════════════════════════════════════════════════════

class TestDaysToExpiry:
    def test_future_expiry(self):
        from strategies.roll_manager import days_to_expiry

        exp = _expiry(30)
        assert days_to_expiry(exp) == 30

    def test_today_is_zero(self):
        from strategies.roll_manager import days_to_expiry

        exp = _expiry(0)
        assert days_to_expiry(exp) == 0

    def test_past_is_negative(self):
        from strategies.roll_manager import days_to_expiry

        exp = _expiry(-5)
        assert days_to_expiry(exp) == -5

    def test_dashed_format_supported(self):
        from strategies.roll_manager import days_to_expiry

        d = date.today() + timedelta(days=10)
        dashed = d.strftime("%Y-%m-%d")
        assert days_to_expiry(dashed) == 10


# ═══════════════════════════════════════════════════════════════════════════════
# Test: ExposureCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestExposureCalculator:
    def test_basic_report_structure(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        pos = _snap(market_value=-500.0, unrealized_pnl=-50.0, avg_cost=5.50)
        report = calc.calculate([pos], account_value_usd=10_000)

        assert report.account_value_usd == 10_000
        assert report.position_count == 1
        assert report.total_exposure_usd == pytest.approx(500.0)
        assert report.total_unrealized_pnl == pytest.approx(-50.0)
        assert "generated_at" in report.to_dict()

    def test_no_alerts_healthy_portfolio(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        pos = _snap(
            market_value=-300.0,
            unrealized_pnl=-30.0,
            avg_cost=3.30,
            quantity=-10,
        )
        report = calc.calculate([pos], account_value_usd=10_000)
        assert report.exposure_ok
        assert report.alerts == []

    def test_stop_loss_alert_fires(self):
        from strategies.risk_engine import ExposureCalculator, RiskConfig

        cfg = RiskConfig(stop_loss_pct=-0.50)
        calc = ExposureCalculator(config=cfg)
        # cost_basis = 10 * 2.00 * 100 = 2000; unrealised = -1200 → -60%
        pos = _snap(
            quantity=-10, avg_cost=2.00, market_value=-800.0,
            unrealized_pnl=-1200.0,
        )
        report = calc.calculate([pos], account_value_usd=50_000)
        stop_alerts = [a for a in report.alerts if "STOP" in a]
        assert len(stop_alerts) == 1
        assert "SPY" in stop_alerts[0]

    def test_take_profit_alert_fires(self):
        from strategies.risk_engine import ExposureCalculator, RiskConfig

        cfg = RiskConfig(take_profit_pct=1.00)
        calc = ExposureCalculator(config=cfg)
        # cost_basis = 10 * 0.50 * 100 = 500; unrealised = +600 → +120%
        pos = _snap(
            quantity=-10, avg_cost=0.50, market_value=-1100.0,
            unrealized_pnl=600.0,
        )
        report = calc.calculate([pos], account_value_usd=50_000)
        tp_alerts = [a for a in report.alerts if "TP" in a]
        assert len(tp_alerts) == 1

    def test_oversized_position_alert(self):
        from strategies.risk_engine import ExposureCalculator, RiskConfig

        cfg = RiskConfig(max_single_position_pct=0.15)
        calc = ExposureCalculator(config=cfg)
        # market_value = -20,000 → 20% of 100k
        pos = _snap(market_value=-20_000.0, unrealized_pnl=0.0, avg_cost=20.0)
        report = calc.calculate([pos], account_value_usd=100_000)
        size_alerts = [a for a in report.alerts if "SIZE" in a]
        assert len(size_alerts) == 1

    def test_total_overexposure_alert(self):
        from strategies.risk_engine import ExposureCalculator, RiskConfig

        cfg = RiskConfig(max_total_exposure_pct=0.80)
        calc = ExposureCalculator(config=cfg)
        # Three positions totalling 90% of 10k account
        positions = [
            _snap(symbol="SPY", market_value=-3_000.0, unrealized_pnl=0.0, avg_cost=3.0),
            _snap(symbol="QQQ", market_value=-3_000.0, unrealized_pnl=0.0, avg_cost=3.0),
            _snap(symbol="HYG", market_value=-3_100.0, unrealized_pnl=0.0, avg_cost=3.0),
        ]
        report = calc.calculate(positions, account_value_usd=10_000)
        over_alerts = [a for a in report.alerts if "OVEREXPOSED" in a]
        assert len(over_alerts) == 1

    def test_oversized_contracts_alert(self):
        from strategies.risk_engine import ExposureCalculator, RiskConfig

        cfg = RiskConfig(max_contracts_per_position=20)
        calc = ExposureCalculator(config=cfg)
        pos = _snap(quantity=-25, sec_type="OPT")  # 25 contracts > 20 limit
        report = calc.calculate([pos], account_value_usd=10_000)
        contract_alerts = [a for a in report.alerts if "CONTRACTS" in a]
        assert len(contract_alerts) == 1

    def test_position_risk_dict_keys(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        pos = _snap()
        report = calc.calculate([pos], account_value_usd=10_000)
        pr_dict = report.positions[0].to_dict()
        required = {
            "symbol", "sec_type", "market_value", "unrealized_pnl",
            "cost_basis", "pnl_pct_of_cost", "account_pct",
            "stop_triggered", "take_profit_triggered",
        }
        assert required <= pr_dict.keys()


class TestCheckNewPosition:
    def test_valid_order_approved(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        ok, reason = calc.check_new_position(
            ticker="SPY",
            size_fraction=0.05,
            account_value_usd=50_000,
            current_positions=[],
        )
        assert ok
        assert reason == ""

    def test_oversized_order_rejected(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        ok, reason = calc.check_new_position(
            ticker="SPY",
            size_fraction=0.20,   # 20% > 15% limit
            account_value_usd=50_000,
            current_positions=[],
        )
        assert not ok
        assert "SPY" in reason

    def test_too_many_contracts_rejected(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        ok, reason = calc.check_new_position(
            ticker="IWM",
            size_fraction=0.05,
            account_value_usd=50_000,
            current_positions=[],
            contracts=25,   # > 20 limit
        )
        assert not ok
        assert "contracts" in reason

    def test_total_exposure_breach_rejected(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        # Already at 75% deployed; adding 10% pushes to 85% > 80% limit
        existing = [
            _snap(symbol="SPY", market_value=-7_500.0, unrealized_pnl=0.0, avg_cost=7.5),
        ]
        ok, reason = calc.check_new_position(
            ticker="QQQ",
            size_fraction=0.10,
            account_value_usd=10_000,
            current_positions=existing,
        )
        assert not ok
        assert "exposure" in reason.lower()

    def test_zero_account_value_rejected(self):
        from strategies.risk_engine import ExposureCalculator

        calc = ExposureCalculator()
        ok, reason = calc.check_new_position("SPY", 0.05, 0.0, [])
        assert not ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test: RollManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollManager:
    def test_hold_far_dated(self):
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager()
        pos = _snap(expiry=_expiry(60))
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.HOLD
        assert not decision.urgent

    def test_roll_at_trigger(self):
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager(roll_trigger_dte=21)
        pos = _snap(expiry=_expiry(15))  # inside 21-DTE trigger, above 7
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.ROLL
        assert decision.urgent

    def test_close_at_7dte(self):
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager()
        pos = _snap(expiry=_expiry(5))
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.CLOSE
        assert decision.urgent

    def test_dead_put_gate(self):
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager()
        pos = _snap(expiry=_expiry(15), market_price=0.01)  # essentially $0
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.DEAD_PUT
        assert "do NOT roll" in decision.reason

    def test_expired_position(self):
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager()
        pos = _snap(expiry=_expiry(-3))
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.EXPIRED
        assert decision.urgent

    def test_non_option_not_evaluated(self):
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager()
        pos = _snap(sec_type="STK", expiry=None)
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.NOT_OPTION
        assert decision.dte is None
        assert not decision.urgent

    def test_no_expiry_returns_hold(self):
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager()
        pos = _snap(expiry=None, sec_type="OPT")
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.HOLD
        assert "Expiry date unavailable" in decision.reason

    def test_urgent_only_filters_correctly(self):
        from strategies.roll_manager import RollManager

        mgr = RollManager()
        positions = [
            _snap(symbol="SPY", expiry=_expiry(60)),    # HOLD — not urgent
            _snap(symbol="QQQ", expiry=_expiry(10)),    # ROLL — urgent
            _snap(symbol="HYG", expiry=_expiry(-2)),    # EXPIRED — urgent
        ]
        urgent = mgr.urgent_only(positions)
        assert len(urgent) == 2
        assert {d.symbol for d in urgent} == {"QQQ", "HYG"}

    def test_summary_structure(self):
        from strategies.roll_manager import RollManager

        mgr = RollManager()
        positions = [
            _snap(symbol="A", expiry=_expiry(30)),
            _snap(symbol="B", expiry=_expiry(15)),
        ]
        summary = mgr.summary(positions)
        assert "total_evaluated" in summary
        assert "urgent_count" in summary
        assert "by_action" in summary
        assert summary["total_evaluated"] == 2

    def test_roll_decision_to_dict(self):
        from strategies.roll_manager import RollManager

        mgr = RollManager()
        pos = _snap(expiry=_expiry(10))
        decision = mgr.evaluate([pos])[0]
        d = decision.to_dict()
        required = {"symbol", "sec_type", "dte", "action", "reason", "urgent"}
        assert required <= d.keys()

    def test_exactly_at_roll_trigger(self):
        """Position at exactly 21 DTE must trigger ROLL."""
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager(roll_trigger_dte=21)
        pos = _snap(expiry=_expiry(21))
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.ROLL

    def test_exactly_at_close_trigger(self):
        """Position at exactly 7 DTE must trigger CLOSE."""
        from strategies.roll_manager import RollAction, RollManager

        mgr = RollManager(close_trigger_dte=7)
        pos = _snap(expiry=_expiry(7))
        decision = mgr.evaluate([pos])[0]
        assert decision.action == RollAction.CLOSE


# ═══════════════════════════════════════════════════════════════════════════════
# Test: PositionSnapshot expiry/strike/right fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionSnapshotOptionFields:
    @pytest.mark.asyncio
    async def test_tracker_maps_expiry_strike_right(self):
        """PositionTracker.refresh() correctly maps expiry/strike/right from raw IBKR data."""
        from TradingExecution.position_tracker import PositionTracker
        from unittest.mock import AsyncMock

        raw = {
            "symbol": "SPY",
            "sec_type": "OPT",
            "quantity": -5,
            "avg_cost": 1.80,
            "market_price": 1.20,
            "market_value": -600.0,
            "unrealized_pnl": -300.0,
            "realized_pnl": 0.0,
            "exchange": "SMART",
            "currency": "USD",
            "expiry": "20261219",
            "strike": 500.0,
            "right": "P",
            "multiplier": "100",
        }

        tracker = PositionTracker(paper=False)
        mock_ibkr = AsyncMock()
        mock_ibkr.get_positions = AsyncMock(return_value=[raw])
        tracker._ibkr = mock_ibkr

        positions = await tracker.refresh()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.expiry == "20261219"
        assert pos.strike == 500.0
        assert pos.right == "P"
        assert pos.multiplier == 100

    @pytest.mark.asyncio
    async def test_tracker_equity_has_no_expiry(self):
        """STK positions have no expiry/strike/right."""
        from TradingExecution.position_tracker import PositionTracker
        from unittest.mock import AsyncMock

        raw = {
            "symbol": "AAPL",
            "sec_type": "STK",
            "quantity": 10,
            "avg_cost": 195.0,
            "market_price": 200.0,
            "market_value": 2000.0,
            "unrealized_pnl": 50.0,
            "realized_pnl": 0.0,
            "exchange": "SMART",
            "currency": "USD",
        }

        tracker = PositionTracker(paper=False)
        mock_ibkr = AsyncMock()
        mock_ibkr.get_positions = AsyncMock(return_value=[raw])
        tracker._ibkr = mock_ibkr

        positions = await tracker.refresh()
        pos = positions[0]
        assert pos.expiry is None
        assert pos.strike is None
        assert pos.right is None

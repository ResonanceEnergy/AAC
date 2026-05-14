"""Tests for CentralAccounting.pnl_tracker — Sprint 4.5.

All tests use in-memory SQLite — no disk I/O, fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import pytest

from CentralAccounting.pnl_tracker import DailyPnlRow, PnLTracker


# ── Helpers ───────────────────────────────────────────────────────────────────


@dataclass
class _FakePosition:
    """Minimal stand-in for TradingExecution.position_tracker.PositionSnapshot."""

    symbol: str
    sec_type: str = "STK"
    quantity: float = 1.0
    avg_cost: float = 100.0
    market_price: float = 105.0
    market_value: float = 105.0
    unrealized_pnl: float = 5.0
    realized_pnl: float = 0.0
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None


def _opt(
    symbol: str = "SPY",
    strike: float = 400.0,
    right: str = "P",
    expiry: str = "20251231",
    unrealized_pnl: float = -20.0,
    market_value: float = 80.0,
) -> _FakePosition:
    return _FakePosition(
        symbol=symbol,
        sec_type="OPT",
        quantity=1.0,
        avg_cost=100.0,
        market_price=market_value,
        market_value=market_value,
        unrealized_pnl=unrealized_pnl,
        expiry=expiry,
        strike=strike,
        right=right,
    )


@pytest.fixture
def tracker():
    t = PnLTracker(":memory:")
    yield t
    t.close()


# ── TestPnlStore ──────────────────────────────────────────────────────────────


class TestPnlStore:
    """Schema init and connectivity."""

    def test_memory_db_creates_tables(self):
        t = PnLTracker(":memory:")
        # If tables are missing, subsequent calls will raise — run a harmless query
        report = t.today_report()
        assert "positions" in report
        t.close()

    def test_double_close_safe(self, tracker):
        tracker.close()
        tracker.close()  # must not raise


# ── TestTakeSnapshot ──────────────────────────────────────────────────────────


class TestTakeSnapshot:
    """take_snapshot() writes daily_pnl and position_snapshots."""

    def test_empty_positions(self, tracker):
        report = tracker.take_snapshot(positions=[], account_value_usd=10_000.0)
        pnl = report["daily_pnl"]
        assert pnl is not None
        assert pnl["account_value_usd"] == 10_000.0
        assert pnl["position_count"] == 0
        assert pnl["total_unrealized_pnl"] == 0.0

    def test_single_equity_position(self, tracker):
        pos = _FakePosition("AAPL", unrealized_pnl=500.0, market_value=10_500.0)
        report = tracker.take_snapshot([pos], account_value_usd=50_000.0, snapshot_date="2025-01-15")
        pnl = report["daily_pnl"]
        assert pnl["total_unrealized_pnl"] == 500.0
        assert pnl["position_count"] == 1
        assert len(report["positions"]) == 1
        assert report["positions"][0]["symbol"] == "AAPL"

    def test_multiple_positions_aggregate(self, tracker):
        positions = [
            _FakePosition("AAPL", unrealized_pnl=200.0, market_value=10_200.0),
            _FakePosition("MSFT", unrealized_pnl=-100.0, market_value=9_900.0),
            _opt(unrealized_pnl=-50.0, market_value=50.0),
        ]
        report = tracker.take_snapshot(positions, account_value_usd=30_000.0, snapshot_date="2025-01-16")
        pnl = report["daily_pnl"]
        assert pnl["total_unrealized_pnl"] == pytest.approx(50.0, abs=0.01)
        assert pnl["position_count"] == 3
        assert pnl["total_exposure_usd"] == pytest.approx(20_150.0, abs=0.01)

    def test_idempotent_upsert_same_day(self, tracker):
        pos = _FakePosition("TSLA", unrealized_pnl=100.0, market_value=5_100.0)
        # First call
        tracker.take_snapshot([pos], account_value_usd=50_000.0, snapshot_date="2025-02-01")
        # Second call on same day — should REPLACE
        tracker.take_snapshot([pos], account_value_usd=55_000.0, snapshot_date="2025-02-01")
        history = tracker.historical_summary(days=5)
        # Only one row for that date
        dates = [r["snapshot_date"] for r in history]
        assert dates.count("2025-02-01") == 1
        row = next(r for r in history if r["snapshot_date"] == "2025-02-01")
        assert row["account_value_usd"] == 55_000.0

    def test_option_fields_persisted(self, tracker):
        pos = _opt(symbol="SPY", strike=450.0, right="P", expiry="20251231")
        tracker.take_snapshot([pos], account_value_usd=20_000.0, snapshot_date="2025-03-01")
        report = tracker.today_report(snapshot_date="2025-03-01")
        snap = report["positions"][0]
        assert snap["sec_type"] == "OPT"
        assert snap["strike"] == 450.0
        assert snap["right"] == "P"
        assert snap["expiry"] == "20251231"

    def test_snapshot_returns_dict_with_required_keys(self, tracker):
        report = tracker.take_snapshot([], account_value_usd=0.0)
        assert "date" in report
        assert "daily_pnl" in report
        assert "positions" in report
        assert "today_trades" in report

    def test_realized_pnl_accumulated(self, tracker):
        pos = _FakePosition("TSLA", unrealized_pnl=0.0, realized_pnl=300.0, market_value=0.0)
        report = tracker.take_snapshot([pos], account_value_usd=50_000.0, snapshot_date="2025-04-01")
        assert report["daily_pnl"]["total_realized_pnl"] == pytest.approx(300.0)


# ── TestLogTrade ──────────────────────────────────────────────────────────────


class TestLogTrade:
    """log_trade() and log_trade_from_confirmation()."""

    def test_log_trade_returns_id(self, tracker):
        row_id = tracker.log_trade("AAPL", "LONG", quantity=10, fill_price=175.0)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_log_trade_multiple_sequential_ids(self, tracker):
        id1 = tracker.log_trade("AAPL", "LONG", 5, 170.0)
        id2 = tracker.log_trade("MSFT", "SHORT", 3, 300.0)
        assert id2 > id1

    def test_log_trade_all_fields(self, tracker):
        tracker.log_trade(
            "NVDA", "LONG", quantity=2, fill_price=800.0,
            order_id="ORD-123", status="filled", strategy="momentum", confidence=0.85
        )
        trades = tracker.recent_trades(limit=1)
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "NVDA"
        assert t["direction"] == "LONG"
        assert t["fill_price"] == 800.0
        assert t["order_id"] == "ORD-123"
        assert t["strategy"] == "momentum"
        assert t["confidence"] == pytest.approx(0.85)

    def test_log_trade_from_confirmation_simple_namespace(self, tracker):
        conf = SimpleNamespace(
            symbol="IWM",
            direction="SHORT",
            quantity=5.0,
            filled_price=195.0,
            order_id="ORD-999",
            status="filled",
            strategy="put_spread",
            confidence=0.72,
        )
        row_id = tracker.log_trade_from_confirmation(conf)
        assert row_id >= 1
        trades = tracker.recent_trades(limit=1)
        t = trades[0]
        assert t["symbol"] == "IWM"
        assert t["direction"] == "SHORT"
        assert t["fill_price"] == pytest.approx(195.0)

    def test_log_trade_from_confirmation_missing_fields_safe(self, tracker):
        """Confirmation with minimal attrs should not crash."""
        conf = SimpleNamespace(symbol="GLD")
        row_id = tracker.log_trade_from_confirmation(conf)
        assert row_id >= 1

    def test_log_trade_from_confirmation_status_enum(self, tracker):
        """status that has .value attr (enum) is coerced to str."""
        class _Status:
            value = "partial_fill"

        conf = SimpleNamespace(
            symbol="QQQ", direction="LONG", quantity=1.0,
            filled_price=None, order_id=None, status=_Status(),
            strategy=None, confidence=None,
        )
        row_id = tracker.log_trade_from_confirmation(conf)
        trades = tracker.recent_trades(1)
        assert trades[0]["status"] == "partial_fill"

    def test_log_trade_no_fill_price(self, tracker):
        """fill_price=None is stored as NULL and doesn't crash."""
        row_id = tracker.log_trade("TLT", "LONG", quantity=1, fill_price=None)
        trades = tracker.recent_trades(1)
        assert trades[0]["fill_price"] is None


# ── TestTodayReport ───────────────────────────────────────────────────────────


class TestTodayReport:
    """today_report() structure."""

    def test_empty_db_returns_empty_report(self, tracker):
        report = tracker.today_report(snapshot_date="2025-01-01")
        assert report["daily_pnl"] is None
        assert report["positions"] == []
        assert report["today_trades"] == []

    def test_report_after_snapshot(self, tracker):
        pos = _FakePosition("AMZN", unrealized_pnl=250.0)
        tracker.take_snapshot([pos], 75_000.0, snapshot_date="2025-05-01")
        report = tracker.today_report(snapshot_date="2025-05-01")
        assert report["daily_pnl"]["total_unrealized_pnl"] == pytest.approx(250.0)
        assert len(report["positions"]) == 1

    def test_report_includes_trades_for_same_day(self, tracker):
        """Trades logged today should appear in today_report()."""
        import datetime
        today = datetime.date.today().isoformat()
        tracker.log_trade("SPY", "LONG", 5, 420.0)
        report = tracker.today_report(snapshot_date=today)
        assert len(report["today_trades"]) == 1

    def test_report_date_field(self, tracker):
        report = tracker.today_report(snapshot_date="2030-12-31")
        assert report["date"] == "2030-12-31"


# ── TestHistoricalSummary ─────────────────────────────────────────────────────


class TestHistoricalSummary:
    """historical_summary() and pnl_delta()."""

    def _fill_two_days(self, tracker):
        tracker.take_snapshot(
            [_FakePosition("AAPL", unrealized_pnl=100.0)],
            50_000.0, snapshot_date="2025-06-01",
        )
        tracker.take_snapshot(
            [_FakePosition("AAPL", unrealized_pnl=200.0)],
            50_500.0, snapshot_date="2025-06-02",
        )

    def test_returns_most_recent_first(self, tracker):
        self._fill_two_days(tracker)
        rows = tracker.historical_summary(days=10)
        assert len(rows) == 2
        assert rows[0]["snapshot_date"] == "2025-06-02"
        assert rows[1]["snapshot_date"] == "2025-06-01"

    def test_limit_respected(self, tracker):
        for i in range(5):
            tracker.take_snapshot([], 10_000.0 + i, snapshot_date=f"2025-07-0{i + 1}")
        rows = tracker.historical_summary(days=3)
        assert len(rows) == 3

    def test_pnl_delta_positive(self, tracker):
        self._fill_two_days(tracker)
        delta = tracker.pnl_delta(days=2)
        assert delta == pytest.approx(100.0, abs=0.01)

    def test_pnl_delta_negative(self, tracker):
        tracker.take_snapshot(
            [_FakePosition("QQQ", unrealized_pnl=500.0)],
            60_000.0, snapshot_date="2025-08-01",
        )
        tracker.take_snapshot(
            [_FakePosition("QQQ", unrealized_pnl=300.0)],
            59_800.0, snapshot_date="2025-08-02",
        )
        delta = tracker.pnl_delta(days=2)
        assert delta == pytest.approx(-200.0, abs=0.01)

    def test_pnl_delta_single_row_returns_zero(self, tracker):
        tracker.take_snapshot([], 50_000.0, snapshot_date="2025-09-01")
        delta = tracker.pnl_delta(days=2)
        assert delta == 0.0

    def test_empty_history_returns_empty_list(self, tracker):
        rows = tracker.historical_summary(days=30)
        assert rows == []


# ── TestAllPnlRows ────────────────────────────────────────────────────────────


class TestAllPnlRows:
    """all_pnl_rows() returns typed DailyPnlRow objects."""

    def test_typed_objects(self, tracker):
        tracker.take_snapshot([], 50_000.0, snapshot_date="2025-10-01")
        rows = tracker.all_pnl_rows()
        assert len(rows) == 1
        assert isinstance(rows[0], DailyPnlRow)
        assert rows[0].account_value_usd == pytest.approx(50_000.0)

    def test_total_pnl_property(self, tracker):
        pos = _FakePosition("X", unrealized_pnl=100.0, realized_pnl=50.0)
        tracker.take_snapshot([pos], 10_000.0, snapshot_date="2025-10-02")
        rows = tracker.all_pnl_rows()
        assert rows[0].total_pnl == pytest.approx(150.0)

    def test_to_dict_keys(self, tracker):
        tracker.take_snapshot([], 1_000.0, snapshot_date="2025-10-03")
        d = tracker.all_pnl_rows()[0].to_dict()
        for key in ("date", "account_value_usd", "total_unrealized_pnl",
                    "total_realized_pnl", "total_pnl", "position_count"):
            assert key in d


# ── TestFormatReport ─────────────────────────────────────────────────────────


class TestFormatReport:
    """format_report() generates non-empty CLI string."""

    def test_empty_report_no_crash(self, tracker):
        report = tracker.today_report(snapshot_date="2020-01-01")
        text = PnLTracker.format_report(report)
        assert isinstance(text, str)
        assert "P&L Report" in text

    def test_report_with_data(self, tracker):
        pos = _FakePosition("BRK", unrealized_pnl=1_200.0, market_value=50_000.0)
        tracker.take_snapshot([pos], 100_000.0, snapshot_date="2025-11-01")
        tracker.log_trade("BRK", "LONG", 1, 48_000.0)
        report = tracker.today_report(snapshot_date="2025-11-01")
        text = PnLTracker.format_report(report)
        assert "BRK" in text
        assert "1,200" in text  # unrealised P&L in the positions table

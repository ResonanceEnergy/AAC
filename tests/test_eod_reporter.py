"""tests/test_eod_reporter.py — Sprint 13: EOD Reporter tests.

Coverage plan (30 tests):
  TestPositionSummary        (3)  — dataclass, to_dict
  TestEodReportDataclass     (5)  — defaults, to_dict, format_text sections
  TestEodReportFormatText    (8)  — header, P&L, positions, roll, API health, alerts
  TestEodReportWriteToFile   (4)  — writes file, path override, dir creation, failure
  TestEodReporterGenerate    (8)  — full pipeline, empty inputs, health parse, pnl parse
  TestMarketSchedulerWiring  (4)  — run_pnl_snapshot calls _run_eod_report
  TestGetUrgentSymbols       (3)  — roll manager wiring
  TestExtractApiHealth       (4)  — snapshot parsing
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.eod_reporter import (
    EodReport,
    EodReporter,
    PositionSummary,
    _extract_api_health,
    _get_urgent_symbols,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_position(symbol="SPY", quantity=10.0, market_value=5000.0, unrealized_pnl=200.0):
    pos = MagicMock()
    pos.symbol = symbol
    pos.quantity = quantity
    pos.market_value = market_value
    pos.unrealized_pnl = unrealized_pnl
    return pos


def _make_pnl_report(
    acct=50000.0,
    unrealized=1200.0,
    realized=300.0,
    pos_count=3,
    trades=None,
):
    return {
        "date": "2026-04-21",
        "daily_pnl": {
            "account_value_usd": acct,
            "total_exposure_usd": 15000.0,
            "total_unrealized_pnl": unrealized,
            "total_realized_pnl": realized,
            "position_count": pos_count,
        },
        "positions": [],
        "today_trades": trades if trades is not None else [{"symbol": "SPY"}],
    }


def _make_health_snap(status_value="OK", api_names=("yfinance", "fred")):
    snap = MagicMock()
    snap.overall_status = MagicMock()
    snap.overall_status.value = status_value
    api_health = {}
    for name in api_names:
        comp = MagicMock()
        comp.status = MagicMock()
        comp.status.value = "OK"
        api_health[name] = comp
    snap.api_health = api_health
    snap.active_alerts = []
    return snap


# ── TestPositionSummary ────────────────────────────────────────────────────────

class TestPositionSummary:
    def test_defaults(self):
        ps = PositionSummary(symbol="IWM", quantity=5, market_value=1000.0, unrealized_pnl=-50.0)
        assert ps.roll_urgent is False

    def test_to_dict_keys(self):
        ps = PositionSummary("SPY", 10, 5000.0, 200.0, roll_urgent=True)
        d = ps.to_dict()
        assert set(d.keys()) == {"symbol", "quantity", "market_value", "unrealized_pnl", "roll_urgent"}

    def test_to_dict_values(self):
        ps = PositionSummary("HYG", 20, 2000.0, -100.0, roll_urgent=True)
        d = ps.to_dict()
        assert d["symbol"] == "HYG"
        assert d["roll_urgent"] is True
        assert d["quantity"] == 20


# ── TestEodReportDataclass ─────────────────────────────────────────────────────

class TestEodReportDataclass:
    def test_defaults(self):
        r = EodReport(report_date="2026-04-21")
        assert r.account_value_usd == 0.0
        assert r.position_count == 0
        assert r.positions == []
        assert r.overall_api_status == "UNKNOWN"

    def test_to_dict_keys(self):
        r = EodReport(report_date="2026-04-21")
        d = r.to_dict()
        expected = {
            "report_date", "account_value_usd", "total_unrealized_pnl",
            "total_realized_pnl", "pnl_delta", "position_count", "positions",
            "roll_urgent_count", "trades_today", "api_health_summary",
            "overall_api_status", "active_alerts", "written_at",
            "drawdown_pct", "drawdown_tripped",
        }
        assert set(d.keys()) == expected

    def test_to_dict_positions_serialized(self):
        ps = PositionSummary("SPY", 5, 2500.0, 100.0)
        r = EodReport(report_date="2026-04-21", positions=[ps])
        d = r.to_dict()
        assert isinstance(d["positions"], list)
        assert d["positions"][0]["symbol"] == "SPY"

    def test_format_text_returns_string(self):
        r = EodReport(report_date="2026-04-21", written_at="2026-04-21 16:05 UTC")
        text = r.format_text()
        assert isinstance(text, str)
        assert len(text) > 100

    def test_format_text_contains_date(self):
        r = EodReport(report_date="2026-04-21")
        assert "2026-04-21" in r.format_text()


# ── TestEodReportFormatText ────────────────────────────────────────────────────

class TestEodReportFormatText:
    def _base_report(self, **kwargs) -> EodReport:
        return EodReport(report_date="2026-04-21", **kwargs)

    def test_header_present(self):
        text = self._base_report().format_text()
        assert "End-of-Day Brief" in text

    def test_pnl_section_shows_values(self):
        r = self._base_report(
            account_value_usd=50000.0,
            total_unrealized_pnl=1200.0,
            total_realized_pnl=300.0,
        )
        text = r.format_text()
        assert "50,000.00" in text
        assert "1,200.00" in text

    def test_pnl_delta_positive_prefix(self):
        r = self._base_report(pnl_delta=150.0)
        assert "+$150" in r.format_text()

    def test_pnl_delta_negative_no_extra_prefix(self):
        r = self._base_report(pnl_delta=-75.0)
        text = r.format_text()
        assert "$-75.00" in text

    def test_positions_section_shows_symbols(self):
        ps = PositionSummary("IWM", 10, 2000.0, -80.0)
        r = self._base_report(positions=[ps])
        assert "IWM" in r.format_text()

    def test_roll_urgent_flag_in_text(self):
        ps = PositionSummary("KRE", 5, 900.0, -200.0, roll_urgent=True)
        r = self._base_report(positions=[ps], roll_urgent_count=1)
        text = r.format_text()
        assert "KRE" in text
        assert "!!" in text

    def test_no_positions_shows_message(self):
        r = self._base_report(positions=[])
        assert "No open positions" in r.format_text()

    def test_api_health_section(self):
        r = self._base_report(
            api_health_summary={"yfinance": "OK", "fred": "DEGRADED"},
            overall_api_status="DEGRADED",
        )
        text = r.format_text()
        assert "yfinance" in text
        assert "DEGRADED" in text

    def test_active_alerts_shown(self):
        r = self._base_report(
            active_alerts=[{"severity": "HIGH", "message": "IBKR disconnected"}]
        )
        assert "IBKR disconnected" in r.format_text()

    def test_no_positions_no_roll_section_clean(self):
        r = self._base_report(roll_urgent_count=0)
        assert "No positions require rolling" in r.format_text()


# ── TestEodReportWriteToFile ──────────────────────────────────────────────────

class TestEodReportWriteToFile:
    def test_writes_file(self, tmp_path):
        r = EodReport(report_date="2026-04-21", written_at="now")
        path = tmp_path / "brief.txt"
        result = r.write_to_file(path)
        assert result == str(path)
        assert path.exists()
        assert "2026-04-21" in path.read_text(encoding="utf-8")

    def test_creates_missing_directory(self, tmp_path):
        r = EodReport(report_date="2026-04-21")
        nested = tmp_path / "a" / "b" / "c" / "brief.txt"
        result = r.write_to_file(nested)
        assert nested.exists()
        assert result == str(nested)

    def test_returns_empty_string_on_write_failure(self):
        r = EodReport(report_date="2026-04-21")
        result = r.write_to_file("/\x00/invalid/path/brief.txt")
        assert result == ""

    def test_path_override_used(self, tmp_path):
        r = EodReport(report_date="2026-04-21")
        p = tmp_path / "custom.txt"
        r.write_to_file(p)
        assert p.exists()


# ── TestEodReporterGenerate ───────────────────────────────────────────────────

class TestEodReporterGenerate:
    def test_returns_eod_report_type(self, tmp_path):
        reporter = EodReporter()
        result = reporter.generate(report_path=tmp_path / "brief.txt")
        assert isinstance(result, EodReport)

    def test_all_none_inputs_safe(self, tmp_path):
        reporter = EodReporter()
        result = reporter.generate(
            pnl_report=None,
            positions=None,
            health_snap=None,
            report_path=tmp_path / "brief.txt",
        )
        assert result.account_value_usd == 0.0
        assert result.trades_today == 0

    def test_pnl_report_values_extracted(self, tmp_path):
        reporter = EodReporter()
        pnl = _make_pnl_report(acct=75000.0, unrealized=2000.0, realized=500.0, pos_count=4)
        result = reporter.generate(pnl_report=pnl, report_path=tmp_path / "brief.txt")
        assert result.account_value_usd == 75000.0
        assert result.total_unrealized_pnl == 2000.0
        assert result.total_realized_pnl == 500.0
        assert result.position_count == 4

    def test_trades_today_counted(self, tmp_path):
        reporter = EodReporter()
        pnl = _make_pnl_report(trades=[{"symbol": "SPY"}, {"symbol": "IWM"}])
        result = reporter.generate(pnl_report=pnl, report_path=tmp_path / "brief.txt")
        assert result.trades_today == 2

    def test_positions_become_summaries(self, tmp_path):
        reporter = EodReporter()
        positions = [_make_position("SPY"), _make_position("IWM")]
        with patch("core.eod_reporter._get_urgent_symbols", return_value=set()):
            result = reporter.generate(positions=positions, report_path=tmp_path / "brief.txt")
        assert len(result.positions) == 2
        assert result.positions[0].symbol == "SPY"

    def test_health_snap_parsed(self, tmp_path):
        reporter = EodReporter()
        snap = _make_health_snap(status_value="DEGRADED", api_names=["yfinance", "fred"])
        result = reporter.generate(health_snap=snap, report_path=tmp_path / "brief.txt")
        assert result.overall_api_status == "DEGRADED"
        assert "yfinance" in result.api_health_summary

    def test_pnl_delta_passed_through(self, tmp_path):
        reporter = EodReporter()
        result = reporter.generate(pnl_delta=250.0, report_path=tmp_path / "brief.txt")
        assert result.pnl_delta == 250.0

    def test_roll_urgency_counted(self, tmp_path):
        reporter = EodReporter()
        positions = [_make_position("KRE"), _make_position("IWM")]
        with patch("core.eod_reporter._get_urgent_symbols", return_value={"KRE"}):
            result = reporter.generate(positions=positions, report_path=tmp_path / "brief.txt")
        assert result.roll_urgent_count == 1
        urgent = [p for p in result.positions if p.roll_urgent]
        assert urgent[0].symbol == "KRE"


# ── TestMarketSchedulerWiring ─────────────────────────────────────────────────

class TestMarketSchedulerWiring:
    """run_pnl_snapshot() must call _run_eod_report() and still return the report."""

    def _make_scheduler(self):
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler.__new__(MarketScheduler)
        sched._stop_event = MagicMock()
        sched._stop_event.is_set.return_value = False
        sched._last_run = {}
        sched._last_date = {}
        sched._paper = False
        sched._position_tracker = MagicMock()
        pnl_mock = MagicMock()
        pnl_mock.take_snapshot.return_value = {"daily_pnl": {}, "today_trades": []}
        pnl_mock.today_report.return_value = {"daily_pnl": {}, "today_trades": []}
        pnl_mock.pnl_delta.return_value = 0.0
        sched._pnl_tracker = pnl_mock
        sched._daily_loss_guard = MagicMock()
        sched._auto_trader = None
        return sched

    def test_run_pnl_snapshot_returns_dict(self):
        sched = self._make_scheduler()
        with (
            patch.object(sched, "_fetch_positions", return_value=[]),
            patch.object(sched, "_run_eod_report") as mock_eod,
        ):
            result = sched.run_pnl_snapshot()
        assert isinstance(result, dict)
        mock_eod.assert_called_once()

    def test_run_eod_report_called_with_positions(self):
        sched = self._make_scheduler()
        positions = [_make_position("SPY")]
        with (
            patch.object(sched, "_fetch_positions", return_value=positions),
            patch.object(sched, "_run_eod_report") as mock_eod,
        ):
            sched.run_pnl_snapshot()
        _, kwargs = mock_eod.call_args
        assert kwargs["positions"] == positions

    def test_eod_report_still_called_when_snapshot_fails(self):
        sched = self._make_scheduler()
        sched._pnl_tracker.take_snapshot.side_effect = RuntimeError("db error")
        with (
            patch.object(sched, "_fetch_positions", return_value=[]),
            patch.object(sched, "_run_eod_report") as mock_eod,
        ):
            result = sched.run_pnl_snapshot()
        assert result == {}
        mock_eod.assert_called_once()

    def test_run_eod_report_exception_does_not_propagate(self):
        sched = self._make_scheduler()
        with (
            patch.object(sched, "_fetch_positions", return_value=[]),
            patch("core.eod_reporter.EodReporter.generate", side_effect=RuntimeError("boom")),
        ):
            # Should not raise
            sched._run_eod_report(report={}, positions=[])


# ── TestGetUrgentSymbols ──────────────────────────────────────────────────────

class TestGetUrgentSymbols:
    def test_returns_set(self):
        with patch("strategies.roll_manager.RollManager") as MockRM:
            decision = MagicMock()
            decision.symbol = "IWM"
            MockRM.return_value.urgent_only.return_value = [decision]
            result = _get_urgent_symbols([MagicMock()])
        assert isinstance(result, set)
        assert "IWM" in result

    def test_returns_empty_on_exception(self):
        with patch("strategies.roll_manager.RollManager", side_effect=RuntimeError("boom")):
            result = _get_urgent_symbols([])
        assert result == set()

    def test_empty_positions_returns_empty(self):
        with patch("strategies.roll_manager.RollManager") as MockRM:
            MockRM.return_value.urgent_only.return_value = []
            result = _get_urgent_symbols([])
        assert result == set()


# ── TestExtractApiHealth ──────────────────────────────────────────────────────

class TestExtractApiHealth:
    def test_extracts_api_names(self):
        snap = _make_health_snap(api_names=["yfinance", "fred", "finnhub"])
        api_map, _ = _extract_api_health(snap)
        assert set(api_map.keys()) == {"yfinance", "fred", "finnhub"}

    def test_extracts_overall_status(self):
        snap = _make_health_snap(status_value="DEGRADED")
        _, overall = _extract_api_health(snap)
        assert overall == "DEGRADED"

    def test_returns_unknown_on_broken_snap(self):
        _, overall = _extract_api_health(None)
        assert overall == "UNKNOWN"

    def test_api_map_empty_on_missing_api_health(self):
        snap = MagicMock()
        del snap.api_health
        snap.overall_status = MagicMock()
        snap.overall_status.value = "OK"
        api_map, _ = _extract_api_health(snap)
        assert api_map == {}

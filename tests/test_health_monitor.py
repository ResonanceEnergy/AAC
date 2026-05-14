from __future__ import annotations

"""Tests for monitoring.health_monitor — Sprint 5."""

import os
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from monitoring.health_monitor import (
    HealthMonitor,
    SystemSnapshot,
    _check_coingecko,
    _check_finnhub,
    _check_fred,
    _check_ibkr_port,
    _check_unusual_whales,
    _check_yfinance,
)
from shared.health_checker import ComponentHealth
from shared.health_checker import HealthStatus


# ── helpers ───────────────────────────────────────────────────────────────


def _make_snapshot(
    *,
    overall: str = "healthy",
    api_health: dict[str, ComponentHealth] | None = None,
    positions_count: int = 3,
    pnl_today: float | None = 42.50,
    last_trade_symbol: str | None = "IWM",
    last_trade_at: str | None = "2026-04-21 10:30",
    active_alerts: list[dict[str, Any]] | None = None,
) -> SystemSnapshot:
    if api_health is None:
        api_health = {
            "yfinance": ComponentHealth(
                name="yfinance",
                status=HealthStatus.HEALTHY,
                details={"note": "SPY $520.00"},
            )
        }
    return SystemSnapshot(
        checked_at=datetime(2026, 4, 21, 12, 0, 0),
        api_health=api_health,
        overall_status=HealthStatus(overall),
        system_resources=ComponentHealth(
            name="system",
            status=HealthStatus.HEALTHY,
            details={"cpu_percent": 12.0, "memory_percent": 45.0, "disk_percent": 60.0},
        ),
        positions_count=positions_count,
        pnl_today=pnl_today,
        last_trade_symbol=last_trade_symbol,
        last_trade_at=last_trade_at,
        active_alerts=active_alerts or [],
    )


def _stub_checks(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return all-healthy check stubs, with optional per-name overrides."""
    defaults: dict[str, Any] = {
        "yfinance": lambda: {"status": "healthy", "note": "SPY ok"},
        "fred": lambda: {"status": "healthy", "note": "key set"},
        "finnhub": lambda: {"status": "healthy", "note": "key set"},
        "coingecko": lambda: {"status": "healthy", "note": "free tier"},
        "ibkr": lambda: {"status": "healthy", "note": "port 7496 reachable"},
        "unusual_whales": lambda: {"status": "degraded", "note": "field parsing broken"},
    }
    if overrides:
        defaults.update(overrides)
    return defaults


def _make_monitor(
    overrides: dict[str, Any] | None = None,
    alert_manager=None,
) -> HealthMonitor:
    return HealthMonitor(checks=_stub_checks(overrides), alert_manager=alert_manager)


def _mock_pnl(positions=None, pnl=None, trades=None):
    """Return a mock PnLTracker class that today_report returns controlled data.

    `pnl` is the desired pnl_today float; we wrap it in the real daily_pnl row
    schema (split arbitrarily into unrealized + 0 realized) so collect_snapshot
    sees the same dict shape as production.
    """
    if pnl is None:
        daily_pnl = None
    elif isinstance(pnl, dict):
        daily_pnl = pnl
    else:
        daily_pnl = {
            "total_unrealized_pnl": float(pnl),
            "total_realized_pnl": 0.0,
        }
    mock_cls = MagicMock()
    mock_cls.return_value.today_report.return_value = {
        "positions": positions or [],
        "daily_pnl": daily_pnl,
        "today_trades": trades or [],
    }
    return mock_cls


# ── _check_yfinance ──────────────────────────────────────────────────────


class TestCheckYfinance:
    def test_success_returns_healthy(self):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 520.0
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _check_yfinance()

        assert result["status"] == "healthy"
        assert "520.00" in result["note"]

    def test_runtime_error_returns_unhealthy(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("connection timeout")

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _check_yfinance()

        assert result["status"] == "unhealthy"
        assert "connection timeout" in result["note"]

    def test_non_numeric_price_still_healthy(self):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = None
        mock_fast_info.__getitem__ = MagicMock(return_value=None)
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _check_yfinance()

        assert result["status"] == "healthy"
        assert result["note"] == "SPY ok"


# ── _check_fred ──────────────────────────────────────────────────────────


class TestCheckFred:
    def test_key_set_is_healthy(self):
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key_xyz"}):
            result = _check_fred()
        assert result["status"] == "healthy"
        assert "configured" in result["note"]

    def test_key_missing_is_degraded(self):
        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = _check_fred()
        assert result["status"] == "degraded"
        assert "not set" in result["note"]


# ── _check_finnhub ───────────────────────────────────────────────────────


class TestCheckFinnhub:
    def test_key_set_is_healthy(self):
        with patch.dict(os.environ, {"FINNHUB_API_KEY": "test_key"}):
            result = _check_finnhub()
        assert result["status"] == "healthy"

    def test_key_missing_is_degraded(self):
        env = {k: v for k, v in os.environ.items() if k != "FINNHUB_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = _check_finnhub()
        assert result["status"] == "degraded"


# ── _check_coingecko ─────────────────────────────────────────────────────


class TestCheckCoingecko:
    def test_with_key_is_healthy(self):
        with patch.dict(os.environ, {"COINGECKO_API_KEY": "mykey"}):
            result = _check_coingecko()
        assert result["status"] == "healthy"
        assert "configured" in result["note"]

    def test_without_key_still_healthy(self):
        env = {k: v for k, v in os.environ.items() if k != "COINGECKO_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = _check_coingecko()
        # Free tier works without a key
        assert result["status"] == "healthy"
        assert "free tier" in result["note"]


# ── _check_ibkr_port ─────────────────────────────────────────────────────


class TestCheckIBKRPort:
    def test_port_7496_open_is_healthy(self):
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=MagicMock())
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("monitoring.health_monitor.socket.create_connection", return_value=mock_cm):
            result = _check_ibkr_port()

        assert result["status"] == "healthy"
        assert "7496" in result["note"]

    def test_both_ports_closed_is_degraded(self):
        with patch(
            "monitoring.health_monitor.socket.create_connection",
            side_effect=OSError("Connection refused"),
        ):
            result = _check_ibkr_port()

        assert result["status"] == "degraded"
        assert "closed" in result["note"]


# ── _check_unusual_whales ────────────────────────────────────────────────


class TestCheckUnusualWhales:
    def test_key_set_always_degraded(self):
        with patch.dict(os.environ, {"UNUSUAL_WHALES_API_KEY": "mykey"}):
            result = _check_unusual_whales()
        # Always degraded — field parsing known broken
        assert result["status"] == "degraded"
        assert "broken" in result["note"]

    def test_no_key_also_degraded(self):
        env = {k: v for k, v in os.environ.items() if k != "UNUSUAL_WHALES_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = _check_unusual_whales()
        assert result["status"] == "degraded"


# ── HealthMonitor.collect_snapshot ───────────────────────────────────────


class TestCollectSnapshot:
    def test_positions_and_pnl_populated(self):
        monitor = _make_monitor()
        with patch(
            "CentralAccounting.pnl_tracker.PnLTracker",
            _mock_pnl(
                positions=[1, 2, 3],
                pnl=100.0,
                trades=[{"symbol": "IWM", "logged_at": "2026-04-21 10:30:00"}],
            ),
        ):
            snap = monitor.collect_snapshot()

        assert snap.positions_count == 3
        assert snap.pnl_today == 100.0
        assert snap.last_trade_symbol == "IWM"
        assert snap.last_trade_at == "2026-04-21 10:30"

    def test_api_health_keys_present(self):
        monitor = _make_monitor()
        with patch(
            "CentralAccounting.pnl_tracker.PnLTracker",
            _mock_pnl(),
        ):
            snap = monitor.collect_snapshot()

        assert "yfinance" in snap.api_health
        assert "ibkr" in snap.api_health
        assert "fred" in snap.api_health

    def test_snapshot_has_timestamp(self):
        monitor = _make_monitor()
        with patch("CentralAccounting.pnl_tracker.PnLTracker", _mock_pnl()):
            snap = monitor.collect_snapshot()
        assert isinstance(snap.checked_at, datetime)

    def test_pnl_tracker_failure_is_graceful(self):
        monitor = _make_monitor()
        with patch(
            "CentralAccounting.pnl_tracker.PnLTracker",
            side_effect=RuntimeError("db unavailable"),
        ):
            snap = monitor.collect_snapshot()

        # Must not raise; positions default to zero
        assert snap.positions_count == 0
        assert snap.pnl_today is None

    def test_unhealthy_api_fires_alert(self):
        from shared.alert_manager import AlertManager

        alert_mgr = AlertManager()
        monitor = _make_monitor(
            overrides={"yfinance": lambda: {"status": "unhealthy", "note": "import error"}},
            alert_manager=alert_mgr,
        )
        with patch("CentralAccounting.pnl_tracker.PnLTracker", _mock_pnl()):
            snap = monitor.collect_snapshot()

        active = alert_mgr.get_active_alerts()
        assert any("yfinance" in a.title for a in active)
        assert len(snap.active_alerts) > 0

    def test_degraded_api_does_not_fire_alert(self):
        from shared.alert_manager import AlertManager

        alert_mgr = AlertManager()
        monitor = _make_monitor(
            overrides={"fred": lambda: {"status": "degraded", "note": "no key"}},
            alert_manager=alert_mgr,
        )
        with patch("CentralAccounting.pnl_tracker.PnLTracker", _mock_pnl()):
            monitor.collect_snapshot()

        assert len(alert_mgr.get_active_alerts()) == 0

    def test_overall_status_healthy_when_all_ok(self):
        monitor = _make_monitor(
            overrides={
                "unusual_whales": lambda: {"status": "healthy", "note": "ok"},
            }
        )
        with patch("CentralAccounting.pnl_tracker.PnLTracker", _mock_pnl()):
            snap = monitor.collect_snapshot()
        assert snap.overall_status.value in ("healthy", "degraded")

    def test_no_trades_defaults_to_none(self):
        monitor = _make_monitor()
        with patch("CentralAccounting.pnl_tracker.PnLTracker", _mock_pnl(trades=[])):
            snap = monitor.collect_snapshot()
        assert snap.last_trade_symbol is None
        assert snap.last_trade_at is None


# ── HealthMonitor.format_terminal ────────────────────────────────────────


class TestFormatTerminal:
    def test_contains_api_names(self):
        snap = _make_snapshot(
            api_health={
                "yfinance": ComponentHealth(
                    name="yfinance",
                    status=HealthStatus.HEALTHY,
                    details={"note": "SPY $520"},
                ),
                "fred": ComponentHealth(
                    name="fred",
                    status=HealthStatus.DEGRADED,
                    details={"note": "no key"},
                ),
            }
        )
        output = HealthMonitor.format_terminal(snap)
        assert "yfinance" in output
        assert "fred" in output
        assert "SPY $520" in output

    def test_positive_pnl_formatted(self):
        snap = _make_snapshot(pnl_today=123.45, positions_count=5)
        output = HealthMonitor.format_terminal(snap)
        assert "+123.45" in output
        assert "5" in output

    def test_none_pnl_shows_na(self):
        snap = _make_snapshot(pnl_today=None)
        output = HealthMonitor.format_terminal(snap)
        assert "n/a" in output

    def test_negative_pnl_formatted(self):
        snap = _make_snapshot(pnl_today=-55.0)
        output = HealthMonitor.format_terminal(snap)
        assert "-55.00" in output

    def test_last_trade_shown(self):
        snap = _make_snapshot(last_trade_symbol="SPY", last_trade_at="2026-04-21 10:30")
        output = HealthMonitor.format_terminal(snap)
        assert "SPY" in output
        assert "2026-04-21 10:30" in output

    def test_no_last_trade_shows_none(self):
        snap = _make_snapshot(last_trade_symbol=None, last_trade_at=None)
        output = HealthMonitor.format_terminal(snap)
        assert "none" in output

    def test_active_alert_shown(self):
        snap = _make_snapshot(
            active_alerts=[{"title": "API DOWN: yfinance", "severity": "error", "count": 2}]
        )
        output = HealthMonitor.format_terminal(snap)
        assert "API DOWN: yfinance" in output
        assert "x2" in output

    def test_no_alerts_message(self):
        snap = _make_snapshot(active_alerts=[])
        output = HealthMonitor.format_terminal(snap)
        assert "No active alerts" in output

    def test_system_resources_shown(self):
        snap = _make_snapshot()
        output = HealthMonitor.format_terminal(snap)
        assert "CPU" in output
        assert "Mem" in output

    def test_overall_healthy_label(self):
        snap = _make_snapshot(overall="healthy")
        output = HealthMonitor.format_terminal(snap)
        assert "OK" in output

    def test_overall_unhealthy_label(self):
        snap = _make_snapshot(overall="unhealthy")
        output = HealthMonitor.format_terminal(snap)
        assert "DOWN" in output

    def test_overall_degraded_label(self):
        snap = _make_snapshot(overall="degraded")
        output = HealthMonitor.format_terminal(snap)
        assert "WARN" in output

    def test_timestamp_in_header(self):
        snap = _make_snapshot()
        output = HealthMonitor.format_terminal(snap)
        assert "2026-04-21 12:00:00" in output

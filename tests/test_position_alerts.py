"""Tests for monitoring.position_alerts — position DTE, stale data, cash, P&L alerts."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from monitoring.position_alerts import (
    check_data_freshness,
    check_low_cash,
    check_pnl,
    check_position_dte,
    parse_expiry,
    run_all_checks,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_data(
    positions=None,
    balance=5000.0,
    updated=None,
    acct_name="test_acct",
    currency="USD",
):
    """Build a minimal account_balances structure."""
    if updated is None:
        updated = datetime.now().isoformat(timespec="seconds")
    return {
        "_meta": {"updated": updated},
        "fx": {"cad_usd": 0.70},
        "accounts": {
            acct_name: {
                "balance": balance,
                "currency": currency,
                "positions": positions or [],
            }
        },
    }


def _make_position(symbol="SPY", strike=400, right="P", expiry_date=None, qty=1, avg_cost=100.0, pnl=0.0):
    """Build a position dict."""
    if expiry_date is None:
        expiry_date = date.today() + timedelta(days=30)
    return {
        "symbol": symbol,
        "strike": strike,
        "right": right,
        "expiry": expiry_date.strftime("%Y%m%d"),
        "qty": qty,
        "avgCost": avg_cost,
        "unrealizedPNL": pnl,
    }


# ── Tests: expiry parsing ────────────────────────────────────────────────

def test_parse_expiry_yyyymmdd():
    pos = {"expiry": "20260417"}
    assert parse_expiry(pos) == date(2026, 4, 17)


def test_parse_expiry_dashed():
    pos = {"expiry": "2026-06-18"}
    assert parse_expiry(pos) == date(2026, 6, 18)


def test_parse_expiry_missing():
    assert parse_expiry({}) is None
    assert parse_expiry({"expiry": ""}) is None


# ── Tests: DTE alerts ───────────────────────────────────────────────────

def test_dte_alert_fires_at_7_days():
    expiry = date.today() + timedelta(days=7)
    pos = _make_position(symbol="ARCC", expiry_date=expiry)
    data = _make_data(positions=[pos])

    alerts = check_position_dte(data)
    assert len(alerts) == 1
    assert alerts[0]["meta"]["dte"] == 7
    assert alerts[0]["severity"] == "warning"


def test_dte_alert_fires_at_1_day():
    expiry = date.today() + timedelta(days=1)
    pos = _make_position(symbol="PFF", expiry_date=expiry)
    data = _make_data(positions=[pos])

    alerts = check_position_dte(data)
    assert len(alerts) == 1
    assert alerts[0]["meta"]["dte"] == 1
    assert alerts[0]["severity"] == "critical"


def test_dte_alert_expired():
    expiry = date.today() - timedelta(days=2)
    pos = _make_position(symbol="KRE", expiry_date=expiry)
    data = _make_data(positions=[pos])

    alerts = check_position_dte(data)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "critical"
    assert alerts[0]["meta"]["dte"] == -2


def test_dte_no_alert_when_far_out():
    expiry = date.today() + timedelta(days=60)
    pos = _make_position(symbol="HYG", expiry_date=expiry)
    data = _make_data(positions=[pos])

    alerts = check_position_dte(data)
    assert len(alerts) == 0


def test_dte_dedup_via_alert_id():
    """Same position produces same alert_id — CMS deduplicates."""
    expiry = date.today() + timedelta(days=3)
    pos = _make_position(symbol="JNK", expiry_date=expiry)
    data = _make_data(positions=[pos])

    alerts1 = check_position_dte(data)
    alerts2 = check_position_dte(data)
    # Pure function returns alerts every time; dedup is caller's job (CMS)
    assert len(alerts1) == 1
    assert len(alerts2) == 1
    assert alerts1[0]["alert_id"] == alerts2[0]["alert_id"]


# ── Tests: data freshness ───────────────────────────────────────────────

def test_stale_data_alert():
    old_time = (datetime.now() - timedelta(hours=48)).isoformat(timespec="seconds")
    data = _make_data(updated=old_time)

    alerts = check_data_freshness(data)
    assert len(alerts) == 1
    assert alerts[0]["meta"]["type"] == "data_stale"


def test_fresh_data_no_alert():
    data = _make_data()  # uses datetime.now()

    alerts = check_data_freshness(data)
    assert len(alerts) == 0


# ── Tests: low cash ─────────────────────────────────────────────────────

def test_low_cash_alert():
    pos = _make_position()
    data = _make_data(positions=[pos], balance=50.0)

    alerts = check_low_cash(data)
    assert len(alerts) == 1
    assert alerts[0]["meta"]["type"] == "low_cash"


def test_adequate_cash_no_alert():
    pos = _make_position()
    data = _make_data(positions=[pos], balance=5000.0)

    alerts = check_low_cash(data)
    assert len(alerts) == 0


def test_low_cash_no_alert_if_no_positions():
    """Low cash with no positions shouldn't alert."""
    data = _make_data(positions=[], balance=10.0)

    alerts = check_low_cash(data)
    assert len(alerts) == 0


# ── Tests: P&L ──────────────────────────────────────────────────────────

def test_pnl_warning():
    pos = _make_position(symbol="PFF", avg_cost=17.0, qty=1, pnl=-13.0)
    data = _make_data(positions=[pos])

    alerts = check_pnl(data)
    assert len(alerts) == 1
    assert alerts[0]["meta"]["type"] == "pnl_warning"


def test_pnl_ok_no_alert():
    pos = _make_position(symbol="LQD", avg_cost=63.0, qty=1, pnl=23.0)
    data = _make_data(positions=[pos])

    alerts = check_pnl(data)
    assert len(alerts) == 0


# ── Tests: run_all_checks ───────────────────────────────────────────────

def test_run_all_checks_with_mixed_data():
    """Test combined check with DTE + stale data + low cash."""
    balances = _make_data(
        positions=[
            _make_position(symbol="SPY", expiry_date=date.today() + timedelta(days=3), pnl=-80.0, avg_cost=100.0),
        ],
        balance=30.0,
        updated=(datetime.now() - timedelta(hours=30)).isoformat(timespec="seconds"),
    )

    alerts = run_all_checks(balances)
    # Should fire: DTE (3d), stale data (30h), low cash ($30), P&L (-80%)
    types = {a["meta"]["type"] for a in alerts}
    assert "position_dte" in types
    assert "data_stale" in types
    assert "low_cash" in types
    assert "pnl_warning" in types


def test_run_all_checks_empty_data():
    """Empty data should return no alerts."""
    alerts = run_all_checks({})
    assert alerts == []


def test_alert_has_required_keys():
    """Every alert from every check must have alert_id, severity, message."""
    expiry = date.today() + timedelta(days=5)
    pos = _make_position(symbol="IWM", expiry_date=expiry)
    data = _make_data(positions=[pos])

    alerts = check_position_dte(data)
    assert len(alerts) == 1
    alert = alerts[0]
    assert "alert_id" in alert
    assert "severity" in alert
    assert "message" in alert
    assert alert["severity"] in ("critical", "warning")

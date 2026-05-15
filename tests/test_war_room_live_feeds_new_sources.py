"""Tests for breadth / COT / ETF-flow fetchers added to war_room_live_feeds."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strategies.war_room_live_feeds import (
    LiveFeedResult,
    fetch_breadth_data,
    fetch_cot_data,
    fetch_etf_flow_data,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ── Breadth ────────────────────────────────────────────────────────────────

def test_fetch_breadth_data_populates_fields():
    snap = SimpleNamespace(
        trin=0.85,
        tick=420.0,
        advance_decline_ratio=2.10,
        mcclellan_oscillator=37.5,
        regime="bullish",
    )

    class _FakeClient:
        def get_snapshot(self):
            return snap

    result = LiveFeedResult()
    with patch("integrations.breadth_client.BreadthClient", lambda: _FakeClient()):
        _run(fetch_breadth_data(result))

    assert result.trin == 0.85
    assert result.tick == 420.0
    assert result.advance_decline_ratio == 2.10
    assert result.mcclellan_oscillator == 37.5
    assert result.breadth_regime == "bullish"
    assert not result.errors


def test_fetch_breadth_data_swallows_errors():
    class _BadClient:
        def get_snapshot(self):
            raise RuntimeError("yfinance down")

    result = LiveFeedResult()
    with patch("integrations.breadth_client.BreadthClient", lambda: _BadClient()):
        _run(fetch_breadth_data(result))

    assert result.trin is None
    assert any("Breadth fetch error" in e for e in result.errors)


# ── COT ────────────────────────────────────────────────────────────────────

def test_fetch_cot_data_populates_three_markets(monkeypatch):
    # Force a fresh client lookup
    monkeypatch.setattr(
        "strategies.war_room_live_feeds._cot_client", None, raising=False
    )
    monkeypatch.setattr(
        "strategies.war_room_live_feeds._cot_cache_payload", {}, raising=False
    )

    es_report = SimpleNamespace(leveraged_net=12_500, report_date="2026-04-08")
    nq_report = SimpleNamespace(leveraged_net=-3_400, report_date="2026-04-08")
    vx_report = SimpleNamespace(leveraged_net=-22_000, report_date="2026-04-08")
    es_sig = SimpleNamespace(signal="extreme_long", z_score=2.4)
    nq_sig = SimpleNamespace(signal="neutral", z_score=0.4)
    vx_sig = SimpleNamespace(signal="extreme_short", z_score=-2.1)

    by_market = {
        "ES": (es_report, es_sig),
        "NQ": (nq_report, nq_sig),
        "VX": (vx_report, vx_sig),
    }

    class _FakeClient:
        def get_latest(self, market):
            return by_market[market][0]

        def get_extreme_signal(self, market, lookback_weeks=52):
            assert lookback_weeks == 52
            return by_market[market][1]

    monkeypatch.setattr(
        "integrations.cftc_cot_client.CFTCCotClient", lambda: _FakeClient()
    )

    result = LiveFeedResult()
    _run(fetch_cot_data(result))

    assert result.cot_es_lev_money_net == 12_500
    assert result.cot_es_extreme == "extreme_long"
    assert result.cot_nq_lev_money_net == -3_400
    assert result.cot_nq_extreme == "neutral"
    assert result.cot_vx_lev_money_net == -22_000
    assert result.cot_vx_extreme == "extreme_short"
    assert result.cot_report_date == "2026-04-08"


def test_fetch_cot_data_handles_missing_market(monkeypatch):
    monkeypatch.setattr(
        "strategies.war_room_live_feeds._cot_client", None, raising=False
    )
    # Reset the shared payload cache so this fetch path actually runs
    monkeypatch.setattr(
        "strategies.war_room_live_feeds._cot_cache_payload", {}, raising=False
    )

    class _PartialClient:
        def get_latest(self, market):
            return None

        def get_extreme_signal(self, market, lookback_weeks=52):
            return None

    monkeypatch.setattr(
        "integrations.cftc_cot_client.CFTCCotClient", lambda: _PartialClient()
    )

    result = LiveFeedResult()
    _run(fetch_cot_data(result))

    assert result.cot_es_lev_money_net is None
    assert result.cot_nq_extreme is None
    assert not result.errors  # missing data is not an error


# ── ETF flow ───────────────────────────────────────────────────────────────

def test_fetch_etf_flow_data_aggregates_top_movers(monkeypatch):
    monkeypatch.setattr(
        "strategies.war_room_live_feeds._etf_flow_client", None, raising=False
    )

    snaps = [
        SimpleNamespace(symbol="SPY", daily_flow_usd=500_000_000.0),
        SimpleNamespace(symbol="QQQ", daily_flow_usd=200_000_000.0),
        SimpleNamespace(symbol="HYG", daily_flow_usd=-150_000_000.0),
        SimpleNamespace(symbol="IBIT", daily_flow_usd=None),  # not enough history
    ]

    class _FakeClient:
        def get_universe_snapshots(self, symbols=None, persist=True):
            return snaps

        def aggregate_flows(self, snapshots):
            return {
                "gross_inflow_usd": 700_000_000.0,
                "gross_outflow_usd": -150_000_000.0,
                "net_flow_usd": 550_000_000.0,
                "samples": 3,
            }

    monkeypatch.setattr(
        "integrations.etf_flow_client.ETFFlowClient", lambda: _FakeClient()
    )

    result = LiveFeedResult()
    _run(fetch_etf_flow_data(result))

    assert result.etf_net_flow_usd == 550_000_000.0
    assert result.etf_gross_inflow_usd == 700_000_000.0
    assert result.etf_gross_outflow_usd == -150_000_000.0
    assert result.etf_flow_samples == 3
    assert result.etf_top_inflow_ticker == "SPY"
    assert result.etf_top_outflow_ticker == "HYG"


def test_fetch_etf_flow_data_swallows_errors(monkeypatch):
    monkeypatch.setattr(
        "strategies.war_room_live_feeds._etf_flow_client", None, raising=False
    )

    class _BadClient:
        def get_universe_snapshots(self, symbols=None, persist=True):
            raise RuntimeError("yfinance down")

        def aggregate_flows(self, snapshots):  # pragma: no cover
            return {}

    monkeypatch.setattr(
        "integrations.etf_flow_client.ETFFlowClient", lambda: _BadClient()
    )

    result = LiveFeedResult()
    _run(fetch_etf_flow_data(result))

    assert result.etf_net_flow_usd is None
    assert any("ETF flow fetch error" in e for e in result.errors)


# ── Scheduler COT refresh ──────────────────────────────────────────────────

def test_scheduler_run_cot_refresh(monkeypatch):
    from core.market_scheduler import MarketScheduler

    es = SimpleNamespace(leveraged_net=12_500, report_date="2026-04-08")
    sig = SimpleNamespace(signal="extreme_long", z_score=2.4)

    class _FakeClient:
        def get_latest(self, market):
            return es

        def get_extreme_signal(self, market, lookback_weeks=52):
            return sig

    monkeypatch.setattr(
        "integrations.cftc_cot_client.CFTCCotClient", lambda: _FakeClient()
    )

    sched = MarketScheduler.__new__(MarketScheduler)  # bypass __init__
    out = MarketScheduler.run_cot_refresh(sched)

    assert set(out.keys()) == {"ES", "NQ", "RTY", "YM", "VX"}
    assert out["ES"]["leveraged_net"] == 12_500
    assert out["ES"]["signal"] == "extreme_long"
    assert out["ES"]["report_date"] == "2026-04-08"

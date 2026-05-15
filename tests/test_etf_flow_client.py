"""Tests for the ETF flow client (yfinance shares-outstanding delta method)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from integrations.etf_flow_client import (
    DEFAULT_UNIVERSE,
    ETFFlowClient,
    ETFFlowSnapshot,
)


def _fake_yf_module(shares: float, price: float) -> MagicMock:
    """Construct a MagicMock yfinance module returning the supplied
    shares/price for any Ticker(...).fast_info access."""
    fake_yf = MagicMock()

    def ticker_factory(_sym: str):
        ticker = MagicMock()
        # fast_info exposes both __getitem__ and attribute access
        fast_info = MagicMock()
        fast_info.__getitem__.side_effect = lambda key: {
            "shares": shares,
            "last_price": price,
            "previous_close": price,
            "market_cap": shares * price,
        }[key]
        ticker.fast_info = fast_info
        ticker.info = {
            "sharesOutstanding": shares,
            "regularMarketPrice": price,
            "marketCap": shares * price,
        }
        return ticker

    fake_yf.Ticker.side_effect = ticker_factory
    return fake_yf


class TestETFFlowSingleSnapshot:
    def test_first_call_has_no_flow(self, tmp_path: Path) -> None:
        client = ETFFlowClient(history_path=tmp_path / "h.json")
        with patch.object(client, "_yfinance", return_value=_fake_yf_module(100_000_000, 500.0)):
            snap = client.get_snapshot("SPY")

        assert isinstance(snap, ETFFlowSnapshot)
        assert snap.symbol == "SPY"
        assert snap.shares_outstanding == 100_000_000
        assert snap.nav_or_price == 500.0
        assert snap.daily_flow_usd is None  # need >=2 samples
        assert any("no prior snapshot" in n for n in snap.notes)
        # History persisted to disk
        assert (tmp_path / "h.json").exists()


class TestETFFlowDelta:
    def test_second_call_computes_flow(self, tmp_path: Path) -> None:
        history_path = tmp_path / "h.json"
        # Pre-seed history with yesterday
        history_path.write_text(json.dumps({
            "SPY": [{
                "date": "2026-03-01",
                "shares_outstanding": 100_000_000,
                "nav_or_price": 500.0,
                "total_assets": 5e10,
            }]
        }))

        client = ETFFlowClient(history_path=history_path)
        # Today: shares went UP by 2M → ~$1B inflow at $510 NAV
        with patch.object(client, "_yfinance", return_value=_fake_yf_module(102_000_000, 510.0)):
            snap = client.get_snapshot("SPY")

        assert snap.prev_shares_outstanding == 100_000_000
        assert snap.prev_date == "2026-03-01"
        assert snap.daily_flow_usd == pytest.approx(2_000_000 * 510.0)

    def test_outflow_when_shares_decline(self, tmp_path: Path) -> None:
        history_path = tmp_path / "h.json"
        history_path.write_text(json.dumps({
            "QQQ": [{"date": "2026-03-01", "shares_outstanding": 500_000_000, "nav_or_price": 400.0}]
        }))
        client = ETFFlowClient(history_path=history_path)
        with patch.object(client, "_yfinance", return_value=_fake_yf_module(498_000_000, 405.0)):
            snap = client.get_snapshot("QQQ")
        assert snap.daily_flow_usd is not None
        assert snap.daily_flow_usd < 0


class TestAggregate:
    def test_aggregate_classifies_directions(self) -> None:
        client = ETFFlowClient()
        snaps = [
            ETFFlowSnapshot(symbol="A", date="d", daily_flow_usd=1_000_000),
            ETFFlowSnapshot(symbol="B", date="d", daily_flow_usd=-500_000),
            ETFFlowSnapshot(symbol="C", date="d", daily_flow_usd=None),
        ]
        agg = client.aggregate_flows(snaps)
        assert agg["gross_inflow_usd"] == 1_000_000
        assert agg["gross_outflow_usd"] == -500_000
        assert agg["net_flow_usd"] == 500_000
        assert agg["samples"] == 2


class TestUniverseConstants:
    def test_default_universe_includes_core_etfs(self) -> None:
        for sym in ("SPY", "QQQ", "IWM", "GLD", "TLT"):
            assert sym in DEFAULT_UNIVERSE

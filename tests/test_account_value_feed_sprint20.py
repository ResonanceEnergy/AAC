from __future__ import annotations

"""tests/test_account_value_feed.py — Sprint 20: Live Account Value Feed.

Covers:
  * extract_net_liq() helper                  (5)
  * AccountValueFeed cache behaviour           (5)
  * Resolution path (ibkr → env → default)    (7)
  * _fetch_from_ibkr() connector interaction  (6)
  * MarketScheduler wiring                     (7)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.account_value_feed import (
    DEFAULT_ACCOUNT_VALUE,
    AccountValueFeed,
    extract_net_liq,
)

# ── helpers ────────────────────────────────────────────────────────────────

_PATCH_POSITION_TRACKER = "TradingExecution.position_tracker.PositionTracker"
_PATCH_PNL_TRACKER = "CentralAccounting.pnl_tracker.PnLTracker"


def _mock_connector(net_liq_usd: float = 12_345.67) -> MagicMock:
    """Build a mock IBKRConnector whose async methods behave correctly."""
    connector = MagicMock()
    connector.connect = AsyncMock()
    connector.get_account_summary = AsyncMock(
        return_value={"NetLiquidation_USD": net_liq_usd}
    )
    connector.disconnect = AsyncMock()
    return connector


# ── TestExtractNetLiq ──────────────────────────────────────────────────────


class TestExtractNetLiq:
    def test_usd_preferred_over_base(self) -> None:
        summary = {"NetLiquidation_USD": 100_000.0, "NetLiquidation_BASE": 130_000.0}
        assert extract_net_liq(summary) == 100_000.0

    def test_base_used_when_no_usd(self) -> None:
        summary = {"NetLiquidation_BASE": 130_000.0}
        assert extract_net_liq(summary) == 130_000.0

    def test_zero_usd_falls_through_to_base(self) -> None:
        summary = {"NetLiquidation_USD": 0.0, "NetLiquidation_BASE": 75_000.0}
        assert extract_net_liq(summary) == 75_000.0

    def test_empty_summary_returns_zero(self) -> None:
        assert extract_net_liq({}) == 0.0

    def test_non_numeric_value_ignored(self) -> None:
        summary = {"NetLiquidation_USD": "N/A", "NetLiquidation_BASE": 55_000.0}
        assert extract_net_liq(summary) == 55_000.0


# ── TestCacheBehavior ──────────────────────────────────────────────────────


class TestCacheBehavior:
    def test_cached_value_returned_within_ttl(self) -> None:
        connector = _mock_connector(25_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        v1 = feed.get()
        v2 = feed.get()
        assert v1 == v2
        # IBKR should only be called once while cache is fresh
        connector.connect.assert_awaited_once()

    def test_stale_cache_triggers_refresh(self) -> None:
        connector = _mock_connector(10_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=0)
        feed.get()
        feed.get()
        assert connector.connect.await_count == 2

    def test_invalidate_clears_cache(self) -> None:
        connector = _mock_connector(10_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        feed.get()
        feed.invalidate()
        feed.get()
        assert connector.connect.await_count == 2

    def test_get_source_returns_ibkr_after_live_fetch(self) -> None:
        connector = _mock_connector(10_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        feed.get()
        assert feed.get_source() == "ibkr"

    def test_get_source_returns_default_before_any_fetch(self) -> None:
        feed = AccountValueFeed(cache_ttl_seconds=300)
        assert feed.get_source() == "default"


# ── TestResolvePaths ───────────────────────────────────────────────────────


class TestResolvePaths:
    def test_ibkr_success_returns_ibkr_value(self) -> None:
        connector = _mock_connector(99_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        assert feed.get() == 99_000.0
        assert feed.get_source() == "ibkr"

    def test_ibkr_failure_falls_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "42000")
        connector = _mock_connector()
        connector.connect = AsyncMock(side_effect=ConnectionRefusedError("offline"))
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        assert feed.get() == 42_000.0
        assert feed.get_source() == "env"

    def test_ibkr_zero_net_liq_falls_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "30000")
        connector = _mock_connector(0.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        assert feed.get() == 30_000.0
        assert feed.get_source() == "env"

    def test_env_parse_error_falls_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "not-a-number")
        connector = _mock_connector(0.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        result = feed.get()
        assert result == DEFAULT_ACCOUNT_VALUE
        assert feed.get_source() == "default"

    def test_no_ibkr_no_env_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ACCOUNT_VALUE_USD", raising=False)
        connector = _mock_connector(0.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        assert feed.get() == DEFAULT_ACCOUNT_VALUE
        assert feed.get_source() == "default"

    def test_all_failures_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ACCOUNT_VALUE_USD", raising=False)
        connector = _mock_connector()
        connector.connect = AsyncMock(side_effect=RuntimeError("crash"))
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        assert feed.get() == DEFAULT_ACCOUNT_VALUE
        assert feed.get_source() == "default"

    def test_get_source_env_after_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "65000")
        connector = _mock_connector(0.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        feed.get()
        assert feed.get_source() == "env"


# ── TestFetchFromIbkr ──────────────────────────────────────────────────────


class TestFetchFromIbkr:
    def test_connect_called(self) -> None:
        connector = _mock_connector(10_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        feed.get()
        connector.connect.assert_awaited_once()

    def test_get_account_summary_called(self) -> None:
        connector = _mock_connector(10_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        feed.get()
        connector.get_account_summary.assert_awaited_once()

    def test_disconnect_called_on_success(self) -> None:
        connector = _mock_connector(10_000.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        feed.get()
        connector.disconnect.assert_awaited_once()

    def test_disconnect_called_even_when_summary_raises(self) -> None:
        connector = _mock_connector()
        connector.get_account_summary = AsyncMock(side_effect=RuntimeError("api_err"))
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        # Falls through to default — must not raise
        feed.get()
        connector.disconnect.assert_awaited_once()

    def test_injected_connector_value_returned(self) -> None:
        connector = _mock_connector(77_777.0)
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        assert feed.get() == 77_777.0

    def test_disconnect_error_does_not_propagate(self) -> None:
        connector = _mock_connector(20_000.0)
        connector.disconnect = AsyncMock(side_effect=OSError("already closed"))
        feed = AccountValueFeed(ibkr_connector=connector, cache_ttl_seconds=300)
        # Should return the value, not crash
        assert feed.get() == 20_000.0


# ── TestMarketSchedulerWiring ──────────────────────────────────────────────


class TestMarketSchedulerWiring:
    @patch("asyncio.run", return_value=[])
    @patch(_PATCH_POSITION_TRACKER)
    @patch(_PATCH_PNL_TRACKER)
    def test_feed_created_at_init(
        self, _MockPT: MagicMock, _MockPos: MagicMock, _mock_run: MagicMock
    ) -> None:
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler()
        from shared.account_value_feed import AccountValueFeed as RealFeed

        assert isinstance(sched._account_value_feed, RealFeed)

    @patch("asyncio.run", return_value=[])
    @patch(_PATCH_POSITION_TRACKER)
    @patch(_PATCH_PNL_TRACKER)
    def test_run_pnl_snapshot_calls_feed_get(
        self, MockPT: MagicMock, _MockPos: MagicMock, _mock_run: MagicMock
    ) -> None:
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler()
        mock_feed = MagicMock()
        mock_feed.get.return_value = 88_000.0
        sched._account_value_feed = mock_feed
        MockPT.return_value.take_snapshot.return_value = {}
        sched.run_pnl_snapshot()
        mock_feed.get.assert_called_once()

    @patch("asyncio.run", return_value=[])
    @patch(_PATCH_POSITION_TRACKER)
    @patch(_PATCH_PNL_TRACKER)
    def test_feed_value_passed_to_pnl_tracker(
        self, MockPT: MagicMock, _MockPos: MagicMock, _mock_run: MagicMock
    ) -> None:
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler()
        mock_feed = MagicMock()
        mock_feed.get.return_value = 123_456.0
        sched._account_value_feed = mock_feed
        snap_mock = MockPT.return_value
        snap_mock.take_snapshot.return_value = {}
        sched.run_pnl_snapshot()
        call_args = snap_mock.take_snapshot.call_args
        assert call_args[0][1] == 123_456.0

    @patch("asyncio.run", return_value=[])
    @patch(_PATCH_POSITION_TRACKER)
    @patch(_PATCH_PNL_TRACKER)
    def test_feed_exception_falls_back_to_env_default(
        self, MockPT: MagicMock, _MockPos: MagicMock, _mock_run: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If feed.get() raises (shouldn't in real code), snapshot still runs."""
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "35000")
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler()
        mock_feed = MagicMock()
        mock_feed.get.side_effect = RuntimeError("feed_dead")
        sched._account_value_feed = mock_feed
        snap_mock = MockPT.return_value
        snap_mock.take_snapshot.return_value = {}
        sched.run_pnl_snapshot()
        # Snapshot must still be called with the env fallback value
        call_args = snap_mock.take_snapshot.call_args
        assert call_args[0][1] == 35_000.0

    @patch("asyncio.run", return_value=[])
    @patch(_PATCH_POSITION_TRACKER)
    @patch(_PATCH_PNL_TRACKER)
    def test_paper_flag_forwarded_to_feed(
        self, _MockPT: MagicMock, _MockPos: MagicMock, _mock_run: MagicMock
    ) -> None:
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler(paper=True)
        assert sched._account_value_feed._paper is True

    @patch("asyncio.run", return_value=[])
    @patch(_PATCH_POSITION_TRACKER)
    @patch(_PATCH_PNL_TRACKER)
    def test_feed_value_passed_to_drawdown_update(
        self, MockPT: MagicMock, _MockPos: MagicMock, _mock_run: MagicMock
    ) -> None:
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler()
        mock_feed = MagicMock()
        mock_feed.get.return_value = 75_000.0
        mock_feed.get_source.return_value = "ibkr"
        sched._account_value_feed = mock_feed

        mock_cb = MagicMock()
        mock_cb.update.return_value = MagicMock(
            drawdown_pct=0.01, tripped=False,
            peak_value=75_000.0, current_value=75_000.0,
        )
        sched._drawdown_circuit_breaker = mock_cb
        MockPT.return_value.take_snapshot.return_value = {}
        sched.run_pnl_snapshot()

        mock_cb.update.assert_called_once_with(75_000.0)

    @patch("asyncio.run", return_value=[])
    @patch(_PATCH_POSITION_TRACKER)
    @patch(_PATCH_PNL_TRACKER)
    def test_default_paper_false(
        self, _MockPT: MagicMock, _MockPos: MagicMock, _mock_run: MagicMock
    ) -> None:
        from core.market_scheduler import MarketScheduler

        with patch.dict("os.environ", {"PAPER_TRADING": "false"}):
            sched = MarketScheduler()
        assert sched._account_value_feed._paper is False

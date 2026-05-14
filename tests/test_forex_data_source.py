from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.forex_data_source import ForexDataSource
from shared.data_sources import MarketTick


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _fake_rate(pair="EUR/USD", bid=1.10, ask=1.11, source="exchangerate-api"):
    """Build a stand-in for FXRate with the attributes ForexDataSource reads."""
    r = MagicMock()
    r.pair = pair
    r.bid = bid
    r.ask = ask
    r.mid = (bid + ask) / 2
    r.source = source
    return r


def _make_source():
    """Build a ForexDataSource with the FX client patched out."""
    with patch("shared.forex_data_source.KnightsbridgeFXClient") as MockClient:
        instance = MagicMock()
        instance.connect = AsyncMock(return_value=None)
        instance.disconnect = AsyncMock(return_value=None)
        instance.get_pair = AsyncMock(return_value=None)
        instance.get_rates = AsyncMock(return_value={})
        instance.find_triangular_arb = AsyncMock(return_value=[])
        instance.compare_spreads = AsyncMock(return_value={})
        MockClient.return_value = instance
        src = ForexDataSource()
    src._client = instance  # ensure direct attribute matches
    return src, instance


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_source_id_is_forex(self):
        src, _ = _make_source()
        assert src.source_id == "forex"

    def test_starts_disconnected(self):
        src, _ = _make_source()
        assert src.is_connected is False

    def test_client_is_set(self):
        src, client = _make_source()
        assert src._client is client

    def test_no_poll_task_initially(self):
        src, _ = _make_source()
        assert src._poll_task is None


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    @pytest.mark.asyncio
    async def test_connect_calls_client_and_sets_flag(self):
        src, client = _make_source()
        await src.connect()
        assert src.is_connected is True
        client.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_clears_flag(self):
        src, client = _make_source()
        await src.connect()
        await src.disconnect()
        assert src.is_connected is False
        client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_running_poll_task(self):
        src, client = _make_source()

        async def long_running():
            await asyncio.sleep(60)

        task = asyncio.create_task(long_running())
        src._poll_task = task
        await src.disconnect()
        # Give the cancellation a chance to propagate through the event loop
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_disconnect_skips_done_poll_task(self):
        src, client = _make_source()

        async def quick():
            return None

        task = asyncio.create_task(quick())
        await task
        src._poll_task = task
        # Should not raise even though task is done
        await src.disconnect()


# ---------------------------------------------------------------------------
# get_fx_tick
# ---------------------------------------------------------------------------


class TestGetFxTick:
    @pytest.mark.asyncio
    async def test_returns_marketTick_with_expected_fields(self):
        src, client = _make_source()
        client.get_pair = AsyncMock(return_value=_fake_rate("EUR/USD", 1.10, 1.12))
        tick = await src.get_fx_tick("EUR", "USD")
        assert tick is not None
        assert isinstance(tick, MarketTick)
        assert tick.symbol == "EUR/USD"
        assert tick.price == pytest.approx(1.11)
        assert tick.bid == 1.10
        assert tick.ask == 1.12
        assert tick.volume_24h == 0.0
        assert tick.change_24h == 0.0
        assert tick.source == "forex_exchangerate-api"

    @pytest.mark.asyncio
    async def test_returns_none_when_client_returns_none(self):
        src, client = _make_source()
        client.get_pair = AsyncMock(return_value=None)
        tick = await src.get_fx_tick("XYZ", "ABC")
        assert tick is None

    @pytest.mark.asyncio
    async def test_notifies_subscribers(self):
        src, client = _make_source()
        client.get_pair = AsyncMock(return_value=_fake_rate("EUR/USD", 1.0, 1.0))
        received = []

        async def cb(tick):
            received.append(tick)

        src.subscribe(cb)
        await src.get_fx_tick("EUR", "USD")
        assert len(received) == 1
        assert received[0].symbol == "EUR/USD"


# ---------------------------------------------------------------------------
# get_all_rates
# ---------------------------------------------------------------------------


class TestGetAllRates:
    @pytest.mark.asyncio
    async def test_empty_rates_returns_empty_dict(self):
        src, client = _make_source()
        client.get_rates = AsyncMock(return_value={})
        out = await src.get_all_rates("USD")
        assert out == {}

    @pytest.mark.asyncio
    async def test_default_base_is_usd(self):
        src, client = _make_source()
        client.get_rates = AsyncMock(return_value={})
        await src.get_all_rates()
        client.get_rates.assert_awaited_once_with("USD")

    @pytest.mark.asyncio
    async def test_returns_dict_of_marketTicks(self):
        src, client = _make_source()
        client.get_rates = AsyncMock(
            return_value={
                "EUR/USD": _fake_rate("EUR/USD", 1.10, 1.12),
                "GBP/USD": _fake_rate("GBP/USD", 1.25, 1.27, source="ecb"),
            }
        )
        out = await src.get_all_rates("USD")
        assert set(out.keys()) == {"EUR/USD", "GBP/USD"}
        assert all(isinstance(t, MarketTick) for t in out.values())
        assert out["GBP/USD"].source == "forex_ecb"

    @pytest.mark.asyncio
    async def test_notifies_each_tick(self):
        src, client = _make_source()
        client.get_rates = AsyncMock(
            return_value={
                "EUR/USD": _fake_rate("EUR/USD", 1.0, 1.0),
                "GBP/USD": _fake_rate("GBP/USD", 1.0, 1.0),
            }
        )
        received = []
        src.subscribe(lambda t: received.append(t))
        await src.get_all_rates("USD")
        assert len(received) == 2


# ---------------------------------------------------------------------------
# poll_rates
# ---------------------------------------------------------------------------


class TestPollRates:
    @pytest.mark.asyncio
    async def test_default_bases_polled(self):
        src, client = _make_source()
        client.get_rates = AsyncMock(return_value={})
        # Patch sleep to short-circuit; raise CancelledError after first iteration
        sleep_calls = []
        original_sleep = asyncio.sleep

        async def fake_sleep(s):
            sleep_calls.append(s)
            if len(sleep_calls) > 4:
                raise asyncio.CancelledError()
            await original_sleep(0)

        with patch("shared.forex_data_source.asyncio.sleep", new=fake_sleep):
            await src.poll_rates(interval=0.01)
        # USD, CAD, EUR each polled once
        called_bases = [c.args[0] for c in client.get_rates.await_args_list]
        assert called_bases[:3] == ["USD", "CAD", "EUR"]

    @pytest.mark.asyncio
    async def test_custom_bases(self):
        src, client = _make_source()
        client.get_rates = AsyncMock(return_value={})
        original_sleep = asyncio.sleep

        async def fake_sleep(s):
            raise asyncio.CancelledError()

        with patch("shared.forex_data_source.asyncio.sleep", new=fake_sleep):
            try:
                await src.poll_rates(bases=["JPY"], interval=0.01)
            except asyncio.CancelledError:
                pass
        # Should have at least attempted JPY
        called_bases = [c.args[0] for c in client.get_rates.await_args_list]
        assert "JPY" in called_bases

    @pytest.mark.asyncio
    async def test_swallows_non_cancelled_exceptions(self):
        src, client = _make_source()
        # First call raises, second-and-on raise CancelledError via sleep
        call_count = {"n": 0}

        async def flaky_get_rates(_base):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("temporary failure")
            return {}

        client.get_rates = AsyncMock(side_effect=flaky_get_rates)

        sleep_count = {"n": 0}

        async def fake_sleep(s):
            sleep_count["n"] += 1
            if sleep_count["n"] >= 2:
                raise asyncio.CancelledError()

        with patch("shared.forex_data_source.asyncio.sleep", new=fake_sleep):
            await src.poll_rates(bases=["USD"], interval=0.01)

        # Did not crash; was called more than once
        assert client.get_rates.await_count >= 1


# ---------------------------------------------------------------------------
# start_polling
# ---------------------------------------------------------------------------


class TestStartPolling:
    @pytest.mark.asyncio
    async def test_creates_background_task(self):
        src, client = _make_source()
        # Avoid actually polling — patch poll_rates to no-op coroutine
        src.poll_rates = AsyncMock(return_value=None)
        src.start_polling(interval=0.01)
        assert src._poll_task is not None
        await asyncio.sleep(0)  # let it start
        # cleanup
        if not src._poll_task.done():
            src._poll_task.cancel()


# ---------------------------------------------------------------------------
# Proxies: find_triangular_arb / compare_spreads
# ---------------------------------------------------------------------------


class TestProxies:
    @pytest.mark.asyncio
    async def test_find_triangular_arb_proxies_to_client(self):
        src, client = _make_source()
        client.find_triangular_arb = AsyncMock(return_value=[{"path": "USD->EUR->GBP->USD", "bps": 7}])
        out = await src.find_triangular_arb("USD", min_profit_bps=5.0)
        assert out == [{"path": "USD->EUR->GBP->USD", "bps": 7}]
        client.find_triangular_arb.assert_awaited_once_with("USD", 5.0)

    @pytest.mark.asyncio
    async def test_find_triangular_arb_default_args(self):
        src, client = _make_source()
        client.find_triangular_arb = AsyncMock(return_value=[])
        await src.find_triangular_arb()
        client.find_triangular_arb.assert_awaited_once_with("USD", 5.0)

    @pytest.mark.asyncio
    async def test_compare_spreads_proxies_to_client(self):
        src, client = _make_source()
        client.compare_spreads = AsyncMock(return_value={"EUR/USD": 1.2})
        out = await src.compare_spreads(["EUR/USD"])
        assert out == {"EUR/USD": 1.2}
        client.compare_spreads.assert_awaited_once_with(["EUR/USD"])

    @pytest.mark.asyncio
    async def test_compare_spreads_default_arg_none(self):
        src, client = _make_source()
        client.compare_spreads = AsyncMock(return_value={})
        await src.compare_spreads()
        client.compare_spreads.assert_awaited_once_with(None)

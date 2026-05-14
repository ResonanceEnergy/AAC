from __future__ import annotations

"""tests/test_alerter_sprint21.py — Sprint 21: Real-Time Alerter.

Test classes
------------
TestFormatAlert(3)          — message formatting helpers
TestAlerterDisabled(5)      — unconfigured alerter always fails-open
TestAlerterEnabled(7)       — configured alerter: HTTP success + failure paths
TestAlerterAiohttpError(3)  — aiohttp exception handling
TestAutoTraderWiring(6)     — alerter receives events when CB/loss guard trips
TestEodReporterWiring(4)    — alerter sends daily brief after write_to_file
TestMarketSchedulerWiring(5)— shared Alerter created + passed to AutoTrader
"""

import asyncio
import os
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_alerter(bot_token: str = "TOK", chat_id: str = "123"):
    """Return a configured Alerter with no real HTTP credentials needed."""
    from shared.alerter import Alerter
    return Alerter(bot_token=bot_token, chat_id=chat_id)


def _unconfigured_alerter():
    from shared.alerter import Alerter
    return Alerter(bot_token="", chat_id="")


def _mock_aiohttp_success():
    """Return a mock aiohttp module that simulates a 200 POST."""
    resp = AsyncMock()
    resp.status = 200
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.post = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    connector = MagicMock()
    resolver = MagicMock()

    resolver_mod = types.ModuleType("aiohttp.resolver")
    resolver_mod.ThreadedResolver = MagicMock(return_value=resolver)

    mod = types.ModuleType("aiohttp")
    mod.ClientSession = MagicMock(return_value=session)
    mod.TCPConnector = MagicMock(return_value=connector)
    mod.ClientTimeout = MagicMock(return_value=MagicMock())
    mod.resolver = resolver_mod  # make aiohttp.resolver attribute access work

    return mod, resolver_mod


def _mock_aiohttp_failure(status: int = 400, body: str = "bad request"):
    """Return a mock aiohttp module that simulates an HTTP error response."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.post = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    connector = MagicMock()
    resolver = MagicMock()

    resolver_mod = types.ModuleType("aiohttp.resolver")
    resolver_mod.ThreadedResolver = MagicMock(return_value=resolver)

    mod = types.ModuleType("aiohttp")
    mod.ClientSession = MagicMock(return_value=session)
    mod.TCPConnector = MagicMock(return_value=connector)
    mod.ClientTimeout = MagicMock(return_value=MagicMock())
    mod.resolver = resolver_mod  # make aiohttp.resolver attribute access work

    return mod, resolver_mod


# ── TestFormatAlert ────────────────────────────────────────────────────────────

class TestFormatAlert:
    def test_known_event_has_icon(self):
        from shared.alerter import format_alert
        msg = format_alert("DRAWDOWN_TRIPPED", "10% breach")
        assert "🔴" in msg
        assert "DRAWDOWN_TRIPPED" in msg
        assert "10% breach" in msg

    def test_unknown_event_has_default_icon(self):
        from shared.alerter import format_alert
        msg = format_alert("MYSTERY_EVENT", "something happened")
        assert "📢" in msg
        assert "MYSTERY_EVENT" in msg

    def test_eod_brief_icon(self):
        from shared.alerter import format_alert
        msg = format_alert("EOD_BRIEF", "P&L: +$500")
        assert "📊" in msg


# ── TestAlerterDisabled ────────────────────────────────────────────────────────

class TestAlerterDisabled:
    def test_empty_credentials_not_enabled(self):
        a = _unconfigured_alerter()
        assert a.enabled is False

    def test_send_returns_false_when_disabled(self):
        a = _unconfigured_alerter()
        result = a.send("TEST", "hello")
        assert result is False

    def test_send_does_not_raise(self):
        a = _unconfigured_alerter()
        # Must never raise regardless of inputs
        a.send("", "")
        a.send("DRAWDOWN_TRIPPED", "x" * 1000)

    def test_missing_token_disables(self):
        from shared.alerter import Alerter
        a = Alerter(bot_token="", chat_id="12345")
        assert a.enabled is False

    def test_missing_chat_id_disables(self):
        from shared.alerter import Alerter
        a = Alerter(bot_token="MYTOKEN", chat_id="")
        assert a.enabled is False


# ── TestAlerterEnabled ─────────────────────────────────────────────────────────

class TestAlerterEnabled:
    def test_enabled_when_both_configured(self):
        a = _make_alerter()
        assert a.enabled is True

    def test_send_returns_true_on_http_200(self):
        a = _make_alerter()
        aio_mod, aio_resolver_mod = _mock_aiohttp_success()
        with patch.dict("sys.modules", {"aiohttp": aio_mod, "aiohttp.resolver": aio_resolver_mod}):
            result = a.send("TEST", "hello")
        assert result is True

    def test_send_returns_false_on_http_400(self):
        a = _make_alerter()
        aio_mod, aio_resolver_mod = _mock_aiohttp_failure(400)
        with patch.dict("sys.modules", {"aiohttp": aio_mod, "aiohttp.resolver": aio_resolver_mod}):
            result = a.send("TEST", "hello")
        assert result is False

    def test_send_correct_url_contains_token(self):
        """ClientSession.post must be called with the token in the URL."""
        a = _make_alerter(bot_token="MYTOKEN123")
        aio_mod, aio_resolver_mod = _mock_aiohttp_success()
        with patch.dict("sys.modules", {"aiohttp": aio_mod, "aiohttp.resolver": aio_resolver_mod}):
            a.send("TEST", "hello")
        call_args = aio_mod.ClientSession.return_value.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", call_args[0][0])
        # url may be first positional arg
        posted_url = call_args[0][0] if call_args.args else str(call_args)
        assert "MYTOKEN123" in str(call_args)

    def test_send_payload_has_chat_id(self):
        """POST payload must include the configured chat_id."""
        a = _make_alerter(chat_id="987654")
        aio_mod, aio_resolver_mod = _mock_aiohttp_success()
        with patch.dict("sys.modules", {"aiohttp": aio_mod, "aiohttp.resolver": aio_resolver_mod}):
            a.send("TEST", "hello")
        # Inspect the json= kwarg passed to session.post
        call_kwargs = aio_mod.ClientSession.return_value.post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("chat_id") == "987654"

    def test_env_var_fallback(self, monkeypatch):
        """Alerter reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from env."""
        from shared.alerter import Alerter
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "ENV_TOKEN")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "ENV_CHAT")
        a = Alerter()
        assert a.enabled is True
        assert a._bot_token == "ENV_TOKEN"
        assert a._chat_id == "ENV_CHAT"

    def test_explicit_args_override_env(self, monkeypatch):
        """Constructor args take precedence over env vars."""
        from shared.alerter import Alerter
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "ENV_TOKEN")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "ENV_CHAT")
        a = Alerter(bot_token="OVERRIDE", chat_id="999")
        assert a._bot_token == "OVERRIDE"
        assert a._chat_id == "999"


# ── TestAlerterAiohttpError ────────────────────────────────────────────────────

class TestAlerterAiohttpError:
    def test_aiohttp_import_error_returns_false(self):
        a = _make_alerter()
        # Remove aiohttp from sys.modules so the import inside _send_async fails
        with patch.dict("sys.modules", {"aiohttp": None, "aiohttp.resolver": None}):
            result = a.send("TEST", "msg")
        assert result is False

    def test_network_exception_returns_false(self):
        a = _make_alerter()
        aio_mod, aio_resolver_mod = _mock_aiohttp_success()
        # Make the session.post raise a network error
        aio_mod.ClientSession.return_value.post = MagicMock(side_effect=OSError("timeout"))
        with patch.dict("sys.modules", {"aiohttp": aio_mod, "aiohttp.resolver": aio_resolver_mod}):
            result = a.send("TEST", "msg")
        assert result is False

    def test_send_never_raises_on_any_error(self):
        a = _make_alerter()
        # asyncio.run itself crashes — must still return False without raising
        with patch("asyncio.run", side_effect=RuntimeError("event loop broken")):
            result = a.send("TEST", "msg")
        assert result is False


# ── TestAutoTraderWiring ───────────────────────────────────────────────────────

class TestAutoTraderWiring:
    def _make_tripped_cb(self):
        cb = MagicMock()
        cb.is_tripped.return_value = True
        cb.max_drawdown_pct = 0.10
        return cb

    def _make_tripped_loss_guard(self):
        dlg = MagicMock()
        dlg.is_limit_reached.return_value = (True, "daily loss -2.5%")
        return dlg

    def test_alerter_param_accepted(self):
        from core.auto_trader import AutoTrader
        a = MagicMock()
        trader = AutoTrader(alerter=a)
        assert trader._alerter is a

    def test_alerter_send_called_on_drawdown_trip(self):
        from core.auto_trader import AutoTrader
        alerter = MagicMock()
        trader = AutoTrader(
            drawdown_circuit_breaker=self._make_tripped_cb(),
            alerter=alerter,
        )
        trader.run_once([])
        alerter.send.assert_called_once()
        call_event = alerter.send.call_args[0][0]
        assert call_event == "DRAWDOWN_TRIPPED"

    def test_alerter_send_called_on_daily_loss_trip(self):
        from core.auto_trader import AutoTrader
        alerter = MagicMock()
        trader = AutoTrader(
            daily_loss_guard=self._make_tripped_loss_guard(),
            alerter=alerter,
        )
        trader.run_once([])
        alerter.send.assert_called_once()
        call_event = alerter.send.call_args[0][0]
        assert call_event == "DAILY_LOSS_TRIPPED"

    def test_no_alert_when_cb_not_tripped(self):
        from core.auto_trader import AutoTrader
        cb = MagicMock()
        cb.is_tripped.return_value = False
        alerter = MagicMock()
        trader = AutoTrader(drawdown_circuit_breaker=cb, alerter=alerter)
        trader.run_once([])
        alerter.send.assert_not_called()

    def test_none_alerter_does_not_crash_on_cb_trip(self):
        from core.auto_trader import AutoTrader
        trader = AutoTrader(
            drawdown_circuit_breaker=self._make_tripped_cb(),
            alerter=None,
        )
        summary = trader.run_once([])
        assert summary.signals_filtered == 0  # no signals received, so filtered=0

    def test_alerter_failure_does_not_block_summary_return(self):
        """Even if alerter.send() raises, AutoTrader still returns a summary."""
        from core.auto_trader import AutoTrader
        alerter = MagicMock()
        alerter.send.side_effect = RuntimeError("Telegram down")
        trader = AutoTrader(
            drawdown_circuit_breaker=self._make_tripped_cb(),
            alerter=alerter,
        )
        summary = trader.run_once([MagicMock()])
        # Should not raise; should return a summary
        assert summary is not None
        assert summary.signals_filtered >= 0


# ── TestEodReporterWiring ──────────────────────────────────────────────────────

class TestEodReporterWiring:
    def test_alerter_param_accepted(self):
        from core.eod_reporter import EodReporter
        a = MagicMock()
        reporter = EodReporter(alerter=a)
        assert reporter._alerter is a

    def test_alerter_send_called_after_generate(self, tmp_path):
        from core.eod_reporter import EodReporter
        alerter = MagicMock()
        reporter = EodReporter(alerter=alerter)
        report = reporter.generate(report_path=str(tmp_path / "brief.txt"))
        alerter.send.assert_called_once()
        event_type = alerter.send.call_args[0][0]
        assert event_type == "EOD_BRIEF"

    def test_alerter_message_contains_date(self, tmp_path):
        from core.eod_reporter import EodReporter
        alerter = MagicMock()
        reporter = EodReporter(alerter=alerter)
        reporter.generate(report_path=str(tmp_path / "brief.txt"))
        message = alerter.send.call_args[0][1]
        assert "Date:" in message

    def test_alerter_failure_does_not_crash_generate(self, tmp_path):
        """If alerter.send() raises, generate() still returns an EodReport."""
        from core.eod_reporter import EodReporter
        alerter = MagicMock()
        alerter.send.side_effect = RuntimeError("Telegram down")
        reporter = EodReporter(alerter=alerter)
        report = reporter.generate(report_path=str(tmp_path / "brief.txt"))
        assert report is not None
        assert report.report_date != ""


# ── TestMarketSchedulerWiring ──────────────────────────────────────────────────

class TestMarketSchedulerWiring:
    def _make_scheduler(self, **kwargs):
        from core.market_scheduler import MarketScheduler
        return MarketScheduler(**kwargs)

    def test_alerter_created_at_init(self):
        from shared.alerter import Alerter
        sched = self._make_scheduler()
        assert hasattr(sched, "_alerter")
        assert isinstance(sched._alerter, Alerter)

    def test_alerter_passed_to_auto_trader_when_auto_execute(self):
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(auto_execute=True)
        assert sched._auto_trader is not None
        assert sched._auto_trader._alerter is sched._alerter

    def test_no_auto_trader_when_auto_execute_false(self):
        sched = self._make_scheduler(auto_execute=False)
        assert sched._auto_trader is None

    def test_eod_report_uses_scheduler_alerter(self, tmp_path):
        """_run_eod_report creates EodReporter with the shared alerter."""
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler(auto_execute=False)
        alerter = MagicMock()
        sched._alerter = alerter  # inject test alerter

        # Patch PnLTracker so no real DB access
        sched._pnl_tracker = MagicMock()
        sched._pnl_tracker.pnl_delta.return_value = 0.0
        sched._pnl_tracker.today_report.return_value = {}

        # Patch HealthMonitor so no real HTTP calls
        with patch("monitoring.health_monitor.HealthMonitor") as mock_hm_cls:
            mock_hm_cls.return_value.collect_snapshot.return_value = None
            with patch("core.eod_reporter.EodReport.write_to_file"):
                sched._run_eod_report({}, [])

        # alerter.send should have been called (EOD_BRIEF)
        alerter.send.assert_called_once()
        assert alerter.send.call_args[0][0] == "EOD_BRIEF"

    def test_alerter_send_failure_in_eod_does_not_crash_scheduler(self, tmp_path):
        """If Telegram send fails during EOD, scheduler never crashes."""
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler(auto_execute=False)
        alerter = MagicMock()
        alerter.send.side_effect = RuntimeError("network down")
        sched._alerter = alerter

        sched._pnl_tracker = MagicMock()
        sched._pnl_tracker.pnl_delta.return_value = 0.0
        sched._pnl_tracker.today_report.return_value = {}

        with patch("core.eod_reporter.EodReport.write_to_file"):
            # Should not raise
            sched._run_eod_report({}, [])

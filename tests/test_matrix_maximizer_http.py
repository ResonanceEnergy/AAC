"""
Tests for MATRIX MAXIMIZER — HTTP Health Check & Wiring
=========================================================
Covers:
  - http_health.py (EndpointResult, HTTPHealthCheck, mocked endpoints)
  - data_feeds.py _http_get User-Agent override
  - Intelligence ← DataFeedManager wiring via runner
  - Intelligence _enrich_from_feeds integration
"""

import json
import ssl
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _mock_urlopen_ok(data):
    """Helper: create mock urlopen context manager that returns 200 + JSON."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.status = 200
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


class TestEndpointResult:
    def test_basic_fields(self):
        from strategies.matrix_maximizer.http_health import EndpointResult
        r = EndpointResult(name="Test", url="https://example.com", status="OK")
        assert r.name == "Test"
        assert r.status == "OK"
        assert r.latency_ms == 0
        assert r.detail == ""

    def test_full_fields(self):
        from strategies.matrix_maximizer.http_health import EndpointResult
        r = EndpointResult(
            name="Polygon Snapshot", url="https://api.polygon.io/v2/snapshot",
            status="FAIL", latency_ms=340.5, http_code=403,
            detail="Subscription required", data_preview="N/A",
        )
        assert r.status == "FAIL"
        assert r.http_code == 403

    def test_status_values(self):
        from strategies.matrix_maximizer.http_health import EndpointResult
        for status in ("OK", "FAIL", "SKIP", "WARN"):
            r = EndpointResult(name="t", url="u", status=status)
            assert r.status == status


# ═══════════════════════════════════════════════════════════════════════════
# HTTP HEALTH CHECK — INIT & KEY INVENTORY
# ═══════════════════════════════════════════════════════════════════════════


class TestHTTPHealthCheckInit:
    @patch.dict("os.environ", {
        "POLYGON_API_KEY": "test_poly_key",
        "FRED_API_KEY": "test_fred_key",
        "FINNHUB_API_KEY": "test_finn_key",
        "UNUSUAL_WHALES_API_KEY": "test_uw_key",
        "TELEGRAM_BOT_TOKEN": "test_tg_token",
        "TELEGRAM_CHAT_ID": "123456",
        "SMTP_HOST": "smtp.test.com",
        "SMTP_USER": "user@test.com",
        "SMTP_PASSWORD": "secret",
    }, clear=False)
    def test_init_reads_env(self):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        hc = HTTPHealthCheck()
        assert hc._polygon == "test_poly_key"
        assert hc._fred == "test_fred_key"
        assert hc._finnhub == "test_finn_key"
        assert hc._uw == "test_uw_key"
        assert hc._telegram_token == "test_tg_token"

    @patch.dict("os.environ", {"POLYGON_API_KEY": "pk_12345678"}, clear=False)
    def test_check_keys_shows_set(self):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        hc = HTTPHealthCheck()
        keys = hc.check_keys()
        assert "POLYGON_API_KEY" in keys
        assert keys["POLYGON_API_KEY"].startswith("SET")

    def test_verbose_flag(self):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        hc = HTTPHealthCheck(verbose=True)
        assert hc.verbose is True
        hc2 = HTTPHealthCheck(verbose=False)
        assert hc2.verbose is False


# ═══════════════════════════════════════════════════════════════════════════
# HTTP HEALTH CHECK — _http_get RETURN VALUE & HEADERS
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthCheckHttpGet:
    @patch("urllib.request.urlopen")
    def test_success_returns_triple(self, mock_urlopen):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        mock_urlopen.return_value = _mock_urlopen_ok({"status": "ok"})
        hc = HTTPHealthCheck()
        code, data, latency = hc._http_get("https://api.example.com/test")
        assert code == 200
        assert data == {"status": "ok"}
        assert latency >= 0

    @patch("urllib.request.urlopen")
    def test_http_error_returns_error_code(self, mock_urlopen):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.example.com", 403, "Forbidden", {}, BytesIO(b"")
        )
        hc = HTTPHealthCheck()
        code, data, latency = hc._http_get("https://api.example.com/test")
        assert code == 403
        assert data is None

    @patch("urllib.request.urlopen")
    def test_url_error_returns_zero_code(self, mock_urlopen):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        hc = HTTPHealthCheck()
        code, data, latency = hc._http_get("https://api.example.com/test")
        assert code == 0

    @patch("urllib.request.urlopen")
    def test_default_user_agent(self, mock_urlopen):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        mock_urlopen.return_value = _mock_urlopen_ok({"ok": True})
        hc = HTTPHealthCheck()
        hc._http_get("https://api.example.com/test")
        req = mock_urlopen.call_args[0][0]
        assert "AAC-MatrixMaximizer" in req.get_header("User-agent")

    @patch("urllib.request.urlopen")
    def test_custom_user_agent_overrides_default(self, mock_urlopen):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        mock_urlopen.return_value = _mock_urlopen_ok({"ok": True})
        hc = HTTPHealthCheck()
        hc._http_get("https://api.example.com/test", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0)",
        })
        req = mock_urlopen.call_args[0][0]
        assert "Mozilla" in req.get_header("User-agent")
        assert "AAC-Matrix" not in req.get_header("User-agent")

    @patch("urllib.request.urlopen")
    def test_custom_headers_applied(self, mock_urlopen):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        mock_urlopen.return_value = _mock_urlopen_ok({"ok": True})
        hc = HTTPHealthCheck()
        hc._http_get("https://x.com", headers={"Referer": "https://cnn.com"})
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Referer") == "https://cnn.com"


# ═══════════════════════════════════════════════════════════════════════════
# HTTP HEALTH CHECK — INDIVIDUAL ENDPOINT TESTS (MOCKED)
# ═══════════════════════════════════════════════════════════════════════════


def _make_hc(**env_overrides):
    """Create HTTPHealthCheck with controlled env vars."""
    env = {
        "POLYGON_API_KEY": "test_key",
        "FRED_API_KEY": "test_key",
        "FINNHUB_API_KEY": "test_key",
        "UNUSUAL_WHALES_API_KEY": "test_key",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
        "SMTP_HOST": "",
        "SMTP_USER": "",
        "SMTP_PASSWORD": "",
    }
    env.update(env_overrides)
    with patch.dict("os.environ", env, clear=False):
        from strategies.matrix_maximizer.http_health import HTTPHealthCheck
        return HTTPHealthCheck()


class TestHealthCheckEndpoints:
    @patch("urllib.request.urlopen")
    def test_polygon_prev_close_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({"results": [{"c": 661.43}]})
        hc = _make_hc()
        result = hc.test_polygon_prev_close()
        assert result.status == "OK"
        assert "661.43" in result.detail

    @patch("urllib.request.urlopen")
    def test_polygon_snapshot_403(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.polygon.io", 403, "Forbidden", {}, BytesIO(b"")
        )
        hc = _make_hc()
        result = hc.test_polygon_snapshot()
        assert result.status == "FAIL"
        assert result.http_code == 403

    @patch("urllib.request.urlopen")
    def test_polygon_options_chain_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({
            "results": [{"ticker": "O:SPY260630P00400000"}] * 5,
        })
        hc = _make_hc()
        result = hc.test_polygon_options_chain()
        assert result.status == "OK"
        assert "5 contracts" in result.detail

    @patch("urllib.request.urlopen")
    def test_fred_vix_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({
            "observations": [{"value": "22.37"}]
        })
        hc = _make_hc()
        result = hc.test_fred_vix()
        assert result.status == "OK"
        assert "22.37" in result.detail

    @patch("urllib.request.urlopen")
    def test_fred_oil_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({
            "observations": [{"value": "93.39"}]
        })
        hc = _make_hc()
        result = hc.test_fred_oil()
        assert result.status == "OK"

    @patch("urllib.request.urlopen")
    def test_finnhub_earnings_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({
            "earningsCalendar": [{"symbol": "AAPL"}] * 78
        })
        hc = _make_hc()
        result = hc.test_finnhub_earnings()
        assert result.status == "OK"

    @patch("urllib.request.urlopen")
    def test_finnhub_news_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok([
            {"headline": "SPY falls"},
            {"headline": "Market turbulence"},
        ])
        hc = _make_hc()
        result = hc.test_finnhub_news()
        assert result.status == "OK"
        assert "2 articles" in result.detail

    @patch("urllib.request.urlopen")
    def test_finnhub_insider_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({
            "data": [{"name": "Tim Cook"}] * 121
        })
        hc = _make_hc()
        result = hc.test_finnhub_insider()
        assert result.status == "OK"

    @patch("urllib.request.urlopen")
    def test_cnn_fear_greed_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({
            "fear_and_greed": {"score": 18.35}
        })
        hc = _make_hc()
        result = hc.test_fear_greed()
        assert result.status == "OK"
        assert "18" in result.detail

    @patch("urllib.request.urlopen")
    def test_cnn_fear_greed_sends_browser_headers(self, mock_urlopen):
        """CNN test must send Referer + browser User-Agent."""
        mock_urlopen.return_value = _mock_urlopen_ok({
            "fear_and_greed": {"score": 25}
        })
        hc = _make_hc()
        hc.test_fear_greed()
        req = mock_urlopen.call_args[0][0]
        assert "cnn.com" in req.get_header("Referer")
        assert "Mozilla" in req.get_header("User-agent")

    @patch("urllib.request.urlopen")
    def test_uw_404_handled(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.unusualwhales.com", 404, "Not Found", {}, BytesIO(b"")
        )
        hc = _make_hc()
        result = hc.test_uw_flow()
        assert result.status == "FAIL"
        assert result.http_code == 404

    @patch("urllib.request.urlopen")
    def test_uw_403_handled(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.unusualwhales.com", 403, "Forbidden", {}, BytesIO(b"")
        )
        hc = _make_hc()
        result = hc.test_uw_dark_pool()
        assert result.status == "FAIL"
        assert "403" in str(result.http_code)

    def test_telegram_skip_without_token(self):
        hc = _make_hc(TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID="")
        result = hc.test_telegram()
        assert result.status == "SKIP"
        assert "TELEGRAM_BOT_TOKEN" in result.detail

    @patch("urllib.request.urlopen")
    def test_telegram_ok(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_ok({
            "ok": True, "result": {"username": "testbot"}
        })
        hc = _make_hc(TELEGRAM_BOT_TOKEN="123:ABC")
        result = hc.test_telegram()
        assert result.status == "OK"
        assert "testbot" in result.detail

    def test_smtp_skip_without_config(self):
        hc = _make_hc(SMTP_HOST="", SMTP_USER="", SMTP_PASSWORD="")
        result = hc.test_smtp()
        assert result.status == "SKIP"

    @patch("smtplib.SMTP")
    def test_smtp_ok(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_server
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        hc = _make_hc(
            SMTP_HOST="smtp.test.com", SMTP_USER="user@test.com",
            SMTP_PASSWORD="pass123",
        )
        result = hc.test_smtp()
        assert result.status == "OK"
        assert "login OK" in result.detail

    @patch("smtplib.SMTP")
    def test_smtp_auth_fail(self, mock_smtp_cls):
        import smtplib as _smtplib
        mock_server = MagicMock()
        mock_server.login.side_effect = _smtplib.SMTPAuthenticationError(535, b"bad")
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_server
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        hc = _make_hc(
            SMTP_HOST="smtp.test.com", SMTP_USER="user@test.com",
            SMTP_PASSWORD="bad",
        )
        result = hc.test_smtp()
        assert result.status == "FAIL"
        assert "auth failed" in result.detail.lower()


# ═══════════════════════════════════════════════════════════════════════════
# HTTP HEALTH CHECK — run_all & REPORT
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthCheckRunAll:
    @patch("urllib.request.urlopen")
    def test_run_all_returns_16_results(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        hc = _make_hc()
        results = hc.run_all()
        assert len(results) == 16

    @patch("urllib.request.urlopen")
    def test_all_no_network_are_fail_or_skip(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        hc = _make_hc()
        results = hc.run_all()
        for r in results:
            assert r.status in ("FAIL", "SKIP")

    @patch("urllib.request.urlopen")
    def test_report_contains_sections(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        hc = _make_hc()
        hc.run_all()
        report = hc.print_report()
        assert "MATRIX MAXIMIZER" in report
        assert "API KEYS" in report
        assert "ENDPOINTS" in report

    @patch("urllib.request.urlopen")
    def test_save_report_creates_file(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("no net")
        hc = _make_hc()
        hc.run_all()
        report = hc.print_report()
        path = hc.save_report(report)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "MATRIX MAXIMIZER" in content


# ═══════════════════════════════════════════════════════════════════════════
# DATA FEEDS — _http_get USER-AGENT OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════


class TestDataFeedsHttpGet:
    @patch("urllib.request.urlopen")
    def test_default_user_agent(self, mock_urlopen):
        """Without headers, default UA is AAC-MatrixMaximizer."""
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        mgr = DataFeedManager()
        mgr._http_get("https://api.example.com/test")
        req = mock_urlopen.call_args[0][0]
        assert "AAC-MatrixMaximizer" in req.get_header("User-agent")

    @patch("urllib.request.urlopen")
    def test_custom_ua_overrides_default(self, mock_urlopen):
        """Custom User-Agent should NOT have AAC-MatrixMaximizer."""
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"score": 25}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        mgr = DataFeedManager()
        mgr._http_get("https://cnn.io/test", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0)",
        })
        req = mock_urlopen.call_args[0][0]
        ua = req.get_header("User-agent")
        assert "Mozilla" in ua
        assert "AAC-Matrix" not in ua

    @patch("urllib.request.urlopen")
    def test_referer_header_applied(self, mock_urlopen):
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": 1}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        mgr = DataFeedManager()
        mgr._http_get("https://cnn.io/test", headers={
            "Referer": "https://www.cnn.com/markets/fear-and-greed",
        })
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Referer") == "https://www.cnn.com/markets/fear-and-greed"

    @patch("urllib.request.urlopen")
    def test_http_get_returns_none_on_failure(self, mock_urlopen):
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        mgr = DataFeedManager()
        result = mgr._http_get("https://api.example.com/test")
        assert result is None

    @patch("urllib.request.urlopen")
    def test_fear_greed_sends_browser_headers(self, mock_urlopen):
        """_fetch_fear_greed must send Referer + browser UA."""
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "fear_and_greed": {"score": 18.35}
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        mgr = DataFeedManager()
        result = mgr._fetch_fear_greed()
        req = mock_urlopen.call_args[0][0]
        assert "cnn.com" in req.get_header("Referer")
        assert "Mozilla" in req.get_header("User-agent")
        assert result == pytest.approx(18.35, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER — INTELLIGENCE ← DATAFEED WIRING
# ═══════════════════════════════════════════════════════════════════════════


class TestRunnerIntelligenceWiring:
    @patch("urllib.request.urlopen")
    def test_intelligence_receives_data_feeds(self, mock_urlopen):
        """IntelligenceEngine must receive DataFeedManager reference."""
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer()
        assert mm.data_feeds is not None
        assert mm.intelligence is not None
        assert mm.intelligence._feeds is mm.data_feeds

    @patch("urllib.request.urlopen")
    def test_init_intelligence_static_method(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        feeds = DataFeedManager()
        intel = MatrixMaximizer._init_intelligence(feeds)
        assert intel is not None
        assert intel._feeds is feeds

    def test_init_intelligence_with_none(self):
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        intel = MatrixMaximizer._init_intelligence(None)
        assert intel is not None
        assert intel._feeds is None

    @patch("urllib.request.urlopen")
    def test_chatbot_gets_intelligence_reference(self, mock_urlopen):
        """Chatbot context should include the wired intelligence engine."""
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer()
        bot = mm.get_chatbot()
        if bot is not None:
            assert bot.ctx.intelligence is mm.intelligence

    @patch("urllib.request.urlopen")
    def test_full_cycle_with_intel_wired(self, mock_urlopen):
        """run_full_cycle should work with intelligence properly wired."""
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer(MatrixConfig(account_size=920, n_simulations=100))
        result = mm.run_full_cycle(prices={"oil": 95, "vix": 28})
        assert result["status"] in ("complete", "blocked")


# ═══════════════════════════════════════════════════════════════════════════
# INTELLIGENCE — _enrich_from_feeds INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


class TestIntelligenceEnrichFromFeeds:
    def test_enrich_with_no_feeds(self):
        """Without feeds (None), gather_intel still returns valid brief."""
        from strategies.matrix_maximizer.intelligence import IntelligenceEngine
        engine = IntelligenceEngine(data_feed_manager=None)
        brief = engine.gather_intel(["SPY", "XLE"])
        assert brief is not None
        assert brief.dark_pool_activity == {}
        assert brief.congress_buys == []
        assert brief.earnings_blackout == []

    def test_enrich_with_mock_feeds(self):
        """With mock feeds, intelligence enriches brief with flow/dark pool/congress."""
        from strategies.matrix_maximizer.intelligence import IntelligenceEngine

        mock_feeds = MagicMock()

        flow_item = MagicMock()
        flow_item.ticker = "SPY"
        flow_item.sentiment = "bearish"
        flow_item.premium = 500_000
        flow_item.option_type = "PUT"
        flow_item.strike = 400
        flow_item.expiry = "2026-06-30"
        mock_feeds.get_unusual_flow.return_value = [flow_item]

        dp_item = MagicMock()
        dp_item.notional = 5_000_000
        mock_feeds.get_dark_pool.return_value = [dp_item]

        mock_feeds.get_congress_trades.return_value = [
            {"ticker": "SPY", "type": "Sale (Partial)"},
            {"ticker": "XLE", "type": "Purchase"},
        ]
        mock_feeds.get_tickers_near_earnings.return_value = ["SPY"]

        engine = IntelligenceEngine(data_feed_manager=mock_feeds)
        brief = engine.gather_intel(["SPY", "XLE"])

        flow_sigs = [s for s in brief.signals if s.source == "unusual_whales"]
        assert len(flow_sigs) >= 1
        assert flow_sigs[0].ticker == "SPY"
        assert flow_sigs[0].direction == "bearish"

        assert "SPY" in brief.dark_pool_activity
        assert brief.dark_pool_activity["SPY"] == 5_000_000

        assert "SPY" in brief.congress_sells
        assert "XLE" in brief.congress_buys
        assert "SPY" in brief.earnings_blackout

    def test_enrich_handles_feed_errors(self):
        """If a feed method raises, enrichment continues without crashing."""
        from strategies.matrix_maximizer.intelligence import IntelligenceEngine

        mock_feeds = MagicMock()
        mock_feeds.get_unusual_flow.side_effect = ConnectionError("API down")
        mock_feeds.get_dark_pool.side_effect = TimeoutError("timeout")
        mock_feeds.get_congress_trades.side_effect = ValueError("bad data")
        mock_feeds.get_tickers_near_earnings.side_effect = RuntimeError("fail")

        engine = IntelligenceEngine(data_feed_manager=mock_feeds)
        brief = engine.gather_intel(["SPY"])
        assert brief is not None
        assert len(brief.dark_pool_activity) == 0

    def test_enrich_flow_strength_scales_with_premium(self):
        """Signal strength should scale with premium size."""
        from strategies.matrix_maximizer.intelligence import IntelligenceEngine

        mock_feeds = MagicMock()
        small_flow = MagicMock(
            ticker="SPY", sentiment="bearish", premium=100_000,
            option_type="PUT", strike=400, expiry="2026-06-30",
        )
        big_flow = MagicMock(
            ticker="SPY", sentiment="bearish", premium=2_000_000,
            option_type="PUT", strike=400, expiry="2026-06-30",
        )
        mock_feeds.get_unusual_flow.return_value = [small_flow, big_flow]
        mock_feeds.get_dark_pool.return_value = []
        mock_feeds.get_congress_trades.return_value = []
        mock_feeds.get_tickers_near_earnings.return_value = []

        engine = IntelligenceEngine(data_feed_manager=mock_feeds)
        brief = engine.gather_intel(["SPY"])
        flow_sigs = [s for s in brief.signals if s.source == "unusual_whales"]
        assert len(flow_sigs) == 2
        assert flow_sigs[0].strength < flow_sigs[1].strength

    def test_ticker_sentiment_computed(self):
        """Ticker sentiment should be computed from all signals."""
        from strategies.matrix_maximizer.intelligence import IntelligenceEngine
        engine = IntelligenceEngine(data_feed_manager=None)
        brief = engine.gather_intel(["SPY"])
        assert "ticker_sentiment" in dir(brief)


# ═══════════════════════════════════════════════════════════════════════════
# __init__.py EXPORTS — HTTPHealthCheck
# ═══════════════════════════════════════════════════════════════════════════


class TestHTTPHealthExport:
    def test_http_health_importable(self):
        from strategies.matrix_maximizer.http_health import EndpointResult, HTTPHealthCheck
        assert HTTPHealthCheck is not None
        assert EndpointResult is not None

    def test_http_health_in_package_all(self):
        import strategies.matrix_maximizer as mm
        assert "HTTPHealthCheck" in mm.__all__

    def test_http_health_accessible_from_package(self):
        from strategies.matrix_maximizer import HTTPHealthCheck
        assert HTTPHealthCheck is not None

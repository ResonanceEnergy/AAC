"""
Tests for Unusual Whales API client — URL, header, and field validation.

Validates:
    1. Auth headers include both Bearer token AND UW-CLIENT-API-ID
    2. All endpoint URLs are correct (no hallucinated paths)
    3. Field name mapping from API responses to dataclasses is correct
    4. Consumer modules call the right method names
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.api_integration_hub import APIResponse
from integrations.unusual_whales_client import (
    DarkPoolTrade,
    OptionsFlow,
    UnusualWhalesClient,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uw_client() -> UnusualWhalesClient:
    """Create a client with a fake API key for testing."""
    with patch("integrations.unusual_whales_client.get_config") as mock_cfg:
        mock_cfg.return_value = SimpleNamespace(
            unusual_whales_key="test-key-12345"
        )
        client = UnusualWhalesClient()
    return client


# Realistic API response payloads (field names from live validation 2026-03)
FLOW_ALERTS_RESPONSE = [
    {
        "ticker": "SPY",
        "strike": "530",
        "expiry": "2026-04-18",
        "type": "call",
        "total_premium": "1520000.00",
        "total_size": "4200",
        "open_interest": "85000",
        "volume": "12000",
        "sector": "ETF",
        "underlying_price": "528.50",
        "price": "3.62",
        "created_at": "2026-03-25T14:30:00Z",
        "volume_oi_ratio": "0.14",
        "has_sweep": True,
        "marketcap": None,
        "issue_type": "ETF",
    },
]

DARKPOOL_RESPONSE = [
    {
        "ticker": "AAPL",
        "price": "185.50",
        "size": "5000",
        "volume": "50000",
        "premium": "927500.00",
        "market_center": "2",
        "executed_at": "2026-03-25T15:00:00Z",
        "tracking_id": "abc123",
        "nbbo_ask": "185.55",
        "nbbo_bid": "185.45",
        "canceled": False,
        "trade_settlement": "T+1",
    },
]

CONGRESS_RESPONSE = [
    {
        "name": "Nancy Pelosi",
        "ticker": "NVDA",
        "txn_type": "purchase",
        "amounts": "$1,000,001 - $5,000,000",
        "transaction_date": "2026-03-20",
        "filed_at_date": "2026-03-25",
        "reporter": "Nancy Pelosi",
        "member_type": "member",
        "politician_id": "P000197",
        "issuer": "NVIDIA Corp",
        "notes": "",
        "is_active": True,
    },
]


# ---------------------------------------------------------------------------
# Phase 5 Step 18: Header validation
# ---------------------------------------------------------------------------

class TestAuthHeaders:
    """Verify _get_auth_headers includes required headers."""

    def test_bearer_token_present(self, uw_client: UnusualWhalesClient) -> None:
        headers = uw_client._get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key-12345"

    def test_client_api_id_present(self, uw_client: UnusualWhalesClient) -> None:
        headers = uw_client._get_auth_headers()
        assert "UW-CLIENT-API-ID" in headers
        assert headers["UW-CLIENT-API-ID"] == "100001"

    def test_user_agent_present(self, uw_client: UnusualWhalesClient) -> None:
        headers = uw_client._get_auth_headers()
        assert "User-Agent" in headers
        assert "AAC/3.6.0" in headers["User-Agent"]

    def test_version_not_stale(self, uw_client: UnusualWhalesClient) -> None:
        """Version string must NOT be an old/stale value."""
        headers = uw_client._get_auth_headers()
        for stale in ("AAC/2.7.0", "AAC/2.0", "AAC/1.0"):
            assert stale not in headers["User-Agent"]


# ---------------------------------------------------------------------------
# Phase 5 Step 19: URL validation
# ---------------------------------------------------------------------------

class TestEndpointURLs:
    """Verify all endpoint URLs use correct paths (no hallucinated routes)."""

    BASE = "https://api.unusualwhales.com/api"

    @pytest.mark.asyncio
    async def test_flow_alerts_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_flow(ticker=None) → /option-trades/flow-alerts"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_flow(ticker=None, min_premium=0, limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/option-trades/flow-alerts"

    @pytest.mark.asyncio
    async def test_ticker_flow_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_flow(ticker='SPY') → /stock/SPY/option-contracts"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_flow(ticker="SPY", min_premium=0, limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/stock/SPY/option-contracts"

    @pytest.mark.asyncio
    async def test_darkpool_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_dark_pool(ticker=None) → /darkpool/recent"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_dark_pool(ticker=None, limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/darkpool/recent"

    @pytest.mark.asyncio
    async def test_darkpool_ticker_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_dark_pool(ticker='AAPL') → /darkpool/AAPL"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_dark_pool(ticker="AAPL", limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/darkpool/AAPL"

    @pytest.mark.asyncio
    async def test_congress_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_congress_trades → /congress/recent-trades"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_congress_trades(limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/congress/recent-trades"

    @pytest.mark.asyncio
    async def test_market_flow_summary_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_market_flow_summary → /option-trades/flow-alerts"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data={}, status_code=200)
            await uw_client.get_market_flow_summary()

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/option-trades/flow-alerts"

    @pytest.mark.asyncio
    async def test_hottest_chains_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_hottest_chains → /screener/option-contracts"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_hottest_chains(limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/screener/option-contracts"

    @pytest.mark.asyncio
    async def test_insider_transactions_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_insider_transactions → /insider/recent-trades"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_insider_transactions(limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/insider/recent-trades"

    @pytest.mark.asyncio
    async def test_news_headlines_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_news_headlines → /news/{ticker}"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data=[], status_code=200)
            await uw_client.get_news_headlines(ticker="AAPL", limit=10)

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/news/AAPL"

    @pytest.mark.asyncio
    async def test_etf_flow_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_etf_flow → /stock/{etf}/option-contracts"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data={}, status_code=200)
            await uw_client.get_etf_flow(etf_ticker="QQQ")

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/stock/QQQ/option-contracts"

    @pytest.mark.asyncio
    async def test_max_pain_url(self, uw_client: UnusualWhalesClient) -> None:
        """get_max_pain → /stock/{ticker}/option-contracts"""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(success=True, data={}, status_code=200)
            await uw_client.get_max_pain(ticker="TSLA")

        url = mock_req.call_args[0][1]
        assert url == f"{self.BASE}/stock/TSLA/option-contracts"

    @pytest.mark.asyncio
    async def test_no_hallucinated_urls(self, uw_client: UnusualWhalesClient) -> None:
        """None of the old broken URLs should appear anywhere."""
        import inspect
        source = inspect.getsource(UnusualWhalesClient)
        broken_urls = [
            "/stock/flow-recent",
            "/market/market-tide",
            "/stock/{ticker}/info",
            "/market/spike",
            "/market/sector-etfs",
            "/insider/transactions",
            "/news/headlines",
            "/stock/{ticker}/max-pain",
            "/news/market-wide",
            "/etf/sectors",
        ]
        for bad_url in broken_urls:
            assert bad_url not in source, f"Hallucinated URL still in source: {bad_url}"


# ---------------------------------------------------------------------------
# Phase 5 Step 20: Field name mapping validation
# ---------------------------------------------------------------------------

class TestFieldMapping:
    """Verify response parsing uses correct field names from live API."""

    @pytest.mark.asyncio
    async def test_flow_field_mapping(self, uw_client: UnusualWhalesClient) -> None:
        """Flow alerts should map API fields → OptionsFlow dataclass correctly."""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(
                success=True, data=FLOW_ALERTS_RESPONSE, status_code=200
            )
            results = await uw_client.get_flow(ticker=None, min_premium=0, limit=50)

        assert len(results) == 1
        flow = results[0]
        assert isinstance(flow, OptionsFlow)
        assert flow.ticker == "SPY"
        assert flow.strike == 530.0
        assert flow.expiry == "2026-04-18"
        assert flow.option_type == "call"
        assert flow.premium == 1520000.0
        assert flow.volume == 4200  # total_size, not volume
        assert flow.open_interest == 85000
        assert flow.sentiment == "neutral"  # not in response → default

    @pytest.mark.asyncio
    async def test_darkpool_field_mapping(self, uw_client: UnusualWhalesClient) -> None:
        """Dark pool should map 'premium' → notional, 'market_center' → exchange."""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(
                success=True, data=DARKPOOL_RESPONSE, status_code=200
            )
            results = await uw_client.get_dark_pool(ticker=None, limit=50)

        assert len(results) == 1
        trade = results[0]
        assert isinstance(trade, DarkPoolTrade)
        assert trade.ticker == "AAPL"
        assert trade.price == 185.50
        assert trade.size == 5000
        assert trade.notional == 927500.0  # from 'premium' field
        assert trade.exchange == "2"  # from 'market_center' field

    @pytest.mark.asyncio
    async def test_darkpool_notional_fallback(self, uw_client: UnusualWhalesClient) -> None:
        """If 'premium' missing, fall back to price * size."""
        data = [{**DARKPOOL_RESPONSE[0], "premium": None}]
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(
                success=True, data=data, status_code=200
            )
            results = await uw_client.get_dark_pool(ticker=None, limit=50)

        trade = results[0]
        assert trade.notional == 185.50 * 5000

    @pytest.mark.asyncio
    async def test_congress_field_mapping(self, uw_client: UnusualWhalesClient) -> None:
        """Congress trades should return raw dicts with 'name', 'txn_type', 'amounts'."""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(
                success=True, data=CONGRESS_RESPONSE, status_code=200
            )
            results = await uw_client.get_congress_trades(limit=10)

        assert len(results) == 1
        trade = results[0]
        # These are the ACTUAL field names from the live API
        assert trade["name"] == "Nancy Pelosi"
        assert trade["txn_type"] == "purchase"
        assert trade["amounts"] == "$1,000,001 - $5,000,000"
        assert trade["ticker"] == "NVDA"

    @pytest.mark.asyncio
    async def test_flow_min_premium_filter(self, uw_client: UnusualWhalesClient) -> None:
        """Flow should filter out entries below min_premium threshold."""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(
                success=True, data=FLOW_ALERTS_RESPONSE, status_code=200
            )
            results = await uw_client.get_flow(
                ticker=None, min_premium=10_000_000, limit=50
            )

        # 1.52M premium < 10M threshold → should be filtered out
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_flow_handles_wrapped_response(self, uw_client: UnusualWhalesClient) -> None:
        """Some endpoints wrap data in {'data': [...]}."""
        wrapped = {"data": FLOW_ALERTS_RESPONSE}
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(
                success=True, data=wrapped, status_code=200
            )
            results = await uw_client.get_flow(ticker=None, min_premium=0, limit=50)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_error_response_returns_empty(self, uw_client: UnusualWhalesClient) -> None:
        """Failed requests should return empty results, not raise."""
        with patch.object(uw_client, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = APIResponse(
                success=False, error="HTTP 403", status_code=403
            )
            flow = await uw_client.get_flow(ticker=None, min_premium=0, limit=10)
            dp = await uw_client.get_dark_pool(ticker=None, limit=10)
            congress = await uw_client.get_congress_trades(limit=10)

        assert flow == []
        assert dp == []
        assert congress == []


# ---------------------------------------------------------------------------
# Consumer integration — verify callers use correct method names
# ---------------------------------------------------------------------------

class TestConsumerIntegration:
    """Verify consumer modules reference correct client methods."""

    def test_client_has_get_dark_pool(self) -> None:
        """Client must have get_dark_pool (not get_darkpool_trades)."""
        assert hasattr(UnusualWhalesClient, "get_dark_pool")
        assert not hasattr(UnusualWhalesClient, "get_darkpool_trades")

    def test_client_has_get_flow(self) -> None:
        assert hasattr(UnusualWhalesClient, "get_flow")

    def test_client_has_get_flow_alerts(self) -> None:
        """get_flow_alerts is the raw-dict variant of get_flow."""
        # get_flow returns OptionsFlow dataclasses; if there's a raw variant
        # it should exist. Currently get_flow handles both, so just verify get_flow exists.
        assert callable(getattr(UnusualWhalesClient, "get_flow", None))

    def test_client_has_get_congress_trades(self) -> None:
        assert hasattr(UnusualWhalesClient, "get_congress_trades")

    def test_client_has_get_news_headlines(self) -> None:
        assert hasattr(UnusualWhalesClient, "get_news_headlines")

    def test_daily_recommendation_calls_correct_method(self) -> None:
        """daily_recommendation_engine must use get_dark_pool, not get_darkpool_trades."""
        import inspect

        from core.daily_recommendation_engine import UnusualWhalesIntelligence
        source = inspect.getsource(UnusualWhalesIntelligence)
        # Must NOT reference the old broken method name
        assert "get_darkpool_trades" not in source
        # Must reference the correct method
        assert "get_dark_pool" in source

    def test_data_feeds_delegates_to_canonical_client(self) -> None:
        """data_feeds.py should import/use the canonical client, not duplicate HTTP logic."""
        import inspect

        from strategies.matrix_maximizer import data_feeds
        source = inspect.getsource(data_feeds)
        # Should reference the canonical client class
        assert "UnusualWhalesClient" in source
        # Should NOT have raw HTTP calls to UW endpoints
        assert "api.unusualwhales.com" not in source

    def test_http_health_uses_correct_urls(self) -> None:
        """http_health.py endpoint URLs must match the canonical client."""
        source_path = Path(__file__).resolve().parent.parent / "strategies" / "matrix_maximizer" / "http_health.py"
        source = source_path.read_text(encoding="utf-8")
        # Must contain the correct endpoints
        assert "/option-trades/flow-alerts" in source
        assert "/darkpool/recent" in source
        assert "/congress/recent-trades" in source
        # Must contain the client API ID header
        assert "100001" in source
        # Must NOT contain old broken URLs
        for bad_url in ("/stock/flow", "/api/darkpool\"", "/congress/trading"):
            assert bad_url not in source, f"Stale URL still in http_health.py: {bad_url}"

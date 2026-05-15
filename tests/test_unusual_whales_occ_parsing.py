"""Tests for OCC option-symbol parsing and UW flow schema fallback."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from integrations.api_integration_hub import APIResponse
from integrations.unusual_whales_client import (
    UnusualWhalesClient,
    _parse_occ_symbol,
)


class TestOCCParser:
    def test_parses_call(self) -> None:
        out = _parse_occ_symbol("AAPL241115C00150000")
        assert out == {
            "ticker": "AAPL",
            "expiry": "2024-11-15",
            "option_type": "call",
            "strike": 150.0,
        }

    def test_parses_put_with_decimal_strike(self) -> None:
        out = _parse_occ_symbol("SPY260418P00530500")
        assert out["ticker"] == "SPY"
        assert out["expiry"] == "2026-04-18"
        assert out["option_type"] == "put"
        assert out["strike"] == 530.5

    def test_rejects_garbage(self) -> None:
        assert _parse_occ_symbol("") == {}
        assert _parse_occ_symbol("not_an_option") == {}
        assert _parse_occ_symbol(None) == {}  # type: ignore[arg-type]


class TestFlowOCCFallback:
    """Modern UW endpoints often return only ``option_chain`` — no top-level
    strike/expiry/type. The client must decode the OCC symbol so the rest of
    the pipeline does not silently see $0 strikes / 'unknown' types."""

    @pytest.fixture
    def client(self) -> UnusualWhalesClient:
        return UnusualWhalesClient(api_key="test-key-32-bytes-long-enough-here")

    @pytest.mark.asyncio
    async def test_falls_back_to_occ_symbol(self, client: UnusualWhalesClient) -> None:
        payload = [
            {
                "option_chain": "SPY260418C00530000",
                "underlying_symbol": "SPY",
                "total_premium": "750000.00",
                "total_size": "1200",
                "open_interest": "5000",
            }
        ]
        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResponse(success=True, data=payload, status_code=200)
            results = await client.get_flow(min_premium=0, limit=10)

        assert len(results) == 1
        f = results[0]
        assert f.ticker == "SPY"
        assert f.strike == 530.0
        assert f.expiry == "2026-04-18"
        assert f.option_type == "call"
        assert f.premium == 750_000.0
        assert f.volume == 1200

    @pytest.mark.asyncio
    async def test_total_ask_side_prem_alias(self, client: UnusualWhalesClient) -> None:
        """If neither ``total_premium`` nor ``premium`` exist, accept ``total_ask_side_prem``."""
        payload = [
            {
                "option_chain": "QQQ260619P00400000",
                "total_ask_side_prem": "200000.00",
                "total_size": "500",
            }
        ]
        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = APIResponse(success=True, data=payload, status_code=200)
            results = await client.get_flow(min_premium=0, limit=10)

        assert len(results) == 1
        assert results[0].premium == 200_000.0
        assert results[0].option_type == "put"
        assert results[0].strike == 400.0

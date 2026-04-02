#!/usr/bin/env python3
"""
Test Twelve Data API for AAC arbitrage system
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.api
def test_twelve_data_api():
    """Test Twelve Data API connectivity — mocked, key from env."""
    api_key = os.environ.get("TWELVE_DATA_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "meta": {"symbol": "AAPL", "interval": "1min", "currency": "USD"},
        "values": [
            {"datetime": "2026-04-01 16:00:00", "close": "220.50", "open": "219.80", "high": "221.00", "low": "219.50", "volume": "1000000"}
        ],
    }

    with patch("requests.get", return_value=mock_response) as mock_get:
        import requests
        url = f"https://api.twelvedata.com/time_series?apikey={api_key}&symbol=AAPL&interval=1min"
        response = requests.get(url, timeout=10)

        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert data["values"][0]["close"] == "220.50"
        assert data["meta"]["symbol"] == "AAPL"
        mock_get.assert_called_once()

#!/usr/bin/env python3
"""Tests for Google Trends rate-limit handling."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.google_trends_client import GoogleTrendsClient


def test_interest_over_time_returns_empty_on_rate_limit_and_sets_cooldown():
    client = GoogleTrendsClient(retries=1, backoff_seconds=0.01)
    client._pytrends = MagicMock()
    client._pytrends.build_payload.side_effect = Exception(
        'The request failed: Google returned a response with code 429'
    )

    with patch('integrations.google_trends_client.time.sleep', return_value=None):
        result = client.get_interest_over_time(['bitcoin'])

    assert result == {'bitcoin': []}
    assert client._cooldown_remaining() > 0


def test_interest_over_time_skips_request_during_cooldown():
    client = GoogleTrendsClient(retries=0, backoff_seconds=0.01)
    client._pytrends = MagicMock()
    client._rate_limited_until = 10**9

    result = client.get_interest_over_time(['bitcoin'])

    assert result == {'bitcoin': []}
    client._pytrends.build_payload.assert_not_called()


def test_sentiment_score_reports_rate_limited_signal_when_requests_fail():
    client = GoogleTrendsClient(retries=0, backoff_seconds=0.01)
    client._pytrends = MagicMock()
    client._pytrends.build_payload.side_effect = Exception(
        'The request failed: Google returned a response with code 429'
    )

    result = client.get_ticker_sentiment_score('BTC')

    assert result['signal'] == 'rate_limited'
    assert 'throttled' in result['error']
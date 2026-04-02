"""Tests for the Unusual Whales snapshot service and FFD integration hooks."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aac.doctrine.ffd.ffd_engine import FFDEngine
from integrations.unusual_whales_service import UnusualWhalesSnapshotService


@pytest.mark.asyncio
async def test_snapshot_service_unconfigured_returns_safe_status():
    service = UnusualWhalesSnapshotService()
    service.client.endpoint.enabled = False

    snapshot = await service.get_snapshot(force_refresh=True)

    assert snapshot["status"] == "unconfigured"
    assert snapshot["configured"] is False


@pytest.mark.asyncio
async def test_ffd_applies_market_intelligence_snapshot():
    engine = FFDEngine()
    engine._apply_market_intelligence(
        {
            "status": "healthy",
            "market_tone": "bearish",
            "put_call_ratio": 1.8,
            "options_flow_signal_count": 120,
        }
    )

    assert engine.metrics.options_flow_signal_count == 120
    assert engine.metrics.capital_flight_signal >= 55.0


def test_ffd_recommended_actions_reflect_state():
    engine = FFDEngine()
    engine.metrics.regulatory_shock_score = 65.0
    engine.metrics.capital_flight_signal = 52.0
    engine.metrics.options_flow_signal_count = 150

    actions = engine.get_recommended_actions()
    action_names = {item["action"] for item in actions}

    assert "enter_safe_mode" in action_names
    assert "throttle_risk" in action_names
    assert "increase_market_surveillance" in action_names

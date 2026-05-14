from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INFRA_BRIDGE = REPO_ROOT / "shared" / "intelligence_infrastructure_bridge.py"
ACCOUNTING_BRIDGE = REPO_ROOT / "shared" / "intelligence_accounting_bridge.py"
TRADING_INFRA_BRIDGE = REPO_ROOT / "shared" / "trading_infrastructure_bridge.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_infrastructure_bridge_fake_literals_removed() -> None:
    src = _source(INFRA_BRIDGE)
    assert "CPU utilization trending upward - consider scaling" not in src
    assert "Current capacity utilization at 75% - monitor closely" not in src
    assert "Disk I/O showing early warning signs - schedule maintenance" not in src
    assert "confidence\": 0.85" not in src
    assert "synthetic score generator" in src


def test_accounting_bridge_fake_literals_removed() -> None:
    src = _source(ACCOUNTING_BRIDGE)
    assert "Market prediction analysis completed" not in src
    assert '"expected_return": 0.085' not in src
    assert '"total_return": 0.072' not in src
    assert "synthetic attribution allocator" in src


def test_trading_infrastructure_bridge_placeholder_comments_removed() -> None:
    src = _source(TRADING_INFRA_BRIDGE)
    assert "placeholder logic" not in src
    assert "placeholder - would escalate based on severity" not in src
    assert "placeholder - would send to infrastructure" not in src


@pytest.mark.asyncio
async def test_infrastructure_bridge_predictive_monitoring_raises() -> None:
    from shared.intelligence_infrastructure_bridge import IntelligenceInfrastructureBridge

    bridge = IntelligenceInfrastructureBridge()
    with pytest.raises(NotImplementedError, match="Sprint 57"):
        await bridge._perform_predictive_monitoring("cpu-node-1", "1h", [1, 2, 3, 4, 5])


@pytest.mark.asyncio
async def test_infrastructure_bridge_anomaly_detection_raises() -> None:
    from shared.intelligence_infrastructure_bridge import IntelligenceInfrastructureBridge

    bridge = IntelligenceInfrastructureBridge()
    with pytest.raises(NotImplementedError, match="Sprint 57"):
        await bridge._perform_anomaly_detection("market-feed", "isolation_forest", 0.95)


@pytest.mark.asyncio
async def test_infrastructure_bridge_resource_optimization_raises() -> None:
    from shared.intelligence_infrastructure_bridge import IntelligenceInfrastructureBridge

    bridge = IntelligenceInfrastructureBridge()
    with pytest.raises(NotImplementedError, match="Sprint 57"):
        await bridge._perform_resource_optimization("cpu", {"utilization": 92}, {})


@pytest.mark.asyncio
async def test_accounting_bridge_research_analytics_returns_none_and_marks_failed() -> None:
    from shared.intelligence_accounting_bridge import IntelligenceAccountingBridge

    bridge = IntelligenceAccountingBridge()
    result = await bridge._perform_research_analytics(
        request_id="req-1",
        analysis_type="market_prediction",
        parameters={"symbol": "SPY"},
    )
    assert result is None
    assert bridge.active_analytics["req-1"]["status"] == "failed"
    assert "Sprint 57" in bridge.active_analytics["req-1"]["error"]


@pytest.mark.asyncio
async def test_accounting_bridge_performance_attribution_raises() -> None:
    from shared.intelligence_accounting_bridge import IntelligenceAccountingBridge

    bridge = IntelligenceAccountingBridge()
    with pytest.raises(NotImplementedError, match="Sprint 57"):
        await bridge._perform_performance_attribution("portfolio-1", "1M", ["macro", "beta"])
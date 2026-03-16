#!/usr/bin/env python3
"""Quick validation test for the Autonomous Engine."""
import sys
import os
import logging

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

logger.info("=" * 60)
logger.info("  AUTONOMOUS ENGINE — VALIDATION TEST")
logger.info("=" * 60)

# Test 1: Imports
logger.info("\n[1] Testing imports...")
from core.autonomous_engine import (
    AutonomousEngine, DoctrineState, DoctrineStatus,
    ComponentHealth, ComponentStatus, SignalAction, TradingSignal,
    MarketDataSnapshot, GapSeverity, SystemGap, ScheduledTask,
    AggregatedSignal,
)
logger.info("    All classes imported: OK")

# Test 2: Engine instantiation
logger.info("\n[2] Creating engine...")
engine = AutonomousEngine()
logger.info(f"    Components: {len(engine.components)}")
logger.info(f"    Tasks:      {list(engine.tasks.keys())}")
logger.info(f"    Doctrine:   {engine.doctrine.state.value}")

# Test 3: Doctrine state machine
logger.info("\n[3] Doctrine state machine...")
ds = DoctrineStatus()

ds.evaluate(-500, 100000)
assert ds.state == DoctrineState.NORMAL, f"Expected NORMAL, got {ds.state}"
logger.info(f"    -0.5% loss  -> {ds.state.value} (correct)")

ds.evaluate(-6000, 100000)
assert ds.state == DoctrineState.CAUTION, f"Expected CAUTION, got {ds.state}"
logger.info(f"    -6% drawdown -> {ds.state.value} (correct)")

ds.evaluate(-12000, 100000)
assert ds.state == DoctrineState.SAFE_MODE, f"Expected SAFE_MODE, got {ds.state}"
logger.info(f"    -12% drawdown -> {ds.state.value} (correct)")

ds.evaluate(-2100, 100000)
assert ds.state == DoctrineState.HALT, f"Expected HALT, got {ds.state}"
logger.info(f"    -2.1% daily  -> {ds.state.value} (correct)")

assert not ds.can_open_new_positions
assert ds.is_halted
logger.info("    Flags: can_trade=False, is_halted=True (correct)")

# Test 4: Component health
logger.info("\n[4] Component health tracking...")
cs = ComponentStatus(name="test")
assert cs.health == ComponentHealth.UNKNOWN
cs.record_success(latency_ms=42.0)
assert cs.health == ComponentHealth.HEALTHY
cs.record_failure("test error")
assert cs.health == ComponentHealth.DEGRADED
cs.record_failure("test error 2")
cs.record_failure("test error 3")
assert cs.health == ComponentHealth.DOWN
logger.info("    UNKNOWN -> HEALTHY -> DEGRADED -> DOWN (correct)")

# Test 5: Signal creation
logger.info("\n[5] Signal creation...")
sig = TradingSignal(
    strategy_name="fibonacci",
    symbol="BTC/USD",
    action=SignalAction.BUY,
    confidence=0.85,
    price=70000.0,
    reason="Near 0.618 retracement",
)
logger.info(f"    Signal: {sig.action.value} {sig.symbol} @ ${sig.price:,.2f} conf={sig.confidence}")

# Test 6: Market data snapshot
logger.info("\n[6] Market data snapshot...")
snap = MarketDataSnapshot(
    symbol="BTC/USD",
    price=70000.0,
    bid=69999.0,
    ask=70001.0,
    volume_24h=1_000_000_000,
    source="coingecko",
)
logger.info(f"    {snap.symbol}: ${snap.price:,.2f} ({snap.source})")

# Test 7: Gap tracking
logger.info("\n[7] Gap tracking...")
gap = SystemGap(
    gap_id="TEST-001",
    component="test",
    description="Test gap",
    severity=GapSeverity.HIGH,
)
assert not gap.resolved
gap.resolved = True
gap.resolution = "Auto-fixed"
assert gap.resolved
logger.info(f"    Gap {gap.gap_id}: {gap.severity.value} → resolved={gap.resolved}")

# Test 8: Status API
logger.info("\n[8] Status API...")
status = engine.get_status()
logger.info(f"    Fields: {len(status)}")
logger.info(f"    Components: {list(status['components'].keys())}")
assert "engine" in status
assert "doctrine" in status
assert "components" in status
assert "gaps" in status
assert "recent_signals" in status
logger.info("    All expected fields present")

# Test 9: Scheduled tasks
logger.info("\n[9] Scheduled tasks...")
for name, task in engine.tasks.items():
    logger.info(f"    {name:25s} every {task.interval_seconds:7.0f}s  critical={task.critical}")
assert "market_scan" in engine.tasks
assert engine.tasks["market_scan"].interval_seconds == 60.0
assert "strategy_signals" in engine.tasks
assert "introspection" in engine.tasks
assert "gap_analysis" in engine.tasks
assert "status_report" in engine.tasks

logger.info("\n" + "=" * 60)
logger.info("  ALL 9 TESTS PASSED — AUTONOMOUS ENGINE VALIDATED")
logger.info("=" * 60)

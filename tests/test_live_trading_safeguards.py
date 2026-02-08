#!/usr/bin/env python3
"""
Live Trading Safeguards Integration Tests
=========================================
Comprehensive testing for the live trading safety system.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.live_trading_safeguards import (
    live_trading_safeguards,
    SafetyAction,
    AlertLevel,
    RiskLevel,
    SafetyRule,
    SafetyAlert
)
from shared.audit_logger import get_audit_logger


async def test_safety_rules():
    """Test safety rule evaluation"""
    print("\n=== Testing Safety Rules ===")

    # Test daily loss limit rule
    rule = live_trading_safeguards.safety_rules["daily_loss_limit"]

    # Simulate high loss
    test_metrics = {"daily_pnl": -6000}  # Above $5000 limit
    triggered = rule.condition(test_metrics)
    assert triggered, "Daily loss limit rule should trigger"
    print("[OK] Daily loss limit rule triggered correctly")

    # Test normal conditions
    test_metrics = {"daily_pnl": -1000}  # Below limit
    triggered = rule.condition(test_metrics)
    assert not triggered, "Daily loss limit rule should not trigger"
    print("[OK] Daily loss limit rule correctly not triggered")

    # Test drawdown limit rule
    rule = live_trading_safeguards.safety_rules["drawdown_limit"]
    test_metrics = {"max_drawdown": 0.15}  # Above 10% limit
    triggered = rule.condition(test_metrics)
    assert triggered, "Drawdown limit rule should trigger"
    print("[OK] Drawdown limit rule triggered correctly")


async def test_trade_safety_checks():
    """Test trade safety validation"""
    print("\n=== Testing Trade Safety Checks ===")

    # Test normal trade
    trade_details = {
        "symbol": "AAPL",
        "quantity": 50,
        "price": 150.0
    }
    safe, message = await live_trading_safeguards.check_trade_safety(trade_details)
    assert safe, f"Normal trade should be approved: {message}"
    print("[OK] Normal trade approved")

    # Test oversized position
    large_trade = {
        "symbol": "AAPL",
        "quantity": 1000,
        "price": 150.0
    }
    safe, message = await live_trading_safeguards.check_trade_safety(large_trade)
    assert not safe, "Large trade should be rejected"
    assert "Position size" in message, "Should mention position size limit"
    print("[OK] Oversized position correctly rejected")

    # Test halted trading
    live_trading_safeguards.trading_halted = True
    safe, message = await live_trading_safeguards.check_trade_safety(trade_details)
    assert not safe, "Trade should be rejected when trading halted"
    assert "halted" in message, "Should mention trading halt"
    print("[OK] Trade correctly rejected during halt")

    # Reset
    live_trading_safeguards.trading_halted = False


async def test_safety_actions():
    """Test safety action execution"""
    print("\n=== Testing Safety Actions ===")

    # Test halt trading action
    alert = SafetyAlert(
        alert_id="test_halt",
        rule_id="test_rule",
        message="Test halt trading",
        level=AlertLevel.CRITICAL,
        triggered_at=asyncio.get_event_loop().time()
    )

    await live_trading_safeguards._execute_safety_action(SafetyAction.HALT_TRADING, alert)
    assert live_trading_safeguards.trading_halted, "Trading should be halted"
    print("[OK] Trading halt action executed")

    # Test emergency shutdown
    alert2 = SafetyAlert(
        alert_id="test_shutdown",
        rule_id="test_rule",
        message="Test emergency shutdown",
        level=AlertLevel.CRITICAL,
        triggered_at=asyncio.get_event_loop().time()
    )

    await live_trading_safeguards._execute_safety_action(SafetyAction.EMERGENCY_SHUTDOWN, alert2)
    assert live_trading_safeguards.emergency_shutdown, "Emergency shutdown should be active"
    print("[OK] Emergency shutdown action executed")

    # Reset for further tests
    live_trading_safeguards.trading_halted = False
    live_trading_safeguards.emergency_shutdown = False


async def test_alert_system():
    """Test alert generation and management"""
    print("\n=== Testing Alert System ===")

    # Execute safety checks to generate alerts
    alerts = await live_trading_safeguards.execute_safety_check()
    print(f"[OK] Generated {len(alerts)} alerts from safety checks")

    # Check active alerts
    active_alerts = live_trading_safeguards.get_active_alerts()
    print(f"[OK] Active alerts: {len(active_alerts)}")

    # Resolve an alert if any exist
    if active_alerts:
        alert_id = active_alerts[0].alert_id
        live_trading_safeguards.resolve_alert(alert_id)

        # Check resolution
        active_after = live_trading_safeguards.get_active_alerts()
        assert len(active_after) < len(active_alerts), "Alert should be resolved"
        print("[OK] Alert resolution working")

    # Check alert history
    history = live_trading_safeguards.alert_history
    print(f"[OK] Alert history contains {len(history)} entries")


async def test_system_health():
    """Test system health monitoring"""
    print("\n=== Testing System Health ===")

    health_score = live_trading_safeguards._check_system_health()
    assert 0.0 <= health_score <= 1.0, f"Health score should be between 0 and 1: {health_score}"
    print(f"[OK] System health score: {health_score:.2f}")

    # Test health in metrics
    metrics = live_trading_safeguards._get_current_metrics()
    assert "system_health" in metrics, "System health should be in metrics"
    print("[OK] System health included in metrics")


async def test_risk_limits():
    """Test risk limit configuration"""
    print("\n=== Testing Risk Limits ===")

    limits = live_trading_safeguards.risk_limits

    # Check default limits
    assert limits.max_daily_loss == 5000.0, "Default daily loss limit should be $5000"
    assert limits.max_drawdown == 0.1, "Default drawdown limit should be 10%"
    assert limits.max_position_size == 10000.0, "Default position size should be $10000"
    print("[OK] Default risk limits loaded correctly")

    # Test limit enforcement in trade checks
    risky_trade = {
        "symbol": "TSLA",
        "quantity": 1000,
        "price": 200.0  # $200,000 position - above limit
    }
    safe, message = await live_trading_safeguards.check_trade_safety(risky_trade)
    assert not safe, "Risky trade should be rejected"
    print("[OK] Risk limits enforced in trade validation")


async def test_custom_rules():
    """Test custom safety rule addition"""
    print("\n=== Testing Custom Rules ===")

    # Add a custom rule
    custom_rule_config = {
        "rule_id": "test_custom_rule",
        "name": "Test Custom Rule",
        "description": "A test custom safety rule",
        "condition_code": "metrics.get('portfolio_value', 0) > 50000",
        "action": "halt_trading",
        "alert_level": "warning",
        "enabled": True,
        "cooldown_period": 60
    }

    live_trading_safeguards._add_custom_rule(custom_rule_config)

    # Check if rule was added
    assert "test_custom_rule" in live_trading_safeguards.safety_rules, "Custom rule should be added"
    rule = live_trading_safeguards.safety_rules["test_custom_rule"]
    assert rule.name == "Test Custom Rule", "Rule name should match"
    print("[OK] Custom rule added successfully")

    # Test custom rule condition
    test_metrics = {"portfolio_value": 60000}  # Above threshold
    triggered = rule.condition(test_metrics)
    assert triggered, "Custom rule should trigger on high portfolio value"
    print("[OK] Custom rule condition working")


async def test_monitoring_loops():
    """Test monitoring and health check loops"""
    print("\n=== Testing Monitoring Loops ===")

    # Let monitoring run for a short time
    await asyncio.sleep(2)

    # Check that metrics are being updated
    metrics = live_trading_safeguards._get_current_metrics()
    assert "last_updated" in str(metrics), "Metrics should have timestamps"
    print("[OK] Monitoring loop updating metrics")

    # Check health monitoring
    health_score = live_trading_safeguards._check_system_health()
    assert isinstance(health_score, float), "Health score should be a float"
    print("[OK] Health monitoring active")


async def test_safety_status():
    """Test safety status reporting"""
    print("\n=== Testing Safety Status ===")

    status = live_trading_safeguards.get_safety_status()

    required_keys = [
        "trading_halted", "emergency_shutdown", "active_alerts",
        "total_alerts_today", "circuit_breakers", "risk_limits", "current_metrics"
    ]

    for key in required_keys:
        assert key in status, f"Status should include {key}"

    print("[OK] Safety status contains all required information")
    print(f"  Trading halted: {status['trading_halted']}")
    print(f"  Emergency shutdown: {status['emergency_shutdown']}")
    print(f"  Active alerts: {status['active_alerts']}")


async def test_trade_frequency_limits():
    """Test trading frequency limit enforcement"""
    print("\n=== Testing Trade Frequency Limits ===")

    # Reset counters
    live_trading_safeguards.hourly_trade_count = 0
    live_trading_safeguards.daily_trade_count = 0

    # Test normal trading
    for i in range(3):
        safe, message = await live_trading_safeguards.check_trade_safety({
            "symbol": f"TEST{i}",
            "quantity": 100,
            "price": 100.0
        })
        assert safe, f"Trade {i+1} should be approved"

    print("[OK] Normal trading frequency allowed")

    # Simulate high frequency trading
    live_trading_safeguards.hourly_trade_count = 55  # Above 50 limit

    safe, message = await live_trading_safeguards.check_trade_safety({
        "symbol": "TEST",
        "quantity": 100,
        "price": 100.0
    })
    assert not safe, "High frequency trade should be rejected"
    assert "Hourly trade limit" in message, "Should mention hourly limit"
    print("[OK] High frequency trading correctly rejected")


async def test_circuit_breakers():
    """Test circuit breaker functionality"""
    print("\n=== Testing Circuit Breakers ===")

    # Check initial state
    breakers = live_trading_safeguards.circuit_breakers
    assert isinstance(breakers, dict), "Circuit breakers should be a dict"
    assert "daily_loss_limit" in breakers, "Should have daily loss breaker"
    print("[OK] Circuit breakers initialized")

    # Test breaker activation (would be set by safety rules in real usage)
    live_trading_safeguards.circuit_breakers["volatility_spike"] = True
    status = live_trading_safeguards.get_safety_status()
    assert status["circuit_breakers"]["volatility_spike"], "Circuit breaker should be active"
    print("[OK] Circuit breaker activation working")


async def run_stress_test():
    """Run stress test with multiple concurrent safety checks"""
    print("\n=== Stress Test ===")

    import time
    start_time = time.time()

    # Run 100 concurrent safety checks
    tasks = []
    for i in range(100):
        task = live_trading_safeguards.check_trade_safety({
            "symbol": f"STRESS{i % 10}",
            "quantity": 10 + (i % 50),
            "price": 100.0 + (i % 200)
        })
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    end_time = time.time()

    approved = sum(1 for safe, _ in results if safe)
    rejected = len(results) - approved

    print(f"[OK] Processed {len(results)} safety checks in {end_time - start_time:.2f} seconds")
    print(f"  Approved: {approved}")
    print(f"  Rejected: {rejected}")
    print(f"  Checks per second: {len(results) / (end_time - start_time):.1f}")


async def main():
    """Run all live trading safeguards tests"""
    print("[SHIELD]️  Starting Live Trading Safeguards Integration Tests")
    print("=" * 60)

    # Initialize the safeguards system
    from shared.live_trading_safeguards import initialize_live_trading_safeguards
    await initialize_live_trading_safeguards()

    # Run tests
    tests = [
        test_safety_rules,
        test_trade_safety_checks,
        test_safety_actions,
        test_alert_system,
        test_system_health,
        test_risk_limits,
        test_custom_rules,
        test_monitoring_loops,
        test_safety_status,
        test_trade_frequency_limits,
        test_circuit_breakers,
        run_stress_test,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"[CROSS] {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"[MONITOR] Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("[CELEBRATION] All live trading safeguards tests passed!")
    else:
        print("[WARN]️  Some tests failed. Check output above.")

    # Final safety status
    print("\n📈 Final Safety Status:")
    status = live_trading_safeguards.get_safety_status()
    print(f"  Trading Status: {'HALTED' if status['trading_halted'] else 'ACTIVE'}")
    print(f"  Emergency Shutdown: {'ACTIVE' if status['emergency_shutdown'] else 'INACTIVE'}")
    print(f"  Active Alerts: {status['active_alerts']}")
    print(f"  System Health: {status['current_metrics'].get('system_health', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
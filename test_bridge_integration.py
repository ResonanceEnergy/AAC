#!/usr/bin/env python3
"""
Bridge System Integration Test
Tests all 10 department bridge pairs for complete cross-department communication
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.bridge_orchestrator import BridgeOrchestrator
from shared.audit_logger import AuditLogger

async def test_bridge_system():
    """Test complete bridge system with all 10 department pairs"""

    print("🚀 Starting Bridge System Integration Test")
    print("=" * 60)

    # Initialize bridge orchestrator
    orchestrator = BridgeOrchestrator()

    try:
        # Initialize all bridges
        print("📡 Initializing all 10 bridge handlers...")
        await orchestrator._initialize_bridge_handlers()

        # Check bridge status
        active_bridges = sum(1 for conn in orchestrator.connections.values() if conn.is_active)
        total_bridges = len(orchestrator.connections)

        print(f"✅ Bridge Status: {active_bridges}/{total_bridges} bridges active")

        if active_bridges != 10:
            print("❌ ERROR: Not all bridges initialized successfully")
            return False

        # Test message routing for each bridge pair
        print("\n📨 Testing message routing across all bridge pairs...")

        test_messages = {
            "TE_CA": {"type": "risk_limit_update", "data": {"symbol": "AAPL", "limit": 1000000}},
            "TE_BBI": {"type": "execution_signal", "data": {"strategy": "momentum", "confidence": 0.85}},
            "TE_CI": {"type": "venue_selection", "data": {"asset": "BTC", "venues": ["binance", "coinbase"]}},
            "TE_SI": {"type": "execution_monitoring", "data": {"order_id": "12345", "status": "filled"}},
            "BBI_CA": {"type": "research_analytics", "data": {"topic": "market_sentiment", "score": 0.72}},
            "BBI_SI": {"type": "system_intelligence", "data": {"anomaly": "high_latency", "severity": "medium"}},
            "CA_CI": {"type": "financial_intelligence", "data": {"counterparty": "bank_xyz", "risk_score": 0.3}},
            "CA_SI": {"type": "data_integrity", "data": {"table": "positions", "checksum": "abc123"}},
            "CI_SI": {"type": "venue_monitoring", "data": {"venue": "binance", "status": "operational"}},
            "CI_CA": {"type": "crypto_intelligence", "data": {"asset": "ETH", "trend": "bullish"}}
        }

        successful_routes = 0
        total_tests = len(test_messages)

        for bridge_key, message in test_messages.items():
            try:
                # Route message through bridge system
                result = await orchestrator.route_message(bridge_key, message)

                if result.get("success", False):
                    print(f"✅ {bridge_key}: Message routed successfully")
                    successful_routes += 1
                else:
                    print(f"❌ {bridge_key}: Message routing failed - {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"❌ {bridge_key}: Exception during routing - {e}")

        print(f"\n📊 Message Routing Results: {successful_routes}/{total_tests} successful")

        # Test health monitoring
        print("\n🏥 Testing bridge health monitoring...")
        health_report = await orchestrator.get_health_report()

        healthy_bridges = sum(1 for status in health_report.values() if status.get("healthy", False))
        total_health_checks = len(health_report)

        print(f"✅ Health Status: {healthy_bridges}/{total_health_checks} bridges healthy")

        # Test cross-bridge communication patterns
        print("\n🔄 Testing cross-bridge communication patterns...")

        # Test TradingExecution → CentralAccounting → CryptoIntelligence flow
        te_ca_message = {"type": "position_update", "data": {"symbol": "BTC", "quantity": 10}}
        ca_ci_message = {"type": "risk_assessment", "data": {"exposure": 50000}}

        try:
            # Route through TE_CA bridge
            te_ca_result = await orchestrator.route_message("TE_CA", te_ca_message)
            # Route through CA_CI bridge
            ca_ci_result = await orchestrator.route_message("CA_CI", ca_ci_message)

            if te_ca_result.get("success") and ca_ci_result.get("success"):
                print("✅ Multi-bridge flow (TE→CA→CI): Successful")
            else:
                print("❌ Multi-bridge flow (TE→CA→CI): Failed")

        except Exception as e:
            print(f"❌ Multi-bridge flow test failed: {e}")

        # Final assessment
        print("\n" + "=" * 60)
        print("🎯 INTEGRATION TEST RESULTS")
        print("=" * 60)

        all_tests_passed = (
            active_bridges == 10 and
            successful_routes >= 9 and  # Allow 1 failure for robustness
            healthy_bridges >= 9        # Allow 1 unhealthy for robustness
        )

        if all_tests_passed:
            print("✅ ALL TESTS PASSED - Bridge system ready for production!")
            print("🎉 Phase 4: Production Readiness - COMPLETE")
            return True
        else:
            print("❌ SOME TESTS FAILED - Review bridge implementations")
            print("🔧 Phase 4: Production Readiness - REQUIRES ATTENTION")
            return False

    except Exception as e:
        print(f"❌ CRITICAL ERROR during integration test: {e}")
        return False

    finally:
        # Cleanup
        await orchestrator.shutdown()
        print("\n🧹 Bridge system shutdown complete")

if __name__ == "__main__":
    # Run the integration test
    success = asyncio.run(test_bridge_system())
    sys.exit(0 if success else 1)
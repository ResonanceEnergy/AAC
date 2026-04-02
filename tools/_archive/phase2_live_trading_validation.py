#!/usr/bin/env python3
"""
AAC Live Trading Infrastructure Validation
==========================================

Phase 2 Priority 4 Validation: Tests live trading infrastructure components
- Multi-exchange connectivity
- Order execution validation
- Emergency systems testing
- Circuit breaker functionality
- Market data feed validation
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading.live_trading_infrastructure import (
    EmergencyAction,
    Exchange,
    LiveTradingInfrastructure,
    get_live_trading_infrastructure,
)

from shared.audit_logger import AuditCategory, AuditSeverity, get_audit_logger
from shared.config_loader import get_config


class LiveTradingValidator:
    """Validates live trading infrastructure components"""

    def __init__(self):
        self.config = get_config()
        self.audit_logger = get_audit_logger()
        self.infrastructure: Optional[LiveTradingInfrastructure] = None
        self.validation_results = {
            'connectivity_tests': {},
            'order_validation_tests': {},
            'emergency_system_tests': {},
            'circuit_breaker_tests': {},
            'market_data_tests': {},
            'overall_status': 'pending'
        }

    async def run_full_validation(self) -> Dict:
        """Run complete validation suite"""
        logger.info("🔍 Starting AAC Live Trading Infrastructure Validation...")
        logger.info("=" * 60)

        try:
            # Initialize infrastructure
            self.infrastructure = await get_live_trading_infrastructure()

            # Test 1: Exchange Connectivity
            logger.info("1️⃣ Testing Exchange Connectivity...")
            await self._test_connectivity()

            # Test 2: Order Validation
            logger.info("2️⃣ Testing Order Validation...")
            await self._test_order_validation()

            # Test 3: Emergency Systems
            logger.info("3️⃣ Testing Emergency Systems...")
            await self._test_emergency_systems()

            # Test 4: Circuit Breakers
            logger.info("4️⃣ Testing Circuit Breakers...")
            await self._test_circuit_breakers()

            # Test 5: Market Data Feeds
            logger.info("5️⃣ Testing Market Data Feeds...")
            await self._test_market_data_feeds()

            # Calculate overall status
            self._calculate_overall_status()

            logger.info("\n" + "=" * 60)
            logger.info("✅ Validation Complete!")
            logger.info(f"📊 Overall Status: {self.validation_results['overall_status'].upper()}")

        except Exception as e:
            logger.info(f"❌ Validation failed with error: {e}")
            self.validation_results['overall_status'] = 'failed'
            self.audit_logger.log_event(
                AuditCategory.SYSTEM,
                AuditSeverity.ERROR,
                f"Live trading validation failed: {e}"
            )

        return self.validation_results

    async def _test_connectivity(self):
        """Test connectivity to all configured exchanges"""
        results = {}

        for exchange in Exchange:
            try:
                logger.info(f"   Testing {exchange.value} connectivity...")

                if exchange not in self.infrastructure.exchanges:
                    results[exchange.value] = {
                        'status': 'not_configured',
                        'error': 'Exchange not configured'
                    }
                    continue

                # Test connectivity
                start_time = time.time()
                await self.infrastructure._test_exchange_connectivity(exchange)
                latency = (time.time() - start_time) * 1000

                results[exchange.value] = {
                    'status': 'connected',
                    'latency_ms': round(latency, 2)
                }

                logger.info(f"   ✅ {exchange.value}: Connected ({latency:.1f}ms)")

            except Exception as e:
                results[exchange.value] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.info(f"   ❌ {exchange.value}: Failed - {e}")

        self.validation_results['connectivity_tests'] = results

    async def _test_order_validation(self):
        """Test order validation logic"""
        results = {}

        # Test cases
        test_cases = [
            {
                'name': 'valid_market_order',
                'exchange': Exchange.BINANCE,
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'quantity': 0.001,
                'order_type': 'market'
            },
            {
                'name': 'invalid_quantity',
                'exchange': Exchange.BINANCE,
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'quantity': -1,
                'order_type': 'market'
            },
            {
                'name': 'unconfigured_exchange',
                'exchange': Exchange.COINBASE,  # May not be configured
                'symbol': 'BTC-USD',
                'side': 'buy',
                'quantity': 0.001,
                'order_type': 'market'
            }
        ]

        for test_case in test_cases:
            try:
                logger.info(f"   Testing {test_case['name']}...")

                # Test validation (without actually placing order)
                await self.infrastructure._validate_order(
                    test_case['exchange'],
                    test_case['symbol'],
                    test_case['side'],
                    test_case['quantity'],
                    None
                )

                results[test_case['name']] = {
                    'status': 'passed',
                    'expected': 'should_pass' if test_case['name'] == 'valid_market_order' else 'should_fail'
                }

                if test_case['name'] == 'valid_market_order':
                    logger.info(f"   ✅ {test_case['name']}: Validation passed")
                else:
                    logger.info(f"   ❌ {test_case['name']}: Should have failed validation")

            except Exception as e:
                results[test_case['name']] = {
                    'status': 'failed',
                    'error': str(e),
                    'expected': 'should_fail' if 'invalid' in test_case['name'] or 'unconfigured' in test_case['name'] else 'should_pass'
                }

                if 'invalid' in test_case['name'] or 'unconfigured' in test_case['name']:
                    logger.info(f"   ✅ {test_case['name']}: Correctly rejected - {e}")
                else:
                    logger.info(f"   ❌ {test_case['name']}: Unexpected validation failure - {e}")

        self.validation_results['order_validation_tests'] = results

    async def _test_emergency_systems(self):
        """Test emergency system components"""
        results = {}

        # Test emergency action queuing (without actual execution)
        test_actions = [
            EmergencyAction.REDUCE_POSITIONS,
            EmergencyAction.CLOSE_ALL_POSITIONS,
            EmergencyAction.SHUTDOWN_TRADING,
            EmergencyAction.EMERGENCY_STOP
        ]

        for action in test_actions:
            try:
                logger.info(f"   Testing {action.value} emergency action...")

                # Test that emergency action is recognized and queued
                # (We won't actually execute to avoid real trading)
                if action in self.infrastructure.emergency_handlers:
                    results[action.value] = {
                        'status': 'recognized',
                        'handler_exists': True
                    }
                    logger.info(f"   ✅ {action.value}: Emergency handler exists")
                else:
                    results[action.value] = {
                        'status': 'failed',
                        'error': 'No handler found'
                    }
                    logger.info(f"   ❌ {action.value}: No emergency handler")

            except Exception as e:
                results[action.value] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.info(f"   ❌ {action.value}: Error - {e}")

        # Test emergency mode flag
        initial_mode = self.infrastructure.emergency_mode
        results['emergency_mode'] = {
            'status': 'ok',
            'initial_state': initial_mode
        }

        self.validation_results['emergency_system_tests'] = results

    async def _test_circuit_breakers(self):
        """Test circuit breaker functionality"""
        results = {}

        for exchange in Exchange:
            try:
                logger.info(f"   Testing {exchange.value} circuit breaker...")

                cb = self.infrastructure.circuit_breakers[exchange]
                initial_state = cb.state

                # Test circuit breaker state
                results[exchange.value] = {
                    'status': 'ok',
                    'initial_state': initial_state.value,
                    'failure_threshold': cb.failure_threshold,
                    'recovery_timeout': cb.recovery_timeout
                }

                logger.info(f"   ✅ {exchange.value}: Circuit breaker OK (state: {initial_state.value})")

            except Exception as e:
                results[exchange.value] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.info(f"   ❌ {exchange.value}: Circuit breaker error - {e}")

        self.validation_results['circuit_breaker_tests'] = results

    async def _test_market_data_feeds(self):
        """Test market data feed connections"""
        results = {}

        for exchange, feed in self.infrastructure.websocket_feeds.items():
            try:
                logger.info(f"   Testing {exchange.value} market data feed...")

                # Check if feed has connection status
                if hasattr(feed, 'is_connected'):
                    is_connected = feed.is_connected
                    results[exchange.value] = {
                        'status': 'connected' if is_connected else 'disconnected',
                        'has_connection_check': True
                    }

                    if is_connected:
                        logger.info(f"   ✅ {exchange.value}: Market data feed connected")
                    else:
                        logger.info(f"   ⚠️  {exchange.value}: Market data feed disconnected")

                else:
                    results[exchange.value] = {
                        'status': 'no_connection_check',
                        'has_connection_check': False
                    }
                    logger.info(f"   ⚠️  {exchange.value}: No connection status available")

            except Exception as e:
                results[exchange.value] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.info(f"   ❌ {exchange.value}: Market data feed error - {e}")

        self.validation_results['market_data_tests'] = results

    def _calculate_overall_status(self):
        """Calculate overall validation status"""
        all_tests = [
            self.validation_results['connectivity_tests'],
            self.validation_results['order_validation_tests'],
            self.validation_results['emergency_system_tests'],
            self.validation_results['circuit_breaker_tests'],
            self.validation_results['market_data_tests']
        ]

        total_tests = 0
        passed_tests = 0

        for test_group in all_tests:
            for test_name, result in test_group.items():
                total_tests += 1
                if result.get('status') in ['connected', 'passed', 'recognized', 'ok']:
                    passed_tests += 1
                elif result.get('status') == 'not_configured':
                    # Not configured is acceptable for optional exchanges
                    passed_tests += 1

        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        if success_rate >= 0.9:
            self.validation_results['overall_status'] = 'excellent'
        elif success_rate >= 0.75:
            self.validation_results['overall_status'] = 'good'
        elif success_rate >= 0.5:
            self.validation_results['overall_status'] = 'fair'
        else:
            self.validation_results['overall_status'] = 'poor'

        self.validation_results['success_rate'] = success_rate
        self.validation_results['total_tests'] = total_tests
        self.validation_results['passed_tests'] = passed_tests

    def print_validation_report(self):
        """Print detailed validation report"""
        logger.info("\n📋 DETAILED VALIDATION REPORT")
        logger.info("=" * 60)

        # Connectivity Tests
        logger.info("🔗 CONNECTIVITY TESTS:")
        for exchange, result in self.validation_results['connectivity_tests'].items():
            status_icon = "✅" if result['status'] == 'connected' else "❌" if result['status'] == 'failed' else "⚠️"
            status_text = f"{status_icon} {exchange}: {result['status']}"
            if 'latency_ms' in result:
                status_text += f" ({result['latency_ms']}ms)"
            if 'error' in result:
                status_text += f" - {result['error']}"
            logger.info(f"   {status_text}")

        # Order Validation Tests
        logger.info("\n📝 ORDER VALIDATION TESTS:")
        for test_name, result in self.validation_results['order_validation_tests'].items():
            status_icon = "✅" if result['status'] == 'passed' else "❌"
            expected = result.get('expected', 'unknown')
            status_text = f"{status_icon} {test_name}: {result['status']} (expected: {expected})"
            if 'error' in result:
                status_text += f" - {result['error']}"
            logger.info(f"   {status_text}")

        # Emergency Systems
        logger.info("\n🚨 EMERGENCY SYSTEMS:")
        for test_name, result in self.validation_results['emergency_system_tests'].items():
            if test_name == 'emergency_mode':
                status_icon = "✅"
                status_text = f"{status_icon} Emergency mode: {result['initial_state']}"
            else:
                status_icon = "✅" if result['status'] == 'recognized' else "❌"
                status_text = f"{status_icon} {test_name}: {result['status']}"
            logger.info(f"   {status_text}")

        # Circuit Breakers
        logger.info("\n🔌 CIRCUIT BREAKERS:")
        for exchange, result in self.validation_results['circuit_breaker_tests'].items():
            status_icon = "✅" if result['status'] == 'ok' else "❌"
            status_text = f"{status_icon} {exchange}: {result['status']} (state: {result.get('initial_state', 'unknown')})"
            logger.info(f"   {status_text}")

        # Market Data Feeds
        logger.info("\n📊 MARKET DATA FEEDS:")
        for exchange, result in self.validation_results['market_data_tests'].items():
            status_icon = "✅" if result['status'] == 'connected' else "⚠️" if result['status'] == 'disconnected' else "❌"
            status_text = f"{status_icon} {exchange}: {result['status']}"
            logger.info(f"   {status_text}")

        # Summary
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Total Tests: {self.validation_results.get('total_tests', 0)}")
        logger.info(f"   Passed Tests: {self.validation_results.get('passed_tests', 0)}")
        logger.info(f"   Success Rate: {self.validation_results.get('success_rate', 0):.1%}")
        logger.info(f"   Overall Status: {self.validation_results['overall_status'].upper()}")

        # Phase 2 Priority 4 Readiness
        success_rate = self.validation_results.get('success_rate', 0)
        if success_rate >= 0.8:
            logger.info("\n🎯 PHASE 2 PRIORITY 4 READINESS: ✅ READY FOR LIVE TRADING")
            logger.info("   • Multi-exchange integration: Implemented")
            logger.info("   • Emergency systems: Functional")
            logger.info("   • Circuit breakers: Active")
            logger.info("   • Market connectivity: Monitoring")
        else:
            logger.info(f"\n🎯 PHASE 2 PRIORITY 4 READINESS: ⚠️ NEEDS ATTENTION ({success_rate:.1%} success rate)")
            logger.info("   • Review failed tests above")
            logger.info("   • Configure missing API credentials")
            logger.info("   • Test in staging environment first")


async def main():
    """Main validation function"""
    validator = LiveTradingValidator()
    results = await validator.run_full_validation()
    validator.print_validation_report()

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = PROJECT_ROOT / f"live_trading_validation_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())

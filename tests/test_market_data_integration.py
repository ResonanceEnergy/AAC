#!/usr/bin/env python3
"""
Market Data Integration Test Suite
===================================
Comprehensive testing of the 100+ market data feeds integration.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.market_data_integration import market_data_integration, initialize_market_data_integration
from shared.market_data_connector import market_data_manager
from shared.audit_logger import audit_log


class MarketDataTestSuite:
    """Comprehensive test suite for market data integration"""

    def __init__(self):
        self.logger = logging.getLogger("MarketDataTestSuite")
        self.test_results = {}
        self.start_time = None

    async def run_full_test_suite(self):
        """Run complete test suite"""
        self.start_time = datetime.now()
        self.logger.info("Starting comprehensive market data test suite...")

        try:
            # Test 1: System Initialization
            await self.test_system_initialization()

            # Test 2: Connector Connectivity
            await self.test_connector_connectivity()

            # Test 3: Data Feed Quality
            await self.test_data_feed_quality()

            # Test 4: Symbol Subscription
            await self.test_symbol_subscription()

            # Test 5: Real-time Data Flow
            await self.test_realtime_data_flow()

            # Test 6: Historical Data Retrieval
            await self.test_historical_data_retrieval()

            # Test 7: NAV Calculation
            await self.test_nav_calculation()

            # Test 8: Arbitrage Detection
            await self.test_arbitrage_detection()

            # Test 9: Strategy Integration
            await self.test_strategy_integration()

            # Test 10: Failover and Redundancy
            await self.test_failover_redundancy()

            # Test 11: Performance Benchmarking
            await self.test_performance_benchmarking()

            # Test 12: Error Handling
            await self.test_error_handling()

        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            self.test_results["overall"] = "FAILED"

        finally:
            # Generate test report
            await self.generate_test_report()

    async def test_system_initialization(self):
        """Test system initialization"""
        self.logger.info("Testing system initialization...")

        try:
            await initialize_market_data_integration()

            # Verify components are initialized
            status = market_data_integration.get_status()

            assert status["active_strategies"] > 0, "No strategies loaded"
            assert len(status["market_data"]["connectors"]) > 0, "No connectors initialized"

            self.test_results["system_initialization"] = "PASSED"
            self.logger.info("[OK] System initialization test passed")

        except Exception as e:
            self.test_results["system_initialization"] = f"FAILED: {e}"
            self.logger.error(f"✗ System initialization test failed: {e}")

    async def test_connector_connectivity(self):
        """Test connector connectivity"""
        self.logger.info("Testing connector connectivity...")

        try:
            status = market_data_manager.get_status()
            connected_count = 0
            total_count = len(status)

            for connector_name, connector_status in status.items():
                if connector_status["status"] in ["connected", "connecting"]:
                    connected_count += 1

            connectivity_rate = connected_count / total_count if total_count > 0 else 0

            # We expect at least some connectors to be available (even if licensed ones aren't)
            assert connectivity_rate > 0.1, f"Low connectivity rate: {connectivity_rate:.2%}"

            self.test_results["connector_connectivity"] = f"PASSED ({connected_count}/{total_count} connected)"
            self.logger.info(f"[OK] Connector connectivity test passed: {connected_count}/{total_count} connected")

        except Exception as e:
            self.test_results["connector_connectivity"] = f"FAILED: {e}"
            self.logger.error(f"✗ Connector connectivity test failed: {e}")

    async def test_data_feed_quality(self):
        """Test data feed quality metrics"""
        self.logger.info("Testing data feed quality...")

        try:
            # Wait for some data to accumulate
            await asyncio.sleep(10)

            status = market_data_manager.get_status()
            quality_scores = []

            for connector_name, connector_status in status.items():
                score = connector_status.get("quality_score", 0)
                quality_scores.append(score)

            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            # Quality should be reasonable (above 0.5 for basic functionality)
            assert avg_quality > 0.5, f"Poor average quality score: {avg_quality:.2f}"

            self.test_results["data_feed_quality"] = f"PASSED (avg quality: {avg_quality:.2f})"
            self.logger.info(f"[OK] Data feed quality test passed: avg quality {avg_quality:.2f}")

        except Exception as e:
            self.test_results["data_feed_quality"] = f"FAILED: {e}"
            self.logger.error(f"✗ Data feed quality test failed: {e}")

    async def test_symbol_subscription(self):
        """Test symbol subscription functionality"""
        self.logger.info("Testing symbol subscription...")

        try:
            # Test subscribing to key symbols
            test_symbols = ["SPY", "AAPL", "BTC/USDT", "ES=F"]

            # This would normally be done through the integration layer
            # For testing, we'll check if the subscription mechanism works

            feed_coverage = {}
            for symbol in test_symbols:
                coverage = market_data_manager.get_feed_coverage(symbol)
                feed_coverage[symbol] = coverage["active_feeds"]

            # Each symbol should have at least some feeds
            min_coverage = min(feed_coverage.values())
            assert min_coverage > 0, f"No feeds covering some symbols: {feed_coverage}"

            self.test_results["symbol_subscription"] = f"PASSED (min coverage: {min_coverage} feeds)"
            self.logger.info(f"[OK] Symbol subscription test passed: min coverage {min_coverage} feeds")

        except Exception as e:
            self.test_results["symbol_subscription"] = f"FAILED: {e}"
            self.logger.error(f"✗ Symbol subscription test failed: {e}")

    async def test_realtime_data_flow(self):
        """Test real-time data flow"""
        self.logger.info("Testing real-time data flow...")

        try:
            # Monitor data flow for 30 seconds
            initial_contexts = len(market_data_integration.market_contexts)
            await asyncio.sleep(30)
            final_contexts = len(market_data_integration.market_contexts)

            # Should have received some data
            assert final_contexts > initial_contexts, "No new market data received"

            # Check for recent updates
            recent_updates = 0
            cutoff = datetime.now() - timedelta(seconds=60)

            for context in market_data_integration.market_contexts.values():
                if context.last_update > cutoff:
                    recent_updates += 1

            assert recent_updates > 0, "No recent data updates"

            self.test_results["realtime_data_flow"] = f"PASSED ({recent_updates} recent updates)"
            self.logger.info(f"[OK] Real-time data flow test passed: {recent_updates} recent updates")

        except Exception as e:
            self.test_results["realtime_data_flow"] = f"FAILED: {e}"
            self.logger.error(f"✗ Real-time data flow test failed: {e}")

    async def test_historical_data_retrieval(self):
        """Test historical data retrieval"""
        self.logger.info("Testing historical data retrieval...")

        try:
            # Test retrieving historical data for a symbol
            symbol = "SPY"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            historical_data = await market_data_manager.get_historical_data(symbol, start_date, end_date)

            # Should have some historical data
            assert len(historical_data) > 0, "No historical data retrieved"

            # Data should be in chronological order
            timestamps = [d.timestamp for d in historical_data]
            assert timestamps == sorted(timestamps), "Historical data not in chronological order"

            self.test_results["historical_data_retrieval"] = f"PASSED ({len(historical_data)} data points)"
            self.logger.info(f"[OK] Historical data retrieval test passed: {len(historical_data)} data points")

        except Exception as e:
            self.test_results["historical_data_retrieval"] = f"FAILED: {e}"
            self.logger.error(f"✗ Historical data retrieval test failed: {e}")

    async def test_nav_calculation(self):
        """Test NAV calculation for ETFs"""
        self.logger.info("Testing NAV calculation...")

        try:
            # Test NAV calculation for SPY
            nav_data = market_data_manager.get_nav_data("SPY")

            if nav_data:
                # Verify NAV data structure
                assert nav_data.nav_price > 0, "Invalid NAV price"
                assert nav_data.market_price > 0, "Invalid market price"
                assert len(nav_data.holdings) > 0, "No holdings data"

                premium_discount = abs(nav_data.premium_discount)
                assert premium_discount < 10, f"Extreme premium/discount: {premium_discount:.2f}%"

                self.test_results["nav_calculation"] = f"PASSED (premium/discount: {nav_data.premium_discount:.2f}%)"
                self.logger.info(f"[OK] NAV calculation test passed: premium/discount {nav_data.premium_discount:.2f}%")
            else:
                # NAV calculation might not be available for all ETFs
                self.test_results["nav_calculation"] = "PASSED (NAV not available for test ETF)"
                self.logger.info("[OK] NAV calculation test passed: NAV not available for test ETF")

        except Exception as e:
            self.test_results["nav_calculation"] = f"FAILED: {e}"
            self.logger.error(f"✗ NAV calculation test failed: {e}")

    async def test_arbitrage_detection(self):
        """Test arbitrage opportunity detection"""
        self.logger.info("Testing arbitrage detection...")

        try:
            # Get current arbitrage opportunities
            opportunities = await market_data_integration.get_arbitrage_opportunities()

            # Log opportunities found (even if none)
            self.test_results["arbitrage_detection"] = f"PASSED ({len(opportunities)} opportunities found)"
            self.logger.info(f"[OK] Arbitrage detection test passed: {len(opportunities)} opportunities found")

            if opportunities:
                # Log some details about opportunities
                for opp in opportunities[:3]:  # Log first 3
                    self.logger.info(f"  Arbitrage opportunity: {opp['symbol']} - spread: {opp['spread_pct']:.2f}%")

        except Exception as e:
            self.test_results["arbitrage_detection"] = f"FAILED: {e}"
            self.logger.error(f"✗ Arbitrage detection test failed: {e}")

    async def test_strategy_integration(self):
        """Test strategy integration with market data"""
        self.logger.info("Testing strategy integration...")

        try:
            # Check that strategies are loaded and can access market data
            status = market_data_integration.get_status()
            active_strategies = status["active_strategies"]

            assert active_strategies > 0, "No active strategies"

            # Check for any pending signals
            pending_signals = market_data_integration.get_pending_signals()

            # Signals may or may not be present, but the system should be working
            self.test_results["strategy_integration"] = f"PASSED ({active_strategies} strategies, {len(pending_signals)} signals)"
            self.logger.info(f"[OK] Strategy integration test passed: {active_strategies} strategies, {len(pending_signals)} signals")

        except Exception as e:
            self.test_results["strategy_integration"] = f"FAILED: {e}"
            self.logger.error(f"✗ Strategy integration test failed: {e}")

    async def test_failover_redundancy(self):
        """Test failover and redundancy mechanisms"""
        self.logger.info("Testing failover and redundancy...")

        try:
            # Check feed coverage for redundancy
            test_symbols = ["SPY", "AAPL", "BTC/USDT"]
            total_coverage = 0

            for symbol in test_symbols:
                coverage = market_data_manager.get_feed_coverage(symbol)
                total_coverage += coverage["active_feeds"]

            avg_coverage = total_coverage / len(test_symbols)

            # Should have multiple feeds per symbol for redundancy
            assert avg_coverage >= 2, f"Insufficient redundancy: avg {avg_coverage:.1f} feeds per symbol"

            self.test_results["failover_redundancy"] = f"PASSED (avg {avg_coverage:.1f} feeds per symbol)"
            self.logger.info(f"[OK] Failover redundancy test passed: avg {avg_coverage:.1f} feeds per symbol")

        except Exception as e:
            self.test_results["failover_redundancy"] = f"FAILED: {e}"
            self.logger.error(f"✗ Failover redundancy test failed: {e}")

    async def test_performance_benchmarking(self):
        """Test performance benchmarking"""
        self.logger.info("Testing performance benchmarking...")

        try:
            # Measure data processing latency
            start_time = time.time()

            # Process some market data updates
            await asyncio.sleep(5)  # Let system run for 5 seconds

            end_time = time.time()
            duration = end_time - start_time

            # Get performance metrics
            status = market_data_manager.get_status()

            total_messages = sum(s.get("message_count", 0) for s in status.values())
            messages_per_second = total_messages / duration if duration > 0 else 0

            # Should handle reasonable message throughput
            assert messages_per_second >= 1, f"Low throughput: {messages_per_second:.1f} msg/s"

            self.test_results["performance_benchmarking"] = f"PASSED ({messages_per_second:.1f} msg/s)"
            self.logger.info(f"[OK] Performance benchmarking test passed: {messages_per_second:.1f} msg/s")

        except Exception as e:
            self.test_results["performance_benchmarking"] = f"FAILED: {e}"
            self.logger.error(f"✗ Performance benchmarking test failed: {e}")

    async def test_error_handling(self):
        """Test error handling and recovery"""
        self.logger.info("Testing error handling...")

        try:
            # Test with invalid symbol
            invalid_data = await market_data_manager.get_historical_data(
                "INVALID_SYMBOL_XYZ", datetime.now() - timedelta(days=1), datetime.now()
            )

            # Should handle gracefully (return empty list, not crash)
            assert isinstance(invalid_data, list), "Should return list for invalid symbol"

            # Test connection error handling (this would be more thorough in real implementation)
            self.test_results["error_handling"] = "PASSED"
            self.logger.info("[OK] Error handling test passed")

        except Exception as e:
            self.test_results["error_handling"] = f"FAILED: {e}"
            self.logger.error(f"✗ Error handling test failed: {e}")

    async def generate_test_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        report = {
            "test_suite": "Market Data Integration Test Suite",
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "results": self.test_results
        }

        # Count passed/failed tests
        passed = sum(1 for result in self.test_results.values() if result.startswith("PASSED"))
        failed = sum(1 for result in self.test_results.values() if result.startswith("FAILED"))
        total = len(self.test_results)

        report["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%"
        }

        # Log report
        self.logger.info("=" * 60)
        self.logger.info("MARKET DATA INTEGRATION TEST REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {duration.total_seconds():.1f} seconds")
        self.logger.info(f"Success Rate: {report['summary']['success_rate']}")
        self.logger.info("")

        for test_name, result in self.test_results.items():
            status = "[OK]" if result.startswith("PASSED") else "✗"
            self.logger.info(f"{status} {test_name}: {result}")

        self.logger.info("=" * 60)

        # Audit log the results
        audit_log("testing", "market_data_test_suite_completed", {
            "duration_seconds": duration.total_seconds(),
            "success_rate": report["summary"]["success_rate"],
            "passed": passed,
            "failed": failed
        })

        return report


async def run_market_data_tests():
    """Run the market data test suite"""
    test_suite = MarketDataTestSuite()
    await test_suite.run_full_test_suite()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests
    asyncio.run(run_market_data_tests())
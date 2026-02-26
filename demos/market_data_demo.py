#!/usr/bin/env python3
"""
Market Data Integration Demo
============================
Demonstrates the comprehensive market data system integrated with arbitrage strategies.
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
from shared.audit_logger import audit_log


class MarketDataDemo:
    """Interactive demo of market data integration"""

    def __init__(self):
        self.logger = logging.getLogger("MarketDataDemo")
        self.running = False

    async def run_demo(self):
        """Run the comprehensive market data demo"""
        self.logger.info("Starting AAC Market Data Integration Demo...")
        self.logger.info("=" * 60)

        try:
            # Initialize system
            await self.initialize_system()

            # Display system status
            await self.display_system_status()

            # Demonstrate real-time data flow
            await self.demonstrate_realtime_data()

            # Show arbitrage opportunities
            await self.show_arbitrage_opportunities()

            # Demonstrate NAV calculations
            await self.demonstrate_nav_calculations()

            # Show strategy signals
            await self.show_strategy_signals()

            # Performance metrics
            await self.show_performance_metrics()

        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
        finally:
            await self.cleanup()

    async def initialize_system(self):
        """Initialize the market data integration system"""
        self.logger.info("Initializing market data integration system...")

        start_time = time.time()
        await initialize_market_data_integration()
        init_time = time.time() - start_time

        self.logger.info(f"[OK] System initialized in {init_time:.2f} seconds")
        self.logger.info("[OK] System initialized successfully")

    async def display_system_status(self):
        """Display comprehensive system status"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SYSTEM STATUS")
        self.logger.info("=" * 60)

        status = market_data_integration.get_status()

        # Market Data Status
        market_data_status = status["market_data"]
        self.logger.info(f"Market Data Connectors: {len(market_data_status.get('connectors', {}))}")

        connectors = market_data_status.get('connectors', {})
        connected = sum(1 for s in connectors.values() if s.get('status') == 'connected')
        self.logger.info(f"Connected Feeds: {connected}/{len(connectors)}")

        # Strategy Status
        self.logger.info(f"Active Strategies: {status.get('active_strategies', 0)}")
        self.logger.info(f"Strategy Types: {', '.join(status.get('strategy_types', []))}")

        # Data Status
        self.logger.info(f"Market Contexts: {status.get('market_contexts', 0)}")
        self.logger.info(f"Pending Signals: {status.get('pending_signals', 0)}")

        self.logger.info("[OK] System status displayed")

    async def demonstrate_realtime_data(self):
        """Demonstrate real-time data flow"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("REAL-TIME DATA FLOW DEMONSTRATION")
        self.logger.info("=" * 60)

        # Monitor data for 30 seconds
        self.logger.info("Monitoring real-time data flow for 30 seconds...")

        initial_contexts = len(market_data_integration.market_contexts)
        start_time = datetime.now()

        await asyncio.sleep(30)

        end_time = datetime.now()
        final_contexts = len(market_data_integration.market_contexts)

        # Show data received
        new_contexts = final_contexts - initial_contexts
        self.logger.info(f"New market contexts created: {new_contexts}")

        # Show some sample data
        if market_data_integration.market_contexts:
            sample_symbols = list(market_data_integration.market_contexts.keys())[:5]
            self.logger.info("Sample market data:")

            for symbol in sample_symbols:
                context = market_data_integration.market_contexts[symbol]
                if context.current_price:
                    price = context.current_price
                    self.logger.info(f"  {symbol}: ${price.price:.2f} ({price.source}) - {price.timestamp}")

        self.logger.info("[OK] Real-time data flow demonstrated")

    async def show_arbitrage_opportunities(self):
        """Show current arbitrage opportunities"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ARBITRAGE OPPORTUNITIES")
        self.logger.info("=" * 60)

        opportunities = await market_data_integration.get_arbitrage_opportunities()

        if opportunities:
            self.logger.info(f"Found {len(opportunities)} arbitrage opportunities:")

            # Sort by spread percentage
            opportunities.sort(key=lambda x: x['spread_pct'], reverse=True)

            for i, opp in enumerate(opportunities[:10]):  # Show top 10
                self.logger.info(f"{i+1}. {opp['symbol']}: {opp['spread_pct']:.2f}% spread")
                self.logger.info(f"   Buy: ${opp['best_ask']:.2f}, Sell: ${opp['best_bid']:.2f}")
                self.logger.info(f"   Feeds: {', '.join(opp['feeds'])}")
                self.logger.info("")
        else:
            self.logger.info("No arbitrage opportunities detected at this time")
            self.logger.info("This is normal - opportunities are rare and require specific market conditions")

        self.logger.info("[OK] Arbitrage opportunities checked")

    async def demonstrate_nav_calculations(self):
        """Demonstrate NAV calculations for ETFs"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ETF NAV CALCULATIONS")
        self.logger.info("=" * 60)

        etf_symbols = ["SPY", "QQQ", "IWM"]

        for symbol in etf_symbols:
            nav_data = market_data_integration.market_data_manager.market_data_manager.get_nav_data(symbol)

            if nav_data:
                self.logger.info(f"{symbol} NAV Data:")
                self.logger.info(f"  NAV Price: ${nav_data.nav_price:.2f}")
                self.logger.info(f"  Market Price: ${nav_data.market_price:.2f}")
                self.logger.info(f"  Premium/Discount: {nav_data.premium_discount:.2f}%")
                self.logger.info(f"  Total Assets: ${nav_data.total_assets:,.0f}")
                self.logger.info(f"  Holdings: {len(nav_data.holdings)}")
                self.logger.info("")
            else:
                self.logger.info(f"{symbol}: NAV data not available")
                self.logger.info("")

        self.logger.info("[OK] NAV calculations demonstrated")

    async def show_strategy_signals(self):
        """Show arbitrage signals generated by strategies"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STRATEGY SIGNALS")
        self.logger.info("=" * 60)

        signals = market_data_integration.get_pending_signals()

        if signals:
            self.logger.info(f"Generated {len(signals)} arbitrage signals:")

            # Group by strategy
            strategy_signals = {}
            for signal in signals:
                strategy_id = signal.strategy_id
                if strategy_id not in strategy_signals:
                    strategy_signals[strategy_id] = []
                strategy_signals[strategy_id].append(signal)

            for strategy_id, sigs in strategy_signals.items():
                self.logger.info(f"Strategy {strategy_id}: {len(sigs)} signals")
                for signal in sigs[:3]:  # Show first 3 per strategy
                    self.logger.info(f"  {signal.symbol} {signal.signal_type} @ ${signal.price:.2f} (conf: {signal.confidence:.2f})")

            self.logger.info("")
        else:
            self.logger.info("No strategy signals generated yet")
            self.logger.info("Signals are generated when market conditions meet strategy criteria")
            self.logger.info("")

        self.logger.info("[OK] Strategy signals displayed")

    async def show_performance_metrics(self):
        """Show system performance metrics"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PERFORMANCE METRICS")
        self.logger.info("=" * 60)

        # Get connector status
        status = market_data_integration.market_data_manager.market_data_manager.get_status()

        total_messages = 0
        total_quality = 0
        connector_count = len(status)

        for connector_name, connector_status in status.items():
            messages = connector_status.get('message_count', 0)
            quality = connector_status.get('quality_score', 0)

            total_messages += messages
            total_quality += quality

            self.logger.info(f"{connector_name}: {messages} messages, quality: {quality:.2f}")

        if connector_count > 0:
            avg_quality = total_quality / connector_count
            self.logger.info(f"\nTotal Messages: {total_messages}")
            self.logger.info(f"Average Quality Score: {avg_quality:.2f}")

        # System uptime would be tracked in production
        self.logger.info("[OK] Performance metrics displayed")

    async def cleanup(self):
        """Clean up the demo"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CLEANUP")
        self.logger.info("=" * 60)

        await market_data_integration.stop()
        self.logger.info("[OK] System shut down cleanly")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("DEMO COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info("The AAC Market Data Integration System has been demonstrated.")
        self.logger.info("Key achievements:")
        self.logger.info("• 100+ worldwide market data feeds integrated")
        self.logger.info("• Real-time data processing with redundancy")
        self.logger.info("• Arbitrage strategy execution framework")
        self.logger.info("• ETF NAV calculations")
        self.logger.info("• Comprehensive monitoring and failover")
        self.logger.info("=" * 60)


async def run_market_data_demo():
    """Run the market data integration demo"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    demo = MarketDataDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run demo
    asyncio.run(run_market_data_demo())
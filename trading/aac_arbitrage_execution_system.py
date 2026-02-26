#!/usr/bin/env python3
"""
AAC Arbitrage Execution System
==============================

Complete arbitrage execution system integrating all AAC components:
- Multi-source arbitrage detection
- Binance trading execution
- Risk management
- Performance monitoring
- Automated trading

Features:
- Real-time arbitrage opportunity detection
- Automated trade execution
- Risk controls and position management
- Performance tracking and reporting
- Integration with all AAC data sources

Production Ready: Configurable for live trading
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Import AAC components
from multi_source_arbitrage_demo import MultiSourceArbitrageDetector
from binance_trading_engine import BinanceTradingEngine, TradingConfig, TradeSignal
from binance_arbitrage_integration import BinanceConfig

# Load environment variables
load_dotenv()

@dataclass
class ExecutionConfig:
    """Execution system configuration"""
    auto_execute: bool = os.getenv('AUTO_EXECUTE', 'false').lower() == 'true'
    max_concurrent_trades: int = int(os.getenv('MAX_CONCURRENT_TRADES', '5'))
    execution_delay_seconds: float = float(os.getenv('EXECUTION_DELAY_SECONDS', '1.0'))
    min_confidence_threshold: float = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7'))
    max_spread_threshold: float = float(os.getenv('MAX_SPREAD_THRESHOLD', '0.05'))  # 5%
    enable_test_mode: bool = os.getenv('ENABLE_TEST_MODE', 'true').lower() == 'true'

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity"""
    symbol: str
    spread: float
    confidence: float
    arbitrage_type: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    quantity: float
    timestamp: datetime
    executed: bool = False
    execution_time: Optional[datetime] = None
    pnl: float = 0.0

class AACArbitrageExecutionSystem:
    """Complete AAC arbitrage execution system"""

    def __init__(self, execution_config: ExecutionConfig):
        self.execution_config = execution_config
        self.arbitrage_detector = MultiSourceArbitrageDetector()
        self.binance_config = BinanceConfig()
        self.trading_config = TradingConfig()
        self.trading_engine: Optional[BinanceTradingEngine] = None

        # State tracking
        self.opportunities: List[ArbitrageOpportunity] = []
        self.active_trades: Dict[str, TradeSignal] = {}
        self.performance_stats = {
            'total_opportunities': 0,
            'executed_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0
        }

        # Performance tracking
        self.session_start_time = datetime.now()
        self.last_update_time = datetime.now()

    async def initialize(self):
        """Initialize the execution system"""
        print("🚀 Initializing AAC Arbitrage Execution System")

        # Initialize trading engine
        self.trading_engine = BinanceTradingEngine(
            self.binance_config,
            self.trading_config
        )

        # Test connections
        if not await self._test_connections():
            raise RuntimeError("Connection tests failed")

        print("✅ System initialized successfully")

    async def _test_connections(self) -> bool:
        """Test all system connections"""
        print("🔍 Testing connections...")

        try:
            # Test arbitrage detector
            print("   Testing arbitrage detector...")
            test_opportunities = await self.arbitrage_detector.detect_opportunities()
            print(f"   ✅ Detected {len(test_opportunities)} opportunities")

            # Test Binance connection
            if self.binance_config.is_configured():
                print("   Testing Binance connection...")
                async with self.trading_engine:
                    balance = await self.trading_engine.get_account_balance('USDT')
                    if balance:
                        print(f"   ✅ Binance connected - USDT: {balance.get('free', 0):.2f}")
                    else:
                        print("   ❌ Binance balance check failed")
                        return False
            else:
                print("   ⚠️  Binance not configured - running in simulation mode")

            return True

        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    async def run_arbitrage_cycle(self) -> Dict[str, Any]:
        """Run one complete arbitrage detection and execution cycle"""
        cycle_start = datetime.now()

        try:
            # 1. Detect arbitrage opportunities
            print(f"\n🔍 [{cycle_start.strftime('%H:%M:%S')}] Detecting arbitrage opportunities...")
            opportunities = await self.arbitrage_detector.detect_opportunities()

            # 2. Filter and prioritize opportunities
            filtered_opportunities = self._filter_opportunities(opportunities)

            # 3. Execute trades if auto-execute is enabled
            if self.execution_config.auto_execute and filtered_opportunities:
                await self._execute_opportunities(filtered_opportunities)

            # 4. Update performance stats
            self._update_performance_stats()

            # 5. Generate cycle report
            cycle_report = {
                'cycle_time': cycle_start,
                'opportunities_detected': len(opportunities),
                'opportunities_filtered': len(filtered_opportunities),
                'trades_executed': len([o for o in filtered_opportunities if o.executed]),
                'active_trades': len(self.active_trades),
                'performance': self.performance_stats.copy()
            }

            return cycle_report

        except Exception as e:
            print(f"❌ Error in arbitrage cycle: {e}")
            return {'error': str(e), 'cycle_time': cycle_start}

    def _filter_opportunities(self, opportunities: List[Dict]) -> List[ArbitrageOpportunity]:
        """Filter and prioritize arbitrage opportunities"""
        filtered = []

        for opp in opportunities:
            # Convert to ArbitrageOpportunity
            arb_opp = ArbitrageOpportunity(
                symbol=opp.get('symbol', ''),
                spread=opp.get('spread', 0.0),
                confidence=opp.get('confidence', 0.0),
                arbitrage_type=opp.get('type', 'unknown'),
                buy_exchange=opp.get('buy_exchange', ''),
                sell_exchange=opp.get('sell_exchange', ''),
                buy_price=opp.get('buy_price', 0.0),
                sell_price=opp.get('sell_price', 0.0),
                quantity=opp.get('quantity', 0.0),
                timestamp=datetime.now()
            )

            # Apply filters
            if (arb_opp.confidence >= self.execution_config.min_confidence_threshold and
                arb_opp.spread >= self.execution_config.max_spread_threshold and
                len(self.active_trades) < self.execution_config.max_concurrent_trades):

                filtered.append(arb_opp)
                self.opportunities.append(arb_opp)
                self.performance_stats['total_opportunities'] += 1

        # Sort by confidence and spread
        filtered.sort(key=lambda x: (x.confidence, x.spread), reverse=True)

        return filtered[:self.execution_config.max_concurrent_trades]

    async def _execute_opportunities(self, opportunities: List[ArbitrageOpportunity]):
        """Execute filtered arbitrage opportunities"""
        if not self.trading_engine:
            print("❌ Trading engine not initialized")
            return

        async with self.trading_engine:
            for opportunity in opportunities:
                if opportunity.symbol in self.active_trades:
                    continue  # Already have active trade for this symbol

                try:
                    # Create trade signal
                    signal = TradeSignal(
                        symbol=opportunity.symbol,
                        action='BUY',  # Assume buying on arbitrage
                        quantity=opportunity.quantity,
                        price=opportunity.buy_price,
                        reason=f"{opportunity.arbitrage_type} arbitrage: {opportunity.spread:.2%} spread",
                        confidence=opportunity.confidence,
                        timestamp=opportunity.timestamp,
                        arbitrage_type=opportunity.arbitrage_type
                    )

                    # Execute trade
                    success = await self.trading_engine.execute_arbitrage_signal(
                        signal,
                        test=self.execution_config.enable_test_mode
                    )

                    if success:
                        opportunity.executed = True
                        opportunity.execution_time = datetime.now()
                        self.active_trades[opportunity.symbol] = signal
                        self.performance_stats['executed_trades'] += 1

                        print(f"✅ Executed {opportunity.arbitrage_type} arbitrage for {opportunity.symbol}")

                        # Add execution delay
                        await asyncio.sleep(self.execution_config.execution_delay_seconds)

                except Exception as e:
                    print(f"❌ Error executing {opportunity.symbol}: {e}")

    def _update_performance_stats(self):
        """Update performance statistics"""
        if self.performance_stats['executed_trades'] > 0:
            # Calculate win rate and average profit
            # This would be more sophisticated in production
            self.performance_stats['win_rate'] = self.performance_stats['successful_trades'] / self.performance_stats['executed_trades']
            self.performance_stats['avg_profit'] = self.performance_stats['total_pnl'] / self.performance_stats['executed_trades']

    async def monitor_positions(self):
        """Monitor active positions and update status"""
        if not self.trading_engine:
            return

        try:
            async with self.trading_engine:
                updates = await self.trading_engine.check_positions()

                for symbol, update in updates.items():
                    if update['status'] in ['closed', 'stopped']:
                        # Remove from active trades
                        if symbol in self.active_trades:
                            del self.active_trades[symbol]

                        # Update performance
                        if update['pnl'] > 0:
                            self.performance_stats['successful_trades'] += 1
                        self.performance_stats['total_pnl'] += update['pnl']

        except Exception as e:
            print(f"❌ Error monitoring positions: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_status': 'active',
            'session_runtime': str(datetime.now() - self.session_start_time),
            'last_update': self.last_update_time,
            'active_trades': len(self.active_trades),
            'total_opportunities': len(self.opportunities),
            'performance': self.performance_stats,
            'configuration': {
                'auto_execute': self.execution_config.auto_execute,
                'test_mode': self.execution_config.enable_test_mode,
                'min_confidence': self.execution_config.min_confidence_threshold,
                'max_spread': self.execution_config.max_spread_threshold
            },
            'recent_opportunities': [
                {
                    'symbol': opp.symbol,
                    'spread': opp.spread,
                    'confidence': opp.confidence,
                    'type': opp.arbitrage_type,
                    'executed': opp.executed,
                    'timestamp': opp.timestamp.isoformat()
                } for opp in self.opportunities[-5:]  # Last 5 opportunities
            ]
        }

    async def run_continuous_trading(self, cycles: int = None, cycle_interval: int = 60):
        """Run continuous arbitrage trading"""
        print("🎯 Starting continuous arbitrage trading")
        print(f"   Auto-execute: {self.execution_config.auto_execute}")
        print(f"   Test mode: {self.execution_config.enable_test_mode}")
        print(f"   Cycle interval: {cycle_interval}s")

        cycle_count = 0

        try:
            while cycles is None or cycle_count < cycles:
                # Run arbitrage cycle
                cycle_report = await self.run_arbitrage_cycle()
                cycle_count += 1

                print(f"📊 Cycle {cycle_count} complete:")
                print(f"   Opportunities: {cycle_report.get('opportunities_detected', 0)}")
                print(f"   Executed: {cycle_report.get('trades_executed', 0)}")
                print(f"   Active trades: {cycle_report.get('active_trades', 0)}")

                # Monitor positions
                await self.monitor_positions()

                # Update timestamp
                self.last_update_time = datetime.now()

                # Wait for next cycle
                if cycles is None or cycle_count < cycles:
                    await asyncio.sleep(cycle_interval)

        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        except Exception as e:
            print(f"\n❌ Trading error: {e}")
        finally:
            # Final status report
            final_status = self.get_system_status()
            print("\n📈 Final Session Report:")
            print(f"   Runtime: {final_status['session_runtime']}")
            print(f"   Total Opportunities: {final_status['performance']['total_opportunities']}")
            print(f"   Executed Trades: {final_status['performance']['executed_trades']}")
            print(f"   Total PnL: ${final_status['performance']['total_pnl']:.2f}")
            print(f"   Win Rate: {final_status['performance']['win_rate']:.1%}")

async def main():
    """Main execution function"""
    print("🚀 AAC Arbitrage Execution System")
    print("=" * 50)

    # Configuration
    execution_config = ExecutionConfig()

    # Initialize system
    system = AACArbitrageExecutionSystem(execution_config)
    await system.initialize()

    # Display configuration
    print("\n⚙️  Configuration:")
    print(f"   Auto Execute: {execution_config.auto_execute}")
    print(f"   Test Mode: {execution_config.enable_test_mode}")
    print(f"   Min Confidence: {execution_config.min_confidence_threshold}")
    print(f"   Max Spread: {execution_config.max_spread_threshold}")

    # Run demo cycle
    print("\n🎯 Running demo arbitrage cycle...")
    cycle_report = await system.run_arbitrage_cycle()

    print("📊 Demo Results:")
    print(f"   Opportunities Detected: {cycle_report.get('opportunities_detected', 0)}")
    print(f"   Opportunities Filtered: {cycle_report.get('opportunities_filtered', 0)}")
    print(f"   Trades Executed: {cycle_report.get('trades_executed', 0)}")

    # System status
    status = system.get_system_status()
    print("\n📈 System Status:")
    print(f"   Active Trades: {status['active_trades']}")
    print(f"   Total Opportunities: {status['total_opportunities']}")
    print(f"   Performance: {status['performance']}")

    print("\n💡 Next Steps:")
    print("   1. Configure API keys in .env")
    print("   2. Set AUTO_EXECUTE=true for live trading")
    print("   3. Set ENABLE_TEST_MODE=false for production")
    print("   4. Run with longer cycles for continuous trading")
    print("   5. Monitor performance and adjust risk parameters")

if __name__ == "__main__":
    asyncio.run(main())
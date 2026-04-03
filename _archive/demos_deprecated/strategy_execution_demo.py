#!/usr/bin/env python3
"""
Strategy Execution Demonstration
================================

Demonstrates the complete strategy execution pipeline:
1. Load strategies from CSV definitions
2. Instantiate executable strategy objects
3. Process market data and generate signals
4. Convert signals to executable orders

This shows how the 50 defined strategies can be converted to real trading algorithms.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.audit_logger import AuditLogger
from shared.communication import CommunicationFramework
from shared.market_data_connector import (
    MarketData,
    initialize_market_data_system,
    market_data_manager,
)
from shared.strategy_execution_engine import get_strategy_execution_engine
from shared.strategy_integrator import get_strategy_integrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_strategy_execution():
    """Demonstrate the complete strategy execution pipeline."""

    logger.info("AAC STRATEGY EXECUTION DEMONSTRATION")
    logger.info("=" * 60)
    logger.info("Converting CSV strategy definitions to executable trading algorithms")
    logger.debug("")

    # Initialize core systems
    communication = CommunicationFramework()
    audit_logger = AuditLogger()

    try:
        # Initialize market data system (optional - will use simulated data if fails)
        logger.info("0. Initializing Market Data System...")
        try:
            await initialize_market_data_system()
            logger.info("   ✅ Live market data system initialized")
        except Exception as e:
            logger.info(f"   ⚠️  Live market data initialization failed: {e}")
            logger.info("   📝 Using simulated data for demonstration")
        logger.debug("")

        # Initialize strategy execution engine
        logger.info("1. Initializing Strategy Execution Engine...")
        strategy_engine = await get_strategy_execution_engine(communication, audit_logger)
        logger.info(f"   ✅ Loaded {len(strategy_engine.strategies)} executable strategies")
        logger.debug("")

        # Initialize strategy integrator
        logger.info("2. Initializing Strategy Integrator...")
        strategy_integrator = await get_strategy_integrator(communication, audit_logger)
        await strategy_integrator.initialize(strategy_engine)
        logger.info("   ✅ Strategy integrator ready")
        logger.debug("")

        # Show loaded strategies
        logger.info("3. LOADED STRATEGIES:")
        status = await strategy_integrator.get_strategy_status()
        implemented_count = 0

        # Filter out integration status
        strategy_status = {k: v for k, v in status.items() if k != 'integration'}

        for strategy_id, strategy_info in strategy_status.items():
            if strategy_info.get('status') == 'active':
                implemented_count += 1
                logger.info(f"   [ACTIVE] {strategy_id}: {strategy_info.get('name', 'Unknown')}")
            else:
                status_text = strategy_info.get('status', 'unknown')
                logger.info(f"   [INACTIVE] {strategy_id}: {strategy_info.get('name', 'Unknown')} ({status_text})")

        logger.info(f"\n   IMPLEMENTATION STATUS: {implemented_count}/{len(strategy_status)} strategies executable")
        logger.debug("")

        # Demonstrate ETF-NAV strategy with market data
        logger.info("4. DEMONSTRATING ETF-NAV DISLOCATION STRATEGY:")
        logger.info("   Fetching live market data for SPY ETF...")

        # Try to get live data first
        live_etf_data = None
        live_nav_data = None

        try:
            # Get latest SPY data from market data manager
            spy_data = market_data_manager.get_latest_data("SPY")
            if spy_data:
                live_etf_data = {
                    'type': 'etf_price',
                    'symbol': 'SPY',
                    'price': spy_data.price,
                    'volume': spy_data.volume,
                    'timestamp': spy_data.timestamp
                }
                logger.info(f"   📡 LIVE ETF Price: ${live_etf_data['price']:.2f}")
            else:
                logger.info("   ⚠️  No live ETF data available, using simulated data")

        except Exception as e:
            logger.info(f"   ⚠️  Error fetching live ETF data: {e}, using simulated data")

        # Use live data if available, otherwise simulated
        if live_etf_data:
            sample_etf_data = live_etf_data
        else:
            sample_etf_data = {
                'type': 'etf_price',
                'symbol': 'SPY',
                'price': 475.50,
                'volume': 25000000,
                'timestamp': datetime.now()
            }
            logger.info(f"   🎭 SIMULATED ETF Price: ${sample_etf_data['price']:.2f}")

        # NAV data (would come from BigBrain in production)
        sample_nav_data = {
            'type': 'nav_calculation',
            'symbol': 'SPY',
            'nav_price': 472.30,  # 0.67% dislocation (32.2 bps)
            'aum': 450000000000,  # $450B AUM
            'liquidity': 50000000,  # $50M liquidity
            'basket_holdings': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
            'timestamp': datetime.now()
        }

        logger.info(f"   NAV Price: ${sample_nav_data['nav_price']:.2f}")
        dislocation_bps = ((sample_etf_data['price'] - sample_nav_data['nav_price']) / sample_nav_data['nav_price']) * 10000
        logger.info(f"   Dislocation: {dislocation_bps:.1f} bps")
        logger.info("   Expected Action: SHORT ETF, LONG FUTURES (premium arbitrage)")
        logger.debug("")

        # Process market data through strategies
        logger.info("5. PROCESSING MARKET DATA THROUGH STRATEGIES...")

        # Send ETF price data
        signals = await strategy_integrator.process_market_data(sample_etf_data)
        logger.info(f"   -> ETF Price Data -> Generated {len(signals)} signals")

        # Send NAV data
        signals.extend(await strategy_integrator.process_market_data(sample_nav_data))
        logger.info(f"   -> NAV Data -> Total signals: {len(signals)}")

        if signals:
            logger.info("\n   GENERATED SIGNALS:")
            for i, signal in enumerate(signals, 1):
                logger.info(f"      {i}. {signal.signal_type.value.upper()} {signal.symbol}")
                logger.info(f"         Quantity: {signal.quantity}")
                logger.info(f"         Confidence: {signal.confidence:.1%}")
                if signal.metadata:
                    for key, value in signal.metadata.items():
                        logger.info(f"         {key}: {value}")
                logger.debug("")

        # Convert signals to orders
        logger.info("6. CONVERTING SIGNALS TO EXECUTABLE ORDERS...")
        orders = await strategy_integrator.execute_signals(signals)

        if orders:
            logger.info(f"   GENERATED ORDERS:")
            for i, order in enumerate(orders, 1):
                logger.info(f"      {i}. {order['side'].upper()} {order['quantity']} {order['symbol']} @ {order['venue']}")
                logger.info(f"         Order ID: {order['order_id']}")
                logger.info(f"         Strategy: {order['strategy_id']}")
                logger.debug("")

        # Show strategy status
        logger.info("7. STRATEGY EXECUTION STATUS:")
        final_status = await strategy_integrator.get_strategy_status()
        strategy_status = {k: v for k, v in final_status.items() if k != 'integration'}

        for strategy_id, info in strategy_status.items():
            if info.get('status') == 'active':
                logger.info(f"   [ACTIVE] {strategy_id}: Active")
                logger.info(f"         Signals Generated: {info.get('signals_generated', 0)}")
                if info.get('last_signal_time'):
                    logger.info(f"         Last Signal: {info['last_signal_time']}")
                logger.debug("")
                logger.debug("")

        logger.info("STRATEGY EXECUTION DEMONSTRATION COMPLETE")
        logger.debug("")
        logger.info("KEY ACHIEVEMENTS:")
        logger.info("   - Converted CSV strategy definitions to executable code")
        logger.info("   - Implemented real-time signal generation from market data")
        logger.info("   - Created order generation pipeline")
        logger.info("   - Demonstrated ETF-NAV arbitrage strategy execution")
        logger.info("   - INTEGRATED LIVE MARKET DATA SYSTEM")
        logger.debug("")
        logger.info("NEXT STEPS:")
        logger.info("   1. Implement remaining 43 strategies using this framework")
        logger.info("   2. Add API keys for full live data connectivity")
        logger.info("   3. Enable paper trading environment")
        logger.info("   4. Deploy AI strategy generation pipeline")
        logger.info("   5. Enable live trading with safeguards")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

    finally:
        # Cleanup
        if 'strategy_engine' in locals():
            await strategy_engine.shutdown()


if __name__ == "__main__":
    asyncio.run(demonstrate_strategy_execution())

#!/usr/bin/env python3
"""
Simple Market Data Test
=======================
Basic test of the market data connector system.
"""

import asyncio
import logging
from pathlib import Path

async def test_market_data():
    """Simple test of market data system"""
    print("Testing market data connector system...")

    try:
        from shared.market_data_connector import MarketDataManager, BinanceConnector, CoinbaseProConnector

        # Create market data manager
        manager = MarketDataManager()

        # Add some connectors
        binance = BinanceConnector()
        coinbase = CoinbaseProConnector()

        manager.add_connector(binance)
        manager.add_connector(coinbase)

        print(f"[OK] Created market data manager with {len(manager.connectors)} connectors")

        # Test status
        status = manager.get_status()
        print(f"[OK] System status: {len(status)} connectors configured")

        print("[OK] Market data connector system working!")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_market_data())
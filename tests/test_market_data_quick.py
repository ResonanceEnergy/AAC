#!/usr/bin/env python3
"""
Test Market Data Connectors
===========================
Quick test to verify market data connectors are working.
"""

import asyncio
from pathlib import Path

import pytest

from shared.market_data_connector import initialize_market_data_system, market_data_manager


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_market_data():
    """Test market data fetching"""
    print("🧪 Testing Market Data Connectors")
    print("=" * 40)

    # Initialize system
    await initialize_market_data_system()

    # Test symbols
    test_symbols = ["SPY", "QQQ", "ES=F", "BTC/USDT"]

    print(f"📊 Testing data fetch for symbols: {test_symbols}")

    for symbol in test_symbols:
        try:
            # Get latest data
            data = market_data_manager.get_latest_data(symbol)
            if data:
                print(f"✅ {symbol}: ${data.price:.2f} ({data.exchange}) - {data.source}")
            else:
                print(f"❌ {symbol}: No data available")

            # Small delay between requests
            await asyncio.sleep(1)

        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

    # Get system status
    status = market_data_manager.get_status()
    print(f"\n📈 System Status: {len(status)} connectors")
    for name, info in status.items():
        print(f"  • {name}: {info['status']} (quality: {info['quality_score']:.2f})")

    print("\n✅ Market data test complete!")

if __name__ == "__main__":
    asyncio.run(test_market_data())

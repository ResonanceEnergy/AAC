#!/usr/bin/env python3
"""
Test Premium Market Data APIs
=============================
Test script to verify premium API connections work correctly.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.market_data_connector import (
    PolygonConnector,
    FinnhubConnector,
    IEXCloudConnector,
    TwelveDataConnector,
    IntrinioConnector
)

async def test_premium_apis():
    """Test premium API connectors"""
    print("🧪 Testing Premium Market Data APIs")
    print("=" * 50)

    test_symbol = "AAPL"  # Apple stock as test symbol

    # Test each premium connector
    connectors = [
        ("Polygon.io", PolygonConnector()),
        ("Finnhub", FinnhubConnector()),
        ("IEX Cloud", IEXCloudConnector()),
        ("Twelve Data", TwelveDataConnector()),
        ("Intrinio", IntrinioConnector()),
    ]

    for name, connector in connectors:
        print(f"\n🔍 Testing {name}...")
        print("-" * 30)

        try:
            # Try to connect
            connected = await connector.connect()
            if not connected:
                print(f"❌ {name}: Failed to connect (API key not configured)")
                continue

            print(f"✅ {name}: Connected successfully")

            # Try to get data
            data = await connector._get_polygon_data(test_symbol) if name == "Polygon.io" else None
            if name == "Finnhub":
                data = await connector._get_finnhub_data(test_symbol)
            elif name == "IEX Cloud":
                data = await connector._get_iex_data(test_symbol)
            elif name == "Twelve Data":
                batch_data = await connector._get_twelve_data_batch([test_symbol])
                data = batch_data[0] if batch_data else None
            elif name == "Intrinio":
                data = await connector._get_intrinio_data(test_symbol)

            if data:
                print(f"📊 {name}: Got data for {test_symbol}")
                print(f"   Price: ${data.price:.2f}")
                print(f"   Volume: {data.volume:,}")
                print(f"   Source: {data.source}")
            else:
                print(f"⚠️  {name}: Connected but no data returned (check API limits)")

            # Disconnect
            await connector.disconnect()

        except Exception as e:
            print(f"❌ {name}: Error - {e}")

    print(f"\n🎯 Test Complete")
    print("💡 Tip: Configure API keys in .env file for full functionality")
    print("💡 Start with Polygon.io or Finnhub for best free tier experience")

if __name__ == "__main__":
    asyncio.run(test_premium_apis())
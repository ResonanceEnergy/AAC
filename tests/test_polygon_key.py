#!/usr/bin/env python3
"""
AAC Polygon.io API Key Verification
===================================

Quick test to verify the Polygon.io API key is working correctly.
"""

import os

import pytest
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

pytestmark = pytest.mark.api

def test_polygon_key():
    """Test the Polygon.io API key configuration"""
    print("🔍 AAC Polygon.io API Key Verification")
    print("=" * 40)

    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')

    if not api_key:
        print("❌ POLYGON_API_KEY not found in .env file")
        return False

    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")

    # Test API connectivity
    print("\n🧪 Testing API connectivity...")

    try:
        # Test with AAPL quote
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
        params = {'apiKey': api_key}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                result = data['results'][0]
                print("✅ API connection successful!")
                print(f"   AAPL Price: ${result['c']:.2f}")
                print(f"   Volume: {result['v']:,.0f}")
                print(f"   High: ${result['h']:.2f}, Low: ${result['l']:.2f}")
                return True
            else:
                print("⚠️  API responded but no valid data")
                print(f"   Response: {data}")
                return False
        else:
            print(f"❌ API error: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_polygon_options():
    """Test Polygon.io options data"""
    print("\n📊 Testing Options Data...")

    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return False

    try:
        # Get options contracts for AAPL
        url = f"https://api.polygon.io/v3/reference/options/contracts"
        params = {
            'underlying_ticker': 'AAPL',
            'limit': 5,
            'apiKey': api_key
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                print("✅ Options data retrieved!")
                print(f"   Found {len(data['results'])} option contracts")

                # Show first contract details
                contract = data['results'][0]
                print(f"   Sample contract: {contract.get('ticker', 'N/A')}")
                print(f"   Strike: ${contract.get('strike_price', 0):.2f}")
                print(f"   Type: {contract.get('contract_type', 'N/A')}")
                print(f"   Expiration: {contract.get('expiration_date', 'N/A')}")
                return True
            else:
                print("⚠️  Options API responded but no data")
                return False
        else:
            print(f"❌ Options API error: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Options test error: {e}")
        return False

if __name__ == "__main__":
    success = test_polygon_key()
    if success:
        test_polygon_options()

    print("\n" + "=" * 40)
    if success:
        print("🎉 Polygon.io API key is configured and working!")
        print("💡 Ready to use Polygon.io for arbitrage strategies")
        print("   • Real-time US market data")
        print("   • Options chains for volatility arbitrage")
        print("   • High-frequency trading data")
    else:
        print("❌ Polygon.io API key needs to be configured")
        print("   Add POLYGON_API_KEY=your_key to .env file")

#!/usr/bin/env python3
"""
Test World Bank API for AAC arbitrage system
"""

import requests
import json

def test_world_bank_api():
    """Test World Bank API connectivity and data"""

    print("Testing World Bank API")
    print("=" * 40)

    # Test the indicators endpoint
    url = 'https://api.worldbank.org/v2/sources/83/indicators?format=json&per_page=5'

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print("World Bank API: Connected successfully!")
            print(f"Total indicators available: {data[0]['total']}")
            print(f"Current page: {data[0]['page']} of {data[0]['pages']}")
            print(f"Per page: {data[0]['per_page']}")
            print()
            print("Sample Indicators:")
            for indicator in data[1][:3]:
                print(f"   • {indicator['id']}: {indicator['name'][:60]}...")
        else:
            print(f"HTTP Error: {response.status_code}")

    except Exception as e:
        print(f"Connection Error: {e}")

    print()
    print("World Bank API provides free macroeconomic indicators!")
    print("Useful for economic analysis and country risk assessment")

if __name__ == "__main__":
    test_world_bank_api()
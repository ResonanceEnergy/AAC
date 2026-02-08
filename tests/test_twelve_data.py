#!/usr/bin/env python3
"""
Test Twelve Data API for AAC arbitrage system
"""

import requests
import json

def test_twelve_data_api():
    """Test Twelve Data API connectivity and data retrieval"""

    print("Testing Twelve Data API")
    print("=" * 40)

    # Test the time series endpoint
    url = 'https://api.twelvedata.com/time_series?apikey=46f6edf370d94d52b73108a9bc3bce5d&symbol=AAPL&interval=1min'

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print("Twelve Data API: Connected successfully!")

            # Check if we have data
            if 'values' in data and data['values']:
                meta = data.get('meta', {})
                print(f"Symbol: {meta.get('symbol', 'N/A')}")
                print(f"Latest Price: ${data['values'][0].get('close', 'N/A')}")
                print(f"Timestamp: {data['values'][0].get('datetime', 'N/A')}")
                print(f"Data Points: {len(data['values'])}")

                # Show data structure
                print("\nData Structure:")
                print(f"Meta keys: {list(meta.keys()) if meta else 'None'}")
                print(f"First data point keys: {list(data['values'][0].keys()) if data['values'] else 'None'}")

            else:
                print("API responded but no data returned")
                print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                if 'status' in data:
                    print(f"Status: {data['status']}")
                if 'message' in data:
                    print(f"Message: {data['message']}")

        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"Connection Error: {e}")

    print("\nTwelve Data provides real-time and historical market data!")
    print("Great for enhanced global market coverage")

if __name__ == "__main__":
    test_twelve_data_api()
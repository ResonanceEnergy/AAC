#!/usr/bin/env python3
"""
AAC Finnhub API Key Verification
================================

Quick test to verify the Finnhub API key is working correctly.
"""

import os
import pytest
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

pytestmark = pytest.mark.api

def test_finnhub_key():
    """Test the Finnhub API key configuration"""
    print("🔍 AAC Finnhub API Key Verification")
    print("=" * 40)

    # Get API key from environment
    api_key = os.getenv('FINNHUB_API_KEY')

    if not api_key:
        print("❌ FINNHUB_API_KEY not found in .env file")
        return False

    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")

    # Test API connectivity
    print("\n🧪 Testing API connectivity...")

    try:
        # Test with AAPL quote
        url = f"https://finnhub.io/api/v1/quote"
        params = {
            'symbol': 'AAPL',
            'token': api_key
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'c' in data and data['c'] > 0:
                print("✅ API connection successful!")
                print(f"   AAPL Price: ${data['c']:.2f}")
                print(f"   Change: {data['d']:+.2f} ({data['dp']:+.2f}%)")
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

def test_finnhub_sentiment():
    """Test Finnhub news sentiment"""
    print("\n📰 Testing News Sentiment...")

    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        return False

    try:
        url = f"https://finnhub.io/api/v1/news-sentiment"
        params = {
            'symbol': 'AAPL',
            'token': api_key
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'sentiment' in data:
                sentiment = data['sentiment']
                buzz = data.get('buzz', {})

                print("✅ News sentiment data retrieved!")
                print(f"   Sentiment: {sentiment.get('label', 'unknown')}")
                print(f"   Score: {sentiment.get('score', 0):.3f}")
                print(f"   Articles in last week: {buzz.get('articlesInLastWeek', 0)}")
                return True
            else:
                print("⚠️  Sentiment API responded but no data")
                return False
        else:
            print(f"❌ Sentiment API error: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Sentiment test error: {e}")
        return False

if __name__ == "__main__":
    success = test_finnhub_key()
    if success:
        test_finnhub_sentiment()

    print("\n" + "=" * 40)
    if success:
        print("🎉 Finnhub API key is configured and working!")
        print("💡 Ready to use Finnhub for arbitrage strategies")
    else:
        print("❌ Finnhub API key needs to be configured")
        print("   Add FINNHUB_API_KEY=your_key to .env file")
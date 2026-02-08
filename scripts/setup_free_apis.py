#!/usr/bin/env python3
"""
Free Worldwide API Quick Setup for AAC
Helps you configure the best free worldwide APIs for arbitrage trading.
"""

import os
import sys
from pathlib import Path

def setup_free_apis():
    """Quick setup for the best free worldwide APIs"""
    print("🚀 AAC Free Worldwide API Quick Setup")
    print("=" * 50)

    print("\n🎯 RECOMMENDED FREE APIs FOR WORLDWIDE ARBITRAGE:")
    print("1. 🥇 Alpha Vantage - Best all-around (50+ global markets)")
    print("2. 🥈 CoinGecko - No API key needed (10,000+ cryptos)")
    print("3. 🥉 Twelve Data - Excellent global coverage (60+ exchanges)")
    print("4. 🔄 Fixer.io - Forex data (170+ currencies)")
    print()

    # Load existing .env
    env_path = Path('.env')
    env_data = {}

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_data[key.strip()] = value.strip()

    print("🔑 QUICK SETUP - Get API Keys:")
    print("-" * 40)

    # Alpha Vantage (Best for global stocks)
    print("\n1. 📈 Alpha Vantage (RECOMMENDED FIRST)")
    print("   🌐 https://www.alphavantage.co/support/#api-key")
    print("   📊 50+ global markets, forex, crypto")
    print("   🆓 25 calls/day free")

    av_key = input("   Enter Alpha Vantage API Key (or press Enter to skip): ").strip()
    if av_key:
        env_data['ALPHAVANTAGE_API_KEY'] = av_key
        print("   ✅ Alpha Vantage configured!")
    else:
        print("   ⏭️  Skipped Alpha Vantage")

    # CoinGecko (No key needed)
    print("\n2. ₿ CoinGecko (NO API KEY NEEDED)")
    print("   🌐 https://www.coingecko.com/en/api")
    print("   📊 10,000+ cryptocurrencies")
    print("   🆓 No rate limits (fair use)")
    print("   ✅ No API key required - ready to use!")

    # Twelve Data (Good global coverage)
    print("\n3. 🌍 Twelve Data (EXCELLENT GLOBAL COVERAGE)")
    print("   🌐 https://twelvedata.com/pricing")
    print("   📊 60+ global exchanges")
    print("   🆓 800 calls/day free")

    td_key = input("   Enter Twelve Data API Key (or press Enter to skip): ").strip()
    if td_key:
        env_data['TWELVE_DATA_API_KEY'] = td_key
        print("   ✅ Twelve Data configured!")
    else:
        print("   ⏭️  Skipped Twelve Data")

    # Fixer.io (Forex)
    print("\n4. 💱 Fixer.io (FOREX SPECIALIST)")
    print("   🌐 https://fixer.io/signup/free")
    print("   📊 170+ currencies")
    print("   🆓 1,000 calls/month free")

    fixer_key = input("   Enter Fixer.io API Key (or press Enter to skip): ").strip()
    if fixer_key:
        env_data['FIXER_API_KEY'] = fixer_key
        print("   ✅ Fixer.io configured!")
    else:
        print("   ⏭️  Skipped Fixer.io")

    # Save to .env file
    if env_data:
        # Read existing content
        existing_content = []
        if env_path.exists():
            with open(env_path, 'r') as f:
                existing_content = f.readlines()

        # Update existing keys and add new ones
        updated_lines = []
        keys_added = set()

        for line in existing_content:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                key = line_stripped.split('=', 1)[0].strip()
                if key in env_data:
                    updated_lines.append(f"{key}={env_data[key]}\n")
                    keys_added.add(key)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        # Add new keys
        if keys_added != set(env_data.keys()):
            updated_lines.append("\n# Free Worldwide APIs\n")
            for key, value in env_data.items():
                if key not in keys_added:
                    updated_lines.append(f"{key}={value}\n")

        with open(env_path, 'w') as f:
            f.writelines(updated_lines)

        print("\n✅ FREE APIs CONFIGURED!")
        print("-" * 30)

        configured_count = len([k for k in env_data.keys() if k in ['ALPHAVANTAGE_API_KEY', 'TWELVE_DATA_API_KEY', 'FIXER_API_KEY'] and env_data[k]])
        print(f"📊 Configured: {configured_count}/4 free worldwide APIs")

        if 'ALPHAVANTAGE_API_KEY' in env_data and env_data['ALPHAVANTAGE_API_KEY']:
            print("📈 Alpha Vantage: ✅ Ready")
        if 'TWELVE_DATA_API_KEY' in env_data and env_data['TWELVE_DATA_API_KEY']:
            print("🌍 Twelve Data: ✅ Ready")
        if 'FIXER_API_KEY' in env_data and env_data['FIXER_API_KEY']:
            print("💱 Fixer.io: ✅ Ready")
        print("₿ CoinGecko: ✅ Ready (no key needed)")

        print("\n🧪 Test your setup:")
        print("   python test_free_apis.py")

        print("\n🚀 Next steps:")
        print("   1. Run triangular arbitrage strategy")
        print("   2. Test worldwide market data")
        print("   3. Add more APIs as needed")

    else:
        print("\nℹ️  No APIs configured.")

def test_free_apis():
    """Test the configured free APIs"""
    print("🧪 Testing Free Worldwide APIs")
    print("=" * 40)

    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    import requests

    # Test Alpha Vantage
    av_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if av_key:
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={av_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'Global Quote' in data and data['Global Quote']:
                    price = data['Global Quote'].get('05. price', 'N/A')
                    print(f"✅ Alpha Vantage: IBM @ ${price}")
                else:
                    print("⚠️  Alpha Vantage: Connected but no data")
            else:
                print(f"❌ Alpha Vantage: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Alpha Vantage: {e}")
    else:
        print("⏭️  Alpha Vantage: Not configured")

    # Test CoinGecko (no key needed)
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'bitcoin' in data and 'usd' in data['bitcoin']:
                price = data['bitcoin']['usd']
                print(f"✅ CoinGecko: BTC @ ${price:,.2f}")
            else:
                print("⚠️  CoinGecko: Connected but no data")
        else:
            print(f"❌ CoinGecko: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ CoinGecko: {e}")

    # Test Twelve Data
    td_key = os.getenv('TWELVE_DATA_API_KEY')
    if td_key:
        try:
            url = f"https://api.twelvedata.com/quote?symbol=AAPL&apikey={td_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'close' in data:
                    price = data['close']
                    print(f"✅ Twelve Data: AAPL @ ${price}")
                else:
                    print("⚠️  Twelve Data: Connected but no data")
            else:
                print(f"❌ Twelve Data: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Twelve Data: {e}")
    else:
        print("⏭️  Twelve Data: Not configured")

    # Test Fixer.io
    fixer_key = os.getenv('FIXER_API_KEY')
    if fixer_key:
        try:
            url = f"http://data.fixer.io/api/latest?access_key={fixer_key}&symbols=USD,EUR,GBP"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'rates' in data:
                    usd_rate = data['rates'].get('USD', 'N/A')
                    print(f"✅ Fixer.io: EUR/USD @ {usd_rate}")
                else:
                    print("⚠️  Fixer.io: Connected but no data")
            else:
                print(f"❌ Fixer.io: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Fixer.io: {e}")
    else:
        print("⏭️  Fixer.io: Not configured")

    print("\n🎯 Test Complete!")
    print("💡 Tip: Use these APIs for worldwide arbitrage opportunities")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_free_apis()
    else:
        setup_free_apis()
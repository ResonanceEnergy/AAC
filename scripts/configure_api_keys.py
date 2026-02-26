#!/usr/bin/env python3
"""
AAC Premium API Key Configuration Helper
Helps you configure premium market data API keys for enhanced trading performance.
"""

import os
import sys
from pathlib import Path

def load_env_file():
    """Load existing .env file if it exists"""
    env_path = Path('.env')
    env_data = {}

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_data[key.strip()] = value.strip()

    return env_data

def save_env_file(env_data):
    """Save environment variables to .env file"""
    env_path = Path('.env')

    # Read existing content to preserve comments and structure
    existing_content = []
    if env_path.exists():
        with open(env_path, 'r') as f:
            existing_content = f.readlines()

    # Update or add new keys
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

    # Add any new keys that weren't in the file
    if keys_added != set(env_data.keys()):
        if updated_lines and not updated_lines[-1].endswith('\n'):
            updated_lines.append('\n')

        updated_lines.append("# Premium Market Data API Keys\n")
        for key, value in env_data.items():
            if key not in keys_added:
                updated_lines.append(f"{key}={value}\n")

    with open(env_path, 'w') as f:
        f.writelines(updated_lines)

def main():
    print("🚀 AAC Premium API Key Configuration Helper")
    print("=" * 50)

    # Load existing environment
    env_data = load_env_file()

    print("\n📝 Current API Key Status:")
    api_keys = {
        'POLYGON_API_KEY': 'Polygon.io',
        'FINNHUB_API_KEY': 'Finnhub',
        'IEX_CLOUD_API_KEY': 'IEX Cloud',
        'TWELVE_DATA_API_KEY': 'Twelve Data',
        'INTRINIO_API_KEY': 'Intrinio',
        'ALPHAVANTAGE_API_KEY': 'Alpha Vantage',
        'EODHD_API_KEY': 'EODHD',
        'TRADESTIE_API_KEY': 'TradeStie (Reddit Sentiment)',
        'REDDIT_CLIENT_ID': 'Reddit API (PRAW)',
        'REDDIT_CLIENT_SECRET': 'Reddit API (PRAW)',
        'REDDIT_USER_AGENT': 'Reddit API (PRAW)',
        'REDDIT_USERNAME': 'Reddit API (PRAW)',
        'REDDIT_PASSWORD': 'Reddit API (PRAW)',
        # Free APIs (no key required)
        'WORLD_BANK_API': 'World Bank Data Catalog (Free)'
    }

    configured = 0
    for key, name in api_keys.items():
        if key == 'WORLD_BANK_API':
            # World Bank is always "configured" since it's free
            status = "✅ Available (Free API)"
            configured += 1
        else:
            status = "✅ Configured" if env_data.get(key) else "❌ Not configured"
            if env_data.get(key):
                configured += 1
        print(f"   {name}: {status}")

    print(f"\n📊 Summary: {configured} / {len(api_keys)} APIs configured (including free APIs)")

    print("\n🔑 Enter your API keys below (press Enter to skip):")

    # Collect API keys from user
    new_keys = {}
    for key, name in api_keys.items():
        if key == 'WORLD_BANK_API':
            # Skip World Bank API as it's free and doesn't require configuration
            continue

        current_value = env_data.get(key, '')
        if current_value:
            masked = current_value[:8] + "..." if len(current_value) > 8 else current_value
            print(f"\n{name} (current: {masked}):")
        else:
            print(f"\n{name}:")

        user_input = input(f"   {key} = ").strip()

        if user_input:
            if user_input.lower() == 'clear':
                new_keys[key] = ''
            else:
                new_keys[key] = user_input
        elif current_value:
            # Keep existing value if user doesn't provide new one
            new_keys[key] = current_value

    # Special handling for Intrinio (needs username too)
    if 'INTRINIO_API_KEY' in new_keys and new_keys['INTRINIO_API_KEY']:
        intrinio_username = env_data.get('INTRINIO_USERNAME', '')
        if intrinio_username:
            print(f"\nIntrinio Username (current: {intrinio_username}):")
        else:
            print("\nIntrinio Username:")

        username_input = input("   INTRINIO_USERNAME = ").strip()
        if username_input:
            new_keys['INTRINIO_USERNAME'] = username_input
        elif intrinio_username:
            new_keys['INTRINIO_USERNAME'] = intrinio_username

    # Save to .env file
    if new_keys:
        save_env_file(new_keys)
        print("\n✅ API keys saved to .env file!")

        # Show updated status
        print("\n📊 Updated API Key Status:")
        final_env = load_env_file()
        configured_final = 0
        for key, name in api_keys.items():
            if key == 'WORLD_BANK_API':
                # World Bank is always "configured" since it's free
                status = "✅ Available (Free API)"
                configured_final += 1
            else:
                status = "✅ Configured" if final_env.get(key) else "❌ Not configured"
                if final_env.get(key):
                    configured_final += 1
            print(f"   {name}: {status}")

        print(f"\n🎯 Final Summary: {configured_final} / {len(api_keys)} APIs configured (including free APIs)")

        if configured_final > 0:
            print("\n🧪 Ready to test! Run: python test_premium_apis.py")
    else:
        print("\nℹ️  No changes made.")

if __name__ == "__main__":
    main()
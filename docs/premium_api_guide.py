#!/usr/bin/env python3
"""
AAC Premium Market Data API Setup Guide
========================================
Complete guide for setting up premium market data APIs for enhanced trading performance.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

class PremiumAPIGuide:
    """Guide for setting up premium market data APIs"""

    def __init__(self):
        self.apis = {
            'polygon': {
                'name': 'Polygon.io',
                'website': 'https://polygon.io',
                'free_tier': '5 calls/minute, 5M calls/month',
                'premium_plans': [
                    {'name': 'Stocks Starter', 'price': '$99/month', 'calls': '5M/month'},
                    {'name': 'Stocks Plus', 'price': '$399/month', 'calls': '15M/month'},
                    {'name': 'Stocks Pro', 'price': '$999/month', 'calls': '50M/month'},
                ],
                'features': [
                    'Real-time and historical stock data',
                    'Options data',
                    'Aggregates API',
                    'Reference data',
                    'Excellent documentation'
                ],
                'setup_steps': [
                    '1. Visit https://polygon.io',
                    '2. Sign up for a free account',
                    '3. Get your API key from the dashboard',
                    '4. Add POLYGON_API_KEY=your_key_here to .env file'
                ]
            },
            'finnhub': {
                'name': 'Finnhub',
                'website': 'https://finnhub.io',
                'free_tier': '60 calls/minute, 150 calls/day',
                'premium_plans': [
                    {'name': 'Basic', 'price': '$9.99/month', 'calls': '300/minute'},
                    {'name': 'Standard', 'price': '$49.99/month', 'calls': '600/minute'},
                    {'name': 'Professional', 'price': '$99.99/month', 'calls': '1000/minute'},
                ],
                'features': [
                    'Real-time stock quotes',
                    'Company fundamentals',
                    'Financial statements',
                    'News and social sentiment',
                    'Technical indicators'
                ],
                'setup_steps': [
                    '1. Visit https://finnhub.io',
                    '2. Create a free account',
                    '3. Get your API token from the dashboard',
                    '4. Add FINNHUB_API_KEY=your_token_here to .env file'
                ]
            },
            'iex_cloud': {
                'name': 'IEX Cloud',
                'website': 'https://iexcloud.io',
                'free_tier': '50,000 calls/month',
                'premium_plans': [
                    {'name': 'Launch', 'price': '$9/month', 'calls': '5M/month'},
                    {'name': 'Grow', 'price': '$49/month', 'calls': '50M/month'},
                    {'name': 'Scale', 'price': '$99/month', 'calls': '500M/month'},
                ],
                'features': [
                    'Real-time quotes and trades',
                    'Historical data (15+ years)',
                    'Fundamentals and financials',
                    'News and social sentiment',
                    'Crypto data'
                ],
                'setup_steps': [
                    '1. Visit https://iexcloud.io',
                    '2. Sign up for a free account',
                    '3. Get your API token from the console',
                    '4. Add IEX_CLOUD_API_KEY=your_token_here to .env file'
                ]
            },
            'twelve_data': {
                'name': 'Twelve Data',
                'website': 'https://twelvedata.com',
                'free_tier': '800 calls/day',
                'premium_plans': [
                    {'name': 'Basic', 'price': '$9.99/month', 'calls': '800/day'},
                    {'name': 'Pro', 'price': '$29.99/month', 'calls': '5K/day'},
                    {'name': 'Business', 'price': '$99/month', 'calls': '50K/day'},
                ],
                'features': [
                    'Real-time and historical data',
                    'Stocks, forex, crypto, ETFs',
                    'Technical indicators',
                    'Economic data',
                    'WebSocket streaming'
                ],
                'setup_steps': [
                    '1. Visit https://twelvedata.com',
                    '2. Create a free account',
                    '3. Get your API key from the dashboard',
                    '4. Add TWELVE_DATA_API_KEY=your_key_here to .env file'
                ]
            },
            'intrinio': {
                'name': 'Intrinio',
                'website': 'https://intrinio.com',
                'free_tier': 'None (Premium only)',
                'premium_plans': [
                    {'name': 'Starter', 'price': '$75/month', 'calls': '50K/month'},
                    {'name': 'Developer', 'price': '$200/month', 'calls': '250K/month'},
                    {'name': 'Professional', 'price': '$500/month', 'calls': '1M/month'},
                ],
                'features': [
                    'Institutional-grade data',
                    'Real-time and historical prices',
                    'Fundamentals and financials',
                    'Options data',
                    'Alternative data'
                ],
                'setup_steps': [
                    '1. Visit https://intrinio.com',
                    '2. Sign up for an account',
                    '3. Get your API key and username',
                    '4. Add INTRINIO_API_KEY=your_key_here to .env file',
                    '5. Add INTRINIO_USERNAME=your_username_here to .env file'
                ]
            },
            'alpha_vantage': {
                'name': 'Alpha Vantage',
                'website': 'https://www.alphavantage.co',
                'free_tier': '25 calls/day, 5 calls/minute',
                'premium_plans': [
                    {'name': 'Premium', 'price': '$49.99/month', 'calls': '75/minute'},
                ],
                'features': [
                    'Stock time series data',
                    'Technical indicators',
                    'Sector performance',
                    'Currency exchange rates',
                    'Crypto data'
                ],
                'setup_steps': [
                    '1. Visit https://www.alphavantage.co/support/#api-key',
                    '2. Sign up for a free API key',
                    '3. Verify your email',
                    '4. Add ALPHAVANTAGE_API_KEY=your_key_here to .env file'
                ]
            }
        }

    def print_guide(self):
        """Print the complete setup guide"""
        print("🚀 AAC Premium Market Data API Setup Guide")
        print("=" * 60)
        print()

        print("📊 WHY UPGRADE TO PREMIUM APIs?")
        print("-" * 40)
        print("• Higher rate limits (free APIs have strict limits)")
        print("• More frequent data updates")
        print("• Additional data sources and features")
        print("• Better reliability and uptime")
        print("• Historical data access")
        print("• Priority support")
        print()

        print("💰 CURRENT FREE APIs IN USE:")
        print("-" * 40)
        print("• Yahoo Finance (via yfinance) - No API key needed")
        print("• Binance/Coinbase Pro - For crypto data")
        print("• Alpha Vantage - If API key configured")
        print()

        for api_key, api_info in self.apis.items():
            self._print_api_guide(api_key, api_info)
            print()

    def _print_api_guide(self, api_key: str, api_info: Dict[str, Any]):
        """Print guide for a specific API"""
        print(f"🔑 {api_info['name']} ({api_key.upper()})")
        print("-" * 50)
        print(f"🌐 Website: {api_info['website']}")
        print(f"🆓 Free Tier: {api_info['free_tier']}")

        if api_info['premium_plans']:
            print("💎 Premium Plans:")
            for plan in api_info['premium_plans']:
                print(f"   • {plan['name']}: {plan['price']} ({plan['calls']} calls)")

        print("✨ Features:")
        for feature in api_info['features']:
            print(f"   • {feature}")

        print("📝 Setup Steps:")
        for step in api_info['setup_steps']:
            print(f"   {step}")

    def check_env_file(self) -> Dict[str, bool]:
        """Check which APIs are configured in .env file"""
        configured = {}
        for api_key in self.apis.keys():
            env_var = f"{api_key.upper()}_API_KEY"
            configured[api_key] = bool(os.getenv(env_var, ''))
        return configured

    def print_status(self):
        """Print current API configuration status"""
        print("📊 CURRENT API CONFIGURATION STATUS")
        print("=" * 50)

        configured = self.check_env_file()
        for api_key, is_configured in configured.items():
            status = "✅ Configured" if is_configured else "❌ Not configured"
            name = self.apis[api_key]['name']
            print(f"• {name}: {status}")

        configured_count = sum(configured.values())
        total_count = len(configured)
        print(f"\n📈 Summary: {configured_count}/{total_count} premium APIs configured")

        if configured_count == 0:
            print("💡 Tip: Start with Polygon.io or Finnhub for the best free tier experience!")
        elif configured_count < 3:
            print("💡 Tip: Consider adding 2-3 premium APIs for redundancy and better data quality.")

def main():
    """Main function to display the guide"""
    guide = PremiumAPIGuide()
    guide.print_guide()
    print("\n" + "=" * 60)
    guide.print_status()

if __name__ == "__main__":
    main()
"""
Free Worldwide API Guide for AAC
================================

Comprehensive guide to free APIs offering worldwide market data.
These APIs provide global stock markets, forex, crypto, and economic data.
"""

import requests
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Free Worldwide APIs Database
FREE_WORLDWIDE_APIS = {
    # 🌍 GLOBAL STOCK MARKET APIs
    "alphavantage": {
        "name": "Alpha Vantage",
        "website": "https://www.alphavantage.co",
        "free_tier": "25 calls/day, 5 calls/minute",
        "global_coverage": "50+ global markets (NYSE, NASDAQ, LSE, TSE, etc.)",
        "data_types": ["Stocks", "Forex", "Crypto", "Technical Indicators"],
        "key_required": True,
        "signup_url": "https://www.alphavantage.co/support/#api-key",
        "features": [
            "Real-time and historical data",
            "50+ global exchanges",
            "Forex cross rates",
            "Crypto prices",
            "Technical indicators"
        ]
    },

    "twelve_data": {
        "name": "Twelve Data",
        "website": "https://twelvedata.com",
        "free_tier": "800 calls/day",
        "global_coverage": "60+ global exchanges",
        "data_types": ["Stocks", "Forex", "Crypto", "ETFs"],
        "key_required": True,
        "signup_url": "https://twelvedata.com/pricing",
        "features": [
            "Real-time data",
            "60+ exchanges worldwide",
            "WebSocket streaming",
            "Economic data",
            "Technical indicators"
        ]
    },

    "iex_cloud": {
        "name": "IEX Cloud",
        "website": "https://iexcloud.io",
        "free_tier": "50,000 calls/month",
        "global_coverage": "US markets + international data",
        "data_types": ["Stocks", "ETFs", "Mutual Funds"],
        "key_required": True,
        "signup_url": "https://iexcloud.io/console",
        "features": [
            "US market data",
            "International stocks",
            "15+ years of historical data",
            "Fundamentals and financials"
        ]
    },

    # 💱 FOREX & CURRENCY APIs
    "fixer": {
        "name": "Fixer.io",
        "website": "https://fixer.io",
        "free_tier": "1,000 calls/month",
        "global_coverage": "170+ currencies",
        "data_types": ["Forex", "Currency Conversion"],
        "key_required": True,
        "signup_url": "https://fixer.io/signup/free",
        "features": [
            "170+ currencies",
            "Real-time rates",
            "Historical data (up to 1999)",
            "Currency conversion"
        ]
    },

    "currencyapi": {
        "name": "CurrencyAPI",
        "website": "https://currencyapi.com",
        "free_tier": "300 calls/month",
        "global_coverage": "150+ currencies",
        "data_types": ["Forex", "Crypto"],
        "key_required": True,
        "signup_url": "https://currencyapi.com/pricing",
        "features": [
            "150+ currencies",
            "Crypto currencies",
            "Real-time rates",
            "Historical data"
        ]
    },

    # ₿ CRYPTO APIs
    "coinmarketcap": {
        "name": "CoinMarketCap",
        "website": "https://coinmarketcap.com/api/",
        "free_tier": "10,000 calls/month",
        "global_coverage": "10,000+ cryptocurrencies",
        "data_types": ["Crypto", "Market Data"],
        "key_required": True,
        "signup_url": "https://coinmarketcap.com/api/",
        "features": [
            "10,000+ cryptocurrencies",
            "Global market data",
            "Exchange data",
            "Historical data"
        ]
    },

    "coingecko": {
        "name": "CoinGecko",
        "website": "https://www.coingecko.com/en/api",
        "free_tier": "No rate limits (fair use)",
        "global_coverage": "10,000+ cryptocurrencies",
        "data_types": ["Crypto", "DeFi", "NFTs"],
        "key_required": False,
        "signup_url": "https://www.coingecko.com/en/api",
        "features": [
            "10,000+ cryptocurrencies",
            "DeFi protocols",
            "NFT data",
            "Exchange rates",
            "No API key required"
        ]
    },

    "binance_api": {
        "name": "Binance API",
        "website": "https://binance-docs.github.io/apidocs/",
        "free_tier": "Varies by endpoint",
        "global_coverage": "350+ cryptocurrencies",
        "data_types": ["Crypto", "Spot", "Futures"],
        "key_required": False,  # For public endpoints
        "signup_url": "https://www.binance.com/en/register",
        "features": [
            "350+ trading pairs",
            "Real-time data",
            "Historical data",
            "Order book data"
        ]
    },

    # 📰 NEWS & SENTIMENT APIs
    "newsapi": {
        "name": "NewsAPI",
        "website": "https://newsapi.org",
        "free_tier": "100 calls/day",
        "global_coverage": "70,000+ news sources worldwide",
        "data_types": ["News", "Financial News"],
        "key_required": True,
        "signup_url": "https://newsapi.org/register",
        "features": [
            "70,000+ news sources",
            "Financial news filtering",
            "Real-time updates",
            "Global coverage"
        ]
    },

    # 📊 ECONOMIC DATA APIs
    "world_bank": {
        "name": "World Bank Open Data",
        "website": "https://data.worldbank.org",
        "free_tier": "No limits",
        "global_coverage": "200+ countries",
        "data_types": ["Economic Indicators", "Development Data"],
        "key_required": False,
        "signup_url": "https://data.worldbank.org",
        "features": [
            "200+ countries",
            "Economic indicators",
            "Development data",
            "Time series data",
            "No API key required"
        ]
    },

    # 🏛️ CENTRAL BANK APIs
    "ecb_api": {
        "name": "European Central Bank",
        "website": "https://www.ecb.europa.eu/stats/ecb_statistics/",
        "free_tier": "No limits",
        "global_coverage": "Eurozone + global economic data",
        "data_types": ["Interest Rates", "Economic Indicators"],
        "key_required": False,
        "signup_url": "https://www.ecb.europa.eu/stats/ecb_statistics/",
        "features": [
            "Eurozone data",
            "Interest rates",
            "Exchange rates",
            "Economic indicators"
        ]
    },

    # 📈 MARKET INDICES APIs
    "eodhd": {
        "name": "EODHD",
        "website": "https://eodhd.com",
        "free_tier": "100 calls/day",
        "global_coverage": "50+ exchanges worldwide",
        "data_types": ["Stocks", "Indices", "Forex", "Crypto"],
        "key_required": True,
        "signup_url": "https://eodhd.com/register",
        "features": [
            "50+ exchanges",
            "Global indices",
            "Real-time data",
            "Historical data"
        ]
    }
}

def display_free_api_guide():
    """Display comprehensive guide to free worldwide APIs"""
    logger.info("🌍 FREE WORLDWIDE API GUIDE FOR AAC")
    logger.info("=" * 60)
    logger.info("")

    logger.info("🎯 WHY USE FREE WORLDWIDE APIs?")
    logger.info("-" * 40)
    logger.info("• Global market coverage (not just US)")
    logger.info("• Cost-effective for development/testing")
    logger.info("• Diverse data sources for arbitrage opportunities")
    logger.info("• No subscription fees")
    logger.info("• Good for worldwide arbitrage strategies")
    logger.info("")

    # Group APIs by category
    categories = {
        "🌍 Global Stock Markets": ["alphavantage", "twelve_data", "iex_cloud", "eodhd"],
        "💱 Forex & Currencies": ["fixer", "currencyapi"],
        "₿ Cryptocurrencies": ["coinmarketcap", "coingecko", "binance_api"],
        "📰 News & Sentiment": ["newsapi"],
        "📊 Economic Data": ["world_bank", "ecb_api"]
    }

    for category, apis in categories.items():
        logger.info(f"{category}")
        logger.info("-" * 50)

        for api_key in apis:
            if api_key in FREE_WORLDWIDE_APIS:
                api = FREE_WORLDWIDE_APIS[api_key]
                key_status = "🔑 Required" if api["key_required"] else "✅ No Key"

                logger.info(f"📡 {api['name']}")
                logger.info(f"   🌐 {api['website']}")
                logger.info(f"   🆓 Free Tier: {api['free_tier']}")
                logger.info(f"   🌍 Coverage: {api['global_coverage']}")
                logger.info(f"   🔑 API Key: {key_status}")
                logger.info(f"   📊 Data: {', '.join(api['data_types'])}")
                logger.info(f"   📝 Signup: {api['signup_url']}")
                logger.info("")

    logger.info("🚀 QUICK START RECOMMENDATIONS")
    logger.info("-" * 40)
    logger.info("1. 🥇 Alpha Vantage - Best all-around free API")
    logger.info("2. 🥈 CoinGecko - No API key needed for crypto")
    logger.info("3. 🥉 Twelve Data - Excellent global coverage")
    logger.info("")

    logger.info("💡 AAC INTEGRATION TIPS")
    logger.info("-" * 40)
    logger.info("• Use multiple APIs for data redundancy")
    logger.info("• Implement rate limiting and caching")
    logger.info("• Monitor API limits and switch providers")
    logger.info("• Combine free + premium APIs for best results")
    logger.info("")

    logger.info("🔧 CONFIGURATION EXAMPLE")
    logger.info("-" * 40)
    logger.info("# Add to your .env file:")
    logger.info("ALPHAVANTAGE_API_KEY=your_key_here")
    print("COINGECKO_API_KEY=")  # No key needed
    logger.info("FIXER_API_KEY=your_key_here")
    logger.info("")

def test_free_api_connectivity():
    """Test connectivity to free APIs that don't require keys"""
    logger.info("🧪 TESTING FREE API CONNECTIVITY")
    logger.info("-" * 40)

    # Test CoinGecko (no API key needed)
    try:
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=10)
        if response.status_code == 200:
            logger.info("✅ CoinGecko: Connected successfully")
        else:
            logger.info(f"⚠️  CoinGecko: HTTP {response.status_code}")
    except Exception as e:
        logger.info(f"❌ CoinGecko: {e}")

    # Test World Bank API
    try:
        response = requests.get("https://api.worldbank.org/v2/countries", timeout=10)
        if response.status_code == 200:
            logger.info("✅ World Bank: Connected successfully")
        else:
            logger.info(f"⚠️  World Bank: HTTP {response.status_code}")
    except Exception as e:
        logger.info(f"❌ World Bank: {e}")

    # Test ECB API
    try:
        response = requests.get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml", timeout=10)
        if response.status_code == 200:
            logger.info("✅ ECB: Connected successfully")
        else:
            logger.info(f"⚠️  ECB: HTTP {response.status_code}")
    except Exception as e:
        logger.info(f"❌ ECB: {e}")

    logger.info("")

if __name__ == "__main__":
    display_free_api_guide()
    test_free_api_connectivity()
#!/usr/bin/env python3
"""
Additional Raw Data Sources for AAC Arbitrage Trading
=====================================================

Comprehensive guide to additional data sources beyond our current APIs.
These provide raw data for enhanced arbitrage strategies.
"""

import requests
import json
from datetime import datetime

# Additional Data Sources for Arbitrage
ADDITIONAL_DATA_SOURCES = {
    # 📊 FINANCIAL DATA PROVIDERS
    "polygon": {
        "name": "Polygon.io",
        "website": "https://polygon.io",
        "free_tier": "5M calls/month, 5 calls/minute",
        "data_types": ["Stocks", "Options", "Aggregates", "Reference Data"],
        "relevance": "High-quality US market data, options chains for arbitrage",
        "api_endpoint": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/2024-01-01/2024-01-02?apiKey=demo",
        "features": [
            "Real-time aggregates",
            "Options data",
            "Corporate actions",
            "Financial statements",
            "Perfect for statistical arbitrage"
        ]
    },

    "finnhub": {
        "name": "Finnhub",
        "website": "https://finnhub.io",
        "free_tier": "150 calls/day, 60 calls/minute",
        "data_types": ["Stocks", "News", "Financials", "Technical"],
        "relevance": "Real-time quotes, earnings data, news sentiment",
        "api_endpoint": "https://finnhub.io/api/v1/quote?symbol=AAPL&token=demo",
        "features": [
            "Real-time quotes",
            "Company financials",
            "Earnings data",
            "News sentiment",
            "Insider trading data"
        ]
    },

    "eodhd": {
        "name": "EODHD",
        "website": "https://eodhd.com",
        "free_tier": "100 calls/day",
        "data_types": ["Stocks", "Indices", "Forex", "Crypto", "Futures"],
        "relevance": "Global market data, futures for arbitrage",
        "api_endpoint": "https://eodhd.com/api/real-time/AAPL.US?api_token=demo&fmt=json",
        "features": [
            "50+ exchanges",
            "Futures data",
            "Fundamental data",
            "Economic indicators",
            "Global indices"
        ]
    },

    # 🏛️ CENTRAL BANKS & ECONOMIC DATA
    "federal_reserve": {
        "name": "Federal Reserve Economic Data (FRED)",
        "website": "https://fred.stlouisfed.org/docs/api/fred/",
        "free_tier": "No limits",
        "data_types": ["Economic Indicators", "Interest Rates", "Employment"],
        "relevance": "US economic data for macro arbitrage signals",
        "api_endpoint": "https://api.stlouisfed.org/fred/series/observations?series_id=MORTGAGE30US&api_key=demo&file_type=json",
        "features": [
            "800,000+ economic time series",
            "Interest rates",
            "Employment data",
            "Housing data",
            "GDP and inflation data"
        ]
    },

    "oecd": {
        "name": "OECD Data",
        "website": "https://data.oecd.org/api/",
        "free_tier": "No limits",
        "data_types": ["Economic Indicators", "Trade Data", "Development"],
        "relevance": "Global economic indicators, trade data",
        "api_endpoint": "https://stats.oecd.org/SDMX-JSON/data/MEI_CLI/LOLITOAA.AUS.M/all?startTime=2020-01&endTime=2024-01",
        "features": [
            "Leading economic indicators",
            "Trade statistics",
            "Development indicators",
            "SDMX format data"
        ]
    },

    # 📰 NEWS & SENTIMENT DATA
    "alphavantage_news": {
        "name": "Alpha Vantage News",
        "website": "https://www.alphavantage.co/documentation/#news-sentiment",
        "free_tier": "25 calls/day (shared with other endpoints)",
        "data_types": ["News", "Sentiment Analysis"],
        "relevance": "Market sentiment for arbitrage timing",
        "api_endpoint": "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo",
        "features": [
            "Real-time news",
            "Sentiment analysis",
            "Ticker-specific news",
            "Relevance scores"
        ]
    },

    "newsapi": {
        "name": "NewsAPI",
        "website": "https://newsapi.org",
        "free_tier": "100 calls/day",
        "data_types": ["Financial News", "Market News"],
        "relevance": "Financial news for market impact analysis",
        "api_endpoint": "https://newsapi.org/v2/everything?q=stocks&apiKey=demo",
        "features": [
            "70,000+ news sources",
            "Financial news filtering",
            "Real-time updates",
            "Source credibility scores"
        ]
    },

    "tradestie_reddit": {
        "name": "TradeStie Reddit Sentiment",
        "website": "https://tradestie.com/apps/reddit",
        "free_tier": "20 calls/minute",
        "data_types": ["Reddit Sentiment", "Social Media"],
        "relevance": "WallStreetBets sentiment for retail arbitrage",
        "api_endpoint": "https://api.tradestie.com/v1/apps/reddit",
        "features": [
            "WallStreetBets sentiment",
            "Top 50 discussed stocks",
            "Real-time updates (15 min)",
            "Comment volume analysis",
            "Bullish/bearish signals"
        ]
    },

    "praw_reddit": {
        "name": "Reddit API (PRAW)",
        "website": "https://www.reddit.com/dev/api/",
        "free_tier": "600 requests/10min (authenticated)",
        "data_types": ["Reddit Posts", "Comments", "Sentiment"],
        "relevance": "Direct WallStreetBets access for sentiment arbitrage",
        "api_endpoint": "https://www.reddit.com/r/wallstreetbets/hot.json",
        "features": [
            "Direct Reddit API access",
            "Real-time hot posts",
            "Comment analysis",
            "Ticker extraction",
            "Sentiment scoring",
            "Arbitrage signal generation"
        ]
    },

    # 📈 OPTIONS & DERIVATIVES DATA
    "cboe": {
        "name": "CBOE Global Markets",
        "website": "https://www.cboe.com/us/options/market_statistics/",
        "free_tier": "Limited free data",
        "data_types": ["Options", "VIX", "Volatility"],
        "relevance": "Options data for volatility arbitrage",
        "api_endpoint": "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json",
        "features": [
            "Options chains",
            "VIX data",
            "Volatility products",
            "Put/call ratios"
        ]
    },

    # 🏢 ALTERNATIVE DATA
    "glassnode": {
        "name": "Glassnode",
        "website": "https://glassnode.com",
        "free_tier": "Limited free metrics",
        "data_types": ["On-chain Analytics", "Crypto Metrics"],
        "relevance": "Crypto on-chain data for arbitrage",
        "api_endpoint": "https://api.glassnode.com/v1/metrics/indicators/sopr?a=BTC&api_key=demo",
        "features": [
            "On-chain metrics",
            "Exchange flows",
            "Mining data",
            "Network health indicators"
        ]
    },

    # 🌍 COMMODITIES & FUTURES
    "quandl": {
        "name": "Quandl (Nasdaq Data Link)",
        "website": "https://data.nasdaq.com",
        "free_tier": "50 calls/day",
        "data_types": ["Commodities", "Futures", "Economic Data"],
        "relevance": "Commodities data for cross-market arbitrage",
        "api_endpoint": "https://data.nasdaq.com/api/v3/datasets/WIKI/AAPL/data.json?api_key=demo",
        "features": [
            "Commodities prices",
            "Futures data",
            "Economic indicators",
            "Alternative datasets"
        ]
    },

    # 📊 MARKET MICROSTRUCTURE
    "liquidity": {
        "name": "Liquidity Data Providers",
        "website": "https://www.tradingview.com",
        "free_tier": "Limited",
        "data_types": ["Order Book", "Liquidity Metrics"],
        "relevance": "Market microstructure for HFT arbitrage",
        "api_endpoint": "https://www.tradingview.com/api/v1/symbols/AAPL/orderbook/",
        "features": [
            "Order book depth",
            "Liquidity metrics",
            "Market impact analysis",
            "Spread analysis"
        ]
    }
}

def test_additional_data_sources():
    """Test additional data sources for AAC arbitrage"""

    print("🔍 ADDITIONAL RAW DATA SOURCES FOR AAC ARBITRAGE")
    print("=" * 70)
    print()

    print("🎯 WHY ADD MORE DATA SOURCES?")
    print("-" * 40)
    print("• Diverse data for cross-validation")
    print("• Specialized data for specific arbitrage strategies")
    print("• Alternative data for alpha generation")
    print("• Economic indicators for macro arbitrage")
    print("• News sentiment for timing strategies")
    print("• Options data for volatility arbitrage")
    print()

    # Test some APIs that don't require keys
    test_apis = {
        "Federal Reserve FRED": "https://api.stlouisfed.org/fred/series/observations?series_id=MORTGAGE30US&api_key=demo&file_type=json",
        "OECD Data": "https://stats.oecd.org/SDMX-JSON/data/MEI_CLI/LOLITOAA.AUS.M/all?startTime=2020-01&endTime=2024-01",
        "CBOE Options": "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
    }

    print("🧪 TESTING FREE DATA SOURCES:")
    print("-" * 40)

    for name, url in test_apis.items():
        print(f"\n🔍 Testing {name}...")
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                print(f"   ✅ {name}: Connected successfully!")

                # Show sample data
                if name == "Federal Reserve FRED":
                    data = response.json()
                    if 'observations' in data and data['observations']:
                        latest = data['observations'][-1]
                        print(f"   📊 Latest 30-year mortgage rate: {latest.get('value', 'N/A')}%")
                        print(f"   📅 Date: {latest.get('date', 'N/A')}")

                elif name == "OECD Data":
                    data = response.json()
                    print(f"   📊 OECD data structure available")
                    if 'dataSets' in data:
                        print(f"   📈 Data points available")

                elif name == "CBOE Options":
                    data = response.json()
                    if 'data' in data and data['data']:
                        print(f"   📊 S&P 500 options data available")
                        print(f"   📈 Options chain length: {len(data['data'])}")

            else:
                print(f"   ❌ {name}: HTTP {response.status_code}")

        except Exception as e:
            print(f"   ❌ {name}: {e}")

    print()
    print("📋 RECOMMENDED ADDITIONAL DATA SOURCES:")
    print("-" * 50)

    categories = {
        "🔥 HIGH PRIORITY (Immediate Value)": ["polygon", "finnhub", "federal_reserve"],
        "📊 ECONOMIC DATA": ["oecd", "quandl"],
        "📰 SENTIMENT DATA": ["alphavantage_news", "newsapi"],
        "📈 DERIVATIVES": ["cboe", "eodhd"],
        "₿ CRYPTO ANALYTICS": ["glassnode"],
        "🏗️ INFRASTRUCTURE": ["liquidity"]
    }

    for category, apis in categories.items():
        print(f"\n{category}")
        print("-" * 60)

        for api_key in apis:
            if api_key in ADDITIONAL_DATA_SOURCES:
                api = ADDITIONAL_DATA_SOURCES[api_key]
                print(f"🔹 {api['name']}")
                print(f"   🌐 {api['website']}")
                print(f"   🆓 {api['free_tier']}")
                print(f"   🎯 {api['relevance']}")
                print(f"   ✨ Key Features: {', '.join(api['features'][:2])}")
                print()

    print("🚀 INTEGRATION STRATEGY:")
    print("-" * 30)
    print("1. Start with Polygon.io for high-quality US market data")
    print("2. Add Finnhub for real-time quotes and news")
    print("3. Integrate Federal Reserve data for economic signals")
    print("4. Add CBOE for options and volatility data")
    print("5. Include Glassnode for crypto on-chain analytics")
    print("6. Use NewsAPI for market sentiment analysis")
    print()

    print("💡 ARBITRAGE APPLICATIONS:")
    print("-" * 30)
    print("• Cross-market arbitrage with Polygon + Twelve Data")
    print("• Statistical arbitrage with options data")
    print("• Macro arbitrage with economic indicators")
    print("• Sentiment-based arbitrage with news data")
    print("• Volatility arbitrage with CBOE data")
    print("• Crypto arbitrage with on-chain metrics")

if __name__ == "__main__":
    test_additional_data_sources()
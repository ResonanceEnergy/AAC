#!/usr/bin/env python3
"""
AAC API Registry & Status Tool
================================
Lists ALL APIs used by the AAC system, their signup URLs, current
configuration status (key present/missing), and priority.

Run:  python tools/api_registry.py
      python tools/api_registry.py --json
      python tools/api_registry.py --missing
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass

# ──────────────────────────────────────────────────
# MASTER API REGISTRY
# ──────────────────────────────────────────────────
# Each entry: (env_var, name, website, category, cost, notes)

REGISTRY = [
    # ── EXCHANGES / BROKERS ─────────────────────
    {
        "env_var": "IBKR_ACCOUNT",
        "name": "Interactive Brokers (IBKR)",
        "website": "https://www.interactivebrokers.com",
        "signup": "https://portal.interactivebrokers.com/en/trading/ib-api.php",
        "category": "Exchange/Broker",
        "cost": "Free account, commission per trade",
        "notes": "US stocks, ETFs, futures, options, forex. Requires IB Gateway/TWS running locally.",
        "priority": "HIGH",
    },
    {
        "env_var": "NDAX_API_KEY",
        "name": "NDAX (National Digital Asset Exchange)",
        "website": "https://ndax.io",
        "signup": "https://ndax.io/signup",
        "category": "Exchange/Broker",
        "cost": "Free, 0.2% trading fee",
        "notes": "Canadian crypto exchange. BTC, ETH, USDT, and more CAD pairs.",
        "priority": "HIGH",
    },
    {
        "env_var": "MOOMOO_API_KEY",
        "name": "Moomoo / Futu",
        "website": "https://www.moomoo.com",
        "signup": "https://www.moomoo.com/us/download",
        "category": "Exchange/Broker",
        "cost": "Free, $0 commission stocks",
        "notes": "US, HK, CN, SG stocks & options. Requires OpenD gateway locally.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "MT5_LOGIN",
        "name": "Noxi Rise (MetaTrader 5)",
        "website": "https://noxirise.com",
        "signup": "https://noxirise.com",
        "category": "Exchange/Broker",
        "cost": "Spread-based (no commission)",
        "notes": "MT5-based institutional forex/CFD broker. Windows only for Python API.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "BINANCE_API_KEY",
        "name": "Binance",
        "website": "https://www.binance.com",
        "signup": "https://www.binance.com/en/register",
        "category": "Exchange/Broker",
        "cost": "Free, 0.1% trading fee",
        "notes": "World's largest crypto exchange. Already integrated via CCXT.",
        "priority": "LOW",
    },
    {
        "env_var": "COINBASE_API_KEY",
        "name": "Coinbase",
        "website": "https://www.coinbase.com",
        "signup": "https://www.coinbase.com/signup",
        "category": "Exchange/Broker",
        "cost": "Free, variable fees",
        "notes": "US-regulated crypto exchange. Already integrated via CCXT.",
        "priority": "LOW",
    },
    {
        "env_var": "KRAKEN_API_KEY",
        "name": "Kraken",
        "website": "https://www.kraken.com",
        "signup": "https://www.kraken.com/sign-up",
        "category": "Exchange/Broker",
        "cost": "Free, 0.16-0.26% fees",
        "notes": "Major crypto exchange with advanced order types. Already integrated via CCXT.",
        "priority": "LOW",
    },

    # ── MARKET DATA ─────────────────────────────
    {
        "env_var": "POLYGON_API_KEY",
        "name": "Polygon.io",
        "website": "https://polygon.io",
        "signup": "https://polygon.io/dashboard/signup",
        "category": "Market Data",
        "cost": "Free (5 req/min) | Basic $29/mo | Starter $79/mo",
        "notes": "US stocks, options, forex, crypto. Real-time + historical. Best bang for buck.",
        "priority": "HIGH",
    },
    {
        "env_var": "FINNHUB_API_KEY",
        "name": "Finnhub",
        "website": "https://finnhub.io",
        "signup": "https://finnhub.io/register",
        "category": "Market Data",
        "cost": "Free (60 req/min) | Premium plans available",
        "notes": "Earnings, IPOs, insider trades, SEC filings, news, economic calendar.",
        "priority": "HIGH",
    },
    {
        "env_var": "COINGECKO_API_KEY",
        "name": "CoinGecko",
        "website": "https://www.coingecko.com",
        "signup": "https://www.coingecko.com/en/api/pricing",
        "category": "Market Data",
        "cost": "Free (30 req/min) | Pro $129/mo (500 req/min)",
        "notes": "Crypto prices, market charts, global data. User has PAID subscription.",
        "priority": "HIGH",
    },
    {
        "env_var": "COINMARKETCAP_API_KEY",
        "name": "CoinMarketCap",
        "website": "https://coinmarketcap.com",
        "signup": "https://coinmarketcap.com/api/",
        "category": "Market Data",
        "cost": "Free (333 req/day) | Hobbyist $29/mo",
        "notes": "Crypto listings, market cap, volume, historical data.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "ALPHAVANTAGE_API_KEY",
        "name": "Alpha Vantage",
        "website": "https://www.alphavantage.co",
        "signup": "https://www.alphavantage.co/support/#api-key",
        "category": "Market Data",
        "cost": "Free (25 req/day) | Premium from $49/mo",
        "notes": "Stocks, forex, crypto, technical indicators, fundamentals.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "TWELVE_DATA_API_KEY",
        "name": "Twelve Data",
        "website": "https://twelvedata.com",
        "signup": "https://twelvedata.com/pricing",
        "category": "Market Data",
        "cost": "Free (8 req/min) | Basic $29/mo",
        "notes": "Stocks, forex, crypto, ETFs. Good technical indicator API.",
        "priority": "LOW",
    },
    {
        "env_var": "IEX_CLOUD_API_KEY",
        "name": "IEX Cloud",
        "website": "https://iexcloud.io",
        "signup": "https://iexcloud.io/cloud-login#/register/",
        "category": "Market Data",
        "cost": "Free (50k msg/mo) | Launch $9/mo",
        "notes": "US stocks, ETFs, mutual funds. Good for fundamentals.",
        "priority": "LOW",
    },
    {
        "env_var": "EODHD_API_KEY",
        "name": "EODHD (EOD Historical Data)",
        "website": "https://eodhd.com",
        "signup": "https://eodhd.com/register",
        "category": "Market Data",
        "cost": "Free (20 req/day) | All-In-One $79.99/mo",
        "notes": "70+ exchanges, fundamentals, options, insider trading data.",
        "priority": "LOW",
    },
    {
        "env_var": "INTRINIO_API_KEY",
        "name": "Intrinio",
        "website": "https://intrinio.com",
        "signup": "https://intrinio.com/starter-plan",
        "category": "Market Data",
        "cost": "Free Starter | Developer from $50/mo",
        "notes": "Real-time + historical data, fundamentals, options.",
        "priority": "LOW",
    },

    # ── OPTIONS / ALTERNATIVES ──────────────────
    {
        "env_var": "TRADIER_API_KEY",
        "name": "Tradier",
        "website": "https://developer.tradier.com",
        "signup": "https://developer.tradier.com/user/sign_up",
        "category": "Options/Broker",
        "cost": "Free sandbox | Brokerage $0 commission",
        "notes": "Options chains with full Greeks, options execution, market data.",
        "priority": "HIGH",
    },
    {
        "env_var": "UNUSUAL_WHALES_API_KEY",
        "name": "Unusual Whales",
        "website": "https://unusualwhales.com",
        "signup": "https://unusualwhales.com/pricing",
        "category": "Alternative Data",
        "cost": "From $57/mo",
        "notes": "Options flow, dark pool, congress trading, whale alerts.",
        "priority": "HIGH",
    },

    # ── MACRO / ECONOMIC ────────────────────────
    {
        "env_var": "FRED_API_KEY",
        "name": "FRED (Federal Reserve)",
        "website": "https://fred.stlouisfed.org",
        "signup": "https://fred.stlouisfed.org/docs/api/api_key.html",
        "category": "Macro/Economic",
        "cost": "FREE (no limits)",
        "notes": "GDP, CPI, unemployment, interest rates, yield curve. Essential macro data.",
        "priority": "HIGH",
    },

    # ── ON-CHAIN / CRYPTO ANALYTICS ─────────────
    {
        "env_var": "WHALE_ALERT_API_KEY",
        "name": "Whale Alert",
        "website": "https://whale-alert.io",
        "signup": "https://whale-alert.io/signup",
        "category": "On-Chain Analytics",
        "cost": "Free (10 req/min) | Pro from $8.25/mo",
        "notes": "Real-time whale transaction monitoring across all major blockchains.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "SANTIMENT_API_KEY",
        "name": "Santiment",
        "website": "https://santiment.net",
        "signup": "https://app.santiment.net/",
        "category": "On-Chain Analytics",
        "cost": "Free tier | Pro from $44/mo",
        "notes": "Social volume, dev activity, on-chain metrics, MVRV, NVT ratios.",
        "priority": "MEDIUM",
    },

    # ── NEWS / SENTIMENT ────────────────────────
    {
        "env_var": "NEWS_API_KEY",
        "name": "NewsAPI",
        "website": "https://newsapi.org",
        "signup": "https://newsapi.org/register",
        "category": "News/Sentiment",
        "cost": "Free (100 req/day) | Business $449/mo",
        "notes": "News articles from 80,000+ sources. Good for sentiment analysis.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "REDDIT_CLIENT_ID",
        "name": "Reddit API",
        "website": "https://www.reddit.com/wiki/api/",
        "signup": "https://www.reddit.com/prefs/apps",
        "category": "News/Sentiment",
        "cost": "Free (60 req/min)",
        "notes": "WallStreetBets, Superstonk, crypto subreddits. Already integrated via PRAW.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "TWITTER_BEARER_TOKEN",
        "name": "X/Twitter API",
        "website": "https://developer.x.com",
        "signup": "https://developer.x.com/en/portal/petition/essential/basic-info",
        "category": "News/Sentiment",
        "cost": "Basic $100/mo | Pro $5000/mo",
        "notes": "Financial Twitter (FinTwit) sentiment. Useful but expensive.",
        "priority": "LOW",
    },
    {
        "env_var": "TRADESTIE_API_KEY",
        "name": "TradeStie (Reddit Sentiment)",
        "website": "https://tradestie.com",
        "signup": "https://tradestie.com/apps/reddit/api/",
        "category": "News/Sentiment",
        "cost": "Free",
        "notes": "Pre-processed Reddit sentiment scores for stocks.",
        "priority": "LOW",
    },

    # ── NO KEY REQUIRED ─────────────────────────
    {
        "env_var": None,
        "name": "Google Trends (pytrends)",
        "website": "https://trends.google.com",
        "signup": "N/A — library-based, no key needed",
        "category": "Sentiment",
        "cost": "FREE (no key)",
        "notes": "pip install pytrends. Search interest as sentiment proxy. Already built.",
        "priority": "DONE",
    },
    {
        "env_var": None,
        "name": "Fear & Greed Index",
        "website": "https://alternative.me/crypto/fear-and-greed-index/",
        "signup": "N/A — free API, no key needed",
        "category": "Sentiment",
        "cost": "FREE (no key)",
        "notes": "Crypto Fear & Greed Index. Contrarian buy/sell signals. Already built.",
        "priority": "DONE",
    },
    {
        "env_var": None,
        "name": "Yahoo Finance (yfinance)",
        "website": "https://finance.yahoo.com",
        "signup": "N/A — library-based, no key needed",
        "category": "Market Data",
        "cost": "FREE (no key)",
        "notes": "pip install yfinance. Stocks, ETFs, options, historical data.",
        "priority": "DONE",
    },
    {
        "env_var": None,
        "name": "ECB (European Central Bank)",
        "website": "https://data.ecb.europa.eu",
        "signup": "N/A — free, no key needed",
        "category": "Macro/Economic",
        "cost": "FREE (no key)",
        "notes": "Euro exchange rates, monetary policy data.",
        "priority": "DONE",
    },
    {
        "env_var": None,
        "name": "World Bank Open Data",
        "website": "https://data.worldbank.org",
        "signup": "N/A — free, no key needed",
        "category": "Macro/Economic",
        "cost": "FREE (no key)",
        "notes": "Global economic indicators. Already integrated.",
        "priority": "DONE",
    },

    # ── NOTIFICATIONS ───────────────────────────
    {
        "env_var": "TELEGRAM_BOT_TOKEN",
        "name": "Telegram Bot",
        "website": "https://core.telegram.org/bots",
        "signup": "https://t.me/BotFather",
        "category": "Notifications",
        "cost": "FREE",
        "notes": "Trading alerts via Telegram. Talk to @BotFather to create bot.",
        "priority": "MEDIUM",
    },
    {
        "env_var": "DISCORD_WEBHOOK_URL",
        "name": "Discord Webhooks",
        "website": "https://discord.com",
        "signup": "Server Settings → Integrations → Webhooks",
        "category": "Notifications",
        "cost": "FREE",
        "notes": "Trading alerts in Discord channel.",
        "priority": "LOW",
    },
    {
        "env_var": "SLACK_WEBHOOK_URL",
        "name": "Slack Webhooks",
        "website": "https://api.slack.com",
        "signup": "https://api.slack.com/messaging/webhooks",
        "category": "Notifications",
        "cost": "FREE",
        "notes": "Trading alerts in Slack channel.",
        "priority": "LOW",
    },

    # ── BLOCKCHAIN / WEB3 ───────────────────────
    {
        "env_var": "ETH_RPC_URL",
        "name": "Ethereum RPC (Infura/Alchemy)",
        "website": "https://www.infura.io | https://www.alchemy.com",
        "signup": "https://app.infura.io/register | https://dashboard.alchemy.com/signup",
        "category": "Blockchain",
        "cost": "Free tier available on both",
        "notes": "Required for DeFi arbitrage. Infura or Alchemy for Ethereum RPC.",
        "priority": "MEDIUM",
    },
]


def check_env_var(env_var: str) -> str:
    """Check if an environment variable is set and non-empty."""
    if env_var is None:
        return "N/A"
    value = os.environ.get(env_var, '')
    if not value:
        return "❌ MISSING"
    if value in ('0', 'false'):
        return "❌ NOT SET"
    return "✅ SET"


def print_registry(show_missing_only: bool = False, output_json: bool = False):
    """Print the full API registry with status."""
    entries = []
    for api in REGISTRY:
        status = check_env_var(api["env_var"])
        entry = {**api, "status": status}
        if show_missing_only and status not in ("❌ MISSING", "❌ NOT SET"):
            continue
        entries.append(entry)

    if output_json:
        clean = [{k: v for k, v in e.items()} for e in entries]
        print(json.dumps(clean, indent=2))
        return

    # Group by category
    categories = {}
    for entry in entries:
        cat = entry["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(entry)

    # Summary counts
    total = len(REGISTRY)
    configured = sum(1 for api in REGISTRY if check_env_var(api["env_var"]) == "✅ SET")
    no_key = sum(1 for api in REGISTRY if api["env_var"] is None)
    missing = total - configured - no_key

    print("=" * 80)
    print("  AAC API REGISTRY — All Integrations & Their Status")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print(f"\n  📊 Summary: {configured} configured | {missing} need keys | {no_key} free (no key)")
    print(f"     Total APIs: {total}\n")

    for cat, apis in categories.items():
        print(f"\n{'─' * 80}")
        print(f"  📁 {cat.upper()}")
        print(f"{'─' * 80}")

        for api in apis:
            status = api["status"]
            pri = api["priority"]
            pri_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪", "DONE": "✅"}.get(pri, "⚪")

            print(f"\n  {status}  {pri_icon} {api['name']}  [{pri}]")
            print(f"     🌐  {api['website']}")
            print(f"     📝  {api['signup']}")
            print(f"     💰  {api['cost']}")
            if api["env_var"]:
                print(f"     🔑  .env key: {api['env_var']}")
            print(f"     📋  {api['notes']}")

    # Action list
    print(f"\n\n{'=' * 80}")
    print("  🎯 RECOMMENDED ACTION ORDER")
    print(f"{'=' * 80}")

    priority_order = ["HIGH", "MEDIUM", "LOW"]
    action_num = 1
    for pri in priority_order:
        missing_apis = [
            api for api in REGISTRY
            if api["priority"] == pri and check_env_var(api["env_var"]) == "❌ MISSING"
        ]
        if missing_apis:
            print(f"\n  {'🔴' if pri == 'HIGH' else '🟡' if pri == 'MEDIUM' else '⚪'} {pri} PRIORITY:")
            for api in missing_apis:
                print(f"    {action_num}. {api['name']}")
                print(f"       Sign up: {api['signup']}")
                print(f"       Add to .env: {api['env_var']}=your_key_here")
                action_num += 1

    print(f"\n{'=' * 80}")
    print(f"  Run with --missing to show only unconfigured APIs")
    print(f"  Run with --json for machine-readable output")
    print(f"{'=' * 80}\n")


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="AAC API Registry — list all APIs and their configuration status"
    )
    parser.add_argument('--missing', action='store_true',
                        help='Show only APIs that need keys')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    parser.add_argument('--category', type=str, default='',
                        help='Filter by category')
    args = parser.parse_args()

    print_registry(
        show_missing_only=args.missing,
        output_json=args.json,
    )


if __name__ == '__main__':
    main()

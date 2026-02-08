#!/usr/bin/env python3
"""
AAC Advanced Arbitrage Integration
==================================

Comprehensive integration of multiple data sources for enhanced arbitrage:
- Polygon.io: High-quality US market data and options
- Finnhub: Real-time quotes and news sentiment
- Federal Reserve FRED: Economic indicators for macro arbitrage
- OECD Data: Global economic data
- CBOE: Options and volatility data

This script demonstrates cross-validation and multi-source arbitrage strategies.
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data"""
    type: str
    symbol: str
    description: str
    expected_return: float
    confidence: float
    timestamp: datetime
    data_sources: List[str]

@dataclass
class MarketDataPoint:
    """Unified market data point"""
    symbol: str
    price: float
    volume: Optional[int]
    timestamp: datetime
    source: str
    additional_data: Dict[str, Any]

class AdvancedArbitrageEngine:
    """Advanced arbitrage engine using multiple data sources"""

    def __init__(self):
        self.opportunities: List[ArbitrageOpportunity] = []
        self.market_data: Dict[str, List[MarketDataPoint]] = {}

    async def collect_multi_source_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Collect data from multiple sources for given symbols"""
        print(f"📊 Collecting multi-source data for {len(symbols)} symbols...")

        # Initialize data collectors
        collectors = {
            'polygon': self._collect_polygon_data,
            'finnhub': self._collect_finnhub_data,
            'fred': self._collect_fred_data,
            'oecd': self._collect_oecd_data,
            'cboe': self._collect_cboe_data
        }

        all_data = {}

        for symbol in symbols:
            all_data[symbol] = {}

            # Collect from each available source
            for source_name, collector in collectors.items():
                try:
                    data = await collector(symbol)
                    if data:
                        all_data[symbol][source_name] = data
                        print(f"   ✅ {symbol} from {source_name}")
                    else:
                        print(f"   ❌ {symbol} from {source_name} (no data)")
                except Exception as e:
                    print(f"   ⚠️  {symbol} from {source_name}: {str(e)[:50]}...")

        return all_data

    async def _collect_polygon_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect data from Polygon.io"""
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                # Get quote
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
                params = {'apiKey': api_key}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'results' in data and data['results']:
                            result = data['results'][0]
                            return {
                                'price': result['c'],
                                'volume': result['v'],
                                'timestamp': datetime.fromtimestamp(result['t'] / 1000),
                                'source': 'polygon'
                            }
        except Exception as e:
            pass
        return None

    async def _collect_finnhub_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect data from Finnhub"""
        api_key = os.getenv('FINNHUB_API_KEY')
        if not api_key:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                # Get quote
                url = f"https://finnhub.io/api/v1/quote"
                params = {'symbol': symbol, 'token': api_key}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'c' in data:
                            return {
                                'price': data['c'],
                                'change': data['d'],
                                'change_percent': data['dp'],
                                'timestamp': datetime.now(),
                                'source': 'finnhub'
                            }
        except Exception as e:
            pass
        return None

    async def _collect_fred_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect economic data from FRED (for macro arbitrage)"""
        # This is a simplified example - in practice you'd map symbols to economic indicators
        if symbol not in ['GDP', 'UNRATE', 'FEDFUNDS']:
            return None

        try:
            series_id = {
                'GDP': 'GDP',
                'UNRATE': 'UNRATE',
                'FEDFUNDS': 'FEDFUNDS'
            }.get(symbol)

            if not series_id:
                return None

            async with aiohttp.ClientSession() as session:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': 'demo',  # Using demo key for testing
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'observations' in data and data['observations']:
                            obs = data['observations'][0]
                            return {
                                'value': float(obs['value']) if obs['value'] != '.' else None,
                                'date': obs['date'],
                                'source': 'fred'
                            }
        except Exception as e:
            pass
        return None

    async def _collect_oecd_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect data from OECD"""
        # Simplified OECD data collection
        try:
            async with aiohttp.ClientSession() as session:
                # Example: GDP data
                if symbol == 'OECD_GDP':
                    url = "https://stats.oecd.org/SDMX-JSON/data/SNA_TABLE1/AUS+USA+GBR+DEU.B1_GE.HCPC/all?startTime=2023&endTime=2023"

                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                'data_type': 'GDP',
                                'countries': ['AUS', 'USA', 'GBR', 'DEU'],
                                'source': 'oecd'
                            }
        except Exception as e:
            pass
        return None

    async def _collect_cboe_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect options data from CBOE"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get VIX data as example
                if symbol == 'VIX':
                    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/daily_prices_VIX.json"

                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'data' in data and data['data']:
                                latest = data['data'][-1]
                                return {
                                    'price': latest.get('close', 0),
                                    'date': latest.get('date', ''),
                                    'source': 'cboe'
                                }
        except Exception as e:
            pass
        return None

    def analyze_cross_source_arbitrage(self, multi_source_data: Dict[str, Dict[str, Any]]) -> List[ArbitrageOpportunity]:
        """Analyze arbitrage opportunities across multiple data sources"""
        opportunities = []

        for symbol, sources in multi_source_data.items():
            if len(sources) < 2:
                continue

            # Extract prices from different sources
            prices = {}
            for source_name, data in sources.items():
                if 'price' in data and data['price']:
                    prices[source_name] = data['price']

            if len(prices) >= 2:
                # Check for price discrepancies
                price_values = list(prices.values())
                max_price = max(price_values)
                min_price = min(price_values)
                spread = max_price - min_price
                avg_price = sum(price_values) / len(price_values)

                if spread / avg_price > 0.001:  # 0.1% spread
                    opportunity = ArbitrageOpportunity(
                        type="cross_source_price_discrepancy",
                        symbol=symbol,
                        description=f"Price discrepancy across sources: ${min_price:.2f} - ${max_price:.2f}",
                        expected_return=spread / avg_price,
                        confidence=min(0.8, spread / avg_price * 100),  # Higher spread = higher confidence
                        timestamp=datetime.now(),
                        data_sources=list(prices.keys())
                    )
                    opportunities.append(opportunity)

        return opportunities

    def analyze_sentiment_arbitrage(self, multi_source_data: Dict[str, Dict[str, Any]]) -> List[ArbitrageOpportunity]:
        """Analyze sentiment-based arbitrage opportunities"""
        opportunities = []

        for symbol, sources in multi_source_data.items():
            # Look for sentiment data from Finnhub
            if 'finnhub' in sources:
                finnhub_data = sources['finnhub']
                if 'change_percent' in finnhub_data:
                    change_pct = finnhub_data['change_percent']

                    # Extreme sentiment signals
                    if change_pct > 5:  # Strong positive momentum
                        opportunity = ArbitrageOpportunity(
                            type="momentum_arbitrage",
                            symbol=symbol,
                            description=f"Strong positive momentum: +{change_pct:.1f}%",
                            expected_return=abs(change_pct) / 100,
                            confidence=min(0.7, abs(change_pct) / 10),
                            timestamp=datetime.now(),
                            data_sources=['finnhub']
                        )
                        opportunities.append(opportunity)
                    elif change_pct < -5:  # Strong negative momentum
                        opportunity = ArbitrageOpportunity(
                            type="contrarian_arbitrage",
                            symbol=symbol,
                            description=f"Strong negative momentum: {change_pct:.1f}%",
                            expected_return=abs(change_pct) / 100,
                            confidence=min(0.7, abs(change_pct) / 10),
                            timestamp=datetime.now(),
                            data_sources=['finnhub']
                        )
                        opportunities.append(opportunity)

        return opportunities

    def analyze_macro_arbitrage(self, multi_source_data: Dict[str, Dict[str, Any]]) -> List[ArbitrageOpportunity]:
        """Analyze macro-economic arbitrage opportunities"""
        opportunities = []

        # Look for economic indicators
        economic_symbols = ['GDP', 'UNRATE', 'FEDFUNDS', 'VIX']

        for symbol in economic_symbols:
            if symbol in multi_source_data and multi_source_data[symbol]:
                sources = multi_source_data[symbol]

                # Analyze economic signals
                if symbol == 'FEDFUNDS' and 'fred' in sources:
                    rate = sources['fred'].get('value')
                    if rate is not None:
                        if rate > 5.0:  # High interest rates
                            opportunity = ArbitrageOpportunity(
                                type="macro_rate_arbitrage",
                                symbol=symbol,
                                description=f"High Fed Funds Rate: {rate:.2f}% - potential for rate-sensitive arbitrage",
                                expected_return=0.02,  # Estimated 2% opportunity
                                confidence=0.6,
                                timestamp=datetime.now(),
                                data_sources=['fred']
                            )
                            opportunities.append(opportunity)

                elif symbol == 'VIX' and 'cboe' in sources:
                    vix_level = sources['cboe'].get('price')
                    if vix_level and vix_level > 30:  # High volatility
                        opportunity = ArbitrageOpportunity(
                            type="volatility_arbitrage",
                            symbol=symbol,
                            description=f"High VIX: {vix_level:.1f} - volatility arbitrage opportunity",
                            expected_return=0.05,  # Estimated 5% opportunity
                            confidence=min(0.8, vix_level / 50),
                            timestamp=datetime.now(),
                            data_sources=['cboe']
                        )
                        opportunities.append(opportunity)

        return opportunities

async def run_advanced_arbitrage_demo():
    """Run comprehensive arbitrage analysis demo"""
    print("🚀 AAC Advanced Multi-Source Arbitrage Analysis")
    print("=" * 60)

    engine = AdvancedArbitrageEngine()

    # Define test symbols
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    economic_symbols = ['GDP', 'UNRATE', 'FEDFUNDS', 'VIX', 'OECD_GDP']

    all_symbols = stock_symbols + economic_symbols

    # Collect multi-source data
    print(f"\n📊 Phase 1: Collecting Data from {len(all_symbols)} Symbols")
    print("-" * 50)

    multi_source_data = await engine.collect_multi_source_data(all_symbols)

    # Analyze different types of arbitrage
    print("\n🎯 Phase 2: Analyzing Arbitrage Opportunities")
    print("-" * 50)

    # Cross-source price discrepancies
    print("\n🔄 Cross-Source Price Discrepancies:")
    cross_opportunities = engine.analyze_cross_source_arbitrage(multi_source_data)
    for opp in cross_opportunities:
        print(f"   {opp.symbol}: {opp.description}")
        print(f"      Expected Return: {opp.expected_return:.1%}")
        print(f"      Confidence: {opp.confidence:.1f}, Sources: {', '.join(opp.data_sources)}")

    # Sentiment-based opportunities
    print("\n📰 Sentiment-Based Opportunities:")
    sentiment_opportunities = engine.analyze_sentiment_arbitrage(multi_source_data)
    for opp in sentiment_opportunities:
        print(f"   {opp.symbol}: {opp.description}")
        print(f"      Expected Return: {opp.expected_return:.1%}")
        print(f"      Confidence: {opp.confidence:.1f}")

    # Macro arbitrage opportunities
    print("\n📊 Macro Arbitrage Opportunities:")
    macro_opportunities = engine.analyze_macro_arbitrage(multi_source_data)
    for opp in macro_opportunities:
        print(f"   {opp.symbol}: {opp.description}")
        print(f"      Expected Return: {opp.expected_return:.1%}")
        print(f"      Confidence: {opp.confidence:.1f}")

    # Summary
    all_opportunities = cross_opportunities + sentiment_opportunities + macro_opportunities

    print("\n📈 Summary:")
    print(f"   Total opportunities found: {len(all_opportunities)}")
    print(f"   Cross-source: {len(cross_opportunities)}")
    print(f"   Sentiment-based: {len(sentiment_opportunities)}")
    print(f"   Macro: {len(macro_opportunities)}")

    if all_opportunities:
        # Sort by expected return
        sorted_opps = sorted(all_opportunities, key=lambda x: x.expected_return, reverse=True)
        print("\n🏆 Top Opportunity:")
        top = sorted_opps[0]
        print(f"   {top.symbol} ({top.type}): {top.expected_return:.1%} expected return")

    # Data source coverage
    print("\n📊 Data Source Coverage:")
    total_symbols = len(all_symbols)
    sources_count = {}
    for symbol_data in multi_source_data.values():
        for source in symbol_data.keys():
            sources_count[source] = sources_count.get(source, 0) + 1

    for source, count in sources_count.items():
        coverage = count / total_symbols * 100
        print(f"   {source}: {count}/{total_symbols} symbols ({coverage:.1f}%)")

    print("\n💡 Integration Recommendations:")
    print("   1. Add POLYGON_API_KEY and FINNHUB_API_KEY to .env for full coverage")
    print("   2. Implement real-time monitoring for detected opportunities")
    print("   3. Add risk management for multi-source arbitrage strategies")
    print("   4. Combine with existing AAC arbitrage strategies")
    print("   5. Add automated execution for high-confidence opportunities")

if __name__ == "__main__":
    asyncio.run(run_advanced_arbitrage_demo())
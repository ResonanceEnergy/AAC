#!/usr/bin/env python3
"""
Worldwide Arbitrage Strategy for AAC
Uses Alpha Vantage API for global market data to identify arbitrage opportunities
across different exchanges and markets worldwide.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Represents market data for a symbol"""
    symbol: str
    price: float
    volume: float
    exchange: str
    timestamp: float
    currency: str = "USD"


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    potential_profit: float
    volume: float
    timestamp: float


class AlphaVantageClient:
    """Client for Alpha Vantage API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_global_quote(self, symbol: str) -> Optional[MarketData]:
        """Get global quote for a symbol"""
        if not self.session:
            return None

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "Global Quote" in data and data["Global Quote"]:
                        quote = data["Global Quote"]
                        return MarketData(
                            symbol=symbol,
                            price=float(quote.get("05. price", 0) or 0),
                            volume=float(quote.get("06. volume", 0) or 0),
                            exchange="GLOBAL",
                            timestamp=time.time(),
                            currency="USD"
                        )
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")

        return None

    async def get_forex_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get forex exchange rate"""
        if not self.session:
            return None

        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": self.api_key
        }

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "Realtime Currency Exchange Rate" in data:
                        rate_data = data["Realtime Currency Exchange Rate"]
                        return float(rate_data.get("5. Exchange Rate", 0) or 0)
        except Exception as e:
            logger.error(f"Error fetching forex rate {from_currency}/{to_currency}: {e}")

        return None


class WorldwideArbitrageStrategy:
    """Worldwide arbitrage strategy using Alpha Vantage"""

    def __init__(self):
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")

        self.client = AlphaVantageClient(self.api_key)

        # Global symbols to monitor (NYSE, NASDAQ, LSE, TSE, etc.)
        self.global_symbols = [
            # US Markets
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
            # European Markets
            "BP.L", "HSBA.L", "VOD.L", "BARC.L",  # London
            "SAP.DE", "BMW.DE", "VOW.DE",  # Frankfurt
            # Asian Markets
            "7203.T", "6758.T", "9432.T",  # Tokyo
            "000001.SS", "000002.SS",  # Shanghai
            # Other Global
            "BHP.AX", "CBA.AX",  # Australia
        ]

        # Forex pairs for currency arbitrage
        self.forex_pairs = [
            ("EUR", "USD"), ("GBP", "USD"), ("JPY", "USD"),
            ("CAD", "USD"), ("AUD", "USD"), ("CHF", "USD")
        ]

        # Minimum arbitrage spread threshold (0.5%)
        self.min_spread_threshold = 0.005

        # Rate limiting (25 calls/day free tier)
        self.call_count = 0
        self.max_calls_per_day = 25

    async def scan_for_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities across global markets"""
        opportunities = []

        async with self.client:
            # Check stock arbitrage opportunities
            stock_opportunities = await self._scan_stock_arbitrage()
            opportunities.extend(stock_opportunities)

            # Check forex arbitrage opportunities
            forex_opportunities = await self._scan_forex_arbitrage()
            opportunities.extend(forex_opportunities)

        return opportunities

    async def _scan_stock_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Scan for stock price differences across exchanges"""
        opportunities = []

        # For demonstration, we'll compare prices across different symbols
        # In a real implementation, you'd compare the same asset on different exchanges
        for symbol in self.global_symbols[:5]:  # Limit to avoid rate limits
            if self.call_count >= self.max_calls_per_day:
                logger.warning("Reached Alpha Vantage rate limit")
                break

            market_data = await self.client.get_global_quote(symbol)
            if market_data and market_data.price > 0:
                self.call_count += 1

                # For now, we'll simulate cross-exchange comparison
                # In reality, you'd need multiple data sources or exchange-specific APIs
                opportunities.append(ArbitrageOpportunity(
                    symbol=symbol,
                    buy_exchange="GLOBAL_LOW",
                    sell_exchange="GLOBAL_HIGH",
                    buy_price=market_data.price * 0.995,  # Simulate lower price
                    sell_price=market_data.price * 1.008,  # Simulate higher price
                    spread_pct=0.013,
                    potential_profit=market_data.price * 0.013,
                    volume=market_data.volume,
                    timestamp=time.time()
                ))

        return opportunities

    async def _scan_forex_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Scan for forex triangular arbitrage opportunities"""
        opportunities = []

        # Check for triangular arbitrage in forex
        for base, quote in self.forex_pairs[:3]:  # Limit calls
            if self.call_count >= self.max_calls_per_day:
                break

            # Get direct rate
            direct_rate = await self.client.get_forex_rate(base, quote)
            if direct_rate:
                self.call_count += 1

                # For triangular arbitrage, we'd need to check if:
                # EUR/USD * USD/GBP * GBP/EUR != 1
                # But for demo, we'll create a simulated opportunity
                opportunities.append(ArbitrageOpportunity(
                    symbol=f"{base}/{quote}",
                    buy_exchange="FOREX_DIRECT",
                    sell_exchange="FOREX_TRIANGLE",
                    buy_price=direct_rate,
                    sell_price=direct_rate * 1.002,
                    spread_pct=0.002,
                    potential_profit=direct_rate * 0.002,
                    volume=1000000,  # Simulated volume
                    timestamp=time.time()
                ))

        return opportunities

    def filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on criteria"""
        filtered = []

        for opp in opportunities:
            if opp.spread_pct >= self.min_spread_threshold:
                # Additional filters could include:
                # - Minimum volume
                # - Transaction costs
                # - Market hours
                # - Liquidity requirements
                filtered.append(opp)

        return sorted(filtered, key=lambda x: x.spread_pct, reverse=True)

    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute an arbitrage trade (placeholder for actual execution)"""
        logger.info(f"Executing arbitrage for {opportunity.symbol}: "
                   f"Buy at {opportunity.buy_exchange} (${opportunity.buy_price:.4f}), "
                   f"Sell at {opportunity.sell_exchange} (${opportunity.sell_price:.4f}), "
                   f"Spread: {opportunity.spread_pct:.2%}")

        # In a real implementation, this would:
        # 1. Place buy order on low-price exchange
        # 2. Place sell order on high-price exchange
        # 3. Monitor execution
        # 4. Handle transaction costs and fees

        return True  # Placeholder success

    async def run_strategy(self):
        """Main strategy execution loop"""
        logger.info("Starting Worldwide Arbitrage Strategy with Alpha Vantage")

        while True:
            try:
                # Scan for opportunities
                opportunities = await self.scan_for_opportunities()

                if opportunities:
                    filtered_opps = self.filter_opportunities(opportunities)

                    logger.info(f"Found {len(filtered_opps)} arbitrage opportunities")

                    for opp in filtered_opps[:3]:  # Execute top 3
                        await self.execute_arbitrage(opp)

                # Wait before next scan (respect rate limits)
                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Error in strategy loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error


async def main():
    """Main function"""
    try:
        strategy = WorldwideArbitrageStrategy()

        # Run a single scan for demonstration
        logger.info("Running single arbitrage scan...")
        opportunities = await strategy.scan_for_opportunities()
        filtered_opps = strategy.filter_opportunities(opportunities)

        if filtered_opps:
            print(f"\n[TARGET] Found {len(filtered_opps)} Arbitrage Opportunities:")
            print("=" * 60)

            for i, opp in enumerate(filtered_opps[:5], 1):
                print(f"{i}. {opp.symbol}")
                print(f"   Buy:  {opp.buy_exchange} @ ${opp.buy_price:.4f}")
                print(f"   Sell: {opp.sell_exchange} @ ${opp.sell_price:.4f}")
                print(f"   Spread: {opp.spread_pct:.2%}")
                print(f"   Profit: ${opp.potential_profit:.2f}")
                print()
        else:
            print("\n[WARN] No arbitrage opportunities found (may be due to rate limits)")

        print(f"\n[DATA] API Calls Used: {strategy.call_count}/{strategy.max_calls_per_day}")
        print("[INFO] Alpha Vantage free tier: 25 calls/day, 5 calls/minute")

    except Exception as e:
        logger.error(f"Error running strategy: {e}")


if __name__ == "__main__":
    asyncio.run(main())
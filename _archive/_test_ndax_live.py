#!/usr/bin/env python3
"""
Quick NDAX live connection test.
Tests: connect, fetch ticker, fetch balances, fetch order book.
Does NOT place any orders.
"""
import asyncio
import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()


async def main():
    from TradingExecution.exchange_connectors.ndax_connector import NDAXConnector

    api_key = os.getenv("NDAX_API_KEY", "")
    api_secret = os.getenv("NDAX_API_SECRET", "")
    user_id = os.getenv("NDAX_USER_ID", "")

    if not api_key or not api_secret:
        print("FAIL: NDAX_API_KEY or NDAX_API_SECRET not set in .env")
        return

    print(f"NDAX credentials: key={api_key[:4]}****, user_id={user_id}")
    print()

    connector = NDAXConnector(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,  # NDAX doesn't have a public testnet — mainnet with read-only calls is safe
    )

    try:
        # 1. Connect
        print("[1] Connecting to NDAX...")
        ok = await connector.connect()
        print(f"    Connected: {ok}")
        print()

        # 2. Fetch BTC/CAD ticker
        print("[2] Fetching BTC/CAD ticker...")
        ticker = await connector.get_ticker("BTC/CAD")
        print(f"    BTC/CAD: bid=${ticker.bid:,.2f} ask=${ticker.ask:,.2f} last=${ticker.last:,.2f}")
        print(f"    24h volume: ${ticker.volume_24h:,.0f} CAD")
        print()

        # 3. Fetch ETH/CAD ticker
        print("[3] Fetching ETH/CAD ticker...")
        ticker_eth = await connector.get_ticker("ETH/CAD")
        print(f"    ETH/CAD: bid=${ticker_eth.bid:,.2f} ask=${ticker_eth.ask:,.2f} last=${ticker_eth.last:,.2f}")
        print()

        # 4. Fetch order book
        print("[4] Fetching BTC/CAD order book (top 5)...")
        book = await connector.get_orderbook("BTC/CAD", limit=5)
        print(f"    Top bids: {[(f'${b[0]:,.2f}', f'{b[1]:.6f}') for b in book.bids[:3]]}")
        print(f"    Top asks: {[(f'${a[0]:,.2f}', f'{a[1]:.6f}') for a in book.asks[:3]]}")
        spread = book.asks[0][0] - book.bids[0][0] if book.asks and book.bids else 0
        spread_pct = (spread / book.bids[0][0] * 100) if book.bids and book.bids[0][0] > 0 else 0
        print(f"    Spread: ${spread:,.2f} ({spread_pct:.3f}%)")
        print()

        # 5. Fetch balances (requires authenticated API)
        print("[5] Fetching account balances...")
        try:
            balances = await connector.get_balances()
            if balances:
                for asset, bal in sorted(balances.items()):
                    total = bal.free + bal.locked
                    if total > 0.0001:
                        print(f"    {asset}: free={bal.free:.8f} locked={bal.locked:.8f} total={total:.8f}")
            else:
                print("    (no balances found — account may be empty)")
        except Exception as e:
            print(f"    Balance fetch failed: {e}")
            print("    (This is OK if the API key is read-only or trade-only)")

        print()
        print("=" * 50)
        print("NDAX CONNECTION TEST: PASSED")
        print("Exchange is LIVE and responding.")
        print("=" * 50)

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

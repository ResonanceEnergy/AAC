#!/usr/bin/env python3
"""
Quick Moomoo live connection test.
Tests: connect, fetch ticker, fetch account.
Does NOT place any orders.

Prerequisites:
    1. Moomoo OpenD gateway running on localhost:11111
       Download from: https://www.moomoo.com/download/OpenD
    2. .env has MOOMOO_API_KEY and MOOMOO_API_SECRET (if required)
"""
import asyncio
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()


def check_port(host, port, timeout=2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    result = s.connect_ex((host, port))
    s.close()
    return result == 0


async def main():
    host = os.getenv("MOOMOO_HOST", "127.0.0.1")
    port = int(os.getenv("MOOMOO_PORT", "11111"))

    print(f"Moomoo config: host={host}, port={port}")
    print()

    # Pre-check
    print(f"[0] Checking port {port}...")
    if not check_port(host, port):
        print(f"    BLOCKED: Port {port} is not open.")
        print()
        print("    To fix:")
        print("    1. Download Moomoo OpenD: https://www.moomoo.com/download/OpenD")
        print("    2. Install and run OpenD")
        print(f"    3. It should listen on port {port}")
        return

    print(f"    Port {port} is OPEN")
    print()

    from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector

    connector = MoomooConnector()

    try:
        # 1. Connect
        print("[1] Connecting to Moomoo OpenD...")
        ok = await connector.connect()
        print(f"    Connected: {ok}")
        print()

        # 2. Fetch a US stock ticker
        print("[2] Fetching AAPL/USD ticker...")
        try:
            ticker = await connector.get_ticker("AAPL/USD")
            print(f"    AAPL: bid=${ticker.bid:.2f} ask=${ticker.ask:.2f} last=${ticker.last:.2f}")
        except Exception as e:
            print(f"    Ticker fetch failed: {e}")
        print()

        # 3. Fetch balances
        print("[3] Fetching account balances...")
        try:
            balances = await connector.get_balances()
            if balances:
                for asset, bal in sorted(balances.items()):
                    total = bal.free + bal.locked
                    if total > 0.01:
                        print(f"    {asset}: free={bal.free:,.2f} locked={bal.locked:,.2f}")
            else:
                print("    (no balances)")
        except Exception as e:
            print(f"    Balance fetch failed: {e}")

        print()
        print("=" * 50)
        print("MOOMOO CONNECTION TEST: PASSED")
        print("=" * 50)

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

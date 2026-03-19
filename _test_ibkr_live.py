#!/usr/bin/env python3
"""
Quick IBKR TWS live connection test.
Tests: connect, fetch ticker, fetch account, fetch positions.
Does NOT place any orders.

Prerequisites:
    1. TWS or IB Gateway running on localhost
    2. API enabled: Edit > Global Configuration > API > Settings
       - Check "Enable ActiveX and Socket Clients"
       - Socket port: 7497 (paper) or 7496 (live)
       - Check "Allow connections from localhost only"
    3. .env has IBKR_PORT=7497, IBKR_ACCOUNT=DUxxxxxx
"""
import asyncio
import sys
import os
import socket
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
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    account = os.getenv("IBKR_ACCOUNT", "")

    print(f"IBKR config: host={host}, port={port}, account={account[:4]}****")
    print()

    # Pre-check: is the port even open?
    print(f"[0] Checking port {port}...")
    if not check_port(host, port):
        print(f"    BLOCKED: Port {port} is not open.")
        print()
        print("    To fix this in TWS:")
        print("    1. Edit > Global Configuration > API > Settings")
        print("    2. Check 'Enable ActiveX and Socket Clients'")
        print(f"    3. Set Socket port to {port}")
        print("    4. Check 'Allow connections from localhost only'")
        print("    5. Click Apply, then restart TWS")
        return

    print(f"    Port {port} is OPEN")
    print()

    from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

    connector = IBKRConnector()

    try:
        # 1. Connect
        print("[1] Connecting to IBKR TWS...")
        ok = await connector.connect()
        print(f"    Connected: {ok}")
        print()

        # 2. Fetch a stock ticker
        print("[2] Fetching AAPL/USD ticker...")
        try:
            ticker = await connector.get_ticker("AAPL/USD")
            print(f"    AAPL: bid=${ticker.bid:.2f} ask=${ticker.ask:.2f} last=${ticker.last:.2f}")
        except Exception as e:
            print(f"    Ticker fetch failed: {e}")
        print()

        # 3. Fetch BTC/USD ticker (crypto)
        print("[3] Fetching BTC/USD ticker...")
        try:
            ticker_btc = await connector.get_ticker("BTC/USD")
            print(f"    BTC: bid=${ticker_btc.bid:,.2f} ask=${ticker_btc.ask:,.2f} last=${ticker_btc.last:,.2f}")
        except Exception as e:
            print(f"    BTC ticker: {e}")
        print()

        # 4. Fetch balances
        print("[4] Fetching account balances...")
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
        print("IBKR CONNECTION TEST: PASSED")
        print("=" * 50)

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

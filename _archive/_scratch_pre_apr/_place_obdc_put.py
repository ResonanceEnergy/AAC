#!/usr/bin/env python3
"""
AAC War Room -- BUY OBDC PUT (March 27, 2026)
OBDC $7.50P July 17 2026, limit $0.20, max contracts
"""
import asyncio
import io
import os
import sys

if hasattr(sys.stdout, "buffer") and (sys.stdout is None or sys.stdout.encoding.lower() != "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

import nest_asyncio; nest_asyncio.apply()
from shared.config_loader import load_env_file; load_env_file()
os.environ["IBKR_PORT"] = "7496"
from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

# ============================================================================
SYMBOL      = "OBDC"
STRIKE      = 7.50
EXPIRY      = "20260717"   # July 17, 2026
RIGHT       = "P"          # PUT
LIMIT_PRICE = 0.20
QUANTITY    = 11           # Will adjust based on cash


async def main():
    mode = "dry"
    if "--live" in sys.argv:
        mode = "live"
    elif "--price-only" in sys.argv:
        mode = "price"

    print("=" * 70)
    print(f"  AAC WAR ROOM -- BUY OBDC PUT")
    print(f"  Mode: {'*** LIVE EXECUTION ***' if mode == 'live' else 'DRY RUN' if mode == 'dry' else 'PRICE SCAN ONLY'}")
    print("=" * 70)

    print(f"\n  Order:   BUY {QUANTITY}x {SYMBOL} ${STRIKE}P  exp {EXPIRY[:4]}-{EXPIRY[4:6]}-{EXPIRY[6:]}")
    print(f"  Limit:   ${LIMIT_PRICE:.2f} per contract")
    print(f"  Premium: ${QUANTITY * LIMIT_PRICE * 100:,.0f}")

    if mode == "live":
        print("\n  WARNING: LIVE MODE")
        print("  Press Ctrl+C within 5 seconds to abort...")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n  Aborted."); return

    connector = IBKRConnector()
    await connector.connect()
    ib = connector._ib
    acct = ib.managedAccounts()[0]

    # Account
    summaries = ib.accountSummary(acct)
    for s in summaries:
        if s.tag == "SettledCash" and s.currency == "CAD":
            print(f"\n  Settled Cash: ${float(s.value):,.2f} CAD")
        if s.tag == "AvailableFunds" and s.currency == "CAD":
            print(f"  Available:    ${float(s.value):,.2f} CAD")

    # Qualify contract
    from ib_insync import Option as IbOption
    contract = IbOption(SYMBOL, EXPIRY, STRIKE, RIGHT, "SMART", currency="USD")
    qualified = ib.qualifyContracts(contract)

    if not qualified:
        print(f"\n  ERROR: {SYMBOL} ${STRIKE}P {EXPIRY} NOT FOUND.")
        print("  Trying nearby strikes...")
        for s in [5.0, 7.5, 10.0, 12.5, 15.0]:
            test = IbOption(SYMBOL, EXPIRY, s, RIGHT, "SMART", currency="USD")
            q = ib.qualifyContracts(test)
            if q:
                print(f"    FOUND: ${s} strike -> conId={q[0].conId}")
        await connector.disconnect(); return

    contract = qualified[0]
    print(f"\n  Contract qualified: conId={contract.conId} {contract.localSymbol}")

    ib.reqMarketDataType(3)
    ib.reqMktData(contract, "", False, False)
    await asyncio.sleep(3)
    ticker = ib.ticker(contract)

    bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
    ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
    last = ticker.last if ticker.last and ticker.last > 0 else 0
    mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else 0

    print(f"\n  LIVE QUOTE:")
    print(f"    Bid:  ${bid:.2f}")
    print(f"    Ask:  ${ask:.2f}")
    print(f"    Mid:  ${mid:.2f}")
    print(f"    Last: ${last:.2f}")
    print(f"    Our limit: ${LIMIT_PRICE:.2f}")

    if mode == "price":
        await connector.disconnect(); return

    if mode == "dry":
        print(f"\n  DRY RUN -- Would place: BUY {QUANTITY}x {SYMBOL} ${STRIKE}P @ ${LIMIT_PRICE:.2f}")
        print(f"  Run with --live to execute.")
        await connector.disconnect(); return

    # LIVE
    print(f"\n{'=' * 70}")
    print(f"  EXECUTING: BUY {QUANTITY}x {SYMBOL} ${STRIKE}P {EXPIRY} @ ${LIMIT_PRICE:.2f}")
    print(f"{'=' * 70}\n")

    try:
        from ib_insync import LimitOrder as IbLimitOrder
        order = IbLimitOrder('BUY', QUANTITY, LIMIT_PRICE)
        order.account = acct
        order.tif = 'DAY'
        trade = ib.placeOrder(contract, order)
        print(f"  Order submitted...")
        await asyncio.sleep(2)
        print(f"  Order ID:  {trade.order.orderId}")
        print(f"  Status:    {trade.orderStatus.status}")
        print(f"  Filled:    {trade.orderStatus.filled}")
        print(f"  Remaining: {trade.orderStatus.remaining}")
        print(f"  Premium:   ${QUANTITY * LIMIT_PRICE * 100:,.0f}")
        if trade.fills:
            for fill in trade.fills:
                print(f"  FILL: {fill.execution.shares}x @ ${fill.execution.price:.2f}")
    except Exception as e:
        print(f"  ORDER FAILED: {e}")

    print("\n  Waiting 5s for fills...")
    await asyncio.sleep(5)
    await connector.disconnect()
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())

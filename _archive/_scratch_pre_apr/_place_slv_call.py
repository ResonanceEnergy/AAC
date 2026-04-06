#!/usr/bin/env python3
r"""
AAC War Room — BUY SLV CALL (March 27, 2026)
=============================================
SLV $65 Call, June 18 2026 expiry, limit $7.95
Budget: $10,000 → 12 contracts ($9,540 premium)

IBKR LIVE port 7496, account U24346218.

Usage:
  .venv\Scripts\python.exe _place_slv_call.py              # DRY RUN (default)
  .venv\Scripts\python.exe _place_slv_call.py --live        # LIVE execution
  .venv\Scripts\python.exe _place_slv_call.py --price-only  # Just scan price
"""

import asyncio
import io
import os
import sys

if hasattr(sys.stdout, "buffer") and (sys.stdout is None or sys.stdout.encoding.lower() != "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

import nest_asyncio

nest_asyncio.apply()

from shared.config_loader import load_env_file

load_env_file()

# Override to live port 7496
os.environ["IBKR_PORT"] = "7496"

from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

# ============================================================================
#  ORDER SPEC
# ============================================================================
SYMBOL      = "SLV"
STRIKE      = 66.0
EXPIRY      = "20260618"   # June 18, 2026
RIGHT       = "C"          # CALL
SIDE        = "buy"
LIMIT_PRICE = 7.95
BUDGET      = 10_000.0

# Calculate quantity: floor(budget / (price * 100))
QUANTITY    = 8  # Reduced from 10 — settled cash constraint


async def main():
    mode = "dry"
    if "--live" in sys.argv:
        mode = "live"
    elif "--price-only" in sys.argv:
        mode = "price"

    print("=" * 70)
    print("  AAC WAR ROOM — BUY SLV CALL")
    print(f"  Mode: {'*** LIVE EXECUTION ***' if mode == 'live' else 'DRY RUN (add --live to execute)' if mode == 'dry' else 'PRICE SCAN ONLY'}")
    print(f"  Date: March 27, 2026")
    print("=" * 70)

    print(f"\n  Order:   BUY {QUANTITY}x SLV ${STRIKE:.0f}C  exp {EXPIRY[:4]}-{EXPIRY[4:6]}-{EXPIRY[6:]}")
    print(f"  Limit:   ${LIMIT_PRICE:.2f} per contract")
    print(f"  Premium: ${QUANTITY * LIMIT_PRICE * 100:,.0f}  (budget ${BUDGET:,.0f})")

    if mode == "live":
        print("\n  WARNING: LIVE MODE — Orders will be placed on IBKR account U24346218")
        print("  Press Ctrl+C within 5 seconds to abort...")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n  Aborted.")
            return

    # Connect to IBKR
    print("\n  Connecting to IBKR (port 7496 LIVE)...")
    connector = IBKRConnector()
    await connector.connect()
    ib = connector._ib
    acct = ib.managedAccounts()[0]

    # Account info
    summaries = ib.accountSummary(acct)
    net_liq = cash = buying_power = 0.0
    for s in summaries:
        if s.currency != "USD":
            continue
        try:
            val = float(s.value)
        except (ValueError, TypeError):
            continue
        if s.tag == "NetLiquidation":
            net_liq = val
        elif s.tag == "TotalCashValue":
            cash = val
        elif s.tag == "BuyingPower":
            buying_power = val

    print(f"\n  Account:       {acct}")
    print(f"  Net Liq:       ${net_liq:,.2f}")
    print(f"  Cash:          ${cash:,.2f}")
    print(f"  Buying Power:  ${buying_power:,.2f}")

    # Show existing SLV positions
    portfolio = ib.portfolio()
    slv_positions = [p for p in portfolio if p.contract.symbol == "SLV"]
    if slv_positions:
        print(f"\n  EXISTING SLV POSITIONS:")
        for pos in slv_positions:
            c = pos.contract
            print(f"    {c.secType} {c.symbol} ${c.strike:.0f}{c.right} exp {c.lastTradeDateOrContractMonth} "
                  f"qty={pos.position:.0f} mktVal=${pos.marketValue:.2f} uPnL=${pos.unrealizedPNL:+.2f}")

    # Check if contract exists
    from ib_insync import Option as IbOption
    contract = IbOption(SYMBOL, EXPIRY, STRIKE, RIGHT, "SMART", currency="USD")
    qualified = ib.qualifyContracts(contract)

    if not qualified:
        print(f"\n  ERROR: Contract SLV ${STRIKE:.0f}C {EXPIRY} NOT FOUND in IBKR chain.")
        print("  Check strike/expiry — the chain may not have this strike.")
        await connector.disconnect()
        return

    contract = qualified[0]
    print(f"\n  Contract qualified: conId={contract.conId} {contract.localSymbol}")

    # Get live quote
    ib.reqMarketDataType(3)  # delayed if no real-time subscription
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
        print("\n  Price scan complete.")
        await connector.disconnect()
        return

    if mode == "dry":
        print(f"\n  DRY RUN — Would place: BUY {QUANTITY}x SLV ${STRIKE:.0f}C {EXPIRY} @ ${LIMIT_PRICE:.2f}")
        print(f"  Total premium: ${QUANTITY * LIMIT_PRICE * 100:,.0f}")
        print("  Run with --live to execute for real.")
        await connector.disconnect()
        return

    # =========================================================================
    #  LIVE EXECUTION — Place order directly with explicit TIF
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  EXECUTING: BUY {QUANTITY}x SLV ${STRIKE:.0f}C {EXPIRY} @ ${LIMIT_PRICE:.2f}")
    print(f"{'=' * 70}\n")

    try:
        from ib_insync import LimitOrder as IbLimitOrder
        order = IbLimitOrder('BUY', QUANTITY, LIMIT_PRICE)
        order.account = acct
        order.tif = 'DAY'

        trade = ib.placeOrder(contract, order)
        print(f"  Order submitted, waiting for acknowledgment...")
        await asyncio.sleep(2)

        # Check status
        print(f"  ORDER PLACED!")
        print(f"  Order ID:  {trade.order.orderId}")
        print(f"  Status:    {trade.orderStatus.status}")
        print(f"  Filled:    {trade.orderStatus.filled}")
        print(f"  Remaining: {trade.orderStatus.remaining}")
        print(f"  Qty:       {QUANTITY}")
        print(f"  Price:     ${LIMIT_PRICE:.2f}")
        print(f"  Premium:   ${QUANTITY * LIMIT_PRICE * 100:,.0f}")

        if trade.fills:
            for fill in trade.fills:
                print(f"  FILL: {fill.execution.shares}x @ ${fill.execution.price:.2f}")
    except Exception as e:
        print(f"  ORDER FAILED: {e}")

    # Wait a moment for any fills
    print("\n  Waiting 5s for fill updates...")
    await asyncio.sleep(5)

    # Check order status
    open_orders = ib.openOrders()
    for o in open_orders:
        print(f"  Open order: {o.orderId} {o.action} {o.totalQuantity}x @ ${o.lmtPrice:.2f} status={o.orderType}")

    await connector.disconnect()
    print("\n  Done. Disconnected from IBKR.")


if __name__ == "__main__":
    asyncio.run(main())

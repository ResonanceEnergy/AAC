#!/usr/bin/env python3
"""
IBKR Fire -- Jan 2027 LEAPS (no prompts)
==========================================
Auto-detects TWS port (7496 live / 7497 paper).
Sizes Jan 2027 LEAPS from live account cash.
Run at 9:30 AM ET for instant execution.

  SLV Jan 2027 $65C  -- 55% of deployment
  XLE Jan 2027 $65C  -- 25% of deployment
  GDX Jul 2026 $90C  -- 20% of deployment

Usage:
  .venv\\Scripts\\python.exe _ibkr_fire.py            # SCAN + DRY RUN (default)
  .venv\\Scripts\\python.exe _ibkr_fire.py --live     # LIVE execution (no prompts)
"""
from __future__ import annotations

import asyncio
import sys
import os
import io
import argparse
from datetime import datetime

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
#  TRADE DEFINITIONS -- Jan 2027 LEAPS
# ============================================================================
LEAPS = [
    {
        "label":    "SLV Jan 2027 $65C LEAPS",
        "symbol":   "SLV",
        "strike":   65.0,
        "right":    "C",
        "expiry":   "20270116",   # Jan 16 2027 (3rd Fri)
        "alt_expiries": ["20270115", "20270117", "20270121"],
        "alloc":    0.55,         # 55% of deploy budget
        "thesis":   "Silver at $63.44. 290 DTE. Iran + de-dollarization = silver moon.",
    },
    {
        "label":    "XLE Jan 2027 $65C LEAPS",
        "symbol":   "XLE",
        "strike":   65.0,
        "right":    "C",
        "expiry":   "20270116",
        "alt_expiries": ["20270115", "20270117", "20270121"],
        "alloc":    0.25,         # 25%
        "thesis":   "XLE $62.56. Oil >$100 Hormuz. Energy LEAPS.",
    },
    {
        "label":    "GDX Jul 2026 $90C",
        "symbol":   "GDX",
        "strike":   90.0,
        "right":    "C",
        "expiry":   "20260717",
        "alt_expiries": ["20260718", "20260716", "20260619"],
        "alloc":    0.20,         # 20%
        "thesis":   "GDX $86. Gold >$4,500. Miners breakout. 120 DTE momentum.",
    },
]

DEPLOY_RATIO = 0.70   # deploy 70% of free cash, keep 30% reserve
IBKR_ACCOUNT = os.getenv("IBKR_ACCOUNT", "U24346218")


async def find_and_connect(ib):
    """Try live port first, then paper. Return (port, mode)."""
    for port, mode in [(7496, "LIVE"), (7497, "PAPER")]:
        try:
            await ib.connectAsync("127.0.0.1", port, clientId=77, timeout=6)
            if ib.isConnected():
                return port, mode
        except Exception:
            pass
    return None, None


async def get_cash(ib, acct):
    """Return TotalCashValue USD."""
    vals = ib.accountValues(acct)
    for v in vals:
        if v.tag == "TotalCashValue" and v.currency == "USD":
            return float(v.value)
    # fallback: NetLiquidation
    for v in vals:
        if v.tag == "NetLiquidation" and v.currency == "USD":
            return float(v.value)
    return 0.0


async def qualify_contract(ib, trade):
    """Qualify option contract, try alt expiries if primary not found."""
    from ib_insync import Option as IbOption
    for exp in [trade["expiry"]] + trade["alt_expiries"]:
        c = IbOption(trade["symbol"], exp, trade["strike"], trade["right"],
                     "SMART", currency="USD")
        q = ib.qualifyContracts(c)
        if q:
            return q[0], exp
    return None, None


async def get_mid(ib, contract):
    """Return (bid, ask, mid, last) for a contract."""
    ib.reqMarketDataType(3)   # delayed OK if no live subscription
    ib.reqMktData(contract, "", False, False)
    await asyncio.sleep(2)
    t = ib.ticker(contract)
    bid  = float(t.bid)  if t.bid  and t.bid  > 0 else 0.0
    ask  = float(t.ask)  if t.ask  and t.ask  > 0 else 0.0
    last = float(t.last) if t.last and t.last > 0 else 0.0
    mid  = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else last
    return bid, ask, mid, last


async def run(live: bool):
    from ib_insync import IB, LimitOrder as IbLimitOrder
    import nest_asyncio
    nest_asyncio.apply()

    print("=" * 70)
    print("  IBKR FIRE -- Jan 2027 LEAPS")
    mode_str = "*** LIVE EXECUTION ***" if live else "DRY RUN (scan only)"
    print(f"  Mode: {mode_str}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    ib = IB()
    port, conn_mode = await find_and_connect(ib)

    if port is None:
        print("\n  ERROR: Could not connect to TWS on port 7496 (live) or 7497 (paper).")
        print("  Action: Open TWS / IB Gateway and ensure API connections are enabled.")
        print("  Live port: 7496 | Paper port: 7497")
        return

    print(f"\n  Connected: port {port} ({conn_mode})")
    if conn_mode == "PAPER" and live:
        print("  !!!")
        print("  !!! WARNING: TWS IS IN PAPER MODE (port 7497) !!!")
        print("  !!! Orders placed on PAPER account -- NOT real money !!!")
        print("  !!! Switch TWS to LIVE mode on port 7496 for real execution !!!")
        print("  !!!")

    acct = ib.managedAccounts()[0]
    print(f"  Account: {acct}")

    cash = await get_cash(ib, acct)
    deploy_budget = round(cash * DEPLOY_RATIO, 2)
    print(f"  Cash (USD):     ${cash:,.2f}")
    print(f"  Deploy ({int(DEPLOY_RATIO*100)}%):  ${deploy_budget:,.2f}")
    print(f"  Reserve (30%):  ${cash - deploy_budget:,.2f}")

    # Show existing positions
    portfolio = ib.portfolio(acct)
    if portfolio:
        symbols = {t["symbol"] for t in LEAPS}
        existing = [p for p in portfolio if p.contract.symbol in symbols]
        if existing:
            print(f"\n  EXISTING POSITIONS in target symbols:")
            for p in existing:
                c = p.contract
                desc = f"{c.symbol} {c.secType}"
                if hasattr(c, "strike") and c.strike:
                    desc += f" ${c.strike:.0f}{c.right} exp {c.lastTradeDateOrContractMonth}"
                print(f"    {desc}  qty={p.position:.0f}  MV=${p.marketValue:.2f}  "
                      f"uPnL={p.unrealizedPNL:+.2f}")

    # Process each LEAP
    print()
    placed = 0
    errors = 0
    total_committed = 0.0

    for trade in LEAPS:
        alloc_usd = round(deploy_budget * trade["alloc"], 2)
        print(f"\n  {'─' * 66}")
        print(f"  {trade['label']}")
        print(f"  Alloc: ${alloc_usd:,.0f} ({int(trade['alloc']*100)}% of ${deploy_budget:,.0f})")

        # Qualify
        contract, used_exp = await qualify_contract(ib, trade)
        if contract is None:
            print(f"  SKIP -- no contract found for {trade['symbol']} "
                  f"${trade['strike']:.0f}{trade['right']}")
            errors += 1
            continue
        print(f"  Contract: {contract.localSymbol}  conId={contract.conId}  exp={used_exp}")

        # Get quote
        bid, ask, mid, last = await get_mid(ib, contract)
        print(f"  Bid ${bid:.2f} | Ask ${ask:.2f} | Mid ${mid:.2f} | Last ${last:.2f}")

        price = mid if mid > 0 else last
        if price <= 0:
            print(f"  SKIP -- no valid price")
            errors += 1
            continue

        qty = max(1, int(alloc_usd / (price * 100)))
        cost = qty * price * 100
        print(f"  Qty: {qty}x @ ${price:.2f} = ${cost:,.0f}  (budget ${alloc_usd:,.0f})")

        if not live:
            print(f"  DRY RUN -- would place BUY {qty}x {contract.localSymbol} @ ${price:.2f}")
            continue

        # LIVE EXECUTION
        try:
            order_obj = IbLimitOrder("BUY", qty, round(price, 2))
            order_obj.account = acct
            order_obj.tif = "DAY"
            trade_obj = ib.placeOrder(contract, order_obj)
            await asyncio.sleep(2)
            oid = trade_obj.order.orderId
            status = trade_obj.orderStatus.status
            filled = trade_obj.orderStatus.filled
            print(f"  ORDER [{oid}] status={status} filled={filled} "
                  f"${cost:,.0f} committed")
            total_committed += cost
            placed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\n  {'=' * 66}")
    if live:
        print(f"  RESULT: {placed} placed | {errors} errors | "
              f"${total_committed:,.0f} committed")
    else:
        print(f"  DRY RUN complete. Run with --live to execute.")
    print(f"  {'=' * 66}\n")

    ib.disconnect()


def main():
    parser = argparse.ArgumentParser(description="IBKR Jan 2027 LEAPS -- fire at market open")
    parser.add_argument("--live", action="store_true",
                        help="Execute LIVE orders (default: dry run / scan)")
    args = parser.parse_args()
    asyncio.run(run(live=args.live))


if __name__ == "__main__":
    main()

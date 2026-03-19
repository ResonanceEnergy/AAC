#!/usr/bin/env python3
"""
IBKR Put Order Script — Private Credit Blowup
Places 8 put orders on IBKR paper account.
Budget: $920 — uses pre-priced contracts from 2026-03-18 session.
"""

import asyncio
import sys
import os

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

import nest_asyncio
nest_asyncio.apply()

from shared.config_loader import load_env_file
load_env_file()

from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

BUDGET = 920.00

# Pre-priced puts from IBKR delayed quotes, sorted cheapest first.
# (symbol, strike, expiry, limit_price, description)
ORDERS = [
    ("ARCC", 17,  "20260417", 0.25, "Ares Capital BDC — largest private credit lender"),
    ("PFF",  29,  "20260417", 0.40, "Preferred stock ETF — cracks first in credit stress"),
    ("LQD",  106, "20260515", 0.64, "IG corporate bonds — spread widening"),
    ("EMB",  90,  "20260515", 0.75, "EM debt — dollar strength + credit contagion"),
    ("MAIN", 50,  "20260417", 0.80, "Main Street Capital BDC — lower middle market"),
    ("JNK",  92,  "20260417", 0.80, "SPDR High Yield — junk bonds"),
    ("KRE",  58,  "20260501", 1.45, "Regional banks — CRE + private credit loans"),
    ("IWM",  230, "20260501", 3.93, "Small caps — most credit-dependent sector"),
]


async def main():
    connector = IBKRConnector()
    await connector.connect()
    ib = connector._ib
    acct = ib.managedAccounts()[0]

    # Read account cash directly (bypass broken get_balances)
    summaries = ib.accountSummary(acct)
    cash = 0.0
    for s in summaries:
        if s.tag in ("TotalCashValue", "NetLiquidation") and s.currency == "USD":
            try:
                cash = float(s.value)
            except (ValueError, TypeError):
                pass
            break

    print(f"Account: {acct}")
    print(f"Cash:    ${cash:,.2f}")
    print(f"Budget:  ${BUDGET:,.2f}")
    print(f"Orders:  {len(ORDERS)} puts\n")

    # Check existing orders to avoid duplicates
    existing = await connector.get_open_orders()
    existing_syms = set()
    for o in existing:
        if o.symbol:
            existing_syms.add(o.symbol.split()[0])
    if existing:
        print(f"Existing orders ({len(existing)}): {', '.join(existing_syms)}")

    # Get fresh quotes to use as limit price (delayed data)
    ib.reqMarketDataType(3)

    spent = 0.0
    placed = 0
    skipped = 0

    for symbol, strike, expiry, fallback_limit, desc in ORDERS:
        cost = fallback_limit * 100
        exp_fmt = f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:]}"

        if symbol in existing_syms:
            print(f"  SKIP {symbol} ${strike}P — order already exists")
            skipped += 1
            continue

        if spent + cost > BUDGET:
            print(f"  SKIP {symbol} ${strike}P — over budget (${spent + cost:.0f} > ${BUDGET:.0f})")
            skipped += 1
            continue

        try:
            # Try to get a fresh quote; fall back to pre-priced limit
            limit = fallback_limit
            try:
                quote = await connector.get_option_quote(symbol, expiry, strike, "P")
                ask = quote.get("ask")
                last = quote.get("last")
                close_p = quote.get("close")
                for v in [ask, last, close_p]:
                    if v and v > 0:
                        limit = v
                        break
            except Exception:
                pass  # use fallback_limit

            order = await connector.create_option_order(
                symbol=symbol,
                expiry=expiry,
                strike=float(strike),
                right="P",
                side="buy",
                quantity=1,
                order_type="limit",
                price=round(limit, 2),
            )

            spent += limit * 100
            placed += 1
            print(f"  ORDER #{placed} [{order.order_id}]: BUY 1x {symbol} ${strike}P {exp_fmt} @ ${limit:.2f} — {desc}")

        except Exception as e:
            print(f"  ERROR {symbol} ${strike}P: {e}")

    print()
    print("=" * 70)
    print(f"PLACED: {placed} put orders | SPENT: ${spent:,.0f} / ${BUDGET:,.0f}")
    print(f"SKIPPED: {skipped} | REMAINING: ${BUDGET - spent:,.0f}")
    print("=" * 70)

    # Show open orders
    await asyncio.sleep(1)
    open_orders = await connector.get_open_orders()
    if open_orders:
        print(f"\nAll open orders ({len(open_orders)}):")
        for o in open_orders:
            print(f"  [{o.order_id}] {o.side} {o.amount}x {o.symbol} @ ${o.price or 0:.2f} — {o.status}")

    await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

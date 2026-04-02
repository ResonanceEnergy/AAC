#!/usr/bin/env python3
"""
Price ALL private credit blowup puts on IBKR.
Uses the existing IBKRConnector properly.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

import nest_asyncio

nest_asyncio.apply()

from shared.config_loader import load_env_file

load_env_file()

from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

# Private credit blowup targets — ordered by conviction
TARGETS = [
    # (symbol, otm_pct, dte_days, description)
    # TIER 1: Direct private credit exposure
    ("KRE",  0.08, 45, "Regional banks — CRE + private credit loans"),
    ("IWM",  0.07, 45, "Small caps — most credit-dependent sector"),
    ("LQD",  0.03, 60, "IG corporate bonds — spread widening"),
    ("SPY",  0.05, 45, "Broad market crash hedge"),
    ("QQQ",  0.06, 45, "Nasdaq — liquidity crunch kills valuations"),
    # TIER 2: Private credit direct plays
    ("ARCC", 0.08, 60, "Ares Capital BDC — largest private credit lender"),
    ("MAIN", 0.08, 60, "Main Street Capital BDC — lower middle market"),
    ("PFF",  0.05, 60, "Preferred stock ETF — cracks first in credit stress"),
    ("JNK",  0.04, 60, "SPDR High Yield — junk bonds"),
    ("EMB",  0.05, 60, "EM debt — dollar strength + credit contagion"),
]


async def main():
    connector = IBKRConnector()
    await connector.connect()
    ib = connector._ib
    acct = ib.managedAccounts()[0]
    print(f"Account: {acct}")

    # Get cash via account summary directly (bypass connector bug)
    cash = 0.0
    try:
        summary = ib.accountSummary(acct)
        for item in summary:
            if item.tag == "TotalCashValue" and item.currency == "USD":
                cash = float(item.value)
                break
        if cash == 0:
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    cash = float(item.value)
                    break
    except Exception as e:
        print(f"Balance warning: {e}")
        cash = 920.0  # fallback known balance
    print(f"Cash available: ${cash:,.2f}")
    remaining = cash
    print()

    ib.reqMarketDataType(3)  # delayed (market closed)

    results = []
    import ib_insync as ibi

    for symbol, otm_pct, dte, desc in TARGETS:
        try:
            # Get underlying price
            stock = ibi.Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(stock)
            if not qualified:
                print(f"  {symbol}: CANNOT QUALIFY STOCK")
                continue

            ticker = ib.reqMktData(stock)
            await asyncio.sleep(1.5)
            price = ticker.last if ticker.last and ticker.last > 0 else (
                ticker.close if ticker.close and ticker.close > 0 else None
            )
            ib.cancelMktData(stock)

            if not price:
                print(f"  {symbol}: NO PRICE — skipping")
                continue

            # Target strike
            strike_target = round(price * (1 - otm_pct))

            # Get option chain via connector
            chains = await connector.get_option_chain(symbol)
            if not chains:
                print(f"  {symbol}: NO OPTION CHAIN")
                continue

            # Find best expiry (closest to target DTE)
            target_date = datetime.now() + timedelta(days=dte)
            target_exp = target_date.strftime("%Y%m%d")
            all_expiries = sorted({c["expiry"] for c in chains})
            best_exp = min(all_expiries, key=lambda x: abs(int(x) - int(target_exp)))

            # Find closest valid strike
            all_strikes = set()
            for c in chains:
                if c["expiry"] == best_exp:
                    all_strikes.update(c["strikes"])
            if not all_strikes:
                for c in chains:
                    all_strikes.update(c["strikes"])
            best_strike = min(all_strikes, key=lambda x: abs(x - strike_target))

            # Price the put
            quote = await connector.get_option_quote(symbol, best_exp, best_strike, "P")
            ask = quote.get("ask")
            bid = quote.get("bid")
            last = quote.get("last")
            close_p = quote.get("close")
            ref = None
            for v in [ask, last, close_p, bid]:
                if v and v > 0:
                    ref = v
                    break

            exp_fmt = f"{best_exp[:4]}-{best_exp[4:6]}-{best_exp[6:]}"
            cost_1 = ref * 100 if ref else 0

            results.append({
                "symbol": symbol,
                "price": price,
                "strike": best_strike,
                "exp": best_exp,
                "exp_fmt": exp_fmt,
                "ref": ref,
                "bid": bid,
                "ask": ask,
                "last": last,
                "cost_1": cost_1,
                "desc": desc,
                "otm_pct": (price - best_strike) / price * 100,
            })

            mark = f"${ref:.2f}" if ref else "NO QUOTE"
            print(f"  {symbol:5s} ${price:>8.2f} -> ${best_strike:.0f}P {exp_fmt} = {mark}/ct (${cost_1:.0f}/lot) -- {desc}")

        except Exception as e:
            print(f"  {symbol}: ERROR -- {e}")

    # Shopping list
    print()
    print("=" * 75)
    print("PRIVATE CREDIT BLOWUP -- PUT SHOPPING LIST FOR MARKET OPEN")
    print("=" * 75)
    print(f"Cash: ${cash:,.2f}")
    print()

    affordable = [r for r in results if r["ref"] and r["cost_1"] > 0]
    affordable.sort(key=lambda x: x["cost_1"])

    total = 0
    buy_list = []
    print(f"{'#':>2s} {'Sym':5s} {'Price':>8s} {'Strike':>8s} {'Expiry':>12s} {'OTM%':>6s} {'Ask':>8s} {'Cost':>8s}  Description")
    print("-" * 95)
    for i, r in enumerate(affordable, 1):
        if total + r["cost_1"] <= remaining:
            total += r["cost_1"]
            buy_list.append(r)
            flag = "<-- BUY"
        else:
            flag = "(over budget)"
        print(f"{i:2d} {r['symbol']:5s} ${r['price']:>7.2f}  ${r['strike']:>6.0f}P {r['exp_fmt']:>12s} {r['otm_pct']:>5.1f}% ${r['ref']:>6.2f} ${r['cost_1']:>6.0f}   {r['desc']}  {flag}")

    print(f"\nBuying: {len(buy_list)} puts for ${total:,.0f} / ${remaining:,.0f} available")
    print(f"Cash after: ${remaining - total:,.0f}")

    if buy_list:
        print("\n--- ORDERS TO PLACE AT MARKET OPEN ---")
        for r in buy_list:
            print(f"  BUY 1x {r['symbol']} ${r['strike']:.0f}P exp {r['exp']} @ ${r['ref']:.2f} limit")

    await connector.disconnect()

asyncio.run(main())

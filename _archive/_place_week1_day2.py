#!/usr/bin/env python3
"""
AAC War Room — Week 1 Day 2 Orders (March 20, 2026)
====================================================
Put Pyramid Expansion + Hedge Rocket Side

Based on Morning Brief:
- SPY 600P x12, QQQ 420P x8, XLF 32P x8, XLRE 19P x8  (puts)
- GLD 500C x5, SLV 80C x5                                (hedges)

Budget: ~$5,000 put premium + ~$2,750 hedge premium
Execution: IBKR port 7496 LIVE, limit orders at mid or better.

IMPORTANT: Review all orders before running. This script will
place REAL LIVE orders on your IBKR account.

Usage:
  .venv\Scripts\python.exe _place_week1_day2.py              # DRY RUN (default)
  .venv\Scripts\python.exe _place_week1_day2.py --live        # LIVE execution
  .venv\Scripts\python.exe _place_week1_day2.py --price-only  # Just scan prices
"""

import asyncio
import sys
import os
import io

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

import nest_asyncio
nest_asyncio.apply()

from shared.config_loader import load_env_file
load_env_file()

from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

# ============================================================================
#  ORDER BOOK — Week 1 Day 2  (March 20, 2026)
# ============================================================================
# All weekly expiries: March 21 (Fri) or March 28 (next Fri).
# If March 21 chain is too thin/not available, use March 28.
# Prices are mid estimates from morning brief — script will get fresh quotes.

# PUT PYRAMID CORE (~$5,000 premium budget)
# (symbol, strike, expiry_primary, expiry_fallback, right, qty, limit_price, description)
PUT_ORDERS = [
    ("SPY",  600,  "20260321", "20260328", "P", 12, 1.20, "SPY deep OTM — broad market crash play"),
    ("QQQ",  420,  "20260321", "20260328", "P",  8, 2.50, "QQQ deep OTM — Big Tech / AI layoff exposure"),
    ("XLF",   32,  "20260321", "20260328", "P",  8, 0.65, "XLF banks — private credit + CRE stress"),
    ("XLRE",  19,  "20260321", "20260328", "P",  8, 0.40, "XLRE CRE — redemption wave + vacancy"),
]

# HEDGE ROCKET SIDE (~$2,750 premium budget)
HEDGE_ORDERS = [
    ("GLD",  500,  "20260321", "20260328", "C",  5, 3.50, "GLD calls — gold safe-haven rocket"),
    ("SLV",   80,  "20260321", "20260328", "C",  5, 2.00, "SLV calls — silver breakout play"),
]

ALL_ORDERS = PUT_ORDERS + HEDGE_ORDERS

# Budget caps
PUT_BUDGET = 5500.0   # max premium for puts
HEDGE_BUDGET = 3000.0  # max premium for hedges
TOTAL_BUDGET = PUT_BUDGET + HEDGE_BUDGET


async def get_fresh_price(connector, ib, symbol, expiry, strike, right, fallback):
    """Try to get a live quote; fall back to estimate."""
    try:
        quote = await connector.get_option_quote(symbol, expiry, strike, right)
        bid = quote.get("bid", 0) or 0
        ask = quote.get("ask", 0) or 0
        last = quote.get("last", 0) or 0
        close_p = quote.get("close", 0) or 0

        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        # Prefer mid, then last, then close, then fallback
        for v in [mid, last, close_p]:
            if v and v > 0:
                return round(v, 2), {"bid": bid, "ask": ask, "last": last, "mid": mid}
    except Exception:
        pass
    return fallback, {"bid": 0, "ask": 0, "last": 0, "mid": 0, "note": "using fallback"}


async def check_chain_exists(connector, symbol, expiry, strike, right):
    """Verify the specific contract exists in IBKR chain."""
    try:
        from ib_insync import Option as IbOption
        contract = IbOption(symbol, expiry, strike, right, "SMART")
        qualified = await connector._conn.qualifyContractsAsync(contract)
        return len(qualified) > 0
    except Exception:
        return False


async def main():
    mode = "dry"
    if "--live" in sys.argv:
        mode = "live"
    elif "--price-only" in sys.argv:
        mode = "price"

    print("=" * 80)
    print("  AAC WAR ROOM — WEEK 1 DAY 2 ORDERS")
    print(f"  Mode: {'*** LIVE EXECUTION ***' if mode == 'live' else 'DRY RUN (add --live to execute)' if mode == 'dry' else 'PRICE SCAN ONLY'}")
    print(f"  Date: March 20, 2026")
    print("=" * 80)

    if mode == "live":
        print("\n  WARNING: LIVE MODE — Orders will be placed on IBKR account U24346218")
        print("  Press Ctrl+C within 5 seconds to abort...")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n  Aborted.")
            return

    # Connect
    connector = IBKRConnector()
    await connector.connect()
    ib = connector._ib
    acct = ib.managedAccounts()[0]

    # Account info
    summaries = ib.accountSummary(acct)
    net_liq = 0.0
    cash = 0.0
    buying_power = 0.0
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

    # Show existing positions
    portfolio = ib.portfolio()
    if portfolio:
        print(f"\n  EXISTING POSITIONS ({len(portfolio)}):")
        print(f"  {'Sym':<8} {'Type':<5} {'Strike':>7} {'Expiry':>10} {'Qty':>5} {'MktVal':>10} {'uPnL':>10}")
        print("  " + "-" * 60)
        for pos in portfolio:
            c = pos.contract
            print(f"  {c.symbol:<8} {c.secType:<5} {c.strike:>7.1f} {c.lastTradeDateOrContractMonth:>10} "
                  f"{pos.position:>5.0f} ${pos.marketValue:>9.2f} ${pos.unrealizedPNL:>+9.2f}")

    # Check for duplicate orders
    existing_orders = await connector.get_open_orders()
    existing_descs = set()
    for o in existing_orders:
        if o.symbol:
            existing_descs.add(o.symbol.split()[0])

    # Request delayed data
    ib.reqMarketDataType(3)

    # Price all contracts
    print(f"\n{'=' * 80}")
    print("  ORDER BOOK — PRICING")
    print(f"{'=' * 80}\n")

    order_plan = []
    for symbol, strike, exp1, exp2, right, qty, limit, desc in ALL_ORDERS:
        # Try primary expiry first, fallback if chain doesn't exist
        expiry = exp1
        exists = await check_chain_exists(connector, symbol, expiry, strike, right)
        if not exists and exp2:
            expiry = exp2
            exists = await check_chain_exists(connector, symbol, expiry, strike, right)

        price, quote_data = await get_fresh_price(connector, ib, symbol, expiry, strike, right, limit)
        premium = price * qty * 100
        is_put = right == "P"
        budget_type = "PUT" if is_put else "HEDGE"

        status = "OK"
        if symbol in existing_descs:
            status = "DUP"
        elif not exists:
            status = "NO_CHAIN"

        exp_fmt = f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:]}"
        side = "BUY"
        r_label = "Put" if right == "P" else "Call"

        order_plan.append({
            "symbol": symbol, "strike": strike, "expiry": expiry, "right": right,
            "qty": qty, "price": price, "premium": premium, "desc": desc,
            "quote": quote_data, "status": status, "budget_type": budget_type,
            "exists": exists,
        })

        q = quote_data
        print(f"  {status:>8} | {side} {qty:>3}x {symbol:<5} ${strike:>5}{r_label[0]} {exp_fmt} "
              f"@ ${price:.2f}  (bid ${q.get('bid', 0):.2f} / ask ${q.get('ask', 0):.2f}) "
              f"= ${premium:,.0f} premium  [{budget_type}]")

    # Summary
    put_total = sum(o["premium"] for o in order_plan if o["budget_type"] == "PUT" and o["status"] == "OK")
    hedge_total = sum(o["premium"] for o in order_plan if o["budget_type"] == "HEDGE" and o["status"] == "OK")
    total = put_total + hedge_total

    print(f"\n  PUT Premium:    ${put_total:>8,.0f} / ${PUT_BUDGET:,.0f} budget")
    print(f"  HEDGE Premium:  ${hedge_total:>8,.0f} / ${HEDGE_BUDGET:,.0f} budget")
    print(f"  TOTAL Premium:  ${total:>8,.0f} / ${TOTAL_BUDGET:,.0f} budget")

    actionable = [o for o in order_plan if o["status"] == "OK"]
    skippable = [o for o in order_plan if o["status"] != "OK"]

    if skippable:
        print(f"\n  SKIPPING {len(skippable)} orders:")
        for o in skippable:
            print(f"    {o['status']}: {o['symbol']} ${o['strike']}{o['right']} — {o['desc']}")

    if mode == "price":
        print("\n  Price scan complete. Use --live to execute.")
        await connector.disconnect()
        return

    if mode == "dry":
        print(f"\n  DRY RUN — {len(actionable)} orders would be placed.")
        print("  Run with --live to execute for real.")
        await connector.disconnect()
        return

    # LIVE EXECUTION
    print(f"\n{'=' * 80}")
    print(f"  EXECUTING {len(actionable)} ORDERS — LIVE")
    print(f"{'=' * 80}\n")

    placed = 0
    total_spent = 0.0

    for o in actionable:
        symbol = o["symbol"]
        strike = float(o["strike"])
        expiry = o["expiry"]
        right = o["right"]
        qty = o["qty"]
        price = o["price"]
        premium = o["premium"]

        # Budget check
        is_put = right == "P"
        budget_cap = PUT_BUDGET if is_put else HEDGE_BUDGET
        bucket_spent = sum(
            x["premium"] for x in order_plan[:order_plan.index(o)]
            if x["budget_type"] == o["budget_type"] and x.get("executed")
        )
        if bucket_spent + premium > budget_cap:
            print(f"  SKIP {symbol} — over {o['budget_type']} budget")
            continue

        try:
            order_result = await connector.create_option_order(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                side="buy",
                quantity=qty,
                order_type="limit",
                price=round(price, 2),
            )
            o["executed"] = True
            placed += 1
            total_spent += premium
            exp_fmt = f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:]}"
            r_label = "P" if right == "P" else "C"
            print(f"  ORDER #{placed} [{order_result.order_id}]: BUY {qty}x {symbol} ${strike:.0f}{r_label} {exp_fmt} "
                  f"@ ${price:.2f} = ${premium:,.0f} — {o['desc']}")
        except Exception as e:
            print(f"  ERROR {symbol} ${strike}{right}: {e}")

    print(f"\n{'=' * 80}")
    print(f"  EXECUTION COMPLETE")
    print(f"  Placed: {placed} orders  |  Premium: ${total_spent:,.0f}")
    print(f"{'=' * 80}")

    # Show all open orders after execution
    await asyncio.sleep(2)
    open_orders = await connector.get_open_orders()
    if open_orders:
        print(f"\n  ALL OPEN ORDERS ({len(open_orders)}):")
        for o in open_orders:
            print(f"    [{o.order_id}] {o.side} {o.amount}x {o.symbol} @ ${o.price or 0:.2f} — {o.status}")

    await connector.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

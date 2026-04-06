"""
Execute all Black Swan scanner opportunities as GTC limit orders on Polymarket.
===============================================================================
- Scans for all opportunities (<=25c)
- Shows a full payout table BEFORE executing
- Places limit BUY orders via py-clob-client SDK
- Logs every order result

Usage:
    python execute_90_bets.py              # dry-run (show table only)
    python execute_90_bets.py --live       # actually place orders
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

BALANCE = 317.00  # USDC.e remaining after 40 prior orders
MIN_ORDER_SIZE = 1.0  # Polymarket minimum order ~$1


def init_clob_client():
    """Initialize the py-clob-client SDK with pre-derived L2 creds."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    api_key = os.getenv("POLYMARKET_API_KEY", "")
    api_secret = os.getenv("POLYMARKET_API_SECRET", "")
    api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
    sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))

    if not all([private_key, api_key, api_secret, api_passphrase]):
        print("ERROR: Missing Polymarket credentials in .env")
        sys.exit(1)

    creds = ApiCreds(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
    )
    client = ClobClient(
        "https://clob.polymarket.com",
        key=private_key,
        chain_id=chain_id,
        signature_type=sig_type,
        funder=funder if sig_type in (1, 2) else None,
        creds=creds,
    )
    return client


def place_limit_buy(client, token_id: str, price: float, size: float,
                    tick_size: str = "0.01", neg_risk: bool = False):
    """Place a GTC limit BUY order. Returns order result dict or error string."""
    from py_clob_client.clob_types import (
        OrderArgs, OrderType, PartialCreateOrderOptions,
    )
    from py_clob_client.order_builder.constants import BUY

    order = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=BUY,
    )
    options = PartialCreateOrderOptions(
        tick_size=tick_size,
        neg_risk=neg_risk,
    )
    signed = client.create_order(order, options)
    result = client.post_order(signed, OrderType.GTC)
    return result


async def main():
    live = "--live" in sys.argv

    print()
    print("=" * 130)
    print("  POLYMARKET BLACK SWAN -- 90-BET EXECUTION PLAN")
    print("  Mode: {}".format("*** LIVE ORDERS ***" if live else "DRY RUN (add --live to execute)"))
    print("  Balance: ${:.2f} USDC.e".format(BALANCE))
    print("=" * 130)

    # 1. Scan for opportunities
    scanner = PolymarketBlackSwanScanner()
    try:
        opps = await scanner.scan()
    finally:
        await scanner.close()

    if not opps:
        print("  No opportunities found!")
        return

    n = len(opps)
    bet_size = round(BALANCE / n, 2)
    if bet_size < MIN_ORDER_SIZE:
        bet_size = MIN_ORDER_SIZE
        n = int(BALANCE / MIN_ORDER_SIZE)
        opps = opps[:n]
        print(f"  WARNING: Capped at {n} bets (min order ${MIN_ORDER_SIZE})")

    print(f"  Opportunities: {n}")
    print(f"  Bet size: ${bet_size:.2f} each (equal weight)")
    print(f"  Total deployed: ${bet_size * n:.2f}")
    print()

    # 2. Calculate shares and max payout for each
    rows = []
    total_cost = 0.0
    total_max_payout = 0.0

    for i, opp in enumerate(opps, 1):
        price = opp.market_price
        if price <= 0:
            continue
        shares = bet_size / price  # shares = dollars / price_per_share
        max_payout = shares * 1.0  # each share pays $1 if YES resolves
        profit = max_payout - bet_size
        multiplier = max_payout / bet_size if bet_size > 0 else 0

        rows.append({
            "num": i,
            "cat": opp.category,
            "outcome": opp.outcome,
            "price": price,
            "bet": bet_size,
            "shares": shares,
            "max_payout": max_payout,
            "profit": profit,
            "mult": multiplier,
            "question": opp.market_question[:70],
            "token_id": opp.token_id,
            "edge": opp.edge,
            "kelly": opp.kelly_fraction,
        })
        total_cost += bet_size
        total_max_payout += max_payout

    # 3. Print the full payout table
    hdr = "  {:>3} {:14} {:>8} {:>6} {:>7} {:>9} {:>11} {:>9} {:>5}  {}"
    print(hdr.format("#", "Category", "BET", "Price", "Spend", "Shares", "MAX PAYOUT", "PROFIT", "MULT", "Market"))
    print("-" * 130)

    for r in rows:
        line = "  {:3} {:14} {:>8} {:6.4f} {:>7.2f} {:>9.1f} {:>11.2f} {:>9.2f} {:>5.1f}x {}"
        print(line.format(
            r["num"], r["cat"], f"BUY {r['outcome']}", r["price"],
            r["bet"], r["shares"], r["max_payout"], r["profit"],
            r["mult"], r["question"],
        ))

    print("-" * 130)
    print("  {:>3} {:14} {:>8} {:>6} {:>7.2f} {:>9} {:>11.2f} {:>9.2f} {:>5.1f}x".format(
        "", "TOTALS", "", "", total_cost, "",
        total_max_payout, total_max_payout - total_cost,
        total_max_payout / total_cost if total_cost > 0 else 0,
    ))
    print()
    print("  TOTAL COST:       ${:>12,.2f}".format(total_cost))
    print("  MAX TOTAL PAYOUT: ${:>12,.2f}  (if ALL bets hit)".format(total_max_payout))
    print("  MAX TOTAL PROFIT: ${:>12,.2f}".format(total_max_payout - total_cost))
    print("  AVG MULTIPLIER:   {:>12.1f}x".format(total_max_payout / total_cost if total_cost > 0 else 0))
    print()

    # Breakdown by category
    cats = {}
    for r in rows:
        c = r["cat"]
        if c not in cats:
            cats[c] = {"count": 0, "cost": 0, "payout": 0}
        cats[c]["count"] += 1
        cats[c]["cost"] += r["bet"]
        cats[c]["payout"] += r["max_payout"]

    print("  CATEGORY BREAKDOWN:")
    print("  {:20} {:>5} {:>10} {:>14} {:>8}".format("Category", "Bets", "Cost", "Max Payout", "Avg Mult"))
    print("  " + "-" * 60)
    for cat, v in sorted(cats.items(), key=lambda x: x[1]["payout"], reverse=True):
        mult = v["payout"] / v["cost"] if v["cost"] > 0 else 0
        print("  {:20} {:5} {:>10.2f} {:>14.2f} {:>7.1f}x".format(
            cat, v["count"], v["cost"], v["payout"], mult))
    print()

    if not live:
        print("  *** DRY RUN — No orders placed ***")
        print("  To execute: python execute_90_bets.py --live")
        print("=" * 130)
        return

    # ═══════════════════════════════════════════════════════════════════════
    # 4. LIVE EXECUTION
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 130)
    print("  EXECUTING {} LIMIT BUY ORDERS...".format(len(rows)))
    print("=" * 130)

    client = init_clob_client()

    results = []
    success = 0
    failed = 0

    for r in rows:
        token_id = r["token_id"]
        if not token_id:
            print(f"  #{r['num']:3} SKIP — no token_id for: {r['question'][:60]}")
            failed += 1
            results.append({"num": r["num"], "status": "SKIP", "reason": "no token_id"})
            continue

        price = r["price"]
        size = r["shares"]

        print(f"  #{r['num']:3} BUY {r['outcome']:3} | {size:>8.1f} shares @ ${price:.4f} | {r['question'][:55]}...", end="  ")

        try:
            result = place_limit_buy(client, token_id, price, size)
            if result and isinstance(result, dict) and result.get("orderID"):
                oid = result["orderID"]
                print(f"OK  [{oid[:12]}...]")
                success += 1
                results.append({"num": r["num"], "status": "OK", "orderID": oid, **r})
            elif result and isinstance(result, dict) and result.get("errorMsg"):
                err = result["errorMsg"]
                print(f"ERR [{err[:40]}]")
                failed += 1
                results.append({"num": r["num"], "status": "ERR", "error": err, **r})
            else:
                print(f"???  [{str(result)[:50]}]")
                failed += 1
                results.append({"num": r["num"], "status": "UNK", "raw": str(result), **r})
        except Exception as e:
            print(f"FAIL [{str(e)[:50]}]")
            failed += 1
            results.append({"num": r["num"], "status": "FAIL", "error": str(e), **r})

        # Rate limit: 100ms between orders to avoid throttling
        time.sleep(0.15)

    print()
    print("=" * 130)
    print(f"  EXECUTION COMPLETE: {success} placed / {failed} failed / {len(rows)} total")
    print("=" * 130)

    # Save results log
    log_path = f"polymarket_execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "LIVE",
            "balance": BALANCE,
            "bet_size": bet_size,
            "total_bets": len(rows),
            "success": success,
            "failed": failed,
            "total_cost": total_cost,
            "max_payout": total_max_payout,
            "orders": results,
        }, f, indent=2, default=str)
    print(f"  Log saved: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())

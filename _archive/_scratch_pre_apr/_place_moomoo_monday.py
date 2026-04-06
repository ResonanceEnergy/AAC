#!/usr/bin/env python3
"""
AAC War Room -- MOOMOO LONG CALL STRATEGY -- Monday March 30, 2026
====================================================================
Pink Moon Fire Peak -- Silver & Oil LONG CALL deployment on Moomoo.

THESIS (from 90-day War Room + Pressure Cooker):
  - Iran war escalation = oil + gold + silver moonshot
  - VIX 31.05 = CRISIS regime -- elevated premium but directional conviction high
  - De-dollarization thesis confirms precious metals / energy bull
  - Consumer credit stress → small bear put hedge

SCRAPED PRICES (Fri Mar 27 close):
  SLV  $63.44 (+4.39%)    XLE  $62.56 (+1.69%)
  GLD  $414.70 (+3.51%)   VIX  31.05 (+13.16%)

MOOMOO ACCOUNT:
  Cash: ~$365 USD (Tier 1, co-primary with IBKR)
  Existing: OWL $20P Jan 2027, SLV Jun calls, XLE Jun calls
  Security Firm: FUTUCA  |  Trade PIN: 069420  |  Port: 11111

STRATEGY (same as capital engine, adjusted for current market):
  Allocation: Silver 57% / Oil 35% / Consumer Bear 8%
  Expiry focus: Jun 18, 2026 (existing cycle) + Sep 2026 scan

  1. SLV Jun 2026 $65C   -- Silver bull (slightly OTM at $63.44)
  2. XLE Jun 2026 $65C   -- Oil bull (slightly OTM at $62.56)
  3. XRT Jun 2026 $60P   -- Consumer bear hedge (deep OTM)

  If Jun premiums too expensive for $365 budget, script scans $70C/$75C
  strikes or pivots to Sep 2026 for cheaper theta.

Usage:
  .venv\\Scripts\\python.exe _place_moomoo_monday.py                # SCAN chains + show positions
  .venv\\Scripts\\python.exe _place_moomoo_monday.py --price-only   # Quick quote scan
  .venv\\Scripts\\python.exe _place_moomoo_monday.py --live         # *** LIVE EXECUTION ***
  .venv\\Scripts\\python.exe _place_moomoo_monday.py --sep          # Scan Sep 2026 cycle too
"""
import argparse
import io
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "buffer") and (sys.stdout is None or sys.stdout.encoding.lower() != "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

from moomoo import (
    RET_OK,
    Currency,
    OpenQuoteContext,
    OpenSecTradeContext,
    SecurityFirm,
    TrdEnv,
    TrdMarket,
    TrdSide,
)
from moomoo import (
    OrderType as MooOrderType,
)

# ============================================================================
#  CONFIG
# ============================================================================
HOST = "127.0.0.1"
PORT = 11111
TRADE_PIN = "069420"
JUN_2026_START = "2026-06-01"
JUN_2026_END = "2026-06-30"
SEP_2026_START = "2026-09-01"
SEP_2026_END = "2026-09-30"

# Friday March 27 close prices
SPOTS = {"SLV": 63.44, "XLE": 62.56, "XRT": 85.0, "GLD": 414.70}

# Capital engine allocation weights
ALLOCATIONS = {
    "silver":   0.57,   # Silver/precious metals bull -- primary thesis
    "oil":      0.35,   # Energy bull -- Iran/Hormuz catalyst
    "consumer": 0.08,   # Consumer bear hedge
}

# ============================================================================
#  TRADE DEFINITIONS -- Adjusted for March 30 entry
# ============================================================================
#  Primary strikes (slightly OTM for leverage):
#  - SLV $65C (2.5% OTM from $63.44) -- cheapest meaningful call
#  - XLE $65C (3.9% OTM from $62.56) -- slightly OTM
#  - XRT $60P (29% OTM from $85) -- cheap tail hedge
#
#  Fallback strikes if primary too expensive:
#  - SLV $70C (10% OTM) -- cheaper but needs bigger move
#  - XLE $70C (12% OTM) -- cheaper
# ============================================================================

TRADES_JUN = [
    {
        "name": "SILVER BULL -- SLV Jun 2026 Long Call",
        "group": "silver",
        "underlying": "US.SLV",
        "strikes": [65.0, 67.0, 70.0],  # Primary, fallback 1, fallback 2
        "right": "CALL",
        "side": "BUY",
        "thesis": "Silver at $63.44, breakout continuation. Iran + de-dollarization catalyst. "
                  "Jun 18 expiry = 80 DTE. Add to existing Moomoo SLV Jun call position.",
    },
    {
        "name": "OIL BULL -- XLE Jun 2026 Long Call",
        "group": "oil",
        "underlying": "US.XLE",
        "strikes": [65.0, 67.0, 70.0],
        "right": "CALL",
        "side": "BUY",
        "thesis": "XLE at $62.56. Hormuz strait risk + oil above $100 thesis. "
                  "Energy sector leverage via XLE. Add to existing Jun call.",
    },
    {
        "name": "CONSUMER BEAR -- XRT Jun 2026 Put (hedge)",
        "group": "consumer",
        "underlying": "US.XRT",
        "strikes": [60.0, 55.0, 50.0],
        "right": "PUT",
        "side": "BUY",
        "thesis": "Consumer credit stress hedge. Deep OTM put for tail risk. "
                  "8% allocation = ~$29. May be too cheap to fill; skip if no contract.",
    },
]


def find_option_code(qctx, underlying, strike, right, start, end):
    """Find the specific option contract code from Moomoo chain."""
    ret, data = qctx.get_option_chain(code=underlying, start=start, end=end)
    if ret != RET_OK:
        return None, f"Chain lookup failed: {data}"
    if data.empty:
        return None, "No contracts found in date range"

    right_str = right.upper()
    matches = data[
        (data["strike_price"] == strike) &
        (data["option_type"] == right_str)
    ]
    if matches.empty:
        avail = data[data["option_type"] == right_str]["strike_price"].unique()
        avail_sorted = sorted(avail)
        return None, f"No {right_str} @ ${strike}. Available: {avail_sorted[:20]}"

    row = matches.iloc[0]
    code = row["code"]
    expiry = row.get("strike_time", "unknown")
    return code, f"Found: {code} (exp {expiry})"


def get_option_quote(qctx, option_code):
    """Get bid/ask/last for an option contract."""
    ret, data = qctx.get_market_snapshot([option_code])
    if ret != RET_OK:
        return None, f"Quote failed: {data}"
    if data.empty:
        return None, "No quote data"

    row = data.iloc[0]
    quote = {
        "bid": float(row.get("bid_price", 0) or 0),
        "ask": float(row.get("ask_price", 0) or 0),
        "last": float(row.get("last_price", 0) or 0),
        "volume": int(row.get("volume", 0) or 0),
        "open_interest": int(row.get("open_interest", 0) or 0),
    }
    if quote["bid"] > 0 and quote["ask"] > 0:
        quote["mid"] = round((quote["bid"] + quote["ask"]) / 2, 2)
    else:
        quote["mid"] = quote["last"]
    return quote, "OK"


def scan_best_strike(qctx, trade, start, end, budget_per_trade):
    """Scan multiple strikes for a trade, return the best one within budget."""
    results = []
    for strike in trade["strikes"]:
        code, msg = find_option_code(
            qctx, trade["underlying"], strike, trade["right"], start, end,
        )
        if code is None:
            results.append({"strike": strike, "code": None, "error": msg})
            continue

        quote, q_msg = get_option_quote(qctx, code)
        if quote is None:
            results.append({"strike": strike, "code": code, "quote": None, "error": q_msg})
            continue

        # Cost for 1 contract
        price = quote["mid"] if quote["mid"] > 0 else quote["last"]
        cost = price * 100
        fits_budget = cost <= budget_per_trade and cost > 0

        results.append({
            "strike": strike,
            "code": code,
            "quote": quote,
            "price": price,
            "cost": cost,
            "fits_budget": fits_budget,
        })

    return results


def print_scan_results(results, trade_name):
    """Pretty-print the scan results for a trade."""
    print(f"\n  --- {trade_name} ---")
    best = None
    for r in results:
        if r.get("code") is None:
            print(f"    ${r['strike']:>6.0f}: NOT FOUND -- {r.get('error', '')}")
            continue
        if r.get("quote") is None:
            print(f"    ${r['strike']:>6.0f}: {r['code']} -- NO QUOTE ({r.get('error', '')})")
            continue

        q = r["quote"]
        fit = "OK" if r["fits_budget"] else "OVER BUDGET"
        print(f"    ${r['strike']:>6.0f}: Bid ${q['bid']:.2f} | Ask ${q['ask']:.2f} | "
              f"Mid ${r['price']:.2f} | Vol {q['volume']} | OI {q['open_interest']} | "
              f"Cost ${r['cost']:.0f} [{fit}]")

        if r["fits_budget"] and best is None:
            best = r  # First strike that fits budget = most aggressive (lowest strike)

    if best:
        print(f"    >>> SELECTED: ${best['strike']} @ ${best['price']:.2f} "
              f"(${best['cost']:.0f}/contract)")
    else:
        # Pick cheapest available even if over budget (user can decide)
        priced = [r for r in results if r.get("cost") and r["cost"] > 0]
        if priced:
            cheapest = min(priced, key=lambda x: x["cost"])
            print(f"    >>> CHEAPEST: ${cheapest['strike']} @ ${cheapest['price']:.2f} "
                  f"(${cheapest['cost']:.0f}/contract) -- EXCEEDS BUDGET")
            best = cheapest
        else:
            print(f"    >>> NO VALID CONTRACTS FOUND")
    return best


def main():
    parser = argparse.ArgumentParser(description="Moomoo Long Call Strategy -- Monday March 30")
    parser.add_argument("--live", action="store_true", help="LIVE execution (default: scan only)")
    parser.add_argument("--price-only", action="store_true", help="Quick price scan, no execution logic")
    parser.add_argument("--sep", action="store_true", help="Also scan Sep 2026 expiry cycle")
    args = parser.parse_args()

    is_live = args.live
    scan_sep = args.sep

    print("=" * 72)
    print("  AAC WAR ROOM -- MOOMOO LONG CALL STRATEGY")
    print("  Monday March 30, 2026 -- Pink Moon Fire Peak")
    print(f"  Mode: {'*** LIVE EXECUTION ***' if is_live else 'SCAN ONLY (dry run)'}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # ── Connect ──
    print("\n[1/6] CONNECTING TO MOOMOO...")
    qctx = OpenQuoteContext(host=HOST, port=PORT)
    print("  Quote context: OK")

    tctx = OpenSecTradeContext(
        host=HOST, port=PORT,
        security_firm=SecurityFirm.FUTUCA,
        filter_trdmarket=TrdMarket.US,
    )
    ret_unlock, unlock_msg = tctx.unlock_trade(password=TRADE_PIN)
    if ret_unlock == RET_OK:
        print("  Trade unlock: OK")
    else:
        print(f"  Trade unlock: FAILED ({unlock_msg})")
        if is_live:
            print("  *** Cannot execute live trades without unlock. Switching to scan mode. ***")
            is_live = False

    # ── Account + Positions ──
    print("\n[2/6] ACCOUNT STATUS & EXISTING POSITIONS...")
    ret_acc, data_acc = tctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
    cash = 365.0  # Fallback
    if ret_acc == RET_OK:
        total_assets = float(data_acc["total_assets"].iloc[0] or 0)
        cash = float(data_acc["cash"].iloc[0] or 0)
        frozen = float(data_acc.get("frozen_cash", [0]).iloc[0] or 0)
        mkt_val = float(data_acc.get("market_val", [0]).iloc[0] or 0)
        print(f"  Total Assets: ${total_assets:,.2f}")
        print(f"  Cash:         ${cash:,.2f}")
        print(f"  Market Value: ${mkt_val:,.2f}")
        print(f"  Frozen:       ${frozen:,.2f}")
    else:
        print(f"  Account query failed: {data_acc}")
        print(f"  Using fallback cash estimate: ${cash:,.2f}")

    # Show existing positions (especially SLV/XLE calls, OWL puts)
    ret_pos, data_pos = tctx.position_list_query(trd_env=TrdEnv.REAL)
    existing_calls = []
    if ret_pos == RET_OK and not data_pos.empty:
        print("\n  Existing Positions:")
        for _, row in data_pos.iterrows():
            code = row.get("code", "")
            qty = int(row.get("qty", 0))
            mv = row.get("market_val", 0)
            pl = row.get("pl_val", 0)
            cost_price = row.get("cost_price", 0)
            print(f"    {code}: {qty}x | Cost ${cost_price} | MV ${mv} | P&L ${pl}")
            # Track existing call positions for thesis summary
            code_upper = str(code).upper()
            if "SLV" in code_upper or "XLE" in code_upper:
                existing_calls.append(code)
    else:
        print("  No position data (or empty)")

    if existing_calls:
        print(f"\n  ** Existing long call positions: {', '.join(existing_calls)}")
        print("  ** This script ADDS to these positions (same thesis, new contracts)")

    deployable = cash * 0.95  # 5% buffer for commissions
    print(f"\n  Deployable Cash: ${deployable:,.2f} (95% of ${cash:,.2f})")

    # Budget per group
    budgets = {}
    for group, weight in ALLOCATIONS.items():
        budgets[group] = round(deployable * weight, 2)
    print(f"  Silver: ${budgets['silver']:,.2f} | Oil: ${budgets['oil']:,.2f} | "
          f"Consumer: ${budgets['consumer']:,.2f}")

    # ── Scan Jun 2026 Chains ──
    print("\n[3/6] SCANNING JUN 2026 OPTION CHAINS...")
    order_plan = []

    for trade in TRADES_JUN:
        budget = budgets[trade["group"]]
        results = scan_best_strike(qctx, trade, JUN_2026_START, JUN_2026_END, budget)
        best = print_scan_results(results, trade["name"])

        if best and best.get("code"):
            order_plan.append({
                "trade": trade,
                "code": best["code"],
                "strike": best["strike"],
                "price": best["price"],
                "cost": best["cost"],
                "quote": best.get("quote"),
                "expiry": "Jun 2026",
            })

    # ── Scan Sep 2026 (optional) ──
    if scan_sep:
        print("\n[4/6] SCANNING SEP 2026 OPTION CHAINS (--sep)...")
        for trade in TRADES_JUN:
            budget = budgets[trade["group"]]
            results = scan_best_strike(qctx, trade, SEP_2026_START, SEP_2026_END, budget)
            best = print_scan_results(results, f"{trade['name']} (SEP)")
            # Don't add to order plan -- Sep is informational unless Jun is empty
    else:
        print("\n[4/6] SEP 2026 SCAN: skipped (use --sep to include)")

    if args.price_only:
        print("\n  --price-only mode. Done.")
        qctx.close()
        tctx.close()
        return

    # ── Position Sizing ──
    print("\n[5/6] POSITION SIZING...")
    final_orders = []
    total_cost = 0.0

    for op in order_plan:
        qty = 1
        cost = op["cost"] * qty
        remaining = deployable - total_cost

        if cost > remaining:
            print(f"  {op['trade']['name']}: SKIP (${cost:.0f} > ${remaining:.0f} remaining)")
            continue

        total_cost += cost
        final_orders.append({**op, "qty": qty})
        print(f"  {op['trade']['name']}:")
        print(f"    {op['code']} ${op['strike']} | {qty}x @ ${op['price']:.2f} | "
              f"Cost: ${cost:.0f} | Left: ${deployable - total_cost:.0f}")

    print(f"\n  TOTAL: ${total_cost:,.0f} / ${deployable:,.0f} deployable")
    print(f"  REMAINING: ${deployable - total_cost:,.0f}")

    if not final_orders:
        print("\n  *** NO ORDERS TO PLACE -- all strikes exceed budget or no contracts found ***")
        print("  Try --sep to scan September 2026 cycle (cheaper premiums)")
        qctx.close()
        tctx.close()
        return

    # ── Execute ──
    print(f"\n[6/6] {'EXECUTING LIVE ORDERS' if is_live else 'DRY RUN SUMMARY'}...")

    placed = 0
    errors = 0

    for op in final_orders:
        code = op["code"]
        qty = op["qty"]
        price = op["price"]
        trade_name = op["trade"]["name"]

        if not is_live:
            side_str = op["trade"]["side"]
            print(f"\n  [DRY RUN] {trade_name}")
            print(f"    Would {side_str} {qty}x {code} @ ${price:.2f} LIMIT")
            print(f"    Thesis: {op['trade']['thesis'][:80]}...")
            continue

        # LIVE EXECUTION
        print(f"\n  [LIVE] {trade_name}")
        side = TrdSide.BUY if op["trade"]["side"] == "BUY" else TrdSide.SELL
        limit_price = round(price, 2)

        try:
            ret, data = tctx.place_order(
                price=limit_price,
                qty=qty,
                code=code,
                trd_side=side,
                order_type=MooOrderType.NORMAL,  # LIMIT
                trd_env=TrdEnv.REAL,
            )
            if ret == RET_OK:
                order_id = data.iloc[0].get("order_id", "???")
                print(f"    ORDER [{order_id}]: BUY {qty}x {code} @ ${limit_price:.2f}")
                placed += 1
            else:
                print(f"    FAILED: {data}")
                errors += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            errors += 1

    # ── Summary ──
    print("\n" + "=" * 72)
    if is_live:
        print(f"  LIVE: {placed} orders placed | {errors} errors")
    else:
        print(f"  DRY RUN COMPLETE -- {len(final_orders)} trades scanned")
        print(f"  Estimated cost: ${total_cost:,.0f}")
        print(f"  Use --live to execute for real")
    print()
    print("  LONG CALL THESIS: Silver + Oil bull via SLV/XLE calls")
    print("  CATALYST: Iran/Hormuz escalation, de-dollarization, VIX 31 crisis")
    print("  RISK: 80 DTE Jun 2026. Full premium at risk. Max loss = premium paid.")
    print("=" * 72)

    qctx.close()
    tctx.close()


if __name__ == "__main__":
    main()

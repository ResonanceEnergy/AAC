#!/usr/bin/env python3
r"""
AAC Capital Engine — Moomoo Options Deployment (ADJUSTED)
==========================================================
March 24, 2026 — Deploy capital engine positions to Moomoo REAL account.

Original capital engine signals adjusted to market reality:
  1. Silver bull 57%:  SLV Jun 2026 $70 Call   (OTM ~$7 above spot)
  2. Oil bull 35%:     XLE Jun 2026 $60 Call   (OTM ~$8 above spot)
  3. Consumer bear 8%: XRT Jun 2026 $65 Put    (OTM ~$20 below spot)
  Gold leg dropped — cheapest GLD call ($500C) = $455/ct, exceeds budget.
  SLV covers precious metals thesis as proxy.

Budget: $923 of ~$948 cash. $25 buffer for commissions.

Usage:
  .venv\Scripts\python.exe _moomoo_deploy_options.py              # SCAN ONLY (default)
  .venv\Scripts\python.exe _moomoo_deploy_options.py --live       # LIVE EXECUTION
  .venv\Scripts\python.exe _moomoo_deploy_options.py --price-only # Just get quotes
"""
import argparse
import io
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
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

# Capital engine allocation weights (adjusted for no gold)
ALLOCATIONS = {
    "silver":   0.57,   # 53/(53+30) ≈ 57% — includes precious metals proxy
    "oil":      0.35,   # 30/(53+30) ≈ 35%
    "consumer": 0.08,   # remaining budget
}

# ============================================================================
#  TRADE DEFINITIONS — Adjusted to available strikes
# ============================================================================
TRADES = [
    {
        "name": "SILVER BULL -- SLV Jun 2026 $70 Call",
        "group": "silver",
        "legs": [
            {"underlying": "US.SLV", "strike": 70.0, "right": "CALL", "side": "BUY"},
        ],
    },
    {
        "name": "OIL BULL -- XLE Jun 2026 $60 Call",
        "group": "oil",
        "legs": [
            {"underlying": "US.XLE", "strike": 60.0, "right": "CALL", "side": "BUY"},
        ],
    },
    {
        "name": "CONSUMER SHORT -- XRT Jun 2026 $65 Put",
        "group": "consumer",
        "legs": [
            {"underlying": "US.XRT", "strike": 65.0, "right": "PUT", "side": "BUY"},
        ],
    },
]


def find_option_code(qctx, underlying, strike, right, start, end):
    """Find the specific option contract code from Moomoo chain."""
    ret, data = qctx.get_option_chain(code=underlying, start=start, end=end)
    if ret != RET_OK:
        return None, f"Chain lookup failed: {data}"
    if data.empty:
        return None, "No contracts found in date range"

    # Filter for matching strike and right
    right_str = right.upper()
    matches = data[
        (data["strike_price"] == strike) &
        (data["option_type"] == right_str)
    ]
    if matches.empty:
        # Show available strikes for debugging
        avail = data[data["option_type"] == right_str]["strike_price"].unique()
        avail_sorted = sorted(avail)
        return None, f"No {right_str} @ ${strike}. Available: {avail_sorted[:20]}"

    # Pick the first (closest expiry within the month)
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
    # Calculate mid
    if quote["bid"] > 0 and quote["ask"] > 0:
        quote["mid"] = round((quote["bid"] + quote["ask"]) / 2, 2)
    else:
        quote["mid"] = quote["last"]
    return quote, "OK"


def main():
    parser = argparse.ArgumentParser(description="Moomoo Capital Engine Options Deployment")
    parser.add_argument("--live", action="store_true", help="LIVE execution (default: scan only)")
    parser.add_argument("--price-only", action="store_true", help="Only scan prices, no execution")
    args = parser.parse_args()

    is_live = args.live
    price_only = args.price_only

    print("=" * 70)
    print("  AAC CAPITAL ENGINE — MOOMOO OPTIONS DEPLOYMENT")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {'*** LIVE EXECUTION ***' if is_live else 'SCAN ONLY (dry run)'}")
    print("=" * 70)

    # ── Connect ──
    print("\n[1/5] CONNECTING TO MOOMOO...")
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

    # ── Account Info ──
    print("\n[2/5] ACCOUNT STATUS...")
    ret_acc, data_acc = tctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
    cash = 0.0
    total_assets = 0.0
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
        cash = 948.0  # Fallback from last known

    # Current positions
    ret_pos, data_pos = tctx.position_list_query(trd_env=TrdEnv.REAL)
    if ret_pos == RET_OK and not data_pos.empty:
        print("\n  Current Positions:")
        for _, row in data_pos.iterrows():
            code = row.get("code", "")
            qty = row.get("qty", 0)
            mv = row.get("market_val", 0)
            pl = row.get("pl_val", 0)
            print(f"    {code}: {int(qty)}x | MV: ${mv} | P&L: ${pl}")
    else:
        print("  No existing positions (besides OWL puts)")

    deployable = cash * 0.97  # Keep 3% buffer for commissions (~$28)
    print(f"\n  Deployable Cash: ${deployable:,.2f} (97% of ${cash:,.2f})")

    # ── Scan Option Chains ──
    print("\n[3/5] SCANNING OPTION CHAINS...")
    trade_plans = []

    for trade in TRADES:
        print(f"\n  --- {trade['name']} ---")
        legs_info = []
        trade_debit = 0.0
        trade_valid = True

        for leg in trade["legs"]:
            code, msg = find_option_code(
                qctx, leg["underlying"], leg["strike"], leg["right"],
                JUN_2026_START, JUN_2026_END,
            )
            if code is None:
                print(f"    {leg['side']} {leg['underlying']} ${leg['strike']} {leg['right']}: NOT FOUND")
                print(f"      {msg}")
                trade_valid = False
                legs_info.append({"code": None, "error": msg, **leg})
                continue

            print(f"    {leg['side']} {code}")
            quote, q_msg = get_option_quote(qctx, code)
            if quote is None:
                print(f"      Quote: FAILED ({q_msg})")
                # Still valid — we can use a limit order with estimate
                legs_info.append({"code": code, "quote": None, **leg})
                continue

            print(f"      Bid: ${quote['bid']:.2f} | Ask: ${quote['ask']:.2f} | "
                  f"Last: ${quote['last']:.2f} | Mid: ${quote['mid']:.2f} | "
                  f"Vol: {quote['volume']} | OI: {quote['open_interest']}")

            # Use bid for BUY orders (better fill price), ask for SELL
            if leg["side"] == "BUY":
                price = quote["bid"] if quote["bid"] > 0 else (quote["mid"] if quote["mid"] > 0 else quote["last"])
                trade_debit += price * 100  # per contract
            else:
                price = quote["ask"] if quote["ask"] > 0 else (quote["mid"] if quote["mid"] > 0 else quote["last"])
                trade_debit -= price * 100  # credit from short leg

            legs_info.append({"code": code, "quote": quote, "price": price, **leg})

        if trade_debit > 0:
            print(f"    Net Debit Per Contract: ${trade_debit / 100:.2f} (${trade_debit:.0f} total)")
        elif trade_debit < 0:
            print(f"    Net Credit Per Contract: ${abs(trade_debit) / 100:.2f}")

        trade_plans.append({
            "trade": trade,
            "legs": legs_info,
            "debit_per_contract": trade_debit,
            "valid": trade_valid,
        })

    if price_only:
        print("\n  --price-only mode. Exiting.")
        qctx.close()
        tctx.close()
        return

    # ── Calculate Position Sizing ──
    print("\n[4/5] POSITION SIZING...")

    # Size each trade — 1 contract per leg, enforce budget strictly
    order_plan = []
    total_cost = 0.0

    for tp in trade_plans:
        if not tp["valid"]:
            print(f"  {tp['trade']['name']}: SKIP (contracts not found)")
            continue

        debit = tp["debit_per_contract"]
        qty = 1
        cost = max(0, qty * debit)
        remaining = deployable - total_cost

        if cost > remaining:
            print(f"  {tp['trade']['name']}: SKIP (${cost:.0f} exceeds ${remaining:.0f} remaining)")
            continue

        total_cost += cost

        print(f"  {tp['trade']['name']}:")
        print(f"    Debit/contract: ${debit:.0f} | Qty: {qty} | Cost: ${cost:,.0f} | Remaining: ${deployable - total_cost:,.0f}")

        order_plan.append({
            "trade": tp["trade"],
            "legs": tp["legs"],
            "qty": qty,
            "cost": cost,
        })

    print(f"\n  TOTAL ESTIMATED COST: ${total_cost:,.0f} / ${deployable:,.0f} deployable")
    print(f"  REMAINING AFTER:     ${deployable - total_cost:,.0f}")

    # ── Execute ──
    print(f"\n[5/5] {'EXECUTING LIVE ORDERS' if is_live else 'DRY RUN (no orders placed)'}...")

    placed = 0
    errors = 0

    for op in order_plan:
        trade_name = op["trade"]["name"]
        qty = op["qty"]

        if not is_live:
            print(f"\n  [DRY RUN] {trade_name} x{qty}")
            for leg in op["legs"]:
                if leg.get("code"):
                    side_str = "BUY" if leg["side"] == "BUY" else "SELL"
                    price = leg.get("price", 0)
                    print(f"    Would {side_str} {qty}x {leg['code']} @ ${price:.2f} LIMIT")
            continue

        # LIVE execution
        print(f"\n  [LIVE] {trade_name} x{qty}")
        for leg in op["legs"]:
            code = leg.get("code")
            if not code:
                print(f"    SKIP — no contract code found")
                errors += 1
                continue

            side = TrdSide.BUY if leg["side"] == "BUY" else TrdSide.SELL
            price = leg.get("price", 0)

            if price <= 0:
                print(f"    SKIP {code} — no valid price")
                errors += 1
                continue

            # Use limit order at mid price
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
                    side_str = "BUY" if leg["side"] == "BUY" else "SELL"
                    print(f"    ORDER [{order_id}]: {side_str} {qty}x {code} @ ${limit_price:.2f}")
                    placed += 1
                else:
                    print(f"    FAILED {code}: {data}")
                    errors += 1
            except Exception as e:
                print(f"    ERROR {code}: {e}")
                errors += 1

    # ── Summary ──
    print("\n" + "=" * 70)
    if is_live:
        print(f"  PLACED: {placed} orders | ERRORS: {errors}")
    else:
        print(f"  DRY RUN COMPLETE — {len(order_plan)} trades scanned")
        print(f"  Use --live to execute for real")
    print("=" * 70)

    qctx.close()
    tctx.close()


if __name__ == "__main__":
    main()

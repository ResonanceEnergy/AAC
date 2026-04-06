#!/usr/bin/env python3
"""
AAC War Room -- MONDAY MARCH 30, 2026 MARKET OPEN
===================================================
Pink Moon Fire Peak Entry Window (LEAPS Playbook + Weekly Puts)

SCRAPED PRICES (Fri Mar 27 close):
  SLV  $63.44 (+4.39%)    GLD  $414.70 (+3.51%)    XLE  $62.56 (+1.69%)
  SPY  $634.09 (-1.71%)   QQQ  $562.58 (-1.95%)    IWM  $243.10 (-1.75%)
  XLF  $47.81 (-2.53%)    KRE  $63.37 (-1.71%)     JNK  $94.66 (-0.31%)
  HYG  $78.72 (-0.25%)    BKLN $20.33 (-0.73%)     VIX  31.05 (+13.16%)
  ARCC $17.45 (-2.62%)    OBDC $10.83 (-2.52%)     USO  $124.20 (+5.92%)

REGIME: VIX 31.05 = CRISIS/ELEVATED -- confirms put thesis + commodity LEAPS

LEAPS PLAYBOOK ($38K deployment):
  1. SLV  Jan 2027 65C  -- 55% ($20,900) -- 30 contracts
  2. XLE  Jan 2027 75C  -- 25% ($9,500)  -- 25 contracts
  3. GLD  Jan 2027 410C -- 10% ($3,800)  -- 3 contracts
  4. JNK  Jan 2027 90P  --  5% ($1,900)  -- 8 contracts
  5. XLF  Jan 2027 45P  --  5% ($1,900)  -- 8 contracts

WEEKLY PUTS (Mon buy / Wed-Thu exit):
  6. SPY  $615P  Apr 4  -- 3% OTM, delta -0.25/-0.35
  7. QQQ  $540P  Apr 4  -- 4% OTM
  8. KRE  $58P   Apr 4  -- 8% OTM
  9. HYG  $76P   Apr 4  -- 3.5% OTM

IBKR live port 7496, account U24346218.

Usage:
  .venv\\Scripts\\python.exe _place_monday_open.py                  # SCAN ALL (default)
  .venv\\Scripts\\python.exe _place_monday_open.py --leaps          # DRY RUN LEAPS only
  .venv\\Scripts\\python.exe _place_monday_open.py --weeklies       # DRY RUN weeklies only
  .venv\\Scripts\\python.exe _place_monday_open.py --leaps --live   # LIVE LEAPS execution
  .venv\\Scripts\\python.exe _place_monday_open.py --weeklies --live  # LIVE weeklies
  .venv\\Scripts\\python.exe _place_monday_open.py --all --live     # LIVE everything
  .venv\\Scripts\\python.exe _place_monday_open.py --order 1        # DRY single order by #
  .venv\\Scripts\\python.exe _place_monday_open.py --order 1 --live # LIVE single order
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

# Paper trading by default -- change to 7496 for LIVE
os.environ["IBKR_PORT"] = "7496"

from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

# ============================================================================
#  ORDER BOOK -- Monday March 30, 2026
# ============================================================================
# Prices scraped Fri Mar 27 close. Limits are initial estimates --
# the script will show live bid/ask/mid before executing.
# Adjust LIMIT_PRICE before going --live based on the price scan.
#
# IBKR Jan 2027 LEAPS expiry: Jan 15, 2027 (3rd Friday)
# Weekly expiry: Apr 4, 2026 (Friday)
# ============================================================================

ORDERS = [
    # === LEAPS PLAYBOOK (Pink Moon Fire Peak entry) ===
    {
        "id": 1,
        "group": "leaps",
        "label": "SLV LEAPS CALL",
        "symbol": "SLV",
        "strike": 65.0,
        "expiry": "20270116",   # Jan 16, 2027 (3rd Fri)
        "right": "C",
        "side": "BUY",
        "quantity": 30,
        "limit": 7.00,          # ~$21,000 premium
        "budget": 20_900.0,
        "thesis": "Silver breakout above $30 historic. SLV at $63.44. 65C slightly OTM. "
                  "Iran war + de-dollarization = silver moon shot. 55% of LEAPS book.",
        "underlying_price": 63.44,
    },
    {
        "id": 2,
        "group": "leaps",
        "label": "XLE LEAPS CALL",
        "symbol": "XLE",
        "strike": 75.0,
        "expiry": "20270116",   # Jan 16, 2027
        "right": "C",
        "side": "BUY",
        "quantity": 25,
        "limit": 3.50,          # ~$8,750 premium
        "budget": 9_500.0,
        "thesis": "Energy squeeze. XLE at $62.56, USO +5.92%. Iran war escalation = "
                  "oil spike. 75C gives 20% upside capture. 25% of LEAPS book.",
        "underlying_price": 62.56,
    },
    {
        "id": 3,
        "group": "leaps",
        "label": "GLD LEAPS CALL",
        "symbol": "GLD",
        "strike": 430.0,        # Adjusted from 410 -- GLD at $414.70, 430 = 3.7% OTM
        "expiry": "20270116",   # Jan 16, 2027
        "right": "C",
        "side": "BUY",
        "quantity": 3,
        "limit": 15.00,         # ~$4,500 premium (GLD is expensive)
        "budget": 3_800.0,
        "thesis": "Gold at $414.70, futures $4,524. Central banks + war + VIX 31 = gold new ATH. "
                  "430C is 3.7% OTM. 10% of LEAPS book.",
        "underlying_price": 414.70,
    },
    {
        "id": 4,
        "group": "leaps",
        "label": "JNK LEAPS PUT",
        "symbol": "JNK",
        "strike": 90.0,
        "expiry": "20270116",   # Jan 16, 2027
        "right": "P",
        "side": "BUY",
        "quantity": 8,
        "limit": 2.00,          # ~$1,600 premium
        "budget": 1_900.0,
        "thesis": "High yield credit stress. JNK at $94.66, down from $98.24 high. "
                  "VIX 31 + rate fears = credit spread blowout. 90P is 5% OTM. 5% of LEAPS.",
        "underlying_price": 94.66,
    },
    {
        "id": 5,
        "group": "leaps",
        "label": "XLF LEAPS PUT",
        "symbol": "XLF",
        "strike": 45.0,
        "expiry": "20270116",   # Jan 16, 2027
        "right": "P",
        "side": "BUY",
        "quantity": 8,
        "limit": 2.00,          # ~$1,600 premium
        "budget": 1_900.0,
        "thesis": "Financials cracking. XLF at $47.81 (-2.53%), JPM -3.02%. "
                  "Regional banks KRE -1.71%. 45P is 5.9% OTM. 5% of LEAPS.",
        "underlying_price": 47.81,
    },
    # === WEEKLY PUTS (Master Plan -- Mon buy / Wed-Thu exit) ===
    {
        "id": 6,
        "group": "weeklies",
        "label": "SPY WEEKLY PUT",
        "symbol": "SPY",
        "strike": 615.0,        # 3% OTM from $634.09
        "expiry": "20260404",   # Apr 4, 2026 (Fri)
        "right": "P",
        "side": "BUY",
        "quantity": 2,
        "limit": 3.00,          # ~$600 premium
        "budget": 800.0,
        "thesis": "SPY at $634.09 (-1.71%), 4 weeks of consecutive declines. VIX 31. "
                  "615P = 3% OTM, target delta -0.25/-0.35. Buy Mon, exit Wed-Thu.",
        "underlying_price": 634.09,
    },
    {
        "id": 7,
        "group": "weeklies",
        "label": "QQQ WEEKLY PUT",
        "symbol": "QQQ",
        "strike": 540.0,        # 4% OTM from $562.58
        "expiry": "20260404",   # Apr 4, 2026
        "right": "P",
        "side": "BUY",
        "quantity": 2,
        "limit": 3.00,          # ~$600 premium
        "budget": 800.0,
        "thesis": "QQQ at $562.58 (-1.95%), Nasdaq -2.15%. Tech selloff. AMZN -4.02%. "
                  "540P = 4% OTM. Buy Mon, exit Wed-Thu.",
        "underlying_price": 562.58,
    },
    {
        "id": 8,
        "group": "weeklies",
        "label": "KRE WEEKLY PUT",
        "symbol": "KRE",
        "strike": 58.0,         # 8.5% OTM from $63.37
        "expiry": "20260404",   # Apr 4, 2026
        "right": "P",
        "side": "BUY",
        "quantity": 3,
        "limit": 0.50,          # ~$150 premium
        "budget": 300.0,
        "thesis": "Regional banks under pressure. KRE at $63.37 (-1.71%). "
                  "58P = 8.5% OTM. Cheap lottery if banking stress escalates.",
        "underlying_price": 63.37,
    },
    {
        "id": 9,
        "group": "weeklies",
        "label": "HYG WEEKLY PUT",
        "symbol": "HYG",
        "strike": 76.0,         # 3.5% OTM from $78.72
        "expiry": "20260404",   # Apr 4, 2026
        "right": "P",
        "side": "BUY",
        "quantity": 3,
        "limit": 0.40,          # ~$120 premium
        "budget": 300.0,
        "thesis": "High yield stress indicator. HYG at $78.72 (-0.25%). "
                  "76P = 3.5% OTM. Credit spread blowout play.",
        "underlying_price": 78.72,
    },
]

# ============================================================================
#  MARKET SNAPSHOT (for display)
# ============================================================================
MARKET_SNAPSHOT = {
    "date": "Fri Mar 27, 2026 Close",
    "indices": {
        "S&P 500": ("6,368.85", "-1.67%"),
        "Dow 30": ("45,166.64", "-1.73%"),
        "Nasdaq": ("20,948.36", "-2.15%"),
        "Russell 2000": ("2,449.70", "-1.75%"),
        "VIX": ("31.05", "+13.16%"),
    },
    "commodities": {
        "Gold Futures": "$4,524.30 (+2.62%)",
        "GLD": "$414.70 (+3.51%)",
        "SLV": "$63.44 (+4.39%)",
        "XLE": "$62.56 (+1.69%)",
        "USO": "$124.20 (+5.92%)",
        "GDX": "$85.79 (+4.13%)",
    },
    "equities": {
        "SPY": "$634.09 (-1.71%)",
        "QQQ": "$562.58 (-1.95%)",
        "IWM": "$243.10 (-1.75%)",
    },
    "credit": {
        "JNK": "$94.66 (-0.31%)",
        "HYG": "$78.72 (-0.25%)",
        "BKLN": "$20.33 (-0.73%)",
        "LQD": "$107.62 (-0.24%)",
        "KRE": "$63.37 (-1.71%)",
        "XLF": "$47.81 (-2.53%)",
    },
    "rates_crypto": {
        "10Y": "4.44%",
        "BTC": "$66,463 (+0.50%)",
        "ETH": "$1,994 (+0.03%)",
    },
    "bdc_targets": {
        "OBDC": "$10.83 (-2.52%) AH $10.85",
        "ARCC": "$17.45 (-2.62%)",
    },
    "headlines": [
        "Iran War Started 1 Month Ago -- Barrons",
        "Gold Falls 15% as Central Banks Reassess $4.3 Trillion Reserves",
        "Stocks dropped 4 consecutive weeks",
        "VIX 31.05 = CRISIS regime (above 30 threshold)",
        "NKE earnings Mar 31, CAG earnings Apr 1",
    ],
}


def print_banner():
    print("=" * 78)
    print("  AAC WAR ROOM -- MONDAY MARCH 30, 2026 -- MARKET OPEN ORDERS")
    print("  Pink Moon Fire Peak -- LEAPS Playbook + Weekly Puts")
    print("=" * 78)


def print_market_snapshot():
    ms = MARKET_SNAPSHOT
    print(f"\n  MARKET INTEL ({ms['date']})")
    print("  " + "-" * 74)

    print("  INDICES:")
    for name, (val, chg) in ms["indices"].items():
        arrow = "!!" if name == "VIX" and float(val) > 30 else ""
        print(f"    {name:<16} {val:>12}  ({chg}) {arrow}")

    print("  COMMODITIES:")
    for name, val in ms["commodities"].items():
        print(f"    {name:<16} {val}")

    print("  EQUITIES:")
    for name, val in ms["equities"].items():
        print(f"    {name:<16} {val}")

    print("  CREDIT/FINANCIALS:")
    for name, val in ms["credit"].items():
        print(f"    {name:<16} {val}")

    print("  RATES/CRYPTO:")
    for name, val in ms["rates_crypto"].items():
        print(f"    {name:<16} {val}")

    print("  BDC TARGETS (existing book):")
    for name, val in ms["bdc_targets"].items():
        print(f"    {name:<16} {val}")

    print("\n  HEADLINES:")
    for h in ms["headlines"]:
        print(f"    >> {h}")
    print()


def print_order_table(orders):
    print("  " + "-" * 74)
    print(f"  {'#':<3} {'Label':<22} {'Contract':<24} {'Qty':>4} {'Limit':>7} {'Premium':>9} {'Budget':>9}")
    print("  " + "-" * 74)
    total_premium = 0
    total_budget = 0
    for o in orders:
        contract_str = f"{o['symbol']} ${o['strike']:.0f}{o['right']} {o['expiry'][:6]}"
        premium = o["quantity"] * o["limit"] * 100
        total_premium += premium
        total_budget += o["budget"]
        print(f"  {o['id']:<3} {o['label']:<22} {contract_str:<24} {o['quantity']:>4} "
              f"${o['limit']:>5.2f} ${premium:>8,.0f} ${o['budget']:>8,.0f}")
    print("  " + "-" * 74)
    print(f"  {'':>53} ${total_premium:>8,.0f} ${total_budget:>8,.0f}")
    print()


async def scan_and_execute(orders_to_run, mode="scan"):
    """
    mode: 'scan' = price only, 'dry' = dry run, 'live' = execute
    """
    print(f"\n  Connecting to IBKR (port {os.environ.get('IBKR_PORT', '7496')})...")
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

    # Show existing positions
    portfolio = ib.portfolio()
    symbols_in_play = set(o["symbol"] for o in orders_to_run)
    existing = [p for p in portfolio if p.contract.symbol in symbols_in_play]
    if existing:
        print(f"\n  EXISTING POSITIONS (in target symbols):")
        for pos in existing:
            c = pos.contract
            tag = f"{c.secType} {c.symbol}"
            if hasattr(c, "strike") and c.strike:
                tag += f" ${c.strike:.0f}{c.right} exp {c.lastTradeDateOrContractMonth}"
            print(f"    {tag} qty={pos.position:.0f} mktVal=${pos.marketValue:.2f} "
                  f"uPnL=${pos.unrealizedPNL:+.2f}")

    # Process each order
    from ib_insync import LimitOrder as IbLimitOrder
    from ib_insync import Option as IbOption

    results = []
    for o in orders_to_run:
        print(f"\n  {'=' * 70}")
        print(f"  ORDER #{o['id']}: {o['label']}")
        print(f"  {o['side']} {o['quantity']}x {o['symbol']} ${o['strike']:.0f}{o['right']} "
              f"exp {o['expiry']} @ ${o['limit']:.2f}")
        print(f"  Thesis: {o['thesis'][:100]}")
        print(f"  {'=' * 70}")

        # Qualify contract
        contract = IbOption(o["symbol"], o["expiry"], o["strike"], o["right"],
                            "SMART", currency="USD")
        qualified = ib.qualifyContracts(contract)

        if not qualified:
            print(f"  !! CONTRACT NOT FOUND: {o['symbol']} ${o['strike']:.0f}{o['right']} {o['expiry']}")
            print(f"  !! Trying nearby expiries...")

            # Try +/- 7 days for weekly, or next monthly for LEAPS
            found = False
            if o["group"] == "weeklies":
                # Try nearby Fridays
                import datetime
                base = datetime.datetime.strptime(o["expiry"], "%Y%m%d")
                for delta in [-1, +1, -2, +2, -7, +7]:
                    alt_dt = base + datetime.timedelta(days=delta)
                    alt_exp = alt_dt.strftime("%Y%m%d")
                    alt_contract = IbOption(o["symbol"], alt_exp, o["strike"], o["right"],
                                            "SMART", currency="USD")
                    alt_q = ib.qualifyContracts(alt_contract)
                    if alt_q:
                        contract = alt_q[0]
                        print(f"  >> Found nearby expiry: {alt_exp}")
                        found = True
                        break
            else:
                # For LEAPS, try 20270117, 20270115, 20261218
                for alt_exp in ["20270117", "20270115", "20261218", "20261219", "20270121"]:
                    alt_contract = IbOption(o["symbol"], alt_exp, o["strike"], o["right"],
                                            "SMART", currency="USD")
                    alt_q = ib.qualifyContracts(alt_contract)
                    if alt_q:
                        contract = alt_q[0]
                        print(f"  >> Found LEAPS expiry: {alt_exp}")
                        found = True
                        break

            if not found:
                print(f"  !! SKIPPING -- no valid contract found")
                results.append({"id": o["id"], "label": o["label"], "status": "NOT_FOUND"})
                continue
        else:
            contract = qualified[0]

        print(f"  Contract: conId={contract.conId} {contract.localSymbol}")

        # Get quote
        ib.reqMarketDataType(3)  # delayed if no real-time sub
        ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(3)
        ticker = ib.ticker(contract)

        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
        last = ticker.last if ticker.last and ticker.last > 0 else 0
        mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else 0

        print(f"  Bid: ${bid:.2f}  Ask: ${ask:.2f}  Mid: ${mid:.2f}  Last: ${last:.2f}")
        print(f"  Our limit: ${o['limit']:.2f}")

        if mid > 0:
            adjusted_qty = int(o["budget"] / (mid * 100)) or 1
            print(f"  >> At mid price, budget ${o['budget']:,.0f} buys {adjusted_qty} contracts "
                  f"(${adjusted_qty * mid * 100:,.0f} premium)")

        result = {
            "id": o["id"],
            "label": o["label"],
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "last": last,
            "limit": o["limit"],
            "status": "SCANNED",
        }

        if mode == "scan":
            results.append(result)
            continue

        if mode == "dry":
            print(f"  DRY RUN -- Would place: {o['side']} {o['quantity']}x @ ${o['limit']:.2f}")
            print(f"  Premium: ${o['quantity'] * o['limit'] * 100:,.0f}")
            result["status"] = "DRY_RUN"
            results.append(result)
            continue

        # === LIVE EXECUTION ===
        print(f"\n  >>> EXECUTING: {o['side']} {o['quantity']}x {contract.localSymbol} @ ${o['limit']:.2f}")

        try:
            order_obj = IbLimitOrder(o["side"], o["quantity"], o["limit"])
            order_obj.account = acct
            order_obj.tif = "DAY"

            trade = ib.placeOrder(contract, order_obj)
            await asyncio.sleep(2)

            print(f"  ORDER PLACED!")
            print(f"  Order ID:  {trade.order.orderId}")
            print(f"  Status:    {trade.orderStatus.status}")
            print(f"  Filled:    {trade.orderStatus.filled}")
            print(f"  Remaining: {trade.orderStatus.remaining}")
            print(f"  Premium:   ${o['quantity'] * o['limit'] * 100:,.0f}")

            if trade.fills:
                for fill in trade.fills:
                    print(f"  FILL: {fill.execution.shares}x @ ${fill.execution.price:.2f}")

            result["status"] = trade.orderStatus.status
            result["order_id"] = trade.order.orderId
        except Exception as e:
            print(f"  !! ORDER FAILED: {e}")
            result["status"] = f"FAILED: {e}"

        results.append(result)
        await asyncio.sleep(1)  # Brief pause between orders

    # === SUMMARY ===
    print(f"\n{'=' * 78}")
    print(f"  ORDER SUMMARY -- {mode.upper()}")
    print(f"{'=' * 78}")
    for r in results:
        print(f"  #{r['id']:<3} {r['label']:<22} bid=${r.get('bid', 0):.2f} "
              f"ask=${r.get('ask', 0):.2f} mid=${r.get('mid', 0):.2f} "
              f"STATUS: {r['status']}")

    # Check open orders
    if mode == "live":
        print("\n  Waiting 5s for fill updates...")
        await asyncio.sleep(5)
        open_orders = ib.openOrders()
        if open_orders:
            print(f"\n  OPEN ORDERS ({len(open_orders)}):")
            for oo in open_orders:
                print(f"    ID {oo.orderId}: {oo.action} {oo.totalQuantity}x @ "
                      f"${oo.lmtPrice:.2f} type={oo.orderType}")

    await connector.disconnect()
    print("\n  Disconnected from IBKR. Done.")
    return results


async def main():
    # Parse args
    args = set(sys.argv[1:])
    live = "--live" in args
    scan_only = not ("--leaps" in args or "--weeklies" in args or "--all" in args or "--order" in args)

    # Determine mode
    if scan_only and not live:
        mode = "scan"
    elif live:
        mode = "live"
    else:
        mode = "dry"

    # Filter orders
    if "--order" in args:
        try:
            idx = sys.argv.index("--order")
            order_id = int(sys.argv[idx + 1])
            selected = [o for o in ORDERS if o["id"] == order_id]
        except (IndexError, ValueError):
            print("  ERROR: --order requires a number (1-9)")
            return
    elif "--leaps" in args:
        selected = [o for o in ORDERS if o["group"] == "leaps"]
    elif "--weeklies" in args:
        selected = [o for o in ORDERS if o["group"] == "weeklies"]
    elif "--all" in args:
        selected = ORDERS[:]
    else:
        selected = ORDERS[:]  # scan all by default

    print_banner()

    # Mode display
    if mode == "live":
        print(f"\n  *** LIVE MODE *** -- {len(selected)} orders will be placed on IBKR")
        print(f"  Port: {os.environ.get('IBKR_PORT', '7496')} "
              f"({'PAPER' if os.environ.get('IBKR_PORT') == '7497' else 'LIVE'})")
        print(f"\n  Press Ctrl+C within 5 seconds to abort...")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n  Aborted.")
            return
    else:
        print(f"\n  Mode: {'PRICE SCAN' if mode == 'scan' else 'DRY RUN'}")
        print(f"  Orders selected: {len(selected)}")
        if mode == "scan":
            print("  (Add --leaps/--weeklies/--all + --live to execute)")

    print_market_snapshot()
    print_order_table(selected)

    await scan_and_execute(selected, mode=mode)


if __name__ == "__main__":
    asyncio.run(main())

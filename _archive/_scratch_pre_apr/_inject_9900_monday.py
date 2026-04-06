#!/usr/bin/env python3
"""
AAC War Room -- $9,900 x2 INJECTION: WealthSimple + Moomoo
=============================================================
Monday March 30 + Tuesday March 31, 2026
Pink Moon Fire Peak -- Capital Deployment Plan

INJECTION: $9,900 CAD to WealthSimple + $9,900 CAD to Moomoo = $19,800 CAD total
  WealthSimple: $9,900 CAD + existing = TFSA investable
  Moomoo:       $9,900 CAD + $2,609 existing = $12,509 CAD (~$8,756 USD at 0.70)

SCRAPED PRICES (Fri Mar 27 close):
  Gold  $4,524/oz (+3.5%)     Silver $63.44 (+4.4%)     Oil   $100+
  VIX   31.05 (+13.2%)        SPY    $634.09 (-1.7%)    DXY   declining
  XLE   $62.56 (+1.7%)        GLD    $414.70 (+3.5%)    GDX   $86

REGIME: VIX 31 = CRISIS -- confirms commodity bull + equity bear thesis

============================================================================
WEALTHSIMPLE TFSA -- $9,900 CAD INJECTION ($14,100 CAD total)
============================================================================
  All gains tax-free. Canadian-listed ETFs only.

  1. CGL.TO  -- CI Gold Bullion ETF          $5,400 CAD (55%)  MONDAY
     Gold at $4,524, thesis target $5,000+. Tax-free gold exposure.
     ~216 shares @ ~$25 CAD

  2. XEG.TO  -- iShares S&P/TSX Energy       $2,700 CAD (27%)  MONDAY
     Oil >$100 thesis. Canadian energy producers benefit from Hormuz risk.
     ~180 shares @ ~$15 CAD

  3. HSD.TO  -- BetaPro S&P 500 2x Bear      $1,200 CAD (12%)  TUESDAY
     VIX 31 = crisis. Leveraged S&P bear hedge. 2-4 WEEK MAX HOLD.
     ~50 shares @ ~$24 CAD

  4. CASH RESERVE                             $600 CAD (6%)
     Dips, follow-on buys, mid-week opportunities.

============================================================================
MOOMOO -- $9,900 CAD INJECTION (~$6,930 USD buying power at 0.70 FX)
============================================================================
  Long calls / LEAPS on silver, oil, gold miners. Leveraged thesis exposure.
  Account is CAD-denominated (FUTUCA). US options trade in USD via FX conversion.
  $2,609 existing is LOCKED in positions (OWL/SLV/XLE). Free cash = $9,900 CAD.

  5. SLV Jan 2027 $65C LEAPS                 $3,990 USD (55%)  MONDAY
     Silver at $63.44. Slightly OTM. 290 DTE. Iran + de-dollarization.
     ~3 contracts @ ~$13.30 ($3,990 premium)

  6. XLE Jan 2027 $65C LEAPS                 $1,575 USD (22%)  MONDAY
     XLE at $62.56. Energy LEAPS. Oil >$100 + Hormuz catalyst.
     ~3 contracts @ ~$5.25 ($1,575 premium)

  7. GDX Jul 2026 $90C                       $1,686 USD (23%)  MONDAY
     Gold miners at $86. Gold $4,524 thesis supports miners breakout.
     ~2 contracts @ ~$8.43 ($1,686 premium)

  ALL IN -- no cash reserve (~$67 for commissions).

============================================================================
DEPLOYMENT SCHEDULE
============================================================================

  SUNDAY MARCH 29 (tonight):
    [ ] Transfer $9,900 CAD to WealthSimple TFSA (Interac e-transfer)
    [x] $9,900 CAD funded to Moomoo (DONE)
    [ ] Verify Moomoo OpenD is running on desktop

  MONDAY MARCH 30:
    09:00  Start Moomoo OpenD, confirm connection
    09:30  Market open -- WAIT 30 min for gap/volatility
    10:00  WealthSimple: BUY CGL.TO ($5,400 CAD) -- LIMIT order
    10:15  WealthSimple: BUY XEG.TO ($2,700 CAD) -- LIMIT order
    10:30  WealthSimple: BUY HSD.TO ($1,200 CAD) -- LIMIT order
    10:30  Moomoo: Run price scan --
             .venv\\Scripts\\python.exe _inject_9900_monday.py --moomoo
    10:45  Moomoo: BUY SLV Jan 2027 $65C x3 -- LIMIT at mid
    11:00  Moomoo: BUY XLE Jan 2027 $65C x3 -- LIMIT at mid
    11:15  Moomoo: BUY GDX Jul 2026 $90C x2 -- LIMIT at mid
    15:30  Review all fills. Adjust unfilled limits.

============================================================================

Usage:
  .venv\\Scripts\\python.exe _inject_9900_monday.py              # SHOW PLAN (default)
  .venv\\Scripts\\python.exe _inject_9900_monday.py --moomoo     # Scan Moomoo chains
  .venv\\Scripts\\python.exe _inject_9900_monday.py --live       # Moomoo LIVE execution
"""
import sys
import io
import os
import argparse
from datetime import datetime

if hasattr(sys.stdout, "buffer") and (sys.stdout is None or sys.stdout.encoding.lower() != "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

# ============================================================================
#  WEALTHSIMPLE PLAN (manual execution -- no API trading)
# ============================================================================
WS_PLAN = [
    {
        "day": "MONDAY",
        "ticker": "CGL.TO",
        "name": "CI Gold Bullion ETF",
        "budget_cad": 5400,
        "est_price_cad": 25.0,
        "est_shares": 216,
        "thesis": "Gold at $4,524/oz. Tax-free gold exposure in TFSA. "
                  "Iran war + de-dollarization = gold $5,000+ target.",
        "order_type": "LIMIT at bid/mid -- not market",
    },
    {
        "day": "MONDAY",
        "ticker": "XEG.TO",
        "name": "iShares S&P/TSX Capped Energy",
        "budget_cad": 2700,
        "est_price_cad": 15.0,
        "est_shares": 180,
        "thesis": "Oil above $100. Hormuz strait risk. Canadian energy producers "
                  "benefit from oil thesis + CAD strength.",
        "order_type": "LIMIT at bid/mid -- not market",
    },
    {
        "day": "MONDAY",
        "ticker": "HSD.TO",
        "name": "BetaPro S&P 500 2x Daily Bear",
        "budget_cad": 1200,
        "est_price_cad": 24.0,
        "est_shares": 50,
        "thesis": "VIX 31.05 = crisis. S&P bear thesis via leveraged inverse ETF. "
                  "SHORT-TERM ONLY (2-4 weeks). Daily reset = decay on hold.",
        "order_type": "LIMIT -- enter on any morning weakness",
        "warning": "2x LEVERAGED INVERSE -- DO NOT HOLD >4 WEEKS. Daily compounding decay.",
    },
]

# ============================================================================
#  MOOMOO PLAN (script-assisted execution)
# ============================================================================
HOST = "127.0.0.1"
PORT = 11111
TRADE_PIN = "069420"

MOO_PLAN = [
    {
        "day": "MONDAY",
        "name": "SLV Jan 2027 $65 Call LEAPS",
        "underlying": "US.SLV",
        "strike": 65.0,
        "right": "CALL",
        "expiry_start": "2027-01-01",
        "expiry_end": "2027-01-31",
        "side": "BUY",
        "qty": 3,
        "est_premium": 13.30,
        "budget_usd": 3990,
        "thesis": "Silver at $63.44. $65C slightly OTM. Jan 2027 = 290 DTE. "
                  "Iran + precious metals bull. Adds to existing Jun SLV calls.",
    },
    {
        "day": "MONDAY",
        "name": "XLE Jan 2027 $65 Call LEAPS",
        "underlying": "US.XLE",
        "strike": 65.0,
        "right": "CALL",
        "expiry_start": "2027-01-01",
        "expiry_end": "2027-01-31",
        "side": "BUY",
        "qty": 3,
        "est_premium": 5.25,
        "budget_usd": 1575,
        "thesis": "XLE at $62.56. $65C slightly OTM. Oil >$100 thesis + Hormuz. "
                  "LEAPS for patient exposure. Adds to existing Jun XLE calls.",
    },
    {
        "day": "MONDAY",
        "name": "GDX Jul 2026 $90 Call",
        "underlying": "US.GDX",
        "strike": 90.0,
        "right": "CALL",
        "expiry_start": "2026-07-01",
        "expiry_end": "2026-07-31",
        "side": "BUY",
        "qty": 2,
        "est_premium": 8.43,
        "budget_usd": 1686,
        "thesis": "Gold miners at GDX $86. Gold >$4,500 supports miners. "
                  "War room: research GDX/GDXJ. $90C = 4.7% OTM, 120 DTE.",
    },
]


def show_plan():
    """Display the full deployment plan."""
    # Pull live balances from central config
    try:
        from config.account_balances import Balances
        _moo_bal = Balances.moomoo()
        _ws_bal = Balances.wealthsimple()
        _fx = Balances.fx_cad_usd()
    except Exception:
        _moo_bal = 12509.26
        _ws_bal = 18637.76
        _fx = 0.70

    ws_injection = 9900  # CAD
    moo_injection = 9900  # USD
    ws_existing = round(_ws_bal - ws_injection, 2) if _ws_bal > ws_injection else 0
    moo_existing_locked = round(_moo_bal - moo_injection, 2) if _moo_bal > moo_injection else 0
    cad_usd = _fx

    ws_total = ws_injection + ws_existing
    moo_usd_buying = round(moo_injection * cad_usd, 2)  # CAD → USD

    ws_deploy = sum(t["budget_cad"] for t in WS_PLAN)
    ws_reserve = ws_total - ws_deploy
    moo_deploy = sum(t["budget_usd"] for t in MOO_PLAN)
    moo_reserve_usd = round(moo_usd_buying - moo_deploy, 2)

    print("=" * 74)
    print("  $9,900 x 2 INJECTION PLAN -- WealthSimple CAD + Moomoo CAD")
    print(f"  Monday March 30 + Tuesday March 31, 2026")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 74)

    print(f"\n  TOTAL INJECTION: ${ws_injection + moo_injection:,} CAD "
          f"($9,900 WS + $9,900 Moo)")
    print(f"  FX rate: 1 CAD = {cad_usd} USD")
    print()
    print(f"  WEALTHSIMPLE TFSA:")
    print(f"    Injection:    ${ws_injection:,.2f} CAD")
    print(f"    Existing:     ${ws_existing:,.2f} CAD")
    print(f"    Total:        ${ws_total:,.2f} CAD (~${ws_total * cad_usd:,.0f} USD)")
    print(f"    Deploying:    ${ws_deploy:,} CAD | Reserve: ${ws_reserve:,.2f} CAD")
    print()
    print(f"  MOOMOO (CAD account, US options via FX):")
    print(f"    Injection:    ${moo_injection:,.2f} CAD")
    print(f"    USD buying:   ~${moo_usd_buying:,.2f} USD (at {cad_usd} FX)")
    print(f"    Locked:       ${moo_existing_locked:,.2f} CAD (existing positions)")
    print(f"    Deploying:    ${moo_deploy:,} USD | Reserve: ~${moo_reserve_usd:,.2f} USD")

    # WealthSimple
    print("\n" + "-" * 74)
    print("  WEALTHSIMPLE TFSA -- $9,900 CAD injection (manual via app/web)")
    print("  Account: Tax-Free Savings Account (ALL gains tax-free)")
    print("-" * 74)
    for i, t in enumerate(WS_PLAN, 1):
        print(f"\n  [{i}] {t['day']} -- {t['ticker']} ({t['name']})")
        print(f"      Budget: ${t['budget_cad']:,} CAD | "
              f"~{t['est_shares']} shares @ ~${t['est_price_cad']:.2f}")
        print(f"      Order:  {t['order_type']}")
        print(f"      Thesis: {t['thesis']}")
        if "warning" in t:
            print(f"      *** WARNING: {t['warning']} ***")
    print(f"\n  Cash reserve: ${ws_reserve:,.2f} CAD (dips/opportunities)")
    print(f"  Total WS deployment: ${ws_deploy:,} CAD + ${ws_reserve:,.2f} reserve = ${ws_total:,.2f} CAD (~${ws_total * cad_usd:,.0f} USD)")

    # Moomoo
    print("\n" + "-" * 74)
    print(f"  MOOMOO -- $9,900 CAD injection (~${9900 * cad_usd:,.0f} USD buying power)")
    print("  Account: FUTUCA | PIN: 069420 | Port: 11111 | REAL mode | CAD account")
    print("  Existing (LOCKED): OWL $20P Jan 2027, SLV Jun calls, XLE Jun calls (~$2,609)")
    print("-" * 74)
    for i, t in enumerate(MOO_PLAN, 1):
        cost = t["qty"] * t["est_premium"] * 100
        print(f"\n  [{i + len(WS_PLAN)}] {t['day']} -- {t['name']}")
        print(f"      {t['side']} {t['qty']}x {t['underlying']} "
              f"${t['strike']} {t['right']}")
        print(f"      Est premium: ${t['est_premium']:.2f}/ct | "
              f"Total: ${cost:,.0f} | Budget: ${t['budget_usd']:,}")
        print(f"      Thesis: {t['thesis']}")
    moo_deploy_usd = sum(t["budget_usd"] for t in MOO_PLAN)
    moo_usd_buying = round(9900 * cad_usd, 2)
    moo_reserve_usd = round(moo_usd_buying - moo_deploy_usd, 2)
    print(f"\n  Cash reserve: ~${moo_reserve_usd:,.0f} USD (FX slippage + commissions)")
    print(f"  Moomoo: $9,900 CAD (~${moo_usd_buying:,.0f} USD) | Deploy ${moo_deploy_usd:,} USD | Reserve ~${moo_reserve_usd:,.0f} USD")

    # Schedule
    print("\n" + "-" * 74)
    print("  DEPLOYMENT SCHEDULE")
    print("-" * 74)
    print("""
  SUNDAY MARCH 29 (tonight):
    [ ] Transfer $9,900 CAD to WealthSimple TFSA (Interac e-transfer)
    [x] $9,900 CAD funded to Moomoo (DONE)
    [ ] Verify Moomoo OpenD is running on desktop

  MONDAY MARCH 30:
    09:00  Start Moomoo OpenD, confirm connection
    09:30  Market open -- WAIT 30 min for gap/volatility
    10:00  WealthSimple: BUY CGL.TO ($5,400 CAD) -- LIMIT order
    10:15  WealthSimple: BUY XEG.TO ($2,700 CAD) -- LIMIT order
    10:30  WealthSimple: BUY HSD.TO ($1,200 CAD) -- LIMIT order
    10:30  Moomoo: Run price scan --
             .venv\\Scripts\\python.exe _inject_9900_monday.py --moomoo
    10:45  Moomoo: BUY SLV Jan 2027 $65C x3 -- LIMIT at mid
    11:00  Moomoo: BUY XLE Jan 2027 $65C x3 -- LIMIT at mid
    11:15  Moomoo: BUY GDX Jul 2026 $90C x2 -- LIMIT at mid
    15:30  Review all fills. Cancel any remaining unfilled orders.
    """)

    # Risk
    print("-" * 74)
    print("  RISK MANAGEMENT")
    print("-" * 74)
    print("""
  MAX LOSS (all positions to zero):
    WealthSimple: $9,300 CAD (CGL+XEG decline, HSD decays)
    Moomoo:       $9,400 USD (all options expire worthless)
    TOTAL:        ~$18,700 combined (94% of injection)

  STOP/EXIT RULES:
    CGL.TO   -- Hold unless gold drops below $3,800 (16% drawdown)
    XEG.TO   -- Hold unless oil drops below $75 (25% drawdown)
    HSD.TO   -- EXIT after 4 weeks MAX regardless. Daily decay kills.
    SLV LEAPS -- Take 50% profit if SLV >$75. Cut at 50% loss.
    XLE LEAPS -- Take 50% profit if XLE >$80. Cut at 50% loss.
    GDX call  -- Take profit at 100% gain. Shorter DTE = tighter stops.

  THESIS INVALIDATION:
    - Iran de-escalation (oil drops <$80) --> close oil/energy positions
    - Gold drops below $4,000 --> reduce CGL.TO by 50% (~$2,700 CAD)
    - VIX drops below 18 --> close HSD.TO immediately (bear thesis dead)
    """)

    # Summary table
    print("-" * 74)
    print("  POSITION SUMMARY AFTER DEPLOYMENT")
    print("-" * 74)
    print("""
  WEALTHSIMPLE TFSA (all CAD, tax-free):
    CGL.TO   ~216 shares  $5,400  Gold bullion -- core thesis
    XEG.TO   ~180 shares  $2,700  Canadian energy -- oil thesis
    HSD.TO   ~50 shares   $1,200  S&P 2x bear -- short-term hedge
    Cash                  $600    Reserve
    TOTAL                 $9,900  (+ existing = TFSA total)

  MOOMOO (CAD account, US options via FX at ~0.70):
    Free cash: $9,900 CAD (~$6,930 USD)
    SLV $65C Jan 2027  3x   $3,990  Silver LEAPS -- precious metals
    XLE $65C Jan 2027  3x   $1,575  Energy LEAPS -- oil thesis
    GDX $90C Jul 2026  2x   $1,686  Gold miners -- gold thesis
    ALL IN                   $7,251  (~$67 for commissions)
    EXISTING (locked): OWL $20P, SLV Jun, XLE Jun (~$2,609 CAD)
    TOTAL NEW:               ~$7,251 USD (all in)

  COMBINED THESIS EXPOSURE:
    Gold/Silver: 56%  (CGL.TO + SLV LEAPS + GDX calls)
    Oil/Energy:  28%  (XEG.TO + XLE LEAPS)
    Bear hedge:   7%  (HSD.TO)
    Cash reserve: <1%  (~$67 commissions only)
    Combined:     $19,800 CAD injected ($9,900 CAD WS + $9,900 CAD Moo)
    """)


def scan_moomoo_chains():
    """Connect to Moomoo and scan option chains for the planned trades."""
    try:
        from moomoo import (
            OpenQuoteContext,
            OpenSecTradeContext,
            TrdEnv,
            SecurityFirm,
            TrdMarket,
            Currency,
            RET_OK,
        )
    except ImportError:
        print("  ERROR: moomoo SDK not installed. Run: pip install moomoo-api")
        return

    print("\n" + "=" * 74)
    print("  MOOMOO CHAIN SCAN -- $9,900 USD Deployment")
    print("=" * 74)

    qctx = OpenQuoteContext(host=HOST, port=PORT)
    print("  Quote context: OK")

    tctx = OpenSecTradeContext(
        host=HOST, port=PORT,
        security_firm=SecurityFirm.FUTUCA,
        filter_trdmarket=TrdMarket.US,
    )
    ret_unlock, unlock_msg = tctx.unlock_trade(password=TRADE_PIN)
    print(f"  Trade unlock: {'OK' if ret_unlock == RET_OK else f'FAILED ({unlock_msg})'}")

    # Account status
    ret_acc, data_acc = tctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
    if ret_acc == RET_OK:
        cash = float(data_acc["cash"].iloc[0] or 0)
        total = float(data_acc["total_assets"].iloc[0] or 0)
        print(f"  Cash: ${cash:,.2f} | Total Assets: ${total:,.2f}")
    else:
        print(f"  Account query failed: {data_acc}")

    # Current positions
    ret_pos, data_pos = tctx.position_list_query(trd_env=TrdEnv.REAL)
    if ret_pos == RET_OK and not data_pos.empty:
        print("\n  Current Positions:")
        for _, row in data_pos.iterrows():
            code = row.get("code", "")
            qty = int(row.get("qty", 0))
            mv = row.get("market_val", 0)
            pl = row.get("pl_val", 0)
            print(f"    {code}: {qty}x | MV ${mv} | P&L ${pl}")

    # Scan each planned trade
    for trade in MOO_PLAN:
        print(f"\n  --- {trade['name']} ---")
        ret, data = qctx.get_option_chain(
            code=trade["underlying"],
            start=trade["expiry_start"],
            end=trade["expiry_end"],
        )
        if ret != RET_OK:
            print(f"    Chain lookup FAILED: {data}")
            continue
        if data.empty:
            print(f"    No contracts in {trade['expiry_start']} to {trade['expiry_end']}")
            continue

        # Filter for our strike and right
        right_str = trade["right"].upper()
        target_strike = trade["strike"]

        # Show nearby strikes
        strikes_avail = sorted(
            data[data["option_type"] == right_str]["strike_price"].unique()
        )
        nearby = [s for s in strikes_avail if abs(s - target_strike) <= 10]
        print(f"    Available {right_str} strikes (near ${target_strike}): {nearby}")

        # Get exact match
        matches = data[
            (data["strike_price"] == target_strike) &
            (data["option_type"] == right_str)
        ]
        if matches.empty:
            print(f"    ${target_strike} {right_str} NOT FOUND. Closest: {nearby[:5]}")
            continue

        code = matches.iloc[0]["code"]
        expiry = matches.iloc[0].get("strike_time", "unknown")
        print(f"    Contract: {code} (exp {expiry})")

        # Get quote
        ret_q, data_q = qctx.get_market_snapshot([code])
        if ret_q == RET_OK and not data_q.empty:
            row = data_q.iloc[0]
            bid = float(row.get("bid_price", 0) or 0)
            ask = float(row.get("ask_price", 0) or 0)
            last = float(row.get("last_price", 0) or 0)
            vol = int(row.get("volume", 0) or 0)
            oi = int(row.get("open_interest", 0) or 0)
            mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else last
            cost = mid * 100 * trade["qty"]
            print(f"    Bid ${bid:.2f} | Ask ${ask:.2f} | Mid ${mid:.2f} | "
                  f"Last ${last:.2f}")
            print(f"    Vol {vol} | OI {oi}")
            print(f"    >>> {trade['qty']}x @ ${mid:.2f} = ${cost:,.0f} "
                  f"(budget ${trade['budget_usd']:,})")
        else:
            print(f"    Quote failed: {data_q}")

    qctx.close()
    tctx.close()
    print("\n  Scan complete. Use --live with _place_moomoo_monday.py to execute.")


def execute_moomoo_live():
    """Execute the Moomoo portion of the plan LIVE."""
    try:
        from moomoo import (
            OpenQuoteContext,
            OpenSecTradeContext,
            TrdEnv,
            TrdSide,
            OrderType as MooOrderType,
            SecurityFirm,
            TrdMarket,
            Currency,
            RET_OK,
        )
    except ImportError:
        print("  ERROR: moomoo SDK not installed.")
        return

    print("\n" + "=" * 74)
    print("  *** MOOMOO LIVE EXECUTION -- $9,900 USD ***")
    print("  *** THIS WILL PLACE REAL ORDERS WITH REAL MONEY ***")
    print("=" * 74)

    # Safety confirmation
    confirm = input("\n  Type 'EXECUTE' to proceed with LIVE orders: ").strip()
    if confirm != "EXECUTE":
        print("  Aborted. No orders placed.")
        return

    qctx = OpenQuoteContext(host=HOST, port=PORT)
    tctx = OpenSecTradeContext(
        host=HOST, port=PORT,
        security_firm=SecurityFirm.FUTUCA,
        filter_trdmarket=TrdMarket.US,
    )
    ret_unlock, _ = tctx.unlock_trade(password=TRADE_PIN)
    if ret_unlock != RET_OK:
        print("  Trade unlock FAILED. Cannot execute.")
        qctx.close()
        tctx.close()
        return

    # Pre-flight: verify cash available
    ret_acc, data_acc = tctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
    if ret_acc == RET_OK:
        cash = float(data_acc["cash"].iloc[0] or 0)
        total_budget = sum(t["budget_usd"] for t in MOO_PLAN)
        print(f"\n  Account cash: ${cash:,.2f} | Total budget: ${total_budget:,}")
        if cash < total_budget * 0.8:
            print(f"  WARNING: Cash ${cash:,.2f} may be insufficient for ${total_budget:,} deployment.")
            go = input("  Continue anyway? (y/N): ").strip().lower()
            if go != "y":
                print("  Aborted.")
                qctx.close()
                tctx.close()
                return
    else:
        print(f"  WARNING: Could not verify cash balance. Proceeding with caution.")

    placed = 0
    errors = 0
    total_spent = 0.0

    for trade in MOO_PLAN:
        print(f"\n  [LIVE] {trade['name']}")

        # Find contract
        ret, data = qctx.get_option_chain(
            code=trade["underlying"],
            start=trade["expiry_start"],
            end=trade["expiry_end"],
        )
        if ret != RET_OK or data.empty:
            print(f"    SKIP -- chain lookup failed")
            errors += 1
            continue

        matches = data[
            (data["strike_price"] == trade["strike"]) &
            (data["option_type"] == trade["right"].upper())
        ]
        if matches.empty:
            print(f"    SKIP -- ${trade['strike']} {trade['right']} not found")
            errors += 1
            continue

        code = matches.iloc[0]["code"]

        # Get mid price
        ret_q, data_q = qctx.get_market_snapshot([code])
        if ret_q != RET_OK or data_q.empty:
            print(f"    SKIP -- no quote for {code}")
            errors += 1
            continue

        row = data_q.iloc[0]
        bid = float(row.get("bid_price", 0) or 0)
        ask = float(row.get("ask_price", 0) or 0)
        last = float(row.get("last_price", 0) or 0)
        mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else last

        if mid <= 0:
            print(f"    SKIP -- invalid price (mid=${mid})")
            errors += 1
            continue

        limit_price = round(mid, 2)
        cost = limit_price * 100 * trade["qty"]
        side = TrdSide.BUY

        # Per-trade confirmation with cost display
        print(f"    >>> BUY {trade['qty']}x {code} @ ${limit_price:.2f} "
              f"= ${cost:,.0f} (budget ${trade['budget_usd']:,})")
        if cost > trade["budget_usd"] * 1.15:
            print(f"    WARNING: Cost ${cost:,.0f} exceeds budget by "
                  f"{(cost / trade['budget_usd'] - 1) * 100:.0f}%")
        go = input(f"    Place this order? (y/N): ").strip().lower()
        if go != "y":
            print(f"    SKIPPED by user.")
            continue

        try:
            ret, result = tctx.place_order(
                price=limit_price,
                qty=trade["qty"],
                code=code,
                trd_side=side,
                order_type=MooOrderType.NORMAL,
                trd_env=TrdEnv.REAL,
            )
            if ret == RET_OK:
                order_id = result.iloc[0].get("order_id", "???")
                cost = limit_price * 100 * trade["qty"]
                total_spent += cost
                print(f"    ORDER [{order_id}]: BUY {trade['qty']}x {code} "
                      f"@ ${limit_price:.2f} (${cost:,.0f})")
                print(f"    Running total: ${total_spent:,.0f}")
                placed += 1
            else:
                print(f"    FAILED: {result}")
                errors += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            errors += 1

    print(f"\n  RESULT: {placed} placed | {errors} errors | ${total_spent:,.0f} total committed")
    qctx.close()
    tctx.close()


def fire_all_orders():
    """Place ALL Moomoo orders immediately -- no prompts, no confirmation.
    Use at market open (9:30 AM ET). Options exchanges reject pre-market orders.
    """
    try:
        from moomoo import (
            OpenQuoteContext,
            OpenSecTradeContext,
            TrdEnv,
            TrdSide,
            OrderType as MooOrderType,
            SecurityFirm,
            TrdMarket,
            Currency,
            RET_OK,
        )
    except ImportError:
        print("  ERROR: moomoo SDK not installed.")
        return

    print("\n" + "=" * 74)
    print("  MOOMOO FIRE ALL -- NO PROMPTS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    qctx = OpenQuoteContext(host=HOST, port=PORT)
    tctx = OpenSecTradeContext(
        host=HOST, port=PORT,
        security_firm=SecurityFirm.FUTUCA,
        filter_trdmarket=TrdMarket.US,
    )
    ret_unlock, msg = tctx.unlock_trade(password=TRADE_PIN)
    if ret_unlock != RET_OK:
        print(f"  Trade unlock FAILED: {msg}")
        qctx.close()
        tctx.close()
        return
    print("  Trade unlock: OK")

    # Check cash
    ret_acc, data_acc = tctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
    if ret_acc == RET_OK:
        cash = float(data_acc["cash"].iloc[0] or 0)
        print(f"  Cash: ${cash:,.2f}")
    else:
        cash = 0
        print("  WARNING: Could not read cash balance")

    placed = 0
    errors = 0
    total_spent = 0.0

    for trade in MOO_PLAN:
        print(f"\n  [{trade['name']}]")

        # Find contract
        ret, data = qctx.get_option_chain(
            code=trade["underlying"],
            start=trade["expiry_start"],
            end=trade["expiry_end"],
        )
        if ret != RET_OK or data.empty:
            print(f"    SKIP -- chain lookup failed")
            errors += 1
            continue

        matches = data[
            (data["strike_price"] == trade["strike"]) &
            (data["option_type"] == trade["right"].upper())
        ]
        if matches.empty:
            print(f"    SKIP -- ${trade['strike']} {trade['right']} not found")
            errors += 1
            continue

        code = matches.iloc[0]["code"]

        # Get live mid price
        ret_q, data_q = qctx.get_market_snapshot([code])
        if ret_q != RET_OK or data_q.empty:
            print(f"    SKIP -- no quote for {code}")
            errors += 1
            continue

        row = data_q.iloc[0]
        bid = float(row.get("bid_price", 0) or 0)
        ask = float(row.get("ask_price", 0) or 0)
        last = float(row.get("last_price", 0) or 0)
        mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else last

        if mid <= 0:
            print(f"    SKIP -- invalid price (mid=${mid})")
            errors += 1
            continue

        cost = mid * 100 * trade["qty"]
        print(f"    BUY {trade['qty']}x {code} @ ${mid:.2f} = ${cost:,.0f}")

        try:
            ret, result = tctx.place_order(
                price=mid,
                qty=trade["qty"],
                code=code,
                trd_side=TrdSide.BUY,
                order_type=MooOrderType.NORMAL,
                trd_env=TrdEnv.REAL,
            )
            if ret == RET_OK:
                order_id = result.iloc[0].get("order_id", "???")
                total_spent += cost
                print(f"    OK -- order [{order_id}] | running total: ${total_spent:,.0f}")
                placed += 1
            else:
                print(f"    FAILED: {result}")
                errors += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            errors += 1

    print(f"\n  RESULT: {placed} placed | {errors} errors | ${total_spent:,.0f} committed")
    qctx.close()
    tctx.close()


def main():
    parser = argparse.ArgumentParser(
        description="$9,900 Injection Plan -- WealthSimple + Moomoo"
    )
    parser.add_argument("--moomoo", action="store_true",
                        help="Scan Moomoo option chains for live prices")
    parser.add_argument("--live", action="store_true",
                        help="Execute Moomoo orders LIVE (WealthSimple is always manual)")
    parser.add_argument("--fire", action="store_true",
                        help="FIRE ALL orders immediately -- no prompts (use at market open)")
    args = parser.parse_args()

    if args.fire:
        fire_all_orders()
    elif args.live:
        show_plan()
        execute_moomoo_live()
    elif args.moomoo:
        show_plan()
        scan_moomoo_chains()
    else:
        show_plan()


if __name__ == "__main__":
    main()

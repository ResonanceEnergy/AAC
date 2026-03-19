#!/usr/bin/env python3
"""
IBKR Order Audit + $920 Maximization Plan
==========================================
Audits all open IBKR orders and prints a capital deployment plan
to maximize the remaining dry powder given the current regime.

Usage:
    python strategies/ibkr_order_audit.py
    python strategies/ibkr_order_audit.py --balance 920
    python strategies/ibkr_order_audit.py --cancel-order 56
    python strategies/ibkr_order_audit.py --execute-recs
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# TODAY'S MACRO CONTEXT (March 18 2026 — update daily or pass via env)
# ──────────────────────────────────────────────────────────────────────────────
MACRO = dict(
    vix=21.48,
    hy_spread_bps=380,
    oil=95.50,
    gold=5011.0,
    core_pce=3.1,
    gdp=0.7,
    war_active=True,
    hormuz_blocked=True,
    private_redemptions=11.0,
)

# ──────────────────────────────────────────────────────────────────────────────
# MAXIMIZATION PLAYBOOK
# Each entry: {ticker, expression, qty, est_premium, rationale, dte, otm, min_bal}
# ──────────────────────────────────────────────────────────────────────────────
PLAYBOOK = [
    {
        "ticker": "KRE",
        "expression": "Put Spread",
        "qty": 1,
        "est_premium": 2.50,
        "cost": 250.0,
        "rationale": "Banks = contagion vector for credit stress; gaps not grinds",
        "dte": "14-42 days",
        "otm": "5%",
        "min_bal": 250.0,
        "priority": 1,
    },
    {
        "ticker": "JNK",
        "expression": "Put Spread",
        "qty": 1,
        "est_premium": 1.00,
        "cost": 100.0,
        "rationale": "Credit twin to HYG — reinforces anchor; cheap theta",
        "dte": "14-42 days",
        "otm": "3%",
        "min_bal": 100.0,
        "priority": 2,
    },
    {
        "ticker": "JETS",
        "expression": "ATM Put",
        "qty": 1,
        "est_premium": 1.50,
        "cost": 150.0,
        "rationale": "Oil shock direct hit; airlines bleed on stagflation",
        "dte": "7-21 days",
        "otm": "ATM",
        "min_bal": 150.0,
        "priority": 3,
    },
    {
        "ticker": "ZIM",
        "expression": "ATM Put",
        "qty": 1,
        "est_premium": 1.80,
        "cost": 180.0,
        "rationale": "Hormuz → shipping collapse; extreme operating leverage",
        "dte": "7-21 days",
        "otm": "ATM",
        "min_bal": 180.0,
        "priority": 4,
    },
    {
        "ticker": "ARCC",
        "expression": "OTM Put",
        "qty": 1,
        "est_premium": 0.80,
        "cost": 80.0,
        "rationale": "Private credit NAV mark-to-market risk; slowest but highest P/L ratio",
        "dte": "56-90 days",
        "otm": "8%",
        "min_bal": 80.0,
        "priority": 5,
    },
]

CASH_RESERVE = 50.0  # Always keep $50 buffer


async def connect_ibkr():
    from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
    c = IBKRConnector()
    await c.connect()
    return c


def parse_order(o) -> dict:
    """Convert ExchangeOrder object to plain dict."""
    qty = float(getattr(o, 'quantity', 0))
    price = float(getattr(o, 'price', 0))
    # Options cost = qty contracts × premium × 100 multiplier
    cost = qty * price * 100
    return {
        'order_id': getattr(o, 'order_id', '?'),
        'symbol': getattr(o, 'symbol', '?').replace('/USD', ''),
        'side': getattr(o, 'side', '?'),
        'qty': qty,
        'price': price,
        'cost_usd': round(cost, 2),
        'status': getattr(o, 'status', '?'),
        'ibkr_status': getattr(o, 'raw', {}).get('ibkr_status', '?'),
        'perm_id': getattr(o, 'raw', {}).get('perm_id'),
        'created_at': getattr(o, 'created_at', None),
    }


def build_maximize_plan(remaining: float) -> List[dict]:
    """Select which playbook entries fit in remaining capital."""
    deployable = remaining - CASH_RESERVE
    plan = []
    for entry in sorted(PLAYBOOK, key=lambda x: x['priority']):
        if entry['cost'] <= deployable:
            plan.append(entry)
            deployable -= entry['cost']
    return plan


def print_audit(orders: list, balance: float) -> None:
    total_committed = sum(o['cost_usd'] for o in orders)
    remaining = balance - total_committed

    print()
    print("═" * 65)
    print("  IBKR ORDER AUDIT  —  " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    print("═" * 65)
    print(f"  Account : U24346218  (paper)")
    print(f"  Balance : ${balance:,.2f} USD")
    print(f"  Macro   : VIX={MACRO['vix']}  HY={MACRO['hy_spread_bps']}bps  Oil=${MACRO['oil']}")
    geo = []
    if MACRO['war_active']:
        geo.append("WAR ACTIVE")
    if MACRO['hormuz_blocked']:
        geo.append("HORMUZ BLOCKED")
    if geo:
        print(f"  Geo     : {' | '.join(geo)}")
    print()

    if not orders:
        print("  No open orders.")
    else:
        print(f"  {'#':>3}  {'SYMBOL':6}  {'SIDE':4}  {'QTY':4}  {'PREMIUM':8}  {'COST':7}  {'STATUS'}")
        print("  " + "─" * 58)
        for o in orders:
            print(
                f"  {o['order_id']:>3}  {o['symbol']:6}  {o['side'].upper():4}  "
                f"{o['qty']:4.0f}  ${o['price']:6.2f}   ${o['cost_usd']:6.0f}  "
                f"{o['ibkr_status']}"
            )
        print()
        print(f"  Committed : ${total_committed:,.2f}")

    print(f"  Remaining : ${remaining:,.2f}  (incl. ${CASH_RESERVE} buffer reserve)")

    # ── Maximization Plan
    plan = build_maximize_plan(remaining)
    print()
    print("─" * 65)
    print("  $920 MAXIMIZATION PLAN")
    print("─" * 65)
    if not plan:
        print(f"  Insufficient powder (${remaining:.0f}) for any new position. Hold cash.")
    else:
        plan_cost = sum(p['cost'] for p in plan)
        cash_after = remaining - plan_cost
        print(f"  Deploy ${plan_cost:.0f} across {len(plan)} new position(s).  Buffer after: ${cash_after:.0f}")
        print()
        for p in plan:
            print(f"  [{p['priority']}] {p['ticker']:6}  {p['expression']:17}  "
                  f"×{p['qty']}  ~${p['est_premium']:.2f}/contract  total≈${p['cost']:.0f}  DTE {p['dte']}")
            print(f"       OTM: {p['otm']}  |  {p['rationale']}")
        print()
        print("  EXECUTION ORDER (highest conviction first):")
        for i, p in enumerate(plan, 1):
            print(f"    {i}. {p['ticker']} {p['expression']} — ${p['cost']:.0f}")

    # ── Regime context
    print()
    print("─" * 65)
    print("  REGIME CONTEXT  (run market_forecaster_runner.py for full plan)")
    print("─" * 65)
    try:
        from strategies.regime_engine import RegimeEngine, MacroSnapshot
        from strategies.stock_forecaster import StockForecaster, Horizon

        snap = MacroSnapshot(
            vix=MACRO['vix'],
            hy_spread_bps=MACRO['hy_spread_bps'],
            oil_price=MACRO['oil'],
            gold_price=MACRO['gold'],
            core_pce=MACRO['core_pce'],
            gdp_growth=MACRO['gdp'],
            war_active=MACRO['war_active'],
            hormuz_blocked=MACRO['hormuz_blocked'],
            private_credit_redemption_pct=MACRO['private_redemptions'],
        )
        state = RegimeEngine().evaluate(snap)
        top2 = StockForecaster().two_trade_stack(state)
        print(f"  Regime  : {state.primary_regime.value.upper().replace('_',' ')}"
              f"  (conf {state.regime_confidence:.0%})")
        print(f"  Vol Shock Readiness: {state.vol_shock_readiness:.0f}/100")
        if top2[0]:
            print(f"  Anchor  : {top2[0].primary_ticker}  —  {top2[0].thesis[:60]}")
        if top2[1]:
            print(f"  Contagion: {top2[1].primary_ticker}  —  {top2[1].thesis[:60]}")
        fired = state.top_formulas
        if fired:
            print(f"  Fired   : {', '.join(f.tag.value for f in fired[:3])}")
    except Exception as e:
        print(f"  (Regime engine unavailable: {e})")

    print()
    print("═" * 65)
    print()


async def cancel_order(order_id: str, connector) -> None:
    try:
        result = await connector.cancel_order(order_id)
        print(f"  Cancelled order #{order_id}: {result}")
    except Exception as e:
        print(f"  Cancel failed for #{order_id}: {e}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR Order Audit + $920 Maximization")
    parser.add_argument("--balance", type=float, default=920.0,
                        help="Account balance in USD (default: 920)")
    parser.add_argument("--cancel-order", metavar="ORDER_ID", dest="cancel_id",
                        help="Cancel a specific open order by ID")
    parser.add_argument("--cancel-all", action="store_true",
                        help="Cancel all open PreSubmitted orders")
    parser.add_argument("--execute-recs", action="store_true",
                        help="Execute the maximization recommendations on paper account")
    args = parser.parse_args()

    try:
        connector = await connect_ibkr()
    except Exception as e:
        print(f"\n  IBKR connection failed: {e}")
        print("  Showing plan with estimated order data...\n")
        # Show offline plan with known orders from session
        orders = [
            {'order_id': '60', 'symbol': 'BKLN', 'side': 'buy', 'qty': 3, 'price': 0.49,
             'cost_usd': 147.0, 'status': 'open', 'ibkr_status': 'PreSubmitted', 'perm_id': None, 'created_at': None},
            {'order_id': '56', 'symbol': 'XLF',  'side': 'buy', 'qty': 1, 'price': 2.08,
             'cost_usd': 208.0, 'status': 'open', 'ibkr_status': 'PreSubmitted', 'perm_id': None, 'created_at': None},
            {'order_id': '58', 'symbol': 'HYG',  'side': 'buy', 'qty': 1, 'price': 1.92,
             'cost_usd': 192.0, 'status': 'open', 'ibkr_status': 'PreSubmitted', 'perm_id': None, 'created_at': None},
        ]
        print_audit(orders, args.balance)
        return

    # ── Live path ──
    raw_orders = await connector.get_open_orders()
    orders = [parse_order(o) for o in raw_orders]

    if args.cancel_id:
        await cancel_order(args.cancel_id, connector)
        # Refresh after cancel
        raw_orders = await connector.get_open_orders()
        orders = [parse_order(o) for o in raw_orders]

    if args.cancel_all:
        pre_submitted = [o for o in orders if o['ibkr_status'] == 'PreSubmitted']
        for o in pre_submitted:
            await cancel_order(o['order_id'], connector)
        raw_orders = await connector.get_open_orders()
        orders = [parse_order(o) for o in raw_orders]

    print_audit(orders, args.balance)

    if args.execute_recs:
        total_committed = sum(o['cost_usd'] for o in orders)
        remaining = args.balance - total_committed
        plan = build_maximize_plan(remaining)
        if not plan:
            print("  Nothing to execute — insufficient remaining capital.")
        else:
            print(f"  Submitting {len(plan)} recommendation orders...\n")
            for p in plan:
                try:
                    result = await connector.place_order(
                        symbol=f"{p['ticker']}/USD",
                        side="buy",
                        order_type="limit",
                        quantity=float(p['qty']),
                        price=p['est_premium'],
                    )
                    print(f"  ✓ {p['ticker']} submitted — {result}")
                except Exception as e:
                    print(f"  ✗ {p['ticker']} failed: {e}")

    await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

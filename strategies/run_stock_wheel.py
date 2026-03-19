#!/usr/bin/env python3
"""
Run Stock Wheel Strategy — Moomoo Execution Wrapper
=====================================================
Cash-secured put selling (wheel) on US large-caps via Moomoo simulate mode.

The wheel:
  1. Sell cash-secured put at target strike → collect premium
  2. If assigned, hold stock + sell covered call → collect premium
  3. If called away, restart at step 1

Usage:
    .venv/Scripts/python strategies/run_stock_wheel.py                # Show plan
    .venv/Scripts/python strategies/run_stock_wheel.py --execute      # Place orders
    .venv/Scripts/python strategies/run_stock_wheel.py --status       # Check positions

Note: Moomoo simulate mode doesn't require US market data rights for order
placement. Orders will fill at sim prices.
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_env_file
load_env_file()

logger = logging.getLogger("stock_wheel_runner")


@dataclass
class WheelTarget:
    """A stock to run the wheel strategy on."""
    symbol: str
    shares: int          # Lot size (usually 100 for options, or fractional for shares)
    target_price: float  # Approximate current price (for limit orders if quotes unavailable)
    max_allocation: float  # Max $ to commit


# Conservative large-cap wheel targets
# These are high-quality stocks we'd be happy owning if assigned
WHEEL_TARGETS = [
    WheelTarget("AAPL/USD",  10, 170.00, 50000),
    WheelTarget("MSFT/USD",  10, 390.00, 50000),
    WheelTarget("GOOGL/USD", 10, 155.00, 50000),
    WheelTarget("AMZN/USD",  10, 180.00, 50000),
    WheelTarget("JPM/USD",   10, 230.00, 50000),
    WheelTarget("V/USD",     10, 310.00, 50000),
    WheelTarget("UNH/USD",   10, 500.00, 50000),
    WheelTarget("HD/USD",    10, 370.00, 50000),
]


async def get_moomoo_connector():
    """Connect to Moomoo OpenD."""
    from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector
    connector = MoomooConnector(testnet=True)
    await connector.connect()
    logger.info("Connected to Moomoo (SIMULATE)")
    return connector


async def show_status(conn):
    """Show current account status, positions, open orders."""
    print(f"\n{'='*60}")
    print(f"  MOOMOO ACCOUNT STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Balances
    bals = await conn.get_balances()
    for asset, b in sorted(bals.items()):
        if b.free != 0 or b.locked != 0:
            print(f"  {b.asset:12s}: ${b.free:>14,.2f} free, ${b.locked:>10,.2f} locked")

    # Positions
    positions = await conn.get_positions()
    if positions:
        print(f"\n  POSITIONS ({len(positions)}):")
        for p in positions:
            pnl_str = f"P&L: ${p['unrealized_pnl']:+,.2f}" if p['unrealized_pnl'] else ""
            print(f"    {p['symbol']:10s} qty={p['quantity']:>6.0f} cost=${p['cost_price']:.2f} {pnl_str}")
    else:
        print("\n  No positions.")

    # Open orders
    orders = await conn.get_open_orders()
    if orders:
        print(f"\n  OPEN ORDERS ({len(orders)}):")
        for o in orders:
            print(f"    {o.order_id}: {o.side} {o.quantity}x {o.symbol} @ ${o.price:.2f} [{o.status}]")
    else:
        print("  No open orders.")

    print(f"{'='*60}\n")
    return bals, positions, orders


def generate_buy_plan(bals: dict, positions: list, targets: List[WheelTarget]) -> List[dict]:
    """
    Generate limit buy orders for wheel targets.
    Buy at 3-5% below target price (value entry).
    """
    cash = bals.get('USD', None)
    available = cash.free if cash else 0

    # Track existing positions
    held = {p['symbol'] for p in positions}

    orders = []
    remaining_cash = available

    for target in targets:
        if target.symbol in held:
            continue  # Already holding

        # Buy at ~3% discount
        buy_price = round(target.target_price * 0.97, 2)
        order_cost = buy_price * target.shares
        max_cost = min(target.max_allocation, remaining_cash * 0.15)  # Max 15% per position

        if order_cost > max_cost:
            target_shares = max(1, int(max_cost / buy_price))
        else:
            target_shares = target.shares

        cost = buy_price * target_shares
        if cost > remaining_cash or cost < 100:
            continue

        orders.append({
            'symbol': target.symbol,
            'side': 'buy',
            'order_type': 'limit',
            'quantity': target_shares,
            'price': buy_price,
            'cost': cost,
        })
        remaining_cash -= cost

    return orders


def print_plan(orders: List[dict], available_cash: float):
    """Print the trading plan."""
    print(f"\n{'='*60}")
    print(f"  WHEEL STRATEGY — BUY PLAN")
    print(f"  Cash available: ${available_cash:,.2f}")
    print(f"{'='*60}")

    total = 0
    for o in orders:
        print(
            f"  BUY {o['quantity']:>4}x {o['symbol']:10s} "
            f"@ ${o['price']:>8.2f} = ${o['cost']:>10,.2f}"
        )
        total += o['cost']

    print(f"  {'─'*50}")
    print(f"  Total deployment: ${total:,.2f} ({total/available_cash*100:.1f}% of cash)")
    print(f"  Cash remaining:   ${available_cash - total:,.2f}")
    print(f"{'='*60}\n")


async def execute_orders(conn, orders: List[dict]) -> List[dict]:
    """Place all orders."""
    results = []
    for o in orders:
        try:
            result = await conn.create_order(
                symbol=o['symbol'],
                side=o['side'],
                order_type=o['order_type'],
                quantity=o['quantity'],
                price=o['price'],
            )
            print(f"  OK: {o['side'].upper()} {o['quantity']}x {o['symbol']} @ ${o['price']:.2f} → order_id={result.order_id}")
            results.append({'success': True, **o, 'order_id': result.order_id})
        except Exception as e:
            print(f"  FAIL: {o['symbol']}: {e}")
            results.append({'success': False, **o, 'error': str(e)})
    return results


async def main():
    parser = argparse.ArgumentParser(description="Stock Wheel Strategy — Moomoo")
    parser.add_argument('--execute', action='store_true',
                        help='Execute buy orders on Moomoo simulate')
    parser.add_argument('--status', action='store_true',
                        help='Show account status only')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    conn = await get_moomoo_connector()

    try:
        bals, positions, open_orders = await show_status(conn)

        if args.status:
            return

        # Generate plan
        orders = generate_buy_plan(bals, positions, WHEEL_TARGETS)
        cash = bals.get('USD', None)
        available = cash.free if cash else 0
        print_plan(orders, available)

        if args.execute and orders:
            print("  EXECUTING ORDERS...\n")
            results = await execute_orders(conn, orders)
            success = sum(1 for r in results if r['success'])
            print(f"\n  {success}/{len(results)} orders placed successfully.")
        elif not args.execute:
            print("  DRY RUN — use --execute to place orders.\n")

    finally:
        await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

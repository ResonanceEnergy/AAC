#!/usr/bin/env python3
"""
Run Macro Crisis Put Strategy — Execution Wrapper
===================================================
Wires the MacroCrisisPutEngine to IBKR for live (paper) execution.

Usage:
    .venv/Scripts/python strategies/run_crisis_puts.py              # Dry-run (plan only)
    .venv/Scripts/python strategies/run_crisis_puts.py --execute     # Execute on IBKR paper
    .venv/Scripts/python strategies/run_crisis_puts.py --live        # Execute LIVE (requires --confirm)
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_env, get_env_bool
from strategies.macro_crisis_put_strategy import (
    MacroCrisisPutEngine,
    CrisisMonitor,
    PutOrderSpec,
)

logger = logging.getLogger("crisis_put_runner")


async def get_ibkr_connector():
    """Connect to IBKR via TWS API."""
    from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
    connector = IBKRConnector()
    await connector.connect()
    logger.info(f"Connected to IBKR — account {connector.account}")
    return connector


async def fetch_live_prices(symbols, connector=None):
    """
    Fetch current prices for target symbols.
    Uses IBKR if connected, otherwise falls back to hardcoded estimates.
    """
    if connector is not None:
        prices = {}
        for sym in symbols:
            try:
                ticker = await connector.get_ticker(f"{sym}/USD")
                prices[sym] = ticker.last
                logger.info(f"  {sym}: ${ticker.last:.2f}")
            except Exception as e:
                logger.warning(f"  {sym}: IBKR quote failed ({e}), using estimate")
        if prices:
            return prices

    # Fallback: approximate prices (updated periodically)
    logger.info("Using estimated prices (no live feed)")
    return {
        'SPY': 669.80,
        'QQQ': 545.00,
        'IWM': 195.00,
        'XLF': 45.50,
        'HYG': 74.50,
        'KRE': 51.00,
        'BKLN': 20.80,
        'LQD': 103.00,
    }


async def main():
    parser = argparse.ArgumentParser(description="Macro Crisis Put Strategy Runner")
    parser.add_argument('--execute', action='store_true',
                        help='Execute orders on IBKR paper account')
    parser.add_argument('--live', action='store_true',
                        help='Execute on LIVE account (requires --confirm)')
    parser.add_argument('--confirm', action='store_true',
                        help='Confirm live trading (safety gate)')
    parser.add_argument('--balance', type=float, default=8800.0,
                        help='Account balance for position sizing')
    parser.add_argument('--max-alloc', type=float, default=15.0,
                        help='Max %% of account for puts')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.live and not args.confirm:
        logger.error("LIVE trading requires --confirm flag. Aborting.")
        sys.exit(1)

    paper = not args.live

    # ── Build crisis assessment with current data ──
    engine = MacroCrisisPutEngine(
        account_balance=args.balance,
        max_portfolio_put_allocation_pct=args.max_alloc,
        paper_trading=paper,
    )

    assessment = engine.monitor.assess(
        oil_price=95.50,
        vix_level=28.0,
        gold_price=5003.50,
        core_pce=3.1,
        gdp_growth=0.7,
        credit_spread_bps=380,
        private_credit_redemption_pct=11.0,
        hormuz_blocked=True,
        war_active=True,
    )

    # ── Get prices ──
    connector = None
    if args.execute or args.live:
        try:
            connector = await get_ibkr_connector()
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            logger.info("Falling back to dry-run mode")

    symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'HYG', 'KRE', 'BKLN', 'LQD']
    prices = await fetch_live_prices(symbols, connector)

    # ── Generate orders ──
    orders = engine.generate_orders(assessment, prices)

    # ── Print battle plan ──
    plan = engine.print_battle_plan(assessment, orders)
    print(plan)

    # ── Execute if requested ──
    if connector and (args.execute or args.live):
        print("\n  EXECUTING ORDERS...")
        results = await engine.execute_via_ibkr(orders, connector)
        success = sum(1 for r in results if r['success'])
        failed = len(results) - success
        print(f"\n  Results: {success} filled, {failed} failed")
        for r in results:
            status = "OK" if r['success'] else f"FAIL: {r.get('error', 'unknown')}"
            print(
                f"    {r['symbol']} ${r['strike']}P {r['expiry']} "
                f"x{r['contracts']} — {status}"
            )
        await connector.disconnect()
    elif not args.execute and not args.live:
        print("\n  DRY RUN — no orders executed. Use --execute for paper trading.")


if __name__ == "__main__":
    asyncio.run(main())

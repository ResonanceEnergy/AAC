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

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import nest_asyncio

nest_asyncio.apply()

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_env, get_env_bool
from strategies.macro_crisis_put_strategy import (
    CrisisMonitor,
    MacroCrisisPutEngine,
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
        'SPY': 681.0,
        'QQQ': 590.0,
        'IWM': 256.0,
        'XLF': 45.50,
        'HYG': 74.50,
        'KRE': 51.00,
        'BKLN': 20.80,
        'LQD': 103.00,
        'OWL': 9.15,    # Blue Owl Capital — down 51% YTD (March 18 close)
        'OBDC': 11.45,  # Blue Owl BDC — private credit loan vehicle
    }


async def main():
    parser = argparse.ArgumentParser(description="Macro Crisis Put Strategy Runner")
    parser.add_argument('--execute', action='store_true',
                        help='Execute orders on IBKR paper account')
    parser.add_argument('--live', action='store_true',
                        help='Execute on LIVE account (requires --confirm)')
    parser.add_argument('--confirm', action='store_true',
                        help='Confirm live trading (safety gate)')
    parser.add_argument('--balance', type=float, default=None,
                        help='Account balance for position sizing (USD). Auto-detected from IBKR if not set.')
    parser.add_argument('--max-alloc', type=float, default=25.0,
                        help='Max %% of account for puts (default 25 for small accounts)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.live and not args.confirm:
        logger.error("LIVE trading requires --confirm flag. Aborting.")
        sys.exit(1)

    paper = not args.live

    # ── Connect to IBKR first so we can auto-detect balance ──
    connector = None
    if args.execute or args.live:
        try:
            connector = await get_ibkr_connector()
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            logger.info("Falling back to dry-run mode")

    # ── Auto-detect account balance from IBKR (convert CAD→USD if needed) ──
    account_balance = args.balance
    if account_balance is None:
        if connector is not None:
            try:
                balances = await connector.get_balance()
                cad_val = sum(b.free for k, b in balances.items() if 'CAD' in k or 'TOTAL' in k)
                usd_val = sum(b.free for k, b in balances.items() if 'USD' in k)
                # Use USD directly, or convert CAD at ~0.70
                account_balance = usd_val if usd_val > 0 else cad_val * 0.70
                logger.info(f"Auto-detected balance: ${account_balance:.2f} USD (raw CAD={cad_val:.2f}, USD={usd_val:.2f})")
            except Exception as e:
                logger.warning(f"Balance auto-detect failed: {e} — using $644 USD default")
                account_balance = 644.0
        else:
            account_balance = 644.0
            logger.info(f"No IBKR connection — using default balance ${account_balance:.2f} USD")

    # ── Build crisis assessment with today's live data (March 18, 2026) ──
    engine = MacroCrisisPutEngine(
        account_balance=account_balance,
        max_portfolio_put_allocation_pct=args.max_alloc,
        paper_trading=paper,
    )

    assessment = engine.monitor.assess(
        oil_price=95.50,           # Brent, US-Iran war week 3
        vix_level=21.48,           # Down 4% today — market bounce, good put entry
        gold_price=5011.60,        # Still above $5k
        core_pce=3.1,              # Stagflation persisting
        gdp_growth=0.7,            # Q4 annualized revised down
        credit_spread_bps=380,     # HY spreads still elevated
        private_credit_redemption_pct=11.0,  # Morgan Stanley PIF; Partners Group warns 5%+ defaults
        hormuz_blocked=True,       # Strait of Hormuz
        war_active=True,           # Iran conflict active
    )

    symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'HYG', 'KRE', 'BKLN', 'LQD', 'OWL', 'OBDC']
    prices = await fetch_live_prices(symbols, connector)

    # ── Generate orders ──
    orders = engine.generate_orders(assessment, prices)

    # ── Resolve expiry dates to valid IBKR option expirations ──
    if connector and orders:
        for order in orders:
            try:
                chains = await connector.get_option_chain(order.symbol)
                if chains:
                    all_expiries = sorted({e for c in chains for e in [c['expiry']] if e >= order.expiry})
                    if all_expiries:
                        old = order.expiry
                        order.expiry = all_expiries[0]
                        if old != order.expiry:
                            logger.info(f"  {order.symbol}: snapped expiry {old} -> {order.expiry}")
                    else:
                        # Fall back to nearest available
                        all_expiries = sorted({c['expiry'] for c in chains})
                        if all_expiries:
                            order.expiry = all_expiries[-1]
                            logger.warning(f"  {order.symbol}: no future expiry found, using {order.expiry}")
            except Exception as e:
                logger.warning(f"  {order.symbol}: chain lookup failed ({e}), keeping {order.expiry}")

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

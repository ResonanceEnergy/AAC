#!/usr/bin/env python3
"""
Run Crypto Arbitrage Strategy — NDAX Cross-Exchange Spread Scanner
====================================================================
Monitors NDAX (CAD pairs) vs Kraken (USD pairs), detects spread dislocations,
and executes limit orders when the edge exceeds fee + spread thresholds.

Strategy:
  NDAX trades CAD pairs.  Kraken (free tier, no key needed) provides USD reference.
  We derive the implied FX rate: NDAX_price / KRAKEN_price = implied USD/CAD.
  Compare against NDAX's own USDC/CAD mid to detect mispricing.

  If NDAX bid > implied fair ask  →  SELL on NDAX  (NDAX overpriced)
  If NDAX ask < implied fair bid  →  BUY  on NDAX  (NDAX underpriced)

Usage:
    .venv/Scripts/python strategies/run_crypto_arb.py                # Scan only
    .venv/Scripts/python strategies/run_crypto_arb.py --execute      # Execute on NDAX
    .venv/Scripts/python strategies/run_crypto_arb.py --loop 30      # Continuous every 30s
    .venv/Scripts/python strategies/run_crypto_arb.py --min-edge 25  # Min edge in bps
"""

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_env_bool, load_env_file

load_env_file()

logger = logging.getLogger("crypto_arb_runner")

# ── Pairs to monitor: NDAX CAD pair → Kraken USD pair ──
# Only high-volume, low-spread pairs are worth arbing
PAIR_MAP = {
    "XRP/CAD":  "XRP/USD",
    "BTC/CAD":  "BTC/USD",
    "ETH/CAD":  "ETH/USD",
    "SOL/CAD":  "SOL/USD",
    "ADA/CAD":  "ADA/USD",
    "DOT/CAD":  "DOT/USD",
    "LINK/CAD": "LINK/USD",
    "AVAX/CAD": "AVAX/USD",
    "LTC/CAD":  "LTC/USD",
    "SUI/CAD":  "SUI/USD",
}

# NDAX fee: 0.20% maker/taker.  Kraken: 0.16% maker / 0.26% taker.
# Round-trip cost (buy one side, sell the other) ~ 40-46 bps.
NDAX_FEE_BPS = 20
REFERENCE_FEE_BPS = 26  # Kraken taker worst-case
ROUND_TRIP_FEE_BPS = NDAX_FEE_BPS + REFERENCE_FEE_BPS
DEFAULT_MIN_EDGE_BPS = 25  # additional edge beyond fees

# Position sizing
MAX_ORDER_CAD = 500.0   # Max order size in CAD
MIN_ORDER_CAD = 20.0    # Min order size in CAD


@dataclass
class ArbSignal:
    """A detected arbitrage opportunity."""
    ndax_pair: str
    ref_pair: str
    direction: str          # "BUY_NDAX" or "SELL_NDAX"
    ndax_bid: float
    ndax_ask: float
    ref_bid_cad: float      # reference bid converted to CAD
    ref_ask_cad: float      # reference ask converted to CAD
    edge_bps: float         # net edge after fees
    fx_rate: float          # USDC/CAD mid used
    timestamp: datetime


async def get_ndax_connector():
    """Connect to NDAX mainnet."""
    from TradingExecution.exchange_connectors.ndax_connector import NDAXConnector
    connector = NDAXConnector(testnet=False)
    await connector.connect()
    logger.info("Connected to NDAX mainnet")
    return connector


async def get_reference_prices(pairs_usd: List[str]) -> Dict[str, dict]:
    """
    Fetch USD reference prices from Kraken via ccxt (public, no API key needed).
    Returns {symbol: {'bid': float, 'ask': float}}.
    """
    import ccxt.async_support as ccxt_async
    kraken = ccxt_async.kraken({'enableRateLimit': True})
    prices = {}
    try:
        await kraken.load_markets()
        for pair in pairs_usd:
            try:
                ticker = await kraken.fetch_ticker(pair)
                prices[pair] = {
                    'bid': float(ticker['bid'] or 0),
                    'ask': float(ticker['ask'] or 0),
                }
            except Exception as e:
                logger.warning(f"  Kraken {pair}: {e}")
    finally:
        await kraken.close()
    return prices


async def get_fx_rate(ndax_connector) -> float:
    """Get USD/CAD exchange rate from NDAX's USDC/CAD pair."""
    try:
        ticker = await ndax_connector.get_ticker("USDC/CAD")
        mid = (ticker.bid + ticker.ask) / 2
        if mid > 0:
            return mid
    except Exception as e:
        logger.warning(f"USDC/CAD ticker failed: {e}")
    # Fallback to approximate rate
    return 1.37


async def scan_opportunities(
    ndax_connector,
    min_edge_bps: float,
) -> List[ArbSignal]:
    """
    Scan all monitored pairs for arb opportunities.
    Returns list of ArbSignals where edge > min_edge_bps above fees.
    """
    # 1. Get FX rate
    fx_rate = await get_fx_rate(ndax_connector)
    logger.info(f"USD/CAD rate (USDC/CAD mid): {fx_rate:.4f}")

    # 2. Get NDAX tickers
    ndax_tickers = {}
    for cad_pair in PAIR_MAP:
        try:
            t = await ndax_connector.get_ticker(cad_pair)
            ndax_tickers[cad_pair] = t
        except Exception as e:
            logger.warning(f"  NDAX {cad_pair}: {e}")

    # 3. Get reference prices
    usd_pairs = list(PAIR_MAP.values())
    ref_prices = await get_reference_prices(usd_pairs)

    # 4. Compare
    signals = []
    threshold_bps = ROUND_TRIP_FEE_BPS + min_edge_bps

    for cad_pair, usd_pair in PAIR_MAP.items():
        if cad_pair not in ndax_tickers or usd_pair not in ref_prices:
            continue

        ndax = ndax_tickers[cad_pair]
        ref = ref_prices[usd_pair]

        if ndax.bid <= 0 or ndax.ask <= 0 or ref['bid'] <= 0 or ref['ask'] <= 0:
            continue

        # Convert reference to CAD
        ref_bid_cad = ref['bid'] * fx_rate
        ref_ask_cad = ref['ask'] * fx_rate

        # Fair mid in CAD
        fair_mid = (ref_bid_cad + ref_ask_cad) / 2

        # Check SELL on NDAX: NDAX bid vs reference ask (buy ref, sell NDAX)
        if fair_mid > 0:
            sell_edge_bps = (ndax.bid - ref_ask_cad) / fair_mid * 10000
            if sell_edge_bps > threshold_bps:
                signals.append(ArbSignal(
                    ndax_pair=cad_pair,
                    ref_pair=usd_pair,
                    direction="SELL_NDAX",
                    ndax_bid=ndax.bid,
                    ndax_ask=ndax.ask,
                    ref_bid_cad=ref_bid_cad,
                    ref_ask_cad=ref_ask_cad,
                    edge_bps=sell_edge_bps,
                    fx_rate=fx_rate,
                    timestamp=datetime.now(),
                ))

            # Check BUY on NDAX: reference bid vs NDAX ask (buy NDAX, sell ref)
            buy_edge_bps = (ref_bid_cad - ndax.ask) / fair_mid * 10000
            if buy_edge_bps > threshold_bps:
                signals.append(ArbSignal(
                    ndax_pair=cad_pair,
                    ref_pair=usd_pair,
                    direction="BUY_NDAX",
                    ndax_bid=ndax.bid,
                    ndax_ask=ndax.ask,
                    ref_bid_cad=ref_bid_cad,
                    ref_ask_cad=ref_ask_cad,
                    edge_bps=buy_edge_bps,
                    fx_rate=fx_rate,
                    timestamp=datetime.now(),
                ))

    # Sort by edge (best first)
    signals.sort(key=lambda s: s.edge_bps, reverse=True)
    return signals


def print_scan_report(signals: List[ArbSignal], fx_rate: float):
    """Print a human-readable scan report."""
    print(f"\n{'='*72}")
    print(f"  CRYPTO ARB SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  USD/CAD: {fx_rate:.4f}  |  Fee budget: {ROUND_TRIP_FEE_BPS} bps")
    print(f"{'='*72}")

    if not signals:
        print("  No opportunities above threshold.")
        print(f"{'='*72}\n")
        return

    for s in signals:
        arrow = "BUY " if s.direction == "BUY_NDAX" else "SELL"
        print(
            f"  {arrow} {s.ndax_pair:10s} | "
            f"edge={s.edge_bps:>6.1f}bps | "
            f"NDAX bid/ask={s.ndax_bid:.4f}/{s.ndax_ask:.4f} | "
            f"ref(CAD)={s.ref_bid_cad:.4f}/{s.ref_ask_cad:.4f}"
        )
    print(f"{'='*72}\n")


async def execute_signal(ndax_connector, signal: ArbSignal, dry_run: bool) -> dict:
    """
    Execute an arb signal on NDAX.
    Places a limit order at the current best price.
    """
    # Determine order size
    price = signal.ndax_ask if signal.direction == "BUY_NDAX" else signal.ndax_bid
    if price <= 0:
        return {'success': False, 'error': 'zero price'}

    qty_cad = min(MAX_ORDER_CAD, 500.0)
    qty = qty_cad / price

    # Get min order size from market info
    if ndax_connector._client and ndax_connector._client.markets:
        market = ndax_connector._client.markets.get(signal.ndax_pair, {})
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
        if min_amount and qty < min_amount:
            qty = min_amount

    side = "buy" if signal.direction == "BUY_NDAX" else "sell"

    if dry_run:
        logger.info(
            f"  DRY RUN: would {side} {qty:.6f} {signal.ndax_pair} "
            f"@ {price:.4f} (~${qty * price:.2f} CAD)"
        )
        return {'success': True, 'dry_run': True, 'side': side,
                'qty': qty, 'price': price, 'pair': signal.ndax_pair}

    try:
        order = await ndax_connector.create_order(
            symbol=signal.ndax_pair,
            side=side,
            order_type="limit",
            quantity=qty,
            price=price,
        )
        logger.info(
            f"  ORDER PLACED: {side} {qty:.6f} {signal.ndax_pair} "
            f"@ {price:.4f} — order_id={order.order_id} status={order.status}"
        )
        return {'success': True, 'order_id': order.order_id, 'side': side,
                'qty': qty, 'price': price, 'pair': signal.ndax_pair,
                'status': order.status}
    except Exception as e:
        logger.error(f"  ORDER FAILED: {side} {signal.ndax_pair}: {e}")
        return {'success': False, 'error': str(e), 'pair': signal.ndax_pair}


async def run_cycle(ndax_connector, args) -> List[dict]:
    """Run one scan + execute cycle."""
    signals = await scan_opportunities(ndax_connector, args.min_edge)

    # Get FX rate for display
    fx_rate = await get_fx_rate(ndax_connector)
    print_scan_report(signals, fx_rate)

    results = []
    if signals and args.execute:
        dry_run = get_env_bool('DRY_RUN', True) and not args.execute
        for signal in signals[:3]:  # Max 3 concurrent arb legs
            result = await execute_signal(ndax_connector, signal, dry_run)
            results.append(result)

    return results


async def main():
    parser = argparse.ArgumentParser(description="Crypto Arbitrage Runner — NDAX vs Kraken")
    parser.add_argument('--execute', action='store_true',
                        help='Execute orders on NDAX (otherwise scan only)')
    parser.add_argument('--loop', type=int, default=0,
                        help='Continuous scan interval in seconds (0 = single scan)')
    parser.add_argument('--min-edge', type=float, default=DEFAULT_MIN_EDGE_BPS,
                        help=f'Minimum edge in bps above fees (default {DEFAULT_MIN_EDGE_BPS})')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Connect
    ndax = await get_ndax_connector()

    # Show account status
    try:
        bals = await ndax.get_balances()
        print("\n=== NDAX ACCOUNT ===")
        for asset, b in sorted(bals.items()):
            if b.free > 0 or b.locked > 0:
                print(f"  {b.asset:6s}: {b.free:>14.6f} free, {b.locked:>14.6f} locked")
        print()
    except Exception as e:
        logger.warning(f"Balance query failed: {e}")

    try:
        if args.loop > 0:
            print(f"Starting continuous scan (every {args.loop}s, min edge {args.min_edge} bps)")
            print("Press Ctrl+C to stop.\n")
            while True:
                await run_cycle(ndax, args)
                await asyncio.sleep(args.loop)
        else:
            await run_cycle(ndax, args)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await ndax.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

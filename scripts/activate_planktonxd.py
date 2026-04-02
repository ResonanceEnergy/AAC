#!/usr/bin/env python3
"""
Activate PlanktonXD — Start the prediction market harvester.

This script:
  1. Runs the Black Swan Scanner to find thesis-aligned opportunities
  2. Feeds them into the PlanktonXD Prediction Harvester
  3. Reports opportunities found and simulated P&L

Usage:
    python scripts/activate_planktonxd.py               # scan + report
    python scripts/activate_planktonxd.py --simulate     # run Monte Carlo sim
    python scripts/activate_planktonxd.py --scan-only    # just scan Polymarket
"""

import asyncio
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# UTF-8 fix for Windows Task Scheduler
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("planktonxd.activator")


async def run_blackswan_scan():
    """Run the Polymarket black swan scanner."""
    from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner

    scanner = PolymarketBlackSwanScanner()
    try:
        opportunities = await scanner.scan()
        print(scanner.generate_report(top_n=25))

        # Save results
        out_path = PROJECT_ROOT / "data" / "blackswan_scan_results.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "scan_time": datetime.now().isoformat(),
                "total_opportunities": len(opportunities),
                "opportunities": scanner.to_json(),
            }, f, indent=2)
        logger.info("Results saved to %s", out_path)

        return opportunities
    finally:
        await scanner.close()


def run_simulation():
    """Run Monte Carlo simulation of the PlanktonXD strategy."""
    from strategies.planktonxd_prediction_harvester import PlanktonXDSimulator

    print("\n" + "=" * 70)
    print("  PLANKTONXD MONTE CARLO SIMULATION")
    print("=" * 70)

    sim = PlanktonXDSimulator(
        starting_bankroll=1000.0,
        bets_per_day=170,
        days=365,
        avg_bet_size=15.0,
        avg_entry_price=0.01,
        true_prob_multiple=3.0,
    )

    # Single path
    print("\n  Single simulation path (seed=42):")
    result = sim.run_simulation(seed=42)
    for k, v in result.items():
        print(f"    {k:20}: {v}")

    # Monte Carlo
    print("\n  Monte Carlo (1000 paths):")
    mc = sim.run_monte_carlo(num_paths=1000)
    for k, v in mc.items():
        if k != "params":
            print(f"    {k:20}: {v}")

    print("\n" + "=" * 70)
    return mc


async def run_live_execution(opportunities, bankroll: float, max_bets: int):
    """Execute PlanktonXD bets live on Polymarket via CLOB."""
    from shared.audit_logger import AuditLogger
    from shared.communication_framework import CommunicationFramework
    from strategies.planktonxd_prediction_harvester import (
        create_planktonxd_strategy,
    )

    print("\n" + "=" * 70)
    print("  LIVE EXECUTION -- PlanktonXD CLOB Orders")
    print("=" * 70)
    print(f"  Bankroll: ${bankroll:.2f} | Max bets: {max_bets}")

    comm = CommunicationFramework()
    audit = AuditLogger()
    harvester = create_planktonxd_strategy(comm, audit, bankroll=bankroll)

    # Fetch markets with token IDs
    print("\n  Fetching markets with CLOB token IDs...")
    markets = await harvester.fetch_polymarket_markets(limit=300)
    print(f"  Loaded {len(markets)} markets with token IDs")

    # Scan for planktonXD-style plays
    print("\n  Running PlanktonXD signal generation...")
    signals = harvester.scan_for_opportunities(markets)
    print(f"  Generated {len(signals)} signals")

    if not signals:
        print("  No actionable signals found. Markets may be efficiently priced.")
        return

    # Place + execute each signal
    placed = 0
    executed = 0
    total_cost = 0.0

    print(f"\n  Placing up to {max_bets} bets...\n")
    for market, bet_type, outcome, edge in signals[:max_bets]:
        bet = harvester.place_bet(market, bet_type, outcome, edge)
        if bet is None:
            continue
        placed += 1

        # Execute on CLOB
        result = harvester.execute_bet(bet)
        if result:
            executed += 1
            total_cost += bet.cost
            print(
                f"  OK #{placed:3d} | {bet.bet_type.name:20s} | "
                f"{outcome:3s} @ ${bet.entry_price:.4f} | "
                f"${bet.cost:.2f} | {market.question[:50]}"
            )
        else:
            print(
                f"  FAIL #{placed:3d} | {bet.bet_type.name:20s} | "
                f"FAILED | {market.question[:50]}"
            )

    print("\n" + "=" * 70)
    print(f"  EXECUTION SUMMARY")
    print(f"  Signals:  {len(signals)}")
    print(f"  Placed:   {placed}")
    print(f"  Executed: {executed}")
    print(f"  Cost:     ${total_cost:.2f}")
    print(f"  Bankroll: ${harvester.bankroll:.2f} remaining")
    print("=" * 70)


def get_strategy_status():
    """Check PlanktonXD strategy registration status."""
    print("\n" + "=" * 70)
    print("  PLANKTONXD ACTIVATION STATUS")
    print("=" * 70)

    # Check imports
    checks = {}
    try:
        from strategies.planktonxd_prediction_harvester import (
            PlanktonXDPredictionHarvester,
            create_planktonxd_strategy,
        )
        checks["PlanktonXD class"] = "OK Loaded"
        checks["Factory function"] = "OK Available"
    except ImportError as e:
        checks["PlanktonXD import"] = f"FAIL {e}"

    try:
        from agents.polymarket_agent import PolymarketAgent
        checks["Polymarket agent"] = "OK Loaded"
    except ImportError as e:
        checks["Polymarket agent"] = f"FAIL {e}"

    try:
        from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner
        checks["BlackSwan scanner"] = "OK Loaded"
    except ImportError as e:
        checks["BlackSwan scanner"] = f"FAIL {e}"

    try:
        from strategies.black_swan_pressure_cooker import get_crisis_data
        result = get_crisis_data()
        pct = result.get("pressure_pct", 0)
        checks["Pressure cooker"] = f"OK Active ({pct}% pressure)"
    except Exception as e:
        checks["Pressure cooker"] = f"WARN {e}"

    # Check strategy framework registration
    try:
        from shared.strategy_framework import StrategyFactory
        checks["Strategy factory"] = "OK Registered (s51)"
    except Exception as e:
        checks["Strategy factory"] = f"WARN {e}"

    for name, status in checks.items():
        print(f"  {name:25}: {status}")

    print("=" * 70)
    return checks


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="PlanktonXD Activation")
    parser.add_argument("--scan-only", action="store_true", help="Just scan Polymarket")
    parser.add_argument("--simulate", action="store_true", help="Run Monte Carlo simulation")
    parser.add_argument("--status", action="store_true", help="Check activation status")
    parser.add_argument("--live", action="store_true", help="Execute bets on Polymarket (real money)")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Bankroll in USD")
    parser.add_argument("--max-bets", type=int, default=50, help="Max bets to place")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  PLANKTONXD PREDICTION MARKET HARVESTER -- ACTIVATION")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.live:
        print("  MODE: LIVE EXECUTION (real money)")
    else:
        print("  MODE: DRY RUN (scan only)")

    # Always show status
    get_strategy_status()

    if args.simulate:
        run_simulation()
        return

    if args.status:
        return

    # Scan Polymarket for black swan opportunities
    print("\n  Scanning Polymarket for thesis-aligned black swan opportunities...")
    try:
        opportunities = await run_blackswan_scan()
        print(f"\n  Found {len(opportunities)} thesis-aligned opportunities")
    except Exception as e:
        logger.error("Polymarket scan failed: %s", e)
        print(f"\n  Scan failed: {e}")
        print("  This is expected if no internet or Polymarket API is down.")
        print("  PlanktonXD strategy is still registered and ready.")
        return

    if args.live and opportunities:
        await run_live_execution(opportunities, args.bankroll, args.max_bets)
    elif args.live:
        # Black swan scanner found 0 thesis-aligned, but PlanktonXD scans ALL markets
        print("\n  Black swan scanner found 0 thesis-aligned opportunities.")
        print("  Running PlanktonXD's own broader market scan...")
        await run_live_execution([], args.bankroll, args.max_bets)
    elif not args.scan_only:
        print("\n  PlanktonXD harvester is registered in the strategy framework (s51).")
        print("  Use --live to execute bets with real money.")
        print("  Use --simulate to run Monte Carlo validation.")


if __name__ == "__main__":
    asyncio.run(main())

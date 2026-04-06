"""
Polymarket Division Activator — Unified CLI
=============================================
AAC Polymarket Division

Activates, monitors, and reports on all three Polymarket strategies:
  1. WAR ROOM (Poly)  — Geopolitical thesis bets via Black Swan scanner
  2. PLANKTONXD       — Micro-arbitrage harvester
  3. POLYMC AGENT     — Top 100 markets + Monte Carlo + exit strategies

Usage:
    python -m strategies.polymarket_division.division_activator                # status
    python -m strategies.polymarket_division.division_activator --scan         # scan all 3
    python -m strategies.polymarket_division.division_activator --war-room     # War Room only
    python -m strategies.polymarket_division.division_activator --plankton     # PlanktonXD only
    python -m strategies.polymarket_division.division_activator --polymc       # PolyMC only
    python -m strategies.polymarket_division.division_activator --mc           # Monte Carlo sim
    python -m strategies.polymarket_division.division_activator --monitor      # start monitor
    python -m strategies.polymarket_division.division_activator --json         # JSON output
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Fix imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from strategies.polymarket_division import DIVISION_STRATEGIES, get_division_status
from strategies.polymarket_division.polymc_agent import PolyMCAgent
from strategies.polymarket_division.polymc_monitor import PolyMCMonitor
from strategies.polymarket_division.war_room_poly import WarRoomPoly

logger = logging.getLogger(__name__)

# ============================================================================
# DIVISION STATUS
# ============================================================================

DIVISION_BANNER = """
================================================================================
    POLYMARKET DIVISION -- AAC v2.9.0
    ---------------------------------------------------------------
    Strategy 1: WAR ROOM (Poly)   -- Geopolitical thesis tail-risk bets
    Strategy 2: PLANKTONXD        -- Micro-arbitrage, 170 trades/day
    Strategy 3: POLYMC AGENT      -- Top 100 markets + 100K Monte Carlo
================================================================================
"""


def show_status():
    """Show division status for all 3 strategies."""
    status = get_division_status()
    print(DIVISION_BANNER)

    for key, info in status.items():
        indicator = "[OK]" if info["status"] == "loaded" else "[!!]"
        print(f"  {indicator} {info['name']:<30} {info['class']:<35} {info['status']}")
        print(f"       {info['description']}")
        print()

    # Check dependencies
    deps = {
        "numpy": False,
        "aiohttp": False,
        "py_clob_client": False,
    }
    for pkg in deps:
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            pass

    print("  Dependencies:")
    for pkg, ok in deps.items():
        print(f"    {'[OK]' if ok else '[!!]'} {pkg}")
    print()


# ============================================================================
# SCAN ALL STRATEGIES
# ============================================================================

async def scan_all():
    """Run all three strategy scans and show combined report."""
    print(DIVISION_BANNER)

    # 1. War Room Poly — scan via Black Swan scanner
    print("  [1/3] WAR ROOM: Scanning geopolitical thesis markets...")
    try:
        from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner
        scanner = PolymarketBlackSwanScanner()
        opportunities = await scanner.scan()
        war_room = WarRoomPoly()
        war_room.scan_results(opportunities)
        print(f"         Found {len(war_room.matches)} thesis-aligned markets")
        print(war_room.generate_report())
        await scanner.close()
    except Exception as e:
        print(f"         ERROR: {e}")

    # 2. PlanktonXD — check status
    print("  [2/3] PLANKTONXD: Checking harvester status...")
    try:
        from strategies.planktonxd_prediction_harvester import PlanktonXDPredictionHarvester
        harvester = PlanktonXDPredictionHarvester.__new__(PlanktonXDPredictionHarvester)
        print(f"         PlanktonXD class loaded OK")
        print(f"         Strategy: micro-arbitrage across all categories")
    except Exception as e:
        print(f"         ERROR: {e}")

    # 3. PolyMC Agent — scan + MC
    print("\n  [3/3] POLYMC: Scanning top 100 markets + Monte Carlo...")
    agent = PolyMCAgent()
    try:
        await agent.scan_top_markets(limit=100)
        print(f"         Scanned {len(agent.scanned_markets)} markets")
        agent.simulate_top_markets()
        print(agent.generate_top100_report(top_n=10))
        print(agent.generate_portfolio_report())
        print(agent.generate_mc_report())
    except Exception as e:
        print(f"         ERROR: {e}")
    finally:
        await agent.close()


async def scan_war_room():
    """Run War Room strategy only."""
    print("\n  WAR ROOM: Scanning geopolitical thesis markets...")
    try:
        from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner
        scanner = PolymarketBlackSwanScanner()
        opportunities = await scanner.scan()
        war_room = WarRoomPoly()
        war_room.scan_results(opportunities)
        print(war_room.generate_report())
        await scanner.close()
    except Exception as e:
        print(f"  ERROR: {e}")


async def scan_polymc(json_output: bool = False):
    """Run PolyMC Agent only."""
    agent = PolyMCAgent()
    try:
        print("\n  POLYMC: Scanning top 100 markets...")
        await agent.scan_top_markets(limit=100)
        agent.simulate_top_markets()
        print(agent.generate_top100_report(top_n=20))
        print(agent.generate_portfolio_report())
        if json_output:
            print(json.dumps(agent.get_portfolio_summary(), indent=2))
    finally:
        await agent.close()


async def run_mc():
    """Run Monte Carlo simulation only."""
    agent = PolyMCAgent()
    print(agent.generate_mc_report())


async def start_monitor(interval: int = 3600, max_checks: int = 0):
    """Start the PolyMC hourly monitor."""
    monitor = PolyMCMonitor()
    await monitor.run_loop(interval_seconds=interval, max_checks=max_checks)


# ============================================================================
# CLI
# ============================================================================

async def _main():
    parser = argparse.ArgumentParser(description="Polymarket Division Activator")
    parser.add_argument("--scan", action="store_true", help="Scan all 3 strategies")
    parser.add_argument("--war-room", action="store_true", help="War Room scan only")
    parser.add_argument("--plankton", action="store_true", help="PlanktonXD status")
    parser.add_argument("--polymc", action="store_true", help="PolyMC scan + MC")
    parser.add_argument("--mc", action="store_true", help="Monte Carlo simulation only")
    parser.add_argument("--monitor", action="store_true", help="Start hourly monitor")
    parser.add_argument("--interval", type=int, default=3600, help="Monitor interval (seconds)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.scan:
        await scan_all()
    elif args.war_room:
        await scan_war_room()
    elif args.polymc:
        await scan_polymc(json_output=args.json)
    elif args.mc:
        await run_mc()
    elif args.monitor:
        await start_monitor(interval=args.interval)
    elif args.plankton:
        print("\n  PLANKTONXD STATUS:")
        try:
            from strategies.planktonxd_prediction_harvester import PlanktonXDPredictionHarvester
            print("  [OK] PlanktonXDPredictionHarvester loaded")
            print("  Strategy: micro-arbitrage across 1000s of markets")
            print("  Mode: deep OTM harvesting + spread market-making")
            print("  See: scripts/activate_planktonxd.py for full activation")
        except Exception as e:
            print(f"  [!!] Error: {e}")
    else:
        show_status()
        if args.json:
            print(json.dumps(get_division_status(), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(_main())

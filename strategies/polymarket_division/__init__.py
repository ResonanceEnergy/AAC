"""
Polymarket Division — AAC v2.9.0
=================================

Unified command center for all three Polymarket strategies:

  1. WAR ROOM (Poly)  — Geopolitical thesis-aligned crisis bets
     Maps the Black Swan Pressure Cooker thesis chain onto Polymarket
     markets. Deep OTM tail-risk bets on Iran/USD/gold/oil scenarios.
     Source: strategies/polymarket_blackswan_scanner.py

  2. PLANKTONXD       — Micro-arbitrage harvester
     Emulates planktonXD 0x4ffe strategy: 170 trades/day, deep OTM
     across all categories (sports/crypto/politics/weather).
     Source: strategies/planktonxd_prediction_harvester.py

  3. POLYMC AGENT     — Top 100 lucrative markets + Monte Carlo
     Scrapes top 100 markets by volume/liquidity, runs 100K Monte
     Carlo simulations, monitors hourly with exit strategies.
     Source: strategies/polymarket_division/polymc_agent.py

Division Structure:
  polymarket_division/
    __init__.py           — This file. Registry + unified interface.
    polymc_agent.py       — PolyMC strategy engine (scanner + MC + portfolio)
    polymc_monitor.py     — Hourly monitoring loop with exit strategies
    war_room_poly.py      — War Room adapted for Polymarket thesis bets
    division_activator.py — CLI to activate, monitor, and report on all 3
    active_scanner.py     — Unified live trading engine (scan/monitor/live)
"""

from typing import Any, Dict, List

DIVISION_STRATEGIES = {
    "war_room": {
        "name": "War Room (Polymarket)",
        "module": "strategies.polymarket_blackswan_scanner",
        "class": "PolymarketBlackSwanScanner",
        "description": "Geopolitical thesis-aligned tail-risk bets",
        "status": "active",
    },
    "planktonxd": {
        "name": "PlanktonXD Harvester",
        "module": "strategies.planktonxd_prediction_harvester",
        "class": "PlanktonXDPredictionHarvester",
        "description": "Micro-arbitrage 170 trades/day harvester",
        "status": "active",
    },
    "polymc": {
        "name": "PolyMC Agent",
        "module": "strategies.polymarket_division.polymc_agent",
        "class": "PolyMCAgent",
        "description": "Top 100 markets + 100K Monte Carlo + exit strategies",
        "status": "active",
    },
    "active_scanner": {
        "name": "Active Scanner (Unified)",
        "module": "strategies.polymarket_division.active_scanner",
        "class": "ActiveScanner",
        "description": "Unified live scanner — runs all 3 strategies with execution",
        "status": "active",
    },
}


def get_division_status() -> Dict[str, Any]:
    """Return status of all three strategies."""
    status = {}
    for key, info in DIVISION_STRATEGIES.items():
        try:
            mod = __import__(info["module"], fromlist=[info["class"]])
            cls = getattr(mod, info["class"])
            status[key] = {
                "name": info["name"],
                "status": "loaded",
                "class": info["class"],
                "description": info["description"],
            }
        except Exception as e:
            status[key] = {
                "name": info["name"],
                "status": f"error: {e}",
                "class": info["class"],
                "description": info["description"],
            }
    return status


__all__ = ["DIVISION_STRATEGIES", "get_division_status", "ActiveScanner"]

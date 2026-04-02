"""
Polymarket Division — Unified Account Tracker
===============================================
Tracks ONE Polymarket wallet across all 3 strategies:

  1. WAR ROOM POLY  — Geopolitical thesis: Iran, Hormuz, oil, Israel, etc.
  2. PLANKTONXD      — Micro-arbitrage: everything not War Room or PolyMC
  3. POLYMC AGENT    — 5 target portfolio bets (FIFA, NBA, politics)

All share a single Polymarket proxy wallet. This module queries
on-chain balance + CLOB orders + Data API positions, then classifies
each position/order to its owning strategy for per-strategy P&L.

Usage:
    python -m strategies.polymarket_division.account_tracker
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -- UTF-8 stdout fix for Windows cp1252 terminals --
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, OpenOrderParams
    from py_clob_client.constants import AMOY as _AMOY

    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# STRATEGY CLASSIFICATION RULES
# ═══════════════════════════════════════════════════════════════

# War Room keywords — geopolitical thesis chain
WAR_ROOM_KEYWORDS = [
    "iran", "hormuz", "irgc", "persian gulf", "hezbollah", "israeli",
    "israel strike", "kharg", "ceasefire", "military action", "invade",
    "pahlavi", "netanyahu", "saudi", "brics", "petrodollar", "yuan",
    "de-dollarization", "gold price", "gold above", "gold 3000", "gold 4000",
    "dollar crash", "dxy below", "reserve currency", "crude oil",
    "us forces enter iran", "us x iran", "iran conduct", "arafi",
    "trump.*iran", "military operation.*iran",
]

# PolyMC target portfolio — match by slug fragments
POLYMC_SLUGS = [
    "spain-win.*fifa", "fifa-world-cup",
    "oklahoma-city-thunder", "nba-championship",
    "jd-vance.*presidential", "vance.*2028",
    "gavin-newsom.*democratic", "newsom.*nomination",
    "jd-vance.*republican", "vance.*republican",
]

# PolyMC market name fragments (for Data API which has no slug)
POLYMC_NAME_FRAGMENTS = [
    "spain.*fifa", "fifa.*world.*cup.*2026",
    "thunder.*nba.*champion", "okc.*champion",
    "vance.*president", "vance.*2028",
    "newsom.*democrat", "newsom.*nominat",
    "vance.*republican",
]


def classify_position(market_text: str) -> str:
    """
    Classify a position/order to a strategy based on market text.
    Returns: 'war_room' | 'polymc' | 'planktonxd'
    """
    text_lower = market_text.lower()

    # Check PolyMC first (most specific — 5 known bets)
    for pattern in POLYMC_SLUGS + POLYMC_NAME_FRAGMENTS:
        if re.search(pattern, text_lower):
            return "polymc"

    # Check War Room (geopolitical keywords)
    for kw in WAR_ROOM_KEYWORDS:
        if re.search(kw, text_lower):
            return "war_room"

    # Everything else → PlanktonXD (micro-arbitrage catch-all)
    return "planktonxd"


# ═══════════════════════════════════════════════════════════════
# ACCOUNT STATE DATACLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrategyAllocation:
    """Per-strategy breakdown from the single Polymarket account."""
    name: str
    positions: int = 0
    orders: int = 0
    position_value: float = 0.0
    committed_in_orders: float = 0.0
    total_deployed: float = 0.0  # position_value + committed_in_orders
    top_positions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AccountState:
    """Full Polymarket account state with per-strategy allocation."""
    timestamp: str = ""
    proxy_address: str = ""
    eoa_address: str = ""
    # Wallet
    wallet_balance: float = 0.0
    # Orders
    total_orders: int = 0
    total_committed: float = 0.0
    # Positions
    total_positions: int = 0
    total_position_value: float = 0.0
    # Grand total
    total_account_value: float = 0.0  # wallet + positions + orders
    # Per-strategy breakdown
    war_room: StrategyAllocation = field(default_factory=lambda: StrategyAllocation("War Room Poly"))
    planktonxd: StrategyAllocation = field(default_factory=lambda: StrategyAllocation("PlanktonXD"))
    polymc: StrategyAllocation = field(default_factory=lambda: StrategyAllocation("PolyMC Agent"))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for Matrix Monitor."""
        def _alloc(a: StrategyAllocation) -> Dict[str, Any]:
            return {
                "name": a.name,
                "positions": a.positions,
                "orders": a.orders,
                "position_value": round(a.position_value, 2),
                "committed_in_orders": round(a.committed_in_orders, 2),
                "total_deployed": round(a.total_deployed, 2),
                "top_positions": a.top_positions[:5],
            }

        return {
            "status": "ok",
            "timestamp": self.timestamp,
            "proxy_address": self.proxy_address,
            "wallet_balance": round(self.wallet_balance, 2),
            "total_orders": self.total_orders,
            "total_committed": round(self.total_committed, 2),
            "total_positions": self.total_positions,
            "total_position_value": round(self.total_position_value, 2),
            "total_account_value": round(self.total_account_value, 2),
            "strategies": {
                "war_room": _alloc(self.war_room),
                "planktonxd": _alloc(self.planktonxd),
                "polymc": _alloc(self.polymc),
            },
        }


# ═══════════════════════════════════════════════════════════════
# ACCOUNT TRACKER
# ═══════════════════════════════════════════════════════════════

class PolymarketAccountTracker:
    """
    Queries ONE Polymarket account and splits balances across 3 strategies.
    """

    USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    RPC_URL = "https://polygon-rpc.com"
    DATA_API = "https://data-api.polymarket.com"

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()

        self.proxy = os.getenv(
            "POLYMARKET_FUNDER_ADDRESS",
            "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8",
        )
        self.eoa = "0x6e9b70D1175ecA144111743503441300A9494297"
        self._clob_client: Optional[Any] = None

    def _get_clob_client(self) -> Any:
        """Lazy-init CLOB client."""
        if self._clob_client:
            return self._clob_client
        if not CLOB_AVAILABLE:
            return None

        creds = ApiCreds(
            api_key=os.getenv("POLYMARKET_API_KEY", ""),
            api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
        )
        self._clob_client = ClobClient(
            "https://clob.polymarket.com",
            key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            chain_id=137,
            signature_type=1,
            funder=self.proxy,
            creds=creds,
        )
        return self._clob_client

    # ── On-chain balance ──────────────────────────────────────────

    def get_wallet_balance(self) -> float:
        """Query USDC.e balance for the proxy wallet on Polygon."""
        if not requests:
            return 0.0
        try:
            padded = self.proxy[2:].lower().zfill(64)
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [
                    {"to": self.USDC_E, "data": f"0x70a08231{padded}"},
                    "latest",
                ],
                "id": 1,
            }
            r = requests.post(self.RPC_URL, json=payload, timeout=10)
            result = r.json().get("result", "0x0")
            return int(result, 16) / 1e6
        except Exception as e:
            logger.warning("Wallet balance query failed: %s", e)
            return 0.0

    # ── CLOB open orders ─────────────────────────────────────────

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders from CLOB API."""
        client = self._get_clob_client()
        if not client:
            return []
        try:
            orders = client.get_orders(OpenOrderParams())
            if isinstance(orders, list):
                return [
                    o for o in orders
                    if o.get("status", "").upper() in ("LIVE", "MATCHED", "OPEN")
                ]
            return []
        except Exception as e:
            logger.warning("CLOB orders query failed: %s", e)
            return []

    # ── Data API positions ───────────────────────────────────────

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Polymarket Data API."""
        if not requests:
            return []
        try:
            r = requests.get(
                f"{self.DATA_API}/positions?user={self.proxy}",
                timeout=15,
            )
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Positions query failed: %s", e)
            return []

    # ── Full account snapshot ────────────────────────────────────

    def get_account_state(self) -> AccountState:
        """
        Query wallet + orders + positions, classify each to a strategy,
        return the full AccountState with per-strategy breakdowns.
        """
        state = AccountState(
            timestamp=datetime.now().isoformat(),
            proxy_address=self.proxy,
            eoa_address=self.eoa,
        )

        # 1. Wallet balance
        state.wallet_balance = self.get_wallet_balance()

        # 2. Open orders → classify per strategy
        orders = self.get_open_orders()
        state.total_orders = len(orders)
        for o in orders:
            asset_id = o.get("asset_id", o.get("token_id", ""))
            market = o.get("market", "") or asset_id
            price = float(o.get("price", 0))
            size = float(o.get("original_size", o.get("size", 0)))
            cost = price * size

            state.total_committed += cost
            strategy = classify_position(market)
            alloc = getattr(state, strategy)
            alloc.orders += 1
            alloc.committed_in_orders += cost

        # 3. Positions → classify per strategy
        positions = self.get_positions()
        state.total_positions = len(positions)
        for p in positions:
            title = p.get("title", p.get("market", "")) or p.get("asset", "")
            sz = float(p.get("size", p.get("amount", 0)))
            avg = float(p.get("avgPrice", p.get("avg_price", 0)))
            cur = float(p.get("curPrice", p.get("price", 0)))
            val = sz * cur if cur > 0 else sz * avg
            side = p.get("side", p.get("outcome", "?"))

            state.total_position_value += val
            strategy = classify_position(title)
            alloc = getattr(state, strategy)
            alloc.positions += 1
            alloc.position_value += val
            alloc.top_positions.append({
                "title": title[:55],
                "side": side,
                "size": round(sz, 1),
                "avg_price": round(avg, 4),
                "current_value": round(val, 2),
            })

        # Sort top positions by value descending
        for alloc_name in ("war_room", "planktonxd", "polymc"):
            alloc = getattr(state, alloc_name)
            alloc.top_positions.sort(key=lambda x: x["current_value"], reverse=True)
            alloc.total_deployed = alloc.position_value + alloc.committed_in_orders

        # Grand total
        state.total_account_value = (
            state.wallet_balance + state.total_position_value + state.total_committed
        )

        return state

    # ── Report ───────────────────────────────────────────────────

    def generate_report(self, state: Optional[AccountState] = None) -> str:
        """Generate a full account report with per-strategy breakdown."""
        if state is None:
            state = self.get_account_state()

        lines = []
        lines.append("")
        lines.append("=" * 100)
        lines.append("  POLYMARKET DIVISION — UNIFIED ACCOUNT TRACKER")
        lines.append(f"  Proxy: {state.proxy_address}")
        lines.append(f"  Timestamp: {state.timestamp}")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"  WALLET BALANCE:     ${state.wallet_balance:>12,.2f} USDC.e")
        lines.append(f"  POSITION VALUE:     ${state.total_position_value:>12,.2f} ({state.total_positions} positions)")
        lines.append(f"  COMMITTED ORDERS:   ${state.total_committed:>12,.2f} ({state.total_orders} orders)")
        lines.append(f"  ─────────────────────────────────")
        lines.append(f"  TOTAL ACCOUNT:      ${state.total_account_value:>12,.2f}")
        lines.append("")

        # Per-strategy breakdown
        lines.append("  ╔═══════════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╗")
        lines.append("  ║ Strategy              ║ Pos      ║ Orders   ║ Pos Val  ║ Ord $    ║ Total    ║")
        lines.append("  ╠═══════════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╣")

        for alloc_name in ("war_room", "planktonxd", "polymc"):
            a = getattr(state, alloc_name)
            lines.append(
                f"  ║ {a.name:<21} ║ {a.positions:>8} ║ {a.orders:>8} ║ "
                f"${a.position_value:>7,.2f} ║ ${a.committed_in_orders:>7,.2f} ║ ${a.total_deployed:>7,.2f} ║"
            )

        lines.append("  ╚═══════════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╝")
        lines.append("")

        # Top positions per strategy
        for alloc_name in ("war_room", "planktonxd", "polymc"):
            a = getattr(state, alloc_name)
            icon = {"war_room": "⚔️", "planktonxd": "🐙", "polymc": "🎲"}.get(alloc_name, "")
            if a.positions > 0:
                lines.append(f"  {icon}  {a.name} — Top Positions:")
                for tp in a.top_positions[:5]:
                    lines.append(
                        f"      ${tp['current_value']:>8,.2f}  {tp['side']:>3}  "
                        f"x{tp['size']:>7,.1f} @ {tp['avg_price']:.4f}  {tp['title']}"
                    )
                lines.append("")

        lines.append("=" * 100)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    tracker = PolymarketAccountTracker()
    state = tracker.get_account_state()
    print(tracker.generate_report(state))

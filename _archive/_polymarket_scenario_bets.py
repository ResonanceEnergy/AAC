"""
Polymarket Scenario Betting System — All 43 AAC Crisis Scenarios
================================================================

Searches Polymarket for prediction markets aligned with EVERY crisis
scenario in the Storm Lifeboat ScenarioEngine, evaluates crowd price
vs thesis probability, sizes bets via half-Kelly, and optionally
places live orders.

Uses the CLOB API directly (bulk market fetch + local keyword matching)
because the Gamma API text search doesn't work reliably.

Usage:
    cd C:\dev\AAC_fresh
    .venv\Scripts\python.exe _polymarket_scenario_bets.py [--live] [--bankroll 500] [--max-per-bet 50]
"""

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Setup path & logging ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("poly_scenario_bets")

# ── Imports ──────────────────────────────────────────────────────────────
import aiohttp  # noqa: E402
from strategies.storm_lifeboat.core import SCENARIOS, SCENARIO_MAP  # noqa: E402

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"


# ═════════════════════════════════════════════════════════════════════════
# SCENARIO → POLYMARKET KEYWORD MAPPING
# ═════════════════════════════════════════════════════════════════════════

# Each scenario gets 3-6 search keywords optimised for Polymarket text search.
# These are tuned for the kind of questions that actually appear on Polymarket
# (e.g. "Will oil price exceed $X", "Will Iran…", "Will NATO…").

SCENARIO_KEYWORDS: Dict[str, List[str]] = {
    # ── Global Crisis Tier (1-15) ────────────────────────────────────
    "HORMUZ":            ["hormuz", "iran oil", "persian gulf", "oil blockade", "strait of hormuz"],
    "DEBT_CRISIS":       ["us debt", "debt ceiling", "treasury default", "government shutdown", "us credit downgrade"],
    "TAIWAN":            ["taiwan", "china taiwan", "tsmc", "china invade"],
    "EU_BANKS":          ["european bank", "deutsche bank", "ecb emergency", "banking crisis europe"],
    "DEFI_CASCADE":      ["defi", "stablecoin", "crypto crash", "bitcoin below", "tether depeg"],
    "SUPERCYCLE":        ["gold above", "silver above", "commodity", "gold 5000", "gold 4000"],
    "CRE_COLLAPSE":      ["commercial real estate", "office reit", "cmbs", "bank failure"],
    "AI_BUBBLE":         ["nvidia", "ai bubble", "tech crash", "qqq", "ai stock"],
    "EM_FX_CRISIS":      ["emerging market", "currency crisis", "dollar index", "imf bailout"],
    "FOOD_CRISIS":       ["food crisis", "wheat price", "famine", "food shortage", "rice price"],
    "CLIMATE_SHOCK":     ["climate disaster", "extreme weather", "hurricane landfall", "wildfire", "insurance crisis", "category 5"],
    "MONETARY_RESET":    ["brics currency", "de-dollarization", "gold standard", "dollar reserve", "mbridge"],
    "JAPAN_CRISIS":      ["japan crisis", "japanese yen", "boj rate", "carry trade unwind", "japan bond"],
    "ELECTION_CHAOS":    ["contested election", "election dispute", "election chaos", "trump impeach", "25th amendment"],
    "PANDEMIC_V2":       ["pandemic", "bird flu", "who emergency", "h5n1", "new variant"],

    # ── Middle East & US Withdrawal Tier (16-20) ─────────────────────
    "US_WITHDRAWAL":     ["us troops", "middle east withdrawal", "us military", "troop withdrawal"],
    "IRAN_DEAL":         ["iran deal", "iran nuclear deal", "iran sanctions", "jcpoa"],
    "PETRODOLLAR_SPIRAL":["petrodollar", "saudi oil yuan", "brics", "saudi china", "oil yuan"],
    "IRAN_NUCLEAR":      ["iran nuclear", "iran bomb", "iran enrichment", "iran weapon"],
    "ELITE_EXPOSURE":    ["epstein", "corruption scandal", "financial scandal", "doj investigation"],

    # ── US Western Hemisphere Pivot Tier (21-43) ─────────────────────
    "HEMISPHERE_PIVOT":  ["monroe doctrine", "western hemisphere", "us latin america"],
    "NATO_EXIT":         ["leave nato", "trump nato", "nato withdrawal", "nato article", "nato troops", "nato country"],
    "EUROPE_ABANDON":    ["european defense", "us europe", "transatlantic", "eu army"],
    "CANADA_DECLINE":    ["canada tariff", "us canada", "canadian dollar", "usmca"],
    "GREENLAND_ACQ":     ["greenland", "greenland purchase", "greenland deal", "denmark greenland"],
    "PANAMA_RECLAIM":    ["panama canal", "panama", "panama sovereignty"],
    "LATAM_LOCKIN":      ["latin america trade", "latam trade", "lithium deal", "latin american resource"],
    "ARCTIC_EXPAND":     ["arctic", "icebreaker", "arctic sovereignty", "arctic military"],
    "BORDER_MILITARY":   ["border military", "mexico border", "border wall", "border troops", "border crisis"],
    "VENEZUELA_REGIME":  ["venezuela", "maduro", "pdvsa", "venezuela oil"],
    "LITHIUM_TRIANGLE":  ["lithium", "battery supply", "critical mineral", "lithium mine"],
    "CUBA_EMBARGO":      ["cuba", "cuba embargo", "cuba sanctions"],
    "BRAZIL_ARGENTINA":  ["brazil", "argentina", "mercosur", "brazil trade"],
    "NORTH_BORDER":      ["canada pipeline", "keystone", "canada energy", "us canada border"],
    "MIDEAST_REDEPLOY":  ["centcom", "middle east base", "military redeployment"],
    "ENERGY_HEMISPHERE": ["energy independence", "oil production", "hemisphere energy", "us oil"],
    "STARLINK_DOMINANCE":["starlink", "spacex", "satellite internet", "spacex contract"],
    "FUSION_ROLLOUT":    ["fusion energy", "nuclear reactor", "smr", "small modular reactor"],
    "RARE_EARTH_FORTRESS":["rare earth", "critical minerals", "rare earth mine", "china rare earth"],
    "MIGRATION_SECURITY":["immigration reform", "border security", "immigration policy", "deportation"],
    "FORTRESS_2100":     ["deglobalization", "reshoring", "autarky", "protectionism", "tariff"],
    "ELITE_CAPITAL":     ["capital flight", "institutional rebalancing", "wealth fund", "sovereign fund"],
    "NUCLEAR_AMERICAS":  ["nuclear umbrella", "nuclear deterrence", "nuclear posture", "nuclear arms"],
}


# ═════════════════════════════════════════════════════════════════════════
# THESIS MULTIPLIERS per scenario code
# ═════════════════════════════════════════════════════════════════════════
# How much more likely WE think this is than the crowd.
# Derived from severity × conviction + contagion effects.

_BASE_MULTIPLIERS: Dict[str, float] = {
    "HORMUZ": 5.0,
    "DEBT_CRISIS": 3.5,
    "TAIWAN": 3.0,
    "EU_BANKS": 3.0,
    "DEFI_CASCADE": 2.5,
    "SUPERCYCLE": 4.0,
    "CRE_COLLAPSE": 3.0,
    "AI_BUBBLE": 2.5,
    "EM_FX_CRISIS": 3.0,
    "FOOD_CRISIS": 3.0,
    "CLIMATE_SHOCK": 2.5,
    "MONETARY_RESET": 5.0,
    "JAPAN_CRISIS": 3.5,
    "ELECTION_CHAOS": 2.0,
    "PANDEMIC_V2": 3.0,
    "US_WITHDRAWAL": 4.0,
    "IRAN_DEAL": 3.0,
    "PETRODOLLAR_SPIRAL": 5.0,
    "IRAN_NUCLEAR": 4.0,
    "ELITE_EXPOSURE": 2.5,
    "HEMISPHERE_PIVOT": 3.5,
    "NATO_EXIT": 3.5,
    "EUROPE_ABANDON": 3.0,
    "CANADA_DECLINE": 2.5,
    "GREENLAND_ACQ": 3.0,
    "PANAMA_RECLAIM": 2.5,
    "LATAM_LOCKIN": 2.5,
    "ARCTIC_EXPAND": 2.0,
    "BORDER_MILITARY": 2.0,
    "VENEZUELA_REGIME": 2.5,
    "LITHIUM_TRIANGLE": 2.5,
    "CUBA_EMBARGO": 2.0,
    "BRAZIL_ARGENTINA": 2.0,
    "NORTH_BORDER": 2.0,
    "MIDEAST_REDEPLOY": 3.0,
    "ENERGY_HEMISPHERE": 2.5,
    "STARLINK_DOMINANCE": 2.0,
    "FUSION_ROLLOUT": 2.0,
    "RARE_EARTH_FORTRESS": 3.0,
    "MIGRATION_SECURITY": 2.0,
    "FORTRESS_2100": 3.5,
    "ELITE_CAPITAL": 2.0,
    "NUCLEAR_AMERICAS": 2.5,
}

# Maximum price we consider "cheap tail-risk" — anything above this the crowd
# already thinks it's likely, so our thesis edge is weaker.
MAX_PRICE_THRESHOLD = 0.25  # 25 cents = 25%

# Minimum edge multiple (thesis_prob / market_price) to consider a bet
MIN_EDGE_MULTIPLE = 1.5


# ═════════════════════════════════════════════════════════════════════════
# DATA CLASS — one bet recommendation
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioBet:
    """A recommended Polymarket bet tied to a specific AAC scenario."""
    scenario_code: str
    scenario_name: str
    severity: float
    base_probability: float           # Our scenario engine's base prob
    market_question: str
    condition_id: str
    outcome: str                      # YES or NO
    token_id: str                     # For order placement
    market_price: float               # Crowd-implied probability (0-1)
    thesis_probability: float         # Our adjusted probability (0-1)
    edge: float                       # thesis_prob - market_price
    edge_multiple: float              # thesis_prob / market_price
    kelly_fraction: float             # Half-Kelly recommended fraction
    recommended_bet: float = 0.0      # Dollar amount
    volume: float = 0.0
    liquidity: float = 0.0

    @property
    def payout_multiple(self) -> float:
        """Dollars paid per dollar bet if outcome hits."""
        return 1.0 / max(self.market_price, 0.001)


# ═════════════════════════════════════════════════════════════════════════
# SCENARIO BET ENGINE — CLOB API bulk-fetch architecture
# ═════════════════════════════════════════════════════════════════════════

class ScenarioBetEngine:
    """
    Scans Polymarket across all 43 AAC crisis scenarios using the CLOB API,
    evaluates thesis-aligned edges, and produces a sized bet sheet.

    Architecture: bulk-fetches all active markets from CLOB, matches locally
    via keywords, then prices matched markets via CLOB /midpoint endpoint.
    """

    def __init__(self, bankroll: float = 500.0, max_per_bet: float = 50.0):
        self.bankroll = bankroll
        self.max_per_bet = max_per_bet
        self.bets: List[ScenarioBet] = []
        self._seen_conditions: set = set()
        self._all_markets: List[dict] = []

    async def close(self):
        pass  # No persistent connections needed

    # ── Thesis probability ────────────────────────────────────────────

    def _thesis_prob(self, market_price: float, scenario_code: str) -> float:
        """Our thesis-adjusted probability for a cheap outcome."""
        mult = _BASE_MULTIPLIERS.get(scenario_code, 2.0)
        estimated = market_price * mult
        # Cap at 0.60 — humility on tail events
        return min(estimated, 0.60)

    def _half_kelly(self, market_price: float, thesis_prob: float) -> float:
        """Half-Kelly fraction for bet sizing."""
        if market_price <= 0 or market_price >= 1:
            return 0.0
        b = (1.0 / market_price) - 1.0   # Payout odds
        p = thesis_prob
        q = 1.0 - p
        f = (b * p - q) / b
        return max(f / 2.0, 0.0)  # Half-Kelly for safety

    # ── CLOB API helpers ──────────────────────────────────────────────

    async def _fetch_all_markets(self) -> List[dict]:
        """Fetch all active non-closed markets from Gamma API (paginated).

        Gamma /markets with closed=false returns markets with outcomePrices
        already embedded, so no separate pricing calls are needed.
        """
        markets: List[dict] = []
        offset = 0
        max_pages = 50  # Cap at ~10,000 markets

        async with aiohttp.ClientSession() as session:
            for page in range(max_pages):
                params = {
                    "closed": "false",
                    "active": "true",
                    "limit": "200",
                    "offset": str(offset),
                }
                try:
                    async with session.get(
                        f"{GAMMA_API}/markets", params=params,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("Gamma /markets returned %d", resp.status)
                            break
                        batch = await resp.json()
                except Exception as e:
                    logger.warning("Gamma fetch error: %s", e)
                    break

                if not batch:
                    break

                markets.extend(batch)
                offset += 200

                if page % 10 == 9:
                    logger.info("  ... fetched %d markets so far", len(markets))

                if len(batch) < 200:
                    break

                await asyncio.sleep(0.05)  # Rate-limit courtesy

        return markets

    # Sports/noise terms that cause false positives when matching crisis keywords
    _NOISE_TERMS = frozenset([
        "nba:", "nfl:", "nhl:", "mlb:", "ncaab:", "ncaaf:", "mls:",
        "ufc:", "f1:", "formula 1:", "wwe:", "afl:", "epl:",
        "premier league", "serie a", "la liga", "bundesliga", "ligue 1",
        "fifa world cup", "champions league", "europa league",
        "stanley cup", "super bowl", "copa america",
        "carolina hurricanes", "miami hurricane",
        "win group", "win the 2026 fifa",
        "oscar", "emmy", "grammy", "golden globe", "box office",
        "reality tv", "bachelor", "love island",
    ])

    @staticmethod
    def _match(question: str, keywords: List[str]) -> bool:
        """Check if a market question matches any of the scenario keywords."""
        q_lower = question.lower()
        # Skip obvious sports/noise markets
        if any(noise in q_lower for noise in ScenarioBetEngine._NOISE_TERMS):
            return False
        return any(kw.lower() in q_lower for kw in keywords)

    # ── Evaluate all markets × all scenarios ──────────────────────────

    @staticmethod
    def _parse_prices(mkt: dict) -> List[Tuple[str, float, str]]:
        """Parse Gamma market into (outcome, price, token_id) tuples."""
        results: List[Tuple[str, float, str]] = []
        try:
            raw_prices = mkt.get("outcomePrices", "")
            if isinstance(raw_prices, str):
                prices = json.loads(raw_prices)  # e.g. ["0.11", "0.89"]
            else:
                prices = raw_prices
            outcomes = mkt.get("outcomes", ["Yes", "No"])
            token_ids_raw = mkt.get("clobTokenIds", "")
            if isinstance(token_ids_raw, str):
                token_ids = json.loads(token_ids_raw)
            else:
                token_ids = token_ids_raw

            for idx, (outcome, price_str) in enumerate(zip(outcomes, prices)):
                price = float(price_str)
                token_id = token_ids[idx] if idx < len(token_ids) else ""
                results.append((outcome, price, token_id))
        except (json.JSONDecodeError, ValueError, IndexError, TypeError):
            pass
        return results

    async def _evaluate_all(self) -> List[ScenarioBet]:
        """Match fetched markets against all 43 scenarios and evaluate edge.

        Prices come directly from Gamma outcomePrices — no midpoint API needed.
        """
        bets: List[ScenarioBet] = []
        scenario_by_code = {s.code: s for s in SCENARIOS}

        # Keyword match (fast, local — no API calls)
        matches: List[Tuple[dict, str]] = []  # (market, scenario_code)
        for mkt in self._all_markets:
            question = mkt.get("question", "")
            for code, keywords in SCENARIO_KEYWORDS.items():
                if self._match(question, keywords):
                    dedup_key = (mkt.get("conditionId", mkt.get("condition_id")), code)
                    if dedup_key not in self._seen_conditions:
                        self._seen_conditions.add(dedup_key)
                        matches.append((mkt, code))

        unique_scenarios = len({m[1] for m in matches})
        logger.info(
            "Keyword matching: %d market-scenario pairs across %d scenarios",
            len(matches), unique_scenarios,
        )

        if not matches:
            logger.info("No keyword matches found across all 43 scenarios")
            return bets

        # Evaluate matched markets using embedded prices
        for mkt, code in matches:
            scenario = scenario_by_code.get(code)
            if not scenario:
                continue

            question = mkt.get("question", "")
            condition_id = mkt.get("conditionId", mkt.get("condition_id", ""))
            parsed = self._parse_prices(mkt)

            for outcome, price, token_id in parsed:
                if price <= 0 or price > MAX_PRICE_THRESHOLD:
                    continue

                thesis_p = self._thesis_prob(price, code)
                edge = thesis_p - price
                edge_mult = thesis_p / max(price, 0.001)

                if edge <= 0 or edge_mult < MIN_EDGE_MULTIPLE:
                    continue

                kelly = self._half_kelly(price, thesis_p)

                bet = ScenarioBet(
                    scenario_code=code,
                    scenario_name=scenario.name,
                    severity=scenario.impact_severity,
                    base_probability=scenario.probability,
                    market_question=question,
                    condition_id=condition_id,
                    outcome=outcome,
                    token_id=token_id,
                    market_price=price,
                    thesis_probability=thesis_p,
                    edge=edge,
                    edge_multiple=edge_mult,
                    kelly_fraction=kelly,
                )
                bets.append(bet)
                logger.info(
                    "  BET: [%s] %s — %s @ %.1f%% (thesis %.1f%%, edge %.1f%%)",
                    code, question[:50], outcome,
                    price * 100, thesis_p * 100, edge * 100,
                )

        logger.info(
            "Total opportunities: %d bets from %d matches",
            len(bets), len(matches),
        )
        return bets

    # ── Full scan ─────────────────────────────────────────────────────

    async def scan_all_scenarios(self) -> List[ScenarioBet]:
        """Scan Polymarket for opportunities across all 43 scenarios."""
        logger.info("=" * 70)
        logger.info("POLYMARKET SCENARIO BET SCANNER — All 43 Scenarios")
        logger.info("Bankroll: $%.2f  |  Max per bet: $%.2f",
                     self.bankroll, self.max_per_bet)
        logger.info("=" * 70)

        self.bets = []
        self._seen_conditions = set()

        # Step 1: Bulk-fetch all active markets from CLOB
        logger.info("Fetching all active markets from CLOB API...")
        self._all_markets = await self._fetch_all_markets()
        logger.info("Fetched %d active markets total", len(self._all_markets))

        if not self._all_markets:
            logger.warning("No markets fetched — CLOB API may be down")
            return self.bets

        # Step 2: Match markets to scenarios + evaluate edge + fetch prices
        self.bets = await self._evaluate_all()

        # Sort by edge (best first)
        self.bets.sort(key=lambda b: b.edge, reverse=True)

        # Size the bets
        self._size_bets()

        return self.bets

    # ── Bet sizing ────────────────────────────────────────────────────

    def _size_bets(self):
        """Apply half-Kelly sizing across all bets within bankroll."""
        if not self.bets:
            return

        total_kelly = sum(b.kelly_fraction for b in self.bets)
        if total_kelly <= 0:
            return

        for bet in self.bets:
            # Proportional Kelly: each bet's fraction of total Kelly allocation
            if total_kelly > 0:
                raw_size = self.bankroll * (bet.kelly_fraction / total_kelly)
            else:
                raw_size = 0

            # Cap individual bet
            bet.recommended_bet = min(raw_size, self.max_per_bet)

            # Floor at $1 minimum if any allocation at all
            if 0 < bet.recommended_bet < 1.0:
                bet.recommended_bet = 0.0  # Too small, skip

    # ── Display ───────────────────────────────────────────────────────

    def print_bet_sheet(self):
        """Pretty-print the full bet recommendation sheet."""
        active_bets = [b for b in self.bets if b.recommended_bet >= 1.0]
        skipped_bets = [b for b in self.bets if b.recommended_bet < 1.0]

        print("\n" + "=" * 90)
        print("  POLYMARKET SCENARIO BET SHEET — %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("  Bankroll: $%.2f  |  Bets: %d active  |  %d below minimum"
              % (self.bankroll, len(active_bets), len(skipped_bets)))
        print("=" * 90)

        if not active_bets:
            print("\n  No viable bets found across all 43 scenarios.")
            print("  (Markets may be offline or all priced above %.0f%% threshold)"
                  % (MAX_PRICE_THRESHOLD * 100))
            print("=" * 90)
            return

        total_deployed = 0.0
        total_potential = 0.0

        # Group by scenario
        from collections import defaultdict
        by_scenario: Dict[str, List[ScenarioBet]] = defaultdict(list)
        for b in active_bets:
            by_scenario[b.scenario_code].append(b)

        for code in by_scenario:
            scenario_bets = by_scenario[code]
            s = scenario_bets[0]
            print(f"\n  ┌─ {s.scenario_code}: {s.scenario_name}")
            print(f"  │  Severity: {s.severity:.2f}  Base Prob: {s.base_probability:.0%}")
            print(f"  │  {'Question':<42} {'Side':>4} {'Price':>6} {'Thesis':>7} {'Edge':>6} {'Kelly':>6} {'Bet $':>7} {'Payout':>7}")
            print(f"  │  {'─'*42} {'─'*4} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*7}")

            for b in scenario_bets:
                q = b.market_question[:42]
                potential = b.recommended_bet * b.payout_multiple
                total_deployed += b.recommended_bet
                total_potential += potential
                print(f"  │  {q:<42} {b.outcome:>4} {b.market_price:>5.1%} {b.thesis_probability:>6.1%}"
                      f"  {b.edge:>5.1%} {b.kelly_fraction:>5.1%} ${b.recommended_bet:>6.2f} ${potential:>6.0f}")

            print(f"  └{'─' * 88}")

        print(f"\n  SUMMARY")
        print(f"  Total deployed:  ${total_deployed:>10.2f}")
        print(f"  Max potential:   ${total_potential:>10.0f}")
        print(f"  Avg payout mult: {total_potential / max(total_deployed, 1):>10.1f}x")
        print(f"  Scenarios w/bets: {len(by_scenario)} / 43")
        print("=" * 90)

    # ── Persistence ───────────────────────────────────────────────────

    def save_report(self, path: str = "polymarket_scenario_bets.json"):
        """Save bet sheet to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "bankroll": self.bankroll,
            "max_per_bet": self.max_per_bet,
            "total_bets": len(self.bets),
            "active_bets": len([b for b in self.bets if b.recommended_bet >= 1.0]),
            "bets": [
                {
                    "scenario_code": b.scenario_code,
                    "scenario_name": b.scenario_name,
                    "severity": b.severity,
                    "base_probability": b.base_probability,
                    "market_question": b.market_question,
                    "condition_id": b.condition_id,
                    "outcome": b.outcome,
                    "token_id": b.token_id,
                    "market_price": b.market_price,
                    "thesis_probability": b.thesis_probability,
                    "edge": b.edge,
                    "edge_multiple": b.edge_multiple,
                    "kelly_fraction": b.kelly_fraction,
                    "recommended_bet": b.recommended_bet,
                    "payout_multiple": b.payout_multiple,
                    "volume": b.volume,
                    "liquidity": b.liquidity,
                }
                for b in self.bets
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Bet sheet saved to %s", path)

    # ── Live execution ────────────────────────────────────────────────

    async def execute_bets(self, dry_run: bool = True):
        """
        Place orders on Polymarket for all recommended bets.
        Set dry_run=False to ACTUALLY place orders (requires wallet + SDK).
        """
        active = [b for b in self.bets if b.recommended_bet >= 1.0]
        if not active:
            print("\nNo bets to execute.")
            return

        print(f"\n{'DRY RUN' if dry_run else 'LIVE EXECUTION'} — {len(active)} orders")
        print("-" * 60)

        agent = None
        if not dry_run:
            try:
                from agents.polymarket_agent import PolymarketAgent
                agent = PolymarketAgent()
            except ImportError:
                print("  ✗ PolymarketAgent not available — install py-clob-client")
                print("    pip install py-clob-client")
                return

        placed = 0
        failed = 0
        for b in active:
            # Size in shares: bet_dollars / price_per_share
            shares = b.recommended_bet / max(b.market_price, 0.01)
            side_char = b.outcome[0] if b.outcome else '['
            print(f"  {'[DRY]' if dry_run else '[LIVE]'} BUY {shares:.1f} shares {side_char} "
                  f"@ ${b.market_price:.3f} = ${b.recommended_bet:.2f}")
            print(f"         {b.market_question[:60]}")
            print(f"         Scenario: {b.scenario_code} | Edge: {b.edge:.1%} | Payout: {b.payout_multiple:.1f}x")

            if not dry_run and agent:
                result = agent.place_limit_order(
                    token_id=b.token_id,
                    price=round(b.market_price, 2),
                    size=round(shares, 1),
                    side="BUY",
                )
                if result:
                    placed += 1
                    print(f"         ✓ ORDER PLACED: {result}")
                else:
                    failed += 1
                    print(f"         ✗ ORDER FAILED")

            print()

        if not dry_run:
            print(f"\n{'='*60}")
            print(f"EXECUTION SUMMARY: {placed} placed, {failed} failed, "
                  f"{len(active) - placed - failed} skipped")


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Scenario Bet Scanner")
    parser.add_argument("--live", action="store_true", help="Place orders for real (requires wallet)")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Total bankroll ($)")
    parser.add_argument("--max-per-bet", type=float, default=50.0, help="Max per individual bet ($)")
    args = parser.parse_args()

    engine = ScenarioBetEngine(bankroll=args.bankroll, max_per_bet=args.max_per_bet)
    try:
        bets = await engine.scan_all_scenarios()
        engine.print_bet_sheet()
        engine.save_report()

        if args.live:
            print("\n⚠  LIVE MODE — Orders will be placed on Polymarket!")
            confirm = input("Type YES to confirm: ")
            if confirm.strip().upper() == "YES":
                await engine.execute_bets(dry_run=False)
            else:
                print("Aborted.")
        elif bets:
            print("\nDry run — add --live to place orders")
            await engine.execute_bets(dry_run=True)

    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())

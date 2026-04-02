"""
Polymarket Black Swan Scanner — AAC v2.8.0
============================================

Scans Polymarket for geopolitical / war / crisis events that align
with the Black Swan Pressure Cooker thesis:
  Iran wins → US leaves ME → Gulf adopts yuan → gold reprices → USD collapses

Finds mispriced tail-risk outcomes where the crowd underestimates
"impossible" scenarios that our thesis says are *probable*.

Integration:
  - Uses PolymarketAgent for live API calls
  - Cross-references pressure cooker indicators for thesis alignment
  - Feeds opportunities to planktonxd_prediction_harvester for execution
  - Outputs to crisis center for monitoring dashboards

Usage:
    python strategies/polymarket_blackswan_scanner.py             # full scan
    python strategies/polymarket_blackswan_scanner.py --thesis     # thesis-aligned only
    python strategies/polymarket_blackswan_scanner.py --json       # JSON output
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.polymarket_agent import PolymarketAgent, PolymarketEvent, PolymarketMarket

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# THESIS-ALIGNED SEARCH KEYWORDS
# ═══════════════════════════════════════════════════════════════════════════

# Keywords that map to our pressure cooker thesis chain
# Use longer/more specific phrases to avoid false positives
THESIS_KEYWORDS = {
    "iran_war": ["iran ", "iranian", "tehran", "irgc", "hormuz", "persian gulf"],
    "us_withdrawal": ["us troops", "us military", "middle east war", "centcom", "us pullout"],
    "israel_conflict": ["israel", "netanyahu", "idf ", "gaza", "hezbollah", "lebanon war", "hamas"],
    "oil_shock": ["oil price", "crude oil", "brent ", "opec ", "oil above", "oil below", "barrel"],
    "gold_reprice": ["gold price", "gold above", "gold below", "gold 3000", "gold 4000", "xau"],
    "usd_collapse": ["us dollar", "dollar index", "dxy", "de-dollarization", "dollar crash"],
    "yuan_rise": ["yuan", "renminbi", "petro-yuan", "china currency", "rmb "],
    "gulf_shift": ["saudi arabia", "saudi ", "uae ", "brics", "qatar ", "petrodollar"],
    "fed_trapped": ["fed rate", "interest rate cut", "fed cut", "fed hike", "powell", "fomc"],
    "crypto_crisis": ["bitcoin", "btc ", "ethereum", "crypto ", "bitcoin below", "bitcoin above"],
    "geopolitical": ["world war", "nuclear ", "military strike", "sanctions ", "trade war", "nato "],
    "inflation": ["inflation", "cpi ", "stagflation", "hyperinflation", "consumer price"],
    "recession": ["recession", "gdp ", "economic collapse", "depression"],
    "credit_crisis": ["credit default", "debt ceiling", "treasury bond", "bond yield", "yield curve"],
}
# Negative patterns — markets matching these are NOT thesis-relevant
_EXCLUSION_KEYWORDS = [
    "fifa", "world cup", "nba", "nfl", "nhl", "mlb", "premier league", "champions league",
    "super bowl", "oscar", "grammy", "emmy", "golden globe", "academy award",
    "james bond", "movie", "box office", "tv show", "netflix", "streaming",
    "bachelor", "love island", "survivor", "big brother", "reality tv",
    "election 2028", "miss universe",
]

# Minimum edge threshold — we want outcomes that are underpriced
MIN_EDGE_MULTIPLIER = 2.0  # Market says 2%, we think 4%+

# ═══════════════════════════════════════════════════════════════════════════
# 3 SCAN TIERS — switch ACTIVE_TIER to compare spreads
# ═══════════════════════════════════════════════════════════════════════════
#   "conservative" (15c) — deep OTM lottery tickets only
#   "standard"     (25c) — sweet spot: catches Iran/Oil/Crypto value plays
#   "aggressive"   (40c) — widest net, tighter edge requirements
#
# >>> SET THIS TO CHANGE SCAN MODE <<<
ACTIVE_TIER = "aggressive"  # "conservative" | "standard" | "aggressive"

TIER_PRESETS = {
    "conservative": {
        "cap": 0.15,
        "edge_tiers": [
            (0.05, 1.5, "deep_value"),
            (0.10, 2.0, "value"),
            (0.15, 2.5, "momentum"),
        ],
    },
    "standard": {
        "cap": 0.25,
        "edge_tiers": [
            (0.10, 1.5, "deep_value"),
            (0.25, 2.0, "value"),
        ],
    },
    "aggressive": {
        "cap": 0.40,
        "edge_tiers": [
            (0.10, 1.5, "deep_value"),
            (0.25, 2.0, "value"),
            (0.40, 2.5, "momentum"),
        ],
    },
}

# Resolve active preset
_preset = TIER_PRESETS[ACTIVE_TIER]
CATEGORY_PRICE_CAPS = {k: _preset["cap"] for k in [
    "iran_war", "us_withdrawal", "oil_shock", "gold_reprice", "gulf_shift",
    "usd_collapse", "yuan_rise", "israel_conflict", "credit_crisis",
    "geopolitical", "fed_trapped", "inflation", "recession", "crypto_crisis",
]}
DEFAULT_PRICE_CAP = _preset["cap"]
EDGE_TIERS = _preset["edge_tiers"]

# Minimum price floor — CLOB rejects sub-penny, we want 5c+ for liquidity
MIN_PRICE_FLOOR = 0.05

# Minimum 24h volume — skip dead markets nobody trades
MIN_VOLUME_24H = 1000  # $1K/day minimum activity


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BlackSwanOpportunity:
    """A Polymarket outcome that aligns with our black swan thesis."""
    market_question: str
    condition_id: str
    outcome: str                  # "YES" or "NO" — which side we're betting
    market_price: float           # What Polymarket says (e.g. 0.03 = 3%)
    thesis_probability: float     # What WE think (e.g. 0.15 = 15%)
    edge: float                   # thesis_prob - market_price
    category: str                 # Which thesis keyword group matched
    matched_keywords: List[str] = field(default_factory=list)
    potential_payout_per_dollar: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    tier: str = "deep_value"      # deep_value | value | momentum
    token_id: str = ""            # CLOB token ID for order execution
    detected_at: datetime = field(default_factory=datetime.now)

    @property
    def edge_multiple(self) -> float:
        return self.thesis_probability / max(self.market_price, 0.001)

    @property
    def kelly_fraction(self) -> float:
        """Simplified Kelly criterion: f* = (bp - q) / b where b=payout odds."""
        if self.market_price <= 0 or self.market_price >= 1:
            return 0.0
        b = (1.0 / self.market_price) - 1.0  # Payout odds
        p = self.thesis_probability
        q = 1.0 - p
        f = (b * p - q) / b
        return max(f, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# BLACK SWAN SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class PolymarketBlackSwanScanner:
    """
    Scans Polymarket for crisis/geopolitical markets aligned with
    the Black Swan Pressure Cooker thesis.
    """

    def __init__(self):
        self.agent = PolymarketAgent()
        self.opportunities: List[BlackSwanOpportunity] = []
        self._last_scan: Optional[datetime] = None

    async def close(self):
        await self.agent.close()

    # ─── Thesis probability estimation ────────────────────────────────

    def _estimate_thesis_probability(
        self, question: str, outcome: str, market_price: float, category: str
    ) -> float:
        """
        Estimate a thesis-adjusted probability for a black swan outcome.

        Our thesis says these events are MORE LIKELY than the crowd thinks.
        The pressure cooker is at ~44% — that's our confidence multiplier.
        """
        # Base: the market's implied probability
        implied = market_price

        # Thesis multipliers by category — how much more likely WE think it is
        multipliers = {
            "iran_war": 5.0,       # Market says 2%, we say ~10%
            "oil_shock": 4.0,
            "gold_reprice": 3.5,
            "usd_collapse": 3.0,
            "yuan_rise": 3.0,
            "gulf_shift": 3.5,
            "us_withdrawal": 4.0,
            "israel_conflict": 3.0,
            "fed_trapped": 2.5,
            "crypto_crisis": 2.0,
            "geopolitical": 3.0,
            "inflation": 2.5,
            "recession": 2.5,
            "credit_crisis": 3.0,
        }

        mult = multipliers.get(category, 2.0)

        # Scale multiplier by pressure cooker level (0-100)
        # Higher pressure = more confidence in thesis
        try:
            from strategies.black_swan_pressure_cooker import get_crisis_data
            result = get_crisis_data()
            pressure_pct = result.get("pressure_pct", 44)
            # At 44% pressure, use 88% of the full multiplier
            # At 75%+ pressure, use full multiplier
            pressure_scale = min(pressure_pct / 50.0, 1.0)
            mult = 1.0 + (mult - 1.0) * pressure_scale
        except Exception:
            pass  # Use base multiplier

        estimated = implied * mult
        # Cap at 0.50 — never more than 50% certain on tail events
        return min(estimated, 0.50)

    # ─── Market matching ──────────────────────────────────────────────

    def _match_thesis_category(self, question: str) -> List[tuple]:
        """Check if a market question matches our thesis keywords.

        Returns list of (category, matched_keywords).
        Excludes sports/entertainment false positives via _EXCLUSION_KEYWORDS.
        """
        question_lower = question.lower()
        # Reject if any exclusion keyword matches
        for excl in _EXCLUSION_KEYWORDS:
            if excl in question_lower:
                return []
        matches = []
        for category, keywords in THESIS_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in question_lower]
            if matched:
                matches.append((category, matched))
        return matches

    # ─── Scanning ─────────────────────────────────────────────────────

    async def scan(self, max_pages: int = 5) -> List[BlackSwanOpportunity]:
        """
        Full scan of Polymarket for black swan opportunities.

        Fetches active markets, filters for thesis-aligned questions,
        and evaluates edge on cheap outcomes.
        """
        logger.info("Black Swan Scanner: scanning Polymarket for thesis-aligned opportunities...")
        self.opportunities = []
        all_markets: List[PolymarketMarket] = []
        api_reachable = True

        # Fetch multiple pages of active markets
        for page in range(max_pages):
            try:
                markets = await self.agent.get_active_markets(
                    limit=100, offset=page * 100
                )
                all_markets.extend(markets)
                if len(markets) < 100:
                    break
            except Exception as e:
                logger.warning("Failed to fetch page %d: %s", page, e)
                if page == 0:
                    api_reachable = False
                break

        logger.info("Fetched %d active markets from Polymarket", len(all_markets))

        # Gamma API text search is non-functional (ignores query params).
        # Instead, fetch additional pages sorted by total volume to get
        # older high-volume thesis markets that may rank lower on 24h volume.
        if api_reachable:
            extra_before = len(all_markets)
            try:
                vol_markets = await self.agent.get_active_markets(
                    limit=100, offset=0, order="volume"
                )
                for mkt in vol_markets:
                    if not any(m.condition_id == mkt.condition_id for m in all_markets):
                        all_markets.append(mkt)
            except Exception as e:
                logger.debug("Volume-sorted fetch failed: %s", e)
            added = len(all_markets) - extra_before
            if added:
                logger.info("Added %d unique markets from volume sort", added)

        logger.info("Total unique markets to evaluate: %d", len(all_markets))

        # Evaluate each market against our thesis
        thesis_matches = 0
        price_rejections = 0
        volume_rejections = 0
        edge_rejections = 0
        for mkt in all_markets:
            matches = self._match_thesis_category(mkt.question)
            if not matches:
                continue
            thesis_matches += 1

            # Check YES side (cheap YES = crowd thinks unlikely)
            for category, matched_kws in matches:
                # Tiered price cap — high-conviction categories get wider net
                price_cap = CATEGORY_PRICE_CAPS.get(category, DEFAULT_PRICE_CAP)

                for outcome, price in [("YES", mkt.yes_price), ("NO", mkt.no_price)]:
                    if price > price_cap:
                        price_rejections += 1
                        logger.debug(
                            "PRICE REJECT [%s] %s @ %.4f > cap %.2f: %s",
                            category, outcome, price, price_cap,
                            mkt.question[:60],
                        )
                        continue
                    if price < MIN_PRICE_FLOOR:
                        price_rejections += 1
                        continue

                    # Skip dead markets (optional — 0 volume still tracked)
                    if mkt.volume < MIN_VOLUME_24H and mkt.volume > 0:
                        volume_rejections += 1
                        continue

                    thesis_prob = self._estimate_thesis_probability(
                        mkt.question, outcome, price, category
                    )
                    edge = thesis_prob - price

                    # Tiered edge requirement — higher price = higher bar
                    tier_label = "deep_value"
                    required_mult = MIN_EDGE_MULTIPLIER
                    for max_p, mult, label in EDGE_TIERS:
                        if price <= max_p:
                            required_mult = mult
                            tier_label = label
                            break

                    if edge > 0 and thesis_prob >= price * required_mult:
                        # Resolve token_id for the chosen side
                        tid = mkt.yes_token_id if outcome == "YES" else mkt.no_token_id
                        opp = BlackSwanOpportunity(
                            market_question=mkt.question,
                            condition_id=mkt.condition_id,
                            outcome=outcome,
                            market_price=price,
                            thesis_probability=thesis_prob,
                            edge=edge,
                            category=category,
                            matched_keywords=matched_kws,
                            potential_payout_per_dollar=1.0 / price if price > 0 else 0,
                            volume_24h=mkt.volume,
                            liquidity=mkt.liquidity,
                            tier=tier_label,
                            token_id=tid,
                        )
                        self.opportunities.append(opp)
                    else:
                        edge_rejections += 1
                        logger.debug(
                            "EDGE REJECT [%s] %s @ %.4f, thesis=%.4f, edge=%.4f, need %.1fx: %s",
                            category, outcome, price, thesis_prob, edge,
                            required_mult, mkt.question[:60],
                        )

        logger.info(
            "Filter stats: %d thesis matches, %d price rejects, %d volume rejects, %d edge rejects",
            thesis_matches, price_rejections, volume_rejections, edge_rejections,
        )

        # Sort by edge (best opportunities first)
        self.opportunities.sort(key=lambda o: o.edge, reverse=True)
        self._last_scan = datetime.now()

        logger.info(
            "Black Swan Scanner complete: %d thesis-aligned opportunities found",
            len(self.opportunities),
        )

        # TurboQuant: record Polymarket scan snapshot
        try:
            from strategies.turboquant_integrations import IntegrationHub
            _tq_hub = IntegrationHub()
            _tq_hub.record_polymarket(
                [o.__dict__ if hasattr(o, '__dict__') else o for o in self.opportunities]
            )
            _tq_hub.save_all()
        except Exception as _tq_err:
            logger.debug("TurboQuant record skipped: %s", _tq_err)

        return self.opportunities

    # ─── Output ───────────────────────────────────────────────────────

    def generate_report(self, top_n: int = 20) -> str:
        """Generate a human-readable report of black swan opportunities."""
        lines = [
            "=" * 70,
            "  POLYMARKET BLACK SWAN OPPORTUNITIES",
            "=" * 70,
            f"  Scan Time: {self._last_scan or 'never'}",
            f"  Total Opportunities: {len(self.opportunities)}",
            f"  Showing Top {min(top_n, len(self.opportunities))}",
            "-" * 70,
        ]

        for i, opp in enumerate(self.opportunities[:top_n], 1):
            payout = f"${opp.potential_payout_per_dollar:.0f}"
            tier_tag = opp.tier.upper().replace("_", " ")
            bet_tag = f"BUY {opp.outcome}"
            lines.extend([
                f"\n  #{i}  [{opp.category.upper()}]  ** {bet_tag} ** @ {opp.market_price:.4f}  [{tier_tag}]",
                f"      Q: {opp.market_question[:80]}",
                f"      Market: {opp.market_price:.2%} -> Thesis: {opp.thesis_probability:.2%}  "
                f"(Edge: {opp.edge:.2%}, {opp.edge_multiple:.1f}x)",
                f"      Payout: {payout}/$ | Kelly: {opp.kelly_fraction:.1%} | "
                f"Vol: ${opp.volume_24h:,.0f} | Liq: ${opp.liquidity:,.0f}",
                f"      Keywords: {', '.join(opp.matched_keywords)}",
            ])

        lines.extend([
            "\n" + "=" * 70,
            "  These are thesis-aligned mispriced outcomes.",
            "  Feed to PlanktonXD for automated execution.",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_json(self) -> List[Dict[str, Any]]:
        """Export opportunities as JSON-serializable list."""
        return [
            {
                "question": o.market_question,
                "condition_id": o.condition_id,
                "outcome": o.outcome,
                "token_id": o.token_id,
                "market_price": o.market_price,
                "thesis_probability": o.thesis_probability,
                "edge": o.edge,
                "edge_multiple": o.edge_multiple,
                "category": o.category,
                "keywords": o.matched_keywords,
                "payout_per_dollar": o.potential_payout_per_dollar,
                "kelly_fraction": o.kelly_fraction,
                "volume_24h": o.volume_24h,
                "liquidity": o.liquidity,
                "tier": o.tier,
                "detected_at": o.detected_at.isoformat(),
            }
            for o in self.opportunities
        ]


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Black Swan Scanner")
    parser.add_argument("--thesis", action="store_true", help="Show thesis-aligned only (default)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--top", type=int, default=20, help="Show top N opportunities")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scanner = PolymarketBlackSwanScanner()
    try:
        opps = await scanner.scan()
        if args.json:
            print(json.dumps(scanner.to_json(), indent=2))
        else:
            print(scanner.generate_report(top_n=args.top))
    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(_main())

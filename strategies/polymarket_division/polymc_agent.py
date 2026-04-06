"""
PolyMC Agent — Top 100 Lucrative Markets + 100K Monte Carlo Simulator
======================================================================
AAC Polymarket Division — Strategy #3

Scrapes the top 100 Polymarket markets by volume/liquidity, runs 100,000
Monte Carlo simulations per market to calculate expected value, identifies
high-EV opportunities, and manages a portfolio of targeted bets with
exit strategies (take-profit / stop-loss).

Features:
  - Gamma API scanner: top 100 active markets sorted by volume + liquidity
  - 100K Monte Carlo: numpy Bernoulli trials, seed=42, per-market EV
  - Portfolio: 5 high-EV target bets with specific entry/exit rules
  - Kelly criterion position sizing
  - Exit strategy engine: per-bet TP/SL + portfolio-level EV floor

Usage:
    python -m strategies.polymarket_division.polymc_agent                # scan + report
    python -m strategies.polymarket_division.polymc_agent --monte-carlo  # full MC sim
    python -m strategies.polymarket_division.polymc_agent --portfolio    # show portfolio
    python -m strategies.polymarket_division.polymc_agent --json         # JSON output
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -- UTF-8 stdout fix for Windows cp1252 terminals --
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

MC_SIMULATIONS = 100_000
MC_SEED = 42

# Portfolio allocation
PORTFOLIO_BUDGET = 500.0  # $500 total across 5 bets
BET_SIZE = 100.0          # $100 per bet

# Exit strategy defaults
DEFAULT_TP_MULTIPLIER = 1.3   # Take profit when price rises 30% above entry
DEFAULT_SL_MULTIPLIER = 0.75  # Stop loss when price drops 25% below entry
PORTFOLIO_EV_FLOOR = 0.04     # Exit all if portfolio EV drops below 4%
PARTIAL_PROFIT_THRESHOLD = 1.50  # Take partial profit if single bet +50%


# ============================================================================
# TARGET PORTFOLIO — 5 High-EV Markets
# ============================================================================

@dataclass
class TargetBet:
    """A specific target market bet with entry/exit rules."""
    name: str
    market_slug: str
    condition_id: str
    token_id: str        # CLOB token ID for the YES/NO side we're buying
    side: str            # "YES" or "NO"
    entry_price: float   # Price we're targeting to enter at
    implied_prob: float  # Market's implied probability (= YES price)
    our_prob: float      # Our estimated true probability
    take_profit: float   # Exit price (above entry)
    stop_loss: float     # Exit price (below entry)
    bet_size: float = BET_SIZE
    status: str = "pending"  # pending | active | exited_tp | exited_sl
    fill_price: float = 0.0
    current_price: float = 0.0
    pnl: float = 0.0

    @property
    def shares_at_entry(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return self.bet_size / self.entry_price

    @property
    def max_payout(self) -> float:
        return self.shares_at_entry * 1.0  # Each share pays $1 if YES

    @property
    def ev(self) -> float:
        """Expected value: prob * payout - (1-prob) * cost."""
        return self.our_prob * self.max_payout - (1 - self.our_prob) * self.bet_size

    @property
    def ev_pct(self) -> float:
        if self.bet_size <= 0:
            return 0.0
        return (self.ev / self.bet_size) * 100

    @property
    def kelly_fraction(self) -> float:
        """Kelly criterion: f* = (bp - q) / b, capped at quarter-Kelly."""
        if self.entry_price <= 0 or self.entry_price >= 1:
            return 0.0
        b = (1.0 / self.entry_price) - 1.0
        p = self.our_prob
        q = 1.0 - p
        f = (b * p - q) / b
        # Quarter-Kelly: full Kelly is too aggressive for prediction markets
        return max(f * 0.25, 0.0)


# The 5 target bets from the PolyMC thesis
TARGET_PORTFOLIO: List[TargetBet] = [
    TargetBet(
        name="Spain wins FIFA World Cup 2026",
        market_slug="will-spain-win-the-2026-fifa-world-cup",
        condition_id="",  # Populated on scan
        token_id="",      # Populated on scan
        side="YES",
        entry_price=0.158,
        implied_prob=0.158,
        our_prob=0.19,
        take_profit=0.22,
        stop_loss=0.12,
    ),
    TargetBet(
        name="OKC Thunder win NBA Championship 2026",
        market_slug="will-the-oklahoma-city-thunder-win-the-2024-25-nba-championship",
        condition_id="",
        token_id="",
        side="YES",
        entry_price=0.37,
        implied_prob=0.37,
        our_prob=0.42,
        take_profit=0.45,
        stop_loss=0.30,
    ),
    TargetBet(
        name="JD Vance wins 2028 Presidential Election",
        market_slug="will-jd-vance-win-the-2028-us-presidential-election",
        condition_id="",
        token_id="",
        side="YES",
        entry_price=0.181,
        implied_prob=0.181,
        our_prob=0.20,
        take_profit=0.25,
        stop_loss=0.14,
    ),
    TargetBet(
        name="Gavin Newsom wins 2028 Democratic Nomination",
        market_slug="will-gavin-newsom-win-the-2028-democratic-presidential-nomination",
        condition_id="",
        token_id="",
        side="YES",
        entry_price=0.243,
        implied_prob=0.243,
        our_prob=0.26,
        take_profit=0.32,
        stop_loss=0.18,
    ),
    TargetBet(
        name="JD Vance wins 2028 Republican Nomination",
        market_slug="will-jd-vance-win-the-2028-republican-presidential-nomination",
        condition_id="",
        token_id="",
        side="YES",
        entry_price=0.368,
        implied_prob=0.368,
        our_prob=0.39,
        take_profit=0.45,
        stop_loss=0.30,
    ),
]


# ============================================================================
# TOP 100 MARKET SCANNER
# ============================================================================

@dataclass
class ScannedMarket:
    """A market from the top 100 scan with MC simulation results."""
    condition_id: str
    question: str
    slug: str
    yes_price: float
    no_price: float
    volume: float
    volume_24h: float
    liquidity: float
    yes_token_id: str = ""
    no_token_id: str = ""
    # MC simulation results
    mc_ev: float = 0.0           # Expected value per $1 bet
    mc_ev_pct: float = 0.0       # EV as percentage
    mc_prob_profit: float = 0.0  # Probability of profit from MC
    mc_mean_return: float = 0.0  # Mean return from MC
    mc_max_payout: float = 0.0   # Max payout per $1 from MC
    mc_var_95: float = 0.0       # 95% Value at Risk
    mc_sharpe: float = 0.0       # Sharpe ratio from MC
    best_side: str = "YES"       # Which side has better EV
    kelly: float = 0.0           # Kelly fraction for best side


class PolyMCAgent:
    """
    PolyMC Agent: Top 100 market scanner + 100K Monte Carlo simulator.

    Scrapes Polymarket's most liquid/active markets, runs Monte Carlo
    simulations to find positive-EV opportunities, and manages a
    portfolio of 5 high-conviction bets.
    """

    def __init__(self):
        self.scanned_markets: List[ScannedMarket] = []
        self.portfolio = list(TARGET_PORTFOLIO)
        self._rng = np.random.default_rng(MC_SEED)
        self._last_scan: Optional[datetime] = None
        self._session: Optional[Any] = None

    async def _get_session(self):
        if aiohttp is None:
            raise ImportError("aiohttp required: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ─── Top 100 Market Scanner ───────────────────────────────────────

    async def scan_top_markets(self, limit: int = 100, max_pages: int = 5) -> List[ScannedMarket]:
        """
        Fetch top markets from Gamma API sorted by volume + liquidity.
        Paginates through results to get up to `limit` markets.
        """
        session = await self._get_session()
        markets: List[ScannedMarket] = []
        seen_ids: set = set()
        page_size = min(limit, 50)  # Gamma API paginates at ~50

        for page in range(max_pages):
            offset = page * page_size
            url = (
                f"{GAMMA_API}/markets"
                f"?limit={page_size}&offset={offset}"
                f"&active=true&closed=false"
                f"&order=volume&ascending=false"
            )
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Gamma API returned {resp.status} on page {page}")
                        break
                    data = await resp.json()

                if not data:
                    break

                for m in data:
                    cid = m.get("conditionId", "") or m.get("condition_id", "")
                    if not cid or cid in seen_ids:
                        continue
                    seen_ids.add(cid)

                    # Parse token IDs
                    tokens = m.get("clobTokenIds", "") or ""
                    yes_id, no_id = "", ""
                    if isinstance(tokens, str) and tokens:
                        try:
                            parsed = json.loads(tokens)
                            if isinstance(parsed, list) and len(parsed) >= 2:
                                yes_id, no_id = str(parsed[0]), str(parsed[1])
                            elif isinstance(parsed, list) and len(parsed) == 1:
                                yes_id = str(parsed[0])
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Parse prices
                    raw_prices = m.get("outcomePrices", "0.5,0.5")
                    yes_price = 0.5
                    if isinstance(raw_prices, str):
                        cleaned = raw_prices.strip().strip("[]")
                        parts = [p.strip().strip('"').strip("'") for p in cleaned.split(",")]
                        if parts:
                            try:
                                yes_price = float(parts[0])
                            except (ValueError, IndexError):
                                yes_price = 0.5
                    no_price = 1.0 - yes_price

                    sm = ScannedMarket(
                        condition_id=cid,
                        question=m.get("question", ""),
                        slug=m.get("slug", ""),
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=float(m.get("volume", 0) or 0),
                        volume_24h=float(m.get("volume24hr", 0) or m.get("volume_24hr", 0) or 0),
                        liquidity=float(m.get("liquidity", 0) or 0),
                        yes_token_id=yes_id,
                        no_token_id=no_id,
                    )
                    markets.append(sm)

                    if len(markets) >= limit:
                        break

                if len(markets) >= limit:
                    break
                # Rate limit
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break

        # Sort by combined volume + liquidity score
        markets.sort(key=lambda x: x.volume + x.liquidity, reverse=True)
        self.scanned_markets = markets[:limit]
        self._last_scan = datetime.now()
        logger.info(f"Scanned {len(self.scanned_markets)} top markets")
        return self.scanned_markets

    # ─── Monte Carlo Simulator ────────────────────────────────────────

    def run_monte_carlo(
        self,
        prob: float,
        entry_price: float,
        bet_size: float = 1.0,
        n_sims: int = MC_SIMULATIONS,
    ) -> Dict[str, float]:
        """
        Run 100K Monte Carlo simulation for a single binary outcome bet.

        Each simulation: Bernoulli trial with probability `prob`.
        If YES: payout = shares * $1 = bet_size / entry_price
        If NO:  payout = $0, loss = bet_size

        Returns dict with ev, ev_pct, prob_profit, mean_return, max_payout,
        var_95, cvar_95, sharpe.
        """
        if entry_price <= 0 or entry_price >= 1 or prob <= 0 or prob >= 1:
            return {
                "ev": 0.0, "ev_pct": 0.0, "prob_profit": 0.0,
                "mean_return": 0.0, "max_payout": 0.0,
                "var_95": -bet_size, "cvar_95": -bet_size, "sharpe": 0.0,
            }

        shares = bet_size / entry_price
        payout_win = shares * 1.0  # Each share pays $1 on YES

        # Bernoulli trials: 1 = win, 0 = lose
        outcomes = self._rng.binomial(1, prob, size=n_sims)

        # PnL for each simulation
        pnl = np.where(outcomes == 1, payout_win - bet_size, -bet_size)

        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl))
        prob_profit = float(np.mean(pnl > 0))
        var_95 = float(np.percentile(pnl, 5))
        cvar_95 = float(np.mean(pnl[pnl <= var_95])) if np.any(pnl <= var_95) else var_95
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        return {
            "ev": mean_pnl,
            "ev_pct": (mean_pnl / bet_size) * 100 if bet_size > 0 else 0.0,
            "prob_profit": prob_profit,
            "mean_return": mean_pnl,
            "max_payout": payout_win,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sharpe": sharpe,
        }

    def run_portfolio_monte_carlo(
        self,
        bets: Optional[List[TargetBet]] = None,
        n_sims: int = MC_SIMULATIONS,
    ) -> Dict[str, Any]:
        """
        Run 100K Monte Carlo simulation for the entire portfolio.
        Each sim independently resolves all 5 bets via Bernoulli trials.
        """
        if bets is None:
            bets = self.portfolio

        n_bets = len(bets)
        total_cost = sum(b.bet_size for b in bets)

        # Build probability and shares arrays
        probs = np.array([b.our_prob for b in bets])
        shares = np.array([b.bet_size / max(b.entry_price, 0.001) for b in bets])
        costs = np.array([b.bet_size for b in bets])

        # Simulate: (n_sims, n_bets) matrix of Bernoulli outcomes
        outcomes = self._rng.binomial(1, probs, size=(n_sims, n_bets))

        # Payouts per sim: shares * $1 for wins, $0 for losses
        payouts = outcomes * shares  # (n_sims, n_bets)
        total_payouts = np.sum(payouts, axis=1)  # (n_sims,)
        pnl = total_payouts - total_cost

        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl))
        prob_profit = float(np.mean(pnl > 0))
        max_payout = float(np.max(total_payouts))
        var_95 = float(np.percentile(pnl, 5))
        cvar_95 = float(np.mean(pnl[pnl <= var_95])) if np.any(pnl <= var_95) else var_95
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        # Per-bet stats
        per_bet = []
        for i, bet in enumerate(bets):
            bet_pnl = np.where(outcomes[:, i] == 1, shares[i] - costs[i], -costs[i])
            per_bet.append({
                "name": bet.name,
                "ev": float(np.mean(bet_pnl)),
                "prob_profit": float(np.mean(bet_pnl > 0)),
                "max_payout": float(shares[i]),
            })

        return {
            "total_cost": total_cost,
            "mean_return": mean_pnl,
            "ev_pct": (mean_pnl / total_cost) * 100 if total_cost > 0 else 0.0,
            "prob_profit": prob_profit,
            "max_payout": max_payout,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sharpe": sharpe,
            "n_sims": n_sims,
            "n_bets": n_bets,
            "per_bet": per_bet,
        }

    def simulate_top_markets(self) -> List[ScannedMarket]:
        """Run Monte Carlo on all scanned markets and rank by EV."""
        for m in self.scanned_markets:
            # Try YES side
            yes_mc = self.run_monte_carlo(m.yes_price, m.yes_price, bet_size=1.0)
            # Try NO side
            no_mc = self.run_monte_carlo(m.no_price, m.no_price, bet_size=1.0)

            # Pick the side with better EV (or least negative)
            if yes_mc["ev"] >= no_mc["ev"]:
                best = yes_mc
                m.best_side = "YES"
            else:
                best = no_mc
                m.best_side = "NO"

            m.mc_ev = best["ev"]
            m.mc_ev_pct = best["ev_pct"]
            m.mc_prob_profit = best["prob_profit"]
            m.mc_mean_return = best["mean_return"]
            m.mc_max_payout = best["max_payout"]
            m.mc_var_95 = best["var_95"]
            m.mc_sharpe = best["sharpe"]

            # Kelly
            price = m.yes_price if m.best_side == "YES" else m.no_price
            if 0 < price < 1:
                b = (1.0 / price) - 1.0
                p = price  # Market-implied probability as baseline
                q = 1.0 - p
                m.kelly = max((b * p - q) / b, 0.0)

        # Sort by MC EV descending
        self.scanned_markets.sort(key=lambda x: x.mc_ev, reverse=True)
        return self.scanned_markets

    # ─── Portfolio Management ─────────────────────────────────────────

    async def resolve_portfolio_tokens(self):
        """Look up condition IDs and token IDs for portfolio bets via Gamma API."""
        session = await self._get_session()
        for bet in self.portfolio:
            if bet.token_id:
                continue  # Already resolved

            # Search by slug
            url = f"{GAMMA_API}/markets?slug={bet.market_slug}&limit=1"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Could not resolve {bet.name}: HTTP {resp.status}")
                        continue
                    data = await resp.json()

                if not data:
                    # Try search by question text
                    search_url = f"{GAMMA_API}/markets?limit=5"
                    async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as resp2:
                        data = await resp2.json() if resp2.status == 200 else []
                    # Filter by name match
                    data = [m for m in data if bet.name.lower()[:20] in m.get("question", "").lower()]

                if data:
                    m = data[0]
                    bet.condition_id = m.get("conditionId", "") or m.get("condition_id", "")

                    tokens = m.get("clobTokenIds", "")
                    if isinstance(tokens, str) and tokens:
                        try:
                            parsed = json.loads(tokens)
                            if isinstance(parsed, list) and len(parsed) >= 2:
                                bet.token_id = str(parsed[0]) if bet.side == "YES" else str(parsed[1])
                            elif isinstance(parsed, list) and len(parsed) == 1:
                                bet.token_id = str(parsed[0])
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Update current price
                    raw_prices = m.get("outcomePrices", "")
                    if isinstance(raw_prices, str) and raw_prices:
                        cleaned = raw_prices.strip().strip("[]")
                        parts = [p.strip().strip('"') for p in cleaned.split(",")]
                        if parts:
                            try:
                                yes_p = float(parts[0])
                                bet.current_price = yes_p if bet.side == "YES" else (1.0 - yes_p)
                                bet.implied_prob = yes_p
                            except ValueError:
                                pass

                    logger.info(f"Resolved {bet.name}: token={bet.token_id[:16]}...")

            except Exception as e:
                logger.warning(f"Error resolving {bet.name}: {e}")

            await asyncio.sleep(0.15)  # Rate limit

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Generate portfolio summary with EV calculations."""
        total_cost = sum(b.bet_size for b in self.portfolio)
        total_ev = sum(b.ev for b in self.portfolio)
        total_max = sum(b.max_payout for b in self.portfolio)

        bets_summary = []
        for b in self.portfolio:
            bets_summary.append({
                "name": b.name,
                "side": b.side,
                "entry_price": b.entry_price,
                "our_prob": b.our_prob,
                "bet_size": b.bet_size,
                "shares": round(b.shares_at_entry, 1),
                "max_payout": round(b.max_payout, 2),
                "ev": round(b.ev, 2),
                "ev_pct": round(b.ev_pct, 1),
                "kelly": round(b.kelly_fraction, 4),
                "take_profit": b.take_profit,
                "stop_loss": b.stop_loss,
                "status": b.status,
                "current_price": b.current_price,
            })

        return {
            "total_cost": total_cost,
            "total_ev": round(total_ev, 2),
            "total_ev_pct": round((total_ev / total_cost) * 100, 1) if total_cost > 0 else 0,
            "total_max_payout": round(total_max, 2),
            "avg_multiplier": round(total_max / total_cost, 1) if total_cost > 0 else 0,
            "n_bets": len(self.portfolio),
            "bets": bets_summary,
        }

    # ─── Report Generation ────────────────────────────────────────────

    def generate_top100_report(self, top_n: int = 20) -> str:
        """Generate a formatted report of top markets by MC EV."""
        lines = []
        lines.append("=" * 120)
        lines.append("  POLYMC AGENT -- TOP 100 MARKETS BY VOLUME/LIQUIDITY")
        lines.append(f"  Scanned: {self._last_scan}")
        lines.append(f"  Markets: {len(self.scanned_markets)} | MC Sims: {MC_SIMULATIONS:,} | Seed: {MC_SEED}")
        lines.append("=" * 120)
        lines.append("")
        lines.append(f"  {'#':<4} {'Market':<55} {'Price':>7} {'Side':>5} {'Vol':>12} "
                      f"{'MC EV':>8} {'EV%':>7} {'P(Win)':>7} {'Kelly':>7} {'Sharpe':>7}")
        lines.append("  " + "-" * 116)

        display = self.scanned_markets[:top_n]
        for i, m in enumerate(display, 1):
            price = m.yes_price if m.best_side == "YES" else m.no_price
            q = m.question[:53] if len(m.question) > 53 else m.question
            lines.append(
                f"  {i:<4} {q:<55} ${price:.3f} {m.best_side:>5} "
                f"${m.volume:>10,.0f} "
                f"${m.mc_ev:>6.3f} {m.mc_ev_pct:>6.1f}% "
                f"{m.mc_prob_profit:>6.1%} {m.kelly:>6.3f} {m.mc_sharpe:>6.2f}"
            )

        lines.append("")
        positive_ev = [m for m in self.scanned_markets if m.mc_ev > 0]
        lines.append(f"  Positive EV markets: {len(positive_ev)} / {len(self.scanned_markets)}")
        if positive_ev:
            avg_ev = sum(m.mc_ev_pct for m in positive_ev) / len(positive_ev)
            lines.append(f"  Average EV (positive only): {avg_ev:.1f}%")

        return "\n".join(lines)

    def generate_portfolio_report(self) -> str:
        """Generate formatted portfolio report."""
        summary = self.get_portfolio_summary()
        lines = []
        lines.append("=" * 120)
        lines.append("  POLYMC AGENT -- TARGET PORTFOLIO (5 HIGH-EV BETS)")
        lines.append(f"  Budget: ${summary['total_cost']:.0f} | "
                      f"Total EV: ${summary['total_ev']:.2f} ({summary['total_ev_pct']:.1f}%) | "
                      f"Max Payout: ${summary['total_max_payout']:.2f} | "
                      f"Avg Multiplier: {summary['avg_multiplier']:.1f}x")
        lines.append("=" * 120)
        lines.append("")
        lines.append(f"  {'#':<4} {'Market':<45} {'Side':>5} {'Entry':>7} {'Prob':>7} "
                      f"{'Size':>7} {'Shares':>8} {'MaxPay':>8} {'EV':>8} {'EV%':>6} "
                      f"{'TP':>7} {'SL':>7} {'Status':>10}")
        lines.append("  " + "-" * 116)

        for i, b in enumerate(summary["bets"], 1):
            name = b["name"][:43] if len(b["name"]) > 43 else b["name"]
            lines.append(
                f"  {i:<4} {name:<45} {b['side']:>5} ${b['entry_price']:.3f} "
                f"{b['our_prob']:>6.1%} ${b['bet_size']:>5.0f} "
                f"{b['shares']:>7.1f} ${b['max_payout']:>7.2f} "
                f"${b['ev']:>6.2f} {b['ev_pct']:>5.1f}% "
                f"${b['take_profit']:.2f} ${b['stop_loss']:.2f} "
                f"{b['status']:>10}"
            )

        lines.append("")
        return "\n".join(lines)

    def generate_mc_report(self) -> str:
        """Generate Monte Carlo simulation report for portfolio."""
        mc = self.run_portfolio_monte_carlo()
        lines = []
        lines.append("=" * 120)
        lines.append("  POLYMC AGENT -- 100K MONTE CARLO SIMULATION")
        lines.append(f"  Simulations: {mc['n_sims']:,} | Bets: {mc['n_bets']} | "
                      f"Total Cost: ${mc['total_cost']:.0f}")
        lines.append("=" * 120)
        lines.append("")
        lines.append(f"  Mean Return:     ${mc['mean_return']:>+.2f}")
        lines.append(f"  EV:              {mc['ev_pct']:>+.1f}%")
        lines.append(f"  P(Profit):       {mc['prob_profit']:.1%}")
        lines.append(f"  Max Payout:      ${mc['max_payout']:,.2f}")
        lines.append(f"  VaR (95%):       ${mc['var_95']:>.2f}")
        lines.append(f"  CVaR (95%):      ${mc['cvar_95']:>.2f}")
        lines.append(f"  Sharpe Ratio:    {mc['sharpe']:>.3f}")
        lines.append("")
        lines.append("  Per-Bet Breakdown:")
        lines.append(f"  {'#':<4} {'Market':<50} {'EV':>8} {'P(Win)':>8} {'MaxPay':>10}")
        lines.append("  " + "-" * 82)
        for i, b in enumerate(mc["per_bet"], 1):
            name = b["name"][:48] if len(b["name"]) > 48 else b["name"]
            lines.append(
                f"  {i:<4} {name:<50} ${b['ev']:>+6.2f} {b['prob_profit']:>7.1%} "
                f"${b['max_payout']:>8.2f}"
            )
        lines.append("")
        return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

import asyncio


async def _main():
    parser = argparse.ArgumentParser(description="PolyMC Agent -- Top 100 + Monte Carlo")
    parser.add_argument("--monte-carlo", action="store_true", help="Run portfolio MC simulation")
    parser.add_argument("--portfolio", action="store_true", help="Show target portfolio")
    parser.add_argument("--scan", action="store_true", help="Scan top 100 markets")
    parser.add_argument("--top", type=int, default=20, help="Number of top markets to show")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--resolve", action="store_true", help="Resolve token IDs for portfolio")
    args = parser.parse_args()

    agent = PolyMCAgent()

    try:
        if args.monte_carlo:
            print(agent.generate_mc_report())
        elif args.portfolio:
            if args.resolve:
                await agent.resolve_portfolio_tokens()
            print(agent.generate_portfolio_report())
            if args.json:
                print(json.dumps(agent.get_portfolio_summary(), indent=2))
        elif args.scan:
            print(f"\n  Scanning top {args.top * 5} markets from Gamma API...")
            await agent.scan_top_markets(limit=100)
            print(f"  Running {MC_SIMULATIONS:,} Monte Carlo simulations per market...")
            agent.simulate_top_markets()
            print(agent.generate_top100_report(top_n=args.top))
            if args.json:
                data = [asdict(m) for m in agent.scanned_markets[:args.top]]
                print(json.dumps(data, indent=2, default=str))
        else:
            # Default: portfolio + MC
            print(agent.generate_portfolio_report())
            print(agent.generate_mc_report())
    finally:
        await agent.close()


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    asyncio.run(_main())

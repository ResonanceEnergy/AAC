"""
Polymarket Active Scanner — Unified Live Trading Engine
=========================================================
AAC v3.6.0

Runs all three Polymarket strategies in a unified async loop:
  1. WAR ROOM   — Geopolitical thesis scanning (blackswan pressure cooker)
  2. PLANKTONXD — Deep OTM micro-arbitrage harvester
  3. POLYMC     — Top 100 markets + Monte Carlo + portfolio management

Modes:
  scan     — Single pass: scan all 3 strategies, report opportunities (no trades)
  monitor  — Continuous loop: scan + monitor positions + exit signals
  live     — Continuous loop: scan + execute trades + monitor + exits

Usage:
    python -m strategies.polymarket_division.active_scanner              # scan only
    python -m strategies.polymarket_division.active_scanner --mode live  # live trading
    python -m strategies.polymarket_division.active_scanner --mode monitor --interval 300
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import structlog

_log = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SCAN_INTERVAL = 300  # 5 minutes between scans
DEFAULT_MONITOR_INTERVAL = 3600  # 1 hour between position checks
MAX_POSITION_SIZE_USD = 21.0  # Max per-bet: 5% of typical $421 bankroll
MAX_DAILY_BETS = 5  # Safety cap — was 50, reckless
MIN_EDGE_THRESHOLD = 0.03  # 3% minimum edge to place a bet
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

# ── POST-MORTEM FIXES (2026-04-03) ─────────────────────────────────────
# Root causes of $452 loss: zero diversification, expired markets,
# fantasy thesis probabilities, no exits, wallet drained instantly.

# Diversification: max percentage of bankroll per thesis category
MAX_CATEGORY_PCT = 0.20  # Max 20% of bankroll in one category
CATEGORY_MAP = {
    "geopolitical": ["iran", "hormuz", "hezbollah", "israel", "saudi", "ceasefire", "invade", "regime", "military action", "pahlavi", "netanyahu"],
    "oil_energy": ["crude oil", "oil price", "brent", "wti", "natural gas", "opec"],
    "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth"],
    "sports": ["fifa", "nba", "nfl", "nhl", "world cup", "championship", "super bowl"],
    "us_politics": ["president", "election", "democrat", "republican", "congress", "senate", "nomination"],
    "economics": ["fed", "interest rate", "gdp", "unemployment", "inflation", "recession"],
}

# Time horizon: never buy markets expiring within this many days
MIN_DAYS_TO_EXPIRY = 14

# Bankroll management
CASH_RESERVE_PCT = 0.30  # Always keep 30% in wallet
DAILY_DEPLOY_LIMIT_PCT = 0.10  # Max 10% of bankroll per day
MAX_ACTIVE_POSITIONS = 15  # Cap open positions

# Market quality filters
MIN_MARKET_VOLUME = 50_000  # $50K min 24h volume
MIN_LIQUIDITY = 10_000  # $10K min liquidity
MIN_YES_PRICE = 0.05  # No lottery tickets below 5 cents
MAX_YES_PRICE = 0.60  # No expensive markets above 60 cents


@dataclass
class ScanResult:
    """A single opportunity found by any strategy."""
    strategy: str
    market_id: str
    market_title: str
    token_id: str
    side: str  # BUY or SELL
    price: float
    estimated_edge: float
    size_usd: float
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "market_id": self.market_id,
            "market_title": self.market_title,
            "token_id": self.token_id,
            "side": self.side,
            "price": self.price,
            "estimated_edge": self.estimated_edge,
            "size_usd": self.size_usd,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


@dataclass
class ExecutionResult:
    """Result of an order placement attempt."""
    scan: ScanResult
    success: bool
    order_id: str = ""
    error: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVE SCANNER ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ActiveScanner:
    """Unified Polymarket scanner that runs all 3 division strategies."""

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self._agent = None  # Lazy-loaded PolymarketAgent
        self._war_room = None
        self._polymc = None
        self._plankton = None
        self.daily_bet_count = 0
        self.daily_deploy_usd = 0.0  # Track daily $ deployed
        self.daily_reset_date = datetime.now(timezone.utc).date()
        self.execution_log: list[dict[str, Any]] = []
        self.category_deployed: dict[str, float] = {}  # Per-category $ tracking
        self.bankroll: float = 0.0  # Updated from wallet + positions

    # ── Lazy loaders ────────────────────────────────────────────────────

    def _get_agent(self):
        """Lazy-load the PolymarketAgent (has CLOB client for trading)."""
        if self._agent is None:
            from agents.polymarket_agent import PolymarketAgent
            self._agent = PolymarketAgent()
        return self._agent

    def _get_war_room(self):
        if self._war_room is None:
            from strategies.polymarket_division.war_room_poly import WarRoomPoly
            self._war_room = WarRoomPoly()
        return self._war_room

    def _get_polymc(self):
        if self._polymc is None:
            from strategies.polymarket_division.polymc_agent import PolyMCAgent
            self._polymc = PolyMCAgent()
        return self._polymc

    def _get_plankton(self):
        if self._plankton is None:
            from strategies.planktonxd_prediction_harvester import PlanktonXDPredictionHarvester
            self._plankton = PlanktonXDPredictionHarvester()
        return self._plankton

    # ── Daily bet counter ───────────────────────────────────────────────

    def _check_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.daily_reset_date:
            _log.info("daily_counter_reset", previous_bets=self.daily_bet_count,
                       previous_usd=self.daily_deploy_usd)
            self.daily_bet_count = 0
            self.daily_deploy_usd = 0.0
            self.daily_reset_date = today

    # ── Risk Filters (POST-MORTEM FIXES) ────────────────────────────────

    def _classify_category(self, market_title: str) -> str:
        """Classify market into a category for diversification tracking."""
        title_lower = market_title.lower()
        for category, keywords in CATEGORY_MAP.items():
            for kw in keywords:
                if kw in title_lower:
                    return category
        return "other"

    def _check_diversification(self, scan: "ScanResult") -> str | None:
        """Check if placing this bet would violate diversification limits.
        Returns error message or None if OK."""
        if self.bankroll <= 0:
            return None  # Can't check without bankroll data

        category = self._classify_category(scan.market_title)
        current = self.category_deployed.get(category, 0.0)
        limit = self.bankroll * MAX_CATEGORY_PCT

        if current + scan.size_usd > limit:
            return (f"Category '{category}' at ${current:.0f}/${limit:.0f} "
                    f"(max {MAX_CATEGORY_PCT:.0%} of ${self.bankroll:.0f} bankroll)")
        return None

    def _check_daily_deploy_limit(self, size_usd: float) -> str | None:
        """Check if daily deployment limit would be exceeded."""
        if self.bankroll <= 0:
            return None
        limit = self.bankroll * DAILY_DEPLOY_LIMIT_PCT
        if self.daily_deploy_usd + size_usd > limit:
            return (f"Daily deploy ${self.daily_deploy_usd:.0f}+${size_usd:.0f} "
                    f"exceeds {DAILY_DEPLOY_LIMIT_PCT:.0%} limit (${limit:.0f})")
        return None

    def _update_bankroll(self) -> None:
        """Refresh bankroll from account tracker."""
        try:
            from strategies.polymarket_division.account_tracker import PolymarketAccountTracker
            tracker = PolymarketAccountTracker()
            state = tracker.get_account_state()
            self.bankroll = state.total_account_value
            # Rebuild category deployment from current positions
            self.category_deployed = {}
            for alloc_name in ("war_room", "planktonxd", "polymc"):
                alloc = getattr(state, alloc_name)
                for pos in alloc.top_positions:
                    cat = self._classify_category(pos.get("title", ""))
                    self.category_deployed[cat] = (
                        self.category_deployed.get(cat, 0.0) + pos.get("current_value", 0)
                    )
            _log.info("bankroll_updated", bankroll=self.bankroll,
                       categories=self.category_deployed)
        except Exception as e:
            _log.warning("bankroll_update_failed", error=str(e))

    def filter_opportunities(self, opportunities: list["ScanResult"]) -> list["ScanResult"]:
        """Apply all risk filters to scan results. Returns filtered list."""
        filtered = []
        for opp in opportunities:
            # Price range filter
            if opp.price < MIN_YES_PRICE:
                _log.debug("filtered_too_cheap", market=opp.market_title[:40], price=opp.price)
                continue
            if opp.price > MAX_YES_PRICE:
                _log.debug("filtered_too_expensive", market=opp.market_title[:40], price=opp.price)
                continue

            # Diversification check
            div_err = self._check_diversification(opp)
            if div_err:
                _log.info("filtered_diversification", market=opp.market_title[:40], reason=div_err)
                continue

            # Daily deploy limit
            deploy_err = self._check_daily_deploy_limit(opp.size_usd)
            if deploy_err:
                _log.info("filtered_daily_limit", market=opp.market_title[:40], reason=deploy_err)
                continue

            filtered.append(opp)

        _log.info("risk_filter_applied",
                   input=len(opportunities), output=len(filtered),
                   removed=len(opportunities) - len(filtered))
        return filtered

    # ── Strategy Scanners ───────────────────────────────────────────────

    async def scan_war_room(self) -> list[ScanResult]:
        """Scan War Room strategy — geopolitical thesis markets."""
        results = []
        try:
            wr = self._get_war_room()
            matches = await wr.scan()
            for match in matches:
                # War Room returns ThesisMarketMatch objects
                if match.adjusted_probability > 0 and match.market_price < 0.15:
                    edge = match.adjusted_probability - match.market_price
                    if edge >= MIN_EDGE_THRESHOLD:
                        # Get token_id from the market data
                        token_id = match.market.get("clobTokenIds", [""])[0] if isinstance(match.market, dict) else ""
                        results.append(ScanResult(
                            strategy="war_room",
                            market_id=str(match.market_id),
                            market_title=match.market_title,
                            token_id=token_id,
                            side="BUY",
                            price=match.market_price,
                            estimated_edge=round(edge, 4),
                            size_usd=min(MAX_POSITION_SIZE_USD, 10.0),  # Conservative for thesis bets
                            rationale=f"Thesis: {match.thesis_stage}, pressure-adjusted prob: {match.adjusted_probability:.1%}",
                        ))
            _log.info("war_room_scan_complete", opportunities=len(results))
        except Exception as e:
            _log.error("war_room_scan_failed", error=str(e))
        return results

    async def scan_polymc(self) -> list[ScanResult]:
        """Scan PolyMC strategy — top markets by volume + Monte Carlo."""
        results = []
        try:
            pmc = self._get_polymc()
            # Scan top markets
            markets = await pmc.scan_top_markets(limit=50, max_pages=3)
            for market in markets:
                # Run MC simulation for markets with good volume
                if market.volume > 50000 and 0.05 < market.best_bid < 0.40:
                    mc = pmc.run_monte_carlo(
                        prob=market.best_bid,
                        entry_price=market.best_bid,
                        bet_size=MAX_POSITION_SIZE_USD,
                    )
                    if mc.get("ev", 0) > 0 and mc.get("sharpe", 0) > 0.3:
                        edge = mc["ev"] / MAX_POSITION_SIZE_USD
                        if edge >= MIN_EDGE_THRESHOLD:
                            token_id = market.token_id if hasattr(market, "token_id") else ""
                            results.append(ScanResult(
                                strategy="polymc",
                                market_id=str(market.condition_id) if hasattr(market, "condition_id") else "",
                                market_title=market.question if hasattr(market, "question") else str(market),
                                token_id=token_id,
                                side="BUY",
                                price=market.best_bid,
                                estimated_edge=round(edge, 4),
                                size_usd=MAX_POSITION_SIZE_USD,
                                rationale=f"MC EV: ${mc['ev']:.2f}, Sharpe: {mc['sharpe']:.2f}, VaR5%: ${mc.get('var_5', 0):.2f}",
                            ))
            _log.info("polymc_scan_complete", markets_scanned=len(markets), opportunities=len(results))
        except Exception as e:
            _log.error("polymc_scan_failed", error=str(e))
        return results

    async def scan_plankton(self) -> list[ScanResult]:
        """Scan PlanktonXD strategy — deep OTM micro-arbitrage."""
        results = []
        try:
            plank = self._get_plankton()
            opps = await plank.scan_for_opportunities()
            for opp in opps:
                if hasattr(opp, "estimated_edge") and opp.estimated_edge >= MIN_EDGE_THRESHOLD:
                    token_id = opp.token_id if hasattr(opp, "token_id") else ""
                    results.append(ScanResult(
                        strategy="planktonxd",
                        market_id=opp.market_id if hasattr(opp, "market_id") else "",
                        market_title=opp.question if hasattr(opp, "question") else str(opp),
                        token_id=token_id,
                        side="BUY",
                        price=opp.price if hasattr(opp, "price") else 0.0,
                        estimated_edge=round(opp.estimated_edge, 4),
                        size_usd=min(MAX_POSITION_SIZE_USD, opp.bet_size if hasattr(opp, "bet_size") else 5.0),
                        rationale=f"PlanktonXD {opp.bet_type if hasattr(opp, 'bet_type') else 'deep_otm'}: {opp.category if hasattr(opp, 'category') else 'unknown'}",
                    ))
            _log.info("plankton_scan_complete", opportunities=len(results))
        except Exception as e:
            _log.error("plankton_scan_failed", error=str(e))
        return results

    # ── Unified Scan ────────────────────────────────────────────────────

    async def scan_all(self) -> list[ScanResult]:
        """Run all 3 strategy scanners concurrently, then apply risk filters."""
        _log.info("scan_all_start", dry_run=self.dry_run)
        self._check_daily_reset()
        self._update_bankroll()

        wr_task = asyncio.create_task(self.scan_war_room())
        pmc_task = asyncio.create_task(self.scan_polymc())
        plk_task = asyncio.create_task(self.scan_plankton())

        wr_results, pmc_results, plk_results = await asyncio.gather(
            wr_task, pmc_task, plk_task,
            return_exceptions=True,
        )

        all_results: list[ScanResult] = []
        for batch_name, batch in [("war_room", wr_results), ("polymc", pmc_results), ("plankton", plk_results)]:
            if isinstance(batch, Exception):
                _log.error("strategy_scan_exception", strategy=batch_name, error=str(batch))
            elif isinstance(batch, list):
                all_results.extend(batch)

        # Sort by edge descending
        all_results.sort(key=lambda r: r.estimated_edge, reverse=True)

        # Apply risk filters (diversification, price range, daily limits)
        all_results = self.filter_opportunities(all_results)

        _log.info("scan_all_complete",
                   total_opportunities=len(all_results),
                   war_room=len(wr_results) if isinstance(wr_results, list) else 0,
                   polymc=len(pmc_results) if isinstance(pmc_results, list) else 0,
                   plankton=len(plk_results) if isinstance(plk_results, list) else 0)

        return all_results

    # ── Execution ───────────────────────────────────────────────────────

    def execute_order(self, scan: ScanResult) -> ExecutionResult:
        """Execute a single order from a scan result."""
        self._check_daily_reset()

        if self.daily_bet_count >= MAX_DAILY_BETS:
            _log.warning("daily_bet_limit_reached", count=self.daily_bet_count)
            return ExecutionResult(scan=scan, success=False, error="Daily bet limit reached")

        if not scan.token_id:
            _log.warning("no_token_id", market=scan.market_title)
            return ExecutionResult(scan=scan, success=False, error="No token_id available")

        # Check cash reserve before deploying
        if self.bankroll > 0:
            reserve_needed = self.bankroll * CASH_RESERVE_PCT
            from strategies.polymarket_division.account_tracker import PolymarketAccountTracker
            try:
                tracker = PolymarketAccountTracker()
                wallet_bal = tracker.get_wallet_balance()
                if wallet_bal - scan.size_usd < reserve_needed:
                    _log.warning("cash_reserve_violation",
                                  wallet=wallet_bal, reserve=reserve_needed, bet=scan.size_usd)
                    return ExecutionResult(
                        scan=scan, success=False,
                        error=f"Would breach {CASH_RESERVE_PCT:.0%} cash reserve "
                              f"(wallet ${wallet_bal:.2f}, reserve ${reserve_needed:.2f})")
            except Exception as e:
                _log.warning("cash_reserve_check_failed", error=str(e))

        if self.dry_run:
            _log.info("dry_run_order",
                       strategy=scan.strategy,
                       market=scan.market_title[:50],
                       side=scan.side,
                       price=scan.price,
                       size=scan.size_usd)
            self.daily_bet_count += 1
            self.daily_deploy_usd += scan.size_usd
            cat = self._classify_category(scan.market_title)
            self.category_deployed[cat] = self.category_deployed.get(cat, 0.0) + scan.size_usd
            result = ExecutionResult(scan=scan, success=True, order_id="DRY_RUN")
            self.execution_log.append({"type": "dry_run", **scan.to_dict()})
            return result

        # LIVE execution
        agent = self._get_agent()
        order_result = agent.place_limit_order(
            token_id=scan.token_id,
            price=scan.price,
            size=scan.size_usd / scan.price if scan.price > 0 else 0,
            side=scan.side,
        )
        if order_result:
            self.daily_bet_count += 1
            self.daily_deploy_usd += scan.size_usd
            cat = self._classify_category(scan.market_title)
            self.category_deployed[cat] = self.category_deployed.get(cat, 0.0) + scan.size_usd
            order_id = order_result.get("orderID", order_result.get("id", "unknown"))
            _log.info("order_placed",
                       strategy=scan.strategy,
                       market=scan.market_title[:50],
                       order_id=order_id)
            result = ExecutionResult(scan=scan, success=True, order_id=str(order_id))
            self.execution_log.append({"type": "live", "order_id": order_id, **scan.to_dict()})
            return result
        else:
            _log.error("order_failed", market=scan.market_title[:50])
            return ExecutionResult(scan=scan, success=False, error="Order placement returned None")

    # ── Monitor Loop ────────────────────────────────────────────────────

    async def run_monitor(self, scan_interval: int = DEFAULT_SCAN_INTERVAL,
                          max_cycles: int = 0) -> None:
        """Continuous monitoring loop: scan -> (optionally execute) -> wait."""
        cycle = 0
        _log.info("monitor_start",
                   mode="live" if not self.dry_run else "dry_run",
                   scan_interval=scan_interval,
                   max_cycles=max_cycles or "infinite")

        while True:
            cycle += 1
            _log.info("monitor_cycle", cycle=cycle)

            try:
                opportunities = await self.scan_all()

                # CHECK EXITS FIRST (before placing new bets)
                try:
                    from strategies.polymarket_division.polymc_monitor import PolyMCMonitor
                    monitor = PolyMCMonitor()
                    signals = await monitor.check_exits()
                    if signals:
                        _log.info("exit_signals_found", count=len(signals))
                        for sig in signals:
                            sig_type = sig.signal_type if hasattr(sig, "signal_type") else str(sig)
                            market = sig.market if hasattr(sig, "market") else ""
                            _log.info("exit_signal",
                                       signal_type=sig_type, market=market,
                                       urgency=getattr(sig, "urgency", "unknown"))
                            # Execute exit if live mode and signal is immediate
                            if not self.dry_run and getattr(sig, "urgency", "") == "immediate":
                                _log.info("executing_exit", market=market, signal=sig_type)
                                # TODO: implement sell order execution via CLOB
                except Exception as e:
                    _log.warning("exit_check_failed", error=str(e))

                # THEN place new bets (capped at 3 per cycle, not 10)
                if not self.dry_run and opportunities:
                    for opp in opportunities[:3]:  # Max 3 per cycle (was 10)
                        if self.daily_bet_count >= MAX_DAILY_BETS:
                            break
                        self.execute_order(opp)
                elif opportunities:
                    _log.info("dry_run_opportunities", count=len(opportunities))
                    for opp in opportunities[:5]:
                        _log.info("  opportunity",
                                   strategy=opp.strategy,
                                   market=opp.market_title[:60],
                                   edge=f"{opp.estimated_edge:.1%}",
                                   price=opp.price,
                                   size=f"${opp.size_usd:.2f}")
                else:
                    _log.info("no_opportunities_found")

            except Exception as e:
                _log.error("monitor_cycle_error", cycle=cycle, error=str(e))

            if max_cycles and cycle >= max_cycles:
                _log.info("monitor_max_cycles_reached", cycles=cycle)
                break

            _log.info("monitor_sleeping", seconds=scan_interval)
            await asyncio.sleep(scan_interval)

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, opportunities: list[ScanResult]) -> str:
        """Generate a human-readable scan report."""
        lines = [
            "=" * 70,
            "  POLYMARKET ACTIVE SCANNER REPORT",
            f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE'}",
            f"  Daily bets: {self.daily_bet_count}/{MAX_DAILY_BETS}",
            "=" * 70,
            "",
        ]

        by_strategy: dict[str, list[ScanResult]] = {}
        for opp in opportunities:
            by_strategy.setdefault(opp.strategy, []).append(opp)

        for strat_name, strat_opps in by_strategy.items():
            lines.append(f"  [{strat_name.upper()}] — {len(strat_opps)} opportunities")
            lines.append("-" * 50)
            for opp in strat_opps[:10]:
                lines.append(f"    {opp.market_title[:55]}")
                lines.append(f"      Price: {opp.price:.3f}  Edge: {opp.estimated_edge:.1%}  Size: ${opp.size_usd:.2f}")
                lines.append(f"      {opp.rationale}")
                lines.append("")

        if not opportunities:
            lines.append("  No opportunities found meeting edge threshold.")
            lines.append(f"  (Min edge: {MIN_EDGE_THRESHOLD:.0%})")

        lines.append("=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Polymarket Active Scanner")
    parser.add_argument("--mode", choices=["scan", "monitor", "live"], default="scan",
                        help="scan=one-shot, monitor=continuous dry-run, live=continuous+execute")
    parser.add_argument("--interval", type=int, default=DEFAULT_SCAN_INTERVAL,
                        help=f"Seconds between scan cycles (default: {DEFAULT_SCAN_INTERVAL})")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Max monitor cycles (0=infinite)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    dry_run = args.mode != "live"
    scanner = ActiveScanner(dry_run=dry_run)

    if args.mode == "scan":
        opportunities = await scanner.scan_all()
        if args.json:
            print(json.dumps([o.to_dict() for o in opportunities], indent=2))
        else:
            print(scanner.generate_report(opportunities))
        return 0

    # monitor or live
    await scanner.run_monitor(
        scan_interval=args.interval,
        max_cycles=args.max_cycles,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))

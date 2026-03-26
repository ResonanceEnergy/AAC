#!/usr/bin/env python3
from __future__ import annotations

"""
Strategy Advisor Engine — Paper-Proof & Leaderboard
====================================================
Evaluates ALL available strategies in ADVISOR mode:
- Runs each strategy against live market data
- Generates paper recommendations with confidence scores
- Tracks paper P&L for forward-testing (proofing)
- Requires manual approval before live execution
- Ranks strategies by paper-proof performance (Sharpe, win rate, P&L)

This is the engine the owner requested:
    "122 strategies = ADVISOR ROLE + PAPER TRADING PROOFING — needs engine"

Integration:
    - Orchestrator calls evaluate_all() periodically (every 30 min)
    - Results displayed on Matrix Monitor as "STRATEGY ADVISOR LEADERBOARD"
    - All recommendations relayed to NCL BRAIN via NCCRelayClient
    - Manual approval gate — no strategy goes live without confirmation
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("StrategyAdvisorEngine")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATEGIES_CSV = PROJECT_ROOT / "50_arbitrage_strategies.csv"
ADVISOR_STATE_FILE = PROJECT_ROOT / "data" / "advisor_state.json"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdvisorRecommendation:
    """A single paper recommendation from a strategy."""
    strategy_name: str
    ticker: str
    direction: str           # "long" | "short"
    confidence: float        # 0.0 - 1.0
    entry_price: float
    target_price: float
    stop_loss: float
    thesis: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PaperPosition:
    """A tracked paper position for forward-testing."""
    strategy_name: str
    ticker: str
    direction: str
    entry_price: float
    current_price: float
    target_price: float
    stop_loss: float
    quantity: float          # notional units (normalised to $1000 per trade)
    entry_time: str
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    status: str = "open"     # "open" | "closed_target" | "closed_stop" | "closed_manual"

    def mark_to_market(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price
        multiplier = 1.0 if self.direction == "long" else -1.0
        self.unrealised_pnl = (price - self.entry_price) * self.quantity * multiplier

    def check_exits(self, price: float) -> bool:
        """Return True if position was closed by target or stop."""
        self.mark_to_market(price)
        if self.direction == "long":
            if price >= self.target_price:
                return self._close("closed_target", price)
            if price <= self.stop_loss:
                return self._close("closed_stop", price)
        else:
            if price <= self.target_price:
                return self._close("closed_target", price)
            if price >= self.stop_loss:
                return self._close("closed_stop", price)
        return False

    def _close(self, reason: str, price: float) -> bool:
        self.status = reason
        multiplier = 1.0 if self.direction == "long" else -1.0
        self.realised_pnl = (price - self.entry_price) * self.quantity * multiplier
        self.unrealised_pnl = 0.0
        return True


@dataclass
class AdvisorPerformance:
    """Rolling performance metrics for a single strategy."""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    approved_for_live: bool = False
    last_evaluated: str = ""

    def update(self, closed_positions: List[PaperPosition]) -> None:
        """Recalculate metrics from closed positions."""
        if not closed_positions:
            return
        wins = [p for p in closed_positions if p.realised_pnl > 0]
        losses = [p for p in closed_positions if p.realised_pnl <= 0]
        self.total_trades = len(closed_positions)
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.total_pnl = sum(p.realised_pnl for p in closed_positions)
        self.win_rate = self.winning_trades / max(1, self.total_trades)
        self.avg_win = sum(p.realised_pnl for p in wins) / max(1, len(wins))
        self.avg_loss = sum(p.realised_pnl for p in losses) / max(1, len(losses))

        # Running drawdown
        equity_curve = []
        running = 0.0
        for p in closed_positions:
            running += p.realised_pnl
            equity_curve.append(running)
        peak = 0.0
        dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = min(dd, eq - peak)
        self.max_drawdown = dd

        # Simplified Sharpe (mean return / std dev of returns)
        returns = [p.realised_pnl for p in closed_positions]
        if len(returns) >= 2:
            mean_r = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = variance ** 0.5
            self.sharpe_ratio = (mean_r / std_r) if std_r > 0 else 0.0
        self.last_evaluated = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY ADAPTER — loads strategies from CSV + active engines
# ═══════════════════════════════════════════════════════════════════════════


class StrategyAdapter:
    """Wraps a strategy definition (from CSV or live engine) for advisor evaluation."""

    def __init__(self, name: str, one_liner: str, category: str = "csv_advisor",
                 engine: Any = None):
        self.name = name
        self.one_liner = one_liner
        self.category = category
        self.engine = engine  # Optional live engine instance

    def evaluate(self, market_snapshot: Dict[str, Any]) -> Optional[AdvisorRecommendation]:
        """
        Run strategy logic against market data and return a paper recommendation.
        If engine is attached, delegate to it. Otherwise, use heuristic scoring.
        """
        # Live engine path — call engine's recommendation method if available
        if self.engine is not None:
            try:
                if hasattr(self.engine, "get_mandate"):
                    mandate = self.engine.get_mandate()
                    if hasattr(mandate, "direction") and hasattr(mandate, "ticker"):
                        return AdvisorRecommendation(
                            strategy_name=self.name,
                            ticker=getattr(mandate, "ticker", "SPY"),
                            direction=getattr(mandate, "direction", "long"),
                            confidence=getattr(mandate, "confidence", 0.5),
                            entry_price=market_snapshot.get("spy_price", 0),
                            target_price=market_snapshot.get("spy_price", 0) * 1.05,
                            stop_loss=market_snapshot.get("spy_price", 0) * 0.97,
                            thesis=str(mandate),
                        )
                if hasattr(self.engine, "scan"):
                    result = self.engine.scan()
                    if result:
                        return AdvisorRecommendation(
                            strategy_name=self.name,
                            ticker="SPY",
                            direction="short" if "bear" in str(result).lower() else "long",
                            confidence=0.5,
                            entry_price=market_snapshot.get("spy_price", 0),
                            target_price=market_snapshot.get("spy_price", 0) * 1.03,
                            stop_loss=market_snapshot.get("spy_price", 0) * 0.98,
                            thesis=str(result)[:200],
                        )
            except Exception as exc:
                logger.debug("Engine evaluate failed for %s: %s", self.name, exc)

        # CSV / heuristic path — generate paper signal based on snapshot
        vix = market_snapshot.get("vix", 20)
        spy_price = market_snapshot.get("spy_price", 500)
        if spy_price <= 0:
            return None

        # Simple regime-based heuristic for paper tracking
        direction = "short" if vix > 25 else "long"
        confidence = min(1.0, vix / 40) if direction == "short" else min(1.0, (40 - vix) / 40)
        if confidence < 0.3:
            return None  # Below threshold, no recommendation

        target_mult = 1.03 if direction == "long" else 0.97
        stop_mult = 0.98 if direction == "long" else 1.02

        return AdvisorRecommendation(
            strategy_name=self.name,
            ticker="SPY",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=spy_price,
            target_price=round(spy_price * target_mult, 2),
            stop_loss=round(spy_price * stop_mult, 2),
            thesis=f"[{self.category}] {self.one_liner[:120]}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════


class StrategyAdvisorEngine:
    """
    Evaluates ALL strategies in ADVISOR mode with paper-proof tracking.

    Usage:
        engine = StrategyAdvisorEngine()
        engine.load_strategies()
        recs = engine.evaluate_all({"spy_price": 510, "vix": 22})
        engine.paper_proof_cycle({"SPY": 512})
        board = engine.get_leaderboard()
    """

    def __init__(self):
        self._strategies: List[StrategyAdapter] = []
        self._paper_positions: Dict[str, List[PaperPosition]] = {}  # strategy -> positions
        self._closed_positions: Dict[str, List[PaperPosition]] = {}
        self._performance: Dict[str, AdvisorPerformance] = {}
        self._loaded = False

    # ── Loading ─────────────────────────────────────────────────────

    def load_strategies(self) -> int:
        """Load strategies from CSV + register any live engine wrappers."""
        if self._loaded:
            return len(self._strategies)

        # Load 50 arbitrage strategies from CSV
        if STRATEGIES_CSV.exists():
            try:
                with STRATEGIES_CSV.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("strategy_name", row.get("name", ""))
                        liner = row.get("one_liner", row.get("description", ""))
                        if name:
                            self._strategies.append(
                                StrategyAdapter(name=name, one_liner=liner, category="arb_csv")
                            )
                logger.info("Loaded %d CSV strategies from %s", len(self._strategies), STRATEGIES_CSV.name)
            except Exception as exc:
                logger.warning("Failed to load CSV strategies: %s", exc)

        # Register the 7 active war-room strategies (engines loaded lazily by integrator)
        active_7 = [
            ("War Room Engine", "Crisis options + Monte Carlo milestones"),
            ("Storm Lifeboat Capital", "Gold-oil-silver see-saw rotation"),
            ("Matrix Maximizer", "Geopolitical bear market options engine"),
            ("Exploitation Matrix", "8 investment verticals for crisis plays"),
            ("Polymarket BlackSwan", "Prediction market probability harvester"),
            ("BlackSwan Authority", "4 expert authority feeds consensus"),
            ("Zero DTE Gamma", "0DTE options gamma scalping engine"),
        ]
        for name, liner in active_7:
            self._strategies.append(StrategyAdapter(name=name, one_liner=liner, category="active_7"))

        # Init performance tracking
        for s in self._strategies:
            if s.name not in self._performance:
                self._performance[s.name] = AdvisorPerformance(strategy_name=s.name)
            if s.name not in self._paper_positions:
                self._paper_positions[s.name] = []
            if s.name not in self._closed_positions:
                self._closed_positions[s.name] = []

        self._loaded = True
        logger.info("Strategy Advisor Engine loaded %d total strategies", len(self._strategies))
        return len(self._strategies)

    # ── Evaluation ──────────────────────────────────────────────────

    def evaluate_all(self, market_snapshot: Dict[str, Any]) -> List[AdvisorRecommendation]:
        """
        Run all strategies, return ranked recommendations.
        Opens paper positions for tracking.
        """
        if not self._loaded:
            self.load_strategies()

        recommendations: List[AdvisorRecommendation] = []
        for strategy in self._strategies:
            try:
                rec = strategy.evaluate(market_snapshot)
                if rec and rec.confidence >= 0.3:
                    recommendations.append(rec)
                    # Open paper position for tracking
                    if rec.entry_price > 0:
                        notional = 1000.0  # normalised $1000 per paper trade
                        qty = notional / rec.entry_price
                        pos = PaperPosition(
                            strategy_name=rec.strategy_name,
                            ticker=rec.ticker,
                            direction=rec.direction,
                            entry_price=rec.entry_price,
                            current_price=rec.entry_price,
                            target_price=rec.target_price,
                            stop_loss=rec.stop_loss,
                            quantity=round(qty, 4),
                            entry_time=rec.timestamp,
                        )
                        self._paper_positions[rec.strategy_name].append(pos)
            except Exception as exc:
                logger.debug("Strategy %s evaluation error: %s", strategy.name, exc)

        # Sort by confidence descending
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        logger.info("Evaluated %d strategies, %d recommendations", len(self._strategies), len(recommendations))
        return recommendations

    # ── Paper Proof Cycle ───────────────────────────────────────────

    def paper_proof_cycle(self, live_prices: Dict[str, float]) -> Dict[str, int]:
        """
        Update all paper positions with live prices, close those hitting targets/stops.
        Returns {closed: N, still_open: M}.
        """
        closed_count = 0
        open_count = 0

        for strat_name, positions in self._paper_positions.items():
            still_open = []
            for pos in positions:
                price = live_prices.get(pos.ticker, pos.current_price)
                if pos.check_exits(price):
                    self._closed_positions.setdefault(strat_name, []).append(pos)
                    closed_count += 1
                else:
                    pos.mark_to_market(price)
                    still_open.append(pos)
                    open_count += 1
            self._paper_positions[strat_name] = still_open

        # Recalculate performance
        for strat_name, closed in self._closed_positions.items():
            if strat_name in self._performance:
                self._performance[strat_name].update(closed)

        return {"closed": closed_count, "still_open": open_count}

    # ── Leaderboard ─────────────────────────────────────────────────

    def get_leaderboard(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Rank strategies by paper-proof performance (Sharpe + win rate + total P&L)."""
        board = []
        for name, perf in self._performance.items():
            if perf.total_trades == 0:
                continue
            # Composite score: 40% Sharpe + 30% win_rate + 30% normalised P&L
            pnl_norm = max(-1.0, min(1.0, perf.total_pnl / 1000.0))
            composite = 0.4 * perf.sharpe_ratio + 0.3 * perf.win_rate + 0.3 * pnl_norm
            board.append({
                "strategy": name,
                "trades": perf.total_trades,
                "win_rate": round(perf.win_rate, 3),
                "total_pnl": round(perf.total_pnl, 2),
                "sharpe": round(perf.sharpe_ratio, 3),
                "max_dd": round(perf.max_drawdown, 2),
                "composite_score": round(composite, 4),
                "approved_live": perf.approved_for_live,
            })

        board.sort(key=lambda x: x["composite_score"], reverse=True)
        return board[:top_n]

    # ── Approval Gate ───────────────────────────────────────────────

    def approve_for_live(self, strategy_name: str) -> bool:
        """
        Promote a strategy from paper-proof to live execution.
        Requires minimum 10 trades and positive Sharpe.
        """
        perf = self._performance.get(strategy_name)
        if not perf:
            logger.warning("Strategy %s not found in performance tracker", strategy_name)
            return False
        if perf.total_trades < 10:
            logger.warning("Strategy %s has only %d trades (min 10)", strategy_name, perf.total_trades)
            return False
        if perf.sharpe_ratio <= 0:
            logger.warning("Strategy %s has negative Sharpe (%.3f)", strategy_name, perf.sharpe_ratio)
            return False
        perf.approved_for_live = True
        logger.info("Strategy %s APPROVED for live trading", strategy_name)
        return True

    def revoke_live(self, strategy_name: str) -> bool:
        """Revoke live trading approval."""
        perf = self._performance.get(strategy_name)
        if perf:
            perf.approved_for_live = False
            return True
        return False

    # ── State Persistence ───────────────────────────────────────────

    def save_state(self) -> None:
        """Persist advisor state to disk."""
        ADVISOR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "performance": {k: asdict(v) for k, v in self._performance.items()},
            "open_positions": {
                k: [asdict(p) for p in v] for k, v in self._paper_positions.items() if v
            },
            "closed_count": sum(len(v) for v in self._closed_positions.values()),
        }
        ADVISOR_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info("Advisor state saved to %s", ADVISOR_STATE_FILE)

    def load_state(self) -> bool:
        """Restore advisor state from disk."""
        if not ADVISOR_STATE_FILE.exists():
            return False
        try:
            state = json.loads(ADVISOR_STATE_FILE.read_text(encoding="utf-8"))
            for name, perf_dict in state.get("performance", {}).items():
                self._performance[name] = AdvisorPerformance(**perf_dict)
            logger.info("Advisor state restored (%d strategies)", len(self._performance))
            return True
        except Exception as exc:
            logger.warning("Failed to restore advisor state: %s", exc)
            return False

    # ── Summary for Matrix Monitor ──────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return summary data for dashboard display."""
        total_open = sum(len(v) for v in self._paper_positions.values())
        total_closed = sum(len(v) for v in self._closed_positions.values())
        approved = [n for n, p in self._performance.items() if p.approved_for_live]
        return {
            "total_strategies": len(self._strategies),
            "total_open_positions": total_open,
            "total_closed_positions": total_closed,
            "approved_for_live": approved,
            "approved_count": len(approved),
            "leaderboard_top5": self.get_leaderboard(top_n=5),
        }

    # ── NCL Relay Payload ───────────────────────────────────────────

    def get_relay_payload(self) -> Dict[str, Any]:
        """Build payload for NCL BRAIN relay."""
        return {
            "engine": "strategy_advisor",
            "summary": self.get_summary(),
            "leaderboard": self.get_leaderboard(top_n=10),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL ACCESSOR
# ═══════════════════════════════════════════════════════════════════════════

_advisor_engine: Optional[StrategyAdvisorEngine] = None


def get_strategy_advisor_engine() -> StrategyAdvisorEngine:
    """Get or create the singleton StrategyAdvisorEngine."""
    global _advisor_engine
    if _advisor_engine is None:
        _advisor_engine = StrategyAdvisorEngine()
    return _advisor_engine

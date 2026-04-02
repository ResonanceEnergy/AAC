"""
MATRIX MAXIMIZER — Chatbot Interface
=========================================
Natural language interface for MATRIX MAXIMIZER:
  - Query system state, positions, P&L
  - Run cycles on demand
  - Ask about risk, mandates, picks
  - Conversation history for context
  - Pluggable into Telegram, CLI, or web

Commands:
  status     — System status snapshot
  positions  — Show open positions
  pnl        — P&L summary
  picks      — Latest scanner picks
  run        — Run a full cycle
  risk       — Risk assessment
  mandate    — Current mandate
  regime     — Current regime context
  schedule   — Scheduler status
  backtest   — Run a quick backtest
  report     — Generate daily report
  help       — Show commands
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Single conversation message."""
    role: str       # "user" or "assistant"
    content: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class ChatContext:
    """Shared state for chatbot commands."""
    runner: Any = None           # MatrixMaximizer runner
    execution: Any = None        # ExecutionEngine
    dashboard: Any = None        # MatrixDashboard
    alerts: Any = None           # AlertManager
    scheduler: Any = None        # MatrixScheduler
    backtester: Any = None       # MatrixBacktester
    data_feeds: Any = None       # DataFeedManager
    intelligence: Any = None     # IntelligenceEngine
    advanced: Any = None         # AdvancedStrategyEngine
    last_cycle: Optional[Dict[str, Any]] = None


class MatrixChatbot:
    """Natural language interface for MATRIX MAXIMIZER.

    Usage:
        ctx = ChatContext(runner=runner, execution=engine, ...)
        bot = MatrixChatbot(ctx)
        response = bot.handle("show me my positions")
        response = bot.handle("run a cycle")
        response = bot.handle("what's the risk level?")
    """

    def __init__(self, context: ChatContext) -> None:
        self.ctx = context
        self._history: List[ChatMessage] = []
        self._commands: Dict[str, Callable[[str], str]] = {
            "status": self._cmd_status,
            "positions": self._cmd_positions,
            "pos": self._cmd_positions,
            "pnl": self._cmd_pnl,
            "picks": self._cmd_picks,
            "run": self._cmd_run,
            "cycle": self._cmd_run,
            "risk": self._cmd_risk,
            "mandate": self._cmd_mandate,
            "regime": self._cmd_regime,
            "schedule": self._cmd_schedule,
            "backtest": self._cmd_backtest,
            "report": self._cmd_report,
            "advanced": self._cmd_advanced,
            "spreads": self._cmd_advanced,
            "help": self._cmd_help,
        }

        # Pattern matchers for natural language
        self._patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"(show|what|list).*position", re.I), "positions"),
            (re.compile(r"(p&l|pnl|profit|loss|how.*(doing|money))", re.I), "pnl"),
            (re.compile(r"(run|execute|start).*cycle", re.I), "run"),
            (re.compile(r"(scan|pick|recommend|what.*buy)", re.I), "picks"),
            (re.compile(r"(risk|danger|safe|circuit)", re.I), "risk"),
            (re.compile(r"(mandate|conviction|should.*trade)", re.I), "mandate"),
            (re.compile(r"(regime|macro|market.*state)", re.I), "regime"),
            (re.compile(r"(schedule|when|next.*run|cron)", re.I), "schedule"),
            (re.compile(r"(backtest|hist|simulate|past)", re.I), "backtest"),
            (re.compile(r"(report|summary|daily|dashboard)", re.I), "report"),
            (re.compile(r"(status|overview|how.*system)", re.I), "status"),
            (re.compile(r"(spread|collar|straddle|iron.*condor|butterfly)", re.I), "advanced"),
            (re.compile(r"(help|command|what.*can)", re.I), "help"),
        ]

    def handle(self, user_input: str) -> str:
        """Process user input and return response.

        Resolves commands via:
          1. Exact command match
          2. Natural language pattern matching
          3. Fallback to help
        """
        self._history.append(ChatMessage(role="user", content=user_input))

        # Clean input
        text = user_input.strip().lower()

        # 1. Exact command match
        first_word = text.split()[0] if text else ""
        if first_word in self._commands:
            response = self._commands[first_word](text)
        else:
            # 2. Pattern matching
            matched_cmd = None
            for pattern, cmd in self._patterns:
                if pattern.search(text):
                    matched_cmd = cmd
                    break

            if matched_cmd:
                response = self._commands[matched_cmd](text)
            else:
                response = self._cmd_fallback(text)

        self._history.append(ChatMessage(role="assistant", content=response))
        return response

    def get_history(self) -> List[ChatMessage]:
        return self._history

    def clear_history(self) -> None:
        self._history.clear()

    # ═══════════════════════════════════════════════════════════════════════
    # COMMANDS
    # ═══════════════════════════════════════════════════════════════════════

    def _cmd_status(self, text: str) -> str:
        """System status overview."""
        lines = ["🔧 MATRIX MAXIMIZER STATUS\n"]

        # Execution engine
        if self.ctx.execution:
            snap = self.ctx.execution.get_account_snapshot()
            lines.extend([
                f"  Mode: {snap.mode}",
                f"  Equity: ${snap.total_value:,.2f}",
                f"  Positions: {snap.positions_count}",
                f"  Exposure: ${snap.put_exposure:,.2f}",
                f"  P&L: ${snap.unrealized_pnl + snap.realized_pnl:+,.2f}",
            ])
        else:
            lines.append("  Execution: not connected")

        # Scheduler
        if self.ctx.scheduler:
            status = self.ctx.scheduler.get_status()
            lines.extend([
                f"\n  Scheduler: {'RUNNING' if status['running'] else 'STOPPED'}",
                f"  Market: {'OPEN' if status['is_market_hours'] else 'CLOSED'}",
            ])

        # Last cycle
        if self.ctx.last_cycle:
            cb = self.ctx.last_cycle.get("circuit_breaker", "?")
            mandate = self.ctx.last_cycle.get("mandate", {})
            lines.extend([
                f"\n  Last Cycle:",
                f"    Circuit Breaker: {cb}",
                f"    Mandate: {mandate.get('level', '?')} ({mandate.get('conviction', 0):.0%})",
            ])

        return "\n".join(lines)

    def _cmd_positions(self, text: str) -> str:
        """Show positions."""
        if not self.ctx.execution:
            return "⚠️ Execution engine not connected"

        positions = self.ctx.execution.get_positions(include_closed="all" in text or "closed" in text)
        if not positions:
            return "📭 No positions tracked"

        return self.ctx.execution.print_positions()

    def _cmd_pnl(self, text: str) -> str:
        """P&L summary."""
        if not self.ctx.execution:
            return "⚠️ Execution engine not connected"

        snap = self.ctx.execution.get_account_snapshot()
        return (
            f"💰 P&L SUMMARY\n"
            f"  Unrealized: ${snap.unrealized_pnl:+,.2f}\n"
            f"  Realized:   ${snap.realized_pnl:+,.2f}\n"
            f"  Total:      ${snap.unrealized_pnl + snap.realized_pnl:+,.2f}\n"
            f"  Equity:     ${snap.total_value:,.2f}\n"
            f"  Put Exp:    ${snap.put_exposure:,.2f}"
        )

    def _cmd_picks(self, text: str) -> str:
        """Show latest picks."""
        if self.ctx.last_cycle and "picks" in self.ctx.last_cycle:
            picks = self.ctx.last_cycle["picks"]
            if not picks:
                return "📝 No picks from last cycle"

            lines = ["🎯 LATEST PICKS:\n"]
            for i, p in enumerate(picks[:10], 1):
                lines.append(
                    f"  {i}. {p.get('ticker', ''):<6s} ${p.get('strike', 0):.0f}P  "
                    f"${p.get('premium', 0):.2f}  Δ={p.get('delta', 0):.2f}  "
                    f"Score={p.get('score', 0):.0f}"
                )
            return "\n".join(lines)

        return "📝 No cycle run yet — try 'run cycle'"

    def _cmd_run(self, text: str) -> str:
        """Run a full cycle."""
        if not self.ctx.runner:
            return "⚠️ Runner not configured"

        try:
            result = self.ctx.runner.run_full_cycle(
                positions=[], prices={}, daily_pnl=0, cumulative_pnl=0,
            )
            self.ctx.last_cycle = result if isinstance(result, dict) else {"raw": str(result)}
            return "✅ Cycle completed!\n\n" + self._cmd_status(text)
        except Exception as exc:
            return f"❌ Cycle failed: {exc}"

    def _cmd_risk(self, text: str) -> str:
        """Risk assessment."""
        if self.ctx.last_cycle:
            risk = self.ctx.last_cycle.get("risk", {})
            cb = self.ctx.last_cycle.get("circuit_breaker", "?")

            lines = [
                f"🛡️ RISK ASSESSMENT\n",
                f"  Circuit Breaker: {cb}",
            ]

            checks = self.ctx.last_cycle.get("risk_checks", [])
            if checks:
                lines.append("  Checks:")
                for check in checks:
                    status = "✅" if check.get("passed") else "❌"
                    lines.append(f"    {status} {check.get('name', '?')}")

            return "\n".join(lines)

        return "⚠️ No cycle data — run a cycle first"

    def _cmd_mandate(self, text: str) -> str:
        """Current mandate."""
        if self.ctx.last_cycle:
            mandate = self.ctx.last_cycle.get("mandate", {})
            return (
                f"📋 MANDATE\n"
                f"  Level: {mandate.get('level', 'OBSERVE')}\n"
                f"  Conviction: {mandate.get('conviction', 0):.0%}\n"
                f"  Thesis: {mandate.get('thesis', 'N/A')}"
            )
        return "⚠️ No cycle data"

    def _cmd_regime(self, text: str) -> str:
        """Current regime context."""
        if self.ctx.last_cycle:
            weights = self.ctx.last_cycle.get("scenario_weights", {})
            regime = self.ctx.last_cycle.get("regime", {})
            return (
                f"🌍 REGIME CONTEXT\n"
                f"  Scenarios: Base={weights.get('base', 0):.0%} "
                f"Bear={weights.get('bear', 0):.0%} "
                f"Bull={weights.get('bull', 0):.0%}\n"
                f"  Regime: {regime.get('primary', 'unknown')}\n"
                f"  VIX: {regime.get('vix', '?')} | Oil: ${regime.get('oil', '?')}"
            )
        return "⚠️ No cycle data"

    def _cmd_schedule(self, text: str) -> str:
        """Scheduler status."""
        if not self.ctx.scheduler:
            return "⚠️ Scheduler not configured"
        return self.ctx.scheduler.print_schedule()

    def _cmd_backtest(self, text: str) -> str:
        """Run a quick backtest."""
        if not self.ctx.backtester:
            return "⚠️ Backtester not configured"

        # Parse days from text (default 30)
        days = 30
        match = re.search(r"(\d+)\s*(day|d)", text)
        if match:
            days = int(match.group(1))

        scenarios = self.ctx.backtester.generate_historical_scenarios(days=days)
        result = self.ctx.backtester.backtest(scenarios)
        return f"📊 BACKTEST ({days} days)\n\n{result.print_card()}"

    def _cmd_report(self, text: str) -> str:
        """Generate and save daily report."""
        if not self.ctx.dashboard:
            return "⚠️ Dashboard not configured"

        positions = None
        account = None
        if self.ctx.execution:
            positions = self.ctx.execution.get_positions()
            account = self.ctx.execution.get_account_snapshot()

        report = self.ctx.dashboard.daily_report(
            cycle_output=self.ctx.last_cycle,
            positions=positions,
            account=account,
        )

        path = self.ctx.dashboard.save_report(report, "daily")
        return f"📄 Report generated and saved to {path}\n\n{report}"

    def _cmd_advanced(self, text: str) -> str:
        """Show advanced strategy recommendations."""
        if not self.ctx.advanced:
            return "⚠️ Advanced strategy engine not configured"

        # Default to SPY
        ticker = "SPY"
        match = re.search(r"\b([A-Z]{2,5})\b", text.upper())
        if match and match.group(1) not in ("SHOW", "THE", "FOR"):
            ticker = match.group(1)

        from strategies.matrix_maximizer.core import ASSET_VOLATILITIES, DEFAULT_PRICES, Asset
        spot = DEFAULT_PRICES.get(Asset(ticker), DEFAULT_PRICES.get(Asset.SPY, 560))
        sigma = ASSET_VOLATILITIES.get(Asset(ticker), 0.25)

        strategies = self.ctx.advanced.recommend_strategies(ticker, spot, sigma, 30)

        if not strategies:
            return f"📝 No advanced strategies found for {ticker}"

        lines = [f"🎯 ADVANCED STRATEGIES for {ticker}\n"]
        for s in strategies[:5]:
            lines.append(s.print_card())
            lines.append("")

        return "\n".join(lines)

    def _cmd_help(self, text: str) -> str:
        """Show available commands."""
        return (
            "🤖 MATRIX MAXIMIZER CHATBOT\n\n"
            "Commands:\n"
            "  status     — System overview (equity, positions, mode)\n"
            "  positions  — Show open positions (add 'all' for closed too)\n"
            "  pnl        — P&L summary (unrealized + realized)\n"
            "  picks      — Latest scanner recommendations\n"
            "  run        — Run a full analysis cycle\n"
            "  risk       — Risk assessment & circuit breaker\n"
            "  mandate    — Current trading mandate\n"
            "  regime     — Market regime context\n"
            "  schedule   — Scheduler status & upcoming slots\n"
            "  backtest   — Run a quick backtest (e.g. 'backtest 60 days')\n"
            "  report     — Generate daily report\n"
            "  spreads    — Advanced multi-leg strategy recommendations\n"
            "  help       — This help message\n\n"
            "You can also ask naturally:\n"
            '  "How are my positions doing?"\n'
            '  "What should I buy?"\n'
            '  "Run a cycle"\n'
            '  "Show me spreads for QQQ"'
        )

    def _cmd_fallback(self, text: str) -> str:
        """Fallback for unrecognized input."""
        return (
            f"🤔 I didn't understand: \"{text}\"\n\n"
            "Try 'help' to see available commands, or ask naturally like:\n"
            '  "show positions" / "run a cycle" / "what\'s the risk?"'
        )

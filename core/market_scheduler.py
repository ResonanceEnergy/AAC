from __future__ import annotations

"""market_scheduler.py — Sprint 7: Automation & Scheduling.

Runs four recurring tasks unattended:

  * Signal scan      — every 15 min, NYSE market hours only  (7.1)
  * Roll check       — once daily at market open (9:30 ET)    (7.2)
  * P&L snapshot     — once daily at market close (16:00 ET)  (7.3)
  * Health check     — every 5 min, always                    (7.4)

Graceful restart (7.5): ``run_forever()`` re-enters the loop up to
``max_restarts`` times on unexpected crashes.

Usage::

    from core.market_scheduler import MarketScheduler
    sched = MarketScheduler()
    sched.run()           # blocks until stop() called
    # — or —
    sched.run_forever()   # auto-restarts on crash
"""

import os
import threading
import time as _time
from datetime import date, datetime, time
from typing import Any, Callable

import structlog

try:
    from zoneinfo import ZoneInfo  # Python 3.9+

    _ET: Any = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    import datetime as _dt

    _ET = _dt.timezone(_dt.timedelta(hours=-5))  # EST fallback (no DST)

_log = structlog.get_logger(__name__)

# ── schedule constants ─────────────────────────────────────────────────────

_SCAN_INTERVAL: int = 900  # 15 min
_HEALTH_INTERVAL: int = 300  # 5 min
_TICK: int = 30  # scheduler sleep granularity (seconds)

# Market open / close windows (ET wall-clock)
_OPEN_START = time(9, 30)
_OPEN_END = time(9, 40)  # 10-min window to catch the trigger
_CLOSE_START = time(16, 0)
_CLOSE_END = time(16, 10)
_MARKET_START = time(9, 30)
_MARKET_END = time(16, 0)

# AI briefing windows (Phase 4 — local LLM agent swarm)
_MORNING_BRIEF_START = time(8, 0)
_MORNING_BRIEF_END = time(8, 30)
_EVENING_BRIEF_START = time(16, 30)
_EVENING_BRIEF_END = time(17, 0)


# ── MarketScheduler ────────────────────────────────────────────────────────


class MarketScheduler:
    """Runs scheduled trading-day tasks unattended.

    Args:
        scan_interval:   Seconds between signal scans (default 900 = 15 min).
        health_interval: Seconds between health checks (default 300 = 5 min).
        tick:            Inner sleep granularity in seconds (default 30).
        paper:           Forward to PositionTracker (paper=True uses paper account).
        auto_execute:    If True, execute approved signals after each scan.
                         Reads DRY_RUN / PAPER_TRADING env vars for trade mode.
        auto_trader:     Inject a pre-configured AutoTrader (overrides auto_execute).
    """

    def __init__(
        self,
        scan_interval: int = _SCAN_INTERVAL,
        health_interval: int = _HEALTH_INTERVAL,
        tick: int = _TICK,
        paper: bool | None = None,
        auto_execute: bool = False,
        auto_trader: Any | None = None,
        enable_ai_briefings: bool = True,
    ) -> None:
        self._scan_interval = scan_interval
        self._health_interval = health_interval
        self._tick = tick
        if paper is None:
            paper = os.getenv("PAPER_TRADING", "false").lower() == "true"
        self._paper = paper

        self._stop = threading.Event()
        self._last_run: dict[str, float] = {}  # task → monotonic timestamp
        self._last_roll_date: date | None = None
        self._last_pnl_date: date | None = None
        self._last_cot_date: date | None = None
        self._last_morning_brief_date: date | None = None
        self._last_evening_brief_date: date | None = None
        self._enable_ai_briefings = enable_ai_briefings
        self.restart_count: int = 0

        # Shared PositionTracker — one IBKR connection reused by roll check,
        # P&L snapshot, and AutoTrader exposure checks (Sprint 9).
        from TradingExecution.position_tracker import PositionTracker  # noqa: PLC0415
        self._position_tracker: Any = PositionTracker(paper=self._paper)

        # Shared PnLTracker + DailyLossGuard — guard reads what pnl_snapshot writes (Sprint 10).
        from CentralAccounting.pnl_tracker import PnLTracker  # noqa: PLC0415
        from strategies.daily_loss_guard import DailyLossGuard  # noqa: PLC0415
        self._pnl_tracker: Any = PnLTracker()
        self._daily_loss_guard: Any = DailyLossGuard()

        # Shared OrderMonitor — cancels stale submitted orders (Sprint 14).
        from core.order_monitor import OrderMonitor, PendingOrderRegistry  # noqa: PLC0415
        self._order_registry: Any = PendingOrderRegistry()
        self._order_monitor: Any = OrderMonitor(self._order_registry)

        # Shared SignalJournal — logs every approved signal for outcome tracking (Sprint 15).
        from strategies.signal_journal import SignalJournal  # noqa: PLC0415
        self._signal_journal: Any = SignalJournal()

        # Shared RegimeMonitor — persists composite-score regime state (Sprint 16).
        from strategies.regime_monitor import RegimeMonitor  # noqa: PLC0415
        self._regime_monitor: Any = RegimeMonitor()

        # Shared PositionReconciler — compares internal snapshot vs live IBKR (Sprint 17).
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415
        self._position_reconciler: Any = PositionReconciler()

        # Shared DrawdownCircuitBreaker — multi-day drawdown guard (Sprint 18).
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker  # noqa: PLC0415
        self._drawdown_circuit_breaker: Any = DrawdownCircuitBreaker()

        # Live account equity feed — Sprint 20.  IBKR NetLiquidation with env-var fallback.
        from shared.account_value_feed import AccountValueFeed  # noqa: PLC0415
        self._account_value_feed: Any = AccountValueFeed(paper=self._paper)

        # Real-time alerter — Sprint 21.  Sends Telegram notifications on key events.
        from shared.alerter import Alerter  # noqa: PLC0415
        self._alerter: Any = Alerter()

        # Persistent execution throttle — Sprint 23.  Survives process restarts.
        from core.execution_throttle import ExecutionThrottle  # noqa: PLC0415
        self._execution_throttle: Any = ExecutionThrottle()

        # Vol-target position sizer — Sprint 25.  Scales kelly fraction by realized vol.
        from strategies.vol_target_sizer import VolTargetSizer  # noqa: PLC0415
        self._vol_sizer: Any = VolTargetSizer()

        # Correlation contagion gate — Sprint 26.  Blocks non-compulsory signals in contagion regime.
        from strategies.correlation_tracker import CorrelationTracker  # noqa: PLC0415
        self._correlation_tracker: Any = CorrelationTracker()

        # Auto-execution wiring (Sprint 8, extended Sprint 9+10+15)
        if auto_trader is not None:
            self._auto_trader: Any | None = auto_trader
        elif auto_execute:
            from core.auto_trader import AutoTrader  # noqa: PLC0415
            self._auto_trader = AutoTrader(
                paper=self._paper,
                position_tracker=self._position_tracker,
                drawdown_circuit_breaker=self._drawdown_circuit_breaker,
                daily_loss_guard=self._daily_loss_guard,
                pnl_tracker=self._pnl_tracker,
                order_monitor=self._order_monitor,
                signal_journal=self._signal_journal,
                alerter=self._alerter,
                execution_throttle=self._execution_throttle,
                vol_sizer=self._vol_sizer,
                correlation_tracker=self._correlation_tracker,
            )
        else:
            self._auto_trader = None

    # ── stop ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the scheduler to stop after the current tick."""
        self._stop.set()

    # ── market-hour helpers ───────────────────────────────────────────────────

    @staticmethod
    def is_trading_day(now_et: datetime) -> bool:
        """True for Mon–Fri (no holiday calendar)."""
        return now_et.weekday() < 5

    @staticmethod
    def is_market_hours(now_et: datetime) -> bool:
        """True during NYSE regular hours: 9:30–16:00 ET, Mon–Fri."""
        if now_et.weekday() >= 5:
            return False
        t = now_et.time()
        return _MARKET_START <= t < _MARKET_END

    @staticmethod
    def is_market_open_window(now_et: datetime) -> bool:
        """True during the 10-min window after NYSE open (9:30–9:40 ET)."""
        if now_et.weekday() >= 5:
            return False
        t = now_et.time()
        return _OPEN_START <= t < _OPEN_END

    @staticmethod
    def is_market_close_window(now_et: datetime) -> bool:
        """True during the 10-min window after NYSE close (16:00–16:10 ET)."""
        if now_et.weekday() >= 5:
            return False
        t = now_et.time()
        return _CLOSE_START <= t < _CLOSE_END

    @staticmethod
    def is_morning_brief_window(now_et: datetime) -> bool:
        """True during the pre-market AI briefing window (8:00–8:30 ET, Mon–Fri)."""
        if now_et.weekday() >= 5:
            return False
        t = now_et.time()
        return _MORNING_BRIEF_START <= t < _MORNING_BRIEF_END

    @staticmethod
    def is_evening_brief_window(now_et: datetime) -> bool:
        """True during the post-close AI prep window (16:30–17:00 ET, Mon–Fri)."""
        if now_et.weekday() >= 5:
            return False
        t = now_et.time()
        return _EVENING_BRIEF_START <= t < _EVENING_BRIEF_END

    # ── task implementations ──────────────────────────────────────────────────

    def run_signal_scan(self) -> list:
        """Run combined War Room + Vol Premium signal scan.

        If an AutoTrader is configured (via ``auto_execute=True`` or
        ``auto_trader=...``), approved signals are executed immediately after
        the scan.  Stop-loss and take-profit checks are also run here — any
        triggered positions are converted to STOP_CLOSE signals and routed
        through AutoTrader immediately (bypassing throttle/confidence).

        Returns the raw strategy signal list (stop signals are not included
        in the return value — they are dispatched inline).
        """
        from strategies.signal_aggregator import get_combined_signals  # noqa: PLC0415
        from strategies.stop_manager import StopManager  # noqa: PLC0415
        from strategies.stop_signals import stop_decisions_to_signals  # noqa: PLC0415

        signals = get_combined_signals(use_calibration=True)  # Sprint 22: use calibrated weights when ≥5 resolved signals available
        _log.info("signal_scan_complete", n_signals=len(signals))

        # Regime change detection — record composite score, apply confidence filter on transition.
        try:
            composite_score = self._get_composite_score()
            transition = self._regime_monitor.record(composite_score)
            if transition:
                from strategies.regime_alerts import apply_regime_filter, format_alert  # noqa: PLC0415
                alert_msg = format_alert(transition)
                _log.warning("regime_change", alert=alert_msg)
                signals = apply_regime_filter(signals, transition)
        except Exception as exc:
            _log.warning("regime_check_failed", error=str(exc))

        # Correlation snapshot — Sprint 26.  Best-effort; never blocks signal scan.
        try:
            import yfinance as yf  # noqa: PLC0415
            _ETF_UNIVERSE = ["SPY", "TLT", "GLD", "IWM", "HYG"]
            prices_df = yf.download(_ETF_UNIVERSE, period="2y", progress=False, auto_adjust=True)["Close"].dropna()
            if len(prices_df) >= 30:
                self._correlation_tracker.update(prices_df)
        except Exception as exc:
            _log.debug("correlation_update_skipped", error=str(exc))

        if self._auto_trader is not None and signals:
            try:
                summary = self._auto_trader.run_once(signals)
                _log.info(
                    "auto_trader_summary",
                    received=summary.signals_received,
                    approved=summary.signals_approved,
                    executed=summary.signals_executed,
                    dry_run=summary.dry_run,
                )
            except Exception as exc:
                _log.error("auto_trader_run_once_failed", error=str(exc))

        # Stop-loss / take-profit enforcement — runs every scan cycle.
        if self._auto_trader is not None:
            try:
                positions = self._fetch_positions()
                acct_value = getattr(self._auto_trader, "account_value_usd", 0.0)
                stop_decisions = StopManager().scan(positions, acct_value)
                if stop_decisions:
                    stop_sigs = stop_decisions_to_signals(
                        stop_decisions, positions, acct_value
                    )
                    if stop_sigs:
                        summary = self._auto_trader.run_once(stop_sigs)
                        _log.info(
                            "stop_signals_executed",
                            n_signals=len(stop_sigs),
                            executed=summary.signals_executed,
                            dry_run=summary.dry_run,
                        )
            except Exception as exc:
                _log.error("stop_signal_execution_failed", error=str(exc))

        return signals

    def run_roll_check(self) -> list:
        """Check positions for roll/close urgency.  Returns urgent RollDecisions.

        When an AutoTrader is configured, urgent ROLL/CLOSE/DEAD_PUT decisions
        are converted to TradeSignals and executed automatically (Sprint 9).
        """
        from strategies.roll_manager import RollManager  # noqa: PLC0415
        from strategies.roll_signals import roll_decisions_to_signals  # noqa: PLC0415

        positions = self._fetch_positions()

        mgr = RollManager()
        decisions = mgr.urgent_only(positions)
        _log.info(
            "roll_check_complete",
            n_urgent=len(decisions),
            symbols=[d.symbol for d in decisions],
        )

        # Route urgent decisions through AutoTrader as CLOSE signals.
        if self._auto_trader is not None and decisions:
            try:
                acct_value = getattr(self._auto_trader, "account_value_usd", 0.0)
                roll_sigs = roll_decisions_to_signals(
                    decisions, positions, acct_value
                )
                if roll_sigs:
                    summary = self._auto_trader.run_once(roll_sigs)
                    _log.info(
                        "roll_signals_executed",
                        n_signals=len(roll_sigs),
                        executed=summary.signals_executed,
                        dry_run=summary.dry_run,
                    )
            except Exception as exc:
                _log.error("roll_signal_execution_failed", error=str(exc))

        return decisions

    def run_pnl_snapshot(self) -> dict[str, Any]:
        """Take end-of-day P&L snapshot, then generate EOD brief.  Returns the report dict."""
        try:
            account_value = self._account_value_feed.get()
            account_source = self._account_value_feed.get_source()
        except Exception as exc:
            _log.warning("account_value_feed_failed", error=str(exc))
            account_value = float(os.environ.get("ACCOUNT_VALUE_USD", "50000"))
            account_source = "default"
        positions = self._fetch_positions()

        report: dict[str, Any] = {}
        try:
            report = self._pnl_tracker.take_snapshot(positions, account_value)
            _log.info("pnl_snapshot_complete", n_positions=len(positions))
        except Exception as exc:
            _log.error("pnl_snapshot_failed", error=str(exc))

        self._run_eod_report(report=report, positions=positions)
        self._run_reconciliation(positions=positions)
        # Sprint 18: only feed REAL account values into the drawdown circuit breaker.
        # Synthetic/default values would falsely trip the breaker when IBKR is offline.
        if account_source in ("ibkr", "env"):
            try:
                self._run_drawdown_update(account_value)
            except Exception as exc:
                _log.error("run_pnl_snapshot_drawdown_update_error", error=str(exc))
        else:
            _log.info("drawdown_update_skipped_synthetic_value", source=account_source)
        self._run_outcome_resolution()  # Sprint 22: resolve signal outcomes from fills
        return report

    def _run_outcome_resolution(self) -> None:
        """Resolve unresolved signal-journal entries against trade fills.

        Called at every EOD snapshot so calibrated weights stay fresh.
        Never raises — exceptions are logged and swallowed.
        """
        try:
            from strategies.signal_outcome_tracker import SignalOutcomeTracker  # noqa: PLC0415

            tracker = SignalOutcomeTracker()
            outcome_report = tracker.run()
            _log.info(
                "outcome_resolution_complete",
                resolved=outcome_report.resolved,
                hits=outcome_report.hits,
                misses=outcome_report.misses,
                errors=outcome_report.errors,
            )
        except Exception as exc:
            _log.error("outcome_resolution_failed", error=str(exc))

    def _run_drawdown_update(self, account_value: float) -> None:
        """Feed the current account value into the drawdown circuit breaker.

        Called at every EOD snapshot so the circuit breaker always has fresh
        data.  Skipped when ``account_value`` is zero or negative.  Never
        raises — exceptions are logged and swallowed.
        """
        if account_value <= 0:
            _log.debug("drawdown_update_skipped_zero_value", account_value=account_value)
            return
        try:
            state = self._drawdown_circuit_breaker.update(account_value)
            _log.info(
                "drawdown_state_updated",
                drawdown_pct=round(state.drawdown_pct, 4),
                tripped=state.tripped,
                peak=round(state.peak_value, 2),
                current=round(state.current_value, 2),
            )
        except Exception as exc:
            _log.error("drawdown_update_failed", error=str(exc))

    def _run_reconciliation(self, positions: list) -> None:
        """Reconcile internal position snapshot against live positions.  Never raises."""
        try:
            rec_report = self._position_reconciler.reconcile(live_positions=positions)
            if rec_report.has_mismatches:
                _log.warning(
                    "position_reconciliation_mismatches",
                    total=rec_report.mismatch_count,
                    missing=rec_report.missing_count,
                    phantom=rec_report.phantom_count,
                    size_mismatch=rec_report.size_mismatch_count,
                )
        except Exception as exc:
            _log.error("reconciliation_dispatch_failed", error=str(exc))

    def _run_eod_report(self, report: dict[str, Any], positions: list) -> None:
        """Generate and write the end-of-day brief.  Never raises."""
        try:
            from core.eod_reporter import EodReporter  # noqa: PLC0415

            pnl_delta = 0.0
            try:
                pnl_delta = self._pnl_tracker.pnl_delta()
            except Exception:  # pragma: no cover
                pass

            health_snap = None
            try:
                from monitoring.health_monitor import HealthMonitor  # noqa: PLC0415

                health_snap = HealthMonitor().collect_snapshot()
            except Exception:  # pragma: no cover
                pass

            today_report = {}
            try:
                today_report = self._pnl_tracker.today_report()
            except Exception:  # pragma: no cover
                pass

            eod = EodReporter(alerter=self._alerter).generate(
                pnl_report=today_report,
                positions=positions,
                health_snap=health_snap,
                pnl_delta=pnl_delta,
            )
            _log.info(
                "eod_report_complete",
                date=eod.report_date,
                positions=eod.position_count,
                roll_urgent=eod.roll_urgent_count,
                trades=eod.trades_today,
            )
        except Exception as exc:
            _log.error("eod_report_failed", error=str(exc))

    def _get_composite_score(self) -> float:
        """Return the current War Room composite score for regime detection.

        Falls back to 50.0 (mid-WATCH) on any error so regime tracking never
        blocks signal delivery.
        """
        try:
            from strategies.war_room_engine import IndicatorState, compute_composite_score  # noqa: PLC0415

            state = IndicatorState()
            return float(compute_composite_score(state))
        except Exception as exc:  # noqa: BLE001
            _log.warning("composite_score_fetch_failed", error=str(exc))
            return 50.0

    def _fetch_positions(self) -> list:
        """Connect, refresh, and disconnect the shared PositionTracker.

        Returns an empty list if IBKR is unavailable or on any error.
        The tracker reconnects on each call — IBKR connections are cheap
        relative to the hour-level task intervals.
        """
        import asyncio  # noqa: PLC0415

        try:
            tracker = self._position_tracker

            async def _go() -> list:
                await tracker.connect()
                try:
                    return await tracker.refresh()
                finally:
                    await tracker.disconnect()

            positions = asyncio.run(_go())
            _log.debug("positions_refreshed", count=len(positions), paper=self._paper)
            return positions
        except Exception as exc:
            _log.warning("positions_fetch_failed", error=str(exc))
            return []

    def run_order_monitor(self) -> Any:
        """Scan pending orders and cancel any that have exceeded the stale window.

        Returns:
            ``OrderMonitorReport`` from ``OrderMonitor.scan()``.
            Returns an empty report (not None) on any exception so callers
            can always read ``.cancelled`` etc. safely.
        """
        from core.order_monitor import OrderMonitorReport  # noqa: PLC0415

        try:
            report = self._order_monitor.scan()
            _log.info(
                "order_monitor_ran",
                checked=report.checked,
                cancelled=report.cancelled,
                still_pending=report.still_pending,
                errors=report.errors,
            )
            return report
        except Exception as exc:
            _log.error("order_monitor_failed", error=str(exc))
            return OrderMonitorReport()

    def run_health_check(self) -> Any:
        """Run health checks.  Returns SystemSnapshot."""
        from monitoring.health_monitor import HealthMonitor  # noqa: PLC0415

        monitor = HealthMonitor()
        snap = monitor.collect_snapshot()
        _log.info(
            "health_check_complete",
            status=snap.overall_status.value,
            n_alerts=len(snap.active_alerts),
        )
        return snap

    def run_cot_refresh(self) -> dict[str, Any]:
        """Refresh CFTC COT positioning for ES / NQ / VX (once per trading day).

        COT releases on Fridays at 15:30 ET (Tuesday data); we re-pull daily so
        the freshly-released report is picked up the next time the bot runs.
        Falls back to an empty dict on any failure — never raises.
        """
        try:
            from integrations.cftc_cot_client import CFTCCotClient  # noqa: PLC0415
        except Exception as exc:
            _log.error("cot_refresh_import_failed", error=str(exc))
            return {}

        client = CFTCCotClient()
        out: dict[str, Any] = {}
        for market in ("ES", "NQ", "RTY", "YM", "VX"):
            try:
                latest = client.get_latest(market)
                signal = client.get_extreme_signal(market, lookback_weeks=52)
                out[market] = {
                    "report_date": latest.report_date if latest else None,
                    "leveraged_net": latest.leveraged_net if latest else None,
                    "signal": signal.signal if signal else None,
                    "z_score": signal.z_score if signal else None,
                }
            except Exception as exc:  # noqa: BLE001
                _log.warning("cot_refresh_market_failed", market=market, error=str(exc))
                out[market] = {"error": str(exc)}

        _log.info(
            "cot_refresh_complete",
            markets=list(out.keys()),
            es_signal=out.get("ES", {}).get("signal"),
            nq_signal=out.get("NQ", {}).get("signal"),
            vx_signal=out.get("VX", {}).get("signal"),
        )

        # Push to war_room_live_feeds module cache so the async fetcher
        # skips the duplicate HTTP pull.
        try:
            from strategies.war_room_live_feeds import update_cot_cache_from_scheduler  # noqa: PLC0415

            update_cot_cache_from_scheduler(out)
        except Exception as exc:  # noqa: BLE001
            _log.warning("cot_cache_propagate_failed", error=str(exc))

        return out

    # ── AI briefings (Phase 4) ────────────────────────────────────────────────

    def run_daily_briefing(
        self,
        agent: str,
        prompt: str,
        *,
        slot: str,
    ) -> dict[str, Any]:
        """Run an AAC agent and persist the answer as a daily briefing.

        Args:
            agent:  one of "researcher" | "monitor" | "planner".
            prompt: user message handed to the agent.
            slot:   short label for the briefing file (e.g. "morning", "evening").

        Returns the agent result dict (or an `{error: ...}` dict on failure).
        Never raises — Ollama outages must not break the scheduler.
        """
        from pathlib import Path  # noqa: PLC0415

        try:
            from shared.aac_agents import run_agent  # noqa: PLC0415
        except Exception as exc:
            _log.error("ai_briefing_import_failed", agent=agent, slot=slot, error=str(exc))
            return {"error": f"import failed: {exc}"}

        try:
            result = run_agent(agent, prompt)
        except Exception as exc:
            _log.error("ai_briefing_run_failed", agent=agent, slot=slot, error=str(exc))
            return {"error": str(exc)}

        # Persist briefing to data/briefings/YYYY-MM-DD-<slot>-<agent>.md
        try:
            today = datetime.now(_ET).date().isoformat()
            out_dir = Path("data") / "briefings"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{today}-{slot}-{agent}.md"
            tool_lines = "\n".join(
                f"- step {tc['step']}: `{tc['tool']}`" for tc in result.get("tool_calls", [])
            ) or "_(no tool calls)_"
            out_path.write_text(
                f"# AAC Daily Briefing — {today} ({slot})\n\n"
                f"**Agent:** `{agent}`  \n**Prompt:** {prompt}\n\n"
                f"## Tool calls\n{tool_lines}\n\n"
                f"## Answer\n\n{result.get('answer', '')}\n",
                encoding="utf-8",
            )
            _log.info(
                "ai_briefing_complete",
                agent=agent,
                slot=slot,
                tool_calls=len(result.get("tool_calls", [])),
                path=str(out_path),
            )
        except Exception as exc:
            _log.warning("ai_briefing_persist_failed", agent=agent, slot=slot, error=str(exc))

        # Best-effort alerter notification (Telegram, etc.)
        try:
            answer = (result.get("answer") or "").strip()
            if answer and hasattr(self._alerter, "send"):
                preview = answer if len(answer) <= 1500 else answer[:1500] + "…"
                self._alerter.send(
                    f"ai_brief_{slot}",
                    f"[{slot.upper()} BRIEF — {agent}]\n{preview}",
                )
        except Exception as exc:
            _log.debug("ai_briefing_alert_failed", error=str(exc))

        return result

    def _morning_brief_prompt(self) -> str:
        return (
            "Brief me on the next 7 days for the AAC watchlist. Highlight any "
            "CRITICAL or HIGH-importance events, name which watchlist symbols "
            "are exposed, and end with a single 'Top concern today: ...' line."
        )

    def _evening_brief_prompt(self) -> str:
        return (
            "Plan tomorrow's trading day. Call get_positions to see what is "
            "actually held, get_account_value for sizing, then fuse with the "
            "upcoming calendar (calendar_upcoming) and current news "
            "(get_news). For any positions with events in the next 5 days, "
            "call get_option_chain to evaluate roll candidates. Cite file "
            "paths and event dates. Produce a numbered action list with timing."
        )

    # ── interval helpers ──────────────────────────────────────────────────────

    def _should_run(self, task: str, interval: float) -> bool:
        """True if at least ``interval`` seconds have elapsed since last run."""
        last = self._last_run.get(task, 0.0)
        return (_time.monotonic() - last) >= interval

    def _mark_done(self, task: str) -> None:
        self._last_run[task] = _time.monotonic()

    def _run_task(self, name: str, fn: Callable[[], Any]) -> bool:
        """Execute *fn*, mark it done, return True on success."""
        try:
            fn()
            self._mark_done(name)
            return True
        except Exception as exc:
            _log.error("scheduled_task_failed", task=name, error=str(exc))
            return False

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block and run scheduled tasks until ``stop()`` is called."""
        _log.info(
            "market_scheduler_started",
            scan_interval=self._scan_interval,
            health_interval=self._health_interval,
            tick=self._tick,
            paper=self._paper,
        )

        while not self._stop.is_set():
            try:
                now_et = datetime.now(_ET)
                today = now_et.date()

                # ── Health check: every 5 min, always (7.4) ───────────────
                if self._should_run("health_check", self._health_interval):
                    self._run_task("health_check", self.run_health_check)
                if self._stop.is_set():
                    break

                # ── Signal scan: every 15 min, market hours only (7.1) ────
                if self.is_market_hours(now_et):
                    if self._should_run("signal_scan", self._scan_interval):
                        self._run_task("signal_scan", self.run_signal_scan)
                if self._stop.is_set():
                    break

                # ── Roll check: once daily at market open (7.2) ───────────
                if (
                    self.is_trading_day(now_et)
                    and self._last_roll_date != today
                    and self.is_market_open_window(now_et)
                ):
                    if self._run_task("roll_check", self.run_roll_check):
                        self._last_roll_date = today
                if self._stop.is_set():
                    break

                # ── P&L snapshot: once daily at market close (7.3) ────────
                if (
                    self.is_trading_day(now_et)
                    and self._last_pnl_date != today
                    and self.is_market_close_window(now_et)
                ):
                    if self._run_task("pnl_snapshot", self.run_pnl_snapshot):
                        self._last_pnl_date = today

                # ── CFTC COT refresh: once daily, trading day, ≥9:35 ET ───
                # Runs after the market-open window so Friday's 15:30 release
                # gets picked up by Monday morning's first scan.
                if (
                    self.is_trading_day(now_et)
                    and self._last_cot_date != today
                    and now_et.time() >= time(9, 35)
                ):
                    if self._run_task("cot_refresh", self.run_cot_refresh):
                        self._last_cot_date = today

                # ── AI morning briefing: once daily 8:00–8:30 ET (Phase 4) ──
                if (
                    self._enable_ai_briefings
                    and self.is_trading_day(now_et)
                    and self._last_morning_brief_date != today
                    and self.is_morning_brief_window(now_et)
                ):
                    if self._run_task(
                        "morning_brief",
                        lambda: self.run_daily_briefing(
                            "monitor", self._morning_brief_prompt(), slot="morning"
                        ),
                    ):
                        self._last_morning_brief_date = today

                # ── AI evening prep: once daily 16:30–17:00 ET (Phase 4) ────
                if (
                    self._enable_ai_briefings
                    and self.is_trading_day(now_et)
                    and self._last_evening_brief_date != today
                    and self.is_evening_brief_window(now_et)
                ):
                    if self._run_task(
                        "evening_brief",
                        lambda: self.run_daily_briefing(
                            "planner", self._evening_brief_prompt(), slot="evening"
                        ),
                    ):
                        self._last_evening_brief_date = today

            except Exception as exc:
                _log.error("scheduler_loop_error", error=str(exc))

            self._stop.wait(self._tick)

        _log.info("market_scheduler_stopped")

    # ── graceful restart (7.5) ────────────────────────────────────────────────

    def run_forever(self, max_restarts: int = 10) -> None:
        """Run with auto-restart on unexpected crash.

        A clean ``stop()`` call exits immediately without counting as a crash.
        """
        while True:
            try:
                self.run()
                return  # clean stop — exit
            except Exception as exc:
                self.restart_count += 1
                _log.error(
                    "scheduler_crashed",
                    restart=self.restart_count,
                    max=max_restarts,
                    error=str(exc),
                )
                if self.restart_count > max_restarts:
                    _log.error("scheduler_max_restarts_exceeded", restarts=self.restart_count)
                    raise
                # Reset stop event so the next run() iteration starts fresh
                self._stop.clear()

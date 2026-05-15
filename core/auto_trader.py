from __future__ import annotations

"""core/auto_trader.py — Sprint 8: Auto-Execution Loop.

Converts TradeSignal lists into executed orders on IBKR or the paper engine.

Pipeline::

    TradeSignals
        → confidence filter  (< min_confidence → drop)
        → throttle check     (same ticker executed < throttle_seconds ago → skip)
        → exposure gate      (ExposureCalculator.check_new_position → drop if breach)
        → DRY_RUN gate       (DRY_RUN=true → log only, no exchange call)
        → SignalExecutor.execute()

Usage::

    trader = AutoTrader(min_confidence=0.70, paper=True, dry_run=False)
    summary = trader.run_once(signals)
    print(summary.format_report())

Integration with MarketScheduler::

    sched = MarketScheduler(auto_execute=True)   # wires AutoTrader automatically
    sched.run_forever()

The throttle prevents the scheduler from re-entering the same position every
15-minute scan.  Default throttle is 4 hours (one full trading session).
"""

import asyncio
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime

import structlog

_log = structlog.get_logger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────

_MIN_CONFIDENCE: float = 0.70   # higher bar than SignalExecutor's 0.50 gate
_THROTTLE_SECONDS: int = 14_400  # 4 h — don't re-execute same ticker per session


# ── ExecutionSummary ──────────────────────────────────────────────────────────


@dataclass
class ExecutionSummary:
    """Result of one AutoTrader.run_once() cycle.

    Attributes:
        signals_received: Total signals passed in.
        signals_filtered: Signals removed by any filter.
        signals_approved: Signals that passed all gates.
        signals_executed: Signals actually submitted to an exchange (0 if dry_run).
        confirmations:    OrderConfirmation objects from SignalExecutor.
        dry_run:          Whether this cycle ran in dry-run mode.
        paper:            Whether the executor was in paper mode.
        filter_reasons:   Human-readable reasons signals were dropped.
        generated_at:     UTC ISO timestamp of this summary.
    """

    signals_received: int
    signals_filtered: int
    signals_approved: int
    signals_executed: int
    confirmations: list          # list[OrderConfirmation]
    dry_run: bool
    paper: bool
    filter_reasons: list[str]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "signals_received": self.signals_received,
            "signals_filtered": self.signals_filtered,
            "signals_approved": self.signals_approved,
            "signals_executed": self.signals_executed,
            "confirmations": [c.to_dict() for c in self.confirmations],
            "dry_run": self.dry_run,
            "paper": self.paper,
            "filter_reasons": self.filter_reasons,
            "generated_at": self.generated_at,
        }

    def format_report(self) -> str:
        mode = "DRY RUN" if self.dry_run else ("PAPER" if self.paper else "LIVE")
        lines = [
            "=== AUTO-TRADER CYCLE ===",
            f"  Signals received : {self.signals_received}",
            f"  Signals filtered : {self.signals_filtered}",
            f"  Signals approved : {self.signals_approved}",
            f"  Signals executed : {self.signals_executed}",
            f"  Mode             : {mode}",
        ]
        if self.filter_reasons:
            lines.append("  Filtered reasons:")
            for reason in self.filter_reasons[:5]:   # cap display at 5
                lines.append(f"    - {reason}")
        if self.confirmations:
            lines.append("  Confirmations:")
            for conf in self.confirmations:
                lines.append(
                    f"    [{conf.status.value.upper()}] {conf.signal_ticker}"
                    f"  fill={conf.filled_quantity} @ {conf.avg_fill_price}"
                )
        return "\n".join(lines)


# ── AutoTrader ────────────────────────────────────────────────────────────────


class AutoTrader:
    """Converts a TradeSignal list into executed orders.

    Args:
        min_confidence:    Minimum confidence to act on a signal (default 0.70).
        paper:             Paper mode. ``None`` → reads ``PAPER_TRADING`` env var.
        dry_run:           Log only; don't execute. ``None`` → reads ``DRY_RUN`` env var.
        account_value_usd: Account size for risk sizing.
                           0 → reads ``ACCOUNT_VALUE_USD`` env var (default 50 000).
        throttle_seconds:  Min seconds before re-executing the same ticker (default 4 h).
        position_tracker:  Optional ``PositionTracker`` instance.  When set,
                           ``_exposure_ok`` fetches live positions before each
                           run cycle so total-exposure checks see real holdings.
                           ``None`` → exposure check falls back to empty list
                           (single-name and contracts limits still apply).
        drawdown_circuit_breaker: Optional ``DrawdownCircuitBreaker`` instance.
                           When set, ``run_once()`` checks ``is_tripped()``
                           *before* the daily loss guard.  If tripped, the
                           entire cycle is skipped (multi-day drawdown guard).
        daily_loss_guard:  Optional ``DailyLossGuard`` instance.  When set,
                           ``run_once()`` checks whether today's P&L has hit the
                           daily loss ceiling *before* processing any signals.
                           If the ceiling is breached, the entire cycle is skipped.
        pnl_tracker:       Optional ``PnLTracker`` instance.  When set, every
                           filled/submitted ``OrderConfirmation`` is logged to the
                           trade journal automatically after each cycle.
        signal_journal:    Optional ``SignalJournal`` instance.  When set, every
                           approved ``TradeSignal`` is logged before execution so
                           outcome tracking can later measure strategy accuracy.
        vol_sizer:         Optional ``VolTargetSizer`` instance.  When set, every
                           approved signal is passed through ``vol_adjusted_kelly()``
                           using the signal confidence and size; signals whose
                           vol-adjusted kelly falls below 0.01 are filtered out
                           (compulsory ROLL_CLOSE/STOP_CLOSE signals bypass this).
                           Fails-open on exception so a bad import never blocks trading.
    """

    def __init__(
        self,
        min_confidence: float = _MIN_CONFIDENCE,
        paper: bool | None = None,
        dry_run: bool | None = None,
        account_value_usd: float = 0.0,
        throttle_seconds: int = _THROTTLE_SECONDS,
        position_tracker: object | None = None,
        drawdown_circuit_breaker: object | None = None,
        daily_loss_guard: object | None = None,
        pnl_tracker: object | None = None,
        order_monitor: object | None = None,
        signal_journal: object | None = None,
        alerter: object | None = None,
        execution_throttle: object | None = None,
        vol_sizer: object | None = None,
        correlation_tracker: object | None = None,
    ) -> None:
        self.min_confidence = min_confidence

        if paper is None:
            paper = os.getenv("PAPER_TRADING", "false").lower() == "true"
        self.paper = paper

        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.dry_run = dry_run

        if account_value_usd <= 0:
            account_value_usd = float(os.getenv("ACCOUNT_VALUE_USD", "50000"))
        self.account_value_usd = account_value_usd

        self.throttle_seconds = throttle_seconds
        self._position_tracker = position_tracker
        self._drawdown_circuit_breaker = drawdown_circuit_breaker
        self._daily_loss_guard = daily_loss_guard
        self._pnl_tracker = pnl_tracker
        self._order_monitor = order_monitor
        self._signal_journal = signal_journal
        self._alerter = alerter  # Sprint 21 — Telegram alerter (fails-open when None)
        self._execution_throttle = execution_throttle  # Sprint 23 — persistent throttle
        self._vol_sizer = vol_sizer  # Sprint 25 — vol-target position size gate (fails-open)
        self._correlation_tracker = correlation_tracker  # Sprint 26 — contagion regime gate (fails-open)

        # Fallback in-memory throttle (used when no persistent throttle is wired)
        self._last_executed: dict[str, float] = {}
        self._last_summary: ExecutionSummary | None = None

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def last_summary(self) -> ExecutionSummary | None:
        """Summary of the most recent run_once() call."""
        return self._last_summary

    def run_once(self, signals: list) -> ExecutionSummary:
        """Process a list of TradeSignals synchronously.

        Filters by confidence, throttle, and exposure; then executes approved
        signals unless ``dry_run`` is active.

        Returns an ExecutionSummary regardless of outcome — never raises.
        """
        n_received = len(signals)
        filter_reasons: list[str] = []
        approved: list = []

        # ── Drawdown circuit breaker (multi-day drawdown guard) ─────────
        if self._drawdown_circuit_breaker is not None:
            try:
                if self._drawdown_circuit_breaker.is_tripped():
                    _log.warning(
                        "auto_trader_drawdown_halt",
                        max_drawdown_pct=getattr(
                            self._drawdown_circuit_breaker, "max_drawdown_pct", None
                        ),
                    )
                    try:
                        if self._alerter is not None:
                            pct = getattr(self._drawdown_circuit_breaker, "max_drawdown_pct", None)
                            self._alerter.send(
                                "DRAWDOWN_TRIPPED",
                                f"Multi-day drawdown circuit breaker tripped."
                                + (f" Threshold: {pct:.0%}" if pct is not None else ""),
                            )
                    except Exception:  # pragma: no cover
                        pass
                    summary = ExecutionSummary(
                        signals_received=n_received,
                        signals_filtered=n_received,
                        signals_approved=0,
                        signals_executed=0,
                        confirmations=[],
                        dry_run=self.dry_run,
                        paper=self.paper,
                        filter_reasons=["DRAWDOWN_CIRCUIT_BREAKER: multi-day drawdown threshold exceeded"],
                    )
                    self._last_summary = summary
                    return summary
            except Exception as exc:
                _log.warning("drawdown_cb_check_failed", error=str(exc))

        # ── Daily loss guard ─────────────────────────────────────────────
        if self._daily_loss_guard is not None:
            tripped, loss_reason = self._daily_loss_guard.is_limit_reached(
                account_value_usd=self.account_value_usd,
            )
            if tripped:
                _log.warning("auto_trader_daily_loss_halt", reason=loss_reason)
                try:
                    if self._alerter is not None:
                        self._alerter.send("DAILY_LOSS_TRIPPED", f"Daily loss limit reached: {loss_reason}")
                except Exception:  # pragma: no cover
                    pass
                summary = ExecutionSummary(
                    signals_received=n_received,
                    signals_filtered=n_received,
                    signals_approved=0,
                    signals_executed=0,
                    confirmations=[],
                    dry_run=self.dry_run,
                    paper=self.paper,
                    filter_reasons=[f"DAILY_LOSS_GUARD: {loss_reason}"],
                )
                self._last_summary = summary
                return summary

        # Fetch live positions once per cycle so all signals see the same
        # portfolio state (avoids N IBKR connections for N signals).
        current_positions = self._fetch_positions_sync()

        for sig in signals:
            ok, reason = self._should_execute(sig, current_positions)
            if ok:
                approved.append(sig)
            else:
                filter_reasons.append(f"{sig.ticker}: {reason}")

        n_filtered = n_received - len(approved)

        # ── Vol-target gate (Sprint 25) ───────────────────────────────────
        # Compulsory ROLL_CLOSE / STOP_CLOSE signals bypass this gate.
        if self._vol_sizer is not None and approved:
            try:
                from strategies.roll_signals import is_roll_close_signal  # noqa: PLC0415
                from strategies.stop_signals import is_stop_close_signal  # noqa: PLC0415
            except Exception:
                is_roll_close_signal = lambda s: False  # noqa: E731
                is_stop_close_signal = lambda s: False  # noqa: E731

            vol_approved: list = []
            for sig in approved:
                is_compulsory = is_roll_close_signal(sig) or is_stop_close_signal(sig)
                if is_compulsory:
                    vol_approved.append(sig)
                    continue
                try:
                    kelly = self._vol_sizer.vol_adjusted_kelly(
                        win_rate=sig.confidence,
                        avg_win=1.5,
                        avg_loss=1.0,
                        current_vol=sig.size if sig.size > 0 else 0.05,
                        historical_vol=0.05,
                    )
                    if kelly >= 0.01:
                        vol_approved.append(sig)
                    else:
                        n_filtered += 1
                        filter_reasons.append(
                            f"{sig.ticker}: VOL_GATE: kelly {kelly:.4f} below minimum"
                        )
                        _log.info(
                            "auto_trader_vol_gate_filtered",
                            ticker=sig.ticker,
                            kelly=round(kelly, 4),
                        )
                except Exception as exc:
                    _log.warning("auto_trader_vol_gate_error", error=str(exc))
                    vol_approved.append(sig)  # fail-open
            approved = vol_approved

        # ── Contagion gate (Sprint 26) ────────────────────────────────────
        # Block all non-compulsory signals when correlation regime is "contagion".
        # Uses the last cached snapshot — never runs a fresh data fetch inline.
        if self._correlation_tracker is not None and approved:
            try:
                snap = getattr(self._correlation_tracker, "last_snapshot", None)
                if snap is not None and getattr(snap, "regime", "normal") == "contagion":
                    try:
                        from strategies.roll_signals import is_roll_close_signal  # noqa: PLC0415
                        from strategies.stop_signals import is_stop_close_signal  # noqa: PLC0415
                    except Exception:
                        is_roll_close_signal = lambda s: False  # noqa: E731
                        is_stop_close_signal = lambda s: False  # noqa: E731
                    contagion_approved: list = []
                    for sig in approved:
                        is_compulsory = is_roll_close_signal(sig) or is_stop_close_signal(sig)
                        if is_compulsory:
                            contagion_approved.append(sig)
                        else:
                            n_filtered += 1
                            filter_reasons.append(
                                f"{sig.ticker}: CONTAGION_GATE: correlation regime=contagion"
                            )
                            _log.warning(
                                "auto_trader_contagion_gate_filtered",
                                ticker=sig.ticker,
                                absorption=round(getattr(snap, "absorption_ratio", 0.0), 3),
                            )
                    approved = contagion_approved
            except Exception as exc:
                _log.warning("auto_trader_contagion_gate_error", error=str(exc))
                # fail-open: approved is unchanged

        n_approved = len(approved)

        # ── DRY RUN — log but never touch the exchange ────────────────────
        if self.dry_run:
            for sig in approved:
                direction = (
                    sig.direction.value
                    if hasattr(sig.direction, "value")
                    else str(sig.direction)
                )
                _log.info(
                    "auto_trader_dry_run",
                    ticker=sig.ticker,
                    direction=direction,
                    confidence=round(sig.confidence, 3),
                    size=round(sig.size, 4),
                )
            summary = ExecutionSummary(
                signals_received=n_received,
                signals_filtered=n_filtered,
                signals_approved=n_approved,
                signals_executed=0,
                confirmations=[],
                dry_run=True,
                paper=self.paper,
                filter_reasons=filter_reasons,
            )
            self._last_summary = summary
            return summary

        # ── nothing approved ──────────────────────────────────────────────
        if not approved:
            summary = ExecutionSummary(
                signals_received=n_received,
                signals_filtered=n_filtered,
                signals_approved=0,
                signals_executed=0,
                confirmations=[],
                dry_run=False,
                paper=self.paper,
                filter_reasons=filter_reasons,
            )
            self._last_summary = summary
            return summary

        # Journal approved signals before execution (Sprint 15)
        if self._signal_journal is not None:
            self._journal_approved_signals(approved)

        # ── live / paper execution ────────────────────────────────────────
        confirmations: list = []
        try:
            confirmations = asyncio.run(self._execute_approved(approved))
        except Exception as exc:
            _log.error("auto_trader_execution_failed", error=str(exc))

        n_executed = sum(
            1
            for c in confirmations
            if hasattr(c, "status") and c.status.value in ("submitted", "filled")
        )

        # Auto-log fills to the trade journal when a PnlTracker is wired in
        if confirmations and self._pnl_tracker is not None:
            self._log_confirmations_to_journal(confirmations)

        # Register submitted orders with the order monitor (Sprint 14)
        if confirmations and self._order_monitor is not None:
            self._register_pending_with_monitor(confirmations)

        # Persist throttle for all filled/submitted tickers — Sprint 23.
        # Done here (not inside _execute_approved) so tests can mock _execute_approved
        # while still exercising throttle persistence.
        if confirmations and self._execution_throttle is not None:
            for conf in confirmations:
                try:
                    if hasattr(conf, "status") and conf.status.value in ("submitted", "filled"):
                        ticker = getattr(conf, "signal_ticker", None)
                        if ticker:
                            self._execution_throttle.record_execution(ticker)
                except Exception as exc:
                    _log.warning(
                        "auto_trader_throttle_record_error",
                        error=str(exc),
                    )

        summary = ExecutionSummary(
            signals_received=n_received,
            signals_filtered=n_filtered,
            signals_approved=n_approved,
            signals_executed=n_executed,
            confirmations=confirmations,
            dry_run=False,
            paper=self.paper,
            filter_reasons=filter_reasons,
        )
        self._last_summary = summary
        _log.info(
            "auto_trader_cycle_complete",
            received=n_received,
            filtered=n_filtered,
            executed=n_executed,
            paper=self.paper,
        )
        return summary

    # ── internal ─────────────────────────────────────────────────────────────

    def _should_execute(self, signal, current_positions: list) -> tuple[bool, str]:
        """Return (True, '') if signal passes all pre-flight checks.

        ROLL_CLOSE signals (generated by RollManager) and STOP_CLOSE signals
        (generated by StopManager) bypass the confidence threshold and throttle
        — both are compulsory and time-sensitive.
        """
        from shared.signal import Direction  # noqa: PLC0415
        from strategies.roll_signals import is_roll_close_signal  # noqa: PLC0415
        from strategies.stop_signals import is_stop_close_signal  # noqa: PLC0415

        if signal.direction is Direction.FLAT:
            return False, "FLAT signal"

        is_roll = is_roll_close_signal(signal)
        is_stop = is_stop_close_signal(signal)
        is_compulsory = is_roll or is_stop

        if not is_compulsory and signal.confidence < self.min_confidence:
            return (
                False,
                f"confidence {signal.confidence:.2f} < min {self.min_confidence:.2f}",
            )

        # Throttle: don't re-execute the same ticker within the window.
        # ROLL_CLOSE and STOP_CLOSE signals bypass the throttle — position MUST be closed.
        if not is_compulsory:
            if self._execution_throttle is not None:
                # Persistent throttle (survives restarts) — Sprint 23
                try:
                    if not self._execution_throttle.can_execute(signal.ticker):
                        remaining_s = self._execution_throttle.remaining_seconds(signal.ticker)
                        remaining_h = remaining_s / 3600
                        return False, f"throttled ({remaining_h:.1f}h remaining)"
                except Exception as exc:
                    _log.warning("auto_trader_throttle_check_error", ticker=signal.ticker, error=str(exc))
            else:
                # Fallback in-memory throttle (no persistence across restarts)
                last = self._last_executed.get(signal.ticker, 0.0)
                elapsed = _time.monotonic() - last
                if elapsed < self.throttle_seconds:
                    remaining_h = (self.throttle_seconds - elapsed) / 3600
                    return False, f"throttled ({remaining_h:.1f}h remaining)"

        # Portfolio exposure gate (uses live positions when tracker is available)
        ok, reason = self._exposure_ok(signal, current_positions)
        if not ok:
            return False, reason

        # COT extreme positioning veto (added 2026-04-13)
        # If Lev Money is at a contrarian extreme on a related underlying,
        # block any signal that would add exposure in the SAME direction as the crowd.
        # ES extreme_long → block new bullish SPY/QQQ/IWM. extreme_short → block new bearish.
        if not is_compulsory:
            cot_block, cot_reason = self._cot_extreme_veto(signal)
            if cot_block:
                return False, cot_reason

        return True, ""

    def _cot_extreme_veto(self, signal) -> tuple[bool, str]:
        """Block signals that align with crowded Lev Money positioning.

        Uses the most recent LiveFeedResult cached at module level in
        strategies.war_room_live_feeds. Fails open on any exception.
        """
        try:
            from shared.signal import Direction  # noqa: PLC0415
            from strategies import war_room_live_feeds as _wrf  # noqa: PLC0415

            result = getattr(_wrf, "_last_feed_result", None)
            if result is None:
                return False, ""

            # Map signal ticker → COT proxy
            ticker = (signal.ticker or "").upper()
            es_proxies = {"SPY", "ES", "SPX", "IWM", "RTY", "DIA"}
            nq_proxies = {"QQQ", "NQ", "QQQM"}
            vx_proxies = {"VXX", "UVXY", "SVXY", "VIX"}

            if ticker in es_proxies:
                extreme = getattr(result, "cot_es_extreme", None)
                market = "ES"
            elif ticker in nq_proxies:
                extreme = getattr(result, "cot_nq_extreme", None)
                market = "NQ"
            elif ticker in vx_proxies:
                extreme = getattr(result, "cot_vx_extreme", None)
                market = "VX"
            else:
                return False, ""

            if extreme not in ("extreme_long", "extreme_short"):
                return False, ""

            # extreme_long: leveraged crowd is long → block new bullish exposure
            # extreme_short: crowd is short → block new bearish exposure
            if extreme == "extreme_long" and signal.direction is Direction.LONG:
                return True, f"COT veto: {market} Lev Money extreme_long (crowded)"
            if extreme == "extreme_short" and signal.direction is Direction.SHORT:
                return True, f"COT veto: {market} Lev Money extreme_short (crowded)"
            return False, ""
        except Exception as exc:
            _log.warning("cot_veto_check_failed", error=str(exc))
            return False, ""

    def _exposure_ok(self, signal, current_positions: list) -> tuple[bool, str]:
        """Check that adding this position wouldn't breach exposure limits.

        Args:
            signal:            The TradeSignal being evaluated.
            current_positions: Live PositionSnapshot objects (from _fetch_positions_sync).
                               Pass an empty list when no tracker is configured.

        Fails-open on exception (returns True) so a bad risk-engine import
        doesn't silently block all trading.
        """
        try:
            from strategies.risk_engine import ExposureCalculator, RiskConfig  # noqa: PLC0415

            config = RiskConfig()
            calc = ExposureCalculator(config)
            approved, reason = calc.check_new_position(
                ticker=signal.ticker,
                size_fraction=signal.size,
                account_value_usd=self.account_value_usd,
                current_positions=current_positions,
            )
            return approved, reason or ""
        except Exception as exc:
            _log.warning("exposure_check_failed", error=str(exc))
            return True, ""   # fail-open: don't block trading on check error

    def _fetch_positions_sync(self) -> list:
        """Fetch current positions from the live tracker synchronously.

        Returns an empty list if no tracker is configured or on error.
        Each call connects, refreshes, and disconnects — so this should be
        called once per ``run_once()`` cycle, not per-signal.
        """
        if self._position_tracker is None:
            return []
        try:
            tracker = self._position_tracker

            async def _go() -> list:
                await tracker.connect()
                try:
                    return await tracker.refresh()
                finally:
                    await tracker.disconnect()

            positions = asyncio.run(_go())
            _log.debug(
                "positions_fetched",
                count=len(positions),
                paper=self.paper,
            )
            return positions
        except Exception as exc:
            _log.warning("position_fetch_failed", error=str(exc))
            return []

    def _journal_approved_signals(self, signals: list) -> None:
        """Log approved signals to the signal journal before execution (Sprint 15).

        Journalling failures are silently swallowed — they must never block execution.

        Args:
            signals: List of ``TradeSignal`` objects approved for execution.
        """
        try:
            for signal in signals:
                strategy_source = str(getattr(signal, "strategy", "unknown"))
                self._signal_journal.log_signal(  # type: ignore[union-attr]
                    signal,
                    strategy_source=strategy_source,
                )
        except Exception as exc:
            _log.warning("signal_journal_write_failed", error=str(exc))

    def _register_pending_with_monitor(self, confirmations: list) -> None:
        """Register SUBMITTED confirmations with the order monitor.

        Only SUBMITTED (not yet filled) orders need tracking.  Errors are
        silently swallowed — monitoring must never block execution.
        """
        for conf in confirmations:
            try:
                from TradingExecution.signal_executor import ConfirmationStatus  # noqa: PLC0415
                if not hasattr(conf, "status"):
                    continue
                if conf.status is not ConfirmationStatus.SUBMITTED:
                    continue
                order_id = getattr(conf, "order_id", "")
                ticker = getattr(conf, "signal_ticker", "")
                if not order_id:
                    continue
                # Parse submitted_at ISO string back to datetime
                from datetime import datetime, timezone  # noqa: PLC0415
                submitted_at_str = getattr(conf, "submitted_at", "")
                try:
                    submitted_at = datetime.fromisoformat(submitted_at_str).replace(
                        tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    submitted_at = datetime.now(tz=timezone.utc)
                self._order_monitor.register(  # type: ignore[union-attr]
                    order_id=order_id,
                    ticker=ticker,
                    submitted_at=submitted_at,
                )
            except Exception as exc:
                _log.warning(
                    "auto_trader_monitor_register_failed",
                    error=str(exc),
                )

    def _log_confirmations_to_journal(self, confirmations: list) -> None:
        """Log filled/submitted confirmations to PnLTracker trade journal.

        Silently ignores errors — trade journaling must never block execution.
        """
        for conf in confirmations:
            try:
                if not (hasattr(conf, "status") and conf.status.value in ("submitted", "filled")):
                    continue
                self._pnl_tracker.log_trade_from_confirmation(conf)
                _log.debug(
                    "auto_trader_trade_logged",
                    ticker=getattr(conf, "signal_ticker", "?"),
                    status=conf.status.value,
                )
            except Exception as exc:
                _log.warning(
                    "auto_trader_journal_log_failed",
                    error=str(exc),
                    ticker=getattr(conf, "signal_ticker", "?"),
                )

    async def _execute_approved(self, signals: list) -> list:
        """Connect the executor, execute all approved signals, then disconnect."""
        from TradingExecution.signal_executor import SignalExecutor  # noqa: PLC0415

        executor = SignalExecutor(
            paper=self.paper,
            account_value_usd=self.account_value_usd,
        )
        confirmations: list = []
        try:
            connected = await executor.connect()
            if not connected:
                _log.error("auto_trader_executor_connect_failed")
                return confirmations

            for signal in signals:
                try:
                    conf = await executor.execute(signal)
                    confirmations.append(conf)
                    # Record throttle timestamp only on successful submission
                    if conf.status.value in ("submitted", "filled"):
                        self._last_executed[signal.ticker] = _time.monotonic()
                        # NOTE: persistent throttle recording is done in run_once()
                        # after _execute_approved returns, so tests can mock this method.
                    _log.info(
                        "auto_trader_executed",
                        ticker=signal.ticker,
                        status=conf.status.value,
                        order_id=conf.order_id,
                        paper=self.paper,
                    )
                except Exception as exc:
                    _log.error(
                        "auto_trader_signal_error",
                        ticker=signal.ticker,
                        error=str(exc),
                    )
        finally:
            await executor.disconnect()

        return confirmations

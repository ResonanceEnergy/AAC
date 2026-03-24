"""
Pre-Market Scanner — 9:15 AM ET Daily Options Intelligence
=============================================================
Runs Monday-Friday at 9:15 AM Eastern (15 min before market open):

1. Connects to IBKR
2. Runs the full Options Intelligence Pipeline with live data
3. Posts trade recommendations to Gasket Discord
4. Waits for user confirmation (✅ / ❌ reaction)
5. Executes confirmed trades through IBKR
6. Reports results back to Discord

Startup integration:
    Called from startup/phases.py as Phase 7.
    Runs as a daemon thread — does not block the main process.

Health check:
    Exposes status via health_check() for /health endpoint.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from strategies.options_intelligence.discord_notifier import (
    ConfirmationResult,
    ConfirmationStatus,
    GasketDiscordNotifier,
    TradeRecommendation,
)
from strategies.options_intelligence.ibkr_executor import (
    ExecutionSummary,
    IBKRExecutor,
)
from strategies.options_intelligence.pipeline import PipelineResult

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Configurable via env
SCAN_HOUR = int(os.environ.get("PREMARKET_SCAN_HOUR", "9"))
SCAN_MINUTE = int(os.environ.get("PREMARKET_SCAN_MINUTE", "15"))
MIN_SCORE = int(os.environ.get("PREMARKET_MIN_SCORE", "60"))
STRONG_ONLY = os.environ.get("PREMARKET_STRONG_ONLY", "false").lower() == "true"


@dataclass
class ScanResult:
    """Result of a single pre-market scan cycle."""
    scan_time: datetime
    pipeline_result: Optional[PipelineResult] = None
    recommendations: List[TradeRecommendation] = field(default_factory=list)
    confirmation: Optional[ConfirmationResult] = None
    execution: Optional[ExecutionSummary] = None
    error: Optional[str] = None

    @property
    def status(self) -> str:
        if self.error:
            return f"error: {self.error}"
        if self.execution and self.execution.successful > 0:
            return f"executed ({self.execution.successful}/{self.execution.total_attempted})"
        if self.confirmation:
            return self.confirmation.status.value
        if self.pipeline_result:
            return f"scanned ({len(self.recommendations)} recs)"
        return "pending"


class PreMarketScanner:
    """
    Daily pre-market scanner that runs Options Intelligence at 9:15 AM ET.

    Lifecycle:
        scanner = PreMarketScanner()
        scanner.start()   # Launches daemon thread
        scanner.stop()    # Graceful shutdown

    The scanner sleeps between cycles and wakes at the configured time.
    On weekends (Sat/Sun), it skips to Monday.
    """

    def __init__(
        self,
        paper: bool = True,
        dry_run: bool = True,
        strong_only: bool = STRONG_ONLY,
    ):
        self._executor = IBKRExecutor(paper=paper, dry_run=dry_run)
        self._notifier = GasketDiscordNotifier()
        self._paper = paper
        self._dry_run = dry_run
        self._strong_only = strong_only

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_scan: Optional[ScanResult] = None
        self._scan_count = 0
        self._started_at: Optional[datetime] = None

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scanner in a background daemon thread."""
        if self._running:
            logger.warning("PreMarketScanner already running")
            return

        self._running = True
        self._started_at = datetime.now(ET)
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="PreMarketScanner",
        )
        self._thread.start()
        logger.info(
            "PreMarketScanner started (paper=%s, dry_run=%s, scan_time=%02d:%02d ET)",
            self._paper, self._dry_run, SCAN_HOUR, SCAN_MINUTE,
        )

    def stop(self) -> None:
        """Signal the scanner to stop after current cycle."""
        self._running = False
        logger.info("PreMarketScanner stop requested")

    def health_check(self) -> Dict[str, Any]:
        """Return health status for the /health endpoint."""
        now_et = datetime.now(ET)
        next_scan = self._next_scan_time(now_et)

        status: Dict[str, Any] = {
            "component": "premarket_scanner",
            "running": self._running,
            "paper": self._paper,
            "dry_run": self._dry_run,
            "scan_time": f"{SCAN_HOUR:02d}:{SCAN_MINUTE:02d} ET",
            "next_scan": next_scan.strftime("%Y-%m-%d %H:%M ET"),
            "scans_completed": self._scan_count,
            "discord_configured": self._notifier.configured,
        }

        if self._started_at:
            status["uptime_seconds"] = (now_et - self._started_at).total_seconds()

        if self._last_scan:
            status["last_scan"] = {
                "time": self._last_scan.scan_time.isoformat(),
                "status": self._last_scan.status,
                "recommendations": len(self._last_scan.recommendations),
            }
            if self._last_scan.execution:
                status["last_scan"]["executed"] = self._last_scan.execution.successful
                status["last_scan"]["failed"] = self._last_scan.execution.failed

        status["discord"] = self._notifier.health_check()

        return status

    # ── Main Loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main loop: sleep until scan time, run scan, repeat."""
        while self._running:
            try:
                now_et = datetime.now(ET)
                next_scan = self._next_scan_time(now_et)
                sleep_seconds = (next_scan - now_et).total_seconds()

                if sleep_seconds > 0:
                    logger.info(
                        "PreMarketScanner sleeping until %s (%.0f min)",
                        next_scan.strftime("%Y-%m-%d %H:%M ET"),
                        sleep_seconds / 60,
                    )
                    # Sleep in small increments so stop() is responsive
                    end_time = time.monotonic() + sleep_seconds
                    while self._running and time.monotonic() < end_time:
                        time.sleep(min(30, end_time - time.monotonic()))

                if not self._running:
                    break

                # Run the scan
                result = self._run_scan()
                self._last_scan = result
                self._scan_count += 1

                logger.info(
                    "Scan #%d complete: %s", self._scan_count, result.status
                )

            except Exception as e:
                logger.error("PreMarketScanner loop error: %s", e, exc_info=True)
                # Sleep 60s on error to avoid tight loop
                time.sleep(60)

        logger.info("PreMarketScanner stopped")

    def _run_scan(self) -> ScanResult:
        """Execute a single scan cycle: connect → pipeline → notify → execute."""
        now_et = datetime.now(ET)
        result = ScanResult(scan_time=now_et)

        logger.info("=" * 60)
        logger.info("PRE-MARKET SCAN — %s", now_et.strftime("%A %B %d, %Y %H:%M ET"))
        logger.info("=" * 60)

        # Step 1: Connect to IBKR
        loop = asyncio.new_event_loop()
        try:
            connected = loop.run_until_complete(self._executor.connect())
            if not connected:
                result.error = "Failed to connect to IBKR"
                self._notify_error(result.error)
                return result

            # Step 2: Run pipeline
            pipeline_result = self._run_pipeline(loop)
            if not pipeline_result:
                result.error = "Pipeline returned no result"
                self._notify_error(result.error)
                return result

            result.pipeline_result = pipeline_result

            # Step 3: Build recommendations
            orders = (
                pipeline_result.strong_orders
                if self._strong_only
                else pipeline_result.actionable_orders
            )

            if not orders:
                logger.info("No actionable trades found")
                self._notifier.send_status_update(
                    f"📊 **Pre-Market Scan** ({now_et.strftime('%b %d')})\n"
                    f"No actionable trades today. Pipeline returned "
                    f"{len(pipeline_result.base_orders)} base orders, "
                    f"{len(pipeline_result.rejected_orders)} rejected."
                )
                return result

            recommendations = self._build_recommendations(orders)
            result.recommendations = recommendations

            summary = (
                f"Regime: {pipeline_result.flow_convictions.get('_regime', 'unknown')} | "
                f"Scanned: {len(pipeline_result.base_orders)} | "
                f"Passed: {len(orders)}"
            )

            # Step 4: Send to Discord and wait for confirmation
            logger.info(
                "Posting %d recommendations to Discord...",
                len(recommendations),
            )
            confirmation = self._notifier.send_and_confirm(
                recommendations=recommendations,
                pipeline_summary=summary,
            )
            result.confirmation = confirmation

            if confirmation.status == ConfirmationStatus.CONFIRMED:
                # Step 5: Execute trades
                logger.info("Trades CONFIRMED — executing...")
                execution = loop.run_until_complete(
                    self._executor.execute_orders(
                        pipeline_result,
                        strong_only=self._strong_only,
                    )
                )
                result.execution = execution

                # Step 6: Report results
                self._notifier.send_execution_result(
                    summary_text=execution.report(),
                    success=execution.successful > 0,
                )

            elif confirmation.status == ConfirmationStatus.REJECTED:
                logger.info("Trades REJECTED by user")
                self._notifier.send_status_update(
                    "🚫 Trade plan rejected. No orders placed."
                )

            elif confirmation.status == ConfirmationStatus.TIMED_OUT:
                logger.warning("Trade confirmation TIMED OUT — no orders placed")
                self._notifier.send_status_update(
                    "⏰ Confirmation timed out. No orders placed."
                )

        except Exception as e:
            result.error = str(e)
            logger.error("Scan failed: %s", e, exc_info=True)
            self._notify_error(str(e))
        finally:
            loop.run_until_complete(self._executor.disconnect())
            loop.close()

        return result

    def _run_pipeline(self, loop: asyncio.AbstractEventLoop) -> Optional[PipelineResult]:
        """Run the Options Intelligence pipeline with current crisis assessment."""
        try:
            from strategies.macro_crisis_put_strategy import (
                CrisisMonitor,
                CrisisAssessment,
            )

            # Get current crisis assessment
            monitor = CrisisMonitor()
            assessment = monitor.assess()

            logger.info(
                "Crisis assessment: severity=%.2f, vectors=%s",
                assessment.severity,
                [v.name for v in assessment.active_vectors[:5]],
            )

            # Get current VIX
            vix = self._get_vix()

            return loop.run_until_complete(
                self._executor.run_pipeline(
                    assessment=assessment,
                    regime="CRISIS" if assessment.severity > 0.5 else "CAUTIOUS",
                    vix=vix,
                    fetch_chains=True,
                )
            )
        except Exception as e:
            logger.error("Pipeline execution failed: %s", e)
            return None

    def _get_vix(self) -> float:
        """Get current VIX level from available sources."""
        try:
            import yfinance as yf
            vix_data = yf.Ticker("^VIX")
            hist = vix_data.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass

        # Fallback: use env or default
        return float(os.environ.get("CURRENT_VIX", "20.0"))

    @staticmethod
    def _build_recommendations(
        orders: List[Dict[str, Any]],
    ) -> List[TradeRecommendation]:
        """Convert pipeline orders to TradeRecommendation display objects."""
        recs = []
        for entry in orders:
            order = entry["order"]
            score = entry["score"]
            recs.append(TradeRecommendation(
                symbol=order.symbol,
                strike=order.strike,
                expiry=order.expiry,
                contracts=order.contracts,
                max_price=order.max_price,
                score=score.score if hasattr(score, "score") else 0,
                crisis_vectors=(
                    order.crisis_vectors if hasattr(order, "crisis_vectors") else []
                ),
                description=getattr(order, "description", ""),
            ))
        return recs

    def _notify_error(self, error: str) -> None:
        """Send error notification to Discord."""
        self._notifier.send_status_update(
            f"🚨 **Pre-Market Scan Error**\n```\n{error[:500]}\n```"
        )

    # ── Schedule Helpers ────────────────────────────────────────────────

    @staticmethod
    def _next_scan_time(now: datetime) -> datetime:
        """Calculate the next scan time (Mon-Fri at SCAN_HOUR:SCAN_MINUTE ET)."""
        today_scan = now.replace(
            hour=SCAN_HOUR,
            minute=SCAN_MINUTE,
            second=0,
            microsecond=0,
        )

        if now < today_scan and now.weekday() < 5:
            # Today's scan hasn't happened yet and it's a weekday
            return today_scan

        # Move to next weekday
        candidate = today_scan + timedelta(days=1)
        while candidate.weekday() >= 5:  # Skip Sat(5), Sun(6)
            candidate += timedelta(days=1)

        return candidate

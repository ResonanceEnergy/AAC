"""
startup.phases — Consolidated multi-phase startup sequences.

Provides the ``full_startup()`` function that runs the complete boot:
  Phase 1: Pre-flight checks
  Phase 2: Trading gateways (IBKR TWS, Moomoo OpenD)
  Phase 3: Health endpoint
  Phase 4: Paper trading engine
  Phase 5: Matrix Monitor dashboard

Each phase can also be invoked individually.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from threading import Thread

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _python() -> str:
    return sys.executable


# ── Phase 1: Pre-flight ────────────────────────────────────────────────────


def phase_preflight() -> bool:
    """Run pre-flight validation. Returns True if safe to proceed."""
    from startup.preflight import run_all
    return run_all()


# ── Phase 2: Gateways ─────────────────────────────────────────────────────


def phase_gateways() -> dict[str, bool]:
    """Start IBKR + Moomoo gateways. Returns status dict."""
    from startup.gateways import start_all_gateways, gateway_summary
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Phase 2: Trading Gateways")
    logger.info("  ═══════════════════════════════════════")
    results = start_all_gateways()
    logger.info(gateway_summary(results))
    return results


# ── Phase 3: Health endpoint ───────────────────────────────────────────────


def phase_health_endpoint() -> bool:
    """Start the background health HTTP endpoint on port 8080."""
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Phase 3: Health Endpoint")
    logger.info("  ═══════════════════════════════════════")
    try:
        from health_server import start_health_server
        start_health_server(background=True)
        logger.info("  [+] Health endpoint: http://localhost:8080/health")
        return True
    except Exception as e:
        logger.error(f"  [!] Health endpoint failed: {e}")
        return False


# ── Phase 4: Paper trading engine ──────────────────────────────────────────


def phase_paper_engine() -> int:
    """Start the trading engine (paper or live based on env). Returns exit code."""
    logger.info("  ═══════════════════════════════════════")
    is_live = os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"
    mode_label = "LIVE Trading Engine" if is_live else "Paper Trading Engine"
    logger.info(f"  Phase 4: {mode_label}")
    logger.info("  ═══════════════════════════════════════")
    if not is_live:
        os.environ.setdefault("PAPER_TRADING", "true")
        os.environ.setdefault("LIVE_TRADING_ENABLED", "false")
    args = [_python(), "-m", "core.orchestrator"]
    if not is_live:
        args.append("--paper")
    return subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
    ).returncode


# ── Phase 5: Matrix Monitor ───────────────────────────────────────────────


def phase_matrix_monitor(display: str = "terminal", port: int = 8501) -> int:
    """Start the Matrix Monitor dashboard. Returns exit code."""
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Phase 5: Matrix Monitor")
    logger.info("  ═══════════════════════════════════════")
    from startup.matrix_monitor import launch
    return launch(display=display, port=port)


def _start_matrix_background(display: str = "web", port: int = 8501) -> None:
    """Start Matrix Monitor in a background thread (for ``all`` mode)."""
    from startup.matrix_monitor import launch
    launch(display=display, port=port)


# ── Full Startup Sequence ─────────────────────────────────────────────────


def full_startup(
    display: str = "web",
    port: int = 8501,
    skip_preflight: bool = False,
) -> int:
    """
    Run the complete AAC startup sequence.

    1. Pre-flight checks
    2. Trading gateways
    3. Matrix Monitor (background thread — web mode)
    4. Paper trading engine (foreground — blocks until Ctrl+C)
       The orchestrator starts its own health endpoint on port 8080.

    Returns the paper engine exit code.
    """
    logger.info("  ╔══════════════════════════════════════════╗")
    logger.info("  ║   AAC FULL STARTUP — ALL SYSTEMS GO      ║")
    logger.info("  ╚══════════════════════════════════════════╝")
    logger.info("")

    # Phase 1
    if not skip_preflight:
        if not phase_preflight():
            logger.error("  Pre-flight FAILED — aborting.")
            return 1
        logger.info("")

    # Phase 2
    phase_gateways()
    logger.info("")

    # Phase 5 (before 4 so dashboard is ready while engine runs)
    logger.info("  Starting Matrix Monitor in background ...")
    monitor_thread = Thread(
        target=_start_matrix_background,
        args=(display, port),
        daemon=True,
        name="MatrixMonitor",
    )
    monitor_thread.start()
    logger.info(f"  [+] Matrix Monitor launching ({display} on port {port})")
    logger.info("")

    # Phase 4 (blocking — runs until Ctrl+C)
    return phase_paper_engine()

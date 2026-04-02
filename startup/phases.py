"""
startup.phases — Consolidated multi-phase startup sequences.

Provides the ``full_startup()`` function that runs the complete boot:
  Phase 1: Pre-flight checks
  Phase 2: Trading gateways (IBKR TWS, Moomoo OpenD)
  Phase 3: Health endpoint
  Phase 4: Paper trading engine  (supervised by watchdog)
  Phase 5: Matrix Monitor dashboard
  Phase 6: OpenClaw Gateway
  Phase 7: Options Intelligence Pre-Market Scanner

Each phase can also be invoked individually.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from threading import Thread

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Module-level refs so other modules can request shutdown
_watchdog = None
_monitor_thread = None
_health_server = None


def _python() -> str:
    return sys.executable


def get_watchdog():
    """Return the active ProcessWatchdog instance, or None."""
    return _watchdog


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if *host*:*port* is already bound."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except (OSError, ConnectionRefusedError):
        return False


# ── Phase 1: Pre-flight ────────────────────────────────────────────────────


def phase_preflight() -> bool:
    """Run pre-flight validation. Returns True if safe to proceed."""
    from startup.preflight import run_all
    return run_all()


# ── Phase 2: Gateways ─────────────────────────────────────────────────────


def phase_gateways() -> dict[str, bool]:
    """Start IBKR + Moomoo gateways. Returns status dict."""
    from startup.gateways import gateway_summary, start_all_gateways
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Phase 2: Trading Gateways")
    logger.info("  ═══════════════════════════════════════")
    results = start_all_gateways()
    logger.info(gateway_summary(results))
    return results


# ── Phase 3: Health endpoint ───────────────────────────────────────────────


def phase_health_endpoint() -> bool:
    """Start the background health HTTP endpoint on port 8080."""
    global _health_server
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Phase 3: Health Endpoint")
    logger.info("  ═══════════════════════════════════════")
    port = int(os.environ.get("HEALTH_PORT", "8080"))
    if _port_in_use(port):
        logger.warning(f"  [!] Port {port} already in use — health endpoint skipped")
        return True
    try:
        from health_server import start_health_server
        _health_server = start_health_server(background=True)
        logger.info(f"  [+] Health endpoint: http://localhost:{port}/health")
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


# ── Phase 6: OpenClaw Gateway ──────────────────────────────────────────────


def phase_openclaw() -> bool:
    """Connect to OpenClaw Gateway for multi-channel messaging. Non-blocking."""
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Phase 6: OpenClaw Gateway")
    logger.info("  ═══════════════════════════════════════")
    try:
        import asyncio

        from integrations.openclaw_gateway_bridge import (
            OpenClawCronJob,
            OpenClawGatewayBridge,
        )
        gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789")
        bridge = OpenClawGatewayBridge(gateway_url=gateway_url)
        connected = asyncio.run(bridge.connect())
        if connected:
            logger.info(f"  [+] OpenClaw Gateway LIVE at {gateway_url}")
            return True
        else:
            logger.warning("  [!] OpenClaw Gateway connect() returned False")
            return False
    except Exception as e:
        logger.warning(f"  [!] OpenClaw init failed (non-critical): {e}")
        return False


# ── Phase 7: Options Intelligence Pre-Market Scanner ──────────────────────

_premarket_scanner = None  # Module-level ref for health checks


def phase_premarket_scanner() -> bool:
    """Start the Options Intelligence pre-market scanner (9:15 AM ET Mon-Fri)."""
    global _premarket_scanner
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Phase 7: Options Intelligence Pre-Market Scanner")
    logger.info("  ═══════════════════════════════════════")
    try:
        from strategies.options_intelligence.premarket_scanner import PreMarketScanner

        paper = os.environ.get("LIVE_TRADING_ENABLED", "false").lower() != "true"
        dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"

        scanner = PreMarketScanner(paper=paper, dry_run=dry_run)
        scanner.start()
        _premarket_scanner = scanner
        logger.info(
            "  [+] Pre-Market Scanner ACTIVE (paper=%s, dry_run=%s)",
            paper, dry_run,
        )
        return True
    except Exception as e:
        logger.error("  [!] Pre-Market Scanner failed: %s", e)
        return False


def get_premarket_scanner_health() -> dict:
    """Get pre-market scanner health for the health endpoint."""
    if _premarket_scanner is None:
        return {"component": "premarket_scanner", "running": False, "status": "not_started"}
    return _premarket_scanner.health_check()


# ── Full Startup Sequence ─────────────────────────────────────────────────


def full_startup(
    display: str = "web",
    port: int = 8501,
    skip_preflight: bool = False,
) -> int:
    """
    Run the complete AAC v3.6 startup sequence.

    1. Pre-flight checks
    2. Trading gateways (IBKR TWS, Moomoo OpenD)
    6. OpenClaw Gateway (non-critical)
    7. Options Intelligence Pre-Market Scanner (non-critical)
    5. Matrix Monitor (background thread — web mode)
    4. Trading engine — watchdog-supervised (blocks until SIGINT/SIGTERM)

    The watchdog supervises the orchestrator and auto-restarts it on crash.
    Returns 0 on clean shutdown, 1 on startup failure.
    """
    global _watchdog, _monitor_thread

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

    # Phase 6: OpenClaw Gateway (non-critical)
    phase_openclaw()
    logger.info("")

    # Phase 7: Options Intelligence Pre-Market Scanner (non-critical)
    phase_premarket_scanner()
    logger.info("")

    # Phase 5 (before 4 so dashboard is ready while engine runs)
    if _port_in_use(port):
        logger.warning(f"  [!] Port {port} already in use — Matrix Monitor skipped (another instance running?)")
    else:
        logger.info("  Starting Matrix Monitor in background ...")
        _monitor_thread = Thread(
            target=_start_matrix_background,
            args=(display, port),
            daemon=True,
            name="MatrixMonitor",
        )
        _monitor_thread.start()
        logger.info(f"  [+] Matrix Monitor launching ({display} on port {port})")
    logger.info("")

    # Phase 4 (supervised by watchdog — blocks until SIGINT/SIGTERM)
    logger.info("  ═══════════════════════════════════════")
    is_live = os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"
    mode_label = "LIVE Trading Engine" if is_live else "Paper Trading Engine"
    logger.info(f"  Phase 4: {mode_label} (watchdog-supervised)")
    logger.info("  ═══════════════════════════════════════")

    from startup.watchdog import ProcessWatchdog

    _watchdog = ProcessWatchdog(poll_interval=10.0)

    # Build orchestrator command
    orch_cmd = [_python(), "-m", "core.orchestrator"]
    if not is_live:
        os.environ.setdefault("PAPER_TRADING", "true")
        os.environ.setdefault("LIVE_TRADING_ENABLED", "false")
        orch_cmd.append("--paper")

    _watchdog.register(
        "orchestrator",
        orch_cmd,
        max_restarts=3,
        health_url="http://127.0.0.1:8080/health",
        critical=True,
    )

    try:
        # Block until Ctrl+C / SIGTERM
        _watchdog.run_forever()
    finally:
        logger.info("  Shutting down AAC systems ...")
        # Shutdown health server
        if _health_server is not None:
            try:
                _health_server.shutdown()
                logger.info("  [+] Health server stopped")
            except Exception:
                pass
        # Wait briefly for monitor thread to finish
        if _monitor_thread is not None and _monitor_thread.is_alive():
            logger.info("  [+] Matrix Monitor thread stopping (daemon) ...")
            _monitor_thread.join(timeout=3.0)
        logger.info("  AAC shutdown complete.")
    return 0

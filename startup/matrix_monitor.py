"""
startup.matrix_monitor — Launch the AAC Matrix Monitor dashboard.

Wraps monitoring.aac_master_monitoring_dashboard with a clean API
so launch.py and other entry points can start it uniformly.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def launch_terminal(port: int = 8501) -> int:
    """Start the Matrix Monitor in terminal (curses/text) mode."""
    logger.info("  Starting Matrix Monitor — terminal mode ...")
    try:
        from monitoring.aac_master_monitoring_dashboard import (
            get_master_dashboard, DisplayMode,
        )
        dashboard = get_master_dashboard(DisplayMode.TERMINAL)
        asyncio.run(dashboard.start_monitoring())
        return 0
    except KeyboardInterrupt:
        logger.info("  Matrix Monitor stopped.")
        return 0
    except Exception as e:
        logger.error(f"  [X] Matrix Monitor (terminal) failed: {e}")
        return 1


def launch_web(port: int = 8501) -> int:
    """Start the Matrix Monitor Streamlit web UI."""
    logger.info(f"  Starting Matrix Monitor — Streamlit on port {port} ...")
    dashboard_path = PROJECT_ROOT / "monitoring" / "aac_master_monitoring_dashboard.py"
    if not dashboard_path.exists():
        logger.error(f"  [X] Dashboard not found: {dashboard_path}")
        return 1
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
         "--server.port", str(port), "--server.headless", "true"],
        cwd=str(PROJECT_ROOT),
    ).returncode


def launch_dash(port: int = 8502) -> int:
    """Start the Matrix Monitor Plotly Dash analytics dashboard."""
    logger.info(f"  Starting Matrix Monitor — Dash on port {port} ...")
    try:
        from monitoring.aac_master_monitoring_dashboard import (
            get_master_dashboard, DisplayMode,
        )
        dashboard = get_master_dashboard(DisplayMode.DASH)
        if hasattr(dashboard, "initialize"):
            asyncio.run(dashboard.initialize())
        if hasattr(dashboard, "run_dashboard"):
            dashboard.run_dashboard(port=port)
        return 0
    except Exception as e:
        logger.error(f"  [X] Matrix Monitor (dash) failed: {e}")
        return 1


def launch(display: str = "terminal", port: int = 8501) -> int:
    """
    Start the Matrix Monitor in the requested display mode.

    Parameters
    ----------
    display : str
        One of ``terminal``, ``web``, ``dash``.
    port : int
        Port for web/dash modes (default 8501).
    """
    dispatch = {
        "terminal": lambda: launch_terminal(port),
        "web": lambda: launch_web(port),
        "dash": lambda: launch_dash(port),
    }
    handler = dispatch.get(display)
    if handler is None:
        logger.error(f"  [X] Unknown display mode: {display!r}  (use terminal/web/dash)")
        return 1
    return handler()

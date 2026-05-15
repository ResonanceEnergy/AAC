#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  BARREN WUFFET — Unified Launcher                              ║
║  Codename: AZ SUPREME                                         ║
║                                                                ║
║  THE SINGLE LAUNCHER.  All modes, all platforms, one file.     ║
║  See: .github/SINGLE_LAUNCHER_RULE.md                          ║
╚══════════════════════════════════════════════════════════════════╝

Usage
-----
    python launch.py <mode> [options]

Modes
-----
    dashboard   AAC Streamlit dashboard (web UI on :8501)
    healthmon   Terminal health monitor loop
    monitor     System monitor (terminal)
    paper       Paper trading engine
    core        Core orchestrator
    full        Full system (orchestrator + dashboard)
    test        Run pytest suite
    health      Health check
    git-sync    Git add/commit/push, then launch dashboard

Examples
--------
    python launch.py dashboard
    python launch.py paper
    python launch.py git-sync
    python launch.py test --verbose

Windows shortcut
----------------
    launch.bat dashboard
    launch.bat full

Unix / macOS
------------
    ./launch.sh paper
"""
from __future__ import annotations

from __future__ import annotations

import os
import sys

# pythonw.exe sets sys.stdout/stderr to None — redirect to devnull before ANY I/O
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

# Windows cp1252 console can't encode ANSI escapes / unicode glyphs we emit.
# Force UTF-8 on stdout/stderr so logger.info() doesn't crash on color codes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import argparse
import logging
import subprocess
import webbrowser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Python Version Guard ────────────────────────────────────────────────────
# Python 3.14+ causes aiohttp and other C-extension packages to hang.
# Require 3.9–3.13 for stable operation on both QUSAR and QFORGE.
if sys.version_info[:2] >= (3, 14):
    print(
        f"\033[93m  [!] Python {sys.version_info[0]}.{sys.version_info[1]} detected — "
        f"AAC requires Python 3.9–3.13.\033[0m"
    )
    logger.info(
        "\033[93m      Run: python setup_machine.py  to create a .venv with Python 3.12\033[0m"
    )
    _venv_py = (
        Path(__file__).resolve().parent
        / ".venv"
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    if _venv_py.exists():
        logger.info(f"\033[92m  [+] Found .venv — re-launching with {_venv_py}\033[0m")
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    # If no venv, continue but warn

# ── Constants ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

MODES = [
    "all",
    "api",
    "dashboard",
    "dashboard2",
    "deploy",
    "healthmon",
    "lde",
    "matrix",
    "monitor",
    "paper",
    "core",
    "full",
    "gateways",
    "preflight",
    "test",
    "health",
    "git-sync",
    "integrate",
    "war-room",
    "mission-control",
    "13-moon",
    "polymarket",
    "planktonxd-browser",
    "planktonxd-web-dashboard",
    "roadmap",
    "pnl",
    "openclaw",
    "schedule",
    "command",
    "autonomous",
    "console",
    "coder",
    "dfv",
    "dfv-brief",
]

BANNER = r"""
  +----------------------------------------------+
  |       BARREN WUFFET Trading System           |
  |       Codename: AZ SUPREME                   |
  |       Unified Launcher v3.6                   |
  |       517 components -- 12 doctrine packs     |
  +----------------------------------------------+
"""

MODE_DESCRIPTIONS = {
    "all": "Full startup: preflight -> gateways -> matrix monitor -> paper engine",
    "api": "Start FastAPI/uvicorn API server",
    "dashboard": "AAC Streamlit dashboard (web UI on :8501)",
    "dashboard2": "AAC Dashboard v2 — redesigned (sidebar nav, dark theme, port 8502)",
    "deploy": "Run production deployment with config validation",
    "healthmon": "Terminal health monitor loop (no web UI)",
    "matrix": "Matrix Monitor dashboard (--display terminal|web|dash)",
    "monitor": "System monitor (terminal)",
    "paper": "Paper trading engine",
    "core": "Core orchestrator",
    "full": "Full system (orchestrator + dashboard)",
    "gateways": "Start trading gateways (IBKR TWS, Moomoo OpenD)",
    "preflight": "Pre-flight validation (env, imports, config)",
    "test": "Run pytest suite",
    "health": "Health check",
    "git-sync": "Git add/commit/push, then launch dashboard",
    "integrate": "Run Unified Component Integrator -- wire all 550+ components",
    "war-room": "Start War Room Streamlit (kills stale instances, opens War Room + 13 Moon tabs)",
    "mission-control": "Unified Mission Control dashboard — single pane of glass (port 8069)",
    "13-moon": "13-Moon Doctrine live dashboard (Streamlit, port 8503)",
    "polymarket": "Polymarket Division — active scanning + execution (scan/monitor/live)",
    "planktonxd-browser": "PlanktonXD Browser Bot — automated browser-based prediction market harvester (--visible, --continuous)",
    "planktonxd-web-dashboard": "PlanktonXD Web Dashboard — browser-based control panel with clickable buttons for all bot commands (port 8088)",
    "roadmap": "Command Roadmap — daily/weekly tasks + 13-Moon + war room (HTML)",
    "pnl": "P&L report — open positions, today's P&L, 30-day history",
    "openclaw": "Connect AZ SUPREME to the OpenClaw Gateway (ws://127.0.0.1:18789)",
    "schedule": "Automated scheduler — signal scans, roll checks, P&L snapshots, health checks",
    "command": "Unified Command Dashboard (web, port 8400) — all Sprint 1-25 subsystems on one page",
    "autonomous": "Autonomous trading engine (continuous loop with heartbeat to data/autonomous_state.json)",
    "console": "Unified Command Console (terminal) -- same data as command mode, no browser",
    "coder": "Autonomous coder -- scan repo for drift patterns, emit backlog, optional --apply safe fixes",
    "dfv": "DFV / Roaring Kitty 24/7 daemon -- pre-market brief, midday, EOD, weekend DD on schedule",
    "dfv-brief": "DFV one-shot pre-market brief (prints to terminal, saves to agents/dfv/memory/briefs/)",
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _cyan(text: str) -> str:
    """Return text wrapped in ANSI cyan (no-op on dumb terminals)."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[96m{text}\033[0m"


def _green(text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[92m{text}\033[0m"


def _yellow(text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[93m{text}\033[0m"


def _red(text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[91m{text}\033[0m"


def _banner() -> None:
    for line in BANNER.strip().splitlines():
        logger.info(str(_cyan(line)))
    logger.info("")


def _activate_venv() -> None:
    """Activate .venv if present (adds to PATH + sys.path)."""
    if sys.platform == "win32":
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    if venv_python.exists():
        logger.info(str(_green("  [+] Virtual environment detected")))
        # On Windows the .venv/Scripts dir; on Unix .venv/bin
        venv_bin = str(venv_python.parent)
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        os.environ["VIRTUAL_ENV"] = str(PROJECT_ROOT / ".venv")
    else:
        logger.info(str(_yellow("  [!] No .venv found — using system Python")))


def _load_env() -> None:
    """Load .env if it exists (python-dotenv)."""
    env_file = PROJECT_ROOT / ".env"
    template = PROJECT_ROOT / ".env.template"

    if not env_file.exists() and template.exists():
        logger.info(str(_yellow("  [!] No .env found — copying from .env.template")))
        import shutil

        shutil.copy2(template, env_file)

    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
            logger.info(str(_green("  [+] Loaded .env")))
        except ImportError:
            logger.info(
                str(_yellow("  [!] python-dotenv not installed — .env not loaded"))
            )


def _python() -> str:
    """Return the Python interpreter path."""
    return sys.executable


def _run(cmd: list[str], **kwargs) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    logger.info(str(_green(f"  [>] {' '.join(cmd)}")))
    logger.info("")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), **kwargs)
    return result.returncode


# ── Mode Handlers ───────────────────────────────────────────────────────────


def _mode_gateways() -> int:
    """Start all trading gateways and verify connectivity."""
    from startup.gateways import gateway_summary, start_all_gateways

    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Starting Trading Gateways")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("")
    results = start_all_gateways()
    logger.info(gateway_summary(results))
    logger.info("")
    return 0


def _mode_matrix(display: str = "terminal", port: int = 8501) -> int:
    """Start the Matrix Monitor dashboard."""
    from startup.matrix_monitor import launch

    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan(f"  Matrix Monitor — {display} mode")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    return launch(display=display, port=port)


def _mode_preflight() -> int:
    """Run pre-flight validation checks."""
    from startup.preflight import run_all

    return 0 if run_all() else 1


def _mode_all(display: str = "web", port: int = 8501) -> int:
    """Full startup: preflight -> gateways -> health -> matrix monitor -> paper engine."""
    from startup.phases import full_startup

    return full_startup(display=display, port=port)


def _mode_dashboard(port: int = 8501) -> int:
    """Launch the AAC Streamlit dashboard (web UI)."""
    logger.info(str(_cyan(f"  Starting AAC Streamlit Dashboard on :{port} ...")))
    from startup.matrix_monitor import launch_web

    return launch_web(port=port)


def _mode_dashboard2(port: int = 8502) -> int:
    """Launch the redesigned v2 dashboard (sidebar nav, dark theme)."""
    import subprocess

    logger.info(str(_cyan(f"  Starting AAC Dashboard v2 on :{port} ...")))
    target = PROJECT_ROOT / "monitoring" / "dashboard_v2.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(target),
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    return subprocess.call(cmd)


def _mode_healthmon() -> int:
    """Terminal-only health monitor loop (legacy ``dashboard`` mode)."""
    logger.info(str(_cyan("  Starting Terminal Health Monitor ...")))
    from monitoring.health_monitor import HealthMonitor

    return HealthMonitor().run_loop(30)


def _mode_monitor() -> int:
    logger.info(str(_cyan("  Starting Health Monitor ...")))
    from monitoring.health_monitor import HealthMonitor

    return HealthMonitor().run_loop(30)


def _start_health_endpoint():
    """Start background health HTTP endpoint."""
    try:
        from health_server import start_health_server

        start_health_server(background=True)
        logger.info(str(_green("  [+] Health endpoint: http://localhost:8080/health")))
    except Exception as e:
        logger.info(str(_red(f"  [!] Health endpoint failed: {e}")))


def _mode_paper() -> int:
    from startup.gateways import start_all_gateways

    logger.info(str(_cyan("  Starting Paper Trading Engine ...")))
    logger.info(str(_cyan("  Pre-flight: checking gateways ...")))
    start_all_gateways()
    logger.info("")
    os.environ["PAPER_TRADING"] = "true"
    os.environ["LIVE_TRADING_ENABLED"] = "false"
    return _run([_python(), "-m", "core.orchestrator", "--paper"])


def _mode_core() -> int:
    logger.info(str(_cyan("  Starting Core Orchestrator ...")))
    return _run([_python(), "-m", "core.orchestrator"])


def _mode_full() -> int:
    logger.info(str(_cyan("  Starting Full System ...")))
    _run_compliance_preflight()
    return _run([_python(), "-m", "core.aac_master_launcher"])


def _run_compliance_preflight() -> None:
    """Run compliance review checks before full launch (warn-only)."""
    try:
        import asyncio

        from shared.compliance_review import ComplianceReviewSystem

        logger.info(str(_cyan("  Running compliance pre-flight checks ...")))
        system = ComplianceReviewSystem()
        report = asyncio.run(system.run_compliance_review())
        if report.overall_compliant:
            logger.info(str(_green("  [OK] Compliance pre-flight passed")))
        else:
            failed = [k for k, v in report.check_results.items() if not v.get("passed")]
            logger.info(
                str(
                    _red(
                        f"  [!] Compliance pre-flight: {len(failed)} check(s) failed: {', '.join(failed)}"
                    )
                )
            )
            logger.info(str(_red("  Review compliance before going live.")))
    except Exception as exc:
        logger.info(str(_red(f"  [!] Compliance pre-flight skipped: {exc}")))


def _mode_test(extra_args: list[str] | None = None) -> int:
    logger.info(str(_cyan("  Running Test Suite ...")))
    cmd = [
        _python(),
        "-m",
        "pytest",
        "tests/",
        "-q",
        "--tb=short",
        "-m",
        "not live and not exchange and not slow and not integration",
        "--timeout=30",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return _run(cmd)


def _mode_health() -> int:
    logger.info(str(_cyan("  Running Health Check ...")))
    script = PROJECT_ROOT / "scripts" / "health_check.py"
    if not script.exists():
        logger.info(str(_red("  [X] scripts/health_check.py not found")))
        return 1
    return _run([_python(), str(script)])


def _mode_git_sync() -> int:
    """Git add -> commit -> push, then launch dashboard."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Git Sync + Dashboard Launch")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("")

    # Check git status
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        logger.info(str(_red("  [X] Not a git repository")))
        return 1

    if not result.stdout.strip():
        logger.info(str(_green("  [+] Working tree clean — nothing to commit")))
    else:
        # Add all
        logger.info(str(_green("  [+] Adding changes ...")))
        subprocess.run(["git", "add", "."], cwd=str(PROJECT_ROOT))

        # Commit
        from datetime import datetime

        msg = f"Auto-commit: AAC system update — {datetime.now():%Y-%m-%d %H:%M:%S}"
        logger.info(str(_green(f"  [+] Committing: {msg}")))
        commit = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if commit.returncode == 0:
            logger.info(str(_green("  [+] Commit successful")))
            push = subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            if push.returncode == 0:
                logger.info(str(_green("  [+] Push successful")))
            else:
                logger.info(
                    str(_yellow("  [!] Push failed — check remote configuration"))
                )
        else:
            logger.info(str(_yellow("  [!] Nothing to commit")))

    logger.info("")
    return _mode_dashboard()


def _mode_api(port: int = 8000) -> int:
    """Start FastAPI/uvicorn API server."""
    logger.info(str(_cyan("  Starting API Server ...")))
    return _run(
        [
            _python(),
            "-m",
            "uvicorn",
            "core.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    )


def _mode_deploy() -> int:
    """Run production deployment with config validation."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Production Deployment")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("")
    _run_compliance_preflight()
    deploy_script = PROJECT_ROOT / "deployment" / "production_deployment_system.py"
    if not deploy_script.exists():
        logger.info(
            str(_red("  [X] deployment/production_deployment_system.py not found"))
        )
        return 1
    return _run([_python(), str(deploy_script)])


def _mode_integrate() -> int:
    """Run the Unified Component Integrator to wire all components."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Unified Component Integrator")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("")
    try:
        import asyncio

        from core.unified_component_integrator import get_unified_integrator

        integrator = get_unified_integrator(paper_mode=True)
        status = asyncio.run(integrator.integrate_all())
        logger.info("")
        logger.info(str(_green(f"  [+] Components wired: {status.components_wired}")))
        if status.components_failed:
            logger.info(
                str(_red(f"  [!] Components failed: {status.components_failed}"))
            )
        if status.errors:
            for err in status.errors:
                logger.info(str(_red(f"  [!] {err}")))
        return 0 if not status.errors else 1
    except Exception as e:
        logger.info(str(_red(f"  [X] Integration failed: {e}")))
        return 1


def _kill_existing_war_room_processes() -> None:
    """Kill stale War Room Streamlit processes to avoid stale code/port conflicts."""
    if os.name != "nt":
        return

    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -like '*streamlit*' "
        "-and $_.CommandLine -like '*monitoring/war_room_storyboard.py*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _mode_war_room(port: int = 8502, open_browsers: bool = True) -> int:
    """Start War Room Streamlit in background and optionally open browser tabs."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  War Room Startup")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))

    _kill_existing_war_room_processes()

    cmd = [
        _python(),
        "-m",
        "streamlit",
        "run",
        "monitoring/war_room_storyboard.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]

    popen_kwargs = {
        "cwd": str(PROJECT_ROOT),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    try:
        subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        logger.info(str(_red(f"  [X] Failed to start War Room: {e}")))
        return 1

    logger.info(str(_green(f"  [+] War Room started on http://localhost:{port}")))

    # Also export & open 13-Moon HTML storyboard
    try:
        from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine
        from strategies.thirteen_moon_storyboard import export_interactive_storyboard
        d = ThirteenMoonDoctrine()
        moon_path = export_interactive_storyboard(d)
        moon_url = "file:///" + os.path.abspath(moon_path).replace("\\", "/")
        logger.info(str(_green(f"  [+] 13-Moon storyboard exported to {moon_path}")))
    except Exception as e:
        moon_url = None
        logger.info(str(_yellow(f"  [!] Failed to export 13-Moon storyboard: {e}")))

    if open_browsers:
        try:
            war_room_url = f"http://localhost:{port}"
            webbrowser.open(war_room_url, new=2)
            if moon_url:
                webbrowser.open(moon_url, new=2)
            logger.info(str(_green("  [+] Opened War Room + 13 Moon browser tabs")))
        except Exception as e:
            logger.info(str(_yellow(f"  [!] Browser auto-open failed: {e}")))

    return 0


def _mode_thirteen_moon(port: int = 8503, open_browser: bool = True) -> int:
    """Export and open 13-Moon Doctrine HTML storyboard."""
    logger.info(str(_cyan("  13-Moon Doctrine Storyboard")))

    try:
        from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine
        from strategies.thirteen_moon_storyboard import export_interactive_storyboard
        d = ThirteenMoonDoctrine()
        path = export_interactive_storyboard(d)
        logger.info(str(_green(f"  [+] 13-Moon storyboard exported to {path}")))
    except Exception as e:
        logger.info(str(_red(f"  [X] Failed to export 13-Moon storyboard: {e}")))
        return 1

    if open_browser:
        try:
            file_url = "file:///" + os.path.abspath(path).replace("\\", "/")
            webbrowser.open(file_url, new=2)
        except Exception as e:
            logger.info(str(_yellow(f"  [!] Browser auto-open failed: {e}")))

    return 0


def _mode_roadmap(open_browser: bool = True, **_kw: object) -> int:
    """Export and open the unified Command Roadmap HTML dashboard."""
    logger.info(str(_cyan("  Command Roadmap — daily/weekly + 13-Moon + war room")))

    try:
        from strategies.roadmap_storyboard import export_roadmap
        path = export_roadmap()
        logger.info(str(_green(f"  [+] Roadmap exported to {path}")))
    except Exception as e:
        logger.info(str(_red(f"  [X] Failed to export roadmap: {e}")))
        return 1

    if open_browser:
        try:
            file_url = "file:///" + os.path.abspath(path).replace("\\", "/")
            webbrowser.open(file_url, new=2)
        except Exception as e:
            logger.info(str(_yellow(f"  [!] Browser auto-open failed: {e}")))

    return 0


def _mode_mission_control(port: int = 8069, open_browser: bool = True) -> int:
    """Start Mission Control — unified dashboard."""
    logger.info(str(_cyan("  [*] Starting Mission Control on port %d ...") % port))
    from monitoring.mission_control import run
    run(port=port, open_browser=open_browser)
    return 0


def _mode_polymarket() -> int:
    """Start Polymarket Active Scanner — all 3 strategies."""
    import asyncio as _asyncio
    logger.info(str(_cyan("  [*] Starting Polymarket Active Scanner ...")))
    logger.info(str(_cyan("  [*] Strategies: War Room + PlanktonXD + PolyMC")))
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    logger.info(str(_cyan(f"  [*] DRY_RUN={dry_run}")))

    from strategies.polymarket_division.active_scanner import ActiveScanner
    scanner = ActiveScanner(dry_run=dry_run)

    async def _run() -> int:
        opps = await scanner.scan_all()
        print(scanner.generate_report(opps))
        return 0

    return _asyncio.run(_run())


def _mode_planktonxd_browser() -> int:
    """Start PlanktonXD Browser Bot — browser automation prediction market harvester."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  PlanktonXD Browser Bot")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Browser-based PlanktonXD strategy emulation")))
    logger.info(str(_cyan("  Deep OTM harvesting via Selenium automation")))
    logger.info("")

    # Run the browser bot activation script
    script = PROJECT_ROOT / "scripts" / "activate_planktonxd_browser_bot.py"
    if not script.exists():
        logger.info(str(_red("  [X] PlanktonXD Browser Bot script not found")))
        return 1

    return _run([_python(), str(script)])


def _mode_planktonxd_web_dashboard() -> int:
    """Start PlanktonXD Web Dashboard — browser control panel with clickable buttons."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  PlanktonXD Web Dashboard")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Browser-based control panel with buttons")))
    logger.info(str(_cyan("  Real-time monitoring & command execution")))
    logger.info("")

    # Run the web dashboard
    dashboard_script = PROJECT_ROOT / "monitoring" / "planktonxd_browser_dashboard.py"
    if not dashboard_script.exists():
        logger.info(str(_red("  [X] PlanktonXD Web Dashboard script not found")))
        return 1

    logger.info(str(_green("  🚀 Starting PlanktonXD Web Dashboard...")))
    logger.info(str(_green("  🌐 Opening browser to http://localhost:8088")))
    logger.info(str(_green("  🎛️ Click buttons to execute bot commands")))
    logger.info("")

    return _run([_python(), str(dashboard_script)])


def _mode_pnl() -> int:
    """P&L report — positions, today's P&L, 30-day history."""
    from CentralAccounting.pnl_tracker import PnLTracker

    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  AAC P&L Report")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("")

    tracker = PnLTracker()

    # Try to get live positions from IBKR and take a fresh snapshot
    try:
        import asyncio
        from TradingExecution.position_tracker import PositionTracker

        paper = os.environ.get("PAPER_TRADING", "false").lower() == "true"
        pos_tracker = PositionTracker(paper=paper)
        positions = asyncio.run(pos_tracker.refresh())
        account_value = float(os.environ.get("ACCOUNT_VALUE_USD", "50000"))

        if positions:
            report = tracker.take_snapshot(positions, account_value)
            logger.info(str(_green(f"  [+] Snapshot taken: {len(positions)} live positions")))
        else:
            report = tracker.today_report()
            logger.info(str(_yellow("  [!] No live positions returned — reporting from DB")))
    except Exception as exc:
        logger.info(str(_yellow(f"  [!] IBKR not available ({exc}) — reporting from DB only")))
        report = tracker.today_report()

    print(PnLTracker.format_report(report))

    # 30-day history
    history = tracker.historical_summary(days=30)
    if len(history) > 1:
        logger.info(str(_cyan("  30-Day History")))
        logger.info(f"  {'Date':<12} {'Acct Value':>12} {'Unr P&L':>10} {'Rea P&L':>10} {'Pos':>4}")
        logger.info(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*4}")
        for row in history:
            logger.info(
                f"  {row['snapshot_date']:<12} "
                f"${row['account_value_usd']:>10,.2f} "
                f"${row['total_unrealized_pnl']:>8,.2f} "
                f"${row['total_realized_pnl']:>8,.2f} "
                f"{row['position_count']:>4}"
            )
        delta = tracker.pnl_delta(days=2)
        sign = "+" if delta >= 0 else ""
        logger.info(str(_green(f"  [+] 2-day P&L delta: {sign}${delta:,.2f}")))

    tracker.close()
    return 0


def _mode_openclaw() -> int:
    """Connect AZ SUPREME to the OpenClaw Gateway and keep it alive."""
    import asyncio

    async def _run():
        from integrations.openclaw_gateway_bridge import get_openclaw_bridge
        from integrations.openclaw_az_supreme_handler import initialize_az_supreme_openclaw_handler

        logger.info(str(_cyan("  ════════════════════════════════════════")))
        logger.info(str(_cyan("  AZ SUPREME → OpenClaw Gateway")))
        logger.info(str(_cyan("  ════════════════════════════════════════")))

        bridge = get_openclaw_bridge()
        connected = await bridge.connect()
        if not connected:
            logger.error("  ❌ Could not connect to OpenClaw Gateway at %s", bridge.gateway_url)
            logger.error("     Is the NCL OpenClaw server running?")
            return 1

        logger.info(str(_green(f"  🦞 Bridge CONNECTED: {bridge.gateway_url}")))

        await initialize_az_supreme_openclaw_handler(bridge=bridge)

        logger.info(str(_green("  👑 AZ SUPREME registered on OpenClaw")))
        logger.info("  Commands available: /status /briefing /risk /doctrine /agents /strategies /apis /help")
        logger.info("  Press Ctrl+C to disconnect.\n")

        try:
            # bridge.connect() already spawned _message_listener + _heartbeat_loop
            # Just keep the loop alive until disconnected or Ctrl+C
            while bridge._connected:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("  🛑 OpenClaw session terminated by user.")
        except Exception as exc:
            logger.error("  ❌ OpenClaw session error: %s", exc)
            return 1
        finally:
            await bridge.disconnect()
        return 0

    return asyncio.run(_run())


def _mode_schedule() -> int:
    """Automated market scheduler — runs unattended during NYSE hours."""
    from core.market_scheduler import MarketScheduler

    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  AAC Market Scheduler")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("  Tasks:")
    logger.info("    [*] Health check      every 5 min (always on)")
    logger.info("    [*] Signal scan       every 15 min (market hours only)")
    logger.info("    [*] Roll check        once daily at market open (9:30 ET)")
    logger.info("    [*] P&L snapshot      once daily at market close (16:00 ET)")
    import os as _os
    auto_execute = _os.getenv("AUTO_EXECUTE", "false").lower() == "true"
    logger.info(
        "    [*] Auto-execute      "
        + (str(_cyan("ON")) if auto_execute else "OFF  (set AUTO_EXECUTE=true to enable)")
    )
    logger.info("  Press Ctrl+C to stop.")
    logger.info("")

    sched = MarketScheduler(auto_execute=auto_execute)
    try:
        sched.run_forever(max_restarts=10)
    except KeyboardInterrupt:
        logger.info(str(_yellow("  [!] Scheduler interrupted by user.")))
    return 0


def _mode_lde(args: argparse.Namespace) -> int:
    """Launch the Living Doctrine Engine dashboard."""
    port = getattr(args, "port", None) or 8510
    logger.info(str(_cyan(f"  [*] Starting LDE Dashboard on port {port} ...")))
    from monitoring.lde_dashboard import run_dashboard
    run_dashboard(port=port)
    return 0


def _mode_command(args: argparse.Namespace) -> int:
    """Launch the unified Command Dashboard (Sprint 1-25 subsystems on one page)."""
    port = getattr(args, "port", None)
    if not port or port == 8501:
        port = 8400
    logger.info(str(_cyan(f"  [*] Starting AAC Command Dashboard on port {port} ...")))
    from monitoring.command_dashboard import run_dashboard
    return run_dashboard(port=port)


def _mode_autonomous(args: argparse.Namespace) -> int:
    """Launch the Autonomous Engine continuous loop (heartbeats to data/autonomous_state.json)."""
    logger.info(str(_cyan("  [*] Starting AAC Autonomous Engine ...")))
    import asyncio
    from core.autonomous_engine import AutonomousEngine
    engine = AutonomousEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        logger.info(str(_cyan("  [*] Autonomous engine stopped by user")))
    return 0


def _mode_console(args: argparse.Namespace) -> int:
    """Launch the Command Console (terminal-rendered twin of the web command dashboard)."""
    logger.info(str(_cyan("  [*] Starting AAC Command Console ...")))
    from monitoring.command_console import run_console
    interval = float(getattr(args, "interval", None) or 5.0)
    return run_console(interval=interval)


def _mode_coder(args: argparse.Namespace, extra: list[str]) -> int:
    """Run the autonomous coder (deterministic scanner + safe-fix applier)."""
    logger.info(str(_cyan("  [*] Running AAC Autonomous Coder ...")))
    from tools.autonomous_coder import main as coder_main
    return coder_main(extra)


def _mode_dfv() -> int:
    """DFV / Roaring Kitty 24/7 daemon — schedules brief, midday, EOD, weekend DD."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  🐱 DFV — Roaring Kitty Operator (24/7)")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Cadence: pre-market 07:30 · midday 12:00 · EOD 17:00 ET")))
    logger.info(str(_cyan("  Doctrine: config/doctrine/dfv_doctrine.yaml")))
    logger.info(str(_cyan("  Briefs   : agents/dfv/memory/briefs/")))
    logger.info("")
    from agents.dfv.daemon import run_forever
    run_forever()
    return 0


def _mode_dfv_brief() -> int:
    """Run a one-shot DFV pre-market brief and print the headline."""
    from agents.dfv.routines import brief
    out = brief()
    logger.info("")
    logger.info(str(_cyan(f"  🐱 DFV Brief — {out['generated_at']}")))
    logger.info(str(_cyan("  ────────────────────────────────────────")))
    logger.info(f"  {out['headline']}")
    logger.info("")
    ps = out['portfolio_summary']
    logger.info(f"  Equity:    ${(ps['total_equity_usd'] or 0):,.0f}")
    logger.info(f"  Cash:      ${(ps['cash_usd'] or 0):,.0f}")
    logger.info(f"  BP:        ${(ps['buying_power_usd'] or 0):,.0f}")
    logger.info(f"  Positions: {ps['open_positions']}")
    logger.info("")
    logger.info(f"  Saved → {out.get('saved_to', '?')}")
    return 0


# ── Dispatch ────────────────────────────────────────────────────────────────

MODE_DISPATCH = {
    "all": _mode_all,
    "api": _mode_api,
    "dashboard": _mode_dashboard,
    "dashboard2": _mode_dashboard2,
    "deploy": _mode_deploy,
    "healthmon": _mode_healthmon,
    "lde": _mode_lde,
    "matrix": _mode_matrix,
    "monitor": _mode_monitor,
    "paper": _mode_paper,
    "core": _mode_core,
    "full": _mode_full,
    "gateways": _mode_gateways,
    "preflight": _mode_preflight,
    "test": _mode_test,
    "health": _mode_health,
    "git-sync": _mode_git_sync,
    "integrate": _mode_integrate,
    "war-room": _mode_war_room,
    "mission-control": _mode_mission_control,
    "13-moon": _mode_thirteen_moon,
    "polymarket": _mode_polymarket,
    "planktonxd-browser": _mode_planktonxd_browser,
    "planktonxd-web-dashboard": _mode_planktonxd_web_dashboard,
    "roadmap": _mode_roadmap,
    "pnl": _mode_pnl,
    "openclaw": _mode_openclaw,
    "schedule": _mode_schedule,
    "command": _mode_command,
    "autonomous": _mode_autonomous,
    "console": _mode_console,
    "coder": _mode_coder,
    "dfv": _mode_dfv,
    "dfv-brief": _mode_dfv_brief,
}


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    """Main."""
    parser = argparse.ArgumentParser(
        prog="launch",
        description="BARREN WUFFET - Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {m:<12} {d}" for m, d in MODE_DESCRIPTIONS.items()),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="health",
        choices=MODES,
        help="Launch mode (default: health)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Extra output for test / debug modes",
    )
    parser.add_argument(
        "--display",
        choices=["terminal", "web", "dash"],
        default="terminal",
        help="Display mode for matrix monitor (default: terminal)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for matrix/api/war-room modes (default: 8501)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open browser tabs when starting war-room mode",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Refresh interval in seconds for console mode (default: 5)",
    )
    args, extra = parser.parse_known_args()

    _banner()

    # Show Python + mode
    logger.info(str(_green(f"  [+] Python: {sys.executable}")))
    logger.info(str(_green(f"  [+] Mode:   {args.mode}")))
    logger.info("")

    # Environment setup
    os.chdir(PROJECT_ROOT)
    _activate_venv()
    _load_env()

    # Startup config validation (skip for non-trading modes)
    if args.mode not in ("test", "health", "git-sync", "preflight", "pnl", "schedule"):
        try:
            from shared.config_loader import validate_startup_requirements

            if not validate_startup_requirements():
                logger.info(
                    str(_red("  [!] Configuration validation FAILED — check .env"))
                )
                logger.info(str(_red("  [!] Set exchange API keys or DRY_RUN=true")))
                return 1
        except Exception as e:
            logger.info(str(_red(f"  [!] Config validation error: {e}")))

    logger.info("")

    # Dispatch
    handler = MODE_DISPATCH[args.mode]
    if args.mode == "test":
        extra_args = extra or []
        if args.verbose:
            extra_args.append("-v")
        return handler(extra_args)
    elif args.mode == "matrix":
        return handler(display=args.display, port=args.port)
    elif args.mode == "all":
        display = args.display if args.display != "terminal" else "web"
        return handler(display=display, port=args.port)
    elif args.mode == "api":
        return handler(port=args.port)
    elif args.mode == "war-room":
        port = args.port if args.port != 8501 else 8502
        return handler(port=port, open_browsers=not args.no_browser)
    elif args.mode == "mission-control":
        port = args.port if args.port != 8501 else 8069
        return handler(port=port, open_browser=not args.no_browser)
    elif args.mode == "lde":
        return handler(args)
    elif args.mode == "command":
        return handler(args)
    elif args.mode == "autonomous":
        return handler(args)
    elif args.mode == "console":
        return handler(args)
    elif args.mode == "coder":
        return handler(args, extra or [])
    elif args.mode in ("dfv", "dfv-brief"):
        return handler()
    else:
        return handler()


if __name__ == "__main__":
    raise SystemExit(main())

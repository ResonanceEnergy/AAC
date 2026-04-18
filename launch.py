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
    dashboard   Dash monitoring dashboard (web UI)
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

import os
import sys

# pythonw.exe sets sys.stdout/stderr to None — redirect to devnull before ANY I/O
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

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
    "deploy",
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
    "roadmap",
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
    "dashboard": "Dash monitoring dashboard (web UI)",
    "deploy": "Run production deployment with config validation",
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
    "roadmap": "Command Roadmap — daily/weekly tasks + 13-Moon + war room (HTML)",
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


def _mode_dashboard() -> int:
    logger.info(str(_cyan("  Starting Dashboard ...")))
    return _run([_python(), "-m", "monitoring.aac_master_monitoring_dashboard", "--mode", "web"])


def _mode_monitor() -> int:
    logger.info(str(_cyan("  Starting System Monitor ...")))
    return _run([_python(), "-m", "shared.system_monitor"])


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
        "not live and not exchange and not slow",
        "--timeout=15",
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


def _mode_lde(args: argparse.Namespace) -> int:
    """Launch the Living Doctrine Engine dashboard."""
    port = getattr(args, "port", None) or 8510
    logger.info(str(_cyan(f"  [*] Starting LDE Dashboard on port {port} ...")))
    from monitoring.lde_dashboard import run_dashboard
    run_dashboard(port=port)
    return 0


# ── Dispatch ────────────────────────────────────────────────────────────────

MODE_DISPATCH = {
    "all": _mode_all,
    "api": _mode_api,
    "dashboard": _mode_dashboard,
    "deploy": _mode_deploy,
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
    "roadmap": _mode_roadmap,
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
    if args.mode not in ("test", "health", "git-sync", "preflight"):
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
    else:
        return handler()


if __name__ == "__main__":
    raise SystemExit(main())

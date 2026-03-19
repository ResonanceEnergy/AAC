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

import argparse
import os
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Python Version Guard ────────────────────────────────────────────────────
# Python 3.14+ causes aiohttp and other C-extension packages to hang.
# Require 3.9–3.13 for stable operation on both QUSAR and QFORGE.
if sys.version_info[:2] >= (3, 14):
    print(f"\033[93m  [!] Python {sys.version_info[0]}.{sys.version_info[1]} detected — "
          f"AAC requires Python 3.9–3.13.\033[0m")
    logger.info(f"\033[93m      Run: python setup_machine.py  to create a .venv with Python 3.12\033[0m")
    _venv_py = Path(__file__).resolve().parent / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
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
]

BANNER = r"""
  ╔══════════════════════════════════════════════╗
  ║       BARREN WUFFET Trading System           ║
  ║       Codename: AZ SUPREME                   ║
  ║       Unified Launcher v1.0                   ║
  ╚══════════════════════════════════════════════╝
"""

MODE_DESCRIPTIONS = {
    "all":       "Full startup: preflight → gateways → matrix monitor → paper engine",
    "api":       "Start FastAPI/uvicorn API server",
    "dashboard": "Dash monitoring dashboard (web UI)",
    "deploy":    "Run production deployment with config validation",
    "matrix":    "Matrix Monitor dashboard (--display terminal|web|dash)",
    "monitor":   "System monitor (terminal)",
    "paper":     "Paper trading engine",
    "core":      "Core orchestrator",
    "full":      "Full system (orchestrator + dashboard)",
    "gateways":  "Start trading gateways (IBKR TWS, Moomoo OpenD)",
    "preflight": "Pre-flight validation (env, imports, config)",
    "test":      "Run pytest suite",
    "health":    "Health check",
    "git-sync":  "Git add/commit/push, then launch dashboard",
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
            logger.info(str(_yellow("  [!] python-dotenv not installed — .env not loaded")))


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
    from startup.gateways import start_all_gateways, gateway_summary
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
    """Full startup: preflight → gateways → health → matrix monitor → paper engine."""
    from startup.phases import full_startup
    return full_startup(display=display, port=port)


def _mode_dashboard() -> int:
    logger.info(str(_cyan("  Starting Dashboard ...")))
    return _run([_python(), "-m", "core.command_center", "--mode", "dashboard"])


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
            logger.info(str(_red(f"  [!] Compliance pre-flight: {len(failed)} check(s) failed: {', '.join(failed)}")))
            logger.info(str(_red("  Review compliance before going live.")))
    except Exception as exc:
        logger.info(str(_red(f"  [!] Compliance pre-flight skipped: {exc}")))


def _mode_test(extra_args: list[str] | None = None) -> int:
    logger.info(str(_cyan("  Running Test Suite ...")))
    cmd = [
        _python(), "-m", "pytest", "tests/",
        "-q", "--tb=short",
        "-m", "not live and not exchange and not slow",
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
    """Git add → commit → push, then launch dashboard."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Git Sync + Dashboard Launch")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("")

    # Check git status
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
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
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if commit.returncode == 0:
            logger.info(str(_green("  [+] Commit successful")))
            push = subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            if push.returncode == 0:
                logger.info(str(_green("  [+] Push successful")))
            else:
                logger.info(str(_yellow("  [!] Push failed — check remote configuration")))
        else:
            logger.info(str(_yellow("  [!] Nothing to commit")))

    logger.info("")
    return _mode_dashboard()


def _mode_api(port: int = 8000) -> int:
    """Start FastAPI/uvicorn API server."""
    logger.info(str(_cyan("  Starting API Server ...")))
    return _run([
        _python(), "-m", "uvicorn",
        "core.api:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload",
    ])


def _mode_deploy() -> int:
    """Run production deployment with config validation."""
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info(str(_cyan("  Production Deployment")))
    logger.info(str(_cyan("  ════════════════════════════════════════")))
    logger.info("")
    _run_compliance_preflight()
    deploy_script = PROJECT_ROOT / "deployment" / "production_deployment_system.py"
    if not deploy_script.exists():
        logger.info(str(_red("  [X] deployment/production_deployment_system.py not found")))
        return 1
    return _run([_python(), str(deploy_script)])


# ── Dispatch ────────────────────────────────────────────────────────────────

MODE_DISPATCH = {
    "all":       _mode_all,
    "api":       _mode_api,
    "dashboard": _mode_dashboard,
    "deploy":    _mode_deploy,
    "matrix":    _mode_matrix,
    "monitor":   _mode_monitor,
    "paper":     _mode_paper,
    "core":      _mode_core,
    "full":      _mode_full,
    "gateways":  _mode_gateways,
    "preflight": _mode_preflight,
    "test":      _mode_test,
    "health":    _mode_health,
    "git-sync":  _mode_git_sync,
}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    """Main."""
    parser = argparse.ArgumentParser(
        prog="launch",
        description="BARREN WUFFET — Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {m:<12} {d}" for m, d in MODE_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="health",
        choices=MODES,
        help="Launch mode (default: health)",
    )
    parser.add_argument(
        "--verbose", "-v",
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
        help="Port for matrix monitor web/dash modes (default: 8501)",
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
                logger.info(str(_red("  [!] Configuration validation FAILED — check .env")))
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
    else:
        return handler()


if __name__ == "__main__":
    raise SystemExit(main())

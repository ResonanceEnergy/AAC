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

# ── Python Version Guard ────────────────────────────────────────────────────
# Python 3.14+ causes aiohttp and other C-extension packages to hang.
# Require 3.9–3.13 for stable operation on both QUSAR and QFORGE.
if sys.version_info[:2] >= (3, 14):
    print(f"\033[93m  [!] Python {sys.version_info[0]}.{sys.version_info[1]} detected — "
          f"AAC requires Python 3.9–3.13.\033[0m")
    print(f"\033[93m      Run: python setup_machine.py  to create a .venv with Python 3.12\033[0m")
    _venv_py = Path(__file__).resolve().parent / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if _venv_py.exists():
        print(f"\033[92m  [+] Found .venv — re-launching with {_venv_py}\033[0m")
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    # If no venv, continue but warn

# ── Constants ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

MODES = [
    "dashboard",
    "monitor",
    "paper",
    "core",
    "full",
    "gateways",
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
    "dashboard": "Dash monitoring dashboard (web UI)",
    "monitor":   "System monitor (terminal)",
    "paper":     "Paper trading engine",
    "core":      "Core orchestrator",
    "full":      "Full system (orchestrator + dashboard)",
    "gateways":  "Start trading gateways (IBKR TWS, Moomoo OpenD)",
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
        print(_cyan(line))
    print()


def _activate_venv() -> None:
    """Activate .venv if present (adds to PATH + sys.path)."""
    if sys.platform == "win32":
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    if venv_python.exists():
        print(_green("  [+] Virtual environment detected"))
        # On Windows the .venv/Scripts dir; on Unix .venv/bin
        venv_bin = str(venv_python.parent)
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        os.environ["VIRTUAL_ENV"] = str(PROJECT_ROOT / ".venv")
    else:
        print(_yellow("  [!] No .venv found — using system Python"))


def _load_env() -> None:
    """Load .env if it exists (python-dotenv)."""
    env_file = PROJECT_ROOT / ".env"
    template = PROJECT_ROOT / ".env.template"

    if not env_file.exists() and template.exists():
        print(_yellow("  [!] No .env found — copying from .env.template"))
        import shutil
        shutil.copy2(template, env_file)

    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(_green("  [+] Loaded .env"))
        except ImportError:
            print(_yellow("  [!] python-dotenv not installed — .env not loaded"))


def _python() -> str:
    """Return the Python interpreter path."""
    return sys.executable


def _run(cmd: list[str], **kwargs) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    print(_green(f"  [>] {' '.join(cmd)}"))
    print()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), **kwargs)
    return result.returncode


# ── Gateway Paths ───────────────────────────────────────────────────────────

_GATEWAY_CONFIGS = {
    "ibkr": {
        "name": "IBKR TWS",
        "exe": Path(r"C:\Jts\tws.exe"),
        "process_name": "tws",
        "host": "127.0.0.1",
        "port": 7497,
        "wait_secs": 30,
    },
    "moomoo": {
        "name": "Moomoo OpenD",
        "exe": Path(r"C:\FutuOpenD\moomoo_OpenD_10.0.6018_Windows\moomoo_OpenD-GUI_10.0.6018_Windows\moomoo_OpenD-GUI_10.0.6018_Windows.exe"),
        "process_name": "moomoo_OpenD-GUI",
        "host": "127.0.0.1",
        "port": 11111,
        "wait_secs": 15,
    },
}


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _is_process_running(name: str) -> bool:
    """Check if a process with the given name fragment is running."""
    import re
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}*", "/NH"],
        capture_output=True, text=True,
    )
    return name.lower() in result.stdout.lower()


def _start_gateway(key: str) -> bool:
    """Start a gateway if not already running. Returns True if port is live."""
    cfg = _GATEWAY_CONFIGS[key]
    name = cfg["name"]
    host, port = cfg["host"], cfg["port"]

    # Already listening?
    if _port_open(host, port):
        print(_green(f"  [+] {name}: already running on {host}:{port}"))
        return True

    # Exe exists?
    if not cfg["exe"].exists():
        print(_yellow(f"  [!] {name}: not installed ({cfg['exe']})"))
        return False

    # Launch
    print(_cyan(f"  [>] Starting {name} ..."))
    try:
        subprocess.Popen(
            [str(cfg["exe"])],
            cwd=str(cfg["exe"].parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        # WinError 740: needs elevation — use shell start
        os.startfile(str(cfg["exe"]))

    # Wait for port
    import time
    deadline = time.time() + cfg["wait_secs"]
    dots = 0
    while time.time() < deadline:
        if _port_open(host, port, timeout=1.0):
            print(_green(f"\n  [+] {name}: LIVE on {host}:{port}"))
            return True
        dots += 1
        print(".", end="", flush=True)
        time.sleep(2)

    # Port not ready — process may need manual login
    if _is_process_running(cfg["process_name"]):
        print(_yellow(f"\n  [!] {name}: running but port {port} not ready — log in to the GUI"))
    else:
        print(_red(f"\n  [X] {name}: failed to start"))
    return False


def _start_all_gateways() -> dict[str, bool]:
    """Start all configured gateways. Returns status dict."""
    results = {}
    for key in _GATEWAY_CONFIGS:
        results[key] = _start_gateway(key)
    return results


# ── Mode Handlers ───────────────────────────────────────────────────────────

def _mode_gateways() -> int:
    """Start all trading gateways and verify connectivity."""
    print(_cyan("  ════════════════════════════════════════"))
    print(_cyan("  Starting Trading Gateways"))
    print(_cyan("  ════════════════════════════════════════"))
    print()

    results = _start_all_gateways()
    print()

    live = sum(1 for v in results.values() if v)
    total = len(results)
    print(_green(f"  Gateways: {live}/{total} live"))
    for key, ok in results.items():
        cfg = _GATEWAY_CONFIGS[key]
        status = _green("LIVE") if ok else _yellow("WAITING")
        print(f"    {cfg['name']:<20s} {cfg['host']}:{cfg['port']}  {status}")
    print()
    return 0


def _mode_dashboard() -> int:
    print(_cyan("  Starting Dashboard ..."))
    return _run([_python(), "-m", "core.command_center", "--mode", "dashboard"])


def _mode_monitor() -> int:
    print(_cyan("  Starting System Monitor ..."))
    return _run([_python(), "-m", "SharedInfrastructure.system_monitor"])


def _mode_paper() -> int:
    print(_cyan("  Starting Paper Trading Engine ..."))
    print(_cyan("  Pre-flight: checking gateways ..."))
    _start_all_gateways()
    print()
    os.environ["PAPER_TRADING"] = "true"
    os.environ["LIVE_TRADING_ENABLED"] = "false"
    return _run([_python(), "-m", "core.orchestrator", "--paper"])


def _mode_core() -> int:
    print(_cyan("  Starting Core Orchestrator ..."))
    return _run([_python(), "-m", "core.orchestrator"])


def _mode_full() -> int:
    print(_cyan("  Starting Full System ..."))
    print(_cyan("  Pre-flight: checking gateways ..."))
    _start_all_gateways()
    print()
    return _run([_python(), "-m", "core.aac_master_launcher"])


def _mode_test(extra_args: list[str] | None = None) -> int:
    print(_cyan("  Running Test Suite ..."))
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
    print(_cyan("  Running Health Check ..."))
    script = PROJECT_ROOT / "scripts" / "health_check.py"
    if not script.exists():
        print(_red("  [X] scripts/health_check.py not found"))
        return 1
    return _run([_python(), str(script)])


def _mode_git_sync() -> int:
    """Git add → commit → push, then launch dashboard."""
    print(_cyan("  ════════════════════════════════════════"))
    print(_cyan("  Git Sync + Dashboard Launch"))
    print(_cyan("  ════════════════════════════════════════"))
    print()

    # Check git status
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(_red("  [X] Not a git repository"))
        return 1

    if not result.stdout.strip():
        print(_green("  [+] Working tree clean — nothing to commit"))
    else:
        # Add all
        print(_green("  [+] Adding changes ..."))
        subprocess.run(["git", "add", "."], cwd=str(PROJECT_ROOT))

        # Commit
        from datetime import datetime
        msg = f"Auto-commit: AAC system update — {datetime.now():%Y-%m-%d %H:%M:%S}"
        print(_green(f"  [+] Committing: {msg}"))
        commit = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if commit.returncode == 0:
            print(_green("  [+] Commit successful"))
            push = subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            if push.returncode == 0:
                print(_green("  [+] Push successful"))
            else:
                print(_yellow("  [!] Push failed — check remote configuration"))
        else:
            print(_yellow("  [!] Nothing to commit"))

    print()
    return _mode_dashboard()


# ── Dispatch ────────────────────────────────────────────────────────────────

MODE_DISPATCH = {
    "dashboard": _mode_dashboard,
    "monitor":   _mode_monitor,
    "paper":     _mode_paper,
    "core":      _mode_core,
    "full":      _mode_full,
    "gateways":  _mode_gateways,
    "test":      _mode_test,
    "health":    _mode_health,
    "git-sync":  _mode_git_sync,
}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
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

    args, extra = parser.parse_known_args()

    _banner()

    # Show Python + mode
    print(_green(f"  [+] Python: {sys.executable}"))
    print(_green(f"  [+] Mode:   {args.mode}"))
    print()

    # Environment setup
    os.chdir(PROJECT_ROOT)
    _activate_venv()
    _load_env()
    print()

    # Dispatch
    handler = MODE_DISPATCH[args.mode]
    if args.mode == "test":
        extra_args = extra or []
        if args.verbose:
            extra_args.append("-v")
        return handler(extra_args)
    else:
        return handler()


if __name__ == "__main__":
    raise SystemExit(main())

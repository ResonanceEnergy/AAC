#!/usr/bin/env python3
"""
AAC Health Check Script
=======================
Quick diagnostic to verify the AAC system is properly configured and ready to run.

Usage:
    python scripts/health_check.py
    # or via Makefile:
    make health
"""

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Colour helpers ────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


# ── Checks ────────────────────────────────────────────────────────────────────

def check_python_version() -> bool:
    v = sys.version_info
    if v >= (3, 9):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    fail(f"Python {v.major}.{v.minor}.{v.micro} — requires >=3.9")
    return False


def check_env_file() -> bool:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        ok(".env file present")
        return True
    template = PROJECT_ROOT / ".env.template"
    if template.exists():
        warn(".env missing — copy from .env.template")
    else:
        fail(".env missing and no .env.template found")
    return False


def check_venv() -> bool:
    venv = PROJECT_ROOT / ".venv"
    if venv.is_dir():
        ok(f".venv exists ({sys.executable})")
        return True
    warn("No .venv directory found")
    return False


def check_core_imports() -> int:
    """Try importing core modules. Returns count of failures."""
    modules = [
        ("shared.audit_logger", "AuditLogger"),
        ("shared.market_data_connector", "MarketDataConnector"),
        ("shared.monitoring", None),
        ("core.orchestrator", "AACOrchestrator"),
        ("TradingExecution.execution_engine", "ExecutionEngine"),
        ("BigBrainIntelligence.agents", None),
        ("CentralAccounting.database", None),
        ("CryptoIntelligence.crypto_intelligence", None),
    ]
    failures = 0
    for mod_name, attr in modules:
        try:
            mod = importlib.import_module(mod_name)
            if attr and not hasattr(mod, attr):
                warn(f"{mod_name}.{attr} not found")
                failures += 1
            else:
                ok(f"import {mod_name}")
        except Exception as exc:
            fail(f"import {mod_name}: {exc.__class__.__name__}: {exc}")
            failures += 1
    return failures


def check_required_packages() -> int:
    """Check critical pip packages are installed."""
    packages = [
        "dotenv",        # python-dotenv
        "aiohttp",
        "pandas",
        "numpy",
        "ccxt",
        "sqlalchemy",
        "dash",
        "fastapi",
        "pydantic",
        "structlog",
        "rich",
    ]
    failures = 0
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            ok(f"package {pkg}")
        except ImportError:
            fail(f"package {pkg} not installed")
            failures += 1
    return failures


def check_directory_structure() -> int:
    """Verify key directories exist."""
    dirs = [
        "shared",
        "core",
        "strategies",
        "TradingExecution",
        "BigBrainIntelligence",
        "CentralAccounting",
        "CryptoIntelligence",
        "monitoring",
        "tests",
        "agents",
    ]
    failures = 0
    for d in dirs:
        p = PROJECT_ROOT / d
        if p.is_dir():
            py_count = len(list(p.glob("*.py")))
            ok(f"{d}/ ({py_count} .py files)")
        else:
            fail(f"{d}/ missing")
            failures += 1
    return failures


def check_env_vars() -> None:
    """Check key environment variables."""
    env_vars = {
        "AAC_ENV": ("test", "production", "development"),
        "PAPER_TRADING": ("true", "false"),
        "LIVE_TRADING_ENABLED": ("true", "false"),
    }
    for var, valid_values in env_vars.items():
        val = os.environ.get(var)
        if val:
            ok(f"${var} = {val}")
        else:
            warn(f"${var} not set (default will be used)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}AAC System Health Check{RESET}")
    print("=" * 50)

    total_failures = 0

    print(f"\n{BOLD}1. Environment{RESET}")
    if not check_python_version():
        total_failures += 1
    check_venv()
    check_env_file()
    check_env_vars()

    print(f"\n{BOLD}2. Directory Structure{RESET}")
    total_failures += check_directory_structure()

    print(f"\n{BOLD}3. Required Packages{RESET}")
    total_failures += check_required_packages()

    print(f"\n{BOLD}4. Core Module Imports{RESET}")
    total_failures += check_core_imports()

    # Summary
    print("\n" + "=" * 50)
    if total_failures == 0:
        print(f"{GREEN}{BOLD}All checks passed! System is healthy.{RESET}")
    else:
        print(f"{RED}{BOLD}{total_failures} check(s) failed.{RESET}")
        print("Run: pip install -r requirements.txt")

    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

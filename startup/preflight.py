"""
startup.preflight — Pre-flight validation checks.

Quick sanity checks before any trading mode starts.
Extracted from automate.py phases 1-6.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python_version() -> bool:
    """Ensure Python 3.9–3.13 (3.14+ breaks aiohttp)."""
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 14):
        logger.error(f"  [X] Python {major}.{minor} detected — AAC requires 3.9–3.13")
        return False
    if (major, minor) < (3, 9):
        logger.error(f"  [X] Python {major}.{minor} too old — AAC requires 3.9+")
        return False
    logger.info(f"  [OK] Python {major}.{minor}")
    return True


def check_venv() -> bool:
    """Verify .venv exists and we're running inside it (or system Python)."""
    venv_dir = PROJECT_ROOT / ".venv"
    if venv_dir.exists():
        logger.info("  [OK] .venv present")
    else:
        logger.warning("  [!] No .venv — using system Python")
    return True


def check_env_file() -> bool:
    """Verify .env exists (copy from template if not)."""
    env_file = PROJECT_ROOT / ".env"
    template = PROJECT_ROOT / ".env.template"
    if env_file.exists():
        logger.info("  [OK] .env present")
        return True
    if template.exists():
        import shutil
        shutil.copy2(template, env_file)
        logger.info("  [OK] .env created from .env.template")
        return True
    logger.warning("  [!] No .env or .env.template found")
    return False


def check_core_imports() -> bool:
    """Smoke-test critical AAC imports."""
    modules = [
        "shared.config_loader",
        "shared.data_sources",
        "shared.audit_logger",
        "CentralAccounting.database",
        "TradingExecution.execution_engine",
        "strategies.golden_ratio_finance",
    ]
    ok = True
    for mod in modules:
        try:
            importlib.import_module(mod)
            logger.info(f"  [OK] {mod}")
        except Exception as e:
            logger.error(f"  [X] {mod}: {e}")
            ok = False
    return ok


def check_config_validation() -> bool:
    """Run shared.config_loader.validate_startup_requirements()."""
    try:
        from shared.config_loader import validate_startup_requirements
        if validate_startup_requirements():
            logger.info("  [OK] Config validation passed")
            return True
        else:
            logger.error("  [X] Config validation FAILED — check .env")
            return False
    except Exception as e:
        logger.error(f"  [X] Config validation error: {e}")
        return False


def run_all() -> bool:
    """Run all pre-flight checks. Returns True if all critical checks pass."""
    logger.info("  ═══════════════════════════════════════")
    logger.info("  Pre-Flight Checks")
    logger.info("  ═══════════════════════════════════════")

    results = {
        "python_version": check_python_version(),
        "venv": check_venv(),
        "env_file": check_env_file(),
        "core_imports": check_core_imports(),
        "config": check_config_validation(),
    }

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("")
    logger.info(f"  Pre-flight: {passed}/{total} passed")

    # Python version is the only hard blocker
    return results["python_version"] and results["config"]

"""Smoke-test: import every critical module and FAIL LOUDLY if anything is broken.

Run with:  python -m pytest tests/test_smoke_imports.py -v
Or:        .venv/Scripts/python -m pytest tests/test_smoke_imports.py -v

This file intentionally does NOT use try/except — if an import fails,
the test must crash with a full traceback so you can fix it immediately.
"""

import importlib
import pytest


# ── Tier 1: Core infrastructure (MUST work) ────────────────────────────
TIER1_MODULES = [
    "shared.config_loader",
    "shared.data_sources",
    "shared.audit_logger",
    "shared.strategy_execution_engine",
    "CentralAccounting.database",
    "CentralAccounting.financial_analysis_engine",
]

# ── Tier 2: Trading / execution (SHOULD work) ──────────────────────────
TIER2_MODULES = [
    "TradingExecution.execution_engine",
    "TradingExecution.order_manager",
    "TradingExecution.risk_manager",
    "BigBrainIntelligence.agents",
    "BigBrainIntelligence.research_agent",
    "core.sub_agent_spawner",
]

# ── Tier 3: Package-level inits (should load without hiding errors) ────
TIER3_MODULES = [
    "core",
    "config",
    "agents",
]


@pytest.mark.parametrize("module_name", TIER1_MODULES, ids=TIER1_MODULES)
def test_tier1_import(module_name):
    """Tier 1 modules are core infrastructure — failure here is critical."""
    mod = importlib.import_module(module_name)
    assert mod is not None, f"{module_name} imported as None"


@pytest.mark.parametrize("module_name", TIER2_MODULES, ids=TIER2_MODULES)
def test_tier2_import(module_name):
    """Tier 2 modules are trading/analysis — failure blocks main features."""
    mod = importlib.import_module(module_name)
    assert mod is not None, f"{module_name} imported as None"


@pytest.mark.parametrize("module_name", TIER3_MODULES, ids=TIER3_MODULES)
def test_tier3_package_init(module_name):
    """Tier 3: package __init__ files should load without swallowing errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None, f"{module_name} imported as None"


# ── Specific assertions: core exports should NOT be None ────────────────
def test_core_exports_not_none():
    """After fixing except-Exception, real errors surface instead of None."""
    import core

    # These are the exports from core/__init__.py
    # If any are None, it means the import failed (and now we'll see why)
    none_exports = []
    for name in core.__all__:
        val = getattr(core, name, "MISSING")
        if val is None:
            none_exports.append(name)
        elif val == "MISSING":
            none_exports.append(f"{name} (missing from module)")

    if none_exports:
        pytest.fail(
            f"core/__init__.py exported None for: {none_exports}. "
            "Check the import errors above — they're no longer hidden."
        )

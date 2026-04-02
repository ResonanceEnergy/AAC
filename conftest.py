"""
conftest.py — root-level pytest configuration for AAC
Adds all top-level module paths to sys.path so tests can import
any package without installing the project.
"""
import os
import sys
from pathlib import Path

# Project root is the directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

# Directories that contain importable packages / source modules
PACKAGE_ROOTS = [
    PROJECT_ROOT,           # root-level packages: shared/, strategies/, TradingExecution/, etc.
    PROJECT_ROOT / "src",   # src/ layout: src/aac/...
    PROJECT_ROOT / "core",  # core/orchestrator, core/risk_engine, etc.
    PROJECT_ROOT / "agents",  # agent_based_trading_integration, etc.
    PROJECT_ROOT / "scripts",  # scripts/health_check, etc.
]

for path in PACKAGE_ROOTS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# ── Environment ────────────────────────────────────────────────────────────────
# Load .env if present so tests that touch real APIs can pick up keys.
# Missing .env is fine — integration tests will be skipped via pytest marks.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — tests use system environment

# Set defaults for tests so nothing crashes on missing secrets
os.environ.setdefault("AAC_ENV", "test")
os.environ.setdefault("PAPER_TRADING", "true")
os.environ.setdefault("LIVE_TRADING_ENABLED", "false")

# ── Pytest fixtures ────────────────────────────────────────────────────────────
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def paper_trading_env():
    """Ensure all tests run in paper-trading mode."""
    old_pt = os.environ.get("PAPER_TRADING")
    old_lt = os.environ.get("LIVE_TRADING_ENABLED")
    os.environ["PAPER_TRADING"] = "true"
    os.environ["LIVE_TRADING_ENABLED"] = "false"
    yield
    # Restore original values
    if old_pt is None:
        os.environ.pop("PAPER_TRADING", None)
    else:
        os.environ["PAPER_TRADING"] = old_pt
    if old_lt is None:
        os.environ.pop("LIVE_TRADING_ENABLED", None)
    else:
        os.environ["LIVE_TRADING_ENABLED"] = old_lt


@pytest.fixture
def mock_market_data():
    """Provide a dictionary of mock market prices for testing."""
    return {
        "BTC/USDT": 45000.0,
        "ETH/USDT": 2500.0,
        "SOL/USDT": 120.0,
        "ADA/USDT": 0.55,
        "DOGE/USDT": 0.085,
    }


@pytest.fixture
def paper_engine():
    """Create a fresh ExecutionEngine in paper-trading (dry_run=False) mode."""
    try:
        from TradingExecution.execution_engine import ExecutionEngine
        engine = ExecutionEngine()
        engine.dry_run = False
        return engine
    except ImportError:
        pytest.skip("TradingExecution not available")


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite database path for testing."""
    return tmp_path / "test.db"


def pytest_configure(config):
    """Register custom markers so tests don't warn about unknown marks."""
    config.addinivalue_line("markers", "live: mark test as requiring live API credentials")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow (> 5s)")
    config.addinivalue_line("markers", "paper: mark test as requiring paper-trading mode")
    config.addinivalue_line("markers", "exchange: mark test as requiring an exchange connection")
    config.addinivalue_line("markers", "cross_pillar: mark test as cross-pillar integration (needs services)")
    config.addinivalue_line("markers", "api: mark test as making real HTTP calls to external APIs")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked 'api' unless explicitly selected with -m."""
    # If the user passed an explicit -m filter, respect it
    markexpr = config.getoption("-m", default="")
    if "api" in markexpr:
        return
    skip_api = pytest.mark.skip(reason="Skipped by default — run with: pytest -m api")
    for item in items:
        if "api" in item.keywords:
            item.add_marker(skip_api)

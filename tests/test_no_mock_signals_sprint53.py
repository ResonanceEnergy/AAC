"""Sprint 53 — assert mock-signal artifacts are gone from agent code.

Pure source-level assertions (no network, no heavy imports beyond the modules
themselves).  Runs in the default suite -- no ``api`` marker required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# Files purged in Sprint 53.
_PURGED_FILES = [
    _REPO / "agents" / "agent_based_trading.py",
    _REPO / "BigBrainIntelligence" / "agents.py",
    _REPO / "divisions" / "research" / "agents" / "base_agent.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", _PURGED_FILES, ids=lambda p: p.name)
def test_no_mock_signal_string_literals(path: Path):
    src = _read(path)
    for forbidden in (
        "Mock API Signal: Exchange Maintenance",
        "Mock API Signal: New Token Launch",
        "MockExchange",
        "MockDEX",
    ):
        assert forbidden not in src, (
            f"{path.name} still contains forbidden mock literal: {forbidden!r}"
        )


def test_agent_based_trading_no_mockintelligencesource_class():
    src = _read(_REPO / "agents" / "agent_based_trading.py")
    assert "class MockIntelligenceSource" not in src
    # Reference comments are allowed but no actual instantiation.
    assert "MockIntelligenceSource()" not in src


def test_agent_based_trading_no_simulated_fill_random():
    """Simulated random-walk fills replaced with NotImplementedError."""
    src = _read(_REPO / "agents" / "agent_based_trading.py")
    # The execute_trade method must raise rather than fabricate fills.
    assert "NotImplementedError" in src
    # Comments may contain the word random, but no live calls in execute_trade
    # path.  Heuristic: the simulated-fill phrases are gone.
    assert "Realistic P&L based on expected return with some noise" not in src
    assert "trade_successful = random.random()" not in src


def test_master_launcher_does_not_initialize_contest():
    """Sprint 53 disabled the AgentContestOrchestrator launch hook."""
    src = _read(_REPO / "core" / "aac_master_launcher.py")
    assert "self.strategy_orchestrator.initialize_contest()" not in src
    # The function still exists but is now a no-op with a documented warning.
    assert "_launch_strategy_agents" in src


def test_module_attribute_mockintelligencesource_absent():
    import importlib

    import agents.agent_based_trading as mod

    importlib.reload(mod)
    assert not hasattr(mod, "MockIntelligenceSource")


def test_research_agent_returns_empty_list_when_no_signals(monkeypatch):
    """Both research-agent files: when CoinGecko returns nothing the method
    returns the real (possibly empty) signals list -- never injects fakes.
    """
    # We don't run the full async scan (it touches network) -- we only need to
    # confirm the source no longer has the inject-on-empty branch.
    for path in (
        _REPO / "BigBrainIntelligence" / "agents.py",
        _REPO / "divisions" / "research" / "agents" / "base_agent.py",
    ):
        src = _read(path)
        assert "Add mock signals for testing when APIs are unavailable" not in src
        assert "if not signals:" not in src or "signals.extend([" not in src

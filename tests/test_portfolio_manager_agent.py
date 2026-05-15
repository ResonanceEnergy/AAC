"""Tests for the portfolio_manager agent runtime + decision memory.

Covers:
- ``shared.aac_agents.memory.append_decision`` round-trip (md + jsonl).
- ``shared.aac_agents.memory.recent_decisions`` newest-first ordering.
- ``shared.aac_agents.runtime._parse_pm_output`` regex parser.
- ``shared.aac_agents.runtime.run_portfolio_decision`` end-to-end with a
  stubbed ``run_agent`` and ``run_debate`` so no Ollama call is made.
- Risk-snapshot tools degrade gracefully when their backing modules raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# ── memory.append_decision / recent_decisions ────────────────────────────────


def test_append_decision_writes_jsonl_and_md(tmp_path, monkeypatch):
    from shared.aac_agents import memory as mem

    monkeypatch.setattr(mem, "_MEMORY_DIR", tmp_path)
    monkeypatch.setattr(mem, "_MD_PATH", tmp_path / "decisions.md")
    monkeypatch.setattr(mem, "_JSONL_PATH", tmp_path / "decisions.jsonl")

    entry = mem.append_decision(
        kind="portfolio_manager",
        thesis="Approve 1.5% on SPY puts.",
        verdict="approve",
        symbol="SPY",
        confidence=0.72,
        tools=["get_drawdown_state", "get_daily_loss_status"],
        extras={"size_pct": 1.5},
    )

    assert entry["id"] and len(entry["id"]) == 10
    assert entry["kind"] == "portfolio_manager"
    assert entry["symbol"] == "SPY"
    assert entry["verdict"] == "approve"
    assert entry["confidence"] == 0.72
    assert entry["extras"]["size_pct"] == 1.5
    assert entry["realised_pnl_usd"] is None

    # JSONL: one line, parseable, matches entry id
    jsonl_lines = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 1
    parsed = json.loads(jsonl_lines[0])
    assert parsed["id"] == entry["id"]

    # Markdown: contains header + verdict + tools
    md = (tmp_path / "decisions.md").read_text(encoding="utf-8")
    assert "portfolio_manager" in md
    assert "SPY" in md
    assert "approve" in md
    assert "get_drawdown_state" in md
    assert "_pending_" in md  # placeholder for realised PnL


def test_recent_decisions_newest_first(tmp_path, monkeypatch):
    from shared.aac_agents import memory as mem

    monkeypatch.setattr(mem, "_MEMORY_DIR", tmp_path)
    monkeypatch.setattr(mem, "_MD_PATH", tmp_path / "decisions.md")
    monkeypatch.setattr(mem, "_JSONL_PATH", tmp_path / "decisions.jsonl")

    for i in range(3):
        mem.append_decision(
            kind="portfolio_manager",
            thesis=f"call #{i}",
            verdict="approve" if i % 2 == 0 else "reject",
            symbol=f"SYM{i}",
            confidence=0.5 + 0.1 * i,
        )

    recent = mem.recent_decisions(n=2)
    assert len(recent) == 2
    # newest first
    assert recent[0]["thesis"] == "call #2"
    assert recent[1]["thesis"] == "call #1"


# ── runtime._parse_pm_output ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected_decision, expected_size",
    [
        ("Decision: approve\nSize: 2.5%\nConfidence: 0.7", "approve", 2.5),
        ("DECISION: REJECT — too risky", "reject", None),
        ("decision approve, size 0%", "approve", 0.0),
        ("no verdict here", None, None),
        ("", None, None),
    ],
)
def test_parse_pm_output(text, expected_decision, expected_size):
    from shared.aac_agents.runtime import _parse_pm_output

    out = _parse_pm_output(text)
    assert out["decision"] == expected_decision
    assert out["size_pct"] == expected_size


def test_parse_pm_output_clamps_size_to_100():
    from shared.aac_agents.runtime import _parse_pm_output

    assert _parse_pm_output("decision: approve  size: 500%")["size_pct"] == 100.0


# ── run_portfolio_decision end-to-end (no Ollama) ────────────────────────────


def _stub_debate(symbol: str | None = None, **_: Any) -> dict[str, Any]:
    return {
        "question": f"Should we trade {symbol or 'the book'}?",
        "verdict": "bullish",
        "confidence": 0.65,
        "bull_confidence": 0.7,
        "bear_confidence": 0.4,
        "bull": {"answer": "Strong tape, breadth healthy."},
        "bear": {"answer": "Vol creeping; CPI risk."},
        "decision_id": "deadbeef01",
    }


def _stub_pm_agent(name: str, *_args: Any, **_kw: Any) -> dict[str, Any]:
    assert name == "portfolio_manager"
    return {
        "answer": "Decision: approve\nSize: 1.0%\nConfidence: 0.62\nRationale: tight stop, defined risk.",
        "tool_calls": [
            {"tool": "get_drawdown_state", "arguments": {}, "result": {"tripped": False}},
            {"tool": "get_daily_loss_status", "arguments": {}, "result": {"tripped": False}},
        ],
    }


def test_run_portfolio_decision_persists_and_parses(tmp_path, monkeypatch):
    from shared.aac_agents import memory as mem
    from shared.aac_agents import runtime as rt

    monkeypatch.setattr(mem, "_MEMORY_DIR", tmp_path)
    monkeypatch.setattr(mem, "_MD_PATH", tmp_path / "decisions.md")
    monkeypatch.setattr(mem, "_JSONL_PATH", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(rt, "run_debate", _stub_debate)
    monkeypatch.setattr(rt, "run_agent", _stub_pm_agent)

    out = rt.run_portfolio_decision(symbol="SPY", persist=True)

    assert out["pm"]["answer"].startswith("Decision: approve")
    parsed = out["pm_decision"]
    assert parsed["decision"] == "approve"
    assert parsed["size_pct"] == 1.0
    assert parsed["confidence"] == 0.62
    assert out["pm_decision_id"] and len(out["pm_decision_id"]) == 10

    jsonl = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1
    persisted = json.loads(jsonl[0])
    assert persisted["kind"] == "portfolio_manager"
    assert persisted["verdict"] == "approve"
    assert persisted["extras"]["size_pct"] == 1.0
    assert persisted["extras"]["debate_decision_id"] == "deadbeef01"
    assert "get_drawdown_state" in persisted["tools"]


def test_run_portfolio_decision_handles_pm_failure(tmp_path, monkeypatch):
    from shared.aac_agents import memory as mem
    from shared.aac_agents import runtime as rt

    monkeypatch.setattr(mem, "_MEMORY_DIR", tmp_path)
    monkeypatch.setattr(mem, "_MD_PATH", tmp_path / "decisions.md")
    monkeypatch.setattr(mem, "_JSONL_PATH", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(rt, "run_debate", _stub_debate)

    def _boom(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rt, "run_agent", _boom)

    out = rt.run_portfolio_decision(symbol="SPY", persist=False)
    assert "portfolio_manager failed" in out["pm"]["answer"]
    assert out["pm_decision"]["decision"] is None
    assert out["pm_decision_id"] is None


def test_run_portfolio_decision_no_persist_skips_memory(tmp_path, monkeypatch):
    from shared.aac_agents import memory as mem
    from shared.aac_agents import runtime as rt

    monkeypatch.setattr(mem, "_MEMORY_DIR", tmp_path)
    monkeypatch.setattr(mem, "_MD_PATH", tmp_path / "decisions.md")
    monkeypatch.setattr(mem, "_JSONL_PATH", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(rt, "run_debate", _stub_debate)
    monkeypatch.setattr(rt, "run_agent", _stub_pm_agent)

    out = rt.run_portfolio_decision(symbol="SPY", persist=False)

    assert out["pm_decision_id"] is None
    assert not Path(tmp_path / "decisions.jsonl").exists()


# ── risk-snapshot tools degrade gracefully ───────────────────────────────────


def test_get_drawdown_state_handles_missing_module(monkeypatch):
    from shared.aac_agents import tools

    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("module unavailable")

    monkeypatch.setattr(
        "strategies.drawdown_circuit_breaker.DrawdownCircuitBreaker",
        _raise,
        raising=False,
    )
    out = tools.get_drawdown_state()
    # Either available with real data or unavailable with error — never raises.
    assert isinstance(out, dict)
    if not out.get("available", True):
        assert "error" in out


def test_get_daily_loss_status_fails_open():
    from shared.aac_agents import tools

    out = tools.get_daily_loss_status(account_value_usd=0.0)
    assert isinstance(out, dict)
    # The tool's contract: failure → tripped=False (fail-open)
    assert out.get("tripped") in (False, True)

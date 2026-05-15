"""Sprint 0 tests for the DFV decision engine.

Covers the four canonical hard-gate / FOMO veto cases. Uses tmp_path to
avoid touching the real agents/dfv/memory files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agents.dfv import decision_engine as de


@pytest.fixture
def isolated_dfv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> de.DFV:
    """A DFV instance whose memory + doctrine point at tmp_path."""
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "briefs").mkdir()

    doctrine = {
        "autonomy": {"trade_execution": "human_in_loop"},
        "conviction": {
            "5": {"label": "screaming", "max_pct_book": 0.20},
            "4": {"label": "high",      "max_pct_book": 0.10},
            "3": {"label": "starter",   "max_pct_book": 0.03},
            "2": {"label": "watch",     "max_pct_book": 0.0},
            "1": {"label": "lottery",   "max_pct_book": 0.0},
        },
        "memory": {
            "thesis_log":    str(mem / "thesis_log.json"),
            "conviction":    str(mem / "conviction.json"),
            "watchlist":     str(mem / "watchlist.json"),
            "decisions_log": str(mem / "decisions.jsonl"),
        },
    }
    doc_path = tmp_path / "dfv_doctrine.yaml"
    doc_path.write_text(yaml.safe_dump(doctrine), encoding="utf-8")

    # Memory paths in doctrine are relative to REPO_ROOT in the engine.
    # Point REPO_ROOT at tmp_path and rewrite memory paths to be relative.
    monkeypatch.setattr(de, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(de, "DOCTRINE_PATH", doc_path)
    # Also patch ThesisLog / friends' REPO_ROOT.
    from agents.dfv import memory_store as ms
    monkeypatch.setattr(ms, "REPO_ROOT", tmp_path)
    # Make memory paths relative for the (REPO_ROOT / Path(path)) join.
    doctrine["memory"] = {
        "thesis_log":    "memory/thesis_log.json",
        "conviction":    "memory/conviction.json",
        "watchlist":     "memory/watchlist.json",
        "decisions_log": "memory/decisions.jsonl",
    }
    doc_path.write_text(yaml.safe_dump(doctrine), encoding="utf-8")

    return de.DFV()


def test_g1_missing_thesis_vetoes(isolated_dfv: de.DFV) -> None:
    """No thesis on file → G1 hard fail → vetoed."""
    decision = isolated_dfv.evaluate({
        "symbol": "NVDA",
        "action": "buy",
        "size_pct": 0.01,
    })
    assert decision.verdict == "vetoed"
    g1 = next(g for g in decision.gates if g.gate_id == "G1")
    assert g1.outcome == "fail"
    assert g1.severity == "hard"


def test_g2_oversize_vetoes(isolated_dfv: de.DFV) -> None:
    """Thesis + invalidation present, but size exceeds tier cap → G2 hard fail."""
    isolated_dfv.thesis.set(
        "GME",
        thesis="Net cash, brand, console refresh",
        conviction=3,
        horizon="12m",
        catalysts=["earnings"],
        invalidation="cash burn > $200M/qtr",
        target={"raw": "$45 base"},
        sizing={"max_pct_book": 0.03},
    )
    isolated_dfv.conviction.set("GME", 3, reason="test")

    decision = isolated_dfv.evaluate({
        "symbol": "GME",
        "action": "buy",
        "size_pct": 0.50,  # way over tier-3 0.03 cap
        "expected_slippage_pct": 0.005,
        "cash_after_trade": 50_000,
        "portfolio_value": 100_000,
    })
    assert decision.verdict == "vetoed"
    g2 = next(g for g in decision.gates if g.gate_id == "G2")
    assert g2.outcome == "fail"
    assert g2.severity == "hard"


def test_g6_missing_invalidation_vetoes(isolated_dfv: de.DFV) -> None:
    """Thesis exists but invalidation field empty → G6 hard fail."""
    isolated_dfv.thesis.set(
        "AAPL",
        thesis="big tech",
        conviction=3,
        horizon="months",
        catalysts=[],
        invalidation="",  # empty → G6 must reject
        target={"raw": ""},
        sizing={"max_pct_book": 0.03},
    )
    isolated_dfv.conviction.set("AAPL", 3, reason="test")

    decision = isolated_dfv.evaluate({
        "symbol": "AAPL",
        "action": "buy",
        "size_pct": 0.02,
        "expected_slippage_pct": 0.005,
        "cash_after_trade": 50_000,
        "portfolio_value": 100_000,
    })
    g6 = next(g for g in decision.gates if g.gate_id == "G6")
    assert g6.outcome == "fail", decision.to_dict()
    assert g6.severity == "hard"
    assert decision.verdict == "vetoed"


def test_review_prompt_fomo_language_is_vetoed(isolated_dfv: de.DFV) -> None:
    """FOMO trigger phrases get hard-vetoed regardless of context."""
    d = isolated_dfv.review_prompt("Should I YOLO into NVDA calls right now?")
    assert d.verdict == "vetoed"
    assert any("FOMO" in n for n in d.notes)


def test_decisions_log_written(isolated_dfv: de.DFV, tmp_path: Path) -> None:
    """Every evaluate() call must persist to decisions.jsonl."""
    isolated_dfv.evaluate({"symbol": "TSLA", "action": "buy", "size_pct": 0.05})
    log = tmp_path / "memory" / "decisions.jsonl"
    assert log.exists()
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(entry.get("decision", {}).get("verdict") == "vetoed" for entry in lines)

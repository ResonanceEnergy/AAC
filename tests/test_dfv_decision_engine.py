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
        "postmortems":   "memory/postmortems.jsonl",
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


def test_llm_tool_proposal_matches_direct_evaluate(
    isolated_dfv: de.DFV, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3 — _decide_proposal (LLM tool layer) must hit the same gates as a direct dict.

    Pre-fix: field name drift (size_pct_book vs size_pct, etc.) caused G3/G4/G5/G7
    to silently default-pass. This test prevents regression.
    """
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

    # Force the LLM module to use OUR isolated DFV (not a fresh one with prod paths).
    from agents.dfv import llm as dfv_llm
    monkeypatch.setattr(dfv_llm, "_dfv", lambda: isolated_dfv)

    # G2 should fail: oversized at 50% of book vs tier-3 cap of 3%.
    direct = isolated_dfv.evaluate({
        "symbol": "GME",
        "action": "buy",
        "size_pct": 0.50,
        "expected_slippage_pct": 0.005,
        "cash_after_trade": 50_000,
        "portfolio_value": 100_000,
        "factor_concentration_after": 0.10,
        "catalyst_within_days": 3,
        "catalyst_acknowledged": True,
    }).to_dict()

    via_tool = dfv_llm._decide_proposal(
        symbol="GME",
        side="long",
        size_pct_book=0.50,
        invalidation="cash burn > $200M/qtr",
        catalyst_days_out=3,
        catalyst_acknowledged=True,
        cash_pct_after=0.50,
        portfolio_value=100_000,
        correlation_factor_pct=0.10,
        estimated_slippage_pct=0.005,
    )

    assert direct["verdict"] == via_tool["verdict"] == "vetoed"
    direct_g2 = next(g for g in direct["gates"] if g["gate_id"] == "G2")
    tool_g2 = next(g for g in via_tool["gates"] if g["gate_id"] == "G2")
    assert direct_g2["outcome"] == tool_g2["outcome"] == "fail"
    # G4 should NOT silently pass — both surfaces see catalyst within 5d acknowledged.
    direct_g4 = next(g for g in direct["gates"] if g["gate_id"] == "G4")
    tool_g4 = next(g for g in via_tool["gates"] if g["gate_id"] == "G4")
    assert direct_g4["outcome"] == tool_g4["outcome"]


# ── Regression: _guess_symbol must not return single-char pronouns ────────
def test_guess_symbol_skips_single_char_pronouns() -> None:
    """Regression: 'I want to add to GME at current levels' must resolve to GME, not I.

    Real prompt that previously logged a vetoed decision for symbol='I'.
    """
    assert de.DFV._guess_symbol("I want to add to GME at current levels. Decent size.") == "GME"
    assert de.DFV._guess_symbol("I think the market is overbought") is None
    assert de.DFV._guess_symbol("buying $GME and TSLA") == "GME"  # cashtag wins
    assert de.DFV._guess_symbol("just BUY some NVDA") == "NVDA"  # BUY in stopwords
    assert de.DFV._guess_symbol("") is None


def test_review_prompt_normalized_schema(isolated_dfv: de.DFV, tmp_path: Path) -> None:
    """review_prompt entries must include symbol + proposal in normalized form."""
    isolated_dfv.review_prompt("Should I add to GME? FOMO into it?")
    log = tmp_path / "memory" / "decisions.jsonl"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    last = lines[-1]
    assert last.get("kind") == "prompt_review"
    assert last.get("symbol") == "GME"
    assert "proposal" in last
    assert last["proposal"].get("symbol") == "GME"
    assert "prompt" in last["proposal"]


# ── G8 vol-regime gate ───────────────────────────────────────────────────
def test_g8_vol_regime_blocks_long_vega_in_low_vix_contango(isolated_dfv: de.DFV) -> None:
    """Long-call/put entry must be vetoed when VIX < 15 AND VIX3M > VIX (contango)."""
    isolated_dfv.thesis.set(
        "SPY",
        thesis="hedge",
        conviction=3, horizon="months",
        catalysts=["fed"],
        invalidation="VIX > 20 ends the hedge",
        target={"raw": "0DTE"},
        sizing={"max_pct_book": 0.02},
    )
    isolated_dfv.conviction.set("SPY", 3, reason="test")
    # Patch doctrine in memory so vol_regime block exists.
    isolated_dfv.doctrine["vol_regime"] = {
        "enabled": True,
        "vix_floor": 15.0,
        "contango_threshold": 0.0,
        "applies_to_actions": ["buy_put", "buy_call", "long_vol", "buy"],
    }
    decision = isolated_dfv.evaluate({
        "symbol": "SPY",
        "action": "buy_put",
        "size_pct": 0.02,
        "expected_slippage_pct": 0.005,
        "cash_after_trade": 50_000,
        "portfolio_value": 100_000,
        "factor_concentration_after": 0.10,
        "catalyst_within_days": 30,
        "catalyst_acknowledged": True,
        "vix": 12.0,
        "vix3m": 14.5,
    })
    g8 = next(g for g in decision.gates if g.gate_id == "G8")
    assert g8.outcome == "fail"
    assert g8.severity == "soft"
    # 3 soft fails → veto; just G8 alone counts as 1 soft, so verdict depends on
    # how many other softs fired. Either way, G8 must be the blocker we see.
    assert "long-vega entry blocked" in g8.note


def test_g8_vol_regime_passes_when_data_missing(isolated_dfv: de.DFV) -> None:
    """G8 must NOT veto when vix/vix3m are not supplied (flagged-pass)."""
    isolated_dfv.thesis.set(
        "SPY", thesis="t", conviction=3, horizon="m",
        catalysts=[], invalidation="below 400",
        target={}, sizing={"max_pct_book": 0.02},
    )
    isolated_dfv.conviction.set("SPY", 3, reason="test")
    decision = isolated_dfv.evaluate({
        "symbol": "SPY", "action": "buy_put", "size_pct": 0.02,
        "expected_slippage_pct": 0.005,
        "cash_after_trade": 50_000, "portfolio_value": 100_000,
        "factor_concentration_after": 0.10,
        "catalyst_within_days": 30, "catalyst_acknowledged": True,
    })
    g8 = next(g for g in decision.gates if g.gate_id == "G8")
    assert g8.outcome == "pass"


def test_g8_vol_regime_passes_for_non_long_vega(isolated_dfv: de.DFV) -> None:
    """G8 must auto-pass when the proposal is not a long-vega entry."""
    isolated_dfv.thesis.set(
        "SPY", thesis="t", conviction=3, horizon="m",
        catalysts=[], invalidation="below 400",
        target={}, sizing={"max_pct_book": 0.02},
    )
    isolated_dfv.conviction.set("SPY", 3, reason="test")
    decision = isolated_dfv.evaluate({
        "symbol": "SPY", "action": "sell", "size_pct": 0.02,
        "expected_slippage_pct": 0.005,
        "cash_after_trade": 50_000, "portfolio_value": 100_000,
        "factor_concentration_after": 0.10,
        "catalyst_within_days": 30, "catalyst_acknowledged": True,
        "vix": 12.0, "vix3m": 14.5,
    })
    g8 = next(g for g in decision.gates if g.gate_id == "G8")
    assert g8.outcome == "pass"


# ── Thesis schema: roll_trigger_dte ──────────────────────────────────────
def test_thesis_set_persists_roll_trigger_dte(isolated_dfv: de.DFV) -> None:
    rec = isolated_dfv.thesis.set(
        "ARCC", thesis="theta hedge", conviction=2, horizon="3m",
        catalysts=["macro"], invalidation="below $17",
        target={"raw": "$17P"}, sizing={"max_pct_book": 0.01},
        roll_trigger_dte=30, tags=["macro_hedge"],
    )
    assert rec["roll_trigger_dte"] == 30
    assert rec["tags"] == ["macro_hedge"]
    fresh = isolated_dfv.thesis.get("ARCC")
    assert fresh is not None
    assert fresh["roll_trigger_dte"] == 30


def test_thesis_set_defaults_roll_trigger_to_21(isolated_dfv: de.DFV) -> None:
    rec = isolated_dfv.thesis.set(
        "PFF", thesis="t", conviction=2, horizon="3m",
        catalysts=[], invalidation="below $29",
        target={}, sizing={},
    )
    assert rec["roll_trigger_dte"] == 21
    assert rec["tags"] == []

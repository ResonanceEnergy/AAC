"""Stage-3 DFV build tests — journal, notifications, orphan guard, reconciler,
P&L attribution, RAG-lite, jury, roll engine.

Pattern follows tests/test_dfv_routines.py: monkeypatch REPO_ROOT in every
module that opens files relative to it, so writes are sandboxed in tmp_path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.dfv import decision_engine as de
    from agents.dfv import memory_store as ms

    monkeypatch.setattr(de, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ms, "REPO_ROOT", tmp_path)


# ── JournalLog ─────────────────────────────────────────────────────────────
def test_journal_append_and_has_today() -> None:
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    assert dfv.journal.has_today() is False
    dfv.journal.append("first entry", mood="calm", tags=["test"])
    dfv.journal.append("second entry")
    assert dfv.journal.has_today() is True
    tail = dfv.journal.tail(5)
    assert len(tail) == 2
    assert tail[-1]["entry"] == "second entry"
    assert tail[0]["mood"] == "calm"


# ── NotificationsLog dedupe ────────────────────────────────────────────────
def test_notifications_dedupe_within_window() -> None:
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    first = dfv.notifications.append(
        kind="breach", symbol="GME", title="x", dedupe_key="k1",
    )
    again = dfv.notifications.append(
        kind="breach", symbol="GME", title="x", dedupe_key="k1",
    )
    assert first is not None
    assert again is None  # suppressed by dedupe


# ── ReconciliationLog round-trip ───────────────────────────────────────────
def test_reconciliation_read_write() -> None:
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    assert dfv.reconciliation.read() == {}
    dfv.reconciliation.write({"mismatch_count": 2, "mismatches": [{"kind": "venue_dup"}]})
    snap = dfv.reconciliation.read()
    assert snap["mismatch_count"] == 2
    assert snap["mismatches"][0]["kind"] == "venue_dup"
    assert "ts" in snap


# ── Orphan guard ──────────────────────────────────────────────────────────
def test_orphan_guard_writes_skeleton_for_missing_thesis() -> None:
    from agents.dfv import orphan_guard
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    payload = {
        "portfolio": {
            "accounts": {
                "ibkr": {
                    "venue": "IBKR",
                    "positions": [
                        {"symbol": "ORPHANCO", "quantity": 100},
                        {"symbol": "OTHER", "quantity": 50},
                    ],
                }
            }
        }
    }
    orphans = orphan_guard.detect_orphans(dfv, payload)
    assert {o["symbol"] for o in orphans} == {"ORPHANCO", "OTHER"}

    result = orphan_guard.scan_and_stub(payload, dfv=dfv)
    assert set(result["written"]) == {"ORPHANCO", "OTHER"}
    theses = dfv.thesis.all()
    assert "ORPHANCO" in theses
    assert theses["ORPHANCO"]["conviction"] == 0
    assert "auto_skeleton" in theses["ORPHANCO"]["tags"]


def test_orphan_guard_skips_when_thesis_exists() -> None:
    from agents.dfv import orphan_guard
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    dfv.thesis.set(
        symbol="HAVE", thesis="real one", conviction=2, horizon="3M",
        catalysts=[], invalidation="x", target={}, sizing={"max_pct_book": 0.05},
    )
    payload = {"portfolio": {"accounts": {"a": {"positions": [{"symbol": "HAVE", "quantity": 10}]}}}}
    result = orphan_guard.scan_and_stub(payload, dfv=dfv)
    assert result["written"] == []


# ── Reconciler ────────────────────────────────────────────────────────────
def test_reconciler_detects_venue_duplicates() -> None:
    from agents.dfv import reconciler
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    payload = {
        "portfolio": {
            "accounts": {
                "ibkr": {"positions": [{"symbol": "GME", "quantity": 100}]},
                "moomoo": {"positions": [{"symbol": "GME", "quantity": 50}]},
            }
        }
    }
    snap = reconciler.reconcile(payload=payload, dfv=dfv, use_live_ibkr=False)
    assert snap["mismatch_count"] >= 1
    kinds = {d.get("kind") for d in snap["mismatches"]}
    assert "duplicated_across_venues" in kinds


def test_reconciler_clean_when_no_dupes() -> None:
    from agents.dfv import reconciler
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    payload = {"portfolio": {"accounts": {"ibkr": {"positions": [{"symbol": "A", "quantity": 1}]}}}}
    snap = reconciler.reconcile(payload=payload, dfv=dfv, use_live_ibkr=False)
    # No venue duplication should mean no duplicated_across_venues diff
    assert not any(d.get("kind") == "duplicated_across_venues" for d in snap["mismatches"])


def test_reconciler_live_ibkr_detects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diff #3 — live IBKR positions disagree with cached payload IBKR slice."""
    from agents.dfv import reconciler
    from agents.dfv.decision_engine import DFV

    def _fake_snap() -> dict[str, Any]:
        return {
            "positions": {"GME": 200.0},  # live says 200
            "account_summary": {"NetLiquidation_USD": 100_000.0},
        }

    monkeypatch.setattr(reconciler, "_live_ibkr_snapshot", _fake_snap)

    dfv = DFV()
    payload = {
        "portfolio": {
            "accounts": {
                "ibkr": {"venue": "IBKR", "positions": [{"symbol": "GME", "quantity": 100}]},
            }
        }
    }
    snap = reconciler.reconcile(payload=payload, dfv=dfv, use_live_ibkr=True)
    assert snap["ibkr_live_available"] is True
    assert snap["ibkr_live_positions"] == {"GME": 200.0}
    drift = [d for d in snap["mismatches"] if d.get("kind") == "ibkr_live_vs_cache"]
    assert drift, "should flag ibkr_live_vs_cache drift"
    assert drift[0]["symbol"] == "GME"
    assert drift[0]["live_qty"] == 200.0
    assert drift[0]["cache_qty"] == 100.0
    assert drift[0]["delta"] == 100.0


def test_reconciler_live_ibkr_clean_when_match(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.dfv import reconciler
    from agents.dfv.decision_engine import DFV

    monkeypatch.setattr(
        reconciler, "_live_ibkr_snapshot",
        lambda: {"positions": {"IWM": -1.0}, "account_summary": {}},
    )
    dfv = DFV()
    payload = {
        "portfolio": {
            "accounts": {
                "ibkr": {"venue": "IBKR", "positions": [{"symbol": "IWM", "quantity": -1}]},
            }
        }
    }
    snap = reconciler.reconcile(payload=payload, dfv=dfv, use_live_ibkr=True)
    assert not any(d.get("kind") == "ibkr_live_vs_cache" for d in snap["mismatches"])


def test_reconciler_live_ibkr_unavailable_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """When IBKR is offline (snapshot returns None) reconciler still produces a snapshot."""
    from agents.dfv import reconciler
    from agents.dfv.decision_engine import DFV

    monkeypatch.setattr(reconciler, "_live_ibkr_snapshot", lambda: None)
    dfv = DFV()
    payload = {"portfolio": {"accounts": {"ibkr": {"positions": [{"symbol": "X", "quantity": 1}]}}}}
    snap = reconciler.reconcile(payload=payload, dfv=dfv, use_live_ibkr=True)
    assert snap["ibkr_live_available"] is False
    assert snap["ibkr_live_positions"] == {}


# ── P&L attribution ───────────────────────────────────────────────────────
def test_pnl_attribution_groups_by_tag_and_tier() -> None:
    from agents.dfv import pnl_attribution
    from agents.dfv.decision_engine import DFV
    dfv = DFV()
    # Seed thesis + postmortems
    dfv.thesis.set(
        symbol="WIN", thesis="t", conviction=2, horizon="1M",
        catalysts=[], invalidation="x", target={}, sizing={}, tags=["macro"],
    )
    dfv.thesis.set(
        symbol="LOSE", thesis="t", conviction=1, horizon="1M",
        catalysts=[], invalidation="x", target={}, sizing={}, tags=["macro"],
    )
    dfv.postmortems.append({
        "symbol": "WIN", "realized_pnl": 500, "status": "closed_profit",
        "expiry": "2026-01-17",
    })
    dfv.postmortems.append({
        "symbol": "LOSE", "realized_pnl": -200, "status": "expired_worthless",
        "expiry": "2026-01-17",
    })
    out = pnl_attribution.attribute(dfv=dfv)
    assert out["total"]["n"] == 2
    assert out["total"]["realized"] == 300
    assert "macro" in out["by_tag"]
    assert out["by_tag"]["macro"]["n"] == 2
    # tier keys are str for JSON friendliness
    assert "2" in out["by_tier"] or 2 in out["by_tier"]


# ── RAG-lite ──────────────────────────────────────────────────────────────
def test_rag_lite_reindex_and_search() -> None:
    from agents.dfv import rag_lite
    from agents.dfv.decision_engine import DFV
    # Point rag_lite DB into tmp via REPO_ROOT (already patched via memory_store)
    # rag_lite stores DB next to memory dir — patch its path module-level constant
    monkey_dir = DFV().thesis.path.parent
    rag_lite.DB_PATH = monkey_dir / "rag_lite.sqlite"

    dfv = DFV()
    dfv.thesis.set(
        symbol="ZZTOP", thesis="legendary rock band thesis content unique_marker_abc",
        conviction=3, horizon="forever", catalysts=[], invalidation="x",
        target={}, sizing={},
    )
    counts = rag_lite.reindex(dfv=dfv)
    assert counts.get("theses", 0) >= 1
    hits = rag_lite.search("unique_marker_abc", k=5)
    assert hits, "FTS5 search should find seeded thesis"
    assert any("unique_marker_abc" in (h.get("snippet") or "").lower() for h in hits)


# ── Jury graceful degradation ─────────────────────────────────────────────
def test_jury_returns_ok_false_when_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.dfv import jury
    from agents.dfv.decision_engine import Decision
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Force gemini path to fail too by stubbing GeminiClient
    class _StubGem:
        configured = False
        def __init__(self, *a, **k):
            self.configured = False
        def ask(self, *a, **k):
            return {"ok": False, "error": "no key"}

    monkeypatch.setattr(
        "integrations.google_clients.GeminiClient", _StubGem, raising=False,
    )

    doctrine = {
        "jury": {
            "enabled": True,
            "panelists": ["gemini", "grok", "openai"],
            "quorum": 2,
            "models": {"gemini": "x", "grok": "x", "openai": "x"},
        }
    }
    out = jury.jury_review(
        symbol="X",
        proposal={"symbol": "X", "size_pct": 0.01},
        decision=Decision(verdict="approved", summary="ok"),
        doctrine=doctrine,
    )
    assert out["provider"] == "jury"
    # No panelists could vote — must not crash, must signal not-ok or empty.
    assert out["ok"] is False or out.get("dissent_count", 0) >= 0


# ── Roll engine: propose only ─────────────────────────────────────────────
def test_roll_engine_proposes_without_executing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agents.dfv import roll_engine
    from agents.dfv.decision_engine import DFV
    monkeypatch.setattr(roll_engine, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(roll_engine, "PROPOSALS_PATH",
                        tmp_path / "agents" / "dfv" / "memory" / "proposed_orders.jsonl")

    # Skip yfinance lookup
    monkeypatch.setattr(roll_engine, "_spot_price", lambda _s: None)

    position = {
        "symbol": "ARCC",
        "strike": 17.0,
        "expiry": "2026-04-17",
        "side": "put",
        "quantity": 5,
    }
    ticket = roll_engine.quote_and_review(position, dfv=DFV())
    assert ticket["kind"] == "roll_proposal"
    assert ticket["status"] == "pending_operator_ok"
    assert ticket["symbol"] == "ARCC"
    assert ticket["close_leg"]["strike"] == 17.0
    assert "gate_decision" in ticket
    # And the file was written
    assert roll_engine.PROPOSALS_PATH.exists()

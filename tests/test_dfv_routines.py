"""Sprint 1 smoke tests — each cadence routine returns a structured dict.

After 2026-05-15 orphan: routines no longer hit external feeds. We only
need to stub mission_control's payload + isolate brief writes.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_briefs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect brief writes + DFV memory into tmp_path."""
    from agents.dfv import routines as r

    monkeypatch.setattr(r, "BRIEF_DIR", tmp_path / "briefs")
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)

    # Empty payload (no IBKR / mission_control)
    monkeypatch.setattr(r, "_safe_collect_payload", lambda: {})


def test_asia_digest_returns_structured_dict() -> None:
    from agents.dfv.routines import asia_digest
    out = asia_digest()
    assert out["type"] == "asia_digest"
    assert "headline" in out
    assert "held_symbols" in out
    assert "asia_adr_universe" in out


def test_open_bell_prep_handles_no_holdings() -> None:
    from agents.dfv.routines import open_bell_prep
    out = open_bell_prep()
    assert out["type"] == "open_bell_prep"
    assert out["minutes_to_open"] == "T-5"
    assert "war_room" in out


def test_close_debrief_lists_next_actions() -> None:
    from agents.dfv.routines import close_debrief
    out = close_debrief()
    assert out["type"] == "close_debrief"
    assert "stale_theses" in out
    assert "next_actions" in out
    assert "watchlist_size" in out


def test_asia_watch_emits_risk_flag() -> None:
    from agents.dfv.routines import asia_watch
    out = asia_watch()
    assert out["type"] == "asia_watch"
    assert out["risk_flag"] in {"calm", "elevated", "stressed"}
    assert "war_room" in out

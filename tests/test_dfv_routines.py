"""Sprint 1 smoke tests — each cadence routine returns a structured dict
even when network clients fail.  We monkeypatch the data layer so tests
never hit the network.
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

    # Stub the data layer so tests never touch the network
    from agents.dfv import data as d

    monkeypatch.setattr(d, "crypto_snapshot", lambda *a, **k: {
        "ok": True, "ts": "now",
        "prices": {"BITCOIN/USD": {"price": 70000, "change_24h_pct": 1.5, "volume_24h": 1e9},
                   "ETHEREUM/USD": {"price": 2500, "change_24h_pct": -0.5, "volume_24h": 5e8}},
    })
    monkeypatch.setattr(d, "macro_snapshot", lambda *a, **k: {
        "ok": True, "ts": "now",
        "series": {"VIXCLS": {"date": "2026-05-15", "value": 18.2},
                   "DTWEXBGS": {"date": "2026-05-15", "value": 102.1},
                   "DGS10": {"date": "2026-05-15", "value": 4.32}},
    })
    monkeypatch.setattr(d, "quotes", lambda syms, **k: {
        "ok": True, "ts": "now",
        "quotes": {s: {"current": 100.0, "change_pct": 0.5, "high": 101, "low": 99, "prev_close": 99.5}
                   for s in syms},
    })
    monkeypatch.setattr(d, "market_news", lambda *a, **k: {
        "ok": True, "ts": "now", "category": "general",
        "headlines": [{"headline": "test", "source": "x", "url": "", "category": "g", "datetime": "now"}],
    })
    monkeypatch.setattr(d, "earnings_window", lambda **k: {
        "ok": True, "ts": "now", "days_ahead": k.get("days_ahead", 1), "earnings": [],
    })


def test_asia_digest_returns_structured_dict() -> None:
    from agents.dfv.routines import asia_digest
    out = asia_digest()
    assert out["type"] == "asia_digest"
    assert out["tilt"] in {"risk-on", "risk-off", "mixed"}
    assert "headline" in out
    assert out["crypto"]["ok"] is True


def test_open_bell_prep_handles_no_holdings() -> None:
    from agents.dfv.routines import open_bell_prep
    out = open_bell_prep()
    assert out["type"] == "open_bell_prep"
    assert out["minutes_to_open"] == "T-5"
    assert isinstance(out["top_movers_held"], list)


def test_close_debrief_lists_next_actions() -> None:
    from agents.dfv.routines import close_debrief
    out = close_debrief()
    assert out["type"] == "close_debrief"
    assert "stale_theses" in out
    assert "next_actions" in out
    assert "tomorrow_earnings_on_book" in out


def test_asia_watch_emits_risk_flag() -> None:
    from agents.dfv.routines import asia_watch
    out = asia_watch()
    assert out["type"] == "asia_watch"
    assert out["risk_flag"] in {"calm", "elevated", "stressed"}
    assert out["macro"]["ok"] is True

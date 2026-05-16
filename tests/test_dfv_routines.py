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
    from agents.dfv import decision_engine as de
    from agents.dfv import memory_store as ms
    from agents.dfv import routines as r

    monkeypatch.setattr(r, "BRIEF_DIR", tmp_path / "briefs")
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(de, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ms, "REPO_ROOT", tmp_path)

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


# ── Sprint 2 contract tests ─────────────────────────────────────────────


def test_collect_payload_returns_dfv_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """C1/T2 — mission_control.collect_payload() is the single feed for DFV.

    DFV routines depend on the keys: portfolio, war_room, pnl, alerts.
    Stub heavy collectors so the test runs in milliseconds.
    """
    from monitoring import mission_control as mc

    sentinel = {
        "portfolio": {"sentinel": "p"},
        "war_room": {"sentinel": "w"},
        "live_feeds": {},
        "regime": {},
        "doctrine": {},
        "moon": {},
        "health": {},
        "tasks": {},
        "daily_tasks": {},
        "unusual_whales": {},
        "divisions": {},
        "api_feeds": {},
        "polymarket": {},
        "scenarios": {},
        "openclaw": {},
        "backbone": {},
        "pnl": {"sentinel": "pnl"},
        "trade_log": {},
    }
    for key, value in sentinel.items():
        if hasattr(mc, f"collect_{key}"):
            monkeypatch.setattr(mc, f"collect_{key}", lambda v=value: v)
    if hasattr(mc, "_collect_alerts_for_payload"):
        monkeypatch.setattr(mc, "_collect_alerts_for_payload", lambda: [])

    payload = mc.collect_payload()
    assert isinstance(payload, dict)
    for required in ("portfolio", "war_room", "pnl", "alerts", "ts"):
        assert required in payload, f"DFV depends on payload['{required}']"


def test_eod_writes_postmortem_for_expired_thesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """M4+M5 — eod() auto-writes a post-mortem and nudges conviction on expiry losses."""
    from agents.dfv import routines as r
    from agents.dfv.decision_engine import DFV

    dfv = DFV()
    dfv.thesis.set(
        "TEST",
        thesis="expired test name",
        conviction=3,
        horizon="weeks",
        catalysts=[],
        invalidation="below 100",
        target={"raw": "test", "expiry": "2020-01-01", "realized_pnl": -250.0},
        sizing={"max_pct_book": 0.01},
    )
    dfv.conviction.set("TEST", 3, reason="seed")

    monkeypatch.setattr(r, "_safe_collect_payload", lambda: {"pnl": {}})
    out = r.eod()

    assert out["type"] == "eod"
    syms = [e["symbol"] for e in out["expirations_processed"]]
    assert "TEST" in syms, f"expected TEST in {out['expirations_processed']}"
    nudges = {n["symbol"]: n for n in out["conviction_nudges"]}
    assert "TEST" in nudges and nudges["TEST"]["to"] == 2

    # Idempotent: second run does not double-write
    out2 = r.eod()
    assert "TEST" not in [e["symbol"] for e in out2["expirations_processed"]]


def test_brief_flags_invalidation_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    """M3 — brief() surfaces theses where current price has crossed the invalidation level."""
    from agents.dfv import routines as r
    from agents.dfv.decision_engine import DFV

    dfv = DFV()
    dfv.thesis.set(
        "BREACHTST",
        thesis="long thesis",
        conviction=3,
        horizon="months",
        catalysts=[],
        invalidation="below $50",
        target={"raw": "n/a"},
        sizing={"max_pct_book": 0.02},
    )

    fake_payload = {
        "portfolio": {
            "accounts": {
                "ibkr": {"positions": [{"symbol": "BREACHTST", "last_price": 42.0}]}
            }
        },
        "war_room": {"regime": "RISK_OFF", "phase": "test"},
    }
    monkeypatch.setattr(r, "_safe_collect_payload", lambda: fake_payload)
    out = r.brief()
    breaches = out["discipline"]["invalidation_breaches"]
    syms = [b["symbol"] for b in breaches]
    assert "BREACHTST" in syms
    assert "INVALIDATION BREACH" in out["headline"]


def test_brief_surfaces_roll_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """brief() must list long-option positions inside their roll_trigger_dte window."""
    from datetime import datetime, timedelta, timezone

    from agents.dfv import routines as r
    from agents.dfv.decision_engine import DFV

    dfv = DFV()
    dfv.thesis.set(
        "ROLLTST",
        thesis="dividend hedge",
        conviction=3,
        horizon="3m",
        catalysts=[],
        invalidation="below $40",
        target={"raw": "$40P"},
        sizing={"max_pct_book": 0.01},
        roll_trigger_dte=21,
    )

    expiry = (datetime.now(timezone.utc).date() + timedelta(days=10)).isoformat()
    fake_payload = {
        "portfolio": {
            "accounts": {
                "ibkr": {
                    "positions": [{
                        "symbol": "ROLLTST",
                        "qty": 1,
                        "expiry": expiry,
                        "asset_type": "option",
                        "right": "P",
                    }],
                },
            },
        },
        "war_room": {"regime": "RISK_ON", "phase": "test"},
    }
    monkeypatch.setattr(r, "_safe_collect_payload", lambda: fake_payload)
    out = r.brief()
    triggers = out["discipline"].get("roll_triggers") or []
    syms = [t["symbol"] for t in triggers]
    assert "ROLLTST" in syms, triggers
    rec = next(t for t in triggers if t["symbol"] == "ROLLTST")
    assert rec["dte"] == 10
    assert rec["trigger_dte"] == 21
    assert rec["status"] in ("roll_or_kill", "expired")

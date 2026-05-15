from __future__ import annotations

"""DFV daily routines — pre-market brief, midday, EOD, weekend DD.

Each routine is a pure function: takes nothing, returns a dict report.
The daemon calls them on schedule; the CLI calls them on demand.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from agents.dfv.decision_engine import DFV

_log = structlog.get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
BRIEF_DIR = REPO_ROOT / "agents" / "dfv" / "memory" / "briefs"


def _save_brief(name: str, payload: dict[str, Any]) -> Path:
    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = BRIEF_DIR / f"{ts}_{name}.json"
    import json
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Best-effort: index into DFV semantic memory so future `ask` calls can recall it.
    try:
        from agents.dfv import rag as dfv_rag  # noqa: PLC0415
        dfv_rag.index_brief(path, payload)
    except Exception as e:  # noqa: BLE001 — daemon must not die on RAG failures
        _log.warning("dfv.routines.index_brief_failed", error=str(e))
    return path


def _safe_collect_payload() -> dict[str, Any]:
    """Pull current dashboard payload; tolerate failures."""
    try:
        from monitoring.mission_control import collect_payload  # type: ignore
        return collect_payload() or {}
    except Exception as e:  # noqa: BLE001 — daemon must not die on collector errors
        _log.warning("dfv.collect_failed", error=str(e))
        return {}


def brief() -> dict[str, Any]:
    """Pre-market / on-demand brief. The first thing DFV produces every morning."""
    dfv = DFV()
    payload = _safe_collect_payload()
    portfolio = payload.get("portfolio", {})
    war_room = payload.get("war_room", {})

    held_symbols = sorted({
        (p.get("symbol") or p.get("underlying") or "").upper()
        for acct in portfolio.get("accounts", {}).values()
        for p in acct.get("positions", [])
        if (p.get("symbol") or p.get("underlying"))
    })
    held_symbols = [s for s in held_symbols if s]

    theses = dfv.thesis.all()
    missing_theses = [s for s in held_symbols if s not in theses]
    stale_theses = dfv.thesis.needs_review(max_age_days=30)
    watchlist = dfv.watchlist.all()

    report = {
        "type": "brief",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "voice": "DFV / Roaring Kitty",
        "portfolio_summary": {
            "total_equity_usd": portfolio.get("total_assets_usd")
            or portfolio.get("total_equity")
            or portfolio.get("total_value"),
            "cash_usd": portfolio.get("cash_usd"),
            "buying_power_usd": portfolio.get("buying_power_usd"),
            "open_positions": len([
                p for acct in portfolio.get("accounts", {}).values()
                for p in acct.get("positions", [])
            ]),
        },
        "war_room": {
            "composite": war_room.get("composite_score"),
            "regime": war_room.get("regime"),
            "phase": war_room.get("phase"),
            "mandate": war_room.get("mandate"),
        },
        "discipline": {
            "held_symbols": held_symbols,
            "missing_thesis": missing_theses,
            "stale_thesis": stale_theses,
            "watchlist_size": len(watchlist),
        },
        "headline": _headline(missing_theses, stale_theses, war_room),
    }
    path = _save_brief("brief", report)
    report["saved_to"] = str(path.relative_to(REPO_ROOT))
    _log.info("dfv.brief", missing=len(missing_theses), stale=len(stale_theses))
    return report


def midday() -> dict[str, Any]:
    """Midday position drift + flow check on holdings."""
    payload = _safe_collect_payload()
    report = {
        "type": "midday",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "war_room_now": payload.get("war_room", {}),
        "alerts": payload.get("alerts", []),
        "note": "Drift check, flow check on held names. No trades — observe.",
    }
    _save_brief("midday", report)
    return report


def eod() -> dict[str, Any]:
    """End-of-day debrief: P&L attribution, conviction nudges, tomorrow's catalysts."""
    dfv = DFV()
    payload = _safe_collect_payload()
    pnl = payload.get("pnl", {}) or {}
    report = {
        "type": "eod",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pnl": {
            "today_realized": pnl.get("today_realized"),
            "mtd_realized": pnl.get("mtd_realized"),
            "ytd_realized": pnl.get("ytd_realized"),
        },
        "thesis_review_needed": dfv.thesis.needs_review(max_age_days=30),
        "decisions_today": len(dfv.decisions.tail(500)),
        "note": "Update theses. Nudge conviction. Write the lesson down.",
    }
    _save_brief("eod", report)
    return report


def weekend_dd() -> dict[str, Any]:
    """Weekend deep DD slot — read 10-Qs, refresh screens, post-mortems."""
    dfv = DFV()
    report = {
        "type": "weekend_dd",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "open_theses": list(dfv.thesis.all().keys()),
        "to_review_this_weekend": dfv.thesis.needs_review(max_age_days=14),
        "note": "Read the filings. Refresh the deep-value and squeeze screens. "
                "Write up any closed positions from the week.",
    }
    _save_brief("weekend_dd", report)
    return report


def _headline(missing: list[str], stale: list[str], war_room: dict[str, Any]) -> str:
    bits: list[str] = []
    regime = (war_room.get("regime") or "").upper() or "?"
    phase = (war_room.get("phase") or "").lower() or "?"
    bits.append(f"Regime {regime} · phase {phase}.")
    if missing:
        bits.append(f"{len(missing)} held name(s) without a thesis: {', '.join(missing[:5])}"
                    + ("…" if len(missing) > 5 else "") + ". Hard rule #1.")
    if stale:
        bits.append(f"{len(stale)} thesis review(s) overdue.")
    if not missing and not stale:
        bits.append("Discipline clean. I like the book.")
    return " ".join(bits)

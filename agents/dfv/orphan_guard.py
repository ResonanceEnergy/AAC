from __future__ import annotations

"""Orphan-position guard — auto-skeleton thesis writer.

The hard rule: "No position without a written thesis." When a position
appears in any venue without a corresponding thesis, the daemon writes a
TODO skeleton (conviction=0, max_pct_book=0, all fields marked TODO) and
fires a notification. The operator can never plead "I forgot to write one."
"""

from typing import Any

import structlog

from agents.dfv.decision_engine import DFV
from agents.dfv.notifications import notify

_log = structlog.get_logger(__name__)

SKELETON_TAG = "auto_skeleton"
SKELETON_AUTHOR = "DFV-auto-skeleton"
SKELETON_TODO = "TODO: write thesis before next session"


def _iter_positions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten positions from mission_control payload into [{symbol, venue, qty}]."""
    out: list[dict[str, Any]] = []
    portfolio = payload.get("portfolio") or {}
    accounts = portfolio.get("accounts") or {}
    iterable = accounts.items() if isinstance(accounts, dict) else enumerate(accounts)
    for venue_key, acct in iterable:
        if not isinstance(acct, dict):
            continue
        venue = acct.get("venue") or acct.get("name") or str(venue_key)
        for p in acct.get("positions", []) or []:
            sym = (p.get("symbol") or p.get("ticker") or "").upper()
            if not sym:
                continue
            qty = p.get("quantity") or p.get("qty") or p.get("position") or 0
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                qty_f = 0.0
            if qty_f == 0:
                continue
            out.append({"symbol": sym, "venue": str(venue), "qty": qty_f})
    return out


def detect_orphans(dfv: DFV, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Positions present in payload without a thesis on file."""
    orphans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pos in _iter_positions(payload):
        sym = pos["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        if dfv.thesis.has(sym):
            continue
        orphans.append(pos)
    return orphans


def _build_skeleton(symbol: str, venue: str, qty: float) -> dict[str, Any]:
    return {
        "thesis": SKELETON_TODO,
        "conviction": 0,
        "horizon": "TBD",
        "catalysts": [],
        "invalidation": "TODO: define invalidation",
        "target": {"raw": ""},
        "sizing": {"max_pct_book": 0.0},
        "author": SKELETON_AUTHOR,
        "tags": [SKELETON_TAG, f"venue:{venue.lower()}", f"first_qty:{qty}"],
    }


def write_skeletons(dfv: DFV, orphans: list[dict[str, Any]]) -> list[str]:
    """Write TODO skeleton theses for each orphan. Returns symbols touched."""
    written: list[str] = []
    for orphan in orphans:
        sym = orphan["symbol"]
        skel = _build_skeleton(sym, orphan["venue"], orphan["qty"])
        dfv.thesis.set(sym, **skel)
        dfv.conviction.set(sym, 0, reason="auto-skeleton: orphan position detected")
        written.append(sym)
        notify(
            dfv=dfv,
            kind="skeleton_thesis",
            symbol=sym,
            title=f"📝 Auto-skeleton thesis written for {sym}",
            body=f"Orphan position on {orphan['venue']} (qty={orphan['qty']}). Fill in the thesis before next session.",
            severity="warn",
            dedupe_key=f"skeleton:{sym}",
            extra={"venue": orphan["venue"], "qty": orphan["qty"]},
        )
    if written:
        _log.warning("dfv.orphan_guard.skeletons_written", count=len(written), symbols=written)
    return written


def scan_and_stub(payload: dict[str, Any] | None = None, dfv: DFV | None = None) -> dict[str, Any]:
    """Daemon hook: detect orphans + write skeletons. Idempotent."""
    inst = dfv or DFV()
    if not (inst.doctrine.get("orphan_guard") or {}).get("enabled", True):
        return {"enabled": False, "orphans": [], "written": []}
    if payload is None:
        try:
            from agents.dfv import routines as r  # noqa: PLC0415
            payload = r._safe_collect_payload()
        except Exception as exc:  # noqa: BLE001
            _log.warning("dfv.orphan_guard.payload_failed", error=str(exc))
            payload = {}
    orphans = detect_orphans(inst, payload or {})
    written = write_skeletons(inst, orphans)
    return {
        "enabled": True,
        "orphans": [o["symbol"] for o in orphans],
        "written": written,
    }

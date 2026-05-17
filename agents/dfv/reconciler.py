from __future__ import annotations

"""Position reconciler.

Compares positions across venues + against an optional second source
(CentralAccounting if importable). Computes per-symbol qty drift across
venues. Mismatches → red banner in dashboard + notification.

Single source of truth = mission_control payload. CentralAccounting is a
secondary check; absent it, we still report per-symbol totals so the
operator can eyeball discrepancies.
"""

from collections import defaultdict
from typing import Any

import structlog

from agents.dfv.decision_engine import DFV
from agents.dfv.notifications import notify

_log = structlog.get_logger(__name__)


def _venue_positions(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """{venue: {symbol: qty}} from payload."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    portfolio = payload.get("portfolio") or {}
    accounts = portfolio.get("accounts") or {}
    iterable = accounts.items() if isinstance(accounts, dict) else enumerate(accounts)
    for venue_key, acct in iterable:
        if not isinstance(acct, dict):
            continue
        venue = (acct.get("venue") or acct.get("name") or str(venue_key)).upper()
        for p in acct.get("positions", []) or []:
            sym = (p.get("symbol") or p.get("ticker") or "").upper()
            if not sym:
                continue
            qty = p.get("quantity") or p.get("qty") or p.get("position") or 0
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                continue
            if qty_f == 0:
                continue
            out[venue][sym] = out[venue].get(sym, 0.0) + qty_f
    return dict(out)


def _central_accounting_positions() -> dict[str, float] | None:
    """Best-effort secondary source. Returns None when unavailable."""
    try:
        from CentralAccounting import positions as ca_positions  # type: ignore
        snap = ca_positions.snapshot()  # type: ignore[attr-defined]
        if not isinstance(snap, dict):
            return None
        return {str(k).upper(): float(v) for k, v in snap.items() if v}
    except Exception:  # noqa: BLE001
        return None


def _live_ibkr_snapshot() -> dict[str, Any] | None:
    """Third source: live IBKR via TWS/Gateway. {positions: {sym: qty}, account_summary: {...}}.

    Returns None when IBKR is unreachable (TWS down, wrong port, etc.). Never
    raises — reconciler must keep working even if IBKR is offline.
    """
    import asyncio

    try:
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
    except Exception as exc:  # noqa: BLE001
        _log.info("dfv.reconciler.ibkr_unavailable", reason=str(exc))
        return None

    async def _fetch() -> dict[str, Any]:
        conn = IBKRConnector()
        ok = await conn.connect()
        if not ok:
            return {}
        try:
            positions = await conn.get_positions()
            summary = await conn.get_account_summary()
        finally:
            try:
                if hasattr(conn, "disconnect"):
                    await conn.disconnect()
            except Exception:  # noqa: BLE001
                pass
        pos_map: dict[str, float] = {}
        for p in positions or []:
            sym = str(p.get("symbol") or "").upper()
            qty = p.get("quantity") or 0
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                continue
            if not sym or qty_f == 0:
                continue
            pos_map[sym] = pos_map.get(sym, 0.0) + qty_f
        return {"positions": pos_map, "account_summary": summary or {}}

    try:
        # Reconciler runs in sync context (CLI / Streamlit); spin a fresh loop.
        try:
            asyncio.get_running_loop()
            # If we're already inside a loop (rare here), bail — don't deadlock.
            _log.warning("dfv.reconciler.ibkr_in_running_loop")
            return None
        except RuntimeError:
            return asyncio.run(asyncio.wait_for(_fetch(), timeout=15)) or None
    except Exception as exc:  # noqa: BLE001
        _log.info("dfv.reconciler.ibkr_fetch_failed", err=str(exc))
        return None


def reconcile(
    payload: dict[str, Any] | None = None,
    dfv: DFV | None = None,
    *,
    use_live_ibkr: bool | None = None,
) -> dict[str, Any]:
    """Build reconciliation snapshot. Stores to ReconciliationLog + notifies on diff."""
    inst = dfv or DFV()
    cfg = inst.doctrine.get("reconciler") or {}
    if not cfg.get("enabled", True):
        return {"enabled": False, "mismatches": []}

    if payload is None:
        try:
            from agents.dfv import routines as r  # noqa: PLC0415
            payload = r._safe_collect_payload()
        except Exception:  # noqa: BLE001
            payload = {}

    by_venue = _venue_positions(payload or {})
    ca = _central_accounting_positions()
    want_live = use_live_ibkr if use_live_ibkr is not None else cfg.get("use_live_ibkr", True)
    live_ibkr = _live_ibkr_snapshot() if want_live else None

    # Aggregate per-symbol totals across venues
    totals: dict[str, float] = defaultdict(float)
    for venue_map in by_venue.values():
        for sym, qty in venue_map.items():
            totals[sym] += qty

    mismatches: list[dict[str, Any]] = []

    # Diff #1 — CentralAccounting vs portfolio totals
    if ca is not None:
        all_syms = set(totals) | set(ca)
        for sym in sorted(all_syms):
            pf_qty = totals.get(sym, 0.0)
            ca_qty = ca.get(sym, 0.0)
            if abs(pf_qty - ca_qty) > 1e-6:
                mismatches.append({
                    "symbol": sym,
                    "kind": "venue_vs_central_accounting",
                    "portfolio_qty": pf_qty,
                    "central_accounting_qty": ca_qty,
                    "delta": pf_qty - ca_qty,
                })

    # Diff #2 — any symbol present in multiple venues (often a copy-paste bug)
    sym_venues: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for venue, syms in by_venue.items():
        for sym, qty in syms.items():
            sym_venues[sym].append((venue, qty))
    for sym, vens in sym_venues.items():
        if len(vens) > 1:
            mismatches.append({
                "symbol": sym,
                "kind": "duplicated_across_venues",
                "venues": [{"venue": v, "qty": q} for v, q in vens],
            })

    # Diff #3 — live IBKR vs payload IBKR slice (rule #4: act on real numbers, not cache)
    live_ibkr_positions: dict[str, float] = {}
    if live_ibkr is not None:
        live_ibkr_positions = live_ibkr.get("positions") or {}
        payload_ibkr = by_venue.get("IBKR") or {}
        all_syms = set(live_ibkr_positions) | set(payload_ibkr)
        for sym in sorted(all_syms):
            live_qty = float(live_ibkr_positions.get(sym, 0.0))
            cache_qty = float(payload_ibkr.get(sym, 0.0))
            if abs(live_qty - cache_qty) > 1e-6:
                mismatches.append({
                    "symbol": sym,
                    "kind": "ibkr_live_vs_cache",
                    "live_qty": live_qty,
                    "cache_qty": cache_qty,
                    "delta": live_qty - cache_qty,
                })

    snapshot = {
        "by_venue": by_venue,
        "totals": dict(totals),
        "central_accounting_available": ca is not None,
        "ibkr_live_available": live_ibkr is not None,
        "ibkr_live_positions": live_ibkr_positions,
        "ibkr_account_summary": (live_ibkr or {}).get("account_summary", {}),
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
    }
    # Don't clobber a good snapshot with an empty one (collector may have failed silently).
    # Only skip when literally no observation source produced anything.
    no_observation = (not by_venue) and (ca is None) and (live_ibkr is None)
    if no_observation:
        _log.info("dfv.reconciler.no_observation_skipped")
        snapshot["skipped_write_reason"] = "no_observation"
    else:
        inst.reconciliation.write(snapshot)

    if mismatches and cfg.get("notify_on_mismatch", True):
        notify(
            dfv=inst,
            kind="reconcile_mismatch",
            title=f"🚨 Position drift — {len(mismatches)} mismatch(es)",
            body="\n".join(
                f"{m['symbol']}: {m['kind']}" for m in mismatches[:5]
            ),
            severity="warn",
            dedupe_key="reconcile:" + ",".join(sorted({m["symbol"] for m in mismatches})),
            extra={"count": len(mismatches)},
        )
        _log.warning("dfv.reconciler.mismatch", count=len(mismatches))

    return snapshot

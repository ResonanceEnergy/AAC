from __future__ import annotations

"""DFV notification dispatcher.

Single entry-point used by orphan_guard, reconciler, breach detector, and the
roll engine. Every call:
  1. Appends to NotificationsLog (audit + dashboard tail). Always.
  2. Best-effort delivery via shared.alerter.Alerter (Telegram).

Never raises. Returns the appended record (or None on dedupe).
"""

from typing import Any

import structlog

from agents.dfv.decision_engine import DFV

_log = structlog.get_logger(__name__)


_ALERTER_EVENT_MAP: dict[str, str] = {
    "invalidation_breach": "DRAWDOWN_TRIPPED",
    "orphan_position":     "SYSTEM",
    "reconcile_mismatch":  "SYSTEM",
    "roll_or_kill":        "ROLL_CLOSE",
    "skeleton_thesis":     "SYSTEM",
    "journal_prompt":      "EOD_BRIEF",
    "jury_dissent":        "SYSTEM",
}


def notify(
    *,
    kind: str,
    title: str,
    body: str = "",
    symbol: str = "",
    severity: str = "info",
    dedupe_key: str | None = None,
    extra: dict[str, Any] | None = None,
    dfv: DFV | None = None,
) -> dict[str, Any] | None:
    """Audit-log + best-effort push. Returns the appended record or None."""
    inst = dfv or DFV()
    rec = inst.notifications.append(
        kind=kind,
        symbol=symbol,
        title=title,
        body=body,
        severity=severity,
        dedupe_key=dedupe_key,
        extra=extra,
    )
    if rec is None:
        return None
    # Check doctrine before pushing
    cfg = (inst.doctrine.get("breach_alerts") or {})
    push_targets = set(cfg.get("push_via") or [])
    if "telegram" in push_targets:
        _push_telegram(kind, title, body)
    return rec


def _push_telegram(kind: str, title: str, body: str) -> bool:
    try:
        from shared.alerter import Alerter
    except ImportError:
        return False
    try:
        alerter = Alerter()
        if not alerter.enabled:
            return False
        event_type = _ALERTER_EVENT_MAP.get(kind, "SYSTEM")
        msg = title if not body else f"{title}\n{body}"
        return alerter.send(event_type, msg)
    except Exception as exc:  # noqa: BLE001 — notifications must never raise
        _log.warning("dfv.notify.telegram_failed", error=str(exc))
        return False


def tail(n: int = 50, dfv: DFV | None = None) -> list[dict[str, Any]]:
    """Tail the audit log for the dashboard."""
    return (dfv or DFV()).notifications.tail(n)

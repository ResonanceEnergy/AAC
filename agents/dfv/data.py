from __future__ import annotations

"""DFV data helpers — IBKR / mission_control only.

History: this module briefly wrapped CoinGecko, FRED and Finnhub for the
Sprint 1 cadence routines. Those three keys are empty in this repo's
.env (and the surrounding integrations clients have known issues), so
the dependency was orphaned on 2026-05-15. DFV now relies exclusively
on what mission_control already aggregates from IBKR / Moomoo / war
room. If you need an external feed, wire it through mission_control or
shared/data_sources first — never call an external API directly from a
DFV routine.
"""

from datetime import datetime, timezone
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# Universe constants kept as documentation; routines reference them in
# headlines but no longer fan out external quote calls.
ASIA_ADRS = ("BABA", "JD", "PDD", "TSM", "BIDU", "SE", "TM", "NIO")
US_FUTURES_PROXIES = ("SPY", "QQQ", "IWM", "DIA")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty(kind: str) -> dict[str, Any]:
    """Sentinel payload returned when an upstream feed is unavailable."""
    return {"ok": False, "ts": _utc_now(), "kind": kind, "note": "feed orphaned"}

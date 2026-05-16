from __future__ import annotations

"""P&L attribution — group realized P&L by thesis tag, conviction tier, and verdict-at-open.

Inputs:
  - postmortems.jsonl: closed positions with realized $ + symbol
  - thesis_log.json:   tags + final conviction
  - decisions.jsonl:   verdict at open (approved | warned | vetoed)

The point isn't perfect accounting — it's accountability. Did the tier-5
"screaming" positions actually earn? Did the auto-skeleton positions bleed?
"""

from collections import defaultdict
from typing import Any

from agents.dfv.decision_engine import DFV


def _verdict_for_symbol(decisions: list[dict[str, Any]], symbol: str) -> str:
    """Earliest non-vetoed verdict for the symbol — that's the entry verdict."""
    sym = symbol.upper()
    for rec in decisions:
        if (rec.get("symbol") or "").upper() != sym:
            continue
        d = rec.get("decision") or {}
        verdict = str(d.get("verdict", ""))
        if verdict in ("approved", "warned"):
            return verdict
    return "unknown"


def attribute(dfv: DFV | None = None) -> dict[str, Any]:
    """Return attribution grouped by (tag, tier, verdict). Plus overall totals."""
    inst = dfv or DFV()
    postmortems = inst.postmortems.all()
    theses = inst.thesis.all()
    decisions = inst.decisions.tail(10_000)

    by_tag: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "realized": 0.0, "wins": 0, "symbols": []})
    by_tier: dict[int, dict[str, Any]] = defaultdict(lambda: {"n": 0, "realized": 0.0, "wins": 0, "symbols": []})
    by_verdict: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "realized": 0.0, "wins": 0, "symbols": []})

    total = {"n": 0, "realized": 0.0, "wins": 0}

    for pm in postmortems:
        sym = (pm.get("symbol") or "").upper()
        realized = pm.get("realized_pnl") or pm.get("pnl") or 0.0
        try:
            realized_f = float(realized)
        except (TypeError, ValueError):
            realized_f = 0.0

        thesis = theses.get(sym, {})
        tags = thesis.get("tags") or ["untagged"]
        tier = int(thesis.get("conviction") or 0)
        verdict = _verdict_for_symbol(decisions, sym)

        is_win = realized_f > 0
        total["n"] += 1
        total["realized"] += realized_f
        total["wins"] += int(is_win)

        for tag in tags:
            b = by_tag[tag]
            b["n"] += 1
            b["realized"] += realized_f
            b["wins"] += int(is_win)
            b["symbols"].append(sym)
        b = by_tier[tier]
        b["n"] += 1
        b["realized"] += realized_f
        b["wins"] += int(is_win)
        b["symbols"].append(sym)
        b = by_verdict[verdict]
        b["n"] += 1
        b["realized"] += realized_f
        b["wins"] += int(is_win)
        b["symbols"].append(sym)

    def _add_hit_rate(d: dict[Any, dict[str, Any]]) -> dict[Any, dict[str, Any]]:
        for v in d.values():
            v["hit_rate"] = (v["wins"] / v["n"]) if v["n"] else 0.0
            v["symbols"] = sorted(set(v["symbols"]))
        return d

    return {
        "total": {
            **total,
            "hit_rate": (total["wins"] / total["n"]) if total["n"] else 0.0,
        },
        "by_tag":     _add_hit_rate(dict(by_tag)),
        "by_tier":    _add_hit_rate({str(k): v for k, v in by_tier.items()}),
        "by_verdict": _add_hit_rate(dict(by_verdict)),
    }

"""Decisions memory log — TradingAgents-style reflection.

Persists every PM/debate decision as a markdown block in
``data/memory/decisions.md`` so future runs can learn from prior calls.
Entries are also appended to ``data/memory/decisions.jsonl`` for
machine-readable replay (PnL realisation later).

Format (markdown):
    ## 2026-05-15T16:30 | bull_bear_debate | SPY
    **Verdict:** bullish (confidence 0.6)
    **Thesis:** ...
    **Tools:** get_index_flow_pillar, get_call_options_pillar
    **Realised PnL:** _pending_

Public API:
    append_decision(kind, thesis, verdict, *, symbol=None, confidence=None,
                    tools=None, extras=None) -> dict
    recent_decisions(n=5) -> list[dict]
    update_realised_pnl(decision_id, pnl_usd) -> bool
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger().bind(component="aac_agents.memory")

_MEMORY_DIR = Path("data") / "memory"
_MD_PATH = _MEMORY_DIR / "decisions.md"
_JSONL_PATH = _MEMORY_DIR / "decisions.jsonl"


def _ensure_dir() -> None:
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def append_decision(
    kind: str,
    thesis: str,
    verdict: str,
    *,
    symbol: str | None = None,
    confidence: float | None = None,
    tools: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a decision to both the markdown log and the jsonl replay file."""
    _ensure_dir()
    decision_id = uuid.uuid4().hex[:10]
    ts = datetime.now().isoformat(timespec="minutes")
    entry: dict[str, Any] = {
        "id": decision_id,
        "ts": ts,
        "kind": kind,
        "symbol": symbol,
        "verdict": verdict,
        "confidence": confidence,
        "thesis": thesis,
        "tools": tools or [],
        "realised_pnl_usd": None,
        "extras": extras or {},
    }

    # JSONL — single line per entry, easy to replay later
    try:
        with _JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:
        _log.warning("decisions_jsonl_write_failed", error=str(exc))

    # Markdown — human-readable
    sym = f" | {symbol}" if symbol else ""
    conf = f" (confidence {confidence:.2f})" if isinstance(confidence, (int, float)) else ""
    tool_line = ", ".join(f"`{t}`" for t in (tools or [])) or "_(none)_"
    block = (
        f"\n## {ts} | {kind}{sym}  \n"
        f"_id:_ `{decision_id}`  \n"
        f"**Verdict:** {verdict}{conf}  \n"
        f"**Tools:** {tool_line}  \n"
        f"**Thesis:** {thesis.strip()}  \n"
        f"**Realised PnL:** _pending_  \n"
    )
    try:
        if not _MD_PATH.exists():
            _MD_PATH.write_text("# AAC Decisions Log\n\n_TradingAgents-style reflection memory._\n", encoding="utf-8")
        with _MD_PATH.open("a", encoding="utf-8") as f:
            f.write(block)
    except OSError as exc:
        _log.warning("decisions_md_write_failed", error=str(exc))

    _log.info("decision_logged", id=decision_id, kind=kind, verdict=verdict, symbol=symbol)
    return entry


def recent_decisions(n: int = 5) -> list[dict[str, Any]]:
    """Return the last ``n`` decisions from the jsonl log (newest first)."""
    if not _JSONL_PATH.exists():
        return []
    try:
        lines = _JSONL_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _log.warning("decisions_jsonl_read_failed", error=str(exc))
        return []
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= n:
            break
    return out


def update_realised_pnl(decision_id: str, pnl_usd: float) -> bool:
    """Patch the realised PnL on a prior decision. Updates jsonl in-place."""
    if not _JSONL_PATH.exists():
        return False
    try:
        lines = _JSONL_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    updated = False
    new_lines: list[str] = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        if obj.get("id") == decision_id:
            obj["realised_pnl_usd"] = float(pnl_usd)
            updated = True
        new_lines.append(json.dumps(obj, default=str))
    if updated:
        try:
            _JSONL_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        except OSError as exc:
            _log.warning("decisions_jsonl_update_failed", error=str(exc))
            return False
    return updated


def format_for_prompt(n: int = 5) -> str:
    """Render last N decisions as a compact prompt fragment for new runs."""
    decisions = recent_decisions(n=n)
    if not decisions:
        return ""
    lines = ["RECENT DECISIONS (most recent first):"]
    for d in decisions:
        pnl = d.get("realised_pnl_usd")
        pnl_str = f"${pnl:+,.0f}" if isinstance(pnl, (int, float)) else "pending"
        sym = d.get("symbol") or "—"
        lines.append(
            f"- {d.get('ts', '?')} [{d.get('kind', '?')} {sym}] verdict={d.get('verdict', '?')} pnl={pnl_str}"
        )
    return "\n".join(lines)

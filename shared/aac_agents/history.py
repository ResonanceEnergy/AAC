"""Lightweight per-agent conversation memory.

Stores the last N briefings/answers for each agent in a JSON file under
`data/agents/<agent>_history.json`. The runtime injects a 1-line "previous
session" hint into the system prompt so agents can build on prior context
without ballooning the conversation token count.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger().bind(component="aac_agents.history")

_HISTORY_DIR = Path("data") / "agents"
_MAX_ENTRIES = 20  # rolling window per agent


def _path_for(agent: str) -> Path:
    return _HISTORY_DIR / f"{agent}_history.json"


def load_history(agent: str) -> list[dict[str, Any]]:
    """Return the persisted history list for an agent (newest last)."""
    path = _path_for(agent)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("history_read_failed", agent=agent, error=str(exc))
        return []


def append_history(agent: str, prompt: str, answer: str, tool_calls: int) -> None:
    """Append a turn to an agent's rolling history."""
    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history = load_history(agent)
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt[:500],
            "answer_preview": answer[:1500],
            "tool_calls": tool_calls,
        })
        history = history[-_MAX_ENTRIES:]
        _path_for(agent).write_text(
            json.dumps(history, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:
        _log.warning("history_write_failed", agent=agent, error=str(exc))


def recent_summary(agent: str, n: int = 3) -> str:
    """Return a compact prose summary of the last N turns for prompt injection."""
    history = load_history(agent)
    if not history:
        return ""
    tail = history[-n:]
    lines = ["PREVIOUS SESSIONS (most recent last):"]
    for h in tail:
        ts = h.get("timestamp", "")[:10]
        prompt = (h.get("prompt") or "").replace("\n", " ")[:120]
        ans = (h.get("answer_preview") or "").replace("\n", " ")[:200]
        lines.append(f"- [{ts}] Q: {prompt} → A: {ans}")
    return "\n".join(lines)

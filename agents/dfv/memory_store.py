from __future__ import annotations

"""DFV memory store — JSON-backed thesis log, conviction tracker, watchlist, decisions log.

Plain, file-backed, atomic-enough for a single-operator desk.  No DB.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _log.warning("dfv.memory.read_failed", path=str(path), error=str(e))
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Thesis Log ────────────────────────────────────────────────────────────
class ThesisLog:
    """Per-ticker investment thesis. Required before any position."""

    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)
        self._data: dict[str, dict[str, Any]] = _read_json(self.path, {})

    def has(self, symbol: str) -> bool:
        return symbol.upper() in self._data

    def get(self, symbol: str) -> dict[str, Any] | None:
        return self._data.get(symbol.upper())

    def set(
        self,
        symbol: str,
        *,
        thesis: str,
        conviction: int,
        horizon: str,
        catalysts: list[str],
        invalidation: str,
        target: dict[str, Any],
        sizing: dict[str, Any],
        author: str = "DFV",
    ) -> dict[str, Any]:
        sym = symbol.upper()
        prior = self._data.get(sym, {})
        rec = {
            "symbol": sym,
            "thesis": thesis.strip(),
            "conviction": int(conviction),
            "horizon": horizon,
            "catalysts": list(catalysts),
            "invalidation": invalidation,
            "target": target,
            "sizing": sizing,
            "author": author,
            "created": prior.get("created", _utc_now()),
            "updated": _utc_now(),
            "revision": int(prior.get("revision", 0)) + 1,
        }
        self._data[sym] = rec
        _atomic_write_json(self.path, self._data)
        _log.info("dfv.thesis.set", symbol=sym, conviction=rec["conviction"], rev=rec["revision"])
        return rec

    def all(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)

    def needs_review(self, max_age_days: int = 30) -> list[str]:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        stale: list[str] = []
        for sym, rec in self._data.items():
            try:
                updated = datetime.fromisoformat(rec.get("updated", ""))
                if updated < cutoff:
                    stale.append(sym)
            except (ValueError, TypeError):
                stale.append(sym)
        return stale


# ── Conviction Tracker ────────────────────────────────────────────────────
class ConvictionTracker:
    """Conviction tier per ticker, 1–5. Drives sizing limits."""

    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)
        self._data: dict[str, dict[str, Any]] = _read_json(self.path, {})

    def get(self, symbol: str) -> int:
        return int(self._data.get(symbol.upper(), {}).get("tier", 0))

    def set(self, symbol: str, tier: int, *, reason: str = "") -> None:
        sym = symbol.upper()
        prior = self._data.get(sym, {})
        history = list(prior.get("history", []))
        if int(prior.get("tier", -1)) != int(tier):
            history.append({"ts": _utc_now(), "from": prior.get("tier"), "to": int(tier), "reason": reason})
        self._data[sym] = {"tier": int(tier), "updated": _utc_now(), "history": history[-50:]}
        _atomic_write_json(self.path, self._data)

    def all(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)


# ── Watchlist ─────────────────────────────────────────────────────────────
class Watchlist:
    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)
        self._data: dict[str, dict[str, Any]] = _read_json(self.path, {})

    def add(self, symbol: str, reason: str = "", source: str = "manual") -> None:
        sym = symbol.upper()
        if sym not in self._data:
            self._data[sym] = {"added": _utc_now(), "reason": reason, "source": source}
            _atomic_write_json(self.path, self._data)

    def remove(self, symbol: str) -> None:
        if symbol.upper() in self._data:
            del self._data[symbol.upper()]
            _atomic_write_json(self.path, self._data)

    def all(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)


# ── Decisions log (append-only JSONL) ─────────────────────────────────────
class DecisionsLog:
    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict[str, Any]) -> None:
        entry = {"ts": _utc_now(), **entry}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out: list[dict[str, Any]] = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

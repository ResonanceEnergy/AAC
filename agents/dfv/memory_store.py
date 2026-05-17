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
        roll_trigger_dte: int | None = None,
        tags: list[str] | None = None,
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
            "roll_trigger_dte": (
                int(roll_trigger_dte) if roll_trigger_dte is not None
                else int(prior.get("roll_trigger_dte", 21))  # doctrine default
            ),
            "tags": list(tags) if tags is not None else list(prior.get("tags", [])),
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

    def replace_all(self, entries: dict[str, dict[str, Any]]) -> None:
        """Atomically swap the entire watchlist (used by the EOD screener).

        Each entry value should be a dict; missing ``added`` is filled with
        the current UTC timestamp so downstream readers can always sort.
        """
        normalized: dict[str, dict[str, Any]] = {}
        for sym, payload in entries.items():
            data = dict(payload or {})
            data.setdefault("added", _utc_now())
            normalized[sym.upper()] = data
        self._data = normalized
        _atomic_write_json(self.path, self._data)


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


# ── Postmortems (append-only JSONL) ───────────────────────────────────────
class PostMortemLog:
    """Closed-position write-ups. Persona §3.7 — every closed name gets a lesson."""

    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = {"ts": _utc_now(), **entry}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def has(self, symbol: str, expiry: str) -> bool:
        sym = symbol.upper()
        for rec in self.all():
            if (rec.get("symbol", "").upper() == sym
                    and str(rec.get("expiry", "")) == str(expiry)):
                return True
        return False

    def for_symbol(self, symbol: str) -> list[dict[str, Any]]:
        sym = symbol.upper()
        return [r for r in self.all() if r.get("symbol", "").upper() == sym]


# ── Journal log (append-only JSONL) ───────────────────────────────────────
class JournalLog:
    """One-sentence-per-day journal. Prompted by EOD; surfaced in dashboard."""

    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: str, *, mood: str = "", tags: list[str] | None = None) -> dict[str, Any]:
        rec = {
            "ts": _utc_now(),
            "date": datetime.now(timezone.utc).date().isoformat(),
            "entry": entry.strip(),
            "mood": mood,
            "tags": list(tags or []),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def tail(self, n: int = 30) -> list[dict[str, Any]]:
        return self.all()[-n:]

    def has_today(self) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        return any(r.get("date") == today for r in self.all())


# ── Notifications (append-only JSONL + best-effort push) ──────────────────
class NotificationsLog:
    """Outbound notification queue.

    Every notification gets written to JSONL (for audit + dashboard tail);
    the optional `pusher` callable is invoked for fire-and-forget delivery
    (Telegram via shared.alerter, etc.). Never raises.
    """

    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        kind: str,
        symbol: str = "",
        title: str,
        body: str = "",
        severity: str = "info",
        dedupe_key: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Append a notification. If dedupe_key matches one written in the
        last 12h, the notification is suppressed (returns None)."""
        if dedupe_key and self._recently_seen(dedupe_key, hours=12):
            return None
        rec = {
            "ts": _utc_now(),
            "kind": kind,
            "symbol": (symbol or "").upper(),
            "title": title,
            "body": body,
            "severity": severity,
            "dedupe_key": dedupe_key,
            "extra": extra or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

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

    def _recently_seen(self, dedupe_key: str, *, hours: int = 12) -> bool:
        from datetime import timedelta
        if not self.path.exists():
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        # Scan last 500 lines — enough for any practical dedupe window.
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-500:]
        except OSError:
            return False
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("dedupe_key") != dedupe_key:
                continue
            try:
                ts = datetime.fromisoformat(str(rec.get("ts", "")))
                if ts >= cutoff:
                    return True
            except ValueError:
                continue
        return False


# ── Reconciliation state (latest snapshot — single JSON file) ─────────────
class ReconciliationLog:
    """Single-snapshot file: the most recent reconcile result.

    History is implied — every run overwrites. For diff history, scan the
    notifications JSONL (kind='reconcile_mismatch').
    """

    def __init__(self, path: str | Path):
        self.path = REPO_ROOT / Path(path)

    def write(self, snapshot: dict[str, Any]) -> None:
        snapshot = {"ts": _utc_now(), **snapshot}
        _atomic_write_json(self.path, snapshot)

    def read(self) -> dict[str, Any]:
        return _read_json(self.path, {})


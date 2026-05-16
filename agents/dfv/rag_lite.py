from __future__ import annotations

"""DFV RAG-lite — SQLite FTS5 over briefs, theses, postmortems, decisions, journal.

Zero external deps (sqlite3 + FTS5 ships with Python). Rebuild is cheap; we
just drop+re-index the whole virtual table since the corpus is small.

Use:
    from agents.dfv import rag_lite
    rag_lite.reindex()
    hits = rag_lite.search("PFF roll", k=5)
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from agents.dfv.decision_engine import DFV

_log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "agents" / "dfv" / "memory" / "rag_lite.sqlite"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS dfv_docs;
        CREATE VIRTUAL TABLE dfv_docs USING fts5(
            kind, symbol, ts, source, text,
            tokenize = 'porter unicode61'
        );
        """
    )


def _ingest_briefs(conn: sqlite3.Connection, dfv: DFV) -> int:
    brief_dir = REPO_ROOT / "agents" / "dfv" / "memory" / "briefs"
    if not brief_dir.exists():
        return 0
    n = 0
    for path in sorted(brief_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = json.dumps(data, default=str)
        conn.execute(
            "INSERT INTO dfv_docs(kind, symbol, ts, source, text) VALUES (?, ?, ?, ?, ?)",
            ("brief", "", str(data.get("generated_at") or ""), path.name, text[:200_000]),
        )
        n += 1
    return n


def _ingest_theses(conn: sqlite3.Connection, dfv: DFV) -> int:
    n = 0
    for sym, rec in dfv.thesis.all().items():
        text = json.dumps(rec, default=str)
        conn.execute(
            "INSERT INTO dfv_docs(kind, symbol, ts, source, text) VALUES (?, ?, ?, ?, ?)",
            ("thesis", sym, str(rec.get("updated") or ""), "thesis_log", text),
        )
        n += 1
    return n


def _ingest_postmortems(conn: sqlite3.Connection, dfv: DFV) -> int:
    n = 0
    for rec in dfv.postmortems.all():
        sym = (rec.get("symbol") or "").upper()
        text = json.dumps(rec, default=str)
        conn.execute(
            "INSERT INTO dfv_docs(kind, symbol, ts, source, text) VALUES (?, ?, ?, ?, ?)",
            ("postmortem", sym, str(rec.get("ts") or rec.get("closed_at") or ""), "postmortems", text),
        )
        n += 1
    return n


def _ingest_decisions(conn: sqlite3.Connection, dfv: DFV) -> int:
    n = 0
    for rec in dfv.decisions.tail(5_000):
        sym = (rec.get("symbol") or "").upper()
        text = json.dumps(rec, default=str)
        conn.execute(
            "INSERT INTO dfv_docs(kind, symbol, ts, source, text) VALUES (?, ?, ?, ?, ?)",
            ("decision", sym, str(rec.get("ts") or ""), "decisions", text),
        )
        n += 1
    return n


def _ingest_journal(conn: sqlite3.Connection, dfv: DFV) -> int:
    n = 0
    for rec in dfv.journal.all():
        text = json.dumps(rec, default=str)
        conn.execute(
            "INSERT INTO dfv_docs(kind, symbol, ts, source, text) VALUES (?, ?, ?, ?, ?)",
            ("journal", "", str(rec.get("ts") or ""), "journal", text),
        )
        n += 1
    return n


def reindex(dfv: DFV | None = None) -> dict[str, int]:
    """Drop and rebuild the FTS5 index. Returns per-source row counts."""
    inst = dfv or DFV()
    conn = _connect()
    try:
        _init_schema(conn)
        counts = {
            "briefs":      _ingest_briefs(conn, inst),
            "theses":      _ingest_theses(conn, inst),
            "postmortems": _ingest_postmortems(conn, inst),
            "decisions":   _ingest_decisions(conn, inst),
            "journal":     _ingest_journal(conn, inst),
        }
        conn.commit()
    finally:
        conn.close()
    _log.info("dfv.rag_lite.reindex", **counts)
    counts["total"] = sum(counts.values())
    counts["reindexed_at"] = int(datetime.now(timezone.utc).timestamp())
    return counts


def _ensure_index() -> None:
    if not DB_PATH.exists():
        reindex()


def _escape_match(query: str) -> str:
    """Wrap each token in double-quotes to neutralize FTS5 syntax errors."""
    tokens = [t for t in query.split() if t.strip()]
    if not tokens:
        return '""'
    return " ".join(f'"{t.replace(chr(34), chr(39))}"' for t in tokens)


def search(
    query: str,
    *,
    k: int = 8,
    kind: str | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    """Return top-k hits as [{kind, symbol, ts, source, snippet, rank}]."""
    _ensure_index()
    conn = _connect()
    try:
        sql = (
            "SELECT kind, symbol, ts, source, "
            "snippet(dfv_docs, 4, '⟪', '⟫', '…', 16) AS snippet, "
            "rank "
            "FROM dfv_docs WHERE dfv_docs MATCH ?"
        )
        params: list[Any] = [_escape_match(query)]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol.upper())
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            _log.warning("dfv.rag_lite.search_failed", error=str(exc))
            return []
        return [dict(r) for r in rows]
    finally:
        conn.close()


def context_for_prompt(query: str, *, k: int = 5, max_chars: int = 4_000) -> str:
    """Format top-k hits as plain-text context for LLM prompt injection."""
    hits = search(query, k=k)
    if not hits:
        return ""
    lines = ["## DFV memory hits"]
    used = 0
    for h in hits:
        line = f"- [{h['kind']}|{h['symbol'] or '·'}|{h['ts'][:10]}] {h['snippet']}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)

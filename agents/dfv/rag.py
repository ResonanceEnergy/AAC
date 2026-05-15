from __future__ import annotations

"""DFV semantic memory — LanceDB collection of briefs, theses, decisions.

Gives DFV a *recall* layer on top of the structured JSON stores in
`memory_store.py`. The structured stores hold facts; this module makes the
historical record searchable by natural language ("when did I last see this
setup?", "what did I decide last time TICKER printed earnings?").

Reuses the existing aac_rag stack (Ollama embeddings + LanceDB) with a
dedicated table so DFV memory does not pollute the codebase index.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger().bind(component="dfv.rag")

DFV_TABLE = "dfv_memory"


def _embed(texts: list[str], cfg) -> list[list[float]]:
    import ollama

    client = ollama.Client(host=cfg.embedding_endpoint)
    out: list[list[float]] = []
    for t in texts:
        resp = client.embeddings(model=cfg.embedding_model, prompt=t)
        out.append(list(resp["embedding"]))
    return out


def _open_table(cfg):
    """Open or create the DFV memory table (separate from the codebase index)."""
    import lancedb
    import pyarrow as pa

    cfg.lancedb_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(cfg.lancedb_path))

    if DFV_TABLE in db.table_names():
        return db, db.open_table(DFV_TABLE)

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("kind", pa.string()),       # brief | thesis | decision | note
            pa.field("symbol", pa.string()),     # may be "" for general notes/briefs
            pa.field("ts", pa.string()),         # ISO-8601 UTC
            pa.field("source", pa.string()),     # path or origin
            pa.field("text", pa.string()),       # human-readable summary used for embed
            pa.field("payload", pa.string()),    # JSON blob with structured detail
            pa.field("vector", pa.list_(pa.float32(), cfg.embedding_dim)),
        ]
    )
    table = db.create_table(DFV_TABLE, schema=schema, mode="create")
    return db, table


def _cfg():
    """Lazy import so the rest of DFV works without aac_rag's heavy deps loaded."""
    from shared.aac_rag.config import RagConfig
    return RagConfig.load()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_id(kind: str, source: str, ts: str) -> str:
    return hashlib.sha256(f"{kind}|{source}|{ts}".encode()).hexdigest()[:24]


# ── public API ────────────────────────────────────────────────────────────


def index_brief(brief_path: str | Path, payload: dict[str, Any] | None = None) -> str | None:
    """Embed and store a daily brief. Returns the row id or None on failure."""
    try:
        cfg = _cfg()
        path = Path(brief_path)
        if payload is None:
            payload = json.loads(path.read_text(encoding="utf-8"))
        text = _summarize_brief_for_embed(payload)
        ts = payload.get("generated_at") or _utc_now()
        row_id = _make_id("brief", str(path), ts)
        _, table = _open_table(cfg)
        # Avoid duplicates
        existing = table.to_pandas()
        if not existing.empty and (existing["id"] == row_id).any():
            return row_id
        vec = _embed([text], cfg)[0]
        table.add(
            [
                {
                    "id": row_id,
                    "kind": "brief",
                    "symbol": "",
                    "ts": ts,
                    "source": str(path),
                    "text": text,
                    "payload": json.dumps(payload, default=str),
                    "vector": vec,
                }
            ]
        )
        _log.info("dfv.rag.indexed_brief", id=row_id, path=str(path))
        return row_id
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as e:
        _log.warning("dfv.rag.index_brief_failed", error=str(e), path=str(brief_path))
        return None
    except Exception as e:  # noqa: BLE001 — never let RAG kill the brief
        _log.warning("dfv.rag.index_brief_failed", error=str(e), path=str(brief_path))
        return None


def index_thesis(symbol: str, record: dict[str, Any]) -> str | None:
    """Embed and store a thesis record."""
    try:
        cfg = _cfg()
        sym = symbol.upper()
        ts = record.get("updated") or _utc_now()
        rev = record.get("revision", 0)
        row_id = _make_id("thesis", f"{sym}:{rev}", ts)
        text = _summarize_thesis_for_embed(sym, record)
        _, table = _open_table(cfg)
        existing = table.to_pandas()
        if not existing.empty and (existing["id"] == row_id).any():
            return row_id
        vec = _embed([text], cfg)[0]
        table.add(
            [
                {
                    "id": row_id,
                    "kind": "thesis",
                    "symbol": sym,
                    "ts": ts,
                    "source": f"thesis_log:{sym}:r{rev}",
                    "text": text,
                    "payload": json.dumps(record, default=str),
                    "vector": vec,
                }
            ]
        )
        _log.info("dfv.rag.indexed_thesis", id=row_id, symbol=sym, rev=rev)
        return row_id
    except Exception as e:  # noqa: BLE001
        _log.warning("dfv.rag.index_thesis_failed", error=str(e), symbol=symbol)
        return None


def search(query: str, k: int = 5, kind: str | None = None,
           symbol: str | None = None) -> list[dict[str, Any]]:
    """Vector search DFV memory. Returns up to k hits as plain dicts."""
    try:
        cfg = _cfg()
        _, table = _open_table(cfg)
        qvec = _embed([query], cfg)[0]
        q = table.search(qvec).limit(int(k))
        wheres: list[str] = []
        if kind:
            wheres.append(f"kind = '{kind}'")
        if symbol:
            wheres.append(f"symbol = '{symbol.upper()}'")
        if wheres:
            q = q.where(" AND ".join(wheres))
        df = q.to_pandas()
        out: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            out.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "symbol": row["symbol"],
                    "ts": row["ts"],
                    "source": row["source"],
                    "score": round(1.0 / (1.0 + float(row.get("_distance", 0.0))), 4),
                    "text": (row["text"] or "")[:600],
                }
            )
        return out
    except Exception as e:  # noqa: BLE001
        _log.warning("dfv.rag.search_failed", error=str(e))
        return []


def reindex_all_theses() -> int:
    """Re-embed every thesis currently in thesis_log.json. Returns count."""
    from agents.dfv.memory_store import ThesisLog

    t = ThesisLog("agents/dfv/memory/thesis_log.json")
    n = 0
    for sym, rec in t.all().items():
        if index_thesis(sym, rec):
            n += 1
    return n


def reindex_all_briefs() -> int:
    """Re-embed every brief JSON under agents/dfv/memory/briefs/. Returns count."""
    briefs_dir = Path(__file__).resolve().parents[1] / "dfv" / "memory" / "briefs"
    if not briefs_dir.exists():
        return 0
    n = 0
    for p in sorted(briefs_dir.glob("*.json")):
        if index_brief(p):
            n += 1
    return n


# ── summary generators (what gets embedded) ──────────────────────────────


def _summarize_brief_for_embed(b: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append(f"DFV brief {b.get('generated_at', '?')}")
    parts.append(f"Headline: {b.get('headline', '')}")
    market = b.get("market_snapshot") or {}
    if market:
        parts.append(
            "Market: regime=%s phase=%s vix=%s spy=%s"
            % (
                market.get("regime"),
                market.get("phase"),
                market.get("vix"),
                market.get("spy_close"),
            )
        )
    p = b.get("portfolio_summary") or {}
    if p:
        parts.append(
            "Book: equity=$%s cash=$%s positions=%s"
            % (p.get("total_equity_usd"), p.get("cash_usd"), p.get("open_positions"))
        )
    disc = b.get("discipline") or {}
    held = disc.get("held_symbols") or b.get("held_symbols") or []
    if held:
        parts.append("Held: " + ", ".join(map(str, held[:25])))
    missing = disc.get("missing_thesis") or b.get("missing_thesis") or []
    if missing:
        parts.append("Missing thesis: " + ", ".join(map(str, missing[:25])))
    stale = disc.get("stale_thesis") or b.get("stale_thesis") or []
    if stale:
        parts.append("Stale thesis: " + ", ".join(map(str, stale[:25])))
    war = b.get("war_room") or {}
    if war:
        parts.append(
            "WarRoom: composite=%s regime=%s phase=%s mandate=%s"
            % (war.get("composite"), war.get("regime"),
               war.get("phase"), war.get("mandate"))
        )
    pnl = b.get("pnl") or {}
    if pnl:
        parts.append(
            "P&L: today=%s mtd=%s ytd=%s"
            % (pnl.get("today_realized"), pnl.get("mtd_realized"),
               pnl.get("ytd_realized"))
        )
    return "\n".join(parts)


def _summarize_thesis_for_embed(symbol: str, r: dict[str, Any]) -> str:
    cats = r.get("catalysts") or []
    return (
        f"Thesis {symbol} (conviction {r.get('conviction', '?')}, "
        f"horizon {r.get('horizon', '?')}, rev {r.get('revision', '?')}, "
        f"updated {r.get('updated', '?')}): "
        f"{r.get('thesis', '')}\n"
        f"Catalysts: {' | '.join(cats[:8])}\n"
        f"Invalidation: {r.get('invalidation', '')}"
    )

"""File walking, chunking, and indexing into LanceDB.

Incremental: a chunk is re-embedded only when the file's sha256 changes.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import structlog

from .config import REPO_ROOT, RagConfig

_log = structlog.get_logger().bind(component="aac_rag.indexer")

CODE_EXTS = {".py", ".toml", ".cfg", ".ini"}
DOC_EXTS = {".md", ".rst", ".txt"}
CONFIG_EXTS = {".yaml", ".yml", ".json"}


@dataclass
class Chunk:
    id: str
    path: str
    kind: str  # "code" | "doc" | "config"
    chunk_idx: int
    text: str
    sha: str
    mtime: float


def _classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in DOC_EXTS:
        return "doc"
    if suffix in CONFIG_EXTS:
        return "config"
    return "code"


def _matches_any(rel: str, patterns: Iterable[str]) -> bool:
    rel_posix = rel.replace("\\", "/")
    for p in patterns:
        if fnmatch.fnmatch(rel_posix, p):
            return True
    return False


def _walk(cfg: RagConfig) -> list[Path]:
    """Return all files matching include patterns and not excluded."""
    seen: set[Path] = set()
    for pattern in cfg.include:
        for p in REPO_ROOT.glob(pattern):
            if not p.is_file():
                continue
            rel = str(p.relative_to(REPO_ROOT))
            if _matches_any(rel, cfg.exclude):
                continue
            seen.add(p)
    return sorted(seen)


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += step
    return out


_HEADER_RE = re.compile(r"(?m)^#{1,3}\s+.+$")


def _split_markdown(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split markdown on H1/H2/H3 headers; further window large sections."""
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        return _sliding_window(text, max_chars, overlap)
    sections: list[str] = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections.append(text[start:end].strip())
    # Window any oversized section
    out: list[str] = []
    for sec in sections:
        if len(sec) <= max_chars:
            out.append(sec)
        else:
            out.extend(_sliding_window(sec, max_chars, overlap))
    return out


def _chunk_file(path: Path, cfg: RagConfig) -> tuple[str, str, list[Chunk]]:
    """Return (sha, kind, chunks) for a single file. Empty chunks list on read failure."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _log.warning("read_failed", path=str(path), error=str(e))
        return ("", "code", [])

    sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    kind = _classify(path)
    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    mtime = path.stat().st_mtime

    if kind == "doc":
        pieces = _split_markdown(text, cfg.doc_max_chars, cfg.doc_overlap_chars)
    else:
        pieces = _sliding_window(text, cfg.code_chunk_chars, cfg.code_overlap_chars)

    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        cid = hashlib.sha256(f"{rel}:{i}:{sha}".encode()).hexdigest()[:24]
        chunks.append(
            Chunk(
                id=cid,
                path=rel,
                kind=kind,
                chunk_idx=i,
                text=piece,
                sha=sha,
                mtime=mtime,
            )
        )
    return (sha, kind, chunks)


# ── Embedding via Ollama HTTP API ────────────────────────────────────────────


def _embed_batch(texts: list[str], cfg: RagConfig) -> list[list[float]]:
    """Embed a batch of texts via Ollama. Returns one vector per input."""
    import ollama

    client = ollama.Client(host=cfg.embedding_endpoint)
    out: list[list[float]] = []
    for t in texts:
        resp = client.embeddings(model=cfg.embedding_model, prompt=t)
        out.append(list(resp["embedding"]))
    return out


# ── LanceDB I/O ──────────────────────────────────────────────────────────────


def _open_table(cfg: RagConfig):
    import lancedb
    import pyarrow as pa

    cfg.lancedb_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(cfg.lancedb_path))

    if cfg.table_name in db.table_names():
        return db, db.open_table(cfg.table_name)

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("chunk_idx", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("sha", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), cfg.embedding_dim)),
        ]
    )
    table = db.create_table(cfg.table_name, schema=schema, mode="create")
    return db, table


# ── Sha cache (path -> sha) so we know what to re-index ──────────────────────


def _load_sha_cache(cfg: RagConfig) -> dict[str, str]:
    cfg.cache_path.mkdir(parents=True, exist_ok=True)
    p = cfg.cache_path / "sha_index.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_sha_cache(cfg: RagConfig, cache: dict[str, str]) -> None:
    p = cfg.cache_path / "sha_index.json"
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ── Public API ───────────────────────────────────────────────────────────────


def reindex(*, full: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Walk repo, chunk changed files, embed, write to LanceDB.

    Args:
        full: if True, ignore sha cache and re-embed everything.
        verbose: print progress.

    Returns summary dict.
    """
    cfg = RagConfig.load()
    db, table = _open_table(cfg)

    sha_cache = {} if full else _load_sha_cache(cfg)
    files = _walk(cfg)

    if verbose:
        print(f"[rag] discovered {len(files)} files")

    changed: list[Path] = []
    new_cache: dict[str, str] = {}
    file_meta: dict[Path, tuple[str, str]] = {}

    for path in files:
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        new_cache[rel] = sha
        file_meta[path] = (rel, sha)
        if sha_cache.get(rel) != sha:
            changed.append(path)

    if verbose:
        print(f"[rag] changed: {len(changed)} / {len(files)}")

    if not changed:
        _save_sha_cache(cfg, new_cache)
        return {
            "files_total": len(files),
            "files_changed": 0,
            "chunks_added": 0,
            "elapsed_s": 0.0,
        }

    started = time.time()
    chunks_added = 0
    BATCH = 32
    pending_records: list[dict[str, Any]] = []
    pending_paths_to_purge: set[str] = set()

    for n, path in enumerate(changed, 1):
        rel, sha = file_meta[path]
        _, _, chunks = _chunk_file(path, cfg)
        if not chunks:
            continue
        pending_paths_to_purge.add(rel)
        texts = [c.text for c in chunks]
        try:
            vectors = _embed_batch(texts, cfg)
        except Exception as e:
            _log.warning("embed_failed", path=rel, error=str(e))
            continue
        for c, v in zip(chunks, vectors):
            pending_records.append(
                {
                    "id": c.id,
                    "path": c.path,
                    "kind": c.kind,
                    "chunk_idx": c.chunk_idx,
                    "text": c.text,
                    "sha": c.sha,
                    "mtime": c.mtime,
                    "vector": v,
                }
            )
        chunks_added += len(chunks)

        if verbose and n % 25 == 0:
            print(f"[rag] embedded {n}/{len(changed)} files, {chunks_added} chunks")

        if len(pending_records) >= BATCH * 8:
            # Purge stale rows for these files
            if pending_paths_to_purge:
                quoted = ",".join(f"'{p}'" for p in pending_paths_to_purge)
                table.delete(f"path IN ({quoted})")
                pending_paths_to_purge.clear()
            table.add(pending_records)
            pending_records = []

    # Flush
    if pending_paths_to_purge:
        quoted = ",".join(f"'{p}'" for p in pending_paths_to_purge)
        table.delete(f"path IN ({quoted})")
    if pending_records:
        table.add(pending_records)

    _save_sha_cache(cfg, new_cache)
    elapsed = time.time() - started

    if verbose:
        print(
            f"[rag] done: {chunks_added} chunks from {len(changed)} files in {elapsed:.1f}s"
        )

    return {
        "files_total": len(files),
        "files_changed": len(changed),
        "chunks_added": chunks_added,
        "elapsed_s": round(elapsed, 1),
    }


def stats() -> dict[str, Any]:
    """Return index statistics."""
    cfg = RagConfig.load()
    if not cfg.lancedb_path.exists():
        return {"status": "not_indexed"}
    import lancedb

    db = lancedb.connect(str(cfg.lancedb_path))
    if cfg.table_name not in db.table_names():
        return {"status": "not_indexed"}
    table = db.open_table(cfg.table_name)
    n = table.count_rows()
    df = table.to_pandas()
    by_kind = df["kind"].value_counts().to_dict() if n else {}
    n_files = df["path"].nunique() if n else 0
    return {
        "status": "indexed",
        "chunks": int(n),
        "files": int(n_files),
        "by_kind": {k: int(v) for k, v in by_kind.items()},
        "lancedb_path": str(cfg.lancedb_path),
    }

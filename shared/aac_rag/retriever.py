"""Hybrid retrieval: vector + BM25, merged via reciprocal rank fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import RagConfig


@dataclass
class Hit:
    id: str
    path: str
    kind: str
    chunk_idx: int
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "kind": self.kind,
            "chunk_idx": self.chunk_idx,
            "score": round(self.score, 4),
            "text": self.text,
        }


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def _embed_query(text: str, cfg: RagConfig) -> list[float]:
    import ollama

    client = ollama.Client(host=cfg.embedding_endpoint)
    resp = client.embeddings(model=cfg.embedding_model, prompt=text)
    return list(resp["embedding"])


def _vector_search(table, query_vec: list[float], k: int, kind: str | None) -> list[Hit]:
    q = table.search(query_vec).limit(k)
    if kind:
        q = q.where(f"kind = '{kind}'")
    df = q.to_pandas()
    out: list[Hit] = []
    for _, row in df.iterrows():
        out.append(
            Hit(
                id=row["id"],
                path=row["path"],
                kind=row["kind"],
                chunk_idx=int(row["chunk_idx"]),
                text=row["text"],
                # LanceDB returns _distance (lower=better); convert to similarity
                score=1.0 / (1.0 + float(row.get("_distance", 0.0))),
            )
        )
    return out


def _bm25ish_search(table, query: str, k: int, kind: str | None) -> list[Hit]:
    """Lightweight keyword scoring: scan all rows, score by token overlap.

    Avoids the BM25 index dependency. AAC repo is small enough (<5k chunks)
    that a full pandas scan is fine for this purpose. Re-rank with vector
    upstream.
    """
    q_tokens = _tokens(query)
    if not q_tokens:
        return []
    df = table.to_pandas()
    if kind:
        df = df[df["kind"] == kind]
    scored: list[tuple[float, dict[str, Any]]] = []
    for _, row in df.iterrows():
        doc_tokens = _tokens(row["text"])
        overlap = len(q_tokens & doc_tokens)
        if overlap == 0:
            continue
        # Simple TF-style score; favor shorter matching docs
        score = overlap / (1.0 + len(doc_tokens) / 100.0)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[Hit] = []
    for s, row in scored[:k]:
        out.append(
            Hit(
                id=row["id"],
                path=row["path"],
                kind=row["kind"],
                chunk_idx=int(row["chunk_idx"]),
                text=row["text"],
                score=float(s),
            )
        )
    return out


def _rrf_merge(lists: list[list[Hit]], k_final: int, k_rrf: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    pool: dict[str, Hit] = {}
    fused: dict[str, float] = {}
    for hits in lists:
        for rank, h in enumerate(hits):
            pool.setdefault(h.id, h)
            fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (k_rrf + rank + 1)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    out: list[Hit] = []
    for cid, score in ordered[:k_final]:
        h = pool[cid]
        h.score = round(score, 4)
        out.append(h)
    return out


def retrieve(
    query: str,
    *,
    k: int | None = None,
    kind: str | None = None,
) -> list[Hit]:
    """Hybrid retrieval. `kind` is None | 'code' | 'doc' | 'config'."""
    import lancedb

    cfg = RagConfig.load()
    db = lancedb.connect(str(cfg.lancedb_path))
    if cfg.table_name not in db.table_names():
        return []
    table = db.open_table(cfg.table_name)

    k_final = k or cfg.k_final

    qvec = _embed_query(query, cfg)
    vec_hits = _vector_search(table, qvec, cfg.k_vector, kind)
    kw_hits = _bm25ish_search(table, query, cfg.k_bm25, kind)

    return _rrf_merge([vec_hits, kw_hits], k_final=k_final)

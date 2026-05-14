"""RAG configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "rag.yaml"


@dataclass
class RagConfig:
    embedding_model: str
    embedding_dim: int
    embedding_endpoint: str
    generation_model: str
    generation_endpoint: str
    lancedb_path: Path
    table_name: str
    cache_path: Path
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    code_chunk_chars: int = 1500
    code_overlap_chars: int = 200
    doc_max_chars: int = 2000
    doc_overlap_chars: int = 150
    k_vector: int = 12
    k_bm25: int = 8
    k_final: int = 8

    @classmethod
    def load(cls, path: Path | None = None) -> RagConfig:
        path = path or CONFIG_PATH
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        emb = raw["embeddings"]
        gen = raw["generation"]
        sto = raw["storage"]
        chk = raw.get("chunking", {})
        ret = raw.get("retrieval", {})

        return cls(
            embedding_model=emb["model"],
            embedding_dim=int(emb.get("dim", 768)),
            embedding_endpoint=emb.get("endpoint", "http://localhost:11434"),
            generation_model=gen["model"],
            generation_endpoint=gen.get("endpoint", "http://localhost:11434"),
            lancedb_path=REPO_ROOT / sto["lancedb_path"],
            table_name=sto.get("table_name", "aac_chunks"),
            cache_path=REPO_ROOT / sto["cache_path"],
            include=list(raw.get("include", [])),
            exclude=list(raw.get("exclude", [])),
            code_chunk_chars=int(chk.get("code_chunk_chars", 1500)),
            code_overlap_chars=int(chk.get("code_overlap_chars", 200)),
            doc_max_chars=int(chk.get("doc_max_chars", 2000)),
            doc_overlap_chars=int(chk.get("doc_overlap_chars", 150)),
            k_vector=int(ret.get("k_vector", 12)),
            k_bm25=int(ret.get("k_bm25", 8)),
            k_final=int(ret.get("k_final", 8)),
        )

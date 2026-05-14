"""AAC RAG — local Ollama + LanceDB retrieval over the AAC repo.

Public API:
    from shared.aac_rag import query, reindex, stats

Quick CLI:
    python -m shared.aac_rag reindex
    python -m shared.aac_rag query "how does the ibkr connector authenticate?"
    python -m shared.aac_rag stats
"""

from __future__ import annotations

from .query import query, ask
from .indexer import reindex, stats

__all__ = ["query", "ask", "reindex", "stats"]

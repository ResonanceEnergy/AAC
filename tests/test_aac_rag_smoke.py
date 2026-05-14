from __future__ import annotations
import pytest
import structlog
from shared.aac_rag import query, stats

logger = structlog.get_logger()

def test_stats_api():
    """Smoke test to ensure stats() returns a dict/object and doesn't crash."""
    try:
        results = stats()
        assert results is not None
        logger.info("RAG stats available", stats=results)
    except Exception as e:
        if "lancedb" in str(e).lower() or "connection" in str(e).lower():
            pytest.skip(f"LanceDB/Ollama backend not available: {e}")
        raise

def test_query_api_smoke():
    """Smoke test for the query() function."""
    try:
        # Use 'k' instead of 'top_k' based on shared.aac_rag.query signature
        results = query("test search", k=1)
        assert isinstance(results, list)
        logger.info("RAG query functional", results_count=len(results))
    except Exception as e:
        if "lancedb" in str(e).lower() or "connection" in str(e).lower():
            pytest.skip(f"LanceDB/Ollama backend not available: {e}")
        raise

def test_public_api_availability():
    """Verify other expected public functions exist."""
    from shared.aac_rag import reindex, ask
    assert reindex is not None
    assert ask is not None

"""High-level query API: retrieve context and (optionally) ask the LLM."""

from __future__ import annotations

from typing import Any

from .config import RagConfig
from .retriever import Hit, retrieve

SYSTEM_PROMPT = (
    "You are an expert engineer working inside the AAC (Accelerated Arbitrage Corp) "
    "trading platform repo. Answer using ONLY the provided code/doc snippets. "
    "Cite file paths in backticks. If the answer is not in the snippets, say so."
)


def query(
    text: str,
    *,
    k: int | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant chunks for `text`. Returns serializable dicts."""
    return [h.to_dict() for h in retrieve(text, k=k, kind=kind)]


def _format_context(hits: list[Hit], max_chars: int = 8000) -> str:
    parts: list[str] = []
    used = 0
    for h in hits:
        header = f"\n--- `{h.path}` (chunk {h.chunk_idx}, kind={h.kind}, score={h.score}) ---\n"
        block = header + h.text + "\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts)


def ask(
    question: str,
    *,
    k: int | None = None,
    kind: str | None = None,
    max_context_chars: int = 8000,
) -> dict[str, Any]:
    """Retrieve context, run local LLM, return {answer, sources}."""
    import ollama

    cfg = RagConfig.load()
    hits = retrieve(question, k=k, kind=kind)
    if not hits:
        return {
            "answer": "No matching context found in the AAC index. Run `python -m shared.aac_rag reindex` first.",
            "sources": [],
        }

    context = _format_context(hits, max_chars=max_context_chars)
    prompt = (
        f"# Question\n{question}\n\n"
        f"# Relevant AAC snippets\n{context}\n\n"
        "# Answer (cite file paths)"
    )

    client = ollama.Client(host=cfg.generation_endpoint)
    resp = client.chat(
        model=cfg.generation_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},
    )
    answer = resp["message"]["content"]
    return {
        "answer": answer,
        "sources": [h.to_dict() for h in hits],
    }

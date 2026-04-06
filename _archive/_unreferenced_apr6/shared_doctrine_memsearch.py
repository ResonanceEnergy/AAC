"""
Memsearch — Semantic Search Over Doctrine Files
================================================

Provides text-based search across BARREN WUFFET doctrine files
(SOUL.md, AGENTS.md, HEARTBEAT.md, RESEARCH.md) using TF-IDF
similarity without heavy ML dependencies.

Part of the SharedInfrastructure department.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DOCTRINE_DIR = Path(__file__).resolve().parent.parent / "aac" / "doctrine"


@dataclass
class SearchResult:
    """A single search hit."""
    file_name: str
    section: str          # Header or line context
    line_number: int
    snippet: str           # Surrounding text (3 lines)
    score: float           # 0.0 to 1.0


@dataclass
class DoctrineDocument:
    """Parsed doctrine file with sections."""
    path: Path
    name: str
    sections: List[Tuple[str, int, str]]  # (header, line_no, content)
    raw_lines: List[str] = field(default_factory=list)


class DoctrineMemsearch:
    """
    Lightweight semantic/keyword search across doctrine files.

    Usage:
        search = DoctrineMemsearch()
        search.index()
        results = search.query("scam detection FrankenClaw", top_k=5)
    """

    def __init__(self, doctrine_dir: Optional[Path] = None) -> None:
        self.doctrine_dir = doctrine_dir or DOCTRINE_DIR
        self.documents: List[DoctrineDocument] = []
        self._indexed = False
        self._idf: Dict[str, float] = {}
        self._doc_vectors: List[Dict[str, float]] = []

    def index(self) -> int:
        """
        Index all Markdown files in the doctrine directory.

        Returns:
            Number of files indexed.
        """
        self.documents = []
        md_files = sorted(self.doctrine_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"No .md files found in {self.doctrine_dir}")
            return 0

        all_term_doc_counts: Counter = Counter()

        for md_file in md_files:
            doc = self._parse_markdown(md_file)
            self.documents.append(doc)
            # Count terms per document (for IDF)
            doc_terms = set()
            for _, _, content in doc.sections:
                doc_terms.update(self._tokenize(content))
            for term in doc_terms:
                all_term_doc_counts[term] += 1

        # Compute IDF
        n_docs = len(self.documents)
        self._idf = {
            term: math.log((n_docs + 1) / (count + 1)) + 1
            for term, count in all_term_doc_counts.items()
        }

        # Build TF-IDF vectors per section
        self._section_entries = []  # (doc_idx, section_idx, vector)
        for doc_idx, doc in enumerate(self.documents):
            for sec_idx, (header, line_no, content) in enumerate(doc.sections):
                tokens = self._tokenize(content)
                tf = Counter(tokens)
                max_tf = max(tf.values()) if tf else 1
                vec = {
                    term: (count / max_tf) * self._idf.get(term, 1.0)
                    for term, count in tf.items()
                }
                self._section_entries.append((doc_idx, sec_idx, vec))

        self._indexed = True
        logger.info(f"Indexed {len(self.documents)} doctrine files, "
                     f"{len(self._section_entries)} sections")
        return len(self.documents)

    def query(self, query_text: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search doctrine files for the given query.

        Args:
            query_text: Natural language query
            top_k: Max results to return

        Returns:
            List of SearchResult, sorted by relevance score descending.
        """
        if not self._indexed:
            self.index()

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        max_qtf = max(query_tf.values())
        query_vec = {
            term: (count / max_qtf) * self._idf.get(term, 1.0)
            for term, count in query_tf.items()
        }

        scored: List[Tuple[float, int, int]] = []
        for doc_idx, sec_idx, sec_vec in self._section_entries:
            score = self._cosine_sim(query_vec, sec_vec)
            if score > 0:
                scored.append((score, doc_idx, sec_idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc_idx, sec_idx in scored[:top_k]:
            doc = self.documents[doc_idx]
            header, line_no, content = doc.sections[sec_idx]

            # Build snippet (first 200 chars)
            snippet_lines = content.strip().split("\n")[:3]
            snippet = "\n".join(snippet_lines)
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."

            results.append(SearchResult(
                file_name=doc.name,
                section=header,
                line_number=line_no,
                snippet=snippet,
                score=round(score, 4),
            ))

        return results

    def get_all_sections(self) -> List[Dict]:
        """Return all indexed sections as dicts (for inspection)."""
        if not self._indexed:
            self.index()
        out = []
        for doc in self.documents:
            for header, line_no, content in doc.sections:
                out.append({
                    "file": doc.name,
                    "section": header,
                    "line": line_no,
                    "length": len(content),
                })
        return out

    # ── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_markdown(path: Path) -> DoctrineDocument:
        """Parse a Markdown file into sections split by headers."""
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        sections: List[Tuple[str, int, str]] = []
        current_header = path.stem
        current_start = 1
        current_lines: List[str] = []

        for i, line in enumerate(lines, start=1):
            if line.startswith("#"):
                # Save previous section
                if current_lines:
                    sections.append((current_header, current_start, "\n".join(current_lines)))
                current_header = line.lstrip("#").strip()
                current_start = i
                current_lines = []
            else:
                current_lines.append(line)

        # Final section
        if current_lines:
            sections.append((current_header, current_start, "\n".join(current_lines)))

        return DoctrineDocument(
            path=path,
            name=path.name,
            sections=sections,
            raw_lines=lines,
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        # Remove very short tokens and stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "shall", "can",
                      "to", "of", "in", "for", "on", "with", "at", "by", "from",
                      "as", "into", "through", "during", "before", "after", "and",
                      "but", "or", "nor", "not", "so", "yet", "both", "either",
                      "this", "that", "these", "those", "it", "its"}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    @staticmethod
    def _cosine_sim(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

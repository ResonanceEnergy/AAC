"""CLI for the AAC RAG system.

Usage:
    python -m shared.aac_rag reindex [--full] [--quiet]
    python -m shared.aac_rag stats
    python -m shared.aac_rag query "text" [-k N] [--kind code|doc|config]
    python -m shared.aac_rag ask   "question" [-k N] [--kind code|doc|config]
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aac_rag", description="AAC local RAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_reidx = sub.add_parser("reindex", help="Walk repo and embed changed files")
    p_reidx.add_argument("--full", action="store_true", help="Re-embed all files")
    p_reidx.add_argument("--quiet", action="store_true")

    sub.add_parser("stats", help="Show index statistics")

    p_q = sub.add_parser("query", help="Retrieve top-k chunks (no LLM)")
    p_q.add_argument("text")
    p_q.add_argument("-k", type=int, default=None)
    p_q.add_argument("--kind", choices=["code", "doc", "config"], default=None)

    p_a = sub.add_parser("ask", help="Retrieve + ask local LLM")
    p_a.add_argument("question")
    p_a.add_argument("-k", type=int, default=None)
    p_a.add_argument("--kind", choices=["code", "doc", "config"], default=None)

    args = p.parse_args(argv)

    if args.cmd == "reindex":
        from .indexer import reindex

        result = reindex(full=args.full, verbose=not args.quiet)
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "stats":
        from .indexer import stats

        print(json.dumps(stats(), indent=2))
        return 0

    if args.cmd == "query":
        from .query import query

        hits = query(args.text, k=args.k, kind=args.kind)
        for h in hits:
            print(f"\n[{h['score']:.3f}] {h['path']} (chunk {h['chunk_idx']}, {h['kind']})")
            preview = h["text"].strip().replace("\n", " ")[:240]
            print(f"   {preview}…")
        return 0

    if args.cmd == "ask":
        from .query import ask

        result = ask(args.question, k=args.k, kind=args.kind)
        print(result["answer"])
        print("\n— Sources —")
        for h in result["sources"]:
            print(f"  • {h['path']} (chunk {h['chunk_idx']}, score {h['score']})")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

"""Xai Council pipeline -- retrieve, analyze, format X posts."""

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import structlog

from councils.xai.analyzer import analyze_posts_extractive, analyze_posts_with_llm
from councils.xai.formatter import format_to_markdown
from councils.xai.models import CouncilEntry
from councils.xai.retriever import get_user_posts_via_grok, search_x_via_grok

_log = structlog.get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_xai_council(
    user: str | None = None,
    search_query: str | None = None,
    provider: str = "xai",
    max_posts: int = 30,
) -> CouncilEntry | None:
    """Run full Xai Council pipeline."""
    source = f"@{user}" if user else search_query or "unknown"
    _log.info("xai_council_starting", source=source, provider=provider)

    # 1. Retrieve posts
    if user:
        posts = get_user_posts_via_grok(user, provider=provider, max_posts=max_posts)
    elif search_query:
        posts = search_x_via_grok(search_query, provider=provider, max_posts=max_posts)
    else:
        _log.warning("no_source_specified")
        return None

    if not posts:
        _log.warning("no_posts_found", source=source)
        return None

    # 2. Analyze
    insights = analyze_posts_with_llm(posts, source, provider=provider)
    if not insights:
        insights = analyze_posts_extractive(posts, source)

    # 3. Format to Markdown
    md_content = format_to_markdown(posts, insights)

    # 4. Save
    safe_source = re.sub(r'[^\w\s-]', '', source)[:40].strip().replace(' ', '_')
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = f"{date_str}_{safe_source}.md"
    md_path = OUTPUT_DIR / filename
    md_path.write_text(md_content, encoding="utf-8")

    # Save raw data JSON
    json_path = OUTPUT_DIR / filename.replace(".md", "_data.json")
    json_data = {
        "source": source,
        "provider": provider,
        "post_count": len(posts),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "insights": asdict(insights),
        "posts": [asdict(p) for p in posts],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    _log.info("xai_council_complete", posts=len(posts), md=str(md_path))

    return CouncilEntry(
        posts=posts,
        insights=insights,
        markdown_path=str(md_path),
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Xai Council -- X/Twitter Analysis via Grok")
    parser.add_argument("--user", type=str, help="X username to analyze (without @)")
    parser.add_argument("--search", type=str, help="Search query for X posts")
    parser.add_argument("--file", type=str, help="File with post URLs (one per line)")
    parser.add_argument("--provider", choices=["xai", "openai"], default="xai", help="LLM provider (default: xai)")
    parser.add_argument("--max-posts", type=int, default=30, help="Max posts to analyze (default: 30)")
    args = parser.parse_args()

    if not any([args.user, args.search, args.file]):
        print("Error: provide --user, --search, or --file")
        sys.exit(1)

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        urls = filepath.read_text(encoding="utf-8").strip().splitlines()
        query = " OR ".join(urls[:10])
        result = run_xai_council(search_query=query, provider=args.provider, max_posts=args.max_posts)
    else:
        result = run_xai_council(
            user=args.user,
            search_query=args.search,
            provider=args.provider,
            max_posts=args.max_posts,
        )

    if result:
        print(f"\nAnalyzed {len(result.posts)} posts. Output: {result.markdown_path}")
    else:
        print("\nNo results. Check API key and connectivity.")


if __name__ == "__main__":
    main()

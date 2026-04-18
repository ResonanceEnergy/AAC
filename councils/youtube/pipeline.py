from __future__ import annotations

"""YouTube Council pipeline — process videos end-to-end."""

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import structlog

from councils.youtube.analyzer import analyze_transcript, analyze_with_llm
from councils.youtube.formatter import format_to_markdown
from councils.youtube.models import CouncilEntry, VideoMeta
from councils.youtube.scraper import download_audio, get_youtube_subtitles, list_channel_videos, transcribe_with_whisper

_log = structlog.get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_video(
    video: VideoMeta,
    use_whisper: bool = False,
    use_llm: str | None = None,
) -> CouncilEntry | None:
    """Full pipeline for a single video: transcribe -> analyze -> format -> save."""
    _log.info("processing_video", title=video.title, video_id=video.video_id)

    # 1. Transcribe
    if use_whisper:
        audio_dir = OUTPUT_DIR / "audio"
        audio_dir.mkdir(exist_ok=True)
        audio_path = download_audio(video.url, audio_dir)
        if not audio_path:
            _log.warning("audio_download_failed", video_id=video.video_id)
            return None
        segments = transcribe_with_whisper(audio_path)
    else:
        segments = get_youtube_subtitles(video.url)

    if not segments:
        _log.warning("no_transcript", video_id=video.video_id)
        return None

    # 2. Analyze
    if use_llm:
        full_text = " ".join(s.text for s in segments)
        insights = analyze_with_llm(full_text, video, provider=use_llm)
        if not insights:
            insights = analyze_transcript(segments, video)
    else:
        insights = analyze_transcript(segments, video)

    # 3. Format to Markdown
    md_content = format_to_markdown(video, segments, insights)

    # 4. Save
    safe_title = re.sub(r'[^\w\s-]', '', video.title)[:60].strip().replace(' ', '_')
    date_str = video.upload_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"{date_str}_{safe_title}.md"
    md_path = OUTPUT_DIR / filename

    md_path.write_text(md_content, encoding="utf-8")
    _log.info("saved_markdown", path=str(md_path))

    # 5. Save insights JSON
    json_path = OUTPUT_DIR / filename.replace(".md", "_insights.json")
    json_path.write_text(
        json.dumps(asdict(insights), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return CouncilEntry(
        meta=video,
        transcript=segments,
        insights=insights,
        markdown_path=str(md_path),
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


def run_youtube_council(
    channel_url: str,
    limit: int = 20,
    max_hours: float = 24.0,
    use_whisper: bool = False,
    use_llm: str | None = None,
) -> list[CouncilEntry]:
    """Run full YouTube Council pipeline on a channel/playlist."""
    _log.info(
        "youtube_council_starting",
        channel=channel_url,
        limit=limit,
        max_hours=max_hours,
        whisper=use_whisper,
        llm=use_llm,
    )

    videos = list_channel_videos(channel_url, limit=limit, max_duration_hours=max_hours)
    if not videos:
        _log.warning("no_videos_found")
        return []

    entries: list[CouncilEntry] = []
    for video in videos:
        entry = process_video(video, use_whisper=use_whisper, use_llm=use_llm)
        if entry:
            entries.append(entry)

    # Save session summary
    summary_path = OUTPUT_DIR / f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    summary = {
        "channel": channel_url,
        "videos_processed": len(entries),
        "total_videos_found": len(videos),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "videos": [
            {
                "title": e.meta.title,
                "video_id": e.meta.video_id,
                "duration_min": round(e.meta.duration_seconds / 60, 1),
                "sentiment": e.insights.sentiment,
                "key_topics": e.insights.key_topics[:5],
                "md_file": e.markdown_path,
            }
            for e in entries
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _log.info("youtube_council_complete", processed=len(entries), summary=str(summary_path))

    return entries


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube Council -- Scrape, Transcribe, Analyze")
    parser.add_argument("--channel", type=str, help="YouTube channel URL")
    parser.add_argument("--url", type=str, help="YouTube playlist or video URL")
    parser.add_argument("--limit", type=int, default=20, help="Max videos to process (default: 20)")
    parser.add_argument("--max-hours", type=float, default=24.0, help="Max total hours (default: 24)")
    parser.add_argument("--whisper", action="store_true", help="Use local Whisper instead of YouTube subs")
    parser.add_argument("--llm", choices=["xai", "openai"], default=None, help="Use LLM for deeper analysis")
    args = parser.parse_args()

    target = args.channel or args.url
    if not target:
        print("Error: provide --channel or --url")
        sys.exit(1)

    entries = run_youtube_council(
        channel_url=target,
        limit=args.limit,
        max_hours=args.max_hours,
        use_whisper=args.whisper,
        use_llm=args.llm,
    )
    print(f"\nProcessed {len(entries)} videos. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

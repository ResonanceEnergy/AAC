from __future__ import annotations

"""YouTube scraper — channel listing, subtitle extraction, audio download."""

import json
import logging
from pathlib import Path

import structlog

from councils.youtube.models import TranscriptSegment, VideoMeta

_log = structlog.get_logger(__name__)

# Silence yt-dlp noise
logging.getLogger("yt_dlp").setLevel(logging.ERROR)


def list_channel_videos(
    channel_url: str,
    limit: int = 50,
    max_duration_hours: float = 24.0,
) -> list[VideoMeta]:
    """List videos from a channel/playlist, respecting duration cap."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        _log.error("yt-dlp not installed. Run: pip install yt-dlp")
        return []

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": limit * 2,
    }

    _log.info("scraping_channel", url=channel_url, limit=limit)
    videos: list[VideoMeta] = []

    url = channel_url.rstrip("/")
    if "/@" in url and "/videos" not in url and "/playlist" not in url:
        url = url + "/videos"

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        entries = list(info.get("entries", [])) if info else []
        channel_name = info.get("channel", info.get("uploader", "Unknown"))

        for entry in entries:
            if not entry:
                continue
            if entry.get("_type") == "playlist":
                continue
            vid = VideoMeta(
                video_id=entry.get("id", ""),
                title=entry.get("title", "Untitled"),
                channel=channel_name,
                upload_date=entry.get("upload_date", ""),
                duration_seconds=entry.get("duration") or 0,
                description=entry.get("description", ""),
                url=f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                view_count=entry.get("view_count") or 0,
                like_count=entry.get("like_count") or 0,
                tags=entry.get("tags") or [],
            )
            videos.append(vid)

    videos.sort(key=lambda v: v.upload_date, reverse=True)

    max_seconds = max_duration_hours * 3600
    selected: list[VideoMeta] = []
    total = 0
    for v in videos:
        dur = v.duration_seconds if v.duration_seconds > 0 else 600
        if total + dur <= max_seconds and len(selected) < limit:
            selected.append(v)
            total += dur

    _log.info("videos_selected", count=len(selected), total_hours=round(total / 3600, 1))
    return selected


def get_youtube_subtitles(video_url: str) -> list[TranscriptSegment]:
    """Extract subtitles/auto-captions from YouTube via yt-dlp. No download needed."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return []

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3",
    }

    segments: list[TranscriptSegment] = []

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        subs = info.get("subtitles", {}) or {}
        auto_subs = info.get("automatic_captions", {}) or {}

        sub_data = subs.get("en") or auto_subs.get("en") or []

        json3_url = None
        for fmt in sub_data:
            if fmt.get("ext") == "json3":
                json3_url = fmt.get("url")
                break

        if not json3_url:
            for fmt in sub_data:
                if fmt.get("ext") == "vtt":
                    json3_url = fmt.get("url")
                    break

        if json3_url:
            import urllib.request

            req = urllib.request.Request(json3_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = json.loads(resp.read().decode("utf-8"))

            events = raw.get("events", [])
            for event in events:
                if "segs" not in event:
                    continue
                text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
                if text:
                    start_ms = event.get("tStartMs", 0)
                    dur_ms = event.get("dDurationMs", 0)
                    segments.append(TranscriptSegment(
                        start=start_ms / 1000.0,
                        end=(start_ms + dur_ms) / 1000.0,
                        text=text,
                    ))

    _log.info("subtitles_extracted", segments=len(segments), source="youtube_auto")
    return segments


def transcribe_with_whisper(audio_path: str) -> list[TranscriptSegment]:
    """Transcribe audio file using faster-whisper (local, GPU-accelerated)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        _log.error("faster-whisper not installed. Run: pip install faster-whisper")
        return []

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"

    _log.info("whisper_loading", device=device)
    model = WhisperModel("base", device=device, compute_type=compute)

    raw_segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=False)
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in raw_segments
        if s.text.strip()
    ]
    _log.info("whisper_done", segments=len(segments))
    return segments


def download_audio(video_url: str, out_dir: Path) -> str | None:
    """Download audio only (mp3) for Whisper transcription."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return None

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        vid_id = info.get("id", "unknown")

    mp3_path = out_dir / f"{vid_id}.mp3"
    if mp3_path.exists():
        return str(mp3_path)

    for f in out_dir.iterdir():
        if f.stem == vid_id and f.suffix in (".mp3", ".m4a", ".opus", ".webm"):
            return str(f)
    return None

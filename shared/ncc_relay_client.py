#!/usr/bin/env python3
from __future__ import annotations

"""
NCC Relay Client — AAC → NCC Event Transport
=============================================
Publishes events from AAC (BANK pillar) to the NCC Relay Server (port 8787).
Events are queued locally in NDJSON outbox files when the relay is unreachable,
and flushed automatically when connectivity is restored.

Event envelope:
    {
        "event_type": "ncl.sync.v1.bank.<event_class>",
        "timestamp": "ISO-8601",
        "source": "aac",
        "pillar": "BANK",
        "data": { ... }
    }

Usage:
    from shared.ncc_relay_client import get_relay_client

    client = get_relay_client()
    client.publish("ncl.sync.v1.bank.heartbeat", {"status": "online"})
    client.flush()
"""

import json
import logging
import os
import threading
import time
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RELAY_URL = os.environ.get("NCC_RELAY_URL", "http://127.0.0.1:8787")
SOURCE = "aac"
PILLAR = "BANK"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTBOX_DIR = Path(
    os.environ.get(
        "AAC_NCC_OUTBOX_DIR",
        str(PROJECT_ROOT / "data" / "ncc_outbox"),
    )
)


class NCCRelayClient:
    """Publish events to NCC Relay with local outbox fallback."""

    def __init__(self, relay_url: str = RELAY_URL, outbox_dir: Path | None = None):
        self.relay_url = relay_url.rstrip("/")
        self.outbox_dir = outbox_dir or OUTBOX_DIR
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._published_count = 0
        self._queued_count = 0
        self._last_relay_ok: float | None = None

    # ── Event Schema ────────────────────────────────────────────

    REQUIRED_EVENT_PREFIX = "ncl.sync.v1."

    # ── Event Construction ──────────────────────────────────────

    def _make_event(self, event_type: str, data: dict) -> dict:
        return {
            "event_type": event_type,
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": SOURCE,
            "pillar": PILLAR,
            "data": data,
        }

    @staticmethod
    def _validate_event(event: dict) -> None:
        """Validate event envelope matches ncl.sync.v1 schema.

        Raises ValueError if event_type is missing or data is not a dict.
        """
        required = ("event_type", "timestamp", "source", "pillar", "data", "correlation_id")
        missing = [k for k in required if k not in event]
        if missing:
            raise ValueError(f"Event envelope missing fields: {missing}")
        if not isinstance(event.get("data"), dict):
            raise ValueError("Event 'data' must be a dict")

    # ── Transport ───────────────────────────────────────────────

    def _post(self, event: dict) -> bool:
        try:
            body = json.dumps(event).encode("utf-8")
            req = urllib.request.Request(
                f"{self.relay_url}/event",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                if resp.status in (200, 202):
                    self._last_relay_ok = time.monotonic()
                    return True
            return False
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _queue(self, event: dict) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        outbox_file = self.outbox_dir / f"{today}.ndjson"
        with self._lock:
            with outbox_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        self._queued_count += 1

    # ── Public API ──────────────────────────────────────────────

    def publish(self, event_type: str, data: dict) -> bool:
        """Publish an event to the relay. Queues locally on failure."""
        event = self._make_event(event_type, data)
        self._validate_event(event)
        if self._post(event):
            self._published_count += 1
            logger.debug("Published %s to relay", event_type)
            return True
        self._queue(event)
        logger.debug("Queued %s (relay unreachable)", event_type)
        return False

    def flush(self) -> dict:
        """Retry all queued outbox events. Returns {sent, failed}."""
        sent, failed = 0, 0
        with self._lock:
            for outbox_file in sorted(self.outbox_dir.glob("*.ndjson")):
                lines = outbox_file.read_text(encoding="utf-8").strip().splitlines()
                if not lines:
                    outbox_file.unlink(missing_ok=True)
                    continue
                remaining = []
                for line in lines:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if self._post(event):
                        sent += 1
                    else:
                        remaining.append(line)
                        failed += 1
                if remaining:
                    outbox_file.write_text(
                        "\n".join(remaining) + "\n", encoding="utf-8"
                    )
                else:
                    outbox_file.unlink(missing_ok=True)
        return {"sent": sent, "failed": failed}

    def relay_health(self) -> dict | None:
        """Check relay /health endpoint."""
        try:
            req = urllib.request.Request(
                f"{self.relay_url}/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    return json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return None

    def outbox_depth(self) -> int:
        """Count of events waiting in the outbox."""
        total = 0
        for f in self.outbox_dir.glob("*.ndjson"):
            total += sum(
                1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        return total

    @property
    def stats(self) -> dict:
        return {
            "relay_url": self.relay_url,
            "published": self._published_count,
            "queued": self._queued_count,
            "outbox_depth": self.outbox_depth(),
            "relay_reachable": self._last_relay_ok is not None
            and (time.monotonic() - self._last_relay_ok) < 120,
        }


# ── Module-level singleton ──────────────────────────────────────

_client: NCCRelayClient | None = None
_client_lock = threading.Lock()


def get_relay_client() -> NCCRelayClient:
    """Get or create the module-level relay client singleton."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = NCCRelayClient()
    return _client

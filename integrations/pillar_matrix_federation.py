"""
Pillar Matrix Federation — Deep Matrix Monitor Data Aggregator
==============================================================
Fetches DEEP matrix monitor data from all 4 pillars + NCC MASTER,
standardises into a unified view, and exposes for dashboard display.

Pillar matrix sources:
    NCC MASTER (:9000)  /matrix/sitrep    → enterprise SITREP
    NCC (:8765)         /ncc/matrix-monitor → governance matrix
    AAC (:8080)         self               → local integrator status
    NCL (:8787)         /health + file     → MatrixReport (checks, SLOs, tiles)
    BRS/DL (:8000)      /matrix/sitrep     → C2 fleet + NERVE status

Usage:
    fed = PillarMatrixFederation()
    deep = await fed.collect_all()
    # deep["pillars"]["NCL"]["matrix_checks_passed"] == 15
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pillar_matrix_federation")

# ── Defaults ──────────────────────────────────────────────────
NCC_MASTER_URL = os.environ.get("NCC_MASTER_URL", "http://localhost:9000")
NCC_COMMAND_URL = os.environ.get("NCC_COMMAND_URL", "http://127.0.0.1:8765")
NCL_RELAY_URL = os.environ.get("NCC_RELAY_URL", "http://127.0.0.1:8787")
BRS_URL = os.environ.get("BRS_URL", "http://localhost:8000")
HTTP_TIMEOUT = 5  # seconds
CACHE_TTL = 25     # seconds


@dataclass
class PillarMatrixSnapshot:
    """Standardised per-pillar matrix data."""
    pillar_id: str
    name: str
    role: str
    health: str = "UNKNOWN"       # GREEN / YELLOW / RED / BLACK
    matrix_status: str = "UNKNOWN"  # ACTIVE / RESPONDING / NO_MATRIX / OFFLINE
    overall_score: float = 0.0
    health_status: str = "UNKNOWN"  # EXCELLENT / GOOD / FAIR / DEGRADED / CRITICAL
    checks_passed: int = 0
    checks_total: int = 0
    slo_violations: int = 0
    active_alerts: int = 0
    latency_ms: Optional[float] = None
    agents_online: int = 0
    version: str = "unknown"
    uptime_s: float = 0.0
    enterprise_score: int = 0
    doctrine_mode: str = "UNKNOWN"
    raw_detail: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pillar_id": self.pillar_id,
            "name": self.name,
            "role": self.role,
            "health": self.health,
            "matrix_status": self.matrix_status,
            "overall_score": self.overall_score,
            "health_status": self.health_status,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "slo_violations": self.slo_violations,
            "active_alerts": self.active_alerts,
            "latency_ms": self.latency_ms,
            "agents_online": self.agents_online,
            "version": self.version,
            "uptime_s": self.uptime_s,
            "enterprise_score": self.enterprise_score,
            "doctrine_mode": self.doctrine_mode,
            "error": self.error,
            "collected_at": self.collected_at,
        }


def _http_get_json(url: str, timeout: int = HTTP_TIMEOUT) -> tuple[Optional[Dict], float, Optional[str]]:
    """Fetch JSON from a URL. Returns (data, latency_ms, error)."""
    try:
        t0 = time.monotonic()
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = (time.monotonic() - t0) * 1000
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body), latency, None
            except json.JSONDecodeError:
                return None, latency, None  # responded but not JSON
    except urllib.error.URLError as exc:
        return None, 0.0, str(getattr(exc, "reason", exc))[:120]
    except Exception as exc:
        return None, 0.0, str(exc)[:120]


class PillarMatrixFederation:
    """Fetches deep matrix monitor data from every pillar and normalises it."""

    def __init__(self) -> None:
        self._cache: Dict[str, PillarMatrixSnapshot] = {}
        self._cache_ts: float = 0.0
        self._ncl_data_path = Path(
            os.environ.get("NCL_DATA_PATH", r"C:\dev\NCL")
        )

    # ── Public API ────────────────────────────────────────────

    async def collect_all(self) -> Dict[str, Any]:
        """Collect deep matrix data from all pillars. Returns unified dict."""
        now = time.time()
        if now - self._cache_ts < CACHE_TTL and self._cache:
            return self._build_result()

        loop = asyncio.get_event_loop()
        # Run HTTP fetches concurrently via threads (urllib is blocking)
        snapshots = await asyncio.gather(
            loop.run_in_executor(None, self._fetch_ncc_master),
            loop.run_in_executor(None, self._fetch_ncc),
            loop.run_in_executor(None, self._fetch_aac),
            loop.run_in_executor(None, self._fetch_ncl),
            loop.run_in_executor(None, self._fetch_brs),
            return_exceptions=True,
        )

        for snap in snapshots:
            if isinstance(snap, PillarMatrixSnapshot):
                self._cache[snap.pillar_id] = snap
            elif isinstance(snap, Exception):
                logger.warning("Pillar fetch exception: %s", snap)

        self._cache_ts = now
        return self._build_result()

    # ── Per-Pillar Fetchers ───────────────────────────────────

    def _fetch_ncc_master(self) -> PillarMatrixSnapshot:
        """NCC MASTER — Supreme Orchestrator on :9000."""
        snap = PillarMatrixSnapshot(
            pillar_id="NCC_MASTER", name="NCC MASTER C2", role="Supreme Orchestrator"
        )
        data, lat, err = _http_get_json(f"{NCC_MASTER_URL}/matrix/sitrep")
        if err:
            snap.health = "RED"
            snap.matrix_status = "OFFLINE"
            snap.error = err
            return snap

        snap.latency_ms = round(lat, 1)
        snap.health = "GREEN"
        snap.matrix_status = "ACTIVE"

        if data:
            snap.raw_detail = data
            snap.doctrine_mode = data.get("doctrine_mode", "UNKNOWN")
            snap.enterprise_score = data.get("enterprise_score", 0)
            # Parse pillar health summary from SITREP
            pillar_healths = data.get("pillar_health", {})
            online = sum(1 for h in pillar_healths.values() if h in ("GREEN", "YELLOW"))
            total = len(pillar_healths)
            snap.checks_passed = online
            snap.checks_total = total
            snap.overall_score = online / total if total > 0 else 0
            snap.health_status = "GOOD" if online == total else "DEGRADED"
            snap.agents_online = data.get("agent_count", 0)
            snap.version = data.get("version", "unknown")
        return snap

    def _fetch_ncc(self) -> PillarMatrixSnapshot:
        """NCC HUB — Governance & Command on :8765."""
        snap = PillarMatrixSnapshot(
            pillar_id="NCC", name="NCC (HUB)", role="Governance & Command"
        )
        data, lat, err = _http_get_json(
            f"{NCC_COMMAND_URL}/ncc/matrix-monitor"
        )
        if err:
            # Try health endpoint as fallback
            h_data, h_lat, h_err = _http_get_json(f"{NCC_COMMAND_URL}/health")
            if h_err:
                snap.health = "RED"
                snap.matrix_status = "OFFLINE"
                snap.error = err
                return snap
            snap.health = "GREEN"
            snap.matrix_status = "NO_MATRIX"
            snap.latency_ms = round(h_lat, 1)
            return snap

        snap.latency_ms = round(lat, 1)
        snap.health = "GREEN"
        snap.matrix_status = "ACTIVE"

        if data:
            snap.raw_detail = data
            snap.doctrine_mode = data.get("doctrine_mode", "UNKNOWN")
            snap.overall_score = data.get("overall_score", 0)
            snap.health_status = data.get("health_status", "UNKNOWN")
            snap.checks_passed = data.get("checks_passed", 0)
            snap.checks_total = data.get("checks_total", 0)
            snap.slo_violations = data.get("slos_in_violation", 0)
            snap.active_alerts = len(data.get("alerts", []))
            snap.uptime_s = data.get("uptime_s", 0)
        return snap

    def _fetch_aac(self) -> PillarMatrixSnapshot:
        """AAC BANK — ourselves. Pull from local integrator."""
        snap = PillarMatrixSnapshot(
            pillar_id="AAC", name="AAC (BANK)", role="Trading & Capital"
        )
        snap.health = "GREEN"
        snap.matrix_status = "ACTIVE"
        snap.latency_ms = 0

        try:
            from core.unified_component_integrator import get_unified_integrator
            integrator = get_unified_integrator(paper_mode=True)
            s = integrator.status
            snap.checks_passed = s.components_wired
            snap.checks_total = s.components_wired + s.components_failed
            snap.overall_score = (
                s.components_wired / snap.checks_total
                if snap.checks_total > 0 else 1.0
            )
            snap.health_status = "EXCELLENT" if s.components_failed == 0 else "DEGRADED"
            snap.error = "; ".join(s.errors) if s.errors else None
        except Exception as exc:
            snap.error = str(exc)[:120]
            snap.health_status = "UNKNOWN"

        return snap

    def _fetch_ncl(self) -> PillarMatrixSnapshot:
        """NCL BRAIN — Cognitive engine on :8787. Reads matrix_latest.json if available."""
        snap = PillarMatrixSnapshot(
            pillar_id="NCL", name="NCL (BRAIN)", role="Cognitive Augmentation"
        )

        # Try health endpoint first
        data, lat, err = _http_get_json(f"{NCL_RELAY_URL}/health")
        if err:
            snap.health = "RED"
            snap.matrix_status = "OFFLINE"
            snap.error = err
        else:
            snap.health = "GREEN"
            snap.latency_ms = round(lat, 1)
            if data:
                snap.raw_detail["health"] = data

        # Try reading the NCL MatrixReport from file (richer data)
        matrix_file = (
            self._ncl_data_path / "ncl_agency_runtime" / "logs" / "matrix_latest.json"
        )
        if matrix_file.exists():
            try:
                report = json.loads(matrix_file.read_text(encoding="utf-8"))
                snap.matrix_status = "ACTIVE"
                snap.overall_score = report.get("overall_score", 0)
                snap.health_status = report.get("health_status", "UNKNOWN")
                snap.checks_passed = report.get("checks_passed", 0)
                snap.checks_total = report.get("checks_total", 0)
                snap.slo_violations = report.get("slos_in_violation", 0)
                snap.active_alerts = len(report.get("alerts", []))
                snap.uptime_s = report.get("uptime_s", 0)
                snap.raw_detail["matrix_report"] = report
            except Exception as exc:
                logger.debug("NCL matrix_latest.json read failed: %s", exc)
                snap.matrix_status = "NO_MATRIX" if snap.health == "GREEN" else "OFFLINE"
        elif snap.health == "GREEN":
            snap.matrix_status = "NO_MATRIX"

        return snap

    def _fetch_brs(self) -> PillarMatrixSnapshot:
        """BRS/DL AGENCY — Digital Labour on :8000."""
        snap = PillarMatrixSnapshot(
            pillar_id="BRS", name="BRS/DL (AGENCY)", role="Digital Labour"
        )
        data, lat, err = _http_get_json(f"{BRS_URL}/matrix/sitrep")
        if err:
            # Try overview endpoint
            data, lat, err2 = _http_get_json(f"{BRS_URL}/monitor/overview")
            if err2:
                snap.health = "RED"
                snap.matrix_status = "OFFLINE"
                snap.error = err
                return snap
            snap.health = "GREEN"
            snap.matrix_status = "NO_MATRIX"
            snap.latency_ms = round(lat, 1)
            if data:
                snap.raw_detail = data
            return snap

        snap.latency_ms = round(lat, 1)
        snap.health = "GREEN"
        snap.matrix_status = "ACTIVE"

        if data:
            snap.raw_detail = data
            snap.agents_online = data.get("agents_online", 0)
            snap.version = data.get("version", "unknown")
            # BRS reports different fields
            snap.overall_score = data.get("health_score", 0) / 100 if data.get("health_score") else 0
            snap.health_status = data.get("health_status", "UNKNOWN")
            snap.enterprise_score = data.get("enterprise_score", 0)
        return snap

    # ── Result Builder ────────────────────────────────────────

    def _build_result(self) -> Dict[str, Any]:
        """Build the unified result dictionary."""
        pillar_order = ["NCC_MASTER", "NCC", "AAC", "NCL", "BRS"]
        pillars: Dict[str, Dict] = {}
        online = 0
        total = 0

        for pid in pillar_order:
            snap = self._cache.get(pid)
            if snap:
                pillars[pid] = snap.to_dict()
                total += 1
                if snap.health in ("GREEN", "YELLOW"):
                    online += 1

        # Aggregate enterprise score
        scores = [s.overall_score for s in self._cache.values() if s.overall_score > 0]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "status": "ok",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "pillars_online": online,
            "pillars_total": total,
            "enterprise_score": round(avg_score * 100),
            "pillars": pillars,
        }


# ── Singleton ─────────────────────────────────────────────────

_federation: Optional[PillarMatrixFederation] = None


def get_pillar_federation() -> PillarMatrixFederation:
    """Get or create the singleton PillarMatrixFederation instance."""
    global _federation
    if _federation is None:
        _federation = PillarMatrixFederation()
    return _federation

"""
Rocket Ship — Geo Plan Engine
================================
Tracks base-of-operations tasks for the physical relocation /
multi-jurisdiction strategy across the Rocket Ship timeline.

Locations:
    Panama    (primary — Moons 1-6 setup, Friendly Nations visa)
    Paraguay  (secondary — Moons 4-9, 0% territorial tax, SUACE)
    UAE       (phase 2 — Moons 10-18, 0% tax, Golden Visa, mBridge hub)
    Canada    (current — wind-down tasks)

Each task has:
    - A target moon range (start_moon – end_moon)
    - A geographic base
    - Priority (CRITICAL / HIGH / MEDIUM)
    - Completion status (persisted to JSON)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from strategies.rocket_ship.core import GeoBase

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GEO TASK
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeoTask:
    """One actionable task in the geo plan."""

    id: str                     # e.g. "PA-01"
    title: str
    base: GeoBase
    description: str
    start_moon: int             # First moon this task is relevant
    end_moon: int               # Last moon to complete by
    priority: str               # "CRITICAL" | "HIGH" | "MEDIUM"
    completed: bool = False
    completed_date: Optional[date] = None
    notes: str = ""

    def is_due(self, moon_number: int) -> bool:
        """Is this task relevant for the given moon?"""
        return self.start_moon <= moon_number <= self.end_moon

    def is_overdue(self, moon_number: int) -> bool:
        """Has the deadline passed without completion?"""
        return not self.completed and moon_number > self.end_moon


# ═══════════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

_TASKS: List[GeoTask] = [
    # ── Canada Wind-Down ─────────────────────────────────────────────────
    GeoTask(
        id="CA-01", title="Consolidate Canadian accounts",
        base=GeoBase.PANAMA,  # Prep for Panama
        description="Close unused bank accounts, consolidate to primary bank + IBKR.",
        start_moon=1, end_moon=4, priority="HIGH",
    ),
    GeoTask(
        id="CA-02", title="Tax residency exit strategy",
        base=GeoBase.PANAMA,
        description="Consult cross-border tax advisor. Document departure date and ties.",
        start_moon=1, end_moon=6, priority="CRITICAL",
    ),
    GeoTask(
        id="CA-03", title="Sell / store physical assets",
        base=GeoBase.PANAMA,
        description="Downsize belongings. Ship essentials. Storage unit for keepers.",
        start_moon=2, end_moon=6, priority="MEDIUM",
    ),

    # ── Panama (Primary) ─────────────────────────────────────────────────
    GeoTask(
        id="PA-01", title="Friendly Nations visa application",
        base=GeoBase.PANAMA,
        description=(
            "Apply for Panama Friendly Nations visa (Canada qualifies). "
            "Requires: clean criminal record, bank reference, $5K deposit, "
            "Panama bank account, health cert. Lawyer recommended."
        ),
        start_moon=1, end_moon=4, priority="CRITICAL",
    ),
    GeoTask(
        id="PA-02", title="Open Panama bank account",
        base=GeoBase.PANAMA,
        description="Required for visa. Banesco, Banco General, or Multibank.",
        start_moon=1, end_moon=3, priority="CRITICAL",
    ),
    GeoTask(
        id="PA-03", title="Secure initial housing (Airbnb → rental)",
        base=GeoBase.PANAMA,
        description="1-month Airbnb in Panama City → scout neighborhoods → lease.",
        start_moon=2, end_moon=5, priority="HIGH",
    ),
    GeoTask(
        id="PA-04", title="Set up local operations",
        base=GeoBase.PANAMA,
        description="Internet, coworking space, local SIM, backup power equipment.",
        start_moon=3, end_moon=6, priority="HIGH",
    ),
    GeoTask(
        id="PA-05", title="Panama corporation formation",
        base=GeoBase.PANAMA,
        description=(
            "Form Sociedad Anónima (S.A.) for trading operations. "
            "Territorial tax = 0% on foreign-sourced income."
        ),
        start_moon=3, end_moon=8, priority="HIGH",
    ),

    # ── Paraguay (Secondary) ─────────────────────────────────────────────
    GeoTask(
        id="PY-01", title="Research Paraguay residency (SUACE)",
        base=GeoBase.PARAGUAY,
        description=(
            "SUACE fast-track residency (2 weeks). Requirements: passport, "
            "criminal record, birth certificate, $5K bank deposit."
        ),
        start_moon=4, end_moon=7, priority="HIGH",
    ),
    GeoTask(
        id="PY-02", title="Open Paraguay bank account",
        base=GeoBase.PARAGUAY,
        description="Banco Continental or Itaú Paraguay. USD account preferred.",
        start_moon=5, end_moon=8, priority="MEDIUM",
    ),
    GeoTask(
        id="PY-03", title="Scout Ciudad del Este / Asunción",
        base=GeoBase.PARAGUAY,
        description="Visit trip: cost of living, crypto community, infrastructure.",
        start_moon=6, end_moon=9, priority="MEDIUM",
    ),

    # ── UAE (Phase 2) ────────────────────────────────────────────────────
    GeoTask(
        id="UAE-01", title="Research UAE freezone options",
        base=GeoBase.UAE,
        description=(
            "DMCC, DIFC, ADGM — compare setup costs, banking, crypto licensing. "
            "Golden Visa via $550K investment or company formation."
        ),
        start_moon=8, end_moon=14, priority="HIGH",
    ),
    GeoTask(
        id="UAE-02", title="UAE freezone company formation",
        base=GeoBase.UAE,
        description="Register trading company in chosen freezone. 0% corporate tax.",
        start_moon=12, end_moon=18, priority="HIGH",
    ),
    GeoTask(
        id="UAE-03", title="Apply for Golden Visa",
        base=GeoBase.UAE,
        description="10-year Golden Visa via investor / entrepreneur route.",
        start_moon=13, end_moon=20, priority="MEDIUM",
    ),
    GeoTask(
        id="UAE-04", title="Open UAE bank + crypto exchange accounts",
        base=GeoBase.UAE,
        description="Emirates NBD or Mashreq Bank. BitOasis / Rain for local crypto.",
        start_moon=13, end_moon=18, priority="HIGH",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# GEO PLAN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

_STATE_FILE = Path("data/geo_plan_state.json")


class GeoPlanEngine:
    """Manages geo relocation tasks and their completion state."""

    ALL_TASKS: List[GeoTask] = []

    def __init__(self) -> None:
        # Deep copy so each engine instance has its own mutable tasks
        self.ALL_TASKS = [
            GeoTask(
                id=t.id, title=t.title, base=t.base,
                description=t.description, start_moon=t.start_moon,
                end_moon=t.end_moon, priority=t.priority,
                completed=t.completed, completed_date=t.completed_date,
                notes=t.notes,
            )
            for t in _TASKS
        ]
        self._task_map: Dict[str, GeoTask] = {t.id: t for t in self.ALL_TASKS}

    # ── Queries ──────────────────────────────────────────────────────────

    def complete_tasks(self) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if t.completed]

    def incomplete_tasks(self) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if not t.completed]

    def tasks_for_moon(self, moon_number: int) -> List[GeoTask]:
        """Tasks that are due (active window) for this moon and not yet done."""
        return [
            t for t in self.ALL_TASKS
            if t.is_due(moon_number) and not t.completed
        ]

    def overdue_tasks(self, moon_number: int) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if t.is_overdue(moon_number)]

    def tasks_by_base(self, base: GeoBase) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if t.base == base]

    # ── Mutations ────────────────────────────────────────────────────────

    def mark_complete(self, task_id: str, notes: str = "") -> bool:
        """Mark a task as completed. Returns False if task_id not found."""
        task = self._task_map.get(task_id)
        if task is None:
            logger.warning("Unknown task ID: %s", task_id)
            return False
        task.completed = True
        task.completed_date = date.today()
        if notes:
            task.notes = notes
        logger.info("Geo task %s completed: %s", task_id, task.title)
        return True

    # ── Persistence ──────────────────────────────────────────────────────

    def save_state(self, path: Path | None = None) -> None:
        out = path or _STATE_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at": date.today().isoformat(),
            "tasks": [
                {
                    "id": t.id,
                    "completed": t.completed,
                    "completed_date": t.completed_date.isoformat() if t.completed_date else None,
                    "notes": t.notes,
                }
                for t in self.ALL_TASKS
            ],
        }
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Geo plan state saved to %s", out)

    def load_state(self, path: Path | None = None) -> bool:
        load_path = path or _STATE_FILE
        if not load_path.exists():
            return False
        try:
            data = json.loads(load_path.read_text(encoding="utf-8"))
            for saved in data.get("tasks", []):
                task = self._task_map.get(saved["id"])
                if task is None:
                    continue
                task.completed = saved.get("completed", False)
                cd = saved.get("completed_date")
                if cd:
                    task.completed_date = date.fromisoformat(cd)
                task.notes = saved.get("notes", "")
            logger.info("Geo plan state loaded from %s", load_path)
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to load geo plan state: %s", exc)
            return False

    # ── Dashboard ────────────────────────────────────────────────────────

    def format_dashboard(
        self,
        current_moon: int = 1,
        base_filter: str | None = None,
    ) -> str:
        """Return a formatted ASCII geo plan dashboard."""
        # Filter tasks
        if base_filter and base_filter != "ALL":
            try:
                base_enum = GeoBase(base_filter.lower())
                tasks = self.tasks_by_base(base_enum)
            except ValueError:
                tasks = self.ALL_TASKS
        else:
            tasks = self.ALL_TASKS

        total = len(tasks)
        done = sum(1 for t in tasks if t.completed)
        active = [t for t in tasks if t.is_due(current_moon) and not t.completed]
        overdue = [t for t in tasks if t.is_overdue(current_moon)]

        lines: list[str] = [
            f"  Moon {current_moon}  |  Total: {total}  |  Done: {done}  |"
            f"  Active: {len(active)}  |  Overdue: {len(overdue)}",
            "",
        ]

        # Active tasks
        if active:
            lines.append("  ACTIVE TASKS (this moon):")
            for t in sorted(active, key=lambda x: x.priority):
                prio_sym = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(
                    t.priority, "⚪"
                )
                base_label = t.base.value.upper()[:3]
                lines.append(
                    f"    {prio_sym} [{t.id}] {t.title}"
                    f"  ({base_label}, moons {t.start_moon}-{t.end_moon})"
                )
            lines.append("")

        # Overdue tasks
        if overdue:
            lines.append("  ⚠️  OVERDUE:")
            for t in overdue:
                lines.append(
                    f"    ❌ [{t.id}] {t.title}"
                    f"  (due by moon {t.end_moon})"
                )
            lines.append("")

        # Completed tasks
        completed = [t for t in tasks if t.completed]
        if completed:
            lines.append(f"  ✅ COMPLETED ({len(completed)}):")
            for t in completed:
                cd = t.completed_date.isoformat() if t.completed_date else "?"
                lines.append(f"    ✓ [{t.id}] {t.title}  (done {cd})")
            lines.append("")

        # Upcoming (next 3 moons)
        upcoming = [
            t for t in tasks
            if not t.completed
            and not t.is_due(current_moon)
            and t.start_moon <= current_moon + 3
            and t.start_moon > current_moon
        ]
        if upcoming:
            lines.append("  📋 UPCOMING (next 3 moons):")
            for t in upcoming:
                base_label = t.base.value.upper()[:3]
                lines.append(
                    f"    ○ [{t.id}] {t.title}"
                    f"  ({base_label}, starts moon {t.start_moon})"
                )
            lines.append("")

        # Progress bar
        pct = done / total * 100 if total else 0
        filled = int(pct / 2.5)
        bar = "█" * filled + "░" * (40 - filled)
        lines.append(f"  Progress: [{bar}] {pct:.0f}%  ({done}/{total})")

        return "\n".join(lines)

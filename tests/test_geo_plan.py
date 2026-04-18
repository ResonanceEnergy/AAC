"""Tests for strategies.rocket_ship.geo_plan — Geo relocation task tracker."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from strategies.rocket_ship.core import GeoBase
from strategies.rocket_ship.geo_plan import GeoPlanEngine, GeoTask


@pytest.fixture
def engine() -> GeoPlanEngine:
    return GeoPlanEngine()


# ═══════════════════════════════════════════════════════════════════════════
# GeoTask dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestGeoTask:
    """GeoTask methods and properties."""

    def test_is_due_within_range(self) -> None:
        task = GeoTask(
            id="T-01", title="Test", base=GeoBase.PANAMA,
            description="", start_moon=3, end_moon=6, priority="HIGH",
        )
        assert task.is_due(3) is True
        assert task.is_due(5) is True
        assert task.is_due(6) is True

    def test_is_due_outside_range(self) -> None:
        task = GeoTask(
            id="T-01", title="Test", base=GeoBase.PANAMA,
            description="", start_moon=3, end_moon=6, priority="HIGH",
        )
        assert task.is_due(2) is False
        assert task.is_due(7) is False

    def test_is_overdue_when_past_deadline(self) -> None:
        task = GeoTask(
            id="T-01", title="Test", base=GeoBase.PANAMA,
            description="", start_moon=1, end_moon=4, priority="HIGH",
        )
        assert task.is_overdue(5) is True
        assert task.is_overdue(4) is False

    def test_completed_task_not_overdue(self) -> None:
        task = GeoTask(
            id="T-01", title="Test", base=GeoBase.PANAMA,
            description="", start_moon=1, end_moon=4, priority="HIGH",
            completed=True,
        )
        assert task.is_overdue(5) is False


# ═══════════════════════════════════════════════════════════════════════════
# GeoPlanEngine queries
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineQueries:
    """Query methods on GeoPlanEngine."""

    def test_all_tasks_populated(self, engine: GeoPlanEngine) -> None:
        assert len(engine.ALL_TASKS) > 10
        assert all(isinstance(t, GeoTask) for t in engine.ALL_TASKS)

    def test_task_ids_unique(self, engine: GeoPlanEngine) -> None:
        ids = [t.id for t in engine.ALL_TASKS]
        assert len(ids) == len(set(ids))

    def test_complete_tasks_initially_empty(self, engine: GeoPlanEngine) -> None:
        assert engine.complete_tasks() == []

    def test_incomplete_tasks_equals_all(self, engine: GeoPlanEngine) -> None:
        assert len(engine.incomplete_tasks()) == len(engine.ALL_TASKS)

    def test_tasks_for_moon_1(self, engine: GeoPlanEngine) -> None:
        active = engine.tasks_for_moon(1)
        assert len(active) > 0
        # All returned tasks should span moon 1
        for t in active:
            assert t.start_moon <= 1 <= t.end_moon

    def test_tasks_for_moon_far_future_empty(self, engine: GeoPlanEngine) -> None:
        active = engine.tasks_for_moon(100)
        assert active == []

    def test_overdue_tasks_at_moon_1_empty(self, engine: GeoPlanEngine) -> None:
        assert engine.overdue_tasks(1) == []

    def test_overdue_tasks_at_moon_50(self, engine: GeoPlanEngine) -> None:
        overdue = engine.overdue_tasks(50)
        # All tasks end before moon 50, so all should be overdue
        assert len(overdue) == len(engine.ALL_TASKS)

    def test_tasks_by_base_panama(self, engine: GeoPlanEngine) -> None:
        panama = engine.tasks_by_base(GeoBase.PANAMA)
        assert len(panama) > 0
        assert all(t.base == GeoBase.PANAMA for t in panama)

    def test_tasks_by_base_uae(self, engine: GeoPlanEngine) -> None:
        uae = engine.tasks_by_base(GeoBase.UAE)
        assert len(uae) > 0
        assert all(t.base == GeoBase.UAE for t in uae)

    def test_tasks_by_base_paraguay(self, engine: GeoPlanEngine) -> None:
        py = engine.tasks_by_base(GeoBase.PARAGUAY)
        assert len(py) > 0
        assert all(t.base == GeoBase.PARAGUAY for t in py)


# ═══════════════════════════════════════════════════════════════════════════
# Mark complete
# ═══════════════════════════════════════════════════════════════════════════

class TestMarkComplete:
    """mark_complete mutation tests."""

    def test_mark_complete_success(self, engine: GeoPlanEngine) -> None:
        task_id = engine.ALL_TASKS[0].id
        assert engine.mark_complete(task_id, notes="done") is True
        task = next(t for t in engine.ALL_TASKS if t.id == task_id)
        assert task.completed is True
        assert task.completed_date == date.today()
        assert task.notes == "done"

    def test_mark_complete_unknown_id(self, engine: GeoPlanEngine) -> None:
        assert engine.mark_complete("NONEXISTENT-99") is False

    def test_mark_complete_updates_queries(self, engine: GeoPlanEngine) -> None:
        task_id = engine.ALL_TASKS[0].id
        before = len(engine.complete_tasks())
        engine.mark_complete(task_id)
        assert len(engine.complete_tasks()) == before + 1

    def test_completed_task_excluded_from_active(self, engine: GeoPlanEngine) -> None:
        # Find a task due at moon 1
        active = engine.tasks_for_moon(1)
        assert len(active) > 0
        first_id = active[0].id
        engine.mark_complete(first_id)
        # Should no longer appear in active
        new_active = engine.tasks_for_moon(1)
        assert all(t.id != first_id for t in new_active)


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistence:
    """save_state / load_state tests."""

    def test_save_and_load_roundtrip(self, engine: GeoPlanEngine, tmp_path: Path) -> None:
        state_file = tmp_path / "geo_state.json"
        engine.mark_complete("CA-01", notes="consolidated")
        engine.mark_complete("PA-01", notes="visa submitted")
        engine.save_state(state_file)

        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "tasks" in data
        assert "saved_at" in data

        # Load into fresh engine
        engine2 = GeoPlanEngine()
        assert engine2.load_state(state_file) is True
        completed_ids = {t.id for t in engine2.complete_tasks()}
        assert "CA-01" in completed_ids
        assert "PA-01" in completed_ids

    def test_load_nonexistent_returns_false(self, engine: GeoPlanEngine, tmp_path: Path) -> None:
        assert engine.load_state(tmp_path / "missing.json") is False

    def test_load_corrupt_json_returns_false(self, engine: GeoPlanEngine, tmp_path: Path) -> None:
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("not valid json{{{", encoding="utf-8")
        assert engine.load_state(bad_file) is False

    def test_load_preserves_notes(self, engine: GeoPlanEngine, tmp_path: Path) -> None:
        state_file = tmp_path / "geo_state.json"
        engine.mark_complete("CA-02", notes="advisor consulted")
        engine.save_state(state_file)

        engine2 = GeoPlanEngine()
        engine2.load_state(state_file)
        task = next(t for t in engine2.ALL_TASKS if t.id == "CA-02")
        assert task.notes == "advisor consulted"


# ═══════════════════════════════════════════════════════════════════════════
# Instance isolation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstanceIsolation:
    """Each engine instance should have independent task state."""

    def test_instances_are_independent(self) -> None:
        e1 = GeoPlanEngine()
        e2 = GeoPlanEngine()
        e1.mark_complete("CA-01")
        # e2 should NOT be affected
        assert len(e2.complete_tasks()) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard formatting
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatDashboard:
    """format_dashboard output tests."""

    def test_returns_string(self, engine: GeoPlanEngine) -> None:
        result = engine.format_dashboard(current_moon=1)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_contains_moon_number(self, engine: GeoPlanEngine) -> None:
        result = engine.format_dashboard(current_moon=5)
        assert "Moon 5" in result

    def test_contains_total_count(self, engine: GeoPlanEngine) -> None:
        result = engine.format_dashboard(current_moon=1)
        total = len(engine.ALL_TASKS)
        assert f"Total: {total}" in result

    def test_base_filter_panama(self, engine: GeoPlanEngine) -> None:
        result = engine.format_dashboard(current_moon=1, base_filter="panama")
        # Should still produce output with Panama tasks
        assert isinstance(result, str)
        assert len(result) > 30

    def test_invalid_base_filter_shows_all(self, engine: GeoPlanEngine) -> None:
        result = engine.format_dashboard(current_moon=1, base_filter="mars")
        total = len(engine.ALL_TASKS)
        assert f"Total: {total}" in result

from __future__ import annotations

from datetime import datetime

import pytest

from shared.incident_manager import (
    Incident,
    IncidentManager,
    IncidentSeverity,
    IncidentStatus,
)


# ---------------------------------------------------------------------------
# Enums & dataclass
# ---------------------------------------------------------------------------


class TestIncidentSeverity:
    def test_values(self):
        assert IncidentSeverity.LOW.value == "low"
        assert IncidentSeverity.MEDIUM.value == "medium"
        assert IncidentSeverity.HIGH.value == "high"
        assert IncidentSeverity.CRITICAL.value == "critical"

    def test_count(self):
        assert len(list(IncidentSeverity)) == 4


class TestIncidentStatus:
    def test_values(self):
        assert IncidentStatus.OPEN.value == "open"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.MITIGATING.value == "mitigating"
        assert IncidentStatus.RESOLVED.value == "resolved"

    def test_count(self):
        assert len(list(IncidentStatus)) == 4


class TestIncidentDataclass:
    def test_required_and_defaults(self):
        inc = Incident(
            incident_id="INC-00001",
            title="t",
            description="d",
            severity=IncidentSeverity.HIGH,
            source="api",
        )
        assert inc.incident_id == "INC-00001"
        assert inc.title == "t"
        assert inc.description == "d"
        assert inc.severity is IncidentSeverity.HIGH
        assert inc.source == "api"
        assert inc.status is IncidentStatus.OPEN
        assert isinstance(inc.created_at, datetime)
        assert isinstance(inc.updated_at, datetime)
        assert inc.resolved_at is None
        assert inc.assignee is None
        assert inc.notes == []

    def test_independent_notes_list(self):
        a = Incident("a", "t", "d", IncidentSeverity.LOW, "s")
        b = Incident("b", "t", "d", IncidentSeverity.LOW, "s")
        a.notes.append("x")
        assert b.notes == []


# ---------------------------------------------------------------------------
# IncidentManager init
# ---------------------------------------------------------------------------


class TestIncidentManagerInit:
    def test_starts_empty(self):
        mgr = IncidentManager()
        assert mgr.incidents == {}
        assert mgr._counter == 0


# ---------------------------------------------------------------------------
# create_incident
# ---------------------------------------------------------------------------


class TestCreateIncident:
    def test_returns_incident_with_default_severity(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("Title", "Desc")
        assert inc.title == "Title"
        assert inc.description == "Desc"
        assert inc.severity is IncidentSeverity.MEDIUM
        assert inc.source == "system"
        assert inc.status is IncidentStatus.OPEN

    def test_id_format(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("a", "b")
        assert inc.incident_id == "INC-00001"

    def test_id_increments(self):
        mgr = IncidentManager()
        ids = [mgr.create_incident(f"t{i}", "d").incident_id for i in range(3)]
        assert ids == ["INC-00001", "INC-00002", "INC-00003"]

    def test_registered_in_dict(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("a", "b", severity=IncidentSeverity.HIGH, source="ibkr")
        assert mgr.incidents[inc.incident_id] is inc
        assert inc.severity is IncidentSeverity.HIGH
        assert inc.source == "ibkr"

    def test_custom_severity_and_source(self):
        mgr = IncidentManager()
        inc = mgr.create_incident(
            "t", "d", severity=IncidentSeverity.CRITICAL, source="watchdog"
        )
        assert inc.severity is IncidentSeverity.CRITICAL
        assert inc.source == "watchdog"


# ---------------------------------------------------------------------------
# escalate
# ---------------------------------------------------------------------------


class TestEscalate:
    def test_escalates_existing(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("t", "d", severity=IncidentSeverity.LOW)
        result = mgr.escalate(inc.incident_id, IncidentSeverity.HIGH)
        assert result is True
        assert inc.severity is IncidentSeverity.HIGH
        assert any("low" in n and "high" in n for n in inc.notes)

    def test_returns_false_for_missing(self):
        mgr = IncidentManager()
        assert mgr.escalate("INC-99999", IncidentSeverity.CRITICAL) is False

    def test_updates_timestamp(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("t", "d")
        original = inc.updated_at
        # advance time deterministically by manually rewinding original
        inc.updated_at = datetime(2020, 1, 1)
        mgr.escalate(inc.incident_id, IncidentSeverity.CRITICAL)
        assert inc.updated_at > datetime(2020, 1, 1)

    def test_escalation_note_added(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("t", "d", severity=IncidentSeverity.MEDIUM)
        mgr.escalate(inc.incident_id, IncidentSeverity.CRITICAL)
        assert len(inc.notes) == 1
        assert "medium" in inc.notes[0]
        assert "critical" in inc.notes[0]


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolves_existing(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("t", "d")
        result = mgr.resolve(inc.incident_id, "fixed it")
        assert result is True
        assert inc.status is IncidentStatus.RESOLVED
        assert inc.resolved_at is not None
        assert any("fixed it" in n for n in inc.notes)

    def test_returns_false_for_missing(self):
        mgr = IncidentManager()
        assert mgr.resolve("INC-99999") is False

    def test_resolve_without_note_adds_no_note(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("t", "d")
        mgr.resolve(inc.incident_id)
        assert inc.notes == []
        assert inc.status is IncidentStatus.RESOLVED


# ---------------------------------------------------------------------------
# get_open_incidents / get_incident
# ---------------------------------------------------------------------------


class TestGetters:
    def test_get_open_excludes_resolved(self):
        mgr = IncidentManager()
        a = mgr.create_incident("a", "d")
        b = mgr.create_incident("b", "d")
        mgr.create_incident("c", "d")
        mgr.resolve(b.incident_id)
        opens = mgr.get_open_incidents()
        ids = {i.incident_id for i in opens}
        assert a.incident_id in ids
        assert b.incident_id not in ids
        assert len(opens) == 2

    def test_get_open_empty_when_none(self):
        mgr = IncidentManager()
        assert mgr.get_open_incidents() == []

    def test_get_incident_returns_existing(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("t", "d")
        assert mgr.get_incident(inc.incident_id) is inc

    def test_get_incident_returns_none_for_missing(self):
        mgr = IncidentManager()
        assert mgr.get_incident("INC-99999") is None


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    def test_empty_metrics(self):
        mgr = IncidentManager()
        m = mgr.get_metrics()
        assert m["total_incidents"] == 0
        assert m["open_incidents"] == 0
        assert m["resolved_incidents"] == 0
        assert m["by_severity"] == {"low": 0, "medium": 0, "high": 0, "critical": 0}

    def test_metrics_after_mixed_incidents(self):
        mgr = IncidentManager()
        mgr.create_incident("a", "d", severity=IncidentSeverity.LOW)
        mgr.create_incident("b", "d", severity=IncidentSeverity.HIGH)
        c = mgr.create_incident("c", "d", severity=IncidentSeverity.CRITICAL)
        mgr.resolve(c.incident_id)

        m = mgr.get_metrics()
        assert m["total_incidents"] == 3
        assert m["open_incidents"] == 2
        assert m["resolved_incidents"] == 1
        assert m["by_severity"]["low"] == 1
        assert m["by_severity"]["high"] == 1
        assert m["by_severity"]["critical"] == 1
        assert m["by_severity"]["medium"] == 0

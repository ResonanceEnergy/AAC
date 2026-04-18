from __future__ import annotations

"""Tests for strategies/roadmap_storyboard.py — Unified Roadmap Dashboard."""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fake data classes for mocking doctrine/war room imports
# ============================================================================

@dataclass
class _FakeDoctrineAction:
    mandate: str = "ACCUMULATE"
    conviction: int = 8
    description: str = "Deploy capital on dips"
    targets: list = field(default_factory=lambda: ["SPY", "QQQ"])


@dataclass
class _FakePhiMarker:
    date: date = field(default_factory=date.today)
    label: str = "Phi^1.618 Node"
    phi_power: float = 1.618
    resonance_strength: float = 0.85


@dataclass
class _FakeEvent:
    date: date = field(default_factory=date.today)
    name: str = "Test Event"
    impact: str = "HIGH"
    description: str = "A test event"


@dataclass
class _FakeMoonCycle:
    moon_number: int = 1
    lunar_phase_name: str = "Seed Moon"
    start_date: date = field(default_factory=lambda: date(2026, 3, 3))
    end_date: date = field(default_factory=lambda: date(2026, 3, 31))
    fire_peak_date: date | None = field(default_factory=lambda: date(2026, 3, 14))
    new_moon_date: date | None = field(default_factory=lambda: date(2026, 3, 3))
    doctrine_action: _FakeDoctrineAction | None = field(default_factory=_FakeDoctrineAction)
    astrology_events: list = field(default_factory=list)
    phi_markers: list = field(default_factory=list)
    financial_events: list = field(default_factory=list)
    world_events: list = field(default_factory=list)
    aac_events: list = field(default_factory=list)


@dataclass
class _FakeAlert:
    event_date: date = field(default_factory=date.today)
    event_name: str = "Approaching deadline"
    event_type: str = "financial"
    days_until: int = 3
    priority: str = "HIGH"
    lead_time_action: str = "Review positions"
    moon_number: int = 1


class _FakeDoctrine:
    """Minimal stand-in for ThirteenMoonDoctrine."""

    def __init__(self) -> None:
        today = date.today()
        self.moon_cycles = [
            _FakeMoonCycle(
                moon_number=1,
                start_date=today - timedelta(days=10),
                end_date=today + timedelta(days=18),
                astrology_events=[_FakeEvent(date=today, name="Mercury Rx")],
                phi_markers=[_FakePhiMarker(date=today + timedelta(days=2))],
            ),
            _FakeMoonCycle(
                moon_number=2,
                lunar_phase_name="Pollination Moon",
                start_date=today + timedelta(days=19),
                end_date=today + timedelta(days=47),
                fire_peak_date=today + timedelta(days=33),
            ),
        ]

    def get_current_moon(self, target: date | None = None) -> _FakeMoonCycle:
        return self.moon_cycles[0]

    def get_events_with_lead_time(
        self, days_ahead: int = 14, target: date | None = None
    ) -> list:
        return [
            _FakeAlert(event_date=date.today() + timedelta(days=3)),
            _FakeAlert(
                event_date=date.today() + timedelta(days=7),
                event_name="Options Expiry",
                event_type="options",
                days_until=7,
                priority="CRITICAL",
                lead_time_action="Roll or close",
            ),
        ]


# ============================================================================
# Import helpers — must patch before importing the module
# ============================================================================

def _import_module():
    """Import roadmap_storyboard lazily so patches take effect."""
    import strategies.roadmap_storyboard as mod
    return mod


# ============================================================================
# Tests — _collect_moon_data
# ============================================================================

class TestCollectMoonData:
    """Tests for _collect_moon_data."""

    def test_moon_data_graceful_fallback(self):
        """When doctrine import fails, return empty structure."""
        mod = _import_module()
        # Force import failure by making the module raise ImportError
        broken = MagicMock()
        broken.ThirteenMoonDoctrine.side_effect = ImportError("no doctrine")
        with patch.dict("sys.modules", {"strategies.thirteen_moon_doctrine": broken}):
            result = mod._collect_moon_data(date.today())
        assert result["cycles"] == []
        assert result["current"] == {}
        assert result["alerts"] == []
        assert result["total_cycles"] == 0

    def test_moon_data_with_mock_doctrine(self):
        """When doctrine is available, collect full moon data."""
        mod = _import_module()
        fake = _FakeDoctrine()

        mock_doctrine_mod = MagicMock()
        mock_doctrine_mod.ThirteenMoonDoctrine.return_value = fake
        mock_doctrine_mod.MOON_BRIEFINGS = {
            1: {"theme": "Seed", "market_implication": "Accumulate"},
            2: {"theme": "Pollin"},
        }
        mock_doctrine_mod.SACRED_GEOMETRY_OVERLAY = {
            1: {"geometry": "Vesica Piscis", "frequency_hz": "432"},
        }

        with patch.dict("sys.modules", {"strategies.thirteen_moon_doctrine": mock_doctrine_mod}):
            result = mod._collect_moon_data(date.today())

        assert len(result["cycles"]) == 2
        assert result["current"]["moon_number"] == 1
        assert result["total_cycles"] == 2
        # Should have alerts
        assert len(result["alerts"]) == 2
        assert result["alerts"][0]["priority"] == "HIGH"

    def test_moon_data_progress_calculation(self):
        """Progress percentage should be reasonable for current moon."""
        mod = _import_module()
        fake = _FakeDoctrine()

        mock_doctrine_mod = MagicMock()
        mock_doctrine_mod.ThirteenMoonDoctrine.return_value = fake
        mock_doctrine_mod.MOON_BRIEFINGS = {}
        mock_doctrine_mod.SACRED_GEOMETRY_OVERLAY = {}

        with patch.dict("sys.modules", {"strategies.thirteen_moon_doctrine": mock_doctrine_mod}):
            result = mod._collect_moon_data(date.today())

        current = result["current"]
        assert 0 <= current["progress_pct"] <= 100
        assert current["days_in"] >= 0
        assert current["days_left"] >= 0


# ============================================================================
# Tests — _collect_daily_tasks
# ============================================================================

class TestCollectDailyTasks:
    """Tests for _collect_daily_tasks."""

    def test_daily_tasks_graceful_fallback(self):
        """When DailyTaskAggregator fails, return empty structure."""
        mod = _import_module()
        with patch.dict("sys.modules", {"monitoring.daily_tasks": None}):
            result = mod._collect_daily_tasks(date.today())
        assert result["total_tasks"] == 0
        assert result["all_tasks"] == []

    def test_daily_tasks_with_mock(self):
        """When DailyTaskAggregator works, pass through its output."""
        mod = _import_module()
        fake_output = {
            "date": "2026-04-10",
            "total_tasks": 5,
            "completed": 1,
            "remaining": 4,
            "by_priority": {"HIGH": 2, "MEDIUM": 3},
            "by_source": {"moon": 2, "options": 3},
            "by_slot": {"pre_market": 3, "market_open": 2},
            "slots": {},
            "today_tasks": [{"name": "Check VIX"}],
            "upcoming_tasks": [],
            "all_tasks": [{"name": "Check VIX"}],
        }

        mock_mod = MagicMock()
        mock_agg = MagicMock()
        mock_agg.collect_all.return_value = fake_output
        mock_mod.DailyTaskAggregator.return_value = mock_agg

        with patch.dict("sys.modules", {"monitoring.daily_tasks": mock_mod}):
            result = mod._collect_daily_tasks(date.today())

        assert result["total_tasks"] == 5
        assert result["completed"] == 1


# ============================================================================
# Tests — _collect_weekly_calendar
# ============================================================================

class TestCollectWeeklyCalendar:
    """Tests for _collect_weekly_calendar."""

    def test_weekly_always_returns_7_days(self):
        """Even with import failure, should return 7-day structure."""
        mod = _import_module()
        with patch.dict("sys.modules", {"strategies.thirteen_moon_doctrine": None}):
            result = mod._collect_weekly_calendar(date.today())
        assert len(result["days"]) == 7
        assert result["days"][0]["is_today"] is True
        assert result["days"][6]["is_today"] is False

    def test_weekly_dates_are_sequential(self):
        """Days should be in order starting from today."""
        mod = _import_module()
        with patch.dict("sys.modules", {"strategies.thirteen_moon_doctrine": None}):
            result = mod._collect_weekly_calendar(date(2026, 7, 1))
        dates = [d["date"] for d in result["days"]]
        assert dates[0] == "2026-07-01"
        assert dates[6] == "2026-07-07"

    def test_weekly_with_mock_events(self):
        """When doctrine is available, events should appear on matching days."""
        mod = _import_module()
        today = date.today()
        fake = _FakeDoctrine()
        # Ensure the fake cycle covers today
        fake.moon_cycles[0].start_date = today - timedelta(days=5)
        fake.moon_cycles[0].end_date = today + timedelta(days=25)

        mock_doctrine_mod = MagicMock()
        mock_doctrine_mod.ThirteenMoonDoctrine.return_value = fake
        mock_doctrine_mod.MOON_BRIEFINGS = {}

        with patch.dict("sys.modules", {"strategies.thirteen_moon_doctrine": mock_doctrine_mod}):
            result = mod._collect_weekly_calendar(today)

        # Today should have the Mercury Rx event we put on today
        today_events = result["days"][0]["events"]
        event_names = [e["name"] for e in today_events]
        assert "Mercury Rx" in event_names


# ============================================================================
# Tests — _collect_war_room
# ============================================================================

class TestCollectWarRoom:
    """Tests for _collect_war_room."""

    def test_war_room_graceful_fallback(self):
        """When war room engine unavailable, return safe defaults."""
        mod = _import_module()
        with patch.dict("sys.modules", {"strategies.war_room_engine": None}):
            result = mod._collect_war_room(date.today())
        assert result["composite_score"] == 0
        assert result["regime"] == "UNKNOWN"
        assert result["indicators"] == []
        assert result["scenarios"] == []

    def test_war_room_regime_thresholds(self):
        """Regime should map to correct threshold bucket."""
        mod = _import_module()

        mock_wre = MagicMock()
        mock_wre.IndicatorState.return_value = MagicMock()
        mock_wre.SCENARIOS = {
            "tariff_war": {"name": "Tariff War", "probability": 0.65, "status": "ACTIVE"},
        }
        mock_wre.MILESTONES = []
        mock_wre.load_milestone_state.return_value = {}
        mock_wre.get_current_phase.return_value = "deployment"

        # Test CRISIS (>70)
        mock_wre.compute_composite_score.return_value = {
            "composite_score": 75.0,
            "individual_scores": {"vix": 90.0},
            "regime": "CRISIS",
            "confidence": 0.75,
        }
        with patch.dict("sys.modules", {"strategies.war_room_engine": mock_wre}):
            result = mod._collect_war_room(date.today())
        assert result["regime"] == "CRISIS"
        assert result["composite_score"] == 75.0

        # Test CALM (<30)
        mock_wre.compute_composite_score.return_value = {
            "composite_score": 15.0,
            "individual_scores": {"vix": 10.0},
            "regime": "CALM",
            "confidence": 0.15,
        }
        with patch.dict("sys.modules", {"strategies.war_room_engine": mock_wre}):
            result = mod._collect_war_room(date.today())
        assert result["regime"] == "CALM"

    def test_war_room_captures_scenarios(self):
        """Scenarios from SCENARIOS dict should be collected."""
        mod = _import_module()

        mock_wre = MagicMock()
        mock_wre.IndicatorState.return_value = MagicMock()
        mock_wre.SCENARIOS = {
            "tariff_war": {"name": "Tariff War", "probability": 0.65, "status": "ACTIVE"},
            "fed_pivot": {"name": "Fed Pivot", "probability": 0.30, "status": "WATCH"},
        }
        mock_wre.MILESTONES = []
        mock_wre.compute_composite_score.return_value = {
            "composite_score": 42.0,
            "individual_scores": {},
            "regime": "WATCH",
            "confidence": 0.42,
        }
        mock_wre.load_milestone_state.return_value = {}
        mock_wre.get_current_phase.return_value = "accumulation"

        with patch.dict("sys.modules", {"strategies.war_room_engine": mock_wre}):
            result = mod._collect_war_room(date.today())

        assert len(result["scenarios"]) == 2
        assert result["scenarios"][0]["name"] == "Tariff War"

    def test_war_room_milestone_count(self):
        """Achieved milestones should be counted."""
        mod = _import_module()

        mock_wre = MagicMock()
        mock_wre.IndicatorState.return_value = MagicMock()
        mock_wre.SCENARIOS = {}
        mock_wre.MILESTONES = []
        mock_wre.compute_composite_score.return_value = {
            "composite_score": 20.0,
            "individual_scores": {},
            "regime": "CALM",
            "confidence": 0.20,
        }
        mock_wre.load_milestone_state.return_value = {"ms1": True, "ms2": True, "ms3": False}
        mock_wre.get_current_phase.return_value = "accumulation"

        with patch.dict("sys.modules", {"strategies.war_room_engine": mock_wre}):
            result = mod._collect_war_room(date.today())

        assert result["milestones"]["achieved"] == 2


# ============================================================================
# Tests — _collect_portfolio
# ============================================================================

class TestCollectPortfolio:
    """Tests for _collect_portfolio."""

    def test_portfolio_fallback_no_files(self, tmp_path):
        """When no state files exist, return defaults from constants."""
        mod = _import_module()

        with patch.object(mod.Path, "__new__", return_value=tmp_path / "fake"):
            # Simpler: just point at an empty dir via __file__ proxy
            pass

        # Easiest: patch glob.glob to return nothing
        with patch("strategies.roadmap_storyboard.glob.glob", return_value=[]):
            result = mod._collect_portfolio()

        assert result["total_balance"] > 0
        assert len(result["accounts"]) == 1
        assert result["accounts"][0]["id"] == "default"
        assert "tickers" in result

    def test_portfolio_reads_state_file(self, tmp_path):
        """Should parse a paper_trading account JSON file."""
        mod = _import_module()

        state = {
            "account_id": "test_acct",
            "balance": 9500.50,
            "equity": 9600.75,
            "total_pnl": 100.25,
            "positions": {"SPY": {"qty": 10}, "AAPL": {"qty": 5}},
        }
        state_file = tmp_path / "test_acct.json"
        state_file.write_text(json.dumps(state), encoding="utf-8")

        with patch("strategies.roadmap_storyboard.glob.glob", return_value=[str(state_file)]):
            with patch("strategies.roadmap_storyboard._load_ticker_cache", return_value=None):
                with patch("strategies.roadmap_storyboard._fetch_live_tickers", return_value={}):
                    result = mod._collect_portfolio()

        assert len(result["accounts"]) == 1
        assert result["accounts"][0]["id"] == "test_acct"
        assert result["accounts"][0]["balance"] == 9500.50
        assert result["accounts"][0]["pnl"] == 100.25
        assert result["accounts"][0]["positions"] == 2
        assert result["total_balance"] == 9500.50
        assert result["total_positions"] == 2

    def test_portfolio_tickers_live(self):
        """Should fetch live prices from yfinance when cache is empty."""
        mod = _import_module()

        fake_tickers = {
            "SPY": {"price": 550.0, "prev_close": 548.0, "change": 2.0,
                     "change_pct": 0.36, "sparkline": [545, 547, 549, 548, 550]},
            "BTC": {"price": 70000.0, "prev_close": 69500.0, "change": 500.0,
                     "change_pct": 0.72, "sparkline": [69000, 69500, 70200, 69800, 70000]},
            "GOLD": {"price": 3200.0, "prev_close": 3180.0, "change": 20.0,
                      "change_pct": 0.63, "sparkline": [3170, 3180, 3190, 3200]},
            "OIL": {"price": 85.0, "prev_close": 86.0, "change": -1.0,
                     "change_pct": -1.16, "sparkline": [87, 86, 85.5, 85]},
            "VIX": {"price": 18.5, "prev_close": 19.0, "change": -0.5,
                     "change_pct": -2.63, "sparkline": [20, 19.5, 19, 18.5]},
            "DXY": {"price": 104.2, "prev_close": 104.0, "change": 0.2,
                     "change_pct": 0.19, "sparkline": [103.8, 104.0, 104.1, 104.2]},
        }

        with patch("strategies.roadmap_storyboard.glob.glob", return_value=[]):
            with patch("strategies.roadmap_storyboard._load_ticker_cache", return_value=None):
                with patch("strategies.roadmap_storyboard._fetch_live_tickers", return_value=fake_tickers):
                    with patch("strategies.roadmap_storyboard._save_ticker_cache") as mock_save:
                        result = mod._collect_portfolio()

        assert result["tickers"]["SPY"]["price"] == 550.0
        assert result["tickers"]["BTC"]["change_pct"] == 0.72
        assert result["tickers"]["OIL"]["change"] == -1.0
        assert len(result["tickers"]["VIX"]["sparkline"]) == 4
        mock_save.assert_called_once_with(fake_tickers)

    def test_portfolio_tickers_from_cache(self):
        """Should use cached ticker data when fresh."""
        mod = _import_module()

        cached = {
            "SPY": {"price": 540.0, "prev_close": 538.0, "change": 2.0,
                     "change_pct": 0.37, "sparkline": [535, 538, 540]},
        }

        with patch("strategies.roadmap_storyboard.glob.glob", return_value=[]):
            with patch("strategies.roadmap_storyboard._load_ticker_cache", return_value=cached):
                result = mod._collect_portfolio()

        assert result["tickers"]["SPY"]["price"] == 540.0
        assert result["tickers"]["SPY"]["sparkline"] == [535, 538, 540]

    def test_portfolio_tickers_fallback(self):
        """When yfinance unavailable and no cache, tickers should be zero-filled."""
        mod = _import_module()

        with patch("strategies.roadmap_storyboard.glob.glob", return_value=[]):
            with patch("strategies.roadmap_storyboard._load_ticker_cache", return_value=None):
                with patch("strategies.roadmap_storyboard._fetch_live_tickers", return_value={}):
                    result = mod._collect_portfolio()

        assert result["tickers"]["SPY"]["price"] == 0
        assert result["tickers"]["BTC"]["price"] == 0
        assert result["tickers"]["VIX"]["sparkline"] == []

    def test_portfolio_skips_corrupt_json(self, tmp_path):
        """Corrupt JSON files should be skipped gracefully."""
        mod = _import_module()

        bad_file = tmp_path / "broken.json"
        bad_file.write_text("{broken", encoding="utf-8")

        with patch("strategies.roadmap_storyboard.glob.glob", return_value=[str(bad_file)]):
            with patch("strategies.roadmap_storyboard._load_ticker_cache", return_value=None):
                with patch("strategies.roadmap_storyboard._fetch_live_tickers", return_value={}):
                    result = mod._collect_portfolio()

        # Should fall through to default account
        assert len(result["accounts"]) == 1
        assert result["accounts"][0]["id"] == "default"

    def test_portfolio_multiple_accounts(self, tmp_path):
        """Multiple state files should be aggregated."""
        mod = _import_module()

        for i, bal in enumerate([5000.0, 3000.0]):
            state = {
                "account_id": f"acct_{i}",
                "balance": bal,
                "equity": bal,
                "total_pnl": 0.0,
                "positions": {},
            }
            f = tmp_path / f"acct_{i}.json"
            f.write_text(json.dumps(state), encoding="utf-8")

        files = sorted([str(tmp_path / f"acct_{i}.json") for i in range(2)])
        with patch("strategies.roadmap_storyboard.glob.glob", return_value=files):
            with patch("strategies.roadmap_storyboard._load_ticker_cache", return_value=None):
                with patch("strategies.roadmap_storyboard._fetch_live_tickers", return_value={}):
                    result = mod._collect_portfolio()

        assert len(result["accounts"]) == 2
        assert result["total_balance"] == 8000.0


# ============================================================================
# Tests — _collect_all_data
# ============================================================================

class TestCollectAllData:
    """Integration-level tests for _collect_all_data."""

    def test_all_data_has_required_keys(self):
        """Output should always have the 7 top-level keys."""
        mod = _import_module()
        # Patch all imports to None so fallbacks kick in
        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            result = mod._collect_all_data()

        assert "generated" in result
        assert "today" in result
        assert "day_name" in result
        assert "portfolio" in result
        assert "moon" in result
        assert "daily" in result
        assert "weekly" in result
        assert "war_room" in result

    def test_all_data_today_is_correct(self):
        """Today string should match actual date."""
        mod = _import_module()
        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            result = mod._collect_all_data()

        assert result["today"] == date.today().isoformat()


# ============================================================================
# Tests — export_roadmap (HTML generation)
# ============================================================================

class TestExportRoadmap:
    """Tests for the export_roadmap public API."""

    def test_export_creates_html_file(self, tmp_path):
        """export_roadmap should create a readable HTML file."""
        mod = _import_module()
        out = str(tmp_path / "roadmap.html")

        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            result = mod.export_roadmap(output_path=out)

        assert os.path.isfile(result)
        content = open(result, encoding="utf-8").read()
        assert "<!DOCTYPE html>" in content
        assert "AAC Command Roadmap" in content

    def test_export_html_contains_tabs(self, tmp_path):
        """HTML should have all 4 tab buttons."""
        mod = _import_module()
        out = str(tmp_path / "roadmap.html")

        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            mod.export_roadmap(output_path=out)

        content = open(out, encoding="utf-8").read()
        assert "TODAY" in content
        assert "THIS WEEK" in content
        assert "ROADMAP" in content
        assert "WAR ROOM" in content

    def test_export_html_contains_data_json(self, tmp_path):
        """HTML should embed the DATA json object."""
        mod = _import_module()
        out = str(tmp_path / "roadmap.html")

        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            mod.export_roadmap(output_path=out)

        content = open(out, encoding="utf-8").read()
        assert "const DATA =" in content

    def test_export_creates_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        mod = _import_module()
        out = str(tmp_path / "nested" / "dir" / "roadmap.html")

        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            result = mod.export_roadmap(output_path=out)

        assert os.path.isfile(result)

    def test_export_default_path(self):
        """Default output path should be data/storyboard/aac_roadmap.html."""
        mod = _import_module()
        assert mod.DEFAULT_OUTPUT == "data/storyboard/aac_roadmap.html"

    def test_export_html_has_hero_section(self, tmp_path):
        """HTML should have the hero banner section."""
        mod = _import_module()
        out = str(tmp_path / "roadmap.html")

        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            mod.export_roadmap(output_path=out)

        content = open(out, encoding="utf-8").read()
        assert "hero" in content.lower()

    def test_export_valid_json_in_html(self, tmp_path):
        """Embedded DATA json should be parseable."""
        mod = _import_module()
        out = str(tmp_path / "roadmap.html")

        with patch.dict("sys.modules", {
            "strategies.thirteen_moon_doctrine": None,
            "monitoring.daily_tasks": None,
            "strategies.war_room_engine": None,
        }):
            mod.export_roadmap(output_path=out)

        content = open(out, encoding="utf-8").read()
        # Extract the JSON between "const DATA = " and the next ";"
        start = content.index("const DATA = ") + len("const DATA = ")
        end = content.index(";", start)
        data_str = content[start:end]
        data = json.loads(data_str)
        assert "today" in data
        assert "moon" in data
        assert "portfolio" in data
        assert "war_room" in data


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_moon_data_no_current_moon(self):
        """When date is outside all cycles, current should be empty."""
        mod = _import_module()
        fake = _FakeDoctrine()
        fake.get_current_moon = lambda target=None: None

        mock_mod = MagicMock()
        mock_mod.ThirteenMoonDoctrine.return_value = fake
        mock_mod.MOON_BRIEFINGS = {}
        mock_mod.SACRED_GEOMETRY_OVERLAY = {}

        with patch.dict("sys.modules", {"strategies.thirteen_moon_doctrine": mock_mod}):
            result = mod._collect_moon_data(date.today())

        assert result["current"] == {}
        # Cycles should still be populated
        assert len(result["cycles"]) == 2

    def test_war_room_elevated_threshold(self):
        """Score 55 should give ELEVATED regime."""
        mod = _import_module()

        mock_wre = MagicMock()
        mock_wre.IndicatorState.return_value = MagicMock()
        mock_wre.SCENARIOS = {}
        mock_wre.MILESTONES = []
        mock_wre.compute_composite_score.return_value = {
            "composite_score": 55.0,
            "individual_scores": {},
            "regime": "ELEVATED",
            "confidence": 0.55,
        }
        mock_wre.load_milestone_state.return_value = {}
        mock_wre.get_current_phase.return_value = "accumulation"

        with patch.dict("sys.modules", {"strategies.war_room_engine": mock_wre}):
            result = mod._collect_war_room(date.today())
        assert result["regime"] == "ELEVATED"

    def test_war_room_watch_threshold(self):
        """Score 35 should give WATCH regime."""
        mod = _import_module()

        mock_wre = MagicMock()
        mock_wre.IndicatorState.return_value = MagicMock()
        mock_wre.SCENARIOS = {}
        mock_wre.MILESTONES = []
        mock_wre.compute_composite_score.return_value = {
            "composite_score": 35.0,
            "individual_scores": {},
            "regime": "WATCH",
            "confidence": 0.35,
        }
        mock_wre.load_milestone_state.return_value = {}
        mock_wre.get_current_phase.return_value = "accumulation"

        with patch.dict("sys.modules", {"strategies.war_room_engine": mock_wre}):
            result = mod._collect_war_room(date.today())
        assert result["regime"] == "WATCH"

    def test_daily_fallback_has_date(self):
        """Fallback daily tasks should include the date."""
        mod = _import_module()
        target = date(2026, 8, 15)
        with patch.dict("sys.modules", {"monitoring.daily_tasks": None}):
            result = mod._collect_daily_tasks(target)
        assert result["date"] == "2026-08-15"
        assert result["day_name"] == "Saturday"

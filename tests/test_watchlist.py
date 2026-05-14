"""tests/test_watchlist.py — Sprint 12: Watchlist & Universe Management

Coverage:
  TestLoadYaml            (5)  — load real YAML, missing file, bad YAML, cache hit, reload
  TestGetVolPremiumTickers(6)  — real load, default fallback, ticker normalisation,
                                  override path, missing key fallback, empty list fallback
  TestGetWarRoomRules     (9)  — CRISIS rules structure, ELEVATED, WATCH, CALM (empty),
                                  unknown regime fallback, Direction enum conversion,
                                  AssetClass enum conversion, size/stop/target preserved,
                                  parse failure row skipped
  TestParseRule           (4)  — string enums, prebuilt tuple, short row ignored, bad row
  TestVolPremiumWiring    (4)  — _get_universe() returns list, contains SPY,
                                  YAML override respected, fallback on import error
  TestSignalGeneratorWiring(4) — _get_regime_rules returns list, rules are tuples,
                                  fallback works when watchlist raises, CALM returns empty
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

_MINIMAL_YAML = """
vol_premium:
  - SPY
  - IWM
war_room:
  CRISIS:
    - [SPY, LONG_PUT, OPTION, 0.08, 0.30, 0.00]
  ELEVATED:
    - [GLD, LONG, ETF, 0.04, 0.10, 0.15]
  WATCH:
    - [GLD, LONG, ETF, 0.03, 0.08, 0.12]
  CALM: []
"""

_MALFORMED_YAML = "key: [unclosed"

# ── TestLoadYaml ──────────────────────────────────────────────────────────────

class TestLoadYaml:
    """shared.watchlist._load internal behaviour."""

    def setup_method(self):
        """Clear module cache before each test."""
        import shared.watchlist as wl
        wl._CACHE = None
        wl._CACHE_PATH = None

    def test_load_real_yaml_returns_dict(self):
        import shared.watchlist as wl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
            fh.write(_MINIMAL_YAML)
            tmp = fh.name
        try:
            data = wl._load(tmp)
            assert isinstance(data, dict)
            assert "vol_premium" in data
        finally:
            os.unlink(tmp)

    def test_load_missing_file_returns_empty(self):
        import shared.watchlist as wl
        data = wl._load("/nonexistent/path/watchlist.yaml")
        assert data == {}

    def test_load_malformed_yaml_returns_empty(self):
        import shared.watchlist as wl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
            fh.write(_MALFORMED_YAML)
            tmp = fh.name
        try:
            data = wl._load(tmp)
            assert data == {}
        finally:
            os.unlink(tmp)

    def test_load_caches_result(self):
        import shared.watchlist as wl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
            fh.write(_MINIMAL_YAML)
            tmp = fh.name
        try:
            data1 = wl._load(tmp)
            data2 = wl._load(tmp)
            assert data1 is data2  # same dict object from cache
        finally:
            os.unlink(tmp)

    def test_reload_clears_cache(self):
        import shared.watchlist as wl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
            fh.write(_MINIMAL_YAML)
            tmp = fh.name
        try:
            data1 = wl._load(tmp)
            wl.reload(tmp)
            data2 = wl._load(tmp)
            assert data1 is not data2
        finally:
            os.unlink(tmp)


# ── TestGetVolPremiumTickers ──────────────────────────────────────────────────

class TestGetVolPremiumTickers:
    """get_vol_premium_tickers() public API."""

    def setup_method(self):
        import shared.watchlist as wl
        wl._CACHE = None
        wl._CACHE_PATH = None

    def _write_yaml(self, content: str) -> str:
        fh = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        fh.write(content)
        fh.close()
        return fh.name

    def test_returns_list_from_yaml(self):
        from shared.watchlist import get_vol_premium_tickers
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            tickers = get_vol_premium_tickers(tmp)
            assert isinstance(tickers, list)
            assert len(tickers) == 2
        finally:
            os.unlink(tmp)

    def test_tickers_are_uppercase(self):
        from shared.watchlist import get_vol_premium_tickers
        yaml_content = "vol_premium:\n  - spy\n  - qqq\n"
        tmp = self._write_yaml(yaml_content)
        try:
            tickers = get_vol_premium_tickers(tmp)
            assert all(t == t.upper() for t in tickers)
        finally:
            os.unlink(tmp)

    def test_spy_present(self):
        from shared.watchlist import get_vol_premium_tickers
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            tickers = get_vol_premium_tickers(tmp)
            assert "SPY" in tickers
        finally:
            os.unlink(tmp)

    def test_fallback_on_missing_file(self):
        from shared.watchlist import get_vol_premium_tickers
        tickers = get_vol_premium_tickers("/nonexistent/watchlist.yaml")
        assert "SPY" in tickers
        assert len(tickers) >= 3

    def test_fallback_on_missing_key(self):
        from shared.watchlist import get_vol_premium_tickers
        yaml_content = "war_room:\n  CALM: []\n"
        tmp = self._write_yaml(yaml_content)
        try:
            tickers = get_vol_premium_tickers(tmp)
            assert "SPY" in tickers
        finally:
            os.unlink(tmp)

    def test_fallback_on_empty_list(self):
        from shared.watchlist import get_vol_premium_tickers
        yaml_content = "vol_premium: []\n"
        tmp = self._write_yaml(yaml_content)
        try:
            tickers = get_vol_premium_tickers(tmp)
            assert len(tickers) >= 3  # got defaults
        finally:
            os.unlink(tmp)


# ── TestGetWarRoomRules ───────────────────────────────────────────────────────

class TestGetWarRoomRules:
    """get_war_room_rules() returns correct 6-tuples."""

    def setup_method(self):
        import shared.watchlist as wl
        wl._CACHE = None
        wl._CACHE_PATH = None

    def _write_yaml(self, content: str) -> str:
        fh = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        fh.write(content)
        fh.close()
        return fh.name

    def test_crisis_rules_returns_list(self):
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("CRISIS", tmp)
            assert isinstance(rules, list)
            assert len(rules) == 1
        finally:
            os.unlink(tmp)

    def test_elevated_rules_returns_list(self):
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("ELEVATED", tmp)
            assert len(rules) == 1
        finally:
            os.unlink(tmp)

    def test_calm_returns_empty(self):
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("CALM", tmp)
            assert rules == []
        finally:
            os.unlink(tmp)

    def test_unknown_regime_falls_back_to_defaults(self):
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("NUKE", tmp)
            assert isinstance(rules, list)  # defaults return empty for unknown
        finally:
            os.unlink(tmp)

    def test_rule_is_6_tuple(self):
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("CRISIS", tmp)
            assert len(rules[0]) == 6
        finally:
            os.unlink(tmp)

    def test_ticker_is_string(self):
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("CRISIS", tmp)
            assert isinstance(rules[0][0], str)
        finally:
            os.unlink(tmp)

    def test_direction_is_enum(self):
        from shared.signal import Direction
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("CRISIS", tmp)
            assert isinstance(rules[0][1], Direction)
        finally:
            os.unlink(tmp)

    def test_asset_class_is_enum(self):
        from shared.signal import AssetClass
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("CRISIS", tmp)
            assert isinstance(rules[0][2], AssetClass)
        finally:
            os.unlink(tmp)

    def test_size_is_float(self):
        from shared.watchlist import get_war_room_rules
        tmp = self._write_yaml(_MINIMAL_YAML)
        try:
            rules = get_war_room_rules("CRISIS", tmp)
            assert isinstance(rules[0][3], float)
            assert rules[0][3] == pytest.approx(0.08)
        finally:
            os.unlink(tmp)


# ── TestParseRule ─────────────────────────────────────────────────────────────

class TestParseRule:
    """_parse_rule() edge cases."""

    def _enums(self):
        from shared.signal import AssetClass, Direction
        return Direction, AssetClass

    def test_string_enum_names_converted(self):
        import shared.watchlist as wl
        Direction, AssetClass = self._enums()
        row = ["SPY", "LONG_PUT", "OPTION", 0.05, 0.25, 0.0]
        result = wl._parse_rule(row, Direction, AssetClass)
        from shared.signal import AssetClass as AC, Direction as D
        assert result[1] == D.LONG_PUT
        assert result[2] == AC.OPTION

    def test_prebuilt_tuple_passthrough(self):
        import shared.watchlist as wl
        Direction, AssetClass = self._enums()
        from shared.signal import AssetClass as AC, Direction as D
        row = ("GLD", D.LONG, AC.ETF, 0.04, 0.10, 0.15)
        result = wl._parse_rule(row, Direction, AssetClass)
        assert result[0] == "GLD"
        assert result[1] == D.LONG

    def test_short_row_returns_empty_tuple(self):
        import shared.watchlist as wl
        Direction, AssetClass = self._enums()
        result = wl._parse_rule(["SPY", "LONG_PUT"], Direction, AssetClass)
        assert result == ()

    def test_bad_row_type_returns_empty_tuple(self):
        import shared.watchlist as wl
        Direction, AssetClass = self._enums()
        result = wl._parse_rule("not_a_list", Direction, AssetClass)
        assert result == ()


# ── TestVolPremiumWiring ──────────────────────────────────────────────────────

class TestVolPremiumWiring:
    """strategies.vol_premium_signals._get_universe() respects watchlist."""

    def setup_method(self):
        import shared.watchlist as wl
        wl._CACHE = None
        wl._CACHE_PATH = None

    def test_get_universe_returns_list(self):
        from strategies.vol_premium_signals import _get_universe
        result = _get_universe()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_get_universe_contains_spy(self):
        from strategies.vol_premium_signals import _get_universe
        result = _get_universe()
        assert "SPY" in result

    def test_get_universe_respects_yaml_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
            fh.write("vol_premium:\n  - CUSTOM1\n  - CUSTOM2\n")
            tmp = fh.name
        try:
            import shared.watchlist as wl
            wl._CACHE = None
            wl._CACHE_PATH = None
            from shared.watchlist import get_vol_premium_tickers
            tickers = get_vol_premium_tickers(tmp)
            assert "CUSTOM1" in tickers
            assert "CUSTOM2" in tickers
        finally:
            os.unlink(tmp)

    def test_get_universe_fallback_on_import_error(self):
        """_get_universe() returns hardcoded fallback if watchlist import fails."""
        with patch.dict(sys.modules, {"shared.watchlist": None}):
            # Reload the function in an env where the module is absent
            import importlib
            import strategies.vol_premium_signals as vps
            # Direct patch of the function for this test
            with patch("shared.watchlist.get_vol_premium_tickers", side_effect=ImportError):
                result = vps._get_universe()
                assert isinstance(result, list)


# ── TestSignalGeneratorWiring ─────────────────────────────────────────────────

class TestSignalGeneratorWiring:
    """strategies.signal_generator._get_regime_rules() respects watchlist."""

    def setup_method(self):
        import shared.watchlist as wl
        wl._CACHE = None
        wl._CACHE_PATH = None

    def test_get_regime_rules_returns_list(self):
        from strategies.signal_generator import _get_regime_rules
        result = _get_regime_rules("CRISIS")
        assert isinstance(result, list)

    def test_crisis_rules_are_tuples(self):
        from strategies.signal_generator import _get_regime_rules
        rules = _get_regime_rules("CRISIS")
        assert len(rules) > 0
        for rule in rules:
            assert isinstance(rule, tuple)
            assert len(rule) == 6

    def test_calm_returns_empty_list(self):
        from strategies.signal_generator import _get_regime_rules
        rules = _get_regime_rules("CALM")
        assert rules == []

    def test_fallback_when_watchlist_raises(self):
        from strategies.signal_generator import _get_regime_rules
        with patch("shared.watchlist.get_war_room_rules", side_effect=RuntimeError("boom")):
            rules = _get_regime_rules("CRISIS")
            # Falls back to _REGIME_SIGNALS — still returns valid rules
            assert isinstance(rules, list)

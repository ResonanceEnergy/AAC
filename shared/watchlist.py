"""shared/watchlist.py — Sprint 12.

Single source of truth for the AAC ticker universe.  Both ``signal_generator``
(War Room) and ``vol_premium_signals`` (VRP) read from here so you can change
the scan universe in one place — ``config/watchlist.yaml`` — without touching
any strategy code.

Usage::

    from shared.watchlist import get_vol_premium_tickers, get_war_room_rules

    tickers = get_vol_premium_tickers()          # ["SPY", "QQQ", ...]
    rules   = get_war_room_rules("CRISIS")       # list of 6-tuples

Design decisions
----------------
* YAML is parsed once and cached in a module-level dict (_CACHE).  Subsequent
  calls in the same process hit the cache — safe for the hot scan loop.
* Falls back to hardcoded defaults if the YAML file is missing or malformed —
  the system NEVER crashes because of a missing config file.
* ``get_war_room_rules()`` converts the YAML list-of-lists into the same
  ``(ticker, Direction, AssetClass, size, stop_pct, target_pct)`` tuple format
  that ``signal_generator.py`` already uses — a drop-in replacement.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# ── Default config path ───────────────────────────────────────────────────────

_DEFAULT_PATH = Path(__file__).parent.parent / "config" / "watchlist.yaml"

# ── Module-level cache  ───────────────────────────────────────────────────────

_CACHE: dict[str, Any] | None = None
_CACHE_PATH: str | None = None


# ── Defaults (fallback when YAML is missing or broken) ───────────────────────

_DEFAULT_VOL_PREMIUM: list[str] = ["SPY", "QQQ", "IWM", "HYG", "JNK"]

_DEFAULT_WAR_ROOM: dict[str, list] = {
    "CRISIS": [
        ("SPY",  "LONG_PUT",  "OPTION", 0.08, 0.30, 0.0),
        ("QQQ",  "LONG_PUT",  "OPTION", 0.05, 0.30, 0.0),
        ("HYG",  "LONG_PUT",  "OPTION", 0.05, 0.20, 0.0),
        ("JNK",  "LONG_PUT",  "OPTION", 0.05, 0.20, 0.0),
        ("GDX",  "LONG",      "ETF",    0.08, 0.12, 0.25),
        ("GLD",  "LONG",      "ETF",    0.05, 0.10, 0.20),
        ("XLF",  "LONG_PUT",  "OPTION", 0.04, 0.25, 0.0),
    ],
    "ELEVATED": [
        ("SPY",  "LONG_PUT",  "OPTION", 0.05, 0.25, 0.0),
        ("GDX",  "LONG",      "ETF",    0.06, 0.12, 0.20),
        ("GLD",  "LONG",      "ETF",    0.04, 0.10, 0.15),
        ("HYG",  "LONG_PUT",  "OPTION", 0.03, 0.20, 0.0),
    ],
    "WATCH": [
        ("GLD",  "LONG",      "ETF",    0.03, 0.08, 0.12),
        ("GDX",  "LONG",      "ETF",    0.02, 0.10, 0.15),
    ],
    "CALM": [],
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_vol_premium_tickers(path: str | Path | None = None) -> list[str]:
    """Return the list of tickers to scan for the volatility-premium strategy.

    Args:
        path: Optional override path to ``watchlist.yaml``.  Defaults to
              ``config/watchlist.yaml`` at the project root.

    Returns:
        List of uppercase ticker strings.  Falls back to
        ``["SPY", "QQQ", "IWM", "HYG", "JNK"]`` if YAML is unavailable.
    """
    data = _load(path)
    tickers = data.get("vol_premium", [])
    if not isinstance(tickers, list) or not tickers:
        _log.warning("watchlist_vol_premium_missing_using_defaults")
        return list(_DEFAULT_VOL_PREMIUM)
    return [str(t).upper() for t in tickers]


def get_war_room_rules(regime: str, path: str | Path | None = None) -> list[tuple]:
    """Return the signal rules for a given War Room regime.

    Each element is a 6-tuple matching the format in ``signal_generator.py``::

        (ticker, Direction, AssetClass, size_fraction, stop_pct, target_pct)

    ``Direction`` and ``AssetClass`` are the enum objects from ``shared.signal``.

    Args:
        regime: One of ``"CRISIS"``, ``"ELEVATED"``, ``"WATCH"``, ``"CALM"``.
        path:   Optional override path to ``watchlist.yaml``.

    Returns:
        List of 6-tuples.  Falls back to hardcoded defaults per regime if YAML
        is unavailable or the regime key is absent.
    """
    from shared.signal import AssetClass, Direction  # noqa: PLC0415

    data = _load(path)
    war_room = data.get("war_room", {})
    regime_upper = regime.upper()

    raw_rules: list = war_room.get(regime_upper, None)
    if raw_rules is None:
        _log.warning(
            "watchlist_war_room_regime_missing_using_defaults",
            regime=regime_upper,
        )
        raw_rules = _DEFAULT_WAR_ROOM.get(regime_upper, [])

    if not isinstance(raw_rules, list):
        raw_rules = []

    return [_parse_rule(r, Direction, AssetClass) for r in raw_rules]


def reload(path: str | Path | None = None) -> None:
    """Force-reload the YAML from disk (clears the cache).

    Useful in tests or after editing ``watchlist.yaml`` at runtime.
    """
    global _CACHE, _CACHE_PATH
    _CACHE = None
    _CACHE_PATH = None
    _load(path)


# ── Internals ─────────────────────────────────────────────────────────────────

def _load(path: str | Path | None) -> dict[str, Any]:
    """Load and cache the watchlist YAML.  Returns {} on any failure."""
    global _CACHE, _CACHE_PATH

    resolved = str(path) if path else str(_DEFAULT_PATH)

    # Cache hit — same path as last call.
    if _CACHE is not None and _CACHE_PATH == resolved:
        return _CACHE

    try:
        import yaml  # noqa: PLC0415

        with open(resolved, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _CACHE = data
        _CACHE_PATH = resolved
        _log.debug("watchlist_loaded", path=resolved)
        return _CACHE

    except FileNotFoundError:
        _log.warning("watchlist_yaml_not_found", path=resolved)
    except Exception as exc:
        _log.error("watchlist_yaml_load_failed", path=resolved, error=str(exc))

    return {}


def _parse_rule(row: Any, Direction: type, AssetClass: type) -> tuple:
    """Convert one YAML list-row to a (ticker, Direction, AssetClass, size, stop, target) tuple.

    Accepts both:
      - A plain list/tuple: ["SPY", "LONG_PUT", "OPTION", 0.08, 0.30, 0.0]
      - A pre-built tuple: ("SPY", Direction.LONG_PUT, AssetClass.OPTION, ...)
    """
    if isinstance(row, (list, tuple)) and len(row) >= 6:
        ticker      = str(row[0]).upper()
        dir_val     = row[1]
        ac_val      = row[2]
        size        = float(row[3])
        stop_pct    = float(row[4])
        target_pct  = float(row[5])

        # Convert string enum names → enum objects if needed.
        if isinstance(dir_val, str):
            dir_val = Direction[dir_val.upper()]
        if isinstance(ac_val, str):
            ac_val = AssetClass[ac_val.upper()]

        return (ticker, dir_val, ac_val, size, stop_pct, target_pct)

    _log.warning("watchlist_rule_parse_failed", row=repr(row))
    return ()

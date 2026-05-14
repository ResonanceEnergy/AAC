"""strategies/signal_generator.py — War Room → TradeSignal conversion.

This is the primary signal source (Sprint 1).  It:
  1. Calls ``generate_mandate()`` from the War Room engine to get the
     current regime + arm allocations.
  2. Converts each arm action into a concrete ``TradeSignal`` with
     realistic entry/stop/target levels derived from live spot prices.
  3. Returns a list ordered by confidence (highest first).

The design is deliberately simple: one strategy, one path, deterministic
output.  No stubs, no pass-throughs — every line is load-bearing.

Sprint 2 will wire the output of ``generate_signals()`` into the IBKR
execution path.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from shared.signal import AssetClass, Direction, TradeSignal

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime → per-arm signal rules
# ---------------------------------------------------------------------------
# Each entry: (ticker, direction, asset_class, size_fraction, stop_pct, target_pct)
# stop_pct / target_pct are relative to entry price (0 = not defined).
# Size fraction is proportion of account to deploy to this position.

_REGIME_SIGNALS: dict[str, list[tuple]] = {
    "CRISIS": [
        # Iran/Oil arm — long puts on SPY/QQQ (already own puts, add to them)
        ("SPY",  Direction.LONG_PUT,  AssetClass.OPTION, 0.08, 0.30, 0.0),
        ("QQQ",  Direction.LONG_PUT,  AssetClass.OPTION, 0.05, 0.30, 0.0),
        # BDC/Credit arm — long puts on high-yield proxies
        ("HYG",  Direction.LONG_PUT,  AssetClass.OPTION, 0.05, 0.20, 0.0),
        ("JNK",  Direction.LONG_PUT,  AssetClass.OPTION, 0.05, 0.20, 0.0),
        # Gold/Metals arm — long miners ETF
        ("GDX",  Direction.LONG,      AssetClass.ETF,    0.08, 0.12, 0.25),
        ("GLD",  Direction.LONG,      AssetClass.ETF,    0.05, 0.10, 0.20),
        # Financials — long puts on banks
        ("XLF",  Direction.LONG_PUT,  AssetClass.OPTION, 0.04, 0.25, 0.0),
    ],
    "ELEVATED": [
        ("SPY",  Direction.LONG_PUT,  AssetClass.OPTION, 0.05, 0.25, 0.0),
        ("GDX",  Direction.LONG,      AssetClass.ETF,    0.06, 0.12, 0.20),
        ("GLD",  Direction.LONG,      AssetClass.ETF,    0.04, 0.10, 0.15),
        ("HYG",  Direction.LONG_PUT,  AssetClass.OPTION, 0.03, 0.20, 0.0),
    ],
    "WATCH": [
        ("GLD",  Direction.LONG,      AssetClass.ETF,    0.03, 0.08, 0.12),
        ("GDX",  Direction.LONG,      AssetClass.ETF,    0.02, 0.10, 0.15),
    ],
    "CALM": [
        # In calm regimes, hold cash — no new positions
    ],
}


def _get_regime_rules(regime: str) -> list:
    """Return signal rules for *regime* from watchlist YAML, with fallback.

    Tries ``shared.watchlist.get_war_room_rules()`` first (reads
    ``config/watchlist.yaml``).  Falls back to the hardcoded ``_REGIME_SIGNALS``
    dict if the module or file is unavailable — so this never raises.
    """
    try:
        from shared.watchlist import get_war_room_rules  # noqa: PLC0415
        rules = get_war_room_rules(regime)
        # filter out any empty tuples produced by parse failures
        return [r for r in rules if r]
    except Exception as exc:
        _log.warning("watchlist_fallback_to_hardcoded regime=%s error=%s", regime, exc)
        return _REGIME_SIGNALS.get(regime, [])


def generate_signals(
    indicators=None,
    live: bool = True,
    run_mc: bool = False,
) -> List[TradeSignal]:
    """Generate trade signals from the War Room engine.

    Args:
        indicators: Optional pre-built ``IndicatorState``.  If None, a fresh
                    one is created and optionally patched with live data.
        live:       Fetch live prices before scoring (default True).
        run_mc:     Run Monte Carlo simulation for mandate (slow, default off
                    here since we just need the regime).

    Returns:
        List of ``TradeSignal`` ordered by confidence descending.
        Empty list on any error (never raises).
    """
    try:
        from strategies.war_room_engine import (
            generate_mandate,
            get_spot_prices,
            IndicatorState,
        )
    except ImportError as exc:
        _log.error("War Room engine not available: %s", exc)
        return []

    try:
        ind = indicators or IndicatorState()
        mandate = generate_mandate(indicators=ind, live=live, run_mc=run_mc)
        spots = get_spot_prices()
    except Exception as exc:
        _log.error("Mandate generation failed: %s", exc)
        return []

    regime = mandate.regime
    composite = mandate.composite_score
    # Scale confidence: CRISIS=0.85, ELEVATED=0.65, WATCH=0.45, CALM=0.20
    _regime_conf = {"CRISIS": 0.85, "ELEVATED": 0.65, "WATCH": 0.45, "CALM": 0.20}
    base_confidence = _regime_conf.get(regime, 0.50)

    rules = _get_regime_rules(regime)
    signals: List[TradeSignal] = []

    for ticker, direction, asset_class, size, stop_pct, target_pct in rules:
        spot = spots.get(ticker.lower(), 0.0)
        if spot == 0.0:
            # Spot not in war room map — use 0 (market order, no stop/target calc)
            entry = 0.0
            stop = 0.0
            target = 0.0
        else:
            entry = spot
            stop = round(entry * (1 - stop_pct), 2) if stop_pct > 0 else 0.0
            target = round(entry * (1 + target_pct), 2) if target_pct > 0 else 0.0
            # For puts: stop is ABOVE entry (if underlying rises, put loses)
            if direction in (Direction.LONG_PUT,):
                stop = round(entry * (1 + stop_pct), 2) if stop_pct > 0 else 0.0
                target = 0.0  # target = expiry worthless on the underlying put

        # Slight per-asset confidence adjustment (±0.05) based on composite
        conf = min(1.0, max(0.0, base_confidence + (composite - 50) / 1000))

        signals.append(TradeSignal(
            ticker=ticker,
            direction=direction,
            confidence=round(conf, 3),
            entry=entry,
            stop=stop,
            target=target,
            size=size,
            strategy="war_room_engine",
            regime=regime,
            asset_class=asset_class,
            notes=f"Composite={composite:.1f} | Regime={regime}",
        ))

    # Sort by confidence descending
    signals.sort(key=lambda s: s.confidence, reverse=True)

    _log.info(
        "Signal generation complete: regime=%s composite=%.1f signals=%d",
        regime, composite, len(signals),
    )
    return signals

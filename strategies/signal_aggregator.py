from __future__ import annotations
"""strategies/signal_aggregator.py — Multi-strategy signal combiner (Sprint 6).

Merges TradeSignal lists from multiple named strategy sources into a single
ranked output.  When two or more strategies agree on the same
(ticker, direction) pair, their confidence scores are combined via weighted
average and boosted by a conviction multiplier to reward corroboration.

Primary entry point: ``get_combined_signals()``
  → runs War Room (weight 0.60) + Vol Premium (weight 0.40)
  → returns List[TradeSignal] ordered by combined confidence descending
"""

import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from shared.signal import TradeSignal

_log = logging.getLogger(__name__)

# Confidence boost per additional agreeing source (beyond the first)
_AGREEMENT_BOOST = 0.05
_MAX_CONFIDENCE = 0.95


# ── Internal data class ──────────────────────────────────────────────────────

@dataclass
class AggregatedSignal:
    """Signal after aggregation, with source metadata."""

    signal: TradeSignal
    sources: list[str]       # strategy names that contributed
    raw_confidence: float    # pre-boost weighted average
    boosted: bool            # True when conviction multiplier applied


# ── Core aggregation ─────────────────────────────────────────────────────────

def _signal_key(s: TradeSignal) -> str:
    return f"{s.ticker}:{s.direction.value}"


def aggregate(
    signal_lists: list[tuple[list[TradeSignal], float]],
) -> list[AggregatedSignal]:
    """Combine weighted signal lists into a single ranked output.

    Args:
        signal_lists: List of ``(signals, weight)`` pairs. ``weight`` is the
                      importance of that strategy (positive float, need not sum
                      to 1 — normalised internally per group).

    Returns:
        List[AggregatedSignal] ordered by final confidence descending.
    """
    # Group by (ticker, direction) key
    groups: dict[str, list[tuple[TradeSignal, float]]] = defaultdict(list)
    for signals, weight in signal_lists:
        for s in signals:
            groups[_signal_key(s)].append((s, weight))

    results: list[AggregatedSignal] = []

    for items in groups.values():
        if not items:
            continue

        total_weight = sum(w for _, w in items)
        if total_weight == 0.0:
            continue

        avg_conf = sum(s.confidence * w for s, w in items) / total_weight
        n_sources = len(items)
        boost = (n_sources - 1) * _AGREEMENT_BOOST
        final_conf = min(_MAX_CONFIDENCE, avg_conf + boost)

        # Use the highest-confidence signal as the structural base
        best = max(items, key=lambda x: x[0].confidence)[0]
        source_names = [s.strategy for s, _ in items]
        merged = dataclasses.replace(
            best,
            confidence=round(final_conf, 3),
            strategy="aggregator",
            notes=f"[{'+'.join(source_names)}] {best.notes}",
        )

        results.append(AggregatedSignal(
            signal=merged,
            sources=source_names,
            raw_confidence=round(avg_conf, 3),
            boosted=n_sources > 1,
        ))

    results.sort(key=lambda r: r.signal.confidence, reverse=True)
    return results


# ── High-level entry point ────────────────────────────────────────────────────

def get_combined_signals(
    war_room_kwargs: dict[str, Any] | None = None,
    vol_premium_kwargs: dict[str, Any] | None = None,
    war_room_weight: float = 0.60,
    vol_premium_weight: float = 0.40,
    use_calibration: bool = False,
) -> list[TradeSignal]:
    """Run both strategies and return a merged, ranked signal list.

    Both strategies use yfinance and env-key APIs only — no external service
    is required beyond what is already configured.

    Args:
        war_room_kwargs:    Keyword args forwarded to ``generate_signals()``.
        vol_premium_kwargs: Keyword args forwarded to
                            ``generate_vol_premium_signals()``.
        war_room_weight:    Importance weight for War Room (default 0.60).
        vol_premium_weight: Importance weight for Vol Premium (default 0.40).
        use_calibration:    When True, attempt to load calibrated weights from
                            the signal journal before falling back to defaults.

    Returns:
        List[TradeSignal] from the merged output, ordered by confidence.
    """
    from strategies.signal_generator import generate_signals  # noqa: PLC0415
    from strategies.vol_premium_signals import generate_vol_premium_signals  # noqa: PLC0415

    if use_calibration:
        try:
            from strategies.signal_outcome_tracker import SignalOutcomeTracker  # noqa: PLC0415
            tracker = SignalOutcomeTracker()
            cal = tracker.calibrated_weights(
                default_war_room=war_room_weight,
                default_vol_premium=vol_premium_weight,
            )
            if cal.calibrated:
                war_room_weight = cal.war_room
                vol_premium_weight = cal.vol_premium
                _log.info(
                    "Using calibrated weights: war_room=%.3f vol_premium=%.3f",
                    war_room_weight, vol_premium_weight,
                )
        except Exception as exc:
            _log.warning("calibration_load_failed_using_defaults: %s", exc)

    wr_signals = generate_signals(**(war_room_kwargs or {}))
    vp_signals = generate_vol_premium_signals(**(vol_premium_kwargs or {}))

    _log.info(
        "Signal sources: war_room=%d vol_premium=%d",
        len(wr_signals), len(vp_signals),
    )

    aggregated = aggregate([
        (wr_signals, war_room_weight),
        (vp_signals, vol_premium_weight),
    ])

    combined = [a.signal for a in aggregated]
    _log.info("Combined signals: %d (boosted=%d)", len(combined), sum(a.boosted for a in aggregated))
    return combined

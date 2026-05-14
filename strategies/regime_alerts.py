from __future__ import annotations

"""regime_alerts.py — Sprint 16: Regime Change Alerting.

Utility functions that act on a ``RegimeTransition``:

* ``is_escalation`` / ``is_de_escalation`` — direction helpers
* ``confidence_multiplier`` — how much to scale signal confidence
* ``format_alert`` — human-readable log string
* ``apply_regime_filter`` — adjust signal confidences in place
"""

import dataclasses
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# ── Severity map (duplicated from Regime enum to avoid circular import) ────

_SEVERITY: dict[str, int] = {
    "CALM": 0,
    "WATCH": 1,
    "ELEVATED": 2,
    "CRISIS": 3,
}

# ── Confidence multipliers ─────────────────────────────────────────────────

_ESCALATION_MULTIPLIER: float = 1.20    # regime worsened → boost short confidence
_DE_ESCALATION_MULTIPLIER: float = 0.85  # regime improved → dampen confidence
_NEUTRAL_MULTIPLIER: float = 1.00
_MAX_CONFIDENCE: float = 0.95


# ── Direction helpers ──────────────────────────────────────────────────────


def is_escalation(transition: Any) -> bool:
    """True when the new regime is more severe than the previous one."""
    return (
        _SEVERITY.get(str(transition.new_regime), 0)
        > _SEVERITY.get(str(transition.prev_regime), 0)
    )


def is_de_escalation(transition: Any) -> bool:
    """True when the new regime is less severe than the previous one."""
    return (
        _SEVERITY.get(str(transition.new_regime), 0)
        < _SEVERITY.get(str(transition.prev_regime), 0)
    )


def confidence_multiplier(transition: Any) -> float:
    """Return the confidence multiplier for this transition.

    * Escalation  → 1.20 (boost bearish confidence on stress increase)
    * De-escalation → 0.85 (dampen confidence on stress relief)
    * Lateral     → 1.00 (no change)
    """
    if is_escalation(transition):
        return _ESCALATION_MULTIPLIER
    if is_de_escalation(transition):
        return _DE_ESCALATION_MULTIPLIER
    return _NEUTRAL_MULTIPLIER


# ── Formatting ─────────────────────────────────────────────────────────────


def format_alert(transition: Any) -> str:
    """Return a human-readable one-line description of the transition."""
    direction = "ESCALATION" if is_escalation(transition) else "DE-ESCALATION"
    arrow = "↑" if is_escalation(transition) else "↓"
    return (
        f"REGIME {arrow} {direction}: "
        f"{transition.prev_regime} → {transition.new_regime} "
        f"(score={transition.composite_score:.1f}) at {transition.detected_at}"
    )


# ── Signal adjustment ──────────────────────────────────────────────────────


def apply_regime_filter(signals: list[Any], transition: Any) -> list[Any]:
    """Return a new signal list with confidence values adjusted for the regime change.

    Uses ``dataclasses.replace()`` so the original signals are not mutated.
    Adjusted confidence is capped at ``_MAX_CONFIDENCE`` (0.95).
    If the multiplier is 1.0 the original list is returned unchanged.
    Any signal that cannot be adjusted is passed through unmodified.
    """
    multiplier = confidence_multiplier(transition)
    if abs(multiplier - _NEUTRAL_MULTIPLIER) < 1e-9:
        return signals

    adjusted: list[Any] = []
    for sig in signals:
        try:
            new_conf = min(sig.confidence * multiplier, _MAX_CONFIDENCE)
            adjusted.append(dataclasses.replace(sig, confidence=new_conf))
        except Exception as exc:  # noqa: BLE001
            _log.warning("regime_filter_error", ticker=getattr(sig, "ticker", "?"), error=str(exc))
            adjusted.append(sig)
    return adjusted

"""Alpha Extraction & Signal Combination Engine.

Implements the unified framework from the working paper:
  1. Demean raw returns → remove drift
  2. Variance estimation → standardisation
  3. Normalisation → consistent scaling
  4. Time-series truncation → observation window
  5. Cross-sectional demeaning → remove market-wide bias
  6. Expected-return estimation over period *d*
  7. Volatility-normalised expected returns
  8. Residual extraction — orthogonalise against common factors
  9. Inverse-volatility signal weighting
 10. Absolute-weight normalisation → sum |w| = 1
 11. Combined signal = Σ w(i)·S(i)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SignalSeries:
    """A named time-series of raw signal values.

    Parameters
    ----------
    name : str
        Human-readable signal name (e.g. ``"youtube_sentiment"``).
    values : list[float]
        Chronologically ordered observations.  Index 0 is oldest.
    """

    name: str
    values: list[float] = field(default_factory=list)


@dataclass
class AlphaResult:
    """Output of the alpha extraction pipeline.

    Attributes
    ----------
    weights : dict[str, float]
        Signal name → normalised weight (sum of absolute values = 1).
    combined_signal : float
        Final combined signal value.
    residuals : dict[str, float]
        Per-signal alpha residual after factor orthogonalisation.
    volatilities : dict[str, float]
        Per-signal estimated volatility.
    expected_returns : dict[str, float]
        Expected return (mean of last *d* observations) per signal.
    details : dict[str, Any]
        Internal diagnostics for debugging / persistence.
    """

    weights: dict[str, float] = field(default_factory=dict)
    combined_signal: float = 0.0
    residuals: dict[str, float] = field(default_factory=dict)
    volatilities: dict[str, float] = field(default_factory=dict)
    expected_returns: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CORE PIPELINE
# ============================================================================

_MIN_VOLATILITY = 1e-9  # floor to prevent division by zero


def _demean(values: list[float]) -> list[float]:
    """Eq (1): X(i,s) = R(i,s) − (1/M)·Σ R(i,s)."""
    if not values:
        return []
    mu = sum(values) / len(values)
    return [v - mu for v in values]


def _variance(demeaned: list[float]) -> float:
    """Eq (2): σ²_i = (1/M)·Σ X(i,s)²."""
    if not demeaned:
        return 0.0
    return sum(x * x for x in demeaned) / len(demeaned)


def _normalise(demeaned: list[float], sigma: float) -> list[float]:
    """Eq (3): Y(i,s) = X(i,s) / σ_i."""
    if sigma < _MIN_VOLATILITY:
        return [0.0] * len(demeaned)
    return [x / sigma for x in demeaned]


def _truncate(series: list[float], keep: int) -> list[float]:
    """Eq (4): keep the last *keep* observations (M−1 periods)."""
    if keep <= 0:
        return []
    return series[-keep:]


def _cross_sectional_demean(
    normalised: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Eq (5): Λ(i,s) = Y(i,s) − (1/N)·Σ_j Y(j,s).

    Removes the market-wide average from each time-step, isolating
    idiosyncratic signal content.
    """
    names = list(normalised)
    if not names:
        return {}

    # All series should be the same length after truncation.
    length = min(len(normalised[n]) for n in names)
    if length == 0:
        return {n: [] for n in names}

    n_signals = len(names)
    result: dict[str, list[float]] = {n: [] for n in names}

    for s in range(length):
        cross_mean = sum(normalised[n][s] for n in names) / n_signals
        for n in names:
            result[n].append(normalised[n][s] - cross_mean)

    return result


def _expected_return(raw_values: list[float], d: int) -> float:
    """Eq (7): E(i) = (1/d)·Σ_{s=M−d+1}^{M} R(i,s).

    Average of the last *d* raw observations.
    """
    if d <= 0 or not raw_values:
        return 0.0
    window = raw_values[-d:]
    return sum(window) / len(window)


def _orthogonalise(
    e_norm: dict[str, float],
    factors: dict[str, list[float]],
) -> dict[str, float]:
    """Eq (9): extract residual ε(i) by regressing E_norm against factors.

    Uses simple OLS projection per signal.  When the factor matrix has
    zero variance the raw E_norm is returned (no projection possible).

    Parameters
    ----------
    e_norm : dict[str, float]
        Volatility-normalised expected returns per signal.
    factors : dict[str, list[float]]
        Cross-sectionally demeaned factor series Λ(i,·).

    Returns
    -------
    dict[str, float]
        Per-signal residual ε(i).
    """
    names = list(e_norm)
    if not names:
        return {}

    # Build a simple factor: the cross-sectional mean across all signals
    # at each time-step (market factor).
    lengths = [len(factors.get(n, [])) for n in names]
    if not lengths or max(lengths) == 0:
        return dict(e_norm)

    factor_len = min(lengths)
    if factor_len == 0:
        return dict(e_norm)

    # Market factor = average Λ across signals per time-step.
    n_signals = len(names)
    market_factor = [
        sum(factors[n][s] for n in names) / n_signals
        for s in range(factor_len)
    ]

    # Factor variance (for denominator of OLS β).
    factor_var = sum(f * f for f in market_factor) / factor_len
    if factor_var < _MIN_VOLATILITY:
        # All factors are zero — nothing to orthogonalise against.
        return dict(e_norm)

    residuals: dict[str, float] = {}
    for n in names:
        s_values = factors.get(n, [])
        if len(s_values) < factor_len:
            residuals[n] = e_norm[n]
            continue

        # β = cov(signal, factor) / var(factor)
        cov = sum(s_values[s] * market_factor[s]
                  for s in range(factor_len)) / factor_len
        beta = cov / factor_var

        # ε = E_norm − β·⟨factor⟩
        factor_mean = sum(market_factor) / factor_len
        residuals[n] = e_norm[n] - beta * factor_mean

    return residuals


def _compute_weights(
    residuals: dict[str, float],
    volatilities: dict[str, float],
) -> dict[str, float]:
    """Eq (10–11): w(i) = η · ε(i)/σ_i, with Σ|w| = 1.

    Inverse-volatility weighting of residuals, normalised to unity.
    """
    raw: dict[str, float] = {}
    for name in residuals:
        sigma = max(volatilities.get(name, 1.0), _MIN_VOLATILITY)
        raw[name] = residuals[name] / sigma

    abs_sum = sum(abs(v) for v in raw.values())
    if abs_sum < _MIN_VOLATILITY:
        # All residuals are zero → equal weight.
        n = len(raw) or 1
        return {name: 1.0 / n for name in raw}

    eta = 1.0 / abs_sum
    return {name: raw[name] * eta for name in raw}


def _combined_signal(
    weights: dict[str, float],
    latest_values: dict[str, float],
) -> float:
    """Eq (12): Combined = Σ w(i)·S(i)."""
    return sum(weights.get(n, 0.0) * latest_values.get(n, 0.0)
               for n in weights)


# ============================================================================
# PUBLIC API
# ============================================================================

def extract_alpha(
    signals: list[SignalSeries],
    *,
    estimation_period: int = 5,
    truncation_window: int | None = None,
) -> AlphaResult:
    """Run the complete alpha extraction and signal combination pipeline.

    Parameters
    ----------
    signals : list[SignalSeries]
        One or more named signal series (raw observations, oldest first).
    estimation_period : int
        Number of recent observations to average for expected-return
        estimation (parameter *d* in eq 7).  Default 5.
    truncation_window : int | None
        If given, truncate each series to the last *truncation_window*
        observations before processing.  If ``None``, use ``M − 1``
        (full length minus one).

    Returns
    -------
    AlphaResult
        Weights, combined signal, diagnostics.
    """
    if not signals:
        return AlphaResult(details={"error": "no signals provided"})

    # Filter out empty series.
    signals = [s for s in signals if s.values]
    if not signals:
        return AlphaResult(details={"error": "all signals empty"})

    # ── Step 1-3: demean, variance, normalise per signal ───────────
    demeaned: dict[str, list[float]] = {}
    sigmas: dict[str, float] = {}
    normalised: dict[str, list[float]] = {}

    for sig in signals:
        dm = _demean(sig.values)
        var = _variance(dm)
        sigma = math.sqrt(max(var, 0.0))
        sigma = max(sigma, _MIN_VOLATILITY)

        demeaned[sig.name] = dm
        sigmas[sig.name] = sigma
        normalised[sig.name] = _normalise(dm, sigma)

    # ── Step 4: truncation ─────────────────────────────────────────
    for name in normalised:
        m = len(normalised[name])
        keep = truncation_window if truncation_window is not None else max(m - 1, 1)
        normalised[name] = _truncate(normalised[name], keep)

    # ── Step 5: cross-sectional demeaning ──────────────────────────
    factors = _cross_sectional_demean(normalised)

    # ── Step 6: final trimming (uniform length) ────────────────────
    min_len = min((len(f) for f in factors.values()), default=0)
    for name in factors:
        factors[name] = factors[name][-min_len:] if min_len > 0 else []

    # ── Step 7-8: expected returns, vol-normalised ─────────────────
    e_raw: dict[str, float] = {}
    e_norm: dict[str, float] = {}

    for sig in signals:
        er = _expected_return(sig.values, estimation_period)
        e_raw[sig.name] = er
        sigma = sigmas[sig.name]
        e_norm[sig.name] = er / sigma if sigma > _MIN_VOLATILITY else 0.0

    # ── Step 9: orthogonalise ──────────────────────────────────────
    residuals = _orthogonalise(e_norm, factors)

    # ── Step 10: inverse-vol weights, normalise ────────────────────
    weights = _compute_weights(residuals, sigmas)

    # ── Step 11-12: latest signal values → combined signal ─────────
    latest: dict[str, float] = {
        sig.name: sig.values[-1] for sig in signals
    }
    combined = _combined_signal(weights, latest)

    return AlphaResult(
        weights=weights,
        combined_signal=combined,
        residuals=residuals,
        volatilities=sigmas,
        expected_returns=e_raw,
        details={
            "n_signals": len(signals),
            "estimation_period": estimation_period,
            "series_lengths": {s.name: len(s.values) for s in signals},
        },
    )


# ============================================================================
# COUNCIL ADAPTER — maps AAC council history to SignalSeries
# ============================================================================

# In-memory rolling buffer for council signal history.
# Keyed by signal name → list of float observations.
_signal_history: dict[str, list[float]] = {}

_MAX_HISTORY = 200  # rolling window cap


def record_council_observation(name: str, value: float) -> None:
    """Append a new observation for a council signal channel.

    Called from ``apply_council_to_indicators`` each cycle to build up
    the time-series that ``extract_alpha`` needs.
    """
    buf = _signal_history.setdefault(name, [])
    buf.append(value)
    if len(buf) > _MAX_HISTORY:
        _signal_history[name] = buf[-_MAX_HISTORY:]


def get_signal_history(name: str) -> list[float]:
    """Return the recorded history for *name* (oldest first)."""
    return list(_signal_history.get(name, []))


def clear_signal_history() -> None:
    """Reset all signal history (tests / restarts)."""
    _signal_history.clear()


def alpha_combine_councils(
    *,
    estimation_period: int = 5,
    min_observations: int = 3,
) -> AlphaResult | None:
    """Build ``SignalSeries`` from recorded council history and run the pipeline.

    Returns ``None`` when insufficient history is available (cold-start).
    """
    series: list[SignalSeries] = []
    for name, values in _signal_history.items():
        if len(values) >= min_observations:
            series.append(SignalSeries(name=name, values=list(values)))

    if len(series) < 2:
        # Need at least 2 signals for cross-sectional demeaning.
        return None

    return extract_alpha(series, estimation_period=estimation_period)

"""
Storm Lifeboat Matrix — PlanckPhire Coherence Model
=====================================================
Golden-ratio harmonic analysis based on Dan Winter's PlanckPhire framework.

The core insight: biological and financial systems exhibit maximum coherence
(order, predictability, power) when their oscillation frequencies nest in
golden-ratio (phi = 1.618...) proportions. Incoherence (chaos, disorder)
manifests when harmonics become dissonant.

Applied to markets:
    - Price oscillations across multiple timeframes (daily, weekly, monthly)
    - When shorter-term oscillations are phi-proportional to longer-term ones,
      the market is "coherent" — trends are strong, breakouts are real
    - When harmonics are dissonant, the market is "incoherent" — choppy,
      false breakouts, mean-reversion dominant

Coherence score (0-1):
    0.0 = Maximum incoherence (chaos, all systems dissonant)
    0.5 = Neutral (mixed signals)
    1.0 = Maximum coherence (all harmonic ratios near phi)

Inputs:
    - Price returns at multiple timeframes
    - Scenario alignment (how many scenarios point the same direction)
    - Lunar phi alignment (from lunar_phi engine)
    - Volatility ratio analysis

This is NOT mystical — it's a systematic way to measure multi-timeframe
trend alignment using a specific mathematical constant (phi) as the
ideal harmonic ratio.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from strategies.storm_lifeboat.core import (
    Asset,
    MoonPhase,
    ScenarioDefinition,
    ScenarioStatus,
    VolRegime,
)

logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2  # 1.6180339887...
PHI_SQ = PHI ** 2              # 2.6180339887...
INV_PHI = 1 / PHI             # 0.6180339887...


@dataclass
class CoherenceResult:
    """Output of the PlanckPhire coherence analysis."""
    overall_score: float           # 0-1 composite coherence
    harmonic_ratio: float          # How close multi-TF oscillations are to phi
    scenario_alignment: float      # 0-1, how aligned active scenarios are
    lunar_alignment: float         # 0-1, phase/phi window alignment
    regime_stability: float        # 0-1, how stable the current regime is
    dominant_frequency: str        # "bullish_coherent", "bearish_coherent", "chaotic"
    confidence: float              # 0-1, confidence in the coherence reading


def _phi_proximity(ratio: float) -> float:
    """Measure how close a ratio is to phi, phi^2, or 1/phi.

    Returns 0-1 where 1 = exact phi match.
    """
    targets = [PHI, PHI_SQ, INV_PHI, 1.0 / PHI_SQ]
    min_dist = min(abs(ratio - t) for t in targets)
    # Exponential decay: score = exp(-distance * 3)
    return math.exp(-min_dist * 3.0)


def compute_harmonic_coherence(
    returns_daily: List[float],
    returns_weekly: List[float],
    returns_monthly: List[float],
) -> float:
    """Compute harmonic coherence from multi-timeframe returns.

    Measures whether the ratio of oscillation amplitudes across
    timeframes approximates golden ratio proportions.

    Args:
        returns_daily: Last 5-10 daily returns
        returns_weekly: Last 4-8 weekly returns
        returns_monthly: Last 3-6 monthly returns

    Returns:
        Harmonic coherence score (0-1)
    """
    if not returns_daily or not returns_weekly or not returns_monthly:
        return 0.5  # Neutral if insufficient data

    # Compute amplitude (standard deviation) at each timeframe
    def _std(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.01
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return max(math.sqrt(var), 0.001)

    amp_d = _std(returns_daily)
    amp_w = _std(returns_weekly)
    amp_m = _std(returns_monthly)

    # Ratios between timeframe amplitudes
    ratio_w_d = amp_w / amp_d if amp_d > 0 else 1.0
    ratio_m_w = amp_m / amp_w if amp_w > 0 else 1.0
    ratio_m_d = amp_m / amp_d if amp_d > 0 else 1.0

    # Check how close each ratio is to a phi proportion
    score_1 = _phi_proximity(ratio_w_d)
    score_2 = _phi_proximity(ratio_m_w)
    score_3 = _phi_proximity(ratio_m_d)

    # Direction alignment: are all timeframes trending the same way?
    mean_d = sum(returns_daily) / len(returns_daily)
    mean_w = sum(returns_weekly) / len(returns_weekly)
    mean_m = sum(returns_monthly) / len(returns_monthly)

    same_direction = (
        (mean_d > 0 and mean_w > 0 and mean_m > 0)
        or (mean_d < 0 and mean_w < 0 and mean_m < 0)
    )
    direction_bonus = 0.15 if same_direction else 0.0

    harmonic = (score_1 + score_2 + score_3) / 3.0 + direction_bonus
    return min(1.0, max(0.0, harmonic))


def compute_scenario_alignment(
    active_scenarios: List[ScenarioDefinition],
) -> float:
    """Measure how aligned active scenarios are in direction.

    If all active scenarios point in the same direction (all bearish
    for equities, all bullish for commodities), alignment is high.
    Mixed signals = low alignment.
    """
    if not active_scenarios:
        return 0.5

    # Count net beneficiary/victim tilts
    beneficiary_count: Dict[Asset, int] = {}
    victim_count: Dict[Asset, int] = {}

    for sc in active_scenarios:
        weight = sc.probability * sc.impact_severity
        for a in sc.beneficiary_assets:
            beneficiary_count[a] = beneficiary_count.get(a, 0) + 1
        for a in sc.victim_assets:
            victim_count[a] = victim_count.get(a, 0) + 1

    # For each asset, check if scenarios agree on direction
    agreements = 0
    disagreements = 0
    for a in set(list(beneficiary_count.keys()) + list(victim_count.keys())):
        b = beneficiary_count.get(a, 0)
        v = victim_count.get(a, 0)
        if b > 0 and v > 0:
            disagreements += min(b, v)
            agreements += abs(b - v)
        else:
            agreements += b + v

    total = agreements + disagreements
    if total == 0:
        return 0.5

    return min(1.0, agreements / total)


def compute_regime_stability(
    current_regime: VolRegime,
    recent_regimes: List[VolRegime],
) -> float:
    """Measure how stable the current regime has been.

    If the regime has been consistent for the last N observations,
    stability is high. Frequent regime changes = low stability.
    """
    if not recent_regimes:
        return 0.5

    same_count = sum(1 for r in recent_regimes if r == current_regime)
    return same_count / len(recent_regimes)


class CoherenceEngine:
    """PlanckPhire harmonic coherence analyzer.

    Combines multi-timeframe harmonic ratios, scenario alignment,
    lunar phi position, and regime stability into a single
    coherence score that modulates the trading mandate.
    """

    def analyze(
        self,
        returns_daily: Optional[List[float]] = None,
        returns_weekly: Optional[List[float]] = None,
        returns_monthly: Optional[List[float]] = None,
        active_scenarios: Optional[List[ScenarioDefinition]] = None,
        moon_phase: MoonPhase = MoonPhase.NEW,
        lunar_phi_coherence: float = 0.5,
        current_regime: VolRegime = VolRegime.CRISIS,
        recent_regimes: Optional[List[VolRegime]] = None,
    ) -> CoherenceResult:
        """Run full PlanckPhire coherence analysis.

        Returns a CoherenceResult with scores across all dimensions.
        """
        # 1. Harmonic ratio from price data
        harmonic = compute_harmonic_coherence(
            returns_daily or [],
            returns_weekly or [],
            returns_monthly or [],
        )

        # 2. Scenario alignment
        scenario_align = compute_scenario_alignment(active_scenarios or [])

        # 3. Lunar alignment — FULL phase during phi window = max alignment
        lunar_base = {
            MoonPhase.NEW: 0.30,
            MoonPhase.WAXING: 0.60,
            MoonPhase.FULL: 0.90,
            MoonPhase.WANING: 0.50,
        }[moon_phase]
        lunar_score = lunar_base * (0.5 + 0.5 * lunar_phi_coherence)

        # 4. Regime stability
        regime_stab = compute_regime_stability(
            current_regime,
            recent_regimes or [current_regime] * 5,
        )

        # Composite: weighted average
        weights = (0.35, 0.25, 0.20, 0.20)  # harmonic, scenario, lunar, regime
        overall = (
            weights[0] * harmonic
            + weights[1] * scenario_align
            + weights[2] * lunar_score
            + weights[3] * regime_stab
        )
        overall = min(1.0, max(0.0, overall))

        # Determine dominant frequency
        if overall > 0.7:
            # Check direction from scenario alignment
            if active_scenarios:
                # Count bearish vs bullish scenarios
                bear_weight = sum(
                    s.probability for s in active_scenarios
                    if Asset.SPY in s.victim_assets
                )
                bull_weight = sum(
                    s.probability for s in active_scenarios
                    if Asset.SPY in s.beneficiary_assets
                )
                freq = "bearish_coherent" if bear_weight > bull_weight else "bullish_coherent"
            else:
                freq = "bullish_coherent"
        elif overall < 0.3:
            freq = "chaotic"
        else:
            freq = "mixed"

        # Confidence: higher when we have more data
        data_count = sum([
            1 if returns_daily else 0,
            1 if returns_weekly else 0,
            1 if returns_monthly else 0,
            1 if active_scenarios else 0,
            1 if recent_regimes else 0,
        ])
        confidence = data_count / 5.0

        return CoherenceResult(
            overall_score=round(overall, 4),
            harmonic_ratio=round(harmonic, 4),
            scenario_alignment=round(scenario_align, 4),
            lunar_alignment=round(lunar_score, 4),
            regime_stability=round(regime_stab, 4),
            dominant_frequency=freq,
            confidence=round(confidence, 2),
        )

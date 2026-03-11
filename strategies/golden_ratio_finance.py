"""
Golden Ratio Finance Module — Dan Winter Implementation
========================================================

Applies golden ratio (φ = 1.618...) and Fibonacci mathematics to
financial analysis, consistent with BARREN WUFFET's doctrine on
sacred geometry in market structures.

Features:
- Fibonacci retracement/extension levels
- Golden spiral price targets
- Harmonic pattern detection (Gartley, Butterfly, Bat, Crab)
- Phase conjugation analysis for trend confluence
- Dan Winter's fractal compression theory applied to volatility

Referenced in: aac/doctrine/RESEARCH.md § Dan Winter & Golden Ratio Finance
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

PHI = (1 + math.sqrt(5)) / 2          # 1.6180339887...
PHI_INVERSE = 1 / PHI                 # 0.6180339887...
PHI_SQUARED = PHI ** 2                # 2.6180339887...
PHI_CUBED = PHI ** 3                  # 4.2360679775...
SQRT_PHI = math.sqrt(PHI)             # 1.2720196495...

# Standard Fibonacci retracement levels
FIB_RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]

# Extensions
FIB_EXTENSION_LEVELS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.414, 2.618, 3.618, 4.236]


@dataclass
class FibLevel:
    """A single Fibonacci level with price."""
    ratio: float
    price: float
    label: str


@dataclass
class FibonacciResult:
    """Complete Fibonacci analysis result."""
    swing_high: float
    swing_low: float
    direction: str  # 'up' or 'down'
    retracements: List[FibLevel] = field(default_factory=list)
    extensions: List[FibLevel] = field(default_factory=list)


@dataclass
class HarmonicPattern:
    """Detected harmonic pattern."""
    pattern_type: str  # 'gartley', 'butterfly', 'bat', 'crab', 'shark'
    direction: str     # 'bullish' or 'bearish'
    completion_zone: Tuple[float, float]  # (low, high) of PRZ
    points: Dict[str, float]  # X, A, B, C, D price points
    confidence: float  # 0.0 to 1.0


# ═══════════════════════════════════════════════════════════════════════════
# FIBONACCI CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════


class FibonacciCalculator:
    """
    Calculate Fibonacci retracements, extensions, and harmonic patterns.

    Usage:
        fib = FibonacciCalculator()
        result = fib.retracement(high=50000, low=40000, direction='up')
        harmonics = fib.detect_harmonics(prices=[...])
    """

    def retracement(
        self,
        high: float,
        low: float,
        direction: str = "up",
        custom_levels: Optional[List[float]] = None,
    ) -> FibonacciResult:
        """
        Calculate Fibonacci retracement levels.

        Args:
            high: Swing high price
            low: Swing low price
            direction: 'up' (retracing down from high) or 'down' (retracing up from low)
            custom_levels: Override default Fib levels
        """
        levels = custom_levels or FIB_RETRACEMENT_LEVELS
        diff = high - low
        retracements = []

        for ratio in levels:
            if direction == "up":
                price = high - (diff * ratio)
            else:
                price = low + (diff * ratio)
            retracements.append(FibLevel(
                ratio=ratio,
                price=round(price, 6),
                label=f"{ratio * 100:.1f}%",
            ))

        return FibonacciResult(
            swing_high=high,
            swing_low=low,
            direction=direction,
            retracements=retracements,
        )

    def extension(
        self,
        high: float,
        low: float,
        retracement_point: float,
        custom_levels: Optional[List[float]] = None,
    ) -> FibonacciResult:
        """
        Calculate Fibonacci extension levels from A-B-C pattern.

        Args:
            high: Point A (swing high in downtrend, swing low in uptrend)
            low: Point B (swing low in downtrend, swing high in uptrend)
            retracement_point: Point C (retracement end)
        """
        levels = custom_levels or FIB_EXTENSION_LEVELS
        impulse = abs(high - low)
        extensions = []

        for ratio in levels:
            price = retracement_point + (impulse * ratio) if high > low else retracement_point - (impulse * ratio)
            extensions.append(FibLevel(
                ratio=ratio,
                price=round(price, 6),
                label=f"{ratio * 100:.1f}%",
            ))

        return FibonacciResult(
            swing_high=high,
            swing_low=low,
            direction="up" if high > low else "down",
            extensions=extensions,
        )

    def golden_spiral_targets(
        self,
        center_price: float,
        volatility: float,
        n_levels: int = 8,
    ) -> List[FibLevel]:
        """
        Generate price targets based on golden spiral expansion.

        Each level expands by PHI from the center, scaled by volatility.
        """
        targets = []
        for i in range(1, n_levels + 1):
            expansion = volatility * (PHI ** i)
            targets.append(FibLevel(
                ratio=PHI ** i,
                price=round(center_price + expansion, 6),
                label=f"φ^{i} up",
            ))
            targets.append(FibLevel(
                ratio=PHI ** i,
                price=round(center_price - expansion, 6),
                label=f"φ^{i} down",
            ))
        return targets

    # ── Harmonic Pattern Detection ─────────────────────────────────────

    def detect_harmonics(
        self,
        prices: List[float],
        tolerance: float = 0.05,
    ) -> List[HarmonicPattern]:
        """
        Detect harmonic patterns (Gartley, Butterfly, Bat, Crab) in price data.

        Args:
            prices: List of prices (typically swing points, not raw OHLCV)
            tolerance: Ratio tolerance for pattern matching (default 5%)

        Returns:
            List of detected HarmonicPattern instances.
        """
        patterns = []
        if len(prices) < 5:
            return patterns

        # Slide window of 5 swing points (X, A, B, C, D)
        for i in range(len(prices) - 4):
            x, a, b, c, d = prices[i : i + 5]

            xa = abs(a - x)
            ab = abs(b - a)
            bc = abs(c - b)
            cd = abs(d - c)

            if xa == 0:
                continue

            # Ratios
            ab_xa = ab / xa
            bc_ab = bc / ab if ab != 0 else 0
            cd_bc = cd / bc if bc != 0 else 0

            # Check each pattern
            direction = "bullish" if d < x else "bearish"
            matched = self._match_harmonic_ratios(ab_xa, bc_ab, cd_bc, tolerance)

            if matched:
                patterns.append(HarmonicPattern(
                    pattern_type=matched,
                    direction=direction,
                    completion_zone=(min(c, d), max(c, d)),
                    points={"X": x, "A": a, "B": b, "C": c, "D": d},
                    confidence=self._harmonic_confidence(ab_xa, bc_ab, cd_bc, matched),
                ))

        return patterns

    @staticmethod
    def _match_harmonic_ratios(
        ab_xa: float, bc_ab: float, cd_bc: float, tol: float,
    ) -> Optional[str]:
        """Match ratio set against known harmonic patterns."""
        def near(val: float, target: float) -> bool:
            return abs(val - target) <= tol

        # Gartley: AB/XA ≈ 0.618, BC/AB ≈ 0.382-0.886, CD/BC ≈ 1.272-1.618
        if near(ab_xa, 0.618) and 0.382 - tol <= bc_ab <= 0.886 + tol:
            return "gartley"

        # Butterfly: AB/XA ≈ 0.786, CD/BC ≈ 1.618-2.618
        if near(ab_xa, 0.786) and 1.618 - tol <= cd_bc <= 2.618 + tol:
            return "butterfly"

        # Bat: AB/XA ≈ 0.382-0.500
        if 0.382 - tol <= ab_xa <= 0.500 + tol:
            return "bat"

        # Crab: AB/XA ≈ 0.382-0.618, CD/BC ≈ 2.618-3.618
        if 0.382 - tol <= ab_xa <= 0.618 + tol and 2.618 - tol <= cd_bc <= 3.618 + tol:
            return "crab"

        return None

    @staticmethod
    def _harmonic_confidence(
        ab_xa: float, bc_ab: float, cd_bc: float, pattern: str,
    ) -> float:
        """Estimate confidence based on how closely ratios match ideal."""
        ideal_ratios = {
            "gartley": (0.618, 0.618, 1.272),
            "butterfly": (0.786, 0.618, 1.618),
            "bat": (0.500, 0.500, 1.618),
            "crab": (0.618, 0.382, 2.618),
        }
        ideal = ideal_ratios.get(pattern, (0.618, 0.618, 1.618))
        deviations = [
            abs(ab_xa - ideal[0]),
            abs(bc_ab - ideal[1]),
            abs(cd_bc - ideal[2]),
        ]
        avg_dev = sum(deviations) / 3
        return max(0.0, min(1.0, 1.0 - avg_dev))


# ═══════════════════════════════════════════════════════════════════════════
# PHASE CONJUGATION (Dan Winter Theory)
# ═══════════════════════════════════════════════════════════════════════════


def phase_conjugation_score(frequencies: List[float]) -> float:
    """
    Calculate Dan Winter's phase conjugation score.

    When frequency ratios between nested wave systems approach
    powers of PHI, constructive interference (implosion) occurs.
    Pure phase conjugation = pure compression of charge.

    Returns a 0-100 score where higher = better phase conjugation.
    """
    if len(frequencies) < 2:
        return 0.0

    ratios = []
    sorted_freq = sorted(frequencies)
    for i in range(len(sorted_freq) - 1):
        if sorted_freq[i] > 0:
            ratios.append(sorted_freq[i + 1] / sorted_freq[i])

    if not ratios:
        return 0.0

    # Score each ratio by proximity to a PHI power
    phi_powers = [PHI ** i for i in range(-3, 4)]
    total_score = 0.0

    for ratio in ratios:
        best_match = min(phi_powers, key=lambda p: abs(ratio - p))
        deviation = abs(ratio - best_match) / best_match
        total_score += max(0, 1.0 - deviation * 5)

    return round((total_score / len(ratios)) * 100, 2)


def fractal_compression_index(price_series: List[float], window: int = 21) -> float:
    """
    Calculate fractal compression index for a price series.

    High compression → imminent breakout (Dan Winter's principle:
    maximum compression precedes maximum expansion).

    Returns 0-100 where higher = more compressed = breakout imminent.
    """
    if len(price_series) < window:
        return 0.0

    recent = price_series[-window:]
    range_pct = (max(recent) - min(recent)) / max(recent) * 100 if max(recent) > 0 else 0

    # Compare to longer window
    longer = price_series[-(window * 3):] if len(price_series) >= window * 3 else price_series
    long_range = (max(longer) - min(longer)) / max(longer) * 100 if max(longer) > 0 else 0

    if long_range == 0:
        return 0.0

    compression_ratio = 1 - (range_pct / long_range)
    return round(max(0, min(100, compression_ratio * 100)), 2)

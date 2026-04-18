"""
IV Skew Optimizer — Strike Selection via Implied Volatility Surface
====================================================================
Analyzes the IV smile/skew across strikes and expiries to find optimal
put strikes where implied volatility is richest (expensive to sell) or
cheapest (cheap to buy) relative to realized vol and the overall surface.

Key concepts:
    - IV Skew: Puts typically have higher IV than ATM (fear premium)
    - Skew steepness: How fast IV rises as strikes go OTM
    - Term structure: IV differences across expiries
    - Vol-of-vol: Instability in the skew itself

For AAC's put buying:
    - We WANT to buy where IV is CHEAPEST relative to neighbors
    - Avoid overpaying for extreme skew at deep OTM
    - Find kink points where skew flattens — best value
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StrikeIV:
    """IV data for a single strike."""
    strike: float
    iv: float
    delta: float
    bid: float
    ask: float
    volume: int
    open_interest: int

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as % of mid."""
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid if mid > 0 else 1.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def liquid(self) -> bool:
        """Is this strike liquid enough to trade?"""
        return self.volume >= 10 and self.open_interest >= 50 and self.spread_pct < 0.15


@dataclass
class SkewAnalysis:
    """Analysis of the IV skew for a single expiry."""
    ticker: str
    expiry: str
    spot: float
    atm_iv: float
    skew_slope: float           # dIV/dStrike (normalized)
    skew_curvature: float       # d2IV/dStrike2 (smile convexity)
    put_skew_richness: float    # How much extra IV puts have over ATM
    cheapest_strike: float      # Strike with lowest IV-to-delta ratio
    richest_strike: float       # Strike with highest IV (avoid buying here)
    kink_strike: float          # Where skew flattens (best value)
    term_structure_slope: float  # 0 = flat, positive = contango, negative = backwardation
    strikes: List[StrikeIV] = field(default_factory=list)

    @property
    def skew_is_steep(self) -> bool:
        """Steep skew = expensive OTM puts."""
        return self.skew_slope < -0.002

    @property
    def skew_is_flat(self) -> bool:
        """Flat skew = puts are relatively fair."""
        return abs(self.skew_slope) < 0.0005


@dataclass
class OptimalStrike:
    """Recommended strike from skew analysis."""
    ticker: str
    strike: float
    expiry: str
    delta: float
    iv: float
    iv_percentile: float            # Where this IV sits in the chain (0-100)
    value_score: float              # 0-100, higher = better value
    spread_cost: float              # Bid-ask spread in dollars
    reasoning: str
    skew_analysis: Optional[SkewAnalysis] = None


class SkewOptimizer:
    """
    Analyzes IV skew to find optimal put strikes.

    The optimizer:
    1. Maps the IV smile across all strikes for a given expiry
    2. Identifies the skew slope, curvature, and kink points
    3. Scores each strike on value (IV vs delta vs liquidity)
    4. Returns the optimal strike(s) for put buying

    Usage:
        optimizer = SkewOptimizer()
        # From pre-fetched chain data
        optimal = optimizer.find_optimal_strike(chain_data, spot, target_delta=-0.30)
        # Multi-expiry analysis
        analysis = optimizer.analyze_term_structure(chains_by_expiry, spot)
    """

    # Strike selection parameters
    MIN_DELTA = -0.50            # Don't go deeper ITM
    MAX_DELTA = -0.10            # Don't go too far OTM
    TARGET_DELTA = -0.30         # Center of sweet spot
    MIN_VOLUME = 10
    MIN_OI = 50
    MAX_SPREAD_PCT = 0.15        # 15% max bid-ask spread

    def find_optimal_strike(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
        target_delta: float = -0.30,
        min_delta: float = -0.50,
        max_delta: float = -0.10,
        ticker: str = "",
        expiry: str = "",
    ) -> Optional[OptimalStrike]:
        """
        Find the optimal put strike from an options chain.

        Args:
            chain: List of option contract dicts with keys:
                strike, iv, delta, bid, ask, volume, open_interest
            spot: Current underlying price
            target_delta: Desired delta (e.g., -0.30)
            min_delta/max_delta: Delta range to consider
            ticker: Underlying symbol
            expiry: Expiry date string

        Returns:
            OptimalStrike or None if no suitable strike found
        """
        strikes = self._parse_chain(chain)
        if not strikes:
            logger.warning("No strikes parsed from chain for %s", ticker)
            return None

        # Filter to tradeable delta range
        candidates = [
            s for s in strikes
            if min_delta <= s.delta <= max_delta and s.liquid
        ]

        if not candidates:
            # Relax liquidity filter
            candidates = [
                s for s in strikes
                if min_delta <= s.delta <= max_delta
            ]

        if not candidates:
            logger.warning("No candidates in delta range [%.2f, %.2f] for %s",
                           min_delta, max_delta, ticker)
            return None

        # Compute skew analysis
        skew = self._analyze_skew(strikes, spot, ticker, expiry)

        # Score each candidate
        scored = []
        iv_values = [c.iv for c in candidates]
        iv_min = min(iv_values) if iv_values else 0
        iv_max = max(iv_values) if iv_values else 1
        iv_range = iv_max - iv_min if iv_max > iv_min else 0.01

        for c in candidates:
            score = self._score_strike(c, target_delta, iv_min, iv_range, spot)
            scored.append((c, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best_strike, best_score = scored[0]

        # IV percentile within this chain
        iv_pct = ((best_strike.iv - iv_min) / iv_range) * 100 if iv_range > 0 else 50

        # Build reasoning
        reasoning = self._build_reasoning(best_strike, target_delta, skew, iv_pct)

        return OptimalStrike(
            ticker=ticker,
            strike=best_strike.strike,
            expiry=expiry,
            delta=best_strike.delta,
            iv=best_strike.iv,
            iv_percentile=iv_pct,
            value_score=best_score,
            spread_cost=best_strike.ask - best_strike.bid,
            reasoning=reasoning,
            skew_analysis=skew,
        )

    def analyze_skew(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
        ticker: str = "",
        expiry: str = "",
    ) -> SkewAnalysis:
        """Analyze IV skew for a single expiry chain."""
        strikes = self._parse_chain(chain)
        return self._analyze_skew(strikes, spot, ticker, expiry)

    def analyze_term_structure(
        self,
        chains_by_expiry: Dict[str, List[Dict[str, Any]]],
        spot: float,
        ticker: str = "",
    ) -> List[SkewAnalysis]:
        """
        Analyze IV term structure across multiple expiries.

        Returns skew analyses sorted by expiry, with term_structure_slope
        computed from the ATM IV curve across time.
        """
        analyses = []
        for expiry, chain in sorted(chains_by_expiry.items()):
            strikes = self._parse_chain(chain)
            skew = self._analyze_skew(strikes, spot, ticker, expiry)
            analyses.append(skew)

        # Compute term structure slope from ATM IVs
        if len(analyses) >= 2:
            atm_ivs = [(i, a.atm_iv) for i, a in enumerate(analyses) if a.atm_iv > 0]
            if len(atm_ivs) >= 2:
                slope = (atm_ivs[-1][1] - atm_ivs[0][1]) / max(1, atm_ivs[-1][0] - atm_ivs[0][0])
                for a in analyses:
                    a.term_structure_slope = slope

        return analyses

    def find_best_expiry(
        self,
        chains_by_expiry: Dict[str, List[Dict[str, Any]]],
        spot: float,
        target_delta: float = -0.30,
        min_dte: int = 14,
        max_dte: int = 45,
        ticker: str = "",
    ) -> Optional[OptimalStrike]:
        """
        Find the best strike across all expiries in the target DTE range.
        """
        best: Optional[OptimalStrike] = None

        for expiry, chain in sorted(chains_by_expiry.items()):
            optimal = self.find_optimal_strike(
                chain, spot, target_delta=target_delta,
                ticker=ticker, expiry=expiry,
            )
            if optimal and (best is None or optimal.value_score > best.value_score):
                best = optimal

        return best

    # ═══════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _parse_chain(self, chain: List[Dict[str, Any]]) -> List[StrikeIV]:
        """Parse raw chain dicts into StrikeIV objects."""
        strikes = []
        for c in chain:
            try:
                strikes.append(StrikeIV(
                    strike=float(c.get("strike", 0)),
                    iv=float(c.get("iv", c.get("implied_volatility", 0))),
                    delta=float(c.get("delta", 0)),
                    bid=float(c.get("bid", 0)),
                    ask=float(c.get("ask", 0)),
                    volume=int(c.get("volume", 0)),
                    open_interest=int(c.get("open_interest", c.get("oi", 0))),
                ))
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed strike: %s", exc)
        return sorted(strikes, key=lambda s: s.strike)

    def _analyze_skew(
        self,
        strikes: List[StrikeIV],
        spot: float,
        ticker: str,
        expiry: str,
    ) -> SkewAnalysis:
        """Compute skew metrics from parsed strikes."""
        if not strikes:
            return SkewAnalysis(
                ticker=ticker, expiry=expiry, spot=spot,
                atm_iv=0, skew_slope=0, skew_curvature=0,
                put_skew_richness=0, cheapest_strike=0,
                richest_strike=0, kink_strike=0, term_structure_slope=0,
            )

        # Find ATM strike (closest to spot)
        atm = min(strikes, key=lambda s: abs(s.strike - spot))
        atm_iv = atm.iv

        # OTM puts only (strike < spot)
        otm_puts = [s for s in strikes if s.strike < spot and s.iv > 0]

        if len(otm_puts) < 2:
            return SkewAnalysis(
                ticker=ticker, expiry=expiry, spot=spot,
                atm_iv=atm_iv, skew_slope=0, skew_curvature=0,
                put_skew_richness=0, cheapest_strike=atm.strike,
                richest_strike=atm.strike, kink_strike=atm.strike,
                term_structure_slope=0, strikes=strikes,
            )

        # Skew slope: linear regression of IV vs moneyness
        moneyness = [(s.strike / spot, s.iv) for s in otm_puts]
        n = len(moneyness)
        sum_x = sum(m[0] for m in moneyness)
        sum_y = sum(m[1] for m in moneyness)
        sum_xy = sum(m[0] * m[1] for m in moneyness)
        sum_x2 = sum(m[0] ** 2 for m in moneyness)
        denom = n * sum_x2 - sum_x ** 2
        skew_slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0

        # Curvature: second derivative approximation
        curvature = 0.0
        if len(otm_puts) >= 3:
            mid = len(otm_puts) // 2
            if mid > 0 and mid < len(otm_puts) - 1:
                iv_left = otm_puts[mid - 1].iv
                iv_mid = otm_puts[mid].iv
                iv_right = otm_puts[mid + 1].iv
                dk = (otm_puts[mid + 1].strike - otm_puts[mid - 1].strike) / (2 * spot)
                if dk > 0:
                    curvature = (iv_left - 2 * iv_mid + iv_right) / (dk ** 2)

        # Put skew richness: avg OTM put IV / ATM IV - 1
        avg_otm_iv = sum(s.iv for s in otm_puts) / len(otm_puts)
        put_skew_richness = (avg_otm_iv / atm_iv - 1) if atm_iv > 0 else 0

        # Finding cheapest, richest, and kink strikes
        cheapest = min(otm_puts, key=lambda s: s.iv)
        richest = max(otm_puts, key=lambda s: s.iv)

        # Kink point: where second derivative of IV changes sign
        # (where the skew starts to flatten)
        kink = self._find_kink(otm_puts, spot)

        return SkewAnalysis(
            ticker=ticker,
            expiry=expiry,
            spot=spot,
            atm_iv=atm_iv,
            skew_slope=skew_slope,
            skew_curvature=curvature,
            put_skew_richness=put_skew_richness,
            cheapest_strike=cheapest.strike,
            richest_strike=richest.strike,
            kink_strike=kink,
            term_structure_slope=0.0,
            strikes=strikes,
        )

    def _find_kink(self, otm_puts: List[StrikeIV], spot: float) -> float:
        """Find where IV skew flattens (kink point)."""
        if len(otm_puts) < 3:
            return otm_puts[0].strike if otm_puts else 0.0

        # Compute local IV slopes between adjacent strikes
        slopes = []
        for i in range(len(otm_puts) - 1):
            dk = otm_puts[i + 1].strike - otm_puts[i].strike
            if dk != 0:
                div = (otm_puts[i + 1].iv - otm_puts[i].iv) / dk
                slopes.append((otm_puts[i].strike, div))

        if len(slopes) < 2:
            return otm_puts[len(otm_puts) // 2].strike

        # Kink = where slope magnitude drops the most
        max_drop = 0.0
        kink_strike = otm_puts[len(otm_puts) // 2].strike

        for i in range(len(slopes) - 1):
            drop = abs(slopes[i][1]) - abs(slopes[i + 1][1])
            if drop > max_drop:
                max_drop = drop
                kink_strike = slopes[i + 1][0]

        return kink_strike

    def _score_strike(
        self,
        strike: StrikeIV,
        target_delta: float,
        iv_min: float,
        iv_range: float,
        spot: float,
    ) -> float:
        """
        Score a candidate strike for put buying.

        Factors (weighted):
            40% — Delta proximity to target
            25% — IV cheapness (lower = better for buying)
            20% — Liquidity (tighter spread, more volume)
            15% — OI depth (more stable pricing)
        """
        # Delta score: closer to target = better
        delta_dist = abs(strike.delta - target_delta)
        delta_score = max(0, 100 - delta_dist * 400)  # 0.25 away = 0

        # IV cheapness: lower IV within the chain = better buy
        iv_norm = (strike.iv - iv_min) / iv_range if iv_range > 0 else 0.5
        iv_score = (1 - iv_norm) * 100  # Cheaper = higher score

        # Liquidity: tighter spread = better
        if strike.spread_pct < 0.03:
            liq_score = 100
        elif strike.spread_pct < 0.08:
            liq_score = 70
        elif strike.spread_pct < 0.15:
            liq_score = 40
        else:
            liq_score = 10

        # OI depth score
        if strike.open_interest >= 1000:
            oi_score = 100
        elif strike.open_interest >= 500:
            oi_score = 80
        elif strike.open_interest >= 100:
            oi_score = 50
        else:
            oi_score = 20

        return (
            delta_score * 0.40 +
            iv_score * 0.25 +
            liq_score * 0.20 +
            oi_score * 0.15
        )

    def _build_reasoning(
        self,
        strike: StrikeIV,
        target_delta: float,
        skew: SkewAnalysis,
        iv_pct: float,
    ) -> str:
        """Build reasoning for strike selection."""
        parts = []

        delta_dist = abs(strike.delta - target_delta)
        if delta_dist < 0.05:
            parts.append(f"delta {strike.delta:.2f} near target {target_delta:.2f}")
        else:
            parts.append(f"delta {strike.delta:.2f} (target {target_delta:.2f})")

        if iv_pct < 30:
            parts.append(f"IV at {iv_pct:.0f}th pct of chain (cheap)")
        elif iv_pct > 70:
            parts.append(f"IV at {iv_pct:.0f}th pct of chain (expensive)")
        else:
            parts.append(f"IV at {iv_pct:.0f}th pct of chain")

        if skew.skew_is_steep:
            parts.append("steep skew — OTM puts expensive")
        elif skew.skew_is_flat:
            parts.append("flat skew — fair pricing")

        if strike.spread_pct < 0.05:
            parts.append("tight spread")
        elif strike.spread_pct > 0.10:
            parts.append(f"wide spread ({strike.spread_pct:.0%})")

        return "; ".join(parts) + "."

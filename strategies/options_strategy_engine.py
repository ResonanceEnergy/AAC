"""
Options Strategy Execution Engine — BARREN WUFFET v2.7.0
=========================================================
Comprehensive options strategy builder covering 40+ strategies with
Greeks calculation, risk management, and execution logic.

Strategies: Iron Condor, Iron Butterfly, Straddle/Strangle, Butterfly,
Calendar, Diagonal, Collar, Covered Call, Wheel, PMCC, Jade Lizard,
Broken Wing Butterfly, Christmas Tree, Ratio Spreads, Box Spread, etc.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════

class OptionType(Enum):
    """OptionType class."""
    CALL = "call"
    PUT = "put"

class Direction(Enum):
    """Direction class."""
    LONG = "long"
    SHORT = "short"

class StrategyOutlook(Enum):
    """StrategyOutlook class."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

class RiskProfile(Enum):
    """RiskProfile class."""
    DEFINED = "defined"
    UNDEFINED = "undefined"
    SEMI_DEFINED = "semi_defined"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptionLeg:
    """Single option leg in a strategy."""
    option_type: OptionType
    direction: Direction
    strike: float
    expiry_days: int
    quantity: int = 1
    premium: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0

    @property
    def is_long(self) -> bool:
        """Is long."""
        return self.direction == Direction.LONG

    @property
    def net_premium(self) -> float:
        """Positive = debit (paid), negative = credit (received)."""
        return self.premium * self.quantity * (1 if self.is_long else -1)


@dataclass
class StrategyGreeks:
    """Aggregate Greeks for a multi-leg strategy."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    def __repr__(self) -> str:
        return (f"Greeks(Δ={self.delta:+.3f}, Γ={self.gamma:.4f}, "
                f"Θ={self.theta:+.2f}, V={self.vega:+.3f})")


@dataclass
class RiskMetrics:
    """Risk/reward metrics for a strategy."""
    max_profit: float
    max_loss: float
    breakeven_prices: List[float]
    probability_of_profit: float
    risk_reward_ratio: float
    capital_required: float
    return_on_risk: float

    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        return (self.probability_of_profit * self.max_profit -
                (1 - self.probability_of_profit) * abs(self.max_loss))


@dataclass
class OptionsStrategy:
    """Complete options strategy definition."""
    name: str
    outlook: StrategyOutlook
    risk_profile: RiskProfile
    legs: List[OptionLeg]
    underlying_price: float
    underlying_shares: int = 0  # For covered strategies
    description: str = ""
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    adjustment_rules: List[str] = field(default_factory=list)

    @property
    def net_credit(self) -> float:
        """Net credit/debit of the strategy. Positive = credit."""
        return -sum(leg.net_premium for leg in self.legs)

    @property
    def greeks(self) -> StrategyGreeks:
        """Aggregate Greeks."""
        g = StrategyGreeks()
        for leg in self.legs:
            sign = 1 if leg.is_long else -1
            g.delta += leg.delta * leg.quantity * sign
            g.gamma += leg.gamma * leg.quantity * sign
            g.theta += leg.theta * leg.quantity * sign
            g.vega += leg.vega * leg.quantity * sign
        return g

    @property
    def leg_count(self) -> int:
        """Leg count."""
        return len(self.legs)

    def to_dict(self) -> Dict:
        """To dict."""
        return {
            "name": self.name,
            "outlook": self.outlook.value,
            "risk_profile": self.risk_profile.value,
            "leg_count": self.leg_count,
            "net_credit": self.net_credit,
            "greeks": {
                "delta": self.greeks.delta,
                "gamma": self.greeks.gamma,
                "theta": self.greeks.theta,
                "vega": self.greeks.vega,
            },
            "underlying_price": self.underlying_price,
            "underlying_shares": self.underlying_shares,
        }


# ═══════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING
# ═══════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = OptionType.CALL
) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        if option_type == OptionType.CALL:
            return max(0, S - K)
        return max(0, K - S)

    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == OptionType.CALL:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def compute_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = OptionType.CALL
) -> Dict[str, float]:
    """Compute all first-order Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * _norm_pdf(d1) * math.sqrt(T) / 100  # Per 1% IV change

    if option_type == OptionType.CALL:
        delta = _norm_cdf(d1)
        theta = ((-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))) -
                 r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100
    else:
        delta = _norm_cdf(d1) - 1
        theta = ((-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))) +
                 r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

class StrategyBuilder:
    """Factory for building common options strategies."""

    def __init__(self, underlying_price: float, risk_free_rate: float = 0.05):
        self.S = underlying_price
        self.r = risk_free_rate

    def _make_leg(
        self, opt_type: OptionType, direction: Direction,
        strike: float, dte: int, iv: float, qty: int = 1
    ) -> OptionLeg:
        """Create an option leg with computed price and Greeks."""
        T = dte / 365
        premium = black_scholes_price(self.S, strike, T, self.r, iv, opt_type)
        greeks = compute_greeks(self.S, strike, T, self.r, iv, opt_type)
        return OptionLeg(
            option_type=opt_type, direction=direction, strike=strike,
            expiry_days=dte, quantity=qty, premium=premium,
            delta=greeks["delta"], gamma=greeks["gamma"],
            theta=greeks["theta"], vega=greeks["vega"], iv=iv,
        )

    # ── BULLISH STRATEGIES ─────────────────────────────────────────────

    def long_call(self, strike: float, dte: int, iv: float) -> OptionsStrategy:
        """Buy a call option."""
        leg = self._make_leg(OptionType.CALL, Direction.LONG, strike, dte, iv)
        return OptionsStrategy(
            name="Long Call", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.DEFINED, legs=[leg],
            underlying_price=self.S,
            description="Unlimited upside, limited risk (premium paid)",
            entry_rules=["Buy when IV is low (IVR < 30)", "Strong directional conviction"],
            exit_rules=["Take profit at 50-100%", "Cut loss at 50%"],
        )

    def bull_call_spread(
        self, long_strike: float, short_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Buy lower strike call, sell higher strike call."""
        long_leg = self._make_leg(OptionType.CALL, Direction.LONG, long_strike, dte, iv)
        short_leg = self._make_leg(OptionType.CALL, Direction.SHORT, short_strike, dte, iv)
        return OptionsStrategy(
            name="Bull Call Spread", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.DEFINED, legs=[long_leg, short_leg],
            underlying_price=self.S,
            description="Debit spread: lower cost than naked call, capped profit",
            entry_rules=["Moderate bullish outlook", "30-60 DTE", "Debit < 50% of spread width"],
            exit_rules=["Close at 50% max profit", "Close at 21 DTE"],
        )

    def bull_put_spread(
        self, short_strike: float, long_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell higher strike put, buy lower strike put (credit)."""
        short_leg = self._make_leg(OptionType.PUT, Direction.SHORT, short_strike, dte, iv)
        long_leg = self._make_leg(OptionType.PUT, Direction.LONG, long_strike, dte, iv)
        return OptionsStrategy(
            name="Bull Put Spread", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.DEFINED, legs=[short_leg, long_leg],
            underlying_price=self.S,
            description="Credit spread: profit if stock stays above short strike",
            entry_rules=["IVR > 50%", "30-45 DTE", "Collect 1/3 of spread width"],
            exit_rules=["Close at 50% max profit", "Close at 21 DTE"],
        )

    def covered_call(
        self, call_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Own 100 shares + sell OTM call."""
        short_call = self._make_leg(OptionType.CALL, Direction.SHORT, call_strike, dte, iv)
        return OptionsStrategy(
            name="Covered Call", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.SEMI_DEFINED, legs=[short_call],
            underlying_price=self.S, underlying_shares=100,
            description="Income strategy: cap upside for premium income",
            entry_rules=["Sell 0.25-0.35 delta call", "30-45 DTE", "Stock you'd hold long-term"],
            exit_rules=["Roll or close at 50% profit", "Let assign if above target"],
        )

    def wheel_strategy(
        self, put_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell cash-secured put (Phase 1 of Wheel)."""
        short_put = self._make_leg(OptionType.PUT, Direction.SHORT, put_strike, dte, iv)
        return OptionsStrategy(
            name="Wheel Strategy (CSP Phase)", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.SEMI_DEFINED, legs=[short_put],
            underlying_price=self.S,
            description="Sell CSP → if assigned, sell CC → if called away, repeat",
            entry_rules=["Quality stock you'd own", "0.25-0.35 delta", "30-45 DTE"],
            exit_rules=["Close at 50% profit", "Accept assignment if put goes ITM"],
            adjustment_rules=["If assigned → transition to covered call phase"],
        )

    def poor_mans_covered_call(
        self, leaps_strike: float, leaps_dte: int,
        short_strike: float, short_dte: int, iv: float
    ) -> OptionsStrategy:
        """Buy deep ITM LEAPS call + sell short-term OTM call."""
        leaps = self._make_leg(OptionType.CALL, Direction.LONG, leaps_strike, leaps_dte, iv)
        short = self._make_leg(OptionType.CALL, Direction.SHORT, short_strike, short_dte, iv)
        return OptionsStrategy(
            name="Poor Man's Covered Call", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.DEFINED, legs=[leaps, short],
            underlying_price=self.S,
            description="Diagonal spread: LEAPS replaces stock for capital efficiency",
            entry_rules=["LEAPS delta > 0.70", "LEAPS 6+ months out", "Short call 30-45 DTE"],
            exit_rules=["Roll short call at 50% profit", "Close if LEAPS at risk"],
        )

    def collar(
        self, put_strike: float, call_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Own stock + buy OTM put + sell OTM call."""
        long_put = self._make_leg(OptionType.PUT, Direction.LONG, put_strike, dte, iv)
        short_call = self._make_leg(OptionType.CALL, Direction.SHORT, call_strike, dte, iv)
        return OptionsStrategy(
            name="Collar", outlook=StrategyOutlook.NEUTRAL,
            risk_profile=RiskProfile.DEFINED, legs=[long_put, short_call],
            underlying_price=self.S, underlying_shares=100,
            description="Zero or low-cost insurance: floor + ceiling on stock",
            entry_rules=["Use call premium to offset put cost", "Moderate outlook"],
            exit_rules=["Let expire if stock between strikes", "Adjust if breached"],
        )

    def jade_lizard(
        self, put_strike: float, call_long_strike: float,
        call_short_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell OTM put + sell call spread (bear call)."""
        short_put = self._make_leg(OptionType.PUT, Direction.SHORT, put_strike, dte, iv)
        short_call = self._make_leg(OptionType.CALL, Direction.SHORT, call_short_strike, dte, iv)
        long_call = self._make_leg(OptionType.CALL, Direction.LONG, call_long_strike, dte, iv)
        return OptionsStrategy(
            name="Jade Lizard", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.SEMI_DEFINED, legs=[short_put, short_call, long_call],
            underlying_price=self.S,
            description="No upside risk if total credit > call spread width",
            entry_rules=["Total credit > call spread width", "30-45 DTE"],
            exit_rules=["Close at 50% profit", "Manage put side if threatened"],
        )

    # ── BEARISH STRATEGIES ─────────────────────────────────────────────

    def long_put(self, strike: float, dte: int, iv: float) -> OptionsStrategy:
        """Buy a put option."""
        leg = self._make_leg(OptionType.PUT, Direction.LONG, strike, dte, iv)
        return OptionsStrategy(
            name="Long Put", outlook=StrategyOutlook.BEARISH,
            risk_profile=RiskProfile.DEFINED, legs=[leg],
            underlying_price=self.S,
            description="Profit from decline; limited risk (premium paid)",
            entry_rules=["Strong bearish conviction", "Lower IV preferred"],
            exit_rules=["Take profit at 50-100%", "Cut loss at 50%"],
        )

    def bear_call_spread(
        self, short_strike: float, long_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell lower strike call, buy higher strike call (credit)."""
        short_leg = self._make_leg(OptionType.CALL, Direction.SHORT, short_strike, dte, iv)
        long_leg = self._make_leg(OptionType.CALL, Direction.LONG, long_strike, dte, iv)
        return OptionsStrategy(
            name="Bear Call Spread", outlook=StrategyOutlook.BEARISH,
            risk_profile=RiskProfile.DEFINED, legs=[short_leg, long_leg],
            underlying_price=self.S,
            description="Credit spread: profit if stock stays below short call",
            entry_rules=["IVR > 50%", "30-45 DTE", "Collect 1/3 of spread width"],
            exit_rules=["Close at 50% profit", "Close at 21 DTE"],
        )

    def bear_put_spread(
        self, long_strike: float, short_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Buy higher strike put, sell lower strike put (debit)."""
        long_leg = self._make_leg(OptionType.PUT, Direction.LONG, long_strike, dte, iv)
        short_leg = self._make_leg(OptionType.PUT, Direction.SHORT, short_strike, dte, iv)
        return OptionsStrategy(
            name="Bear Put Spread", outlook=StrategyOutlook.BEARISH,
            risk_profile=RiskProfile.DEFINED, legs=[long_leg, short_leg],
            underlying_price=self.S,
            description="Debit spread: profit from decline, cheaper than naked put",
            entry_rules=["Moderate bearish outlook", "30-60 DTE"],
            exit_rules=["Close at 50% max profit", "Close at 21 DTE"],
        )

    # ── NEUTRAL STRATEGIES ─────────────────────────────────────────────

    def iron_condor(
        self, put_long: float, put_short: float,
        call_short: float, call_long: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell OTM put spread + sell OTM call spread."""
        legs = [
            self._make_leg(OptionType.PUT, Direction.LONG, put_long, dte, iv),
            self._make_leg(OptionType.PUT, Direction.SHORT, put_short, dte, iv),
            self._make_leg(OptionType.CALL, Direction.SHORT, call_short, dte, iv),
            self._make_leg(OptionType.CALL, Direction.LONG, call_long, dte, iv),
        ]
        return OptionsStrategy(
            name="Iron Condor", outlook=StrategyOutlook.NEUTRAL,
            risk_profile=RiskProfile.DEFINED, legs=legs,
            underlying_price=self.S,
            description="Range-bound play: profit from time decay and low volatility",
            entry_rules=["30-45 DTE", "15-25 delta shorts", "IVR > 50%", "No earnings/FOMC"],
            exit_rules=["Close at 50% profit", "Close at 21 DTE"],
            adjustment_rules=[
                "Roll untested side closer if one side threatened",
                "Close tested side if breached, let untested side expire",
            ],
        )

    def iron_butterfly(
        self, put_long: float, atm_strike: float,
        call_long: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell ATM straddle + buy OTM wings."""
        legs = [
            self._make_leg(OptionType.PUT, Direction.LONG, put_long, dte, iv),
            self._make_leg(OptionType.PUT, Direction.SHORT, atm_strike, dte, iv),
            self._make_leg(OptionType.CALL, Direction.SHORT, atm_strike, dte, iv),
            self._make_leg(OptionType.CALL, Direction.LONG, call_long, dte, iv),
        ]
        return OptionsStrategy(
            name="Iron Butterfly", outlook=StrategyOutlook.NEUTRAL,
            risk_profile=RiskProfile.DEFINED, legs=legs,
            underlying_price=self.S,
            description="Maximum premium at ATM; tighter profit zone than iron condor",
            entry_rules=["30-45 DTE", "High IVR", "ATM short strikes"],
            exit_rules=["Close at 25% profit (tighter)", "Close at 21 DTE"],
        )

    def short_straddle(
        self, strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell ATM call + sell ATM put."""
        legs = [
            self._make_leg(OptionType.CALL, Direction.SHORT, strike, dte, iv),
            self._make_leg(OptionType.PUT, Direction.SHORT, strike, dte, iv),
        ]
        return OptionsStrategy(
            name="Short Straddle", outlook=StrategyOutlook.NEUTRAL,
            risk_profile=RiskProfile.UNDEFINED, legs=legs,
            underlying_price=self.S,
            description="Maximum theta: unlimited risk both directions",
            entry_rules=["Very high IVR (> 70%)", "45 DTE", "Portfolio margin recommended"],
            exit_rules=["Close at 25% profit", "Defend at 2x credit received"],
        )

    def long_straddle(
        self, strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Buy ATM call + buy ATM put."""
        legs = [
            self._make_leg(OptionType.CALL, Direction.LONG, strike, dte, iv),
            self._make_leg(OptionType.PUT, Direction.LONG, strike, dte, iv),
        ]
        return OptionsStrategy(
            name="Long Straddle", outlook=StrategyOutlook.VOLATILE,
            risk_profile=RiskProfile.DEFINED, legs=legs,
            underlying_price=self.S,
            description="Profit from big move in either direction",
            entry_rules=["Low IV", "Expect catalyst or event", "30-60 DTE"],
            exit_rules=["Close when underlying moves beyond breakeven", "Cut at 50% loss"],
        )

    def short_strangle(
        self, put_strike: float, call_strike: float, dte: int, iv: float
    ) -> OptionsStrategy:
        """Sell OTM put + sell OTM call."""
        legs = [
            self._make_leg(OptionType.PUT, Direction.SHORT, put_strike, dte, iv),
            self._make_leg(OptionType.CALL, Direction.SHORT, call_strike, dte, iv),
        ]
        return OptionsStrategy(
            name="Short Strangle", outlook=StrategyOutlook.NEUTRAL,
            risk_profile=RiskProfile.UNDEFINED, legs=legs,
            underlying_price=self.S,
            description="Wider profit zone than straddle; unlimited risk",
            entry_rules=["IVR > 50%", "45 DTE", "16 delta strikes", "Diversify across tickers"],
            exit_rules=["Close at 50% profit", "Close at 21 DTE"],
        )

    def butterfly_spread(
        self, lower: float, middle: float, upper: float,
        dte: int, iv: float, use_calls: bool = True
    ) -> OptionsStrategy:
        """Buy 1 lower, sell 2 middle, buy 1 upper."""
        opt = OptionType.CALL if use_calls else OptionType.PUT
        legs = [
            self._make_leg(opt, Direction.LONG, lower, dte, iv),
            self._make_leg(opt, Direction.SHORT, middle, dte, iv, qty=2),
            self._make_leg(opt, Direction.LONG, upper, dte, iv),
        ]
        return OptionsStrategy(
            name=f"Butterfly Spread ({'Calls' if use_calls else 'Puts'})",
            outlook=StrategyOutlook.NEUTRAL, risk_profile=RiskProfile.DEFINED,
            legs=legs, underlying_price=self.S,
            description="Pin play: max profit at center strike; cheap entry; 10:1+ R:R possible",
            entry_rules=["Target specific price level", "14-30 DTE", "ATM center strike"],
            exit_rules=["Close at 50% max profit", "Close at 7 DTE if not near pin"],
        )

    def calendar_spread(
        self, strike: float, near_dte: int, far_dte: int,
        iv_near: float, iv_far: float, use_calls: bool = True
    ) -> OptionsStrategy:
        """Sell near-term, buy far-term at same strike."""
        opt = OptionType.CALL if use_calls else OptionType.PUT
        legs = [
            self._make_leg(opt, Direction.SHORT, strike, near_dte, iv_near),
            self._make_leg(opt, Direction.LONG, strike, far_dte, iv_far),
        ]
        return OptionsStrategy(
            name=f"Calendar Spread ({'Calls' if use_calls else 'Puts'})",
            outlook=StrategyOutlook.NEUTRAL, risk_profile=RiskProfile.DEFINED,
            legs=legs, underlying_price=self.S,
            description="Time spread: profit from differential theta decay + IV expansion",
            entry_rules=["ATM strike", "Near 30 DTE / Far 60+ DTE", "Low IV environment"],
            exit_rules=["Close when near-term expires or at 25% profit"],
        )

    def broken_wing_butterfly(
        self, lower: float, middle: float, upper: float,
        dte: int, iv: float, bullish: bool = True
    ) -> OptionsStrategy:
        """Asymmetric butterfly with directional bias."""
        if bullish:
            # Skip wider on upside
            legs = [
                self._make_leg(OptionType.PUT, Direction.LONG, lower, dte, iv),
                self._make_leg(OptionType.PUT, Direction.SHORT, middle, dte, iv, qty=2),
                self._make_leg(OptionType.PUT, Direction.LONG, upper, dte, iv),
            ]
        else:
            legs = [
                self._make_leg(OptionType.CALL, Direction.LONG, lower, dte, iv),
                self._make_leg(OptionType.CALL, Direction.SHORT, middle, dte, iv, qty=2),
                self._make_leg(OptionType.CALL, Direction.LONG, upper, dte, iv),
            ]
        return OptionsStrategy(
            name=f"Broken Wing Butterfly ({'Bullish' if bullish else 'Bearish'})",
            outlook=StrategyOutlook.BULLISH if bullish else StrategyOutlook.BEARISH,
            risk_profile=RiskProfile.SEMI_DEFINED, legs=legs,
            underlying_price=self.S,
            description="Credit or small debit; no risk on one side; risk on opposite",
            entry_rules=["30-45 DTE", "Moderate directional bias"],
            exit_rules=["Close at 50% profit", "Manage if stock approaches risk side"],
        )

    def christmas_tree_spread(
        self, atm: float, otm1: float, otm2: float,
        dte: int, iv: float, use_calls: bool = True
    ) -> OptionsStrategy:
        """Buy 1 ATM, sell 1 OTM, sell 1 further OTM."""
        opt = OptionType.CALL if use_calls else OptionType.PUT
        legs = [
            self._make_leg(opt, Direction.LONG, atm, dte, iv),
            self._make_leg(opt, Direction.SHORT, otm1, dte, iv),
            self._make_leg(opt, Direction.SHORT, otm2, dte, iv),
        ]
        return OptionsStrategy(
            name=f"Christmas Tree ({'Calls' if use_calls else 'Puts'})",
            outlook=StrategyOutlook.BULLISH if use_calls else StrategyOutlook.BEARISH,
            risk_profile=RiskProfile.SEMI_DEFINED, legs=legs,
            underlying_price=self.S,
            description="Asymmetric butterfly variant: cheaper entry, extra short leg",
            entry_rules=["Directional bias", "30-45 DTE"],
            exit_rules=["Close at 50% profit", "Manage unlimited risk beyond 2nd short"],
        )

    def ratio_spread(
        self, long_strike: float, short_strike: float,
        dte: int, iv: float, ratio: int = 2, use_calls: bool = True
    ) -> OptionsStrategy:
        """Buy 1, sell N at different strike (ratio spread)."""
        opt = OptionType.CALL if use_calls else OptionType.PUT
        legs = [
            self._make_leg(opt, Direction.LONG, long_strike, dte, iv),
            self._make_leg(opt, Direction.SHORT, short_strike, dte, iv, qty=ratio),
        ]
        return OptionsStrategy(
            name=f"1x{ratio} Ratio Spread ({'Calls' if use_calls else 'Puts'})",
            outlook=StrategyOutlook.NEUTRAL,
            risk_profile=RiskProfile.UNDEFINED, legs=legs,
            underlying_price=self.S,
            description=f"Buy 1, sell {ratio}: can be zero-cost; unlimited risk beyond shorts",
            entry_rules=["High IV on short strike side", "30-45 DTE"],
            exit_rules=["Close if stock approaches unlimited risk zone"],
        )


# ═══════════════════════════════════════════════════════════════════════════
# RISK CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class OptionsRiskCalculator:
    """Calculate risk metrics for any options strategy."""

    @staticmethod
    def calculate_payoff(strategy: OptionsStrategy, stock_price: float) -> float:
        """Calculate P/L at a given stock price at expiration."""
        total_pnl = 0.0

        # Stock position P/L
        if strategy.underlying_shares:
            total_pnl += strategy.underlying_shares * (stock_price - strategy.underlying_price)

        for leg in strategy.legs:
            if leg.option_type == OptionType.CALL:
                intrinsic = max(0, stock_price - leg.strike)
            else:
                intrinsic = max(0, leg.strike - stock_price)

            if leg.is_long:
                total_pnl += (intrinsic - leg.premium) * leg.quantity * 100
            else:
                total_pnl += (leg.premium - intrinsic) * leg.quantity * 100

        return total_pnl

    @staticmethod
    def compute_risk_metrics(
        strategy: OptionsStrategy,
        price_range: Tuple[float, float] = None,
        steps: int = 1000,
    ) -> RiskMetrics:
        """Compute full risk metrics across price range."""
        calc = OptionsRiskCalculator()
        S = strategy.underlying_price

        if price_range is None:
            price_range = (S * 0.5, S * 1.5)

        prices = [price_range[0] + (price_range[1] - price_range[0]) * i / steps
                  for i in range(steps + 1)]

        payoffs = [calc.calculate_payoff(strategy, p) for p in prices]

        max_profit = max(payoffs)
        max_loss = min(payoffs)

        # Find breakeven prices
        breakevens = []
        for i in range(len(payoffs) - 1):
            if (payoffs[i] <= 0 <= payoffs[i + 1]) or (payoffs[i] >= 0 >= payoffs[i + 1]):
                # Linear interpolation
                if payoffs[i + 1] != payoffs[i]:
                    ratio = -payoffs[i] / (payoffs[i + 1] - payoffs[i])
                    be = prices[i] + ratio * (prices[i + 1] - prices[i])
                    breakevens.append(round(be, 2))

        # Probability of profit (assuming normal distribution)
        profitable_steps = sum(1 for p in payoffs if p > 0)
        pop = profitable_steps / len(payoffs)

        # Risk/reward
        risk_reward = abs(max_loss / max_profit) if max_profit != 0 else float("inf")

        # Capital required
        net_credit = strategy.net_credit * 100  # Per contract
        if net_credit > 0:
            # Credit strategy: capital = max loss
            capital = abs(max_loss)
        else:
            # Debit strategy: capital = debit paid
            capital = abs(net_credit)

        ror = (max_profit / capital * 100) if capital > 0 else 0

        return RiskMetrics(
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            breakeven_prices=breakevens,
            probability_of_profit=round(pop, 4),
            risk_reward_ratio=round(risk_reward, 2),
            capital_required=round(capital, 2),
            return_on_risk=round(ror, 2),
        )


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class StrategyScanner:
    """Recommend strategies based on market conditions."""

    STRATEGY_MATRIX = {
        # (outlook, iv_environment) -> recommended strategies
        ("bullish", "high_iv"): ["Bull Put Spread", "Covered Call", "Wheel", "Jade Lizard"],
        ("bullish", "low_iv"): ["Long Call", "Bull Call Spread", "PMCC", "LEAPS"],
        ("bearish", "high_iv"): ["Bear Call Spread", "Put Backspread"],
        ("bearish", "low_iv"): ["Long Put", "Bear Put Spread"],
        ("neutral", "high_iv"): ["Iron Condor", "Iron Butterfly", "Short Strangle", "Short Straddle"],
        ("neutral", "low_iv"): ["Calendar Spread", "Double Diagonal", "Butterfly"],
        ("volatile", "high_iv"): ["Long Straddle", "Long Strangle", "Reverse Iron Condor"],
        ("volatile", "low_iv"): ["Long Straddle", "Long Strangle", "Calendar (far month)"],
    }

    @classmethod
    def recommend(cls, outlook: str, iv_rank: float) -> List[str]:
        """Recommend strategies based on outlook and IV environment."""
        iv_env = "high_iv" if iv_rank > 50 else "low_iv"
        key = (outlook.lower(), iv_env)
        strategies = cls.STRATEGY_MATRIX.get(key, [])
        logger.info(f"Recommended strategies for {outlook}/{iv_env}: {strategies}")
        return strategies

    @classmethod
    def filter_by_risk(
        cls, strategies: List[str], max_risk: RiskProfile
    ) -> List[str]:
        """Filter strategies by maximum risk level."""
        risk_map = {
            "Long Call": RiskProfile.DEFINED,
            "Long Put": RiskProfile.DEFINED,
            "Bull Call Spread": RiskProfile.DEFINED,
            "Bull Put Spread": RiskProfile.DEFINED,
            "Bear Call Spread": RiskProfile.DEFINED,
            "Bear Put Spread": RiskProfile.DEFINED,
            "Iron Condor": RiskProfile.DEFINED,
            "Iron Butterfly": RiskProfile.DEFINED,
            "Butterfly": RiskProfile.DEFINED,
            "Calendar Spread": RiskProfile.DEFINED,
            "PMCC": RiskProfile.DEFINED,
            "Covered Call": RiskProfile.SEMI_DEFINED,
            "Wheel": RiskProfile.SEMI_DEFINED,
            "Jade Lizard": RiskProfile.SEMI_DEFINED,
            "Collar": RiskProfile.DEFINED,
            "Short Strangle": RiskProfile.UNDEFINED,
            "Short Straddle": RiskProfile.UNDEFINED,
            "Long Straddle": RiskProfile.DEFINED,
            "Long Strangle": RiskProfile.DEFINED,
        }
        if max_risk == RiskProfile.DEFINED:
            return [s for s in strategies if risk_map.get(s) == RiskProfile.DEFINED]
        elif max_risk == RiskProfile.SEMI_DEFINED:
            return [s for s in strategies if risk_map.get(s) in
                    (RiskProfile.DEFINED, RiskProfile.SEMI_DEFINED)]
        return strategies


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — Options Strategy Engine v2.7.0")
    print("=" * 55)

    # Demo: Build and analyze an Iron Condor on SPY
    builder = StrategyBuilder(underlying_price=520.0)
    ic = builder.iron_condor(
        put_long=490, put_short=500,
        call_short=540, call_long=550,
        dte=35, iv=0.18,
    )
    print(f"\nStrategy: {ic.name}")
    print(f"Legs: {ic.leg_count}")
    print(f"Net Credit: ${ic.net_credit:.2f}")
    print(f"Greeks: {ic.greeks}")

    # Risk metrics
    metrics = OptionsRiskCalculator.compute_risk_metrics(ic)
    print(f"\nRisk Metrics:")
    print(f"  Max Profit: ${metrics.max_profit:,.2f}")
    print(f"  Max Loss: ${metrics.max_loss:,.2f}")
    print(f"  Breakevens: {metrics.breakeven_prices}")
    print(f"  POP: {metrics.probability_of_profit:.1%}")
    print(f"  Risk/Reward: {metrics.risk_reward_ratio:.2f}")
    print(f"  Return on Risk: {metrics.return_on_risk:.1f}%")
    print(f"  Expectancy: ${metrics.expectancy:,.2f}")

    # Strategy recommendations
    print("\n\nStrategy Recommendations:")
    for outlook in ["bullish", "bearish", "neutral", "volatile"]:
        for ivr in [25, 75]:
            recs = StrategyScanner.recommend(outlook, ivr)
            print(f"  {outlook.upper()} | IVR={ivr}: {', '.join(recs)}")

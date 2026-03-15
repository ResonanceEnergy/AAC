"""
0DTE & Intraday Gamma Trading Engine — BARREN WUFFET v2.7.0
=============================================================
Specialized engine for zero-days-to-expiration (0DTE) options trading
and intraday gamma scalping strategies.

From BARREN WUFFET Insights:
  - 0DTE options now account for 40-50% of SPX daily volume
  - Gamma is MASSIVE at expiration → small moves cause large P/L
  - Theta decays exponentially in final hours → time is critical
  - Dealer hedging of 0DTE creates intraday volatility patterns
  - Key times: 10:00 AM (opening range), 2:00 PM (afternoon session), 3:45 PM (close)
  - GEX profile shifts intraday as 0DTE gamma dominates
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import logging
from datetime import datetime, time

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════

class SessionPhase(Enum):
    """SessionPhase class."""
    PRE_MARKET = "pre_market"          # Before 9:30
    OPENING_RANGE = "opening_range"    # 9:30 - 10:00
    MORNING = "morning"                # 10:00 - 11:30
    MIDDAY = "midday"                  # 11:30 - 14:00
    AFTERNOON = "afternoon"            # 14:00 - 15:30
    POWER_HOUR = "power_hour"          # 15:30 - 15:45
    FINAL_15 = "final_15"             # 15:45 - 16:00
    AFTER_HOURS = "after_hours"

class ZeroDTEStrategy(Enum):
    """ZeroDTEStrategy class."""
    SCALP_GAMMA = "scalp_gamma"            # Buy ATM, scalp delta moves
    SELL_IRON_CONDOR = "sell_iron_condor"   # Sell OTM spreads, let decay
    SELL_BUTTERFLY = "sell_butterfly"       # Sell wings for theta
    PIN_PLAY = "pin_play"                  # Bet on max pain pinning
    MOMENTUM_CALLS = "momentum_calls"       # Buy calls for breakout
    MOMENTUM_PUTS = "momentum_puts"         # Buy puts for breakdown
    FADE_OPENING = "fade_opening_range"     # Fade the OR expansion

class RiskMode(Enum):
    """RiskMode class."""
    CONSERVATIVE = "conservative"  # Max 0.5% risk per trade
    MODERATE = "moderate"          # Max 1% risk per trade
    AGGRESSIVE = "aggressive"      # Max 2% risk per trade


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IntradayLevel:
    """Key intraday price level."""
    price: float
    level_type: str  # "support", "resistance", "pivot", "gex", "max_pain"
    strength: float  # 0-100
    source: str


@dataclass
class ZeroDTESetup:
    """0DTE trade setup."""
    strategy: ZeroDTEStrategy
    entry_time: str
    exit_deadline: str
    strikes: List[float]
    estimated_cost: float
    max_risk: float
    target_profit: float
    risk_reward: float
    session_phase: SessionPhase
    conviction: float  # 0-100
    notes: List[str]


@dataclass
class GammaScalpPosition:
    """Active gamma scalping position."""
    symbol: str
    entry_price: float
    option_strike: float
    option_type: str  # "call" or "put"
    option_cost: float
    delta_at_entry: float
    current_delta: float = 0.0
    shares_hedged: int = 0
    realized_scalp_pnl: float = 0.0
    scalp_count: int = 0
    entry_time: str = ""
    last_scalp_time: str = ""

    @property
    def total_pnl(self) -> float:
        """Total P/L including option decay and scalp profits."""
        return self.realized_scalp_pnl - self.option_cost


@dataclass
class IntradayAnalytics:
    """Intraday session analytics."""
    session_date: str
    opening_range_high: float = 0.0
    opening_range_low: float = 0.0
    vwap: float = 0.0
    max_pain: float = 0.0
    current_price: float = 0.0
    session_range: float = 0.0
    volume_profile_poc: float = 0.0  # Point of control
    put_call_ratio_0dte: float = 0.0
    total_0dte_volume: int = 0
    gex_flip_level: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SESSION PHASE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class SessionDetector:
    """Detect current trading session phase and characteristics."""

    PHASE_CHARACTERISTICS = {
        SessionPhase.OPENING_RANGE: {
            "volatility": "highest",
            "theta_decay": "moderate",
            "gamma_impact": "extreme",
            "best_strategies": [
                ZeroDTEStrategy.SCALP_GAMMA,
                ZeroDTEStrategy.MOMENTUM_CALLS,
                ZeroDTEStrategy.MOMENTUM_PUTS,
            ],
            "avoid": [ZeroDTEStrategy.SELL_IRON_CONDOR],
            "note": "Wait for OR to form (9:30-10:00), then trade the break or fade",
        },
        SessionPhase.MORNING: {
            "volatility": "high",
            "theta_decay": "slow",
            "gamma_impact": "high",
            "best_strategies": [
                ZeroDTEStrategy.SCALP_GAMMA,
                ZeroDTEStrategy.SELL_IRON_CONDOR,
            ],
            "avoid": [ZeroDTEStrategy.PIN_PLAY],
            "note": "Trend continuation from OR. Good for directional or range plays",
        },
        SessionPhase.MIDDAY: {
            "volatility": "lowest",
            "theta_decay": "moderate",
            "gamma_impact": "moderate",
            "best_strategies": [
                ZeroDTEStrategy.SELL_IRON_CONDOR,
                ZeroDTEStrategy.SELL_BUTTERFLY,
            ],
            "avoid": [ZeroDTEStrategy.SCALP_GAMMA],
            "note": "Low vol lunch lull. Theta decay starts to help short vol. Range-bound.",
        },
        SessionPhase.AFTERNOON: {
            "volatility": "rising",
            "theta_decay": "accelerating",
            "gamma_impact": "very_high",
            "best_strategies": [
                ZeroDTEStrategy.SCALP_GAMMA,
                ZeroDTEStrategy.PIN_PLAY,
                ZeroDTEStrategy.SELL_IRON_CONDOR,
            ],
            "avoid": [],
            "note": "Theta accelerating. Institutions often re-hedge here. Key inflection point.",
        },
        SessionPhase.POWER_HOUR: {
            "volatility": "high",
            "theta_decay": "extreme",
            "gamma_impact": "extreme",
            "best_strategies": [
                ZeroDTEStrategy.PIN_PLAY,
                ZeroDTEStrategy.FADE_OPENING,
            ],
            "avoid": [ZeroDTEStrategy.SCALP_GAMMA],
            "note": "Maximum gamma. Single ticks move options 20-50%. Pin towards max pain.",
        },
        SessionPhase.FINAL_15: {
            "volatility": "extreme",
            "theta_decay": "terminal",
            "gamma_impact": "infinite",
            "best_strategies": [],
            "avoid": [s for s in ZeroDTEStrategy],
            "note": "Too late. All positions should be closed. Edge is near zero.",
        },
    }

    @classmethod
    def detect_phase(cls, current_time: time) -> SessionPhase:
        """Detect current session phase from time."""
        if current_time < time(9, 30):
            return SessionPhase.PRE_MARKET
        elif current_time < time(10, 0):
            return SessionPhase.OPENING_RANGE
        elif current_time < time(11, 30):
            return SessionPhase.MORNING
        elif current_time < time(14, 0):
            return SessionPhase.MIDDAY
        elif current_time < time(15, 30):
            return SessionPhase.AFTERNOON
        elif current_time < time(15, 45):
            return SessionPhase.POWER_HOUR
        elif current_time < time(16, 0):
            return SessionPhase.FINAL_15
        else:
            return SessionPhase.AFTER_HOURS

    @classmethod
    def get_characteristics(cls, phase: SessionPhase) -> Dict:
        """Get characteristics for a session phase."""
        chars = cls.PHASE_CHARACTERISTICS.get(phase, {})
        return {
            "phase": phase.value,
            **chars,
            "best_strategies": [s.value for s in chars.get("best_strategies", [])],
            "avoid": [s.value for s in chars.get("avoid", [])],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 0DTE STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ZeroDTEEngine:
    """
    Core engine for 0DTE options strategy selection and sizing.
    
    Key principles:
      1. Time is your friend (selling) or enemy (buying) — and it accelerates
      2. Gamma dominates: small moves create outsized P/L
      3. Liquidity: only trade liquid underlyings (SPX, SPY, QQQ, AAPL)
      4. Sizing: smaller than you think (0.25-1% risk per trade)
      5. Hard exits: close ALL positions by 3:45 PM
      6. No overnight: "0DTE" means ZERO residual at EOD
    """

    def __init__(
        self, capital: float,
        risk_mode: RiskMode = RiskMode.MODERATE,
        underlying_price: float = 520.0,
    ):
        self.capital = capital
        self.risk_mode = risk_mode
        self.S = underlying_price
        self.max_risk_pct = {
            RiskMode.CONSERVATIVE: 0.005,
            RiskMode.MODERATE: 0.01,
            RiskMode.AGGRESSIVE: 0.02,
        }[risk_mode]

    def generate_iron_condor(
        self, current_phase: SessionPhase,
        opening_range_width: float,
        atm_iv: float,
    ) -> Optional[ZeroDTESetup]:
        """
        Generate 0DTE iron condor setup.
        
        Best in: morning → midday → afternoon
        Avoid in: opening_range, final_15
        """
        if current_phase in (SessionPhase.OPENING_RANGE, SessionPhase.FINAL_15,
                              SessionPhase.POWER_HOUR):
            return None

        # Short strikes at 1-1.5x opening range
        or_half = opening_range_width / 2
        margin = or_half * 1.3  # 30% buffer beyond OR

        put_short = round(self.S - margin, 0)
        call_short = round(self.S + margin, 0)
        wing_width = 5.0
        put_long = put_short - wing_width
        call_long = call_short + wing_width

        # Estimate credit (higher earlier in day)
        time_factor = {
            SessionPhase.MORNING: 0.45,
            SessionPhase.MIDDAY: 0.35,
            SessionPhase.AFTERNOON: 0.25,
        }.get(current_phase, 0.30)

        est_credit = wing_width * time_factor
        max_loss = (wing_width - est_credit) * 100
        max_risk = self.capital * self.max_risk_pct
        contracts = max(1, int(max_risk / max_loss))

        return ZeroDTESetup(
            strategy=ZeroDTEStrategy.SELL_IRON_CONDOR,
            entry_time=current_phase.value,
            exit_deadline="15:30",
            strikes=[put_long, put_short, call_short, call_long],
            estimated_cost=round(est_credit * contracts * 100, 2),
            max_risk=round(max_loss * contracts, 2),
            target_profit=round(est_credit * 0.5 * contracts * 100, 2),
            risk_reward=round(max_loss / (est_credit * 100), 2),
            session_phase=current_phase,
            conviction=65 if current_phase == SessionPhase.MIDDAY else 55,
            notes=[
                f"Sell {contracts}x {put_short}/{put_long}p — {call_short}/{call_long}c",
                f"Credit target: ~${est_credit:.2f}/spread",
                "Close at 50% profit OR 3:30 PM, whichever first",
                "If tested: close entire position, do NOT defend",
                f"Max risk: ${max_loss * contracts:,.0f} ({self.max_risk_pct * 100:.1f}% of capital)",
            ],
        )

    def generate_gamma_scalp(
        self, current_phase: SessionPhase,
        atm_iv: float,
        scalp_size_pct: float = 0.003,  # 0.3% of capital
    ) -> Optional[ZeroDTESetup]:
        """
        Generate gamma scalp setup.
        
        Gamma scalping: buy ATM option, delta-hedge with shares.
        As stock moves → delta changes → hedge locks in profit.
        Repeat until theta eats you alive.
        
        Best in: high vol sessions, opening range, afternoon
        """
        if current_phase in (SessionPhase.MIDDAY, SessionPhase.FINAL_15):
            return None

        # ATM straddle cost estimate
        # For 0DTE: IV * S * sqrt(hours_left / 252*6.5)
        hours_left = {
            SessionPhase.OPENING_RANGE: 6.5,
            SessionPhase.MORNING: 5.0,
            SessionPhase.AFTERNOON: 2.0,
            SessionPhase.POWER_HOUR: 0.5,
        }.get(current_phase, 4.0)

        option_cost = self.S * atm_iv * math.sqrt(hours_left / (252 * 6.5))
        max_risk = option_cost * 100  # Per contract

        contracts = max(1, int(self.capital * scalp_size_pct / max_risk))

        # Target: need stock to move > straddle cost to profit
        move_needed = option_cost / self.S * 100

        return ZeroDTESetup(
            strategy=ZeroDTEStrategy.SCALP_GAMMA,
            entry_time=current_phase.value,
            exit_deadline="15:30",
            strikes=[round(self.S, 0)],
            estimated_cost=round(max_risk * contracts, 2),
            max_risk=round(max_risk * contracts, 2),
            target_profit=round(max_risk * contracts * 0.5, 2),  # 50% of cost
            risk_reward=0.5,
            session_phase=current_phase,
            conviction=50,
            notes=[
                f"Buy {contracts}x ATM straddle at ~${option_cost:.2f}/side",
                f"Stock must move >{move_needed:.1f}% for profit",
                f"Scalp delta hedges at every $1 move in underlying",
                f"Total risk: ${max_risk * contracts:,.0f}",
                "Close ALL by 3:30 PM — theta is lethal in final hour",
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# GAMMA SCALPING TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class GammaScalpTracker:
    """
    Track and manage gamma scalping positions.
    
    How gamma scalping works:
      1. Buy ATM straddle (long gamma)
      2. Delta-hedge with shares (sell shares when delta positive, buy when negative)
      3. Each time stock reverts, the hedge locks in a small profit
      4. Repeat until theta decay exceeds scalping profits
    
    The breakeven: gamma_scalps - theta_cost = 0
    If realized vol > implied vol → gamma scalps > theta → PROFIT
    If realized vol < implied vol → gamma scalps < theta → LOSS
    """

    def __init__(self, position: GammaScalpPosition, hedge_interval: float = 1.0):
        self.position = position
        self.hedge_interval = hedge_interval  # Hedge every $1 move

    def should_hedge(self, current_price: float, current_delta: float) -> Dict:
        """Check if a delta hedge is needed."""
        price_moved = abs(current_price - self.position.entry_price)
        delta_change = abs(current_delta - self.position.current_delta)

        # Hedge if price has moved by hedge_interval
        if price_moved >= self.hedge_interval or delta_change > 0.10:
            shares_needed = int((current_delta - 0.5) * 100)  # Target delta-neutral
            current_shares = self.position.shares_hedged

            trade_shares = shares_needed - current_shares
            action = "SELL" if trade_shares > 0 else "BUY"

            return {
                "should_hedge": True,
                "action": action,
                "shares": abs(trade_shares),
                "price": current_price,
                "current_delta": current_delta,
                "scalp_pnl_est": abs(price_moved * current_shares * 0.5),
                "note": f"{action} {abs(trade_shares)} shares at ${current_price:.2f}",
            }

        return {
            "should_hedge": False,
            "price_move": price_moved,
            "next_hedge_at": self.position.entry_price + self.hedge_interval,
        }

    def theta_burn_rate(self, hours_remaining: float, option_cost: float) -> Dict:
        """Calculate theta burn vs scalp breakeven."""
        if hours_remaining <= 0:
            return {"status": "EXPIRED", "action": "CLOSE_NOW"}

        # Theta accelerates: ~sqrt relationship
        theta_per_hour = option_cost / (2 * math.sqrt(hours_remaining))
        theta_remaining = option_cost * (1 - math.sqrt(max(0, hours_remaining) / 6.5))

        # Need enough realized vol to offset
        scalp_pnl = self.position.realized_scalp_pnl
        net = scalp_pnl - theta_remaining

        return {
            "theta_per_hour": round(theta_per_hour, 2),
            "theta_remaining": round(theta_remaining, 2),
            "scalp_pnl": round(scalp_pnl, 2),
            "net_pnl": round(net, 2),
            "profitable": net > 0,
            "hours_left": round(hours_remaining, 2),
            "breakeven_scalps_needed": max(0, int((theta_remaining - scalp_pnl) / self.hedge_interval)),
        }


# ═══════════════════════════════════════════════════════════════════════════
# OPENING RANGE BREAKOUT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class OpeningRangeEngine:
    """
    Opening Range Breakout strategy using 0DTE options.
    
    The Opening Range is the first 30 minutes (9:30-10:00).
    Breakouts from the OR tend to follow through, especially:
      - On high-volume days
      - After overnight gaps
      - On FOMC/CPI/NFP days
    """

    @staticmethod
    def compute_opening_range(
        high_9_30_10: float, low_9_30_10: float, vwap: float
    ) -> Dict:
        """Compute opening range levels."""
        width = high_9_30_10 - low_9_30_10
        midpoint = (high_9_30_10 + low_9_30_10) / 2

        # Extension targets (Fibonacci-like)
        ext_1 = width * 1.0   # 100% extension
        ext_1_5 = width * 1.5  # 150% extension
        ext_2 = width * 2.0   # 200% extension

        return {
            "or_high": high_9_30_10,
            "or_low": low_9_30_10,
            "or_width": round(width, 2),
            "or_width_pct": round(width / midpoint * 100, 2),
            "midpoint": round(midpoint, 2),
            "vwap": vwap,
            # Breakout targets
            "break_up_t1": round(high_9_30_10 + ext_1, 2),
            "break_up_t2": round(high_9_30_10 + ext_1_5, 2),
            "break_up_t3": round(high_9_30_10 + ext_2, 2),
            # Breakdown targets
            "break_down_t1": round(low_9_30_10 - ext_1, 2),
            "break_down_t2": round(low_9_30_10 - ext_1_5, 2),
            "break_down_t3": round(low_9_30_10 - ext_2, 2),
        }

    @staticmethod
    def generate_breakout_trade(
        or_levels: Dict, direction: str, capital: float,
        current_price: float
    ) -> Dict:
        """Generate 0DTE breakout trade using options."""
        risk_pct = 0.005  # 0.5% risk
        max_risk = capital * risk_pct

        if direction == "up":
            entry_trigger = or_levels["or_high"]
            target = or_levels["break_up_t1"]
            stop = or_levels["midpoint"]
            option = "call"
        else:
            entry_trigger = or_levels["or_low"]
            target = or_levels["break_down_t1"]
            stop = or_levels["midpoint"]
            option = "put"

        # ATM option ~$2-5 for SPY 0DTE
        est_option_cost = abs(entry_trigger - stop) * 0.5
        contracts = max(1, int(max_risk / (est_option_cost * 100)))

        return {
            "direction": direction,
            "entry_trigger": entry_trigger,
            "target": target,
            "stop_loss": stop,
            "option_type": option,
            "strike": round(entry_trigger, 0),
            "contracts": contracts,
            "est_cost": round(est_option_cost * contracts * 100, 2),
            "max_risk": round(max_risk, 2),
            "risk_reward": round(abs(target - entry_trigger) / abs(entry_trigger - stop), 2),
            "exit_deadline": "15:30",
            "notes": [
                f"Buy {contracts}x {round(entry_trigger, 0)} 0DTE {option}",
                f"Trigger: break {'above' if direction == 'up' else 'below'} ${entry_trigger:.2f}",
                f"Target: ${target:.2f} | Stop: ${stop:.2f}",
                "Confirm with volume > 1.5x average",
                "Close at target OR 3:30 PM, NO exceptions",
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAX PAIN / PIN PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

class MaxPainPredictor:
    """
    Calculate max pain and predict potential pin levels.
    
    Max Pain Theory:
      - Stock price tends to gravitate toward the strike where
        option holders (both call + put) lose the most money
      - This is because market makers (dealers) have opposite positions
        and their hedging pushes price toward max pain
      - More reliable in low-vol weeks; less reliable in crisis
    """

    @staticmethod
    def calculate_max_pain(
        strikes: List[float],
        call_oi: Dict[float, int],
        put_oi: Dict[float, int],
    ) -> Dict:
        """Calculate max pain strike."""
        pain_at_strike = {}

        for test_price in strikes:
            total_pain = 0

            # Call pain: for each call strike, if test_price > strike,
            # call holders profit (pain for writers/MM)
            for strike, oi in call_oi.items():
                if test_price > strike:
                    total_pain += (test_price - strike) * oi * 100

            # Put pain: for each put strike, if test_price < strike,
            # put holders profit
            for strike, oi in put_oi.items():
                if test_price < strike:
                    total_pain += (strike - test_price) * oi * 100

            pain_at_strike[test_price] = total_pain

        # Max pain = strike with MINIMUM total payout
        max_pain_strike = min(pain_at_strike, key=pain_at_strike.get)

        # Also find top 3 magnetic strikes (highest OI)
        combined_oi = {}
        for strike in strikes:
            combined_oi[strike] = call_oi.get(strike, 0) + put_oi.get(strike, 0)
        magnetic = sorted(combined_oi.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "max_pain": max_pain_strike,
            "total_pain_at_mp": pain_at_strike[max_pain_strike],
            "magnetic_strikes": [{"strike": s, "total_oi": oi} for s, oi in magnetic],
            "pain_landscape": {str(k): v for k, v in sorted(pain_at_strike.items())[:10]},
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — 0DTE & Intraday Gamma Engine v2.7.0")
    print("=" * 60)

    # Session detection
    now = datetime.now().time()
    phase = SessionDetector.detect_phase(now)
    chars = SessionDetector.get_characteristics(phase)
    print(f"\nCurrent Session: {phase.value}")
    print(f"  Volatility: {chars.get('volatility', 'N/A')}")
    print(f"  Gamma Impact: {chars.get('gamma_impact', 'N/A')}")
    print(f"  Best Strategies: {chars.get('best_strategies', [])}")
    print(f"  Note: {chars.get('note', '')}")

    # 0DTE Iron Condor
    engine = ZeroDTEEngine(capital=100_000, risk_mode=RiskMode.MODERATE, underlying_price=520)
    ic_setup = engine.generate_iron_condor(
        current_phase=SessionPhase.MORNING,
        opening_range_width=3.5,
        atm_iv=0.20,
    )
    if ic_setup:
        print(f"\n0DTE Iron Condor Setup:")
        print(f"  Strikes: {ic_setup.strikes}")
        print(f"  Cost: ${ic_setup.estimated_cost:,.0f}")
        print(f"  Max Risk: ${ic_setup.max_risk:,.0f}")
        print(f"  Target: ${ic_setup.target_profit:,.0f}")
        print(f"  Conviction: {ic_setup.conviction}%")
        for note in ic_setup.notes:
            print(f"    → {note}")

    # Opening Range
    or_levels = OpeningRangeEngine.compute_opening_range(
        high_9_30_10=521.50, low_9_30_10=518.20, vwap=519.80
    )
    print(f"\nOpening Range:")
    for k, v in or_levels.items():
        print(f"  {k}: {v}")

    breakout = OpeningRangeEngine.generate_breakout_trade(
        or_levels, "up", 100_000, 521.00
    )
    print(f"\nBreakout Trade:")
    for note in breakout["notes"]:
        print(f"  → {note}")

    # Max Pain
    strikes = list(range(510, 535, 5))
    call_oi = {510: 5000, 515: 12000, 520: 25000, 525: 18000, 530: 8000}
    put_oi = {510: 8000, 515: 20000, 520: 22000, 525: 10000, 530: 3000}
    mp = MaxPainPredictor.calculate_max_pain(strikes, call_oi, put_oi)
    print(f"\nMax Pain: ${mp['max_pain']}")
    print(f"  Magnetic Strikes: {mp['magnetic_strikes']}")

"""
Options Income Systems — BARREN WUFFET v2.7.0
==============================================
Systematic options income strategies: Wheel, Covered Calls, Credit Spreads,
Iron Condors with portfolio management, position sizing, and yield tracking.

From BARREN WUFFET Insights 571-600:
  - Systematic premium selling has positive expectancy when:
    1. IV is elevated (IVR > 50%)
    2. Position sizing limits risk to 1-5% of portfolio per trade
    3. Management rules are followed mechanically
    4. Portfolio is diversified across sectors and expiries
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class WheelPhase(Enum):
    """WheelPhase class."""
    CSP = "cash_secured_put"       # Selling puts to enter
    COVERED_CALL = "covered_call"  # Assigned → selling calls
    IDLE = "idle"                  # Between cycles

class ManagementAction(Enum):
    """ManagementAction class."""
    HOLD = "hold"
    CLOSE = "close"
    ROLL_OUT = "roll_out"
    ROLL_UP = "roll_up"
    ROLL_DOWN = "roll_down"
    ROLL_OUT_AND_UP = "roll_out_and_up"
    ROLL_OUT_AND_DOWN = "roll_out_and_down"
    DEFEND = "defend"
    TAKE_ASSIGNMENT = "take_assignment"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IncomePosition:
    """A single income position being tracked."""
    symbol: str
    strategy: str
    entry_date: str
    expiry_date: str
    strikes: List[float]
    credit_received: float
    current_value: float = 0.0
    quantity: int = 1
    dte_at_entry: int = 0
    dte_remaining: int = 0
    delta: float = 0.0
    iv_at_entry: float = 0.0
    current_iv: float = 0.0
    underlying_at_entry: float = 0.0
    underlying_current: float = 0.0
    status: str = "open"  # open, closed, assigned, expired

    @property
    def pnl(self) -> float:
        """Current P/L per contract."""
        return (self.credit_received - self.current_value) * 100

    @property
    def pnl_pct(self) -> float:
        """P/L as percentage of credit received."""
        if self.credit_received == 0:
            return 0
        return (self.credit_received - self.current_value) / self.credit_received * 100

    @property
    def theta_decay_pct(self) -> float:
        """Estimated theta decay captured (time-based)."""
        if self.dte_at_entry == 0:
            return 100
        days_held = self.dte_at_entry - self.dte_remaining
        return min(100, (1 - math.sqrt(self.dte_remaining / self.dte_at_entry)) * 100)


@dataclass
class WheelPosition:
    """Complete Wheel strategy position tracker."""
    symbol: str
    phase: WheelPhase
    cost_basis: float = 0.0        # Adjusted cost basis (reduced by premiums)
    total_premium_collected: float = 0.0
    cycles_completed: int = 0
    shares_held: int = 0
    current_position: Optional[IncomePosition] = None
    history: List[Dict] = field(default_factory=list)
    target_annual_yield: float = 0.0

    @property
    def effective_cost_basis(self) -> float:
        """Cost basis adjusted for all premiums collected."""
        return self.cost_basis - self.total_premium_collected

    @property
    def annualized_yield(self) -> float:
        """Annualized yield from premiums."""
        if self.cost_basis == 0 or not self.history:
            return 0
        first = self.history[0].get("date", "")
        if not first:
            return 0
        try:
            start = datetime.fromisoformat(first)
            days = max(1, (datetime.utcnow() - start).days)
            return (self.total_premium_collected / self.cost_basis) * (365 / days) * 100
        except (ValueError, TypeError):
            return 0


@dataclass
class IncomePortfolio:
    """Portfolio-level income tracking."""
    positions: List[IncomePosition] = field(default_factory=list)
    wheel_positions: List[WheelPosition] = field(default_factory=list)
    total_capital: float = 0.0
    max_position_pct: float = 5.0  # Max 5% per position
    max_sector_pct: float = 20.0   # Max 20% per sector
    target_monthly_income: float = 0.0

    @property
    def total_credits(self) -> float:
        """Total credits."""
        return sum(p.credit_received * p.quantity * 100 for p in self.positions if p.status == "open")

    @property
    def total_pnl(self) -> float:
        """Total pnl."""
        return sum(p.pnl * p.quantity for p in self.positions if p.status == "open")

    @property
    def positions_count(self) -> int:
        """Positions count."""
        return sum(1 for p in self.positions if p.status == "open")

    @property
    def capital_deployed(self) -> float:
        """Estimate capital in use."""
        deployed = 0
        for p in self.positions:
            if p.status != "open":
                continue
            if p.strategy in ("covered_call", "wheel_csp"):
                deployed += p.underlying_at_entry * 100 * p.quantity
            elif p.strategy in ("iron_condor", "bull_put_spread", "bear_call_spread"):
                # Width of widest spread
                if len(p.strikes) >= 2:
                    width = max(p.strikes) - min(p.strikes)
                    deployed += (width - p.credit_received) * 100 * p.quantity
        return deployed

    @property
    def capital_utilization(self) -> float:
        """Capital utilization."""
        if self.total_capital == 0:
            return 0
        return self.capital_deployed / self.total_capital * 100


# ═══════════════════════════════════════════════════════════════════════════
# WHEEL STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class WheelEngine:
    """
    Full Wheel strategy automation engine.
    
    The Wheel:
      1. Sell cash-secured put (CSP) on a stock you'd buy
      2. If assigned → you now own 100 shares at strike - premium
      3. Sell covered calls on those shares
      4. If called away → you sold at strike + premium
      5. Repeat
    
    Edge comes from:
      - Selling elevated IV systematically
      - Lowering cost basis with each cycle
      - Mechanical management rules
    """

    def __init__(self, capital: float, max_allocation_pct: float = 15.0):
        self.capital = capital
        self.max_allocation = max_allocation_pct / 100

    def screen_csp_candidates(
        self, candidates: List[Dict]
    ) -> List[Dict]:
        """
        Screen stocks for CSP selling.
        
        Criteria:
          - Quality company (would own long-term)
          - Liquid options (tight bid-ask)
          - IVR > 50% (selling elevated premium)
          - Delta: 0.25-0.35 (70-80% POP)
          - 30-45 DTE
          - Capital efficient (stock price * 100 < max allocation)
        """
        max_capital_per = self.capital * self.max_allocation
        qualified = []

        for c in candidates:
            stock_price = c.get("price", 0)
            ivr = c.get("iv_rank", 0)
            bid_ask_spread = c.get("bid_ask_spread", 1.0)
            avg_volume = c.get("avg_option_volume", 0)

            # Capital requirement: 100 * strike
            capital_needed = stock_price * 100
            if capital_needed > max_capital_per:
                continue

            # IV Rank filter
            if ivr < 40:
                continue

            # Liquidity filter
            if avg_volume < 500 or bid_ask_spread > 0.15:
                continue

            score = (
                ivr * 0.4 +                              # IV premium
                min(100, avg_volume / 50) * 0.3 +        # Liquidity
                (1 - bid_ask_spread / 0.15) * 100 * 0.2 + # Tight spreads
                (1 - capital_needed / max_capital_per) * 100 * 0.1  # Capital efficiency
            )

            qualified.append({
                **c,
                "score": round(score, 2),
                "capital_needed": capital_needed,
                "capital_pct": round(capital_needed / self.capital * 100, 2),
            })

        qualified.sort(key=lambda x: x["score"], reverse=True)
        return qualified[:20]

    def calculate_csp_yield(
        self, strike: float, premium: float, dte: int
    ) -> Dict:
        """Calculate yield metrics for a cash-secured put."""
        capital = strike * 100
        credit = premium * 100
        annualized = (credit / capital) * (365 / dte) * 100
        monthly = annualized / 12
        breakeven = strike - premium

        return {
            "strike": strike,
            "premium": premium,
            "dte": dte,
            "credit": credit,
            "capital_required": capital,
            "yield_pct": round(credit / capital * 100, 2),
            "annualized_yield_pct": round(annualized, 2),
            "monthly_yield_pct": round(monthly, 2),
            "breakeven": round(breakeven, 2),
        }

    def should_roll(
        self, position: IncomePosition, underlying_price: float
    ) -> Tuple[bool, ManagementAction, str]:
        """
        Determine if a Wheel position should be rolled.
        
        Rules:
          - Close at 50% profit (don't get greedy)
          - Roll at 21 DTE if not profitable
          - If ITM at 14 DTE → roll out for credit
          - Never roll for a debit
          - If no credit available → take assignment
        """
        pnl_pct = position.pnl_pct
        dte = position.dte_remaining
        tested = underlying_price < min(position.strikes)  # CSP tested

        # 50% profit → close
        if pnl_pct >= 50:
            return True, ManagementAction.CLOSE, "50% profit target reached — close and redeploy"

        # 21 DTE and not profitable → roll
        if dte <= 21 and pnl_pct < 40:
            if tested:
                return True, ManagementAction.ROLL_OUT_AND_DOWN, (
                    "Tested at 21 DTE — roll out 30 days and down 1-2 strikes for credit"
                )
            return True, ManagementAction.ROLL_OUT, "21 DTE — roll out 30 days for credit"

        # Deep ITM at 14 DTE → consider assignment
        if dte <= 14 and tested:
            return True, ManagementAction.TAKE_ASSIGNMENT, (
                "Deep ITM near expiry — accept assignment, transition to covered call"
            )

        return False, ManagementAction.HOLD, "Within parameters — hold"


# ═══════════════════════════════════════════════════════════════════════════
# CREDIT SPREAD ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class CreditSpreadEngine:
    """
    Systematic credit spread selling engine.
    
    From BARREN WUFFET Insights:
      - Sell 1/3 of spread width as credit (minimum)
      - 15-25 delta short strike for iron condors
      - 30-45 DTE for optimal theta-to-gamma ratio
      - Close at 50% profit, 21 DTE, or tested
      - Never let a spread go to full loss
    """

    @staticmethod
    def select_spread(
        chain_data: List[Dict],
        outlook: str,
        target_delta: float = 0.20,
        spread_width: float = 5.0,
        min_credit_pct: float = 0.30,
    ) -> Optional[Dict]:
        """
        Select optimal credit spread from option chain.
        
        Args:
            chain_data: List of option chain entries
            outlook: 'bullish' (bull put spread) or 'bearish' (bear call spread)
            target_delta: Target delta for short strike
            spread_width: Width between strikes
            min_credit_pct: Minimum credit as % of spread width
        """
        best = None
        best_score = 0

        for option in chain_data:
            delta = abs(option.get("delta", 0))
            strike = option.get("strike", 0)
            bid = option.get("bid", 0)
            iv = option.get("iv", 0)

            delta_match = 1 - abs(delta - target_delta) / target_delta
            if delta_match < 0.5:
                continue

            credit = bid  # Approximate
            credit_pct = credit / spread_width

            if credit_pct < min_credit_pct:
                continue

            score = delta_match * 0.5 + credit_pct * 0.3 + (iv / 0.30) * 0.2

            if score > best_score:
                best_score = score
                if outlook == "bullish":
                    best = {
                        "strategy": "bull_put_spread",
                        "short_strike": strike,
                        "long_strike": strike - spread_width,
                        "credit": round(credit, 2),
                        "credit_pct": round(credit_pct * 100, 2),
                        "max_loss": round((spread_width - credit) * 100, 2),
                        "max_profit": round(credit * 100, 2),
                        "delta": round(delta, 3),
                        "score": round(score, 3),
                    }
                else:
                    best = {
                        "strategy": "bear_call_spread",
                        "short_strike": strike,
                        "long_strike": strike + spread_width,
                        "credit": round(credit, 2),
                        "credit_pct": round(credit_pct * 100, 2),
                        "max_loss": round((spread_width - credit) * 100, 2),
                        "max_profit": round(credit * 100, 2),
                        "delta": round(delta, 3),
                        "score": round(score, 3),
                    }

        return best

    @staticmethod
    def manage_spread(
        position: IncomePosition,
        underlying_price: float,
    ) -> Tuple[ManagementAction, str]:
        """
        Manage an open credit spread.
        
        Rules (from insights 571-585):
          1. Close at 50% max profit
          2. Close at 21 DTE
          3. If tested (stock near short strike) → roll or close
          4. If untested side can reduce risk → roll closer
          5. Never take max loss — close or roll before expiry
        """
        pnl_pct = position.pnl_pct
        dte = position.dte_remaining

        # Win management
        if pnl_pct >= 50:
            return ManagementAction.CLOSE, "Target hit: 50% profit — close and free capital"

        # Time management
        if dte <= 21 and pnl_pct >= 0:
            return ManagementAction.CLOSE, "21 DTE reached — close to avoid gamma risk"

        if dte <= 21 and pnl_pct < 0:
            return ManagementAction.ROLL_OUT, "21 DTE, losing — roll out 30 days for credit if possible"

        # Threat management
        short_strike = min(position.strikes)  # Assume put spread
        distance_pct = (underlying_price - short_strike) / underlying_price * 100

        if distance_pct < 1.0:
            return ManagementAction.DEFEND, "Stock within 1% of short strike — defend or close"

        if distance_pct < 2.0 and dte <= 14:
            return ManagementAction.ROLL_OUT_AND_DOWN, "Threatened at 14 DTE — roll down and out"

        return ManagementAction.HOLD, "Within parameters"


# ═══════════════════════════════════════════════════════════════════════════
# IRON CONDOR MANAGEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class IronCondorEngine:
    """
    Systematic Iron Condor management.
    
    The iron condor is the bread-and-butter neutral income strategy.
    From BARREN WUFFET insights:
      - Enter at 30-45 DTE, IVR > 50%
      - 15-25 delta short strikes
      - Collect minimum 1/3 of wing width
      - Manage at 50% profit or 21 DTE
      - Adjust by rolling untested side closer when one side tested
    """

    @staticmethod
    def construct(
        put_short: float, put_long: float,
        call_short: float, call_long: float,
        put_credit: float, call_credit: float,
        dte: int,
    ) -> Dict:
        """Build iron condor position metrics."""
        total_credit = put_credit + call_credit
        put_width = put_short - put_long
        call_width = call_long - call_short
        max_width = max(put_width, call_width)
        max_loss = (max_width - total_credit) * 100

        return {
            "put_spread": f"${put_long}/{put_short}",
            "call_spread": f"${call_short}/{call_long}",
            "total_credit": round(total_credit, 2),
            "total_credit_dollars": round(total_credit * 100, 2),
            "max_loss": round(max_loss, 2),
            "risk_reward": round(max_loss / (total_credit * 100), 2),
            "credit_pct_of_width": round(total_credit / max_width * 100, 2),
            "dte": dte,
            "profit_target": round(total_credit * 0.5 * 100, 2),
        }

    @staticmethod
    def adjustment_decision(
        position: IncomePosition,
        underlying_price: float,
    ) -> Dict:
        """
        Iron Condor adjustment logic.
        
        The "untested side roll" technique:
          - When one side is threatened, the other side has mostly decayed
          - Roll the untested side closer to collect additional credit
          - This reduces overall risk and widens the breakeven on tested side
          - Classic "inverted strangle" defense
        """
        strikes = sorted(position.strikes)
        if len(strikes) < 4:
            return {"action": "HOLD", "reason": "Insufficient strike data"}

        put_long, put_short, call_short, call_long = strikes
        pnl_pct = position.pnl_pct
        dte = position.dte_remaining

        # Distance from short strikes
        put_distance = (underlying_price - put_short) / underlying_price * 100
        call_distance = (call_short - underlying_price) / underlying_price * 100

        result = {
            "put_distance_pct": round(put_distance, 2),
            "call_distance_pct": round(call_distance, 2),
            "pnl_pct": round(pnl_pct, 2),
            "dte": dte,
        }

        # 50% profit → close all
        if pnl_pct >= 50:
            result.update({
                "action": "CLOSE_ALL",
                "reason": "50% profit target hit",
                "priority": "HIGH",
            })
            return result

        # 21 DTE → close
        if dte <= 21:
            result.update({
                "action": "CLOSE_ALL",
                "reason": "21 DTE — gamma risk increasing",
                "priority": "MEDIUM",
            })
            return result

        # Put side threatened
        if put_distance < 2.0:
            result.update({
                "action": "ROLL_CALL_SIDE_DOWN",
                "reason": f"Put side within {put_distance:.1f}% — roll untested call spread closer",
                "new_call_short": round(underlying_price + (underlying_price * 0.02), 0),
                "priority": "HIGH",
            })
            return result

        # Call side threatened
        if call_distance < 2.0:
            result.update({
                "action": "ROLL_PUT_SIDE_UP",
                "reason": f"Call side within {call_distance:.1f}% — roll untested put spread closer",
                "new_put_short": round(underlying_price - (underlying_price * 0.02), 0),
                "priority": "HIGH",
            })
            return result

        result.update({
            "action": "HOLD",
            "reason": "Both sides safe — hold for decay",
            "priority": "LOW",
        })
        return result


# ═══════════════════════════════════════════════════════════════════════════
# COVERED CALL SCREENER
# ═══════════════════════════════════════════════════════════════════════════

class CoveredCallScreener:
    """
    Screen for optimal covered call candidates.
    
    From BARREN WUFFET insights:
      - Best CC stocks: moderate volatility, uptrend or range
      - Sell 0.25-0.35 delta for income; 0.15-0.20 for growth
      - Use 30-45 DTE for optimal decay
      - Avoid selling calls through earnings
      - Target 1-3% monthly yield
      - Consider ex-dividend dates (assignment risk)
    """

    @staticmethod
    def screen(
        candidates: List[Dict],
        mode: str = "income",  # "income" or "growth"
    ) -> List[Dict]:
        """Screen for covered call candidates."""
        target_delta = 0.30 if mode == "income" else 0.18
        min_yield = 0.8 if mode == "income" else 0.4

        results = []
        for stock in candidates:
            price = stock.get("price", 0)
            call_premium = stock.get("call_premium", 0)
            call_delta = stock.get("call_delta", 0)
            dte = stock.get("dte", 30)
            iv = stock.get("iv", 0)
            dividend_yield = stock.get("div_yield", 0)
            earnings_in_cycle = stock.get("earnings_in_cycle", False)

            if price <= 0 or call_premium <= 0:
                continue

            # Skip if earnings in cycle (IV crush risk, assignment risk)
            if earnings_in_cycle:
                continue

            # Monthly yield
            monthly_yield = (call_premium / price) * (30 / dte) * 100
            annualized = monthly_yield * 12

            # Total return including dividends
            total_annual = annualized + dividend_yield

            if monthly_yield < min_yield:
                continue

            # Delta check
            delta_score = 1 - abs(call_delta - target_delta) / target_delta

            # Composite score
            score = (
                monthly_yield * 40 +
                delta_score * 30 +
                (iv / 0.30) * 20 +
                dividend_yield * 10
            )

            results.append({
                **stock,
                "monthly_yield_pct": round(monthly_yield, 2),
                "annualized_yield_pct": round(annualized, 2),
                "total_annual_return_pct": round(total_annual, 2),
                "delta_score": round(delta_score, 3),
                "composite_score": round(score, 2),
                "mode": mode,
            })

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results[:15]


# ═══════════════════════════════════════════════════════════════════════════
# YIELD TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class IncomeYieldTracker:
    """Track income strategy yields over time."""

    def __init__(self, starting_capital: float):
        self.capital = starting_capital
        self.trades: List[Dict] = []
        self.monthly_income: Dict[str, float] = {}

    def record_trade(
        self, symbol: str, strategy: str, credit: float,
        result: float, date: str, dte: int,
    ) -> None:
        """Record a completed trade."""
        trade = {
            "symbol": symbol,
            "strategy": strategy,
            "credit": credit,
            "result": result,
            "date": date,
            "dte": dte,
            "return_pct": round(result / self.capital * 100, 4),
        }
        self.trades.append(trade)

        # Track monthly
        month_key = date[:7]  # YYYY-MM
        self.monthly_income[month_key] = self.monthly_income.get(month_key, 0) + result

    def summary(self) -> Dict:
        """Generate yield summary."""
        if not self.trades:
            return {"total_trades": 0}

        total_pnl = sum(t["result"] for t in self.trades)
        winners = [t for t in self.trades if t["result"] > 0]
        losers = [t for t in self.trades if t["result"] <= 0]
        win_rate = len(winners) / len(self.trades) * 100

        avg_win = sum(t["result"] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t["result"] for t in losers) / len(losers) if losers else 0

        monthly_avg = sum(self.monthly_income.values()) / len(self.monthly_income) if self.monthly_income else 0

        return {
            "total_trades": len(self.trades),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / self.capital * 100, 2),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win * len(winners)) / abs(avg_loss * len(losers)), 2) if losers else float("inf"),
            "avg_monthly_income": round(monthly_avg, 2),
            "annualized_return_pct": round(monthly_avg * 12 / self.capital * 100, 2),
            "months_tracked": len(self.monthly_income),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — Options Income Systems v2.7.0")
    print("=" * 55)

    # Demo: Wheel yield calculation
    wheel = WheelEngine(capital=100_000)
    csp_yield = wheel.calculate_csp_yield(strike=495, premium=3.50, dte=35)
    print(f"\nWheel CSP Yield:")
    for k, v in csp_yield.items():
        print(f"  {k}: {v}")

    # Demo: Iron Condor construction
    ic = IronCondorEngine.construct(
        put_short=500, put_long=490,
        call_short=540, call_long=550,
        put_credit=1.50, call_credit=1.20,
        dte=35,
    )
    print(f"\nIron Condor:")
    for k, v in ic.items():
        print(f"  {k}: {v}")

    # Demo: Yield tracking
    tracker = IncomeYieldTracker(100_000)
    tracker.record_trade("SPY", "iron_condor", 2.70, 135, "2025-01-15", 35)
    tracker.record_trade("AAPL", "bull_put_spread", 1.50, 75, "2025-01-22", 30)
    tracker.record_trade("MSFT", "covered_call", 2.00, -50, "2025-02-05", 30)
    tracker.record_trade("QQQ", "iron_condor", 3.10, 155, "2025-02-12", 35)
    tracker.record_trade("AMD", "bull_put_spread", 2.20, 110, "2025-02-20", 30)

    summary = tracker.summary()
    print(f"\nIncome Portfolio Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

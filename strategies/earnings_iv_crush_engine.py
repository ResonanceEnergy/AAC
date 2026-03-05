"""
Earnings IV Crush & Event Volatility Engine — BARREN WUFFET v2.7.0
===================================================================
Systematic strategies for trading around earnings and catalysts:
  - Pre-earnings IV expansion capture
  - Post-earnings IV crush harvesting
  - Straddle pricing vs expected move analysis
  - Event vol surface analysis
  - Calendar spread structures around events

From BARREN WUFFET Insights 456-510:
  - IV rises 2-3 weeks pre-earnings, peaks day before
  - Post-earnings IV crushes 30-70% overnight
  - The "expected move" priced by straddles is exceeded only ~30% of the time
  - Calendar spreads (sell event week, buy post-event) exploit crush
  - Iron condors inside the expected move have ~70% win rate but negative skew
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EarningsEvent:
    """Upcoming earnings event data."""
    symbol: str
    report_date: str
    report_timing: str  # "BMO" (before market open) or "AMC" (after market close)
    stock_price: float
    # IV data
    front_week_iv: float = 0.0
    back_week_iv: float = 0.0
    historical_iv_mean: float = 0.0
    iv_rank: float = 0.0
    # Straddle data
    atm_straddle_price: float = 0.0
    expected_move_dollar: float = 0.0
    expected_move_pct: float = 0.0
    # Historical earnings
    avg_historical_move: float = 0.0
    last_4_moves: List[float] = field(default_factory=list)
    beat_rate: float = 0.0

    @property
    def iv_premium(self) -> float:
        """How much eleveted IV is vs historical."""
        if self.historical_iv_mean == 0:
            return 0
        return (self.front_week_iv / self.historical_iv_mean - 1) * 100

    @property
    def move_vs_expected(self) -> float:
        """Ratio of average actual move to expected move."""
        if self.expected_move_pct == 0:
            return 0
        return self.avg_historical_move / self.expected_move_pct


@dataclass
class CrushTradeSetup:
    """IV crush trade setup."""
    strategy: str
    strikes: List[float]
    credit: float
    max_profit: float
    max_loss: float
    expected_iv_crush: float
    expected_pnl: float
    confidence: float
    edge: str
    risk_notes: List[str]


# ═══════════════════════════════════════════════════════════════════════════
# EXPECTED MOVE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class ExpectedMoveCalculator:
    """
    Calculate the market's expected move from option prices.
    
    Methods:
      1. ATM Straddle: Expected Move ≈ Straddle Price × 0.85
      2. 1-SD Move: Expected Move ≈ Price × IV × sqrt(DTE/365)
      3. Two-Strike: Average of ±1 strike straddle approximation
    
    The market overprice expected moves ~70% of the time.
    This is the core edge for selling premium around events.
    """

    @staticmethod
    def from_straddle(
        stock_price: float, straddle_price: float
    ) -> Dict:
        """Calculate expected move from ATM straddle price."""
        # 85% rule: multiply straddle by 0.85 for expected 1-SD move
        expected_move = straddle_price * 0.85
        expected_pct = expected_move / stock_price * 100

        upper = stock_price + expected_move
        lower = stock_price - expected_move

        return {
            "expected_move": round(expected_move, 2),
            "expected_pct": round(expected_pct, 2),
            "upper_bound": round(upper, 2),
            "lower_bound": round(lower, 2),
            "straddle_cost": round(straddle_price, 2),
            "straddle_pct": round(straddle_price / stock_price * 100, 2),
            "method": "straddle_85pct_rule",
        }

    @staticmethod
    def from_iv(
        stock_price: float, iv: float, dte: int
    ) -> Dict:
        """Calculate expected move from IV."""
        # 1 standard deviation move
        one_sd = stock_price * iv * math.sqrt(dte / 365)
        expected_pct = one_sd / stock_price * 100

        return {
            "expected_move_1sd": round(one_sd, 2),
            "expected_pct_1sd": round(expected_pct, 2),
            "upper_1sd": round(stock_price + one_sd, 2),
            "lower_1sd": round(stock_price - one_sd, 2),
            "expected_move_2sd": round(one_sd * 2, 2),
            "prob_within_1sd": 68.2,
            "prob_within_2sd": 95.4,
            "method": "implied_volatility",
        }

    @staticmethod
    def compare_to_historical(
        expected_move_pct: float,
        historical_moves: List[float],
    ) -> Dict:
        """Compare expected move to historical earnings moves."""
        if not historical_moves:
            return {"comparison": "no_history"}

        avg_move = sum(abs(m) for m in historical_moves) / len(historical_moves)
        max_move = max(abs(m) for m in historical_moves)
        exceeded = sum(1 for m in historical_moves if abs(m) > expected_move_pct)
        exceeded_pct = exceeded / len(historical_moves) * 100

        # Is the expected move overpriced?
        ratio = expected_move_pct / avg_move if avg_move > 0 else 0

        return {
            "expected_move_pct": round(expected_move_pct, 2),
            "avg_historical_move_pct": round(avg_move, 2),
            "max_historical_move_pct": round(max_move, 2),
            "times_exceeded": exceeded,
            "pct_exceeded": round(exceeded_pct, 1),
            "expected_vs_avg_ratio": round(ratio, 2),
            "overpriced": ratio > 1.15,  # >15% over historical avg
            "edge": "SELL_PREMIUM" if ratio > 1.15 else "NEUTRAL" if ratio > 0.85 else "BUY_PREMIUM",
        }


# ═══════════════════════════════════════════════════════════════════════════
# IV CRUSH STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class IVCrushEngine:
    """
    Generate and manage IV crush trading strategies.
    
    Core strategies (ranked by risk/reward):
    
    1. Iron Condor inside expected move (DEFINED, ~70% POP)
       - Sell strikes at ±1 expected move
       - Buy wings $5-10 beyond
       - Credit = max profit
       - Max loss = width - credit
    
    2. Short Straddle/Strangle (UNDEFINED, higher premium)
       - Maximum theta capture
       - Requires portfolio margin
       - Best in high IV environments
    
    3. Calendar Spread (DEFINED, volatility play)
       - Sell event-week expiry, buy post-event expiry
       - Profits from IV crush on front leg
       - Risk: if stock moves too far, both legs lose
    
    4. Butterfly at expected pin (DEFINED, high R:R)
       - Bet on stock staying near current price
       - Cheap entry, 10:1+ potential
       - Works when market overprices the move
    """

    def __init__(self, event: EarningsEvent):
        self.event = event

    def iron_condor_setup(
        self, wing_width: float = 5.0
    ) -> CrushTradeSetup:
        """Build iron condor for IV crush play."""
        S = self.event.stock_price
        em = self.event.expected_move_dollar
        if em == 0:
            em = S * self.event.front_week_iv * math.sqrt(1 / 365)

        # Place short strikes at the expected move
        put_short = round(S - em, 0)
        call_short = round(S + em, 0)
        put_long = put_short - wing_width
        call_long = call_short + wing_width

        # Estimate credit (rough: ~30-40% of width in high IV)
        est_credit = wing_width * 0.35
        max_loss = (wing_width - est_credit) * 100

        # Expected IV crush
        crush = self.event.front_week_iv * 0.5  # 50% typical
        expected_pnl = est_credit * 0.50 * 100  # 50% close

        return CrushTradeSetup(
            strategy="Iron Condor (Earnings IV Crush)",
            strikes=[put_long, put_short, call_short, call_long],
            credit=round(est_credit, 2),
            max_profit=round(est_credit * 100, 2),
            max_loss=round(max_loss, 2),
            expected_iv_crush=round(crush, 4),
            expected_pnl=round(expected_pnl, 2),
            confidence=70 if self.event.move_vs_expected < 1 else 55,
            edge=f"Expected move overpriced {self.event.move_vs_expected:.0%} of time",
            risk_notes=[
                "Close IMMEDIATELY after earnings release",
                "If stock gaps beyond short strike → max loss likely",
                "Do not hold through if uncertain about report timing",
                f"Historical avg move: {self.event.avg_historical_move:.1f}% "
                f"vs expected {self.event.expected_move_pct:.1f}%",
            ],
        )

    def calendar_spread_setup(self) -> CrushTradeSetup:
        """Build calendar spread for IV crush."""
        S = self.event.stock_price
        strike = round(S, 0)  # ATM

        # Front IV crushes, back IV holds → spread widens
        front_iv = self.event.front_week_iv
        back_iv = self.event.back_week_iv or front_iv * 0.85

        iv_diff = front_iv - back_iv
        est_debit = S * 0.015  # ~1.5% of stock price typical cost
        crush_target = front_iv * 0.50  # 50% crush
        expected_pnl = est_debit * 0.30 * 100  # 30% of debit as profit

        return CrushTradeSetup(
            strategy="Calendar Spread (Sell Event / Buy Post-Event)",
            strikes=[strike],
            credit=-round(est_debit, 2),  # Debit trade
            max_profit=round(est_debit * 1.5 * 100, 2),  # Can 1.5x
            max_loss=round(est_debit * 100, 2),
            expected_iv_crush=round(crush_target, 4),
            expected_pnl=round(expected_pnl, 2),
            confidence=65,
            edge=f"Front IV {front_iv:.0%} crushed to ~{front_iv - crush_target:.0%}, back holds at ~{back_iv:.0%}",
            risk_notes=[
                "Stock must stay near strike for max profit",
                f"If stock moves > {S * 0.03:.0f} points, calendar loses",
                "Close morning after earnings; don't hold through theta decay",
                "Best when IV term structure is in backwardation",
            ],
        )

    def butterfly_pin_setup(self, width: float = 5.0) -> CrushTradeSetup:
        """Build butterfly for pin play (stock stays put after earnings)."""
        S = self.event.stock_price
        center = round(S, 0)
        lower = center - width
        upper = center + width

        est_debit = width * 0.10  # ~10% of width
        max_profit = (width - est_debit) * 100
        max_loss = est_debit * 100

        return CrushTradeSetup(
            strategy="ATM Butterfly (Earnings Pin Play)",
            strikes=[lower, center, upper],
            credit=-round(est_debit, 2),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            expected_iv_crush=round(self.event.front_week_iv * 0.5, 4),
            expected_pnl=round(max_profit * 0.15, 2),  # 15% hit rate on pin
            confidence=30,  # Low prob but high R:R
            edge=f"10:1 R:R if stock pins. Cost only ${max_loss:.0f}.",
            risk_notes=[
                "Low probability (~15-20%) but excellent risk/reward",
                "Use as portfolio lottery ticket, not core position",
                "Stock must close within butterfly strikes",
                "Best for stocks with history of small post-earnings moves",
            ],
        )

    def rank_strategies(self) -> List[Dict]:
        """Rank all strategies for this earnings event."""
        setups = [
            self.iron_condor_setup(),
            self.calendar_spread_setup(),
            self.butterfly_pin_setup(),
        ]

        ranked = []
        for setup in setups:
            # Composite score: confidence * edge quality
            score = setup.confidence * 0.5
            if setup.expected_pnl > 0:
                score += min(30, setup.expected_pnl / 100 * 10)
            if setup.max_loss > 0 and setup.max_profit / setup.max_loss > 2:
                score += 10
            if self.event.move_vs_expected < 0.8:
                score -= 10  # Stock moves are BIGGER than expected → don't sell

            ranked.append({
                "strategy": setup.strategy,
                "score": round(score, 1),
                "credit": setup.credit,
                "max_profit": setup.max_profit,
                "max_loss": setup.max_loss,
                "confidence": setup.confidence,
                "edge": setup.edge,
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked


# ═══════════════════════════════════════════════════════════════════════════
# EARNINGS SEASON SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class EarningsSeasonScanner:
    """
    Scan for optimal earnings plays across multiple stocks.
    
    From BARREN WUFFET insights:
      - Best earnings sells: IVR > 60%, move_vs_expected < 0.9
      - Avoid: biotech binary events, first-time reporters, low liquidity
      - Diversify: max 2-3 earnings plays per week
      - Size small: No more than 1-2% risk per earnings play
      - Timing: Enter 1-3 days before, close morning after
    """

    @staticmethod
    def scan(events: List[EarningsEvent]) -> List[Dict]:
        """Scan and rank earnings events for IV crush plays."""
        ranked = []

        for event in events:
            # Scoring criteria
            score = 0
            flags = []

            # IV premium (higher = better for selling)
            if event.iv_premium > 50:
                score += 25
                flags.append("HIGH_IV_PREMIUM")
            elif event.iv_premium > 30:
                score += 15

            # Expected move vs historical (lower ratio = overpriced = edge)
            if event.move_vs_expected < 0.8:
                score += 25
                flags.append("MOVE_OVERPRICED")
            elif event.move_vs_expected < 1.0:
                score += 15

            # IV Rank
            if event.iv_rank > 70:
                score += 20
                flags.append("HIGH_IVR")
            elif event.iv_rank > 50:
                score += 10

            # Historical consistency (low variance in moves = predictable)
            if event.last_4_moves:
                move_std = (sum((m - sum(event.last_4_moves) / len(event.last_4_moves))**2
                                for m in event.last_4_moves) / len(event.last_4_moves)) ** 0.5
                if move_std < 2:
                    score += 15
                    flags.append("CONSISTENT_MOVER")

            ranked.append({
                "symbol": event.symbol,
                "report_date": event.report_date,
                "timing": event.report_timing,
                "score": round(score, 1),
                "iv_premium": round(event.iv_premium, 1),
                "expected_move_pct": round(event.expected_move_pct, 2),
                "avg_actual_move_pct": round(event.avg_historical_move, 2),
                "move_ratio": round(event.move_vs_expected, 2),
                "iv_rank": round(event.iv_rank, 1),
                "flags": flags,
                "recommended": score >= 50,
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — Earnings IV Crush Engine v2.7.0")
    print("=" * 60)

    # Demo: AAPL earnings setup
    event = EarningsEvent(
        symbol="AAPL",
        report_date="2025-07-31",
        report_timing="AMC",
        stock_price=210.0,
        front_week_iv=0.45,
        back_week_iv=0.28,
        historical_iv_mean=0.25,
        iv_rank=85,
        atm_straddle_price=12.50,
        expected_move_dollar=10.63,  # 12.50 * 0.85
        expected_move_pct=5.06,
        avg_historical_move=3.8,
        last_4_moves=[2.5, -4.1, 3.2, -5.7],
        beat_rate=0.75,
    )

    # Expected move analysis
    em = ExpectedMoveCalculator.from_straddle(event.stock_price, event.atm_straddle_price)
    print(f"\nExpected Move (Straddle Method):")
    for k, v in em.items():
        print(f"  {k}: {v}")

    cmp = ExpectedMoveCalculator.compare_to_historical(
        event.expected_move_pct, [abs(m) for m in event.last_4_moves]
    )
    print(f"\nHistorical Comparison:")
    for k, v in cmp.items():
        print(f"  {k}: {v}")

    # Strategy ranking
    engine = IVCrushEngine(event)
    strategies = engine.rank_strategies()
    print(f"\nStrategy Rankings:")
    for i, s in enumerate(strategies, 1):
        print(f"\n  #{i}: {s['strategy']}")
        print(f"      Score: {s['score']} | Confidence: {s['confidence']}%")
        print(f"      Max Profit: ${s['max_profit']:,.0f} | Max Loss: ${s['max_loss']:,.0f}")
        print(f"      Edge: {s['edge']}")

    # Earnings season scan
    events = [
        event,  # AAPL
        EarningsEvent(
            symbol="MSFT", report_date="2025-07-29", report_timing="AMC",
            stock_price=440, front_week_iv=0.38, historical_iv_mean=0.22,
            iv_rank=80, expected_move_pct=4.2, avg_historical_move=3.5,
            last_4_moves=[3.0, -2.8, 4.5, -3.2]
        ),
        EarningsEvent(
            symbol="TSLA", report_date="2025-07-22", report_timing="AMC",
            stock_price=280, front_week_iv=0.65, historical_iv_mean=0.45,
            iv_rank=70, expected_move_pct=8.5, avg_historical_move=9.2,
            last_4_moves=[12.0, -7.5, 8.3, -11.0]
        ),
    ]

    scan = EarningsSeasonScanner.scan(events)
    print(f"\n\nEarnings Season Scan:")
    for s in scan:
        rec = "✅ RECOMMENDED" if s["recommended"] else "⚠️ SKIP"
        print(f"\n  {s['symbol']} ({s['report_date']} {s['timing']}) — {rec}")
        print(f"    Score: {s['score']} | IVR: {s['iv_rank']}%")
        print(f"    Expected Move: {s['expected_move_pct']}% vs Actual Avg: {s['avg_actual_move_pct']}%")
        print(f"    Flags: {', '.join(s['flags']) if s['flags'] else 'None'}")

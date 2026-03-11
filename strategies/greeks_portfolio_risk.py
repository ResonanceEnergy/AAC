"""
Greeks Portfolio Risk Manager — BARREN WUFFET v2.7.0
=====================================================
Portfolio-level Greeks aggregation, risk management, and hedging engine.
Monitors Delta, Gamma, Theta, Vega, and Vanna/Charm across all positions
and generates hedging recommendations.

From BARREN WUFFET Insights 536-570 (Risk Management & Greeks Mastery):
  - Portfolio delta should be managed to target range (e.g., ±50)
  - Gamma risk explodes near expiration → reduce positions at 7 DTE
  - Theta is your friend when short vol, your enemy when long
  - Vega correlation across positions can create hidden risk
  - Second-order Greeks (vanna, charm) drive P/L in trending markets
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionGreeks:
    """Greeks for a single position (possibly multi-leg)."""
    symbol: str
    strategy: str
    quantity: int
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    # Second-order
    vanna: float = 0.0     # dDelta/dIV
    charm: float = 0.0     # dDelta/dTime
    vomma: float = 0.0     # dVega/dIV
    dte: int = 0
    notional: float = 0.0

    @property
    def dollar_delta(self) -> float:
        """Delta in dollar terms."""
        return self.delta * self.notional

    @property
    def gamma_risk_1pct(self) -> float:
        """P/L impact of 1% move from gamma alone."""
        return 0.5 * self.gamma * (self.notional * 0.01) ** 2

    @property
    def theta_per_day(self) -> float:
        """Daily theta in dollars."""
        return self.theta * self.quantity * 100

    @property
    def vega_per_1pct(self) -> float:
        """P/L impact of 1% IV change."""
        return self.vega * self.quantity * 100


@dataclass
class PortfolioRiskSnapshot:
    """Complete portfolio risk snapshot."""
    timestamp: str
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    total_vanna: float = 0.0
    total_charm: float = 0.0
    # Dollar measures
    dollar_delta: float = 0.0
    gamma_risk_1pct: float = 0.0
    daily_theta: float = 0.0
    vega_risk_1pct: float = 0.0
    # Limits
    delta_limit: float = 100.0
    gamma_limit: float = 50.0
    theta_limit: float = -500.0
    vega_limit: float = 200.0
    # Risk level
    risk_level: RiskLevel = RiskLevel.LOW
    alerts: List[str] = field(default_factory=list)
    num_positions: int = 0
    beta_weighted_delta: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "delta": round(self.total_delta, 2),
            "gamma": round(self.total_gamma, 3),
            "theta": round(self.total_theta, 2),
            "vega": round(self.total_vega, 2),
            "dollar_delta": round(self.dollar_delta, 0),
            "daily_theta": round(self.daily_theta, 2),
            "gamma_risk_1pct": round(self.gamma_risk_1pct, 2),
            "vega_risk_1pct": round(self.vega_risk_1pct, 2),
            "risk_level": self.risk_level.value,
            "alerts": self.alerts,
            "positions": self.num_positions,
            "beta_weighted_delta": round(self.beta_weighted_delta, 2),
        }


@dataclass
class HedgeRecommendation:
    """Hedging recommendation to reduce portfolio risk."""
    action: str
    instrument: str
    quantity: int
    rationale: str
    priority: str           # "immediate", "end_of_day", "optional"
    estimated_cost: float
    greeks_impact: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO GREEKS AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════

class PortfolioGreeksEngine:
    """
    Aggregate and analyze Greeks across an entire options portfolio.
    
    Key concepts from BARREN WUFFET doctrine:
      - Beta-weight all deltas to SPY for uniform risk measurement
      - Track "dollar delta" not just option delta
      - Gamma exposure spikes near OPEX → reduce at T-3
      - Vega correlation: selling vol on 5 tech stocks ≠ diversified
      - Theta is NOT free money — it's compensation for gamma risk
    """

    def __init__(
        self,
        delta_limit: float = 100.0,
        gamma_limit: float = 50.0,
        vega_limit: float = 200.0,
        theta_floor: float = -500.0,
    ):
        self.positions: List[PositionGreeks] = []
        self.delta_limit = delta_limit
        self.gamma_limit = gamma_limit
        self.vega_limit = vega_limit
        self.theta_floor = theta_floor

    def add_position(self, pos: PositionGreeks) -> None:
        """Add a position to the portfolio."""
        self.positions.append(pos)

    def clear_positions(self) -> None:
        """Clear all positions."""
        self.positions = []

    def compute_snapshot(
        self, spy_price: float = 520.0,
        beta_map: Optional[Dict[str, float]] = None,
    ) -> PortfolioRiskSnapshot:
        """
        Compute full portfolio risk snapshot.
        
        Args:
            spy_price: SPY price for beta-weighting
            beta_map: {symbol: beta} for beta-weighted delta
        """
        beta_map = beta_map or {}
        snap = PortfolioRiskSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            delta_limit=self.delta_limit,
            gamma_limit=self.gamma_limit,
            theta_limit=self.theta_floor,
            vega_limit=self.vega_limit,
            num_positions=len(self.positions),
        )

        for pos in self.positions:
            snap.total_delta += pos.delta * pos.quantity
            snap.total_gamma += pos.gamma * pos.quantity
            snap.total_theta += pos.theta * pos.quantity
            snap.total_vega += pos.vega * pos.quantity
            snap.total_vanna += pos.vanna * pos.quantity
            snap.total_charm += pos.charm * pos.quantity

            snap.dollar_delta += pos.dollar_delta
            snap.gamma_risk_1pct += pos.gamma_risk_1pct
            snap.daily_theta += pos.theta_per_day
            snap.vega_risk_1pct += pos.vega_per_1pct

            # Beta-weighted delta
            beta = beta_map.get(pos.symbol, 1.0)
            bw_factor = (pos.notional / (spy_price * 100)) * beta if spy_price > 0 else 1
            snap.beta_weighted_delta += pos.delta * pos.quantity * bw_factor

        # Alerts
        snap.alerts = self._generate_alerts(snap)
        snap.risk_level = self._classify_risk(snap)

        return snap

    def _generate_alerts(self, snap: PortfolioRiskSnapshot) -> List[str]:
        """Generate risk alerts based on limits."""
        alerts = []

        if abs(snap.total_delta) > self.delta_limit:
            direction = "long" if snap.total_delta > 0 else "short"
            alerts.append(
                f"DELTA BREACH: Portfolio delta {snap.total_delta:+.1f} "
                f"exceeds ±{self.delta_limit} limit ({direction} bias)"
            )

        if abs(snap.total_gamma) > self.gamma_limit:
            alerts.append(
                f"GAMMA ELEVATED: Portfolio gamma {snap.total_gamma:+.2f} "
                f"exceeds ±{self.gamma_limit} limit"
            )

        if snap.daily_theta < self.theta_floor:
            alerts.append(
                f"THETA DRAG: Daily theta ${snap.daily_theta:,.0f} "
                f"below ${self.theta_floor:,.0f} floor (too much long vol)"
            )

        if abs(snap.total_vega) > self.vega_limit:
            direction = "long" if snap.total_vega > 0 else "short"
            alerts.append(
                f"VEGA RISK: Portfolio vega {snap.total_vega:+.1f} "
                f"exceeds ±{self.vega_limit} limit ({direction} vol)"
            )

        # Gamma near expiration
        near_expiry = [p for p in self.positions if p.dte <= 7 and abs(p.gamma) > 0.01]
        if near_expiry:
            symbols = ", ".join(set(p.symbol for p in near_expiry))
            total_near_gamma = sum(p.gamma * p.quantity for p in near_expiry)
            alerts.append(
                f"EXPIRATION RISK: {len(near_expiry)} positions within 7 DTE "
                f"({symbols}), gamma = {total_near_gamma:+.2f}"
            )

        # Concentration check
        symbol_delta = {}
        for p in self.positions:
            symbol_delta[p.symbol] = symbol_delta.get(p.symbol, 0) + p.delta * p.quantity
        for sym, delta in symbol_delta.items():
            if abs(delta) > self.delta_limit * 0.3:
                alerts.append(
                    f"CONCENTRATION: {sym} delta {delta:+.1f} is "
                    f"{abs(delta / self.delta_limit) * 100:.0f}% of portfolio limit"
                )

        return alerts

    def _classify_risk(self, snap: PortfolioRiskSnapshot) -> RiskLevel:
        """Classify overall portfolio risk level."""
        score = 0
        delta_pct = abs(snap.total_delta) / self.delta_limit * 100
        gamma_pct = abs(snap.total_gamma) / self.gamma_limit * 100
        vega_pct = abs(snap.total_vega) / self.vega_limit * 100

        # Delta scoring
        if delta_pct > 120:
            score += 3
        elif delta_pct > 80:
            score += 2
        elif delta_pct > 50:
            score += 1

        # Gamma scoring
        if gamma_pct > 120:
            score += 3
        elif gamma_pct > 80:
            score += 2
        elif gamma_pct > 50:
            score += 1

        # Vega scoring
        if vega_pct > 120:
            score += 2
        elif vega_pct > 80:
            score += 1

        # Alert count
        score += len(snap.alerts)

        if score >= 8:
            return RiskLevel.CRITICAL
        elif score >= 5:
            return RiskLevel.HIGH
        elif score >= 3:
            return RiskLevel.ELEVATED
        elif score >= 1:
            return RiskLevel.MODERATE
        return RiskLevel.LOW


# ═══════════════════════════════════════════════════════════════════════════
# HEDGING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class HedgingEngine:
    """
    Generate hedging recommendations to reduce portfolio risk.
    
    Hedging hierarchy (from insights):
      1. Delta: hedge with shares or futures (cheapest, most liquid)
      2. Gamma: reduce near-expiry positions or buy protection
      3. Vega: calendar/diagonal adjustments or VIX products
      4. Tail risk: OTM puts or VIX calls for catastrophic protection
    """

    def __init__(self, spy_price: float = 520.0):
        self.spy_price = spy_price

    def generate_hedges(
        self, snapshot: PortfolioRiskSnapshot
    ) -> List[HedgeRecommendation]:
        """Generate prioritized hedging recommendations."""
        hedges = []

        # Delta hedge
        if abs(snapshot.total_delta) > snapshot.delta_limit * 0.7:
            shares = -int(snapshot.total_delta * 100)
            hedges.append(HedgeRecommendation(
                action="BUY_SHARES" if shares > 0 else "SELL_SHARES",
                instrument="SPY" if abs(shares) > 50 else "underlying",
                quantity=abs(shares),
                rationale=(
                    f"Delta {snapshot.total_delta:+.1f} exceeds 70% of limit. "
                    f"{'Buy' if shares > 0 else 'Sell'} {abs(shares)} shares to neutralize."
                ),
                priority="immediate" if abs(snapshot.total_delta) > snapshot.delta_limit else "end_of_day",
                estimated_cost=abs(shares) * self.spy_price * 0.001,  # Slippage estimate
                greeks_impact={"delta": -snapshot.total_delta},
            ))

        # Gamma hedge (near expiry)
        if abs(snapshot.total_gamma) > snapshot.gamma_limit * 0.8:
            hedges.append(HedgeRecommendation(
                action="REDUCE_NEAR_EXPIRY",
                instrument="All positions with DTE < 7",
                quantity=0,
                rationale=(
                    f"Portfolio gamma {snapshot.total_gamma:+.2f} elevated. "
                    f"Close or roll near-expiry positions to reduce intraday risk."
                ),
                priority="immediate",
                estimated_cost=0,
                greeks_impact={"gamma": -snapshot.total_gamma * 0.5},
            ))

        # Vega hedge
        if abs(snapshot.total_vega) > snapshot.vega_limit * 0.8:
            direction = "short" if snapshot.total_vega > 0 else "long"
            hedges.append(HedgeRecommendation(
                action=f"ADD_{direction.upper()}_VOL_HEDGE",
                instrument="Calendar Spread or VIX product",
                quantity=int(abs(snapshot.total_vega) / 10),
                rationale=(
                    f"Portfolio vega {snapshot.total_vega:+.1f} exceeds 80% limit. "
                    f"{'Sell' if direction == 'short' else 'Buy'} vol to reduce exposure."
                ),
                priority="end_of_day",
                estimated_cost=abs(snapshot.total_vega) * 0.5,
                greeks_impact={"vega": -snapshot.total_vega * 0.3},
            ))

        # Tail risk hedge (always consider)
        if not any(h.action.startswith("TAIL") for h in hedges):
            total_notional = abs(snapshot.dollar_delta)
            if total_notional > 100_000:
                hedges.append(HedgeRecommendation(
                    action="TAIL_RISK_PROTECTION",
                    instrument="SPY put spread (5% OTM, 30-60 DTE)",
                    quantity=max(1, int(total_notional / (self.spy_price * 100 * 5))),
                    rationale=(
                        f"Portfolio notional ~${total_notional:,.0f}. "
                        f"Consider 1-2% allocation to tail hedges (OTM put spreads)."
                    ),
                    priority="optional",
                    estimated_cost=total_notional * 0.003,
                    greeks_impact={"delta": -5, "vega": 2},
                ))

        return hedges


# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class PositionSizer:
    """
    Options-specific position sizing.
    
    Rules from BARREN WUFFET insights:
      - Risk no more than 1-5% of capital per trade
      - Maximum portfolio notional: 50% of buying power
      - Adjust size based on IVR (bigger when high, smaller when low)
      - Scale down near portfolio Greek limits
      - Never let a single symbol exceed 15% of total exposure
    """

    def __init__(
        self, total_capital: float,
        max_risk_per_trade: float = 0.03,  # 3%
        max_allocation: float = 0.50,       # 50% deployed
        max_single_name: float = 0.15,      # 15% per name
    ):
        self.capital = total_capital
        self.max_risk = max_risk_per_trade
        self.max_allocation = max_allocation
        self.max_name = max_single_name

    def size_credit_spread(
        self, spread_width: float, credit: float,
        current_allocation: float = 0.0,
    ) -> Dict:
        """Size a credit spread position."""
        max_loss_per = (spread_width - credit) * 100
        max_risk_dollars = self.capital * self.max_risk
        available = self.capital * self.max_allocation - current_allocation

        # By risk
        contracts_by_risk = int(max_risk_dollars / max_loss_per) if max_loss_per > 0 else 0

        # By allocation
        contracts_by_alloc = int(available / (spread_width * 100)) if spread_width > 0 else 0

        # Take minimum
        contracts = max(1, min(contracts_by_risk, contracts_by_alloc))

        return {
            "contracts": contracts,
            "max_loss": round(max_loss_per * contracts, 2),
            "max_profit": round(credit * 100 * contracts, 2),
            "capital_at_risk_pct": round(max_loss_per * contracts / self.capital * 100, 2),
            "rationale": (
                f"{contracts} contracts: risk ${max_loss_per * contracts:,.0f} "
                f"({max_loss_per * contracts / self.capital * 100:.1f}% of capital)"
            ),
        }

    def size_naked_position(
        self, premium: float, underlying_price: float,
        margin_requirement: float = 0.20,  # 20% of underlying
    ) -> Dict:
        """Size a naked (undefined risk) position."""
        margin_per = underlying_price * margin_requirement * 100
        max_risk_dollars = self.capital * self.max_risk * 0.5  # Halve for undefined risk

        contracts_by_risk = int(max_risk_dollars / margin_per) if margin_per > 0 else 0
        contracts = max(1, min(contracts_by_risk, 3))  # Cap at 3 for naked

        return {
            "contracts": contracts,
            "margin_required": round(margin_per * contracts, 2),
            "premium_collected": round(premium * 100 * contracts, 2),
            "capital_deployed_pct": round(margin_per * contracts / self.capital * 100, 2),
            "warning": "UNDEFINED RISK: Use strict stop-loss at 2x premium collected",
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — Greeks Portfolio Risk Manager v2.7.0")
    print("=" * 60)

    engine = PortfolioGreeksEngine(delta_limit=80, gamma_limit=30, vega_limit=150)

    # Add sample positions
    positions = [
        PositionGreeks("SPY", "iron_condor", 5, delta=-3, gamma=0.8, theta=-1.5, vega=-4, dte=30, notional=260_000),
        PositionGreeks("AAPL", "bull_put_spread", 3, delta=8, gamma=0.3, theta=-0.8, vega=-2, dte=25, notional=57_000),
        PositionGreeks("MSFT", "covered_call", 2, delta=45, gamma=-0.2, theta=-0.3, vega=-1, dte=35, notional=84_000),
        PositionGreeks("TSLA", "short_strangle", 2, delta=-5, gamma=1.2, theta=-2.0, vega=-6, dte=14, notional=58_000),
        PositionGreeks("QQQ", "calendar_spread", 4, delta=2, gamma=-0.5, theta=0.4, vega=8, dte=45, notional=92_000),
        PositionGreeks("AMD", "long_straddle", 2, delta=1, gamma=0.9, theta=-1.8, vega=5, dte=5, notional=32_000),
    ]

    for pos in positions:
        engine.add_position(pos)

    # Compute snapshot
    snap = engine.compute_snapshot(
        spy_price=520,
        beta_map={"SPY": 1.0, "AAPL": 1.2, "MSFT": 1.1, "TSLA": 1.8, "QQQ": 1.05, "AMD": 1.5},
    )

    print(f"\nPortfolio Risk Snapshot:")
    for k, v in snap.to_dict().items():
        print(f"  {k}: {v}")

    # Hedging recommendations
    hedger = HedgingEngine(spy_price=520)
    hedges = hedger.generate_hedges(snap)
    print(f"\n{'=' * 40}")
    print(f"Hedging Recommendations ({len(hedges)}):")
    for h in hedges:
        print(f"\n  [{h.priority.upper()}] {h.action}")
        print(f"    Instrument: {h.instrument}")
        print(f"    Qty: {h.quantity}")
        print(f"    Rationale: {h.rationale}")
        print(f"    Est. Cost: ${h.estimated_cost:,.0f}")

    # Position sizing
    sizer = PositionSizer(total_capital=200_000)
    spread_size = sizer.size_credit_spread(spread_width=5, credit=1.50)
    print(f"\nPosition Sizing (Credit Spread $5 wide, $1.50 credit):")
    for k, v in spread_size.items():
        print(f"  {k}: {v}")

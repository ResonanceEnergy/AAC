"""
Macro Crisis Put Strategy — AAC v3.0
=====================================
Systematic put-buying strategy designed to capitalize on macro crisis events:
  - War/geopolitical escalation (Iran-Hormuz crisis, oil shock)
  - Private credit collapse (Blackstone/Morgan Stanley/BlackRock fund redemptions)
  - Insurance/backstop stress (credit default contagion)
  - Stagflation (rising inflation + slowing growth simultaneously)

Thesis (March 2026):
  - Iran war entering week 3, oil >$100/bbl, Brent $95.50
  - Strait of Hormuz blockade disrupting 21% of global oil transit
  - Private credit funds capping redemptions: Blackstone BCRED, Morgan Stanley PIF,
    BlackRock HLEND, Blue Owl all restricting withdrawals
  - JPMorgan marking down private credit loan portfolios
  - Hedge funds "aggressively" shorting financials (Goldman report)
  - Fed trapped: core PCE 3.1%, GDP 0.7% — stagflation setup
  - Fed meeting Wednesday, no rate cut expected, second cut pushed to December
  - Citadel Securities ditched bearish Treasury call after rout
  - Dollar weakening (worst day in over a month)
  - Gold at $5,003/oz — risk-off

Targets: SPY, QQQ, IWM, XLF (financials), HYG (high yield credit),
         LQD (investment grade), KRE (regional banks), BKLN (leveraged loans)

Execution: IBKR via TWS API (options-enabled connector)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CRISIS INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

class CrisisVector(Enum):
    """Classification of macro crisis catalysts."""
    WAR_ESCALATION = "war_escalation"
    OIL_SHOCK = "oil_shock"
    PRIVATE_CREDIT_COLLAPSE = "private_credit_collapse"
    INSURANCE_BACKSTOP_STRESS = "insurance_backstop_stress"
    STAGFLATION = "stagflation"
    CREDIT_CONTAGION = "credit_contagion"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class CrisisSignal:
    """A single crisis indicator reading."""
    vector: CrisisVector
    severity: float  # 0.0 to 1.0
    description: str
    data_source: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_critical(self) -> bool:
        return self.severity >= 0.7


@dataclass
class CrisisAssessment:
    """Composite crisis assessment from all vectors."""
    signals: List[CrisisSignal] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def composite_severity(self) -> float:
        if not self.signals:
            return 0.0
        return sum(s.severity for s in self.signals) / len(self.signals)

    @property
    def max_severity(self) -> float:
        if not self.signals:
            return 0.0
        return max(s.severity for s in self.signals)

    @property
    def critical_count(self) -> int:
        return sum(1 for s in self.signals if s.is_critical)

    @property
    def should_deploy_puts(self) -> bool:
        """Deploy puts if composite > 0.5 OR any 2+ critical signals."""
        return self.composite_severity > 0.5 or self.critical_count >= 2


# ═══════════════════════════════════════════════════════════════════════════
# PUT POSITION SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PutTarget:
    """Specification for a put position."""
    symbol: str
    description: str
    crisis_vectors: List[CrisisVector]
    target_delta: float  # Negative, e.g. -0.30
    target_dte: int  # Days to expiration
    max_premium_pct: float  # Max % of account to spend on this position
    priority: int  # 1 = highest priority
    otm_pct: float  # How far out of the money (e.g. 0.05 = 5% OTM)

    @property
    def strike_multiplier(self) -> float:
        return 1.0 - self.otm_pct


# The playbook — ordered by priority
PUT_PLAYBOOK: List[PutTarget] = [
    # TIER 1: Maximum conviction (war + credit crisis direct exposure)
    PutTarget(
        symbol="SPY",
        description="S&P 500 broad market hedge — war + stagflation + credit contagion",
        crisis_vectors=[CrisisVector.WAR_ESCALATION, CrisisVector.STAGFLATION,
                        CrisisVector.CREDIT_CONTAGION],
        target_delta=-0.30,
        target_dte=45,
        max_premium_pct=3.0,
        priority=1,
        otm_pct=0.05,
    ),
    PutTarget(
        symbol="XLF",
        description="Financials — private credit exposure, bank loan markdowns",
        crisis_vectors=[CrisisVector.PRIVATE_CREDIT_COLLAPSE,
                        CrisisVector.CREDIT_CONTAGION],
        target_delta=-0.35,
        target_dte=45,
        max_premium_pct=2.5,
        priority=1,
        otm_pct=0.07,
    ),
    PutTarget(
        symbol="HYG",
        description="High yield credit — direct private credit / junk bond contagion",
        crisis_vectors=[CrisisVector.PRIVATE_CREDIT_COLLAPSE,
                        CrisisVector.LIQUIDITY_CRISIS],
        target_delta=-0.30,
        target_dte=60,
        max_premium_pct=2.5,
        priority=1,
        otm_pct=0.04,
    ),

    # TIER 2: High conviction
    PutTarget(
        symbol="KRE",
        description="Regional banks — CRE exposure, deposit flight, loan losses",
        crisis_vectors=[CrisisVector.PRIVATE_CREDIT_COLLAPSE,
                        CrisisVector.INSURANCE_BACKSTOP_STRESS],
        target_delta=-0.35,
        target_dte=45,
        max_premium_pct=2.0,
        priority=2,
        otm_pct=0.08,
    ),
    PutTarget(
        symbol="QQQ",
        description="Nasdaq — AI valuation bubble + software sector writedowns",
        crisis_vectors=[CrisisVector.PRIVATE_CREDIT_COLLAPSE,
                        CrisisVector.STAGFLATION],
        target_delta=-0.25,
        target_dte=45,
        max_premium_pct=2.0,
        priority=2,
        otm_pct=0.06,
    ),
    PutTarget(
        symbol="IWM",
        description="Russell 2000 small caps — most rate-sensitive, credit-dependent",
        crisis_vectors=[CrisisVector.STAGFLATION,
                        CrisisVector.CREDIT_CONTAGION],
        target_delta=-0.30,
        target_dte=45,
        max_premium_pct=2.0,
        priority=2,
        otm_pct=0.07,
    ),

    # TIER 3: Targeted plays
    PutTarget(
        symbol="BKLN",
        description="Leveraged loans — direct private credit fund collateral stress",
        crisis_vectors=[CrisisVector.PRIVATE_CREDIT_COLLAPSE,
                        CrisisVector.LIQUIDITY_CRISIS],
        target_delta=-0.30,
        target_dte=60,
        max_premium_pct=1.5,
        priority=3,
        otm_pct=0.04,
    ),
    PutTarget(
        symbol="LQD",
        description="Investment grade corporate — credit spread widening",
        crisis_vectors=[CrisisVector.CREDIT_CONTAGION,
                        CrisisVector.INSURANCE_BACKSTOP_STRESS],
        target_delta=-0.25,
        target_dte=60,
        max_premium_pct=1.5,
        priority=3,
        otm_pct=0.03,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# CRISIS MONITOR — Assesses current macro environment
# ═══════════════════════════════════════════════════════════════════════════

class CrisisMonitor:
    """
    Monitors macro crisis indicators and produces a composite assessment.
    Uses available data sources (CoinGecko for crypto, FRED for macro,
    news feeds for events).
    """

    def __init__(self):
        self.last_assessment: Optional[CrisisAssessment] = None
        self.logger = logging.getLogger("crisis_monitor")

    def assess(
        self,
        oil_price: Optional[float] = None,
        vix_level: Optional[float] = None,
        gold_price: Optional[float] = None,
        ten_year_yield: Optional[float] = None,
        core_pce: Optional[float] = None,
        gdp_growth: Optional[float] = None,
        credit_spread_bps: Optional[float] = None,
        private_credit_redemption_pct: Optional[float] = None,
        hormuz_blocked: bool = False,
        war_active: bool = False,
    ) -> CrisisAssessment:
        """
        Produce a crisis assessment from available macro data.
        Pass None for unavailable data points — they'll be skipped.
        """
        signals = []

        # ── War / Geopolitical ──
        if war_active:
            severity = 0.8
            if hormuz_blocked:
                severity = 0.95
            signals.append(CrisisSignal(
                vector=CrisisVector.WAR_ESCALATION,
                severity=severity,
                description=f"Active war zone. Hormuz blocked={hormuz_blocked}",
                data_source="manual/news",
            ))

        # ── Oil Shock ──
        if oil_price is not None:
            if oil_price > 120:
                severity = 0.95
            elif oil_price > 100:
                severity = 0.80
            elif oil_price > 90:
                severity = 0.60
            elif oil_price > 80:
                severity = 0.30
            else:
                severity = 0.10
            signals.append(CrisisSignal(
                vector=CrisisVector.OIL_SHOCK,
                severity=severity,
                description=f"Oil at ${oil_price:.2f}/bbl",
                data_source="market_data",
            ))

        # ── Private Credit Collapse ──
        if private_credit_redemption_pct is not None:
            if private_credit_redemption_pct > 10:
                severity = 0.90
            elif private_credit_redemption_pct > 7:
                severity = 0.75
            elif private_credit_redemption_pct > 5:
                severity = 0.60
            else:
                severity = 0.30
            signals.append(CrisisSignal(
                vector=CrisisVector.PRIVATE_CREDIT_COLLAPSE,
                severity=severity,
                description=f"Private credit fund redemptions at {private_credit_redemption_pct:.1f}%",
                data_source="reuters/bloomberg",
            ))

        # ── Stagflation ──
        if core_pce is not None and gdp_growth is not None:
            if core_pce > 3.0 and gdp_growth < 1.0:
                severity = 0.85  # Classic stagflation
            elif core_pce > 2.5 and gdp_growth < 1.5:
                severity = 0.60
            else:
                severity = 0.20
            signals.append(CrisisSignal(
                vector=CrisisVector.STAGFLATION,
                severity=severity,
                description=f"Core PCE {core_pce:.1f}%, GDP {gdp_growth:.1f}%",
                data_source="fed/bea",
            ))

        # ── Credit Contagion ──
        if credit_spread_bps is not None:
            if credit_spread_bps > 600:
                severity = 0.95
            elif credit_spread_bps > 450:
                severity = 0.80
            elif credit_spread_bps > 350:
                severity = 0.60
            elif credit_spread_bps > 250:
                severity = 0.40
            else:
                severity = 0.15
            signals.append(CrisisSignal(
                vector=CrisisVector.CREDIT_CONTAGION,
                severity=severity,
                description=f"HY credit spread {credit_spread_bps:.0f}bps",
                data_source="market_data",
            ))

        # ── VIX ──
        if vix_level is not None:
            if vix_level > 40:
                severity = 0.90
            elif vix_level > 30:
                severity = 0.70
            elif vix_level > 25:
                severity = 0.50
            elif vix_level > 20:
                severity = 0.30
            else:
                severity = 0.10
            # VIX contributes to multiple vectors
            signals.append(CrisisSignal(
                vector=CrisisVector.LIQUIDITY_CRISIS,
                severity=severity,
                description=f"VIX at {vix_level:.1f}",
                data_source="market_data",
            ))

        # ── Gold (risk-off indicator) ──
        if gold_price is not None:
            if gold_price > 4500:
                severity = 0.70
            elif gold_price > 4000:
                severity = 0.50
            else:
                severity = 0.20
            signals.append(CrisisSignal(
                vector=CrisisVector.INSURANCE_BACKSTOP_STRESS,
                severity=severity,
                description=f"Gold at ${gold_price:,.0f}/oz (flight to safety)",
                data_source="market_data",
            ))

        assessment = CrisisAssessment(signals=signals)
        self.last_assessment = assessment
        return assessment


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY ENGINE — Converts crisis assessment into actionable put orders
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PutOrderSpec:
    """A fully specified put order ready for execution."""
    symbol: str
    expiry: str  # YYYYMMDD
    strike: float
    contracts: int
    max_price: float  # Max premium per contract to pay
    side: str  # Always 'buy' for protective puts
    order_type: str  # 'limit'
    crisis_vectors: List[str]
    priority: int
    target_delta: float
    description: str


class MacroCrisisPutEngine:
    """
    Core engine that:
    1. Takes a CrisisAssessment
    2. Selects which puts to buy based on active crisis vectors
    3. Sizes positions based on account balance and risk limits
    4. Generates executable PutOrderSpec objects
    5. Can execute through IBKR connector
    """

    def __init__(
        self,
        account_balance: float = 8800.0,
        max_portfolio_put_allocation_pct: float = 15.0,
        max_single_position_pct: float = 3.0,
        paper_trading: bool = True,
    ):
        self.account_balance = account_balance
        self.max_allocation_pct = max_portfolio_put_allocation_pct
        self.max_single_pct = max_single_position_pct
        self.paper_trading = paper_trading
        self.logger = logging.getLogger("macro_crisis_puts")
        self.monitor = CrisisMonitor()

    def generate_orders(
        self,
        assessment: CrisisAssessment,
        underlying_prices: Dict[str, float],
        target_date: Optional[datetime] = None,
    ) -> List[PutOrderSpec]:
        """
        Generate put orders based on crisis assessment and current prices.

        Args:
            assessment: CrisisAssessment from CrisisMonitor
            underlying_prices: Current prices {symbol: price}
            target_date: Reference date for DTE calculation (default: now)

        Returns:
            List of PutOrderSpec objects sorted by priority
        """
        if not assessment.should_deploy_puts:
            self.logger.info(
                f"Crisis severity {assessment.composite_severity:.2f} below threshold — "
                f"no puts deployed"
            )
            return []

        if target_date is None:
            target_date = datetime.now()

        max_total = self.account_balance * (self.max_allocation_pct / 100)
        allocated = 0.0
        orders = []

        # Get active crisis vectors
        active_vectors = {s.vector for s in assessment.signals if s.severity > 0.4}

        # Filter and sort playbook by priority
        relevant_targets = [
            t for t in PUT_PLAYBOOK
            if any(v in active_vectors for v in t.crisis_vectors)
        ]
        relevant_targets.sort(key=lambda t: t.priority)

        for target in relevant_targets:
            if target.symbol not in underlying_prices:
                self.logger.warning(f"No price available for {target.symbol} — skipping")
                continue

            spot = underlying_prices[target.symbol]
            strike = round(spot * target.strike_multiplier, 0)  # Round to whole dollar

            # Calculate expiration
            expiry_date = target_date + timedelta(days=target.target_dte)
            # Snap to nearest Friday (options expire on Fridays)
            days_to_friday = (4 - expiry_date.weekday()) % 7
            expiry_date = expiry_date + timedelta(days=days_to_friday)
            expiry_str = expiry_date.strftime('%Y%m%d')

            # Position sizing — cap per position AND check remaining budget
            max_for_this = min(
                self.account_balance * (target.max_premium_pct / 100),
                max_total - allocated,
            )

            if max_for_this <= 50:
                self.logger.warning(f"Budget exhausted — skipping {target.symbol}")
                continue  # Skip this target but try cheaper ones

            # Estimate premium: rough approximation using OTM %
            # Real premium comes from IBKR quote; this is for sizing
            est_premium_per_share = spot * target.otm_pct * 0.5  # Simple estimate
            est_premium_per_contract = est_premium_per_share * 100

            if est_premium_per_contract <= 0:
                continue

            contracts = max(1, int(max_for_this / est_premium_per_contract))
            actual_cost = contracts * est_premium_per_contract

            # If even 1 contract exceeds per-position budget, skip to cheaper targets
            if est_premium_per_contract > max_for_this:
                self.logger.info(
                    f"  SKIP {target.symbol}: 1 contract ~${est_premium_per_contract:.0f} "
                    f"> budget ${max_for_this:.0f} — too expensive for account size"
                )
                continue

            order = PutOrderSpec(
                symbol=target.symbol,
                expiry=expiry_str,
                strike=strike,
                contracts=contracts,
                max_price=round(est_premium_per_share * 1.2, 2),  # 20% buffer
                side='buy',
                order_type='limit',
                crisis_vectors=[v.value for v in target.crisis_vectors],
                priority=target.priority,
                target_delta=target.target_delta,
                description=target.description,
            )

            allocated += actual_cost
            orders.append(order)

            self.logger.info(
                f"  PUT {target.symbol}: {contracts}x ${strike}P exp {expiry_str} "
                f"(est ${est_premium_per_contract:.0f}/contract, "
                f"total ${contracts * est_premium_per_contract:.0f})"
            )

        return orders

    async def execute_via_ibkr(
        self,
        orders: List[PutOrderSpec],
        ibkr_connector: Any,
    ) -> List[Dict[str, Any]]:
        """
        Execute put orders through the IBKR connector.

        Args:
            orders: List of PutOrderSpec from generate_orders()
            ibkr_connector: IBKRConnector instance (must be connected)

        Returns:
            List of execution results
        """
        results = []

        for order in orders:
            try:
                self.logger.info(
                    f"Executing: BUY {order.contracts}x {order.symbol} "
                    f"${order.strike}P {order.expiry} @ limit ${order.max_price}"
                )

                result = await ibkr_connector.create_option_order(
                    symbol=order.symbol,
                    expiry=order.expiry,
                    strike=order.strike,
                    right='P',
                    side='buy',
                    quantity=order.contracts,
                    order_type='limit',
                    price=order.max_price,
                )

                results.append({
                    'symbol': order.symbol,
                    'strike': order.strike,
                    'expiry': order.expiry,
                    'contracts': order.contracts,
                    'order_id': result.order_id,
                    'status': result.status,
                    'success': True,
                })

                self.logger.info(
                    f"  OK  Order {result.order_id}: {result.status}"
                )

            except Exception as e:
                self.logger.error(f"  FAIL {order.symbol}: {e}")
                results.append({
                    'symbol': order.symbol,
                    'strike': order.strike,
                    'expiry': order.expiry,
                    'contracts': order.contracts,
                    'error': str(e),
                    'success': False,
                })

        return results

    def print_battle_plan(
        self,
        assessment: CrisisAssessment,
        orders: List[PutOrderSpec],
    ) -> str:
        """Print a human-readable battle plan."""
        lines = []
        lines.append("=" * 70)
        lines.append("  MACRO CRISIS PUT STRATEGY — BATTLE PLAN")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        lines.append("=" * 70)

        lines.append("")
        lines.append("  CRISIS ASSESSMENT:")
        lines.append(f"    Composite Severity: {assessment.composite_severity:.0%}")
        lines.append(f"    Critical Signals:   {assessment.critical_count}/{len(assessment.signals)}")
        lines.append(f"    Deploy Puts:        {'YES' if assessment.should_deploy_puts else 'NO'}")
        lines.append("")

        for sig in sorted(assessment.signals, key=lambda s: s.severity, reverse=True):
            indicator = "!!" if sig.is_critical else "  "
            lines.append(
                f"    {indicator} [{sig.severity:.0%}] {sig.vector.value}: {sig.description}"
            )

        lines.append("")
        lines.append("  PUT ORDERS:")
        lines.append("  " + "-" * 66)

        total_premium = 0.0
        for order in orders:
            est = order.contracts * order.max_price * 100
            total_premium += est
            lines.append(
                f"    P{order.priority} | BUY {order.contracts:>2}x {order.symbol:<5} "
                f"${order.strike:>7.0f}P  exp {order.expiry}  "
                f"limit ${order.max_price:>6.2f}  "
                f"~${est:>7.0f}"
            )
            lines.append(f"       {order.description}")

        lines.append("  " + "-" * 66)
        lines.append(
            f"    TOTAL ESTIMATED PREMIUM: ${total_premium:,.0f} "
            f"({total_premium / self.account_balance * 100:.1f}% of account)"
        )
        lines.append("=" * 70)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER — Assess current crisis and generate plan
# ═══════════════════════════════════════════════════════════════════════════

async def run_crisis_assessment():
    """Run the crisis monitor with current macro data and generate put orders."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = MacroCrisisPutEngine(
        account_balance=8800.0,
        max_portfolio_put_allocation_pct=15.0,
        paper_trading=True,
    )

    # ── Current macro data as of March 16, 2026 ──
    assessment = engine.monitor.assess(
        oil_price=95.50,           # Brent crude
        vix_level=28.0,            # Elevated but not panic
        gold_price=5003.50,        # Near record, flight to safety
        ten_year_yield=4.24,       # Treasuries
        core_pce=3.1,              # Above target, rising
        gdp_growth=0.7,            # Q4 annualized, revised down
        credit_spread_bps=380,     # HY spreads widening
        private_credit_redemption_pct=11.0,  # Morgan Stanley PIF 11% redemptions
        hormuz_blocked=True,       # Strait of Hormuz blocked
        war_active=True,           # Iran conflict week 3
    )

    # ── Current approximate prices ──
    prices = {
        'SPY': 669.80,    # S&P 500 ETF
        'QQQ': 545.00,    # Nasdaq 100 ETF
        'IWM': 195.00,    # Russell 2000 ETF
        'XLF': 45.50,     # Financial Select Sector
        'HYG': 74.50,     # iShares High Yield Corporate
        'KRE': 51.00,     # SPDR Regional Banking
        'BKLN': 20.80,    # Invesco Senior Loan
        'LQD': 103.00,    # iShares Investment Grade
    }

    orders = engine.generate_orders(assessment, prices)
    plan = engine.print_battle_plan(assessment, orders)
    print(plan)

    return assessment, orders


if __name__ == "__main__":
    asyncio.run(run_crisis_assessment())

"""
MATRIX MAXIMIZER — 7-Layer Risk Management System
====================================================
Guards the entire options machine with dynamic, oil-shock-specific controls.

7 Core Risk Management Strategies:
    1. Position & Portfolio Sizing Caps
    2. Monte Carlo VaR + CVaR (95% confidence)
    3. Drawdown Circuit Breakers (daily 3%, cumulative 8%)
    4. Greeks Concentration Limits (portfolio delta/vega/gamma caps)
    5. Dynamic Premium Stop-Losses (-40% per position)
    6. Oil/VIX Shock Triggers (escalation/de-escalation)
    7. Hedging Alerts (energy hedge, TLT flight-to-safety)

Integration:
    - NCC governance gates respected (HALT = full stop)
    - Doctrine mode multiplier applied to all limits
    - Risk snapshots pushed to monitoring dashboard
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.matrix_maximizer.core import (
    Asset,
    MatrixConfig,
    MandateLevel,
    PortfolioForecast,
    SystemMandate,
)
from strategies.matrix_maximizer.greeks import GreeksResult
from strategies.matrix_maximizer.scanner import Position, PutRecommendation

logger = logging.getLogger(__name__)


class CircuitBreaker(Enum):
    """Circuit breaker states."""
    GREEN = "green"            # All clear
    YELLOW = "yellow"          # Warning — reduce risk
    RED = "red"                # Halt — close new entries
    BLACK = "black"            # Emergency — unwind all positions


@dataclass
class RiskCheck:
    """Result of a single risk check."""
    name: str
    passed: bool
    value: float               # Current value
    limit: float               # Threshold
    message: str
    severity: CircuitBreaker


@dataclass
class HedgeAlert:
    """Hedging recommendation when risk is concentrated."""
    hedge_type: str            # "energy_long", "tlt_long", "vix_call"
    reason: str
    urgency: str               # "suggested", "recommended", "required"
    ticker: str
    expression: str            # "call spread", "shares", etc.


@dataclass
class RiskSnapshot:
    """Complete risk status at a point in time."""
    timestamp: datetime
    circuit_breaker: CircuitBreaker
    checks: List[RiskCheck]
    passed: int
    failed: int
    hedge_alerts: List[HedgeAlert]

    # Portfolio-level metrics
    total_exposure: float      # Total $ in puts
    exposure_pct: float        # % of account in puts
    portfolio_delta: float     # Sum of position deltas
    portfolio_vega: float      # Sum of position vegas
    portfolio_gamma: float     # Sum of position gammas
    daily_pnl: float           # Today's P&L
    cumulative_pnl: float      # Cumulative P&L
    daily_pnl_pct: float
    cumulative_pnl_pct: float

    # VaR
    var_95_1d: float
    cvar_95_1d: float

    # Mandate
    mandate: SystemMandate

    def print_summary(self) -> str:
        """Human-readable risk snapshot."""
        status_icon = {
            CircuitBreaker.GREEN: "[OK]",
            CircuitBreaker.YELLOW: "[!]",
            CircuitBreaker.RED: "[X]",
            CircuitBreaker.BLACK: "[!!!]",
        }
        lines = [
            "=" * 72,
            "  MATRIX MAXIMIZER — RISK SNAPSHOT",
            f"  {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Circuit Breaker: {status_icon[self.circuit_breaker]} "
            f"{self.circuit_breaker.value.upper()}",
            f"  Checks: {self.passed} passed, {self.failed} failed",
            "-" * 72,
            f"  Exposure: ${self.total_exposure:,.0f} ({self.exposure_pct:.1%} of account)",
            f"  Portfolio Delta: {self.portfolio_delta:.3f} | "
            f"Vega: ${self.portfolio_vega:.0f} | Gamma: {self.portfolio_gamma:.4f}",
            f"  Daily PnL: {self.daily_pnl:+,.0f} ({self.daily_pnl_pct:+.1%})",
            f"  Cumulative PnL: {self.cumulative_pnl:+,.0f} ({self.cumulative_pnl_pct:+.1%})",
            f"  VaR(95,1d): {self.var_95_1d:.1%} | CVaR(95,1d): {self.cvar_95_1d:.1%}",
            f"  Mandate: {self.mandate.level.value.upper()} | "
            f"Risk/Trade: {self.mandate.risk_per_trade_pct:.1%}",
            "-" * 72,
        ]
        for check in self.checks:
            icon = "[OK]" if check.passed else status_icon[check.severity]
            lines.append(f"  {icon} {check.name}: {check.value:.3f} (limit: {check.limit:.3f}) "
                         f"— {check.message}")
        if self.hedge_alerts:
            lines.append("-" * 72)
            lines.append("  HEDGE ALERTS:")
            for alert in self.hedge_alerts:
                lines.append(f"    [{alert.urgency.upper()}] {alert.hedge_type}: {alert.reason}")
                lines.append(f"      -> {alert.ticker} {alert.expression}")
        lines.append("=" * 72)
        return "\n".join(lines)


class RiskManager:
    """7-layer risk management system for the MATRIX MAXIMIZER.

    Usage:
        risk = RiskManager(config)
        snapshot = risk.evaluate(positions, forecast, daily_pnl, cumul_pnl)
        if snapshot.circuit_breaker == CircuitBreaker.RED:
            # Halt all new entries
    """

    def __init__(self, config: MatrixConfig) -> None:
        self.config = config
        self._state_file = Path("data/matrix_maximizer_risk_state.json")
        self._daily_pnl_history: List[float] = []

    def evaluate(
        self,
        positions: List[Position],
        forecast: PortfolioForecast,
        greeks_map: Dict[str, GreeksResult],
        daily_pnl: float = 0.0,
        cumulative_pnl: float = 0.0,
        oil_price: float = 96.5,
        vix: float = 22.0,
        ncc_risk_multiplier: float = 1.0,
    ) -> RiskSnapshot:
        """Run all 7 risk checks and return a complete snapshot.

        Args:
            positions: Open put positions
            forecast: Latest Monte Carlo forecast
            greeks_map: Greeks for each position (key = ticker)
            daily_pnl: Today's P&L in dollars
            cumulative_pnl: Cumulative P&L in dollars
            oil_price: Current oil price (WTI)
            vix: Current VIX
            ncc_risk_multiplier: NCC governance multiplier (1.0 normal, 0.5 caution, 0.0 halt)
        """
        checks: List[RiskCheck] = []
        hedge_alerts: List[HedgeAlert] = []
        account = self.config.account_size

        # Apply NCC governance multiplier to all limits
        effective_max_exposure = self.config.max_portfolio_put_pct * ncc_risk_multiplier
        effective_max_delta = self.config.max_portfolio_delta * ncc_risk_multiplier
        effective_max_daily_loss = self.config.max_daily_loss_pct * ncc_risk_multiplier

        # Calculate portfolio Greeks
        total_exposure = sum(p.cost_basis for p in positions)
        port_delta = sum(g.delta * 100 for g in greeks_map.values())  # Per contract × 100
        port_vega = sum(g.vega * 100 for g in greeks_map.values())
        port_gamma = sum(g.gamma * 100 for g in greeks_map.values())
        exposure_pct = total_exposure / account if account > 0 else 0

        # ─── CHECK 1: Position & Portfolio Sizing ───
        checks.append(RiskCheck(
            name="Portfolio Exposure",
            passed=exposure_pct <= effective_max_exposure,
            value=exposure_pct,
            limit=effective_max_exposure,
            message=f"${total_exposure:,.0f} of ${account:,.0f} ({exposure_pct:.1%})",
            severity=CircuitBreaker.YELLOW if exposure_pct <= effective_max_exposure * 1.2 else CircuitBreaker.RED,
        ))

        # Per-position check
        for pos in positions:
            pos_pct = pos.cost_basis / account if account > 0 else 0
            per_trade_limit = self.config.risk_per_trade_pct * 3  # 3x risk budget = max single
            if pos_pct > per_trade_limit:
                checks.append(RiskCheck(
                    name=f"Position Size {pos.ticker}",
                    passed=False,
                    value=pos_pct,
                    limit=per_trade_limit,
                    message=f"{pos.ticker} at {pos_pct:.1%} exceeds {per_trade_limit:.1%} limit",
                    severity=CircuitBreaker.YELLOW,
                ))

        # ─── CHECK 2: Monte Carlo VaR + CVaR ───
        spy_forecast = forecast.asset_forecasts.get(Asset.SPY)
        var_1d = spy_forecast.var_95_1d if spy_forecast else 0.0
        cvar_1d = spy_forecast.cvar_95_1d if spy_forecast else 0.0

        checks.append(RiskCheck(
            name="VaR(95,1d)",
            passed=var_1d <= self.config.var_pause_threshold,
            value=var_1d,
            limit=self.config.var_pause_threshold,
            message=f"1-day VaR at {var_1d:.1%} (pause at {self.config.var_pause_threshold:.1%})",
            severity=CircuitBreaker.RED if var_1d > self.config.var_pause_threshold else CircuitBreaker.GREEN,
        ))

        checks.append(RiskCheck(
            name="CVaR(95,1d)",
            passed=cvar_1d <= self.config.var_pause_threshold * 1.5,
            value=cvar_1d,
            limit=self.config.var_pause_threshold * 1.5,
            message=f"Expected shortfall at {cvar_1d:.1%}",
            severity=CircuitBreaker.YELLOW,
        ))

        # ─── CHECK 3: Drawdown Circuit Breakers ───
        daily_pnl_pct = daily_pnl / account if account > 0 else 0
        cumul_pnl_pct = cumulative_pnl / account if account > 0 else 0

        checks.append(RiskCheck(
            name="Daily Loss",
            passed=daily_pnl_pct >= -effective_max_daily_loss,
            value=daily_pnl_pct,
            limit=-effective_max_daily_loss,
            message=f"Today: {daily_pnl_pct:+.1%} (limit: {-effective_max_daily_loss:.1%})",
            severity=CircuitBreaker.RED if daily_pnl_pct < -effective_max_daily_loss else CircuitBreaker.GREEN,
        ))

        checks.append(RiskCheck(
            name="Cumulative Drawdown",
            passed=cumul_pnl_pct >= -self.config.max_drawdown_pct,
            value=cumul_pnl_pct,
            limit=-self.config.max_drawdown_pct,
            message=f"Cumulative: {cumul_pnl_pct:+.1%} (limit: {-self.config.max_drawdown_pct:.1%})",
            severity=CircuitBreaker.BLACK if cumul_pnl_pct < -self.config.max_drawdown_pct else CircuitBreaker.GREEN,
        ))

        # ─── CHECK 4: Greeks Concentration Limits ───
        max_vega = account * self.config.max_portfolio_vega_ratio

        checks.append(RiskCheck(
            name="Portfolio Delta",
            passed=port_delta >= effective_max_delta,  # Delta is negative for puts
            value=port_delta,
            limit=effective_max_delta,
            message=f"Net delta: {port_delta:.3f} (limit: {effective_max_delta:.3f})",
            severity=CircuitBreaker.YELLOW,
        ))

        checks.append(RiskCheck(
            name="Portfolio Vega",
            passed=abs(port_vega) <= max_vega,
            value=abs(port_vega),
            limit=max_vega,
            message=f"Net vega: ${port_vega:.0f} (limit: ${max_vega:.0f})",
            severity=CircuitBreaker.YELLOW,
        ))

        # ─── CHECK 5: Premium Stop-Losses (position-level) ───
        for pos in positions:
            if pos.pnl_pct <= -self.config.premium_stop_loss_pct:
                checks.append(RiskCheck(
                    name=f"Stop Loss {pos.ticker}",
                    passed=False,
                    value=pos.pnl_pct,
                    limit=-self.config.premium_stop_loss_pct,
                    message=f"{pos.ticker} premium down {pos.pnl_pct:.0%} — CLOSE NOW",
                    severity=CircuitBreaker.RED,
                ))

        # ─── CHECK 6: Oil/VIX Shock Triggers ───
        oil_escalated = oil_price > self.config.oil_escalation_threshold
        vix_escalated = vix > self.config.vix_escalation_threshold
        oil_deescalated = oil_price < self.config.oil_deescalation_threshold

        if oil_escalated or vix_escalated:
            checks.append(RiskCheck(
                name="Shock Trigger",
                passed=True,  # Not a failure — an escalation signal
                value=oil_price if oil_escalated else vix,
                limit=self.config.oil_escalation_threshold if oil_escalated else self.config.vix_escalation_threshold,
                message=f"{'Oil' if oil_escalated else 'VIX'} escalation — deeper OTM, halved risk/trade",
                severity=CircuitBreaker.YELLOW,
            ))

        if oil_deescalated:
            checks.append(RiskCheck(
                name="De-escalation",
                passed=True,
                value=oil_price,
                limit=self.config.oil_deescalation_threshold,
                message=f"Oil de-escalation (${oil_price:.0f}) — reduce exposure 50%",
                severity=CircuitBreaker.YELLOW,
            ))

        # ─── CHECK 7: Hedging Alerts ───
        if abs(port_vega) > max_vega * 0.7:
            hedge_alerts.append(HedgeAlert(
                hedge_type="energy_long",
                reason=f"High portfolio vega (${port_vega:.0f}) — natural energy hedge",
                urgency="recommended",
                ticker="XLE",
                expression="call spread 30-45 DTE",
            ))

        if spy_forecast and spy_forecast.prob_down_20 > 0.15:
            hedge_alerts.append(HedgeAlert(
                hedge_type="tlt_long",
                reason=f"Extreme bear tail: P(20%down)={spy_forecast.prob_down_20:.0%} — flight-to-safety",
                urgency="suggested",
                ticker="TLT",
                expression="shares or call 90-day",
            ))

        if vix > 35:
            hedge_alerts.append(HedgeAlert(
                hedge_type="vix_call",
                reason=f"VIX={vix:.1f} — consider VIX call spread for convexity hedge",
                urgency="required" if vix > 40 else "recommended",
                ticker="VIX",
                expression="call spread 1-2 month",
            ))

        # ─── DETERMINE CIRCUIT BREAKER STATE ───
        failed_checks = [c for c in checks if not c.passed]
        passed_count = len(checks) - len(failed_checks)

        # NCC governance override
        if ncc_risk_multiplier <= 0.0:
            circuit = CircuitBreaker.BLACK
        elif any(c.severity == CircuitBreaker.BLACK for c in failed_checks):
            circuit = CircuitBreaker.BLACK
        elif any(c.severity == CircuitBreaker.RED for c in failed_checks):
            circuit = CircuitBreaker.RED
        elif len(failed_checks) >= 3:
            circuit = CircuitBreaker.RED
        elif failed_checks:
            circuit = CircuitBreaker.YELLOW
        else:
            circuit = CircuitBreaker.GREEN

        snapshot = RiskSnapshot(
            timestamp=datetime.utcnow(),
            circuit_breaker=circuit,
            checks=checks,
            passed=passed_count,
            failed=len(failed_checks),
            hedge_alerts=hedge_alerts,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            portfolio_delta=port_delta,
            portfolio_vega=port_vega,
            portfolio_gamma=port_gamma,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            daily_pnl_pct=daily_pnl_pct,
            cumulative_pnl_pct=cumul_pnl_pct,
            var_95_1d=var_1d,
            cvar_95_1d=cvar_1d,
            mandate=forecast.mandate,
        )

        logger.info(
            "Risk check: %s | %d/%d passed | delta=%.3f | vega=$%.0f | exposure=%.1%%",
            circuit.value, passed_count, len(checks), port_delta, port_vega, exposure_pct * 100,
        )

        return snapshot

    def save_state(self, snapshot: RiskSnapshot) -> None:
        """Persist risk state for recovery and audit."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "timestamp": snapshot.timestamp.isoformat(),
            "circuit_breaker": snapshot.circuit_breaker.value,
            "passed": snapshot.passed,
            "failed": snapshot.failed,
            "exposure_pct": snapshot.exposure_pct,
            "portfolio_delta": snapshot.portfolio_delta,
            "daily_pnl": snapshot.daily_pnl,
            "cumulative_pnl": snapshot.cumulative_pnl,
            "mandate": snapshot.mandate.level.value,
            "checks": [
                {"name": c.name, "passed": c.passed, "value": c.value, "limit": c.limit}
                for c in snapshot.checks
            ],
        }
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load previous risk state."""
        if self._state_file.exists():
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        return None

"""
MATRIX MAXIMIZER — Core Data Structures & Scenario Engine
===========================================================
All shared types, enums, constants, and the scenario-probability engine.

Scenario model:
  BASE  (50%) — Partial de-escalation 4-8 weeks, oil -> $85-90
  BEAR  (40%) — Prolonged Hormuz/attacks, oil -> $110+
  BULL  (10%) — Quick resolution, oil -> $80

Oil is the #1 driver. Equity/crypto betas keyed off oil % move:
  SPY  beta ~ -0.4x oil
  QQQ  beta ~ -0.5x oil
  BTC  beta ~ -0.35x oil (high-vol amplifier)
  TLT  beta ~ +0.1x oil (flight-to-safety, partial)
  USO  beta ~ +1.0x oil (direct proxy)
  JETS beta ~ -0.6x oil (diesel victim)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class Asset(Enum):
    """Tracked assets for Monte Carlo + scanner."""
    SPY = "SPY"
    QQQ = "QQQ"
    USO = "USO"
    BITO = "BITO"     # BTC proxy ETF
    TLT = "TLT"
    JETS = "JETS"
    KRE = "KRE"
    HYG = "HYG"
    XLY = "XLY"
    ZIM = "ZIM"
    XLE = "XLE"       # Energy — DO NOT SHORT


class Scenario(Enum):
    """Geopolitical scenario taxonomy."""
    BASE = "base"     # Partial de-escalation 4-8 weeks
    BEAR = "bear"     # Prolonged conflict, Hormuz extended closure
    BULL = "bull"     # Quick resolution


class MandateLevel(Enum):
    """Dynamic mandate escalation levels."""
    DEFENSIVE = "defensive"           # Oil < $85, de-escalation
    STANDARD = "standard"             # Normal bearish operations
    AGGRESSIVE = "aggressive"         # High downside probability
    MAX_CONVICTION = "max_conviction"  # Extreme tail risk confirmed


class RollAction(Enum):
    """Auto-roll outcomes."""
    HOLD = "hold"
    ROLL_DEEPER = "roll_deeper"       # Roll to deeper OTM
    ROLL_WIDER = "roll_wider"         # Roll to wider spread
    PYRAMID = "pyramid"              # Increase position size
    CLOSE = "close"                  # Take profit / stop loss
    ABORT = "abort"                  # Risk limit hit


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO WEIGHTS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioWeights:
    """Probability weights for each scenario — auto-adjusted by oil/VIX."""
    base: float = 0.50
    bear: float = 0.40
    bull: float = 0.10

    def validate(self) -> None:
        total = self.base + self.bear + self.bull
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Scenario weights must sum to 1.0, got {total:.4f}")

    def adjust_for_oil(self, oil_price: float) -> "ScenarioWeights":
        """Shift probabilities based on current oil price."""
        if oil_price > 105:
            return ScenarioWeights(base=0.30, bear=0.60, bull=0.10)
        elif oil_price > 95:
            return ScenarioWeights(base=0.40, bear=0.50, bull=0.10)
        elif oil_price < 85:
            return ScenarioWeights(base=0.50, bear=0.25, bull=0.25)
        elif oil_price < 75:
            return ScenarioWeights(base=0.40, bear=0.15, bull=0.45)
        return ScenarioWeights(base=self.base, bear=self.bear, bull=self.bull)

    def adjust_for_vix(self, vix: float) -> "ScenarioWeights":
        """Further adjust for VIX level."""
        if vix > 30:
            return ScenarioWeights(
                base=max(0.1, self.base - 0.15),
                bear=min(0.8, self.bear + 0.15),
                bull=max(0.05, self.bull),
            )
        elif vix > 25:
            return ScenarioWeights(
                base=max(0.2, self.base - 0.05),
                bear=min(0.7, self.bear + 0.05),
                bull=self.bull,
            )
        return ScenarioWeights(base=self.base, bear=self.bear, bull=self.bull)


# Oil-correlated betas: asset_beta × oil_pct_move → expected asset return
ASSET_OIL_BETAS: Dict[Asset, float] = {
    Asset.SPY:  -0.40,
    Asset.QQQ:  -0.50,
    Asset.USO:   1.00,
    Asset.BITO: -0.35,
    Asset.TLT:   0.10,
    Asset.JETS: -0.60,
    Asset.KRE:  -0.45,
    Asset.HYG:  -0.30,
    Asset.XLY:  -0.35,
    Asset.ZIM:  -0.55,
    Asset.XLE:   0.70,
}

# Annualised volatility (conflict-elevated, March 2026)
ASSET_VOLATILITIES: Dict[Asset, float] = {
    Asset.SPY:  0.22,
    Asset.QQQ:  0.28,
    Asset.USO:  0.45,
    Asset.BITO: 0.55,
    Asset.TLT:  0.18,
    Asset.JETS: 0.35,
    Asset.KRE:  0.32,
    Asset.HYG:  0.15,
    Asset.XLY:  0.25,
    Asset.ZIM:  0.50,
    Asset.XLE:  0.30,
}

# Scenario-weighted drifts (annualised) for 3-month horizon
SCENARIO_DRIFTS: Dict[Scenario, Dict[Asset, float]] = {
    Scenario.BASE: {
        Asset.SPY: -0.07, Asset.QQQ: -0.09, Asset.USO: 0.05,
        Asset.BITO: -0.15, Asset.TLT: 0.02, Asset.JETS: -0.12,
        Asset.KRE: -0.10, Asset.HYG: -0.05, Asset.XLY: -0.08,
        Asset.ZIM: -0.14, Asset.XLE: 0.10,
    },
    Scenario.BEAR: {
        Asset.SPY: -0.18, Asset.QQQ: -0.22, Asset.USO: 0.30,
        Asset.BITO: -0.30, Asset.TLT: 0.05, Asset.JETS: -0.30,
        Asset.KRE: -0.25, Asset.HYG: -0.12, Asset.XLY: -0.18,
        Asset.ZIM: -0.28, Asset.XLE: 0.20,
    },
    Scenario.BULL: {
        Asset.SPY: 0.05, Asset.QQQ: 0.08, Asset.USO: -0.10,
        Asset.BITO: 0.10, Asset.TLT: -0.03, Asset.JETS: 0.08,
        Asset.KRE: 0.06, Asset.HYG: 0.03, Asset.XLY: 0.05,
        Asset.ZIM: 0.07, Asset.XLE: -0.05,
    },
}

# Correlation matrix (simplified: all correlated through oil)
# Row/col order: SPY, QQQ, USO, BITO, TLT, JETS, KRE, HYG, XLY, ZIM, XLE
CORRELATION_MATRIX = [
    # SPY   QQQ   USO   BITO  TLT   JETS  KRE   HYG   XLY   ZIM   XLE
    [1.00,  0.92, -0.40, 0.60, -0.30, 0.75, 0.80, 0.65, 0.85, 0.55, -0.25],  # SPY
    [0.92,  1.00, -0.35, 0.55, -0.25, 0.70, 0.72, 0.58, 0.80, 0.50, -0.20],  # QQQ
    [-0.40, -0.35, 1.00, -0.20, 0.10, -0.55, -0.35, -0.25, -0.30, -0.45, 0.70],  # USO
    [0.60,  0.55, -0.20, 1.00, -0.15, 0.45, 0.50, 0.40, 0.50, 0.35, -0.10],  # BITO
    [-0.30, -0.25, 0.10, -0.15, 1.00, -0.20, -0.25, 0.30, -0.20, -0.15, 0.05],  # TLT
    [0.75,  0.70, -0.55, 0.45, -0.20, 1.00, 0.65, 0.55, 0.70, 0.60, -0.40],  # JETS
    [0.80,  0.72, -0.35, 0.50, -0.25, 0.65, 1.00, 0.70, 0.72, 0.50, -0.20],  # KRE
    [0.65,  0.58, -0.25, 0.40, 0.30,  0.55, 0.70, 1.00, 0.60, 0.45, -0.15],  # HYG
    [0.85,  0.80, -0.30, 0.50, -0.20, 0.70, 0.72, 0.60, 1.00, 0.55, -0.20],  # XLY
    [0.55,  0.50, -0.45, 0.35, -0.15, 0.60, 0.50, 0.45, 0.55, 1.00, -0.30],  # ZIM
    [-0.25, -0.20, 0.70, -0.10, 0.05, -0.40, -0.20, -0.15, -0.20, -0.30, 1.00],  # XLE
]


# Default baseline prices (March 18, 2026)
DEFAULT_PRICES: Dict[Asset, float] = {
    Asset.SPY: 667.0,
    Asset.QQQ: 480.0,
    Asset.USO: 96.5,
    Asset.BITO: 72.0,
    Asset.TLT: 88.0,
    Asset.JETS: 19.5,
    Asset.KRE: 56.0,
    Asset.HYG: 76.0,
    Asset.XLY: 185.0,
    Asset.ZIM: 18.0,
    Asset.XLE: 93.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MatrixConfig:
    """Complete configuration for a MATRIX MAXIMIZER cycle."""
    # Account
    account_size: float = 50000.0
    max_portfolio_put_pct: float = 0.20       # 20% max in puts
    risk_per_trade_pct: float = 0.01          # 1% base risk per trade

    # Monte Carlo
    n_simulations: int = 10000
    horizon_days: int = 90
    risk_free_rate: float = 0.037             # 3-month Treasury (March 2026)
    dividend_yield: float = 0.0109            # SPY dividend yield

    # Scanner
    target_delta_min: float = -0.48
    target_delta_max: float = -0.25
    min_volume: int = 100
    min_open_interest: int = 500
    otm_pct_default: float = 0.10             # 10% OTM default
    max_expiry_days: int = 45                 # Max DTE for weeklies

    # Auto-roll
    delta_decay_threshold: float = 0.20       # 20% delta decay triggers roll
    premium_stop_loss_pct: float = 0.40       # -40% premium = cut
    pyramid_multiplier: float = 1.5           # Roll winners 1.5x

    # Risk
    max_daily_loss_pct: float = 0.03          # 3% daily loss → circuit breaker
    max_drawdown_pct: float = 0.08            # 8% cumulative → pause 24h
    max_portfolio_delta: float = -0.50        # Total portfolio delta cap
    max_portfolio_vega_ratio: float = 0.001   # Vega ≤ account_size / 1000
    var_pause_threshold: float = 0.05         # 1-day VaR > 5% → pause

    # Oil/VIX shock thresholds
    oil_escalation_threshold: float = 105.0
    oil_deescalation_threshold: float = 85.0
    vix_escalation_threshold: float = 30.0

    # Tickers to scan (excludes energy — do NOT short XLE)
    scan_tickers: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "JETS", "KRE", "HYG", "XLY", "ZIM", "BITO",
    ])

    # NCC governance
    ncc_governance_enabled: bool = True
    ncl_intelligence_enabled: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# FORECAST RESULTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssetForecast:
    """Monte Carlo forecast for a single asset."""
    asset: Asset
    current_price: float
    mean_price: float
    median_price: float
    pct_5: float                    # 5th percentile (worst 5%)
    pct_25: float
    pct_75: float
    pct_95: float
    expected_return_pct: float      # Mean return
    prob_down_10: float             # P(drop ≥ 10%)
    prob_down_15: float             # P(drop ≥ 15%)
    prob_down_20: float             # P(drop ≥ 20%)
    prob_up_5: float                # P(gain ≥ 5%)
    var_95_1d: float                # 95% 1-day VaR (% loss)
    cvar_95_1d: float               # Expected shortfall (worst 5% avg)
    var_95_90d: float               # 95% 90-day VaR
    cvar_95_90d: float              # 90-day expected shortfall
    simulated_paths: int            # Number of simulations


@dataclass
class PortfolioForecast:
    """Aggregated forecast across all tracked assets."""
    timestamp: datetime
    scenario_weights: ScenarioWeights
    asset_forecasts: Dict[Asset, AssetForecast]
    weighted_return: float          # Scenario-weighted portfolio return
    portfolio_var_95: float         # Portfolio 95% VaR
    portfolio_cvar_95: float        # Portfolio CVaR
    mandate: "SystemMandate"        # Dynamic mandate based on probabilities
    horizon_days: int
    n_simulations: int

    def print_summary(self) -> str:
        """Human-readable forecast summary."""
        lines = [
            "=" * 72,
            "  MATRIX MAXIMIZER — MONTE CARLO FORECAST",
            f"  {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Simulations: {self.n_simulations:,} | Horizon: {self.horizon_days}d",
            f"  Scenarios: BASE {self.scenario_weights.base:.0%} "
            f"BEAR {self.scenario_weights.bear:.0%} "
            f"BULL {self.scenario_weights.bull:.0%}",
            "=" * 72,
            f"  MANDATE: {self.mandate.level.value.upper()}",
            f"  Risk/Trade: {self.mandate.risk_per_trade_pct:.1%}",
            f"  OTM Target: {self.mandate.otm_pct:.0%}",
            f"  Portfolio VaR(95): {self.portfolio_var_95:.1%}",
            f"  Portfolio CVaR(95): {self.portfolio_cvar_95:.1%}",
            "-" * 72,
            f"  {'ASSET':<6} {'PRICE':>7} {'MEAN':>7} {'RET%':>6} "
            f"{'P(10%↓)':>8} {'P(15%↓)':>8} {'P(20%↓)':>8} "
            f"{'5th%':>7} {'95th%':>7}",
            "-" * 72,
        ]
        for asset in Asset:
            if asset in self.asset_forecasts:
                f = self.asset_forecasts[asset]
                lines.append(
                    f"  {asset.value:<6} {f.current_price:>7.1f} {f.mean_price:>7.1f} "
                    f"{f.expected_return_pct:>5.1f}% "
                    f"{f.prob_down_10:>7.0%} {f.prob_down_15:>7.0%} "
                    f"{f.prob_down_20:>7.0%} "
                    f"{f.pct_5:>7.1f} {f.pct_95:>7.1f}"
                )
        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass
class SystemMandate:
    """Dynamic system mandate computed from Monte Carlo probabilities."""
    level: MandateLevel
    risk_per_trade_pct: float       # Adjusted risk per trade
    otm_pct: float                  # Target OTM % for puts
    max_contracts_per_name: int     # Max contracts per underlying
    pyramid_allowed: bool           # Whether to pyramid winners
    hedge_energy: bool              # Suggest long XLE/USO hedge
    hedge_tlt: bool                 # Suggest TLT long for flight-to-safety
    rationale: str

    @staticmethod
    def from_probabilities(
        prob_10_down: float,
        oil_price: float,
        vix: float,
        config: MatrixConfig,
    ) -> "SystemMandate":
        """Compute mandate from Monte Carlo output probabilities."""
        if prob_10_down > 0.45 or vix > 30:
            return SystemMandate(
                level=MandateLevel.MAX_CONVICTION,
                risk_per_trade_pct=min(0.025, config.risk_per_trade_pct * 2.5),
                otm_pct=0.15,
                max_contracts_per_name=10,
                pyramid_allowed=True,
                hedge_energy=True,
                hedge_tlt=True,
                rationale=f"Extreme bear: P(10%down)={prob_10_down:.0%}, VIX={vix:.1f}",
            )
        elif prob_10_down > 0.35 or oil_price > 105:
            return SystemMandate(
                level=MandateLevel.AGGRESSIVE,
                risk_per_trade_pct=min(0.020, config.risk_per_trade_pct * 2.0),
                otm_pct=0.12,
                max_contracts_per_name=7,
                pyramid_allowed=True,
                hedge_energy=True,
                hedge_tlt=False,
                rationale=f"High bear: P(10%down)={prob_10_down:.0%}, oil=${oil_price:.0f}",
            )
        elif oil_price < 85:
            return SystemMandate(
                level=MandateLevel.DEFENSIVE,
                risk_per_trade_pct=config.risk_per_trade_pct * 0.5,
                otm_pct=0.08,
                max_contracts_per_name=3,
                pyramid_allowed=False,
                hedge_energy=False,
                hedge_tlt=False,
                rationale=f"De-escalation: oil=${oil_price:.0f}, reducing exposure",
            )
        else:
            return SystemMandate(
                level=MandateLevel.STANDARD,
                risk_per_trade_pct=config.risk_per_trade_pct,
                otm_pct=0.10,
                max_contracts_per_name=5,
                pyramid_allowed=True,
                hedge_energy=False,
                hedge_tlt=False,
                rationale=f"Standard bearish: P(10%down)={prob_10_down:.0%}",
            )

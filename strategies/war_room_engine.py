#!/usr/bin/env python3
"""
AAC War Room Engine v2.0
========================
Forward Monte Carlo simulation (100,000 paths), milestone-driven spiderweb
trigger system, Black-Scholes Greeks, 5-arm position management, and
twice-daily CLI runner.

Milestone-driven: timeline measured in DOLLAR GAINS, not dates.
Phase transitions at $150K -> $1M -> $5M -> $100M.

Requires: numpy, scipy (both in .venv)
All output is ASCII-safe for Windows PowerShell cp1252.

Usage:
    python strategies/war_room_engine.py                  # dashboard
    python strategies/war_room_engine.py --monte-carlo    # run 100K MC
    python strategies/war_room_engine.py --milestones     # spiderweb status
    python strategies/war_room_engine.py --greeks         # position Greeks
    python strategies/war_room_engine.py --mandate        # twice-daily mandate
    python strategies/war_room_engine.py --positions      # all positions
    python strategies/war_room_engine.py --arms           # 5-arm breakdown
    python strategies/war_room_engine.py --indicators     # 12-indicator model
    python strategies/war_room_engine.py --scenario NAME  # run scenario
    python strategies/war_room_engine.py --phase          # phase status
    python strategies/war_room_engine.py --json           # JSON output
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# -- UTF-8 stdout fix for Windows cp1252 terminals --
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from scipy.stats import norm

# ============================================================================
# PART 1: CONSTANTS & CONFIGURATION
# ============================================================================

STARTING_CAPITAL_CAD = 45_120.0
CAD_TO_USD = 0.72  # approximate
RISK_FREE_RATE = 0.045  # 4.5% Fed rate
MC_PATHS = 100_000
MC_HORIZON_DAYS = 90
TRADING_DAYS_PER_YEAR = 252

# Phase thresholds (USD)
PHASE_THRESHOLDS = {
    "accumulation": (0, 150_000),
    "growth": (150_000, 1_000_000),
    "rotation": (1_000_000, 5_000_000),
    "preservation": (5_000_000, 100_000_000),
}

# 11 correlated assets
ASSETS = [
    "oil", "gold", "silver", "gdx", "spy", "qqq",
    "xlf", "xlre", "eth", "xrp", "btc",
]

# LIVE prices -- March 19, 2026 7:20 PM MDT
SPOT_PRICES = {
    "oil": 95.0,        # WTI crude ($93.96-$96.60 range)
    "gold": 4861.0,     # COMEX gold -- new ATH territory
    "silver": 78.0,     # COMEX silver -- parabolic
    "gdx": 95.0,        # Gold miners ETF (est. 2-3x gold leverage)
    "spy": 665.0,       # S&P 500 -- holding but risk-off tone
    "qqq": 450.0,       # Nasdaq -- tech under pressure
    "xlf": 35.0,        # Financials -- credit stress visible
    "xlre": 22.0,       # CRE -- crushed (-42% from pre-crisis)
    "eth": 3800.0,      # Ethereum -- surprisingly strong
    "xrp": 2.50,        # XRP -- holding
    "btc": 68000.0,     # Bitcoin -- declining, broke $70K
}

# Forward-looking annualized drifts (crisis ~50% realized as of Mar 19)
CRISIS_DRIFTS = {
    "oil": 0.60,       # +60% forward (Hormuz escalation still live)
    "gold": 0.40,      # +40% forward (parabolic but momentum intact)
    "silver": 0.40,    # +40% forward (industrial + safe haven)
    "gdx": 0.55,       # +55% forward (gold miners leverage)
    "spy": -0.40,      # -40% forward (crash NOT yet started -- big risk)
    "qqq": -0.45,      # -45% forward (tech selloff imminent)
    "xlf": -0.40,      # -40% forward (credit stress deepening)
    "xlre": -0.20,     # -20% forward (most downside done, -42% already)
    "eth": -0.55,      # -55% forward (DeFi contagion not yet realized)
    "xrp": -0.45,      # -45% forward (crypto contagion)
    "btc": -0.50,      # -50% forward (risk-off intensifying)
}

# Annualized volatilities (VIX at 25, elevated regime)
CRISIS_VOLS = {
    "oil": 0.90,       # geopolitical risk premium
    "gold": 0.72,      # parabolic moves = higher vol
    "silver": 0.82,    # industrial + speculative
    "gdx": 1.02,       # leveraged to gold vol
    "spy": 0.48,       # VIX at 25 = ~48% annualized
    "qqq": 0.52,       # tech vol elevated
    "xlf": 0.58,       # credit stress vol
    "xlre": 0.50,      # CRE vol declining (bottoming)
    "eth": 1.20,       # crypto vol elevated
    "xrp": 1.28,       # altcoin vol extreme
    "btc": 1.12,       # bitcoin vol elevated
}

# Correlation matrix (11x11) -- from geopolitical analysis
# Order: oil, gold, silver, gdx, spy, qqq, xlf, xlre, eth, xrp, btc
_RAW_CORR = {
    ("oil", "gold"): 0.55,
    ("oil", "silver"): 0.50,
    ("oil", "gdx"): 0.60,
    ("oil", "spy"): -0.30,
    ("oil", "qqq"): -0.25,
    ("oil", "xlf"): -0.35,
    ("oil", "xlre"): -0.40,
    ("oil", "eth"): -0.35,
    ("oil", "xrp"): -0.30,
    ("oil", "btc"): -0.40,
    ("gold", "silver"): 0.85,
    ("gold", "gdx"): 0.70,
    ("gold", "spy"): -0.25,
    ("gold", "qqq"): -0.20,
    ("gold", "xlf"): -0.30,
    ("gold", "xlre"): -0.35,
    ("gold", "eth"): -0.30,
    ("gold", "xrp"): -0.25,
    ("gold", "btc"): -0.35,
    ("silver", "gdx"): 0.75,
    ("silver", "spy"): -0.20,
    ("silver", "qqq"): -0.15,
    ("silver", "xlf"): -0.25,
    ("silver", "xlre"): -0.30,
    ("silver", "eth"): -0.25,
    ("silver", "xrp"): -0.20,
    ("silver", "btc"): -0.30,
    ("gdx", "spy"): -0.35,
    ("gdx", "qqq"): -0.30,
    ("gdx", "xlf"): -0.40,
    ("gdx", "xlre"): -0.45,
    ("gdx", "eth"): -0.35,
    ("gdx", "xrp"): -0.30,
    ("gdx", "btc"): -0.45,
    ("spy", "qqq"): 0.92,
    ("spy", "xlf"): 0.75,
    ("spy", "xlre"): 0.65,
    ("spy", "eth"): 0.45,
    ("spy", "xrp"): 0.40,
    ("spy", "btc"): 0.50,
    ("qqq", "xlf"): 0.70,
    ("qqq", "xlre"): 0.55,
    ("qqq", "eth"): 0.50,
    ("qqq", "xrp"): 0.45,
    ("qqq", "btc"): 0.55,
    ("xlf", "xlre"): 0.60,
    ("xlf", "eth"): 0.35,
    ("xlf", "xrp"): 0.30,
    ("xlf", "btc"): 0.40,
    ("xlre", "eth"): 0.30,
    ("xlre", "xrp"): 0.25,
    ("xlre", "btc"): 0.35,
    ("eth", "xrp"): 0.80,
    ("eth", "btc"): 0.85,
    ("xrp", "btc"): 0.75,
}


def _build_correlation_matrix() -> np.ndarray:
    """Build symmetric correlation matrix from pairwise dict."""
    n = len(ASSETS)
    corr = np.eye(n)
    for i, a in enumerate(ASSETS):
        for j, b in enumerate(ASSETS):
            if i == j:
                continue
            key = (a, b) if (a, b) in _RAW_CORR else (b, a)
            if key in _RAW_CORR:
                corr[i, j] = _RAW_CORR[key]
    return corr


CORRELATION_MATRIX = _build_correlation_matrix()


# ============================================================================
# PART 2: BLACK-SCHOLES GREEKS ENGINE
# ============================================================================

@dataclass
class GreeksResult:
    """Full Black-Scholes Greeks for a put option."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    vomma: float
    iv: float
    moneyness: str  # ITM, ATM, OTM

    def greek_score(self) -> float:
        """Weighted Greek quality score (0-100) for entry filtering."""
        d = min(abs(self.delta), 0.50)
        g = min(self.gamma, 0.15)
        v = min(self.vega, 0.30)
        t = max(1.0 - abs(self.theta), 0.0)
        score = (0.35 * d / 0.50 + 0.25 * g / 0.15 + 0.25 * v / 0.30 + 0.15 * t) * 100
        return round(min(score, 100.0), 1)


def bs_put(S: float, K: float, T: float, r: float, sigma: float,
           q: float = 0.0) -> GreeksResult:
    """
    Full Black-Scholes put pricing with all Greeks.

    S: spot price
    K: strike price
    T: time to expiry in years
    r: risk-free rate
    sigma: implied volatility
    q: dividend yield
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(K - S, 0.0)
        m = "ITM" if K > S else ("ATM" if abs(K - S) < 0.01 else "OTM")
        return GreeksResult(intrinsic, -1.0 if K > S else 0.0,
                            0.0, 0.0, 0.0, 0.0, sigma, m)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    nd1 = norm.cdf(-d1)
    nd2 = norm.cdf(-d2)
    phi_d1 = norm.pdf(d1)

    eq = math.exp(-q * T)
    er = math.exp(-r * T)

    # Put price
    price = K * er * nd2 - S * eq * nd1

    # Delta (put)
    delta = -eq * nd1

    # Gamma (same for call/put)
    gamma = eq * phi_d1 / (S * sigma * sqrtT)

    # Vega (per 1% IV move)
    vega = S * eq * phi_d1 * sqrtT / 100.0

    # Theta (per day)
    theta = (-(S * eq * phi_d1 * sigma) / (2 * sqrtT)
             + r * K * er * nd2 - q * S * eq * nd1) / 365.0

    # Vomma (d(vega)/d(sigma))
    vomma = vega * d1 * d2 / sigma

    # Moneyness
    ratio = K / S
    if ratio > 1.02:
        m = "ITM"
    elif ratio < 0.98:
        m = "OTM"
    else:
        m = "ATM"

    return GreeksResult(
        price=round(price, 4),
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        vega=round(vega, 4),
        theta=round(theta, 4),
        vomma=round(vomma, 6),
        iv=sigma,
        moneyness=m,
    )


def bs_call(S: float, K: float, T: float, r: float, sigma: float,
            q: float = 0.0) -> GreeksResult:
    """Full Black-Scholes call pricing with all Greeks."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0)
        m = "ITM" if S > K else ("ATM" if abs(S - K) < 0.01 else "OTM")
        return GreeksResult(intrinsic, 1.0 if S > K else 0.0,
                            0.0, 0.0, 0.0, 0.0, sigma, m)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    phi_d1 = norm.pdf(d1)

    eq = math.exp(-q * T)
    er = math.exp(-r * T)

    price = S * eq * nd1 - K * er * nd2
    delta = eq * nd1
    gamma = eq * phi_d1 / (S * sigma * sqrtT)
    vega = S * eq * phi_d1 * sqrtT / 100.0
    theta = (-(S * eq * phi_d1 * sigma) / (2 * sqrtT)
             - r * K * er * nd2 + q * S * eq * nd1) / 365.0
    vomma = vega * d1 * d2 / sigma

    ratio = S / K
    if ratio > 1.02:
        m = "ITM"
    elif ratio < 0.98:
        m = "OTM"
    else:
        m = "ATM"

    return GreeksResult(
        price=round(price, 4),
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        vega=round(vega, 4),
        theta=round(theta, 4),
        vomma=round(vomma, 6),
        iv=sigma,
        moneyness=m,
    )


# ============================================================================
# PART 3: MONTE CARLO SIMULATION ENGINE
# ============================================================================

@dataclass
class MCResult:
    """Results from a Monte Carlo simulation run."""
    n_paths: int
    horizon_days: int
    # Per-asset final price distributions
    asset_means: dict[str, float]
    asset_medians: dict[str, float]
    asset_p5: dict[str, float]       # 5th percentile (bear)
    asset_p25: dict[str, float]      # 25th percentile
    asset_p75: dict[str, float]      # 75th percentile
    asset_p95: dict[str, float]      # 95th percentile (bull)
    # Portfolio-level
    portfolio_mean: float
    portfolio_median: float
    portfolio_p5: float
    portfolio_p95: float
    var_95: float                    # Value at Risk (95%)
    cvar_95: float                   # Conditional VaR (95%)
    # Scenario probabilities
    prob_oil_above_120: float
    prob_gold_above_3500: float
    prob_spy_below_500: float
    prob_btc_below_60k: float
    prob_portfolio_above_150k: float
    prob_portfolio_above_1m: float
    # Timing
    runtime_ms: float


def run_monte_carlo(
    n_paths: int = MC_PATHS,
    horizon_days: int = MC_HORIZON_DAYS,
    spot_prices: Optional[dict[str, float]] = None,
    drifts: Optional[dict[str, float]] = None,
    vols: Optional[dict[str, float]] = None,
    portfolio_value: float = STARTING_CAPITAL_CAD * CAD_TO_USD,
    seed: Optional[int] = None,
) -> MCResult:
    """
    Run multivariate GBM Monte Carlo with Cholesky decomposition.

    100,000 paths x 11 assets x 90 days.
    Returns percentile distributions, VaR/CVaR, scenario probabilities.
    """
    import time
    t0 = time.perf_counter()

    spots = spot_prices or SPOT_PRICES
    mu = drifts or CRISIS_DRIFTS
    sig = vols or CRISIS_VOLS

    n_assets = len(ASSETS)
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    # Build drift and vol vectors in ASSETS order
    drift_vec = np.array([mu[a] for a in ASSETS])
    vol_vec = np.array([sig[a] for a in ASSETS])
    spot_vec = np.array([spots[a] for a in ASSETS])

    # Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(CORRELATION_MATRIX)

    # Random number generation
    rng = np.random.default_rng(seed)

    # Simulate: shape (n_paths, n_assets)
    # For efficiency, simulate terminal values directly using GBM formula:
    # S_T = S_0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    T = horizon_days * dt
    sqrtT = math.sqrt(T)

    # Correlated standard normals: (n_paths, n_assets)
    Z_indep = rng.standard_normal((n_paths, n_assets))
    Z_corr = Z_indep @ L.T  # apply Cholesky correlation

    # GBM terminal values
    exponent = (drift_vec - 0.5 * vol_vec**2) * T + vol_vec * sqrtT * Z_corr
    S_T = spot_vec * np.exp(exponent)  # shape: (n_paths, n_assets)

    # Compute returns for portfolio impact
    returns = (S_T - spot_vec) / spot_vec  # shape: (n_paths, n_assets)

    # Portfolio P&L estimate using crisis-weighted exposure
    # Weights: how the $45K portfolio is exposed to each asset
    # Based on actual positions + planned deployment
    exposure_weights = np.array([
        0.15,  # oil (puts on energy)
        0.12,  # gold (long)
        0.08,  # silver (long)
        0.10,  # gdx (long miners)
        0.15,  # spy (puts)
        0.05,  # qqq (puts)
        0.08,  # xlf (puts on financials)
        0.07,  # xlre (puts on RE)
        0.05,  # eth
        0.05,  # xrp
        0.10,  # btc
    ])

    # For puts: flip the sign -- we profit from declines
    put_assets = {"spy", "qqq", "xlf", "xlre"}
    long_assets = {"oil", "gold", "silver", "gdx"}
    # Mixed: eth, xrp, btc depend on position

    portfolio_returns = np.zeros(n_paths)
    for i, asset in enumerate(ASSETS):
        if asset in put_assets:
            # Long puts: profit from decline, capped loss at premium
            asset_return = -returns[:, i]  # flip sign
            asset_return = np.clip(asset_return, -1.0, None)  # can't lose more than 100%
        elif asset in long_assets:
            asset_return = returns[:, i]
        else:
            # Crypto: currently mixed (some puts, some hold)
            asset_return = returns[:, i] * 0.5  # partial exposure
        portfolio_returns += exposure_weights[i] * asset_return

    portfolio_values = portfolio_value * (1 + portfolio_returns)

    # Per-asset statistics
    asset_means = {}
    asset_medians = {}
    asset_p5 = {}
    asset_p25 = {}
    asset_p75 = {}
    asset_p95 = {}
    for i, asset in enumerate(ASSETS):
        col = S_T[:, i]
        asset_means[asset] = round(float(np.mean(col)), 2)
        asset_medians[asset] = round(float(np.median(col)), 2)
        asset_p5[asset] = round(float(np.percentile(col, 5)), 2)
        asset_p25[asset] = round(float(np.percentile(col, 25)), 2)
        asset_p75[asset] = round(float(np.percentile(col, 75)), 2)
        asset_p95[asset] = round(float(np.percentile(col, 95)), 2)

    # Portfolio statistics
    pf_sorted = np.sort(portfolio_values)
    var_idx = int(n_paths * 0.05)
    var_95 = portfolio_value - float(pf_sorted[var_idx])
    cvar_95 = portfolio_value - float(np.mean(pf_sorted[:var_idx]))

    # Scenario probabilities (thresholds calibrated Mar 19 evening)
    prob_oil_120 = float(np.mean(S_T[:, ASSETS.index("oil")] > 120))
    prob_gold_3500 = float(np.mean(S_T[:, ASSETS.index("gold")] > 5500))
    prob_spy_500 = float(np.mean(S_T[:, ASSETS.index("spy")] < 600))
    prob_btc_60k = float(np.mean(S_T[:, ASSETS.index("btc")] < 55000))
    prob_pf_150k = float(np.mean(portfolio_values > 150_000))
    prob_pf_1m = float(np.mean(portfolio_values > 1_000_000))

    runtime = (time.perf_counter() - t0) * 1000

    return MCResult(
        n_paths=n_paths,
        horizon_days=horizon_days,
        asset_means=asset_means,
        asset_medians=asset_medians,
        asset_p5=asset_p5,
        asset_p25=asset_p25,
        asset_p75=asset_p75,
        asset_p95=asset_p95,
        portfolio_mean=round(float(np.mean(portfolio_values)), 2),
        portfolio_median=round(float(np.median(portfolio_values)), 2),
        portfolio_p5=round(float(np.percentile(portfolio_values, 5)), 2),
        portfolio_p95=round(float(np.percentile(portfolio_values, 95)), 2),
        var_95=round(var_95, 2),
        cvar_95=round(cvar_95, 2),
        prob_oil_above_120=round(prob_oil_120, 4),
        prob_gold_above_3500=round(prob_gold_3500, 4),
        prob_spy_below_500=round(prob_spy_500, 4),
        prob_btc_below_60k=round(prob_btc_60k, 4),
        prob_portfolio_above_150k=round(prob_pf_150k, 4),
        prob_portfolio_above_1m=round(prob_pf_1m, 4),
        runtime_ms=round(runtime, 1),
    )


# ============================================================================
# PART 4: 50-MILESTONE SPIDERWEB SYSTEM
# ============================================================================

class MilestoneCategory(Enum):
    DOLLAR = "dollar"           # Portfolio value milestones
    OIL = "oil"                 # Oil price triggers
    GOLD = "gold"               # Gold price triggers
    CREDIT = "credit"           # Credit/BDC stress triggers
    CRYPTO = "crypto"           # Crypto market triggers
    MACRO = "macro"             # Macro/VIX/Fed triggers
    GEOPOLITICAL = "geopolitical"  # Geopolitical event triggers
    DEFI = "defi"               # DeFi stress triggers
    EQUITY = "equity"           # Equity market triggers
    PHASE = "phase"             # Phase transition milestones


@dataclass
class Milestone:
    """Single milestone in the spiderweb."""
    id: int
    name: str
    category: MilestoneCategory
    trigger_condition: str       # human-readable condition
    threshold_value: float       # numeric threshold
    threshold_field: str         # which field to check
    threshold_op: str            # ">" or "<"
    strategy_action: str         # what to do when triggered
    confidence: float            # 0-1 confidence this happens
    leads_to: list[int]          # IDs of milestones this triggers
    phase: str                   # which phase this belongs to
    triggered: bool = False
    triggered_date: Optional[str] = None


def _build_milestones() -> list[Milestone]:
    """Build the 50-milestone spiderweb."""
    ms = []

    # -- DOLLAR MILESTONES (10) --
    ms.append(Milestone(1, "First $50K", MilestoneCategory.DOLLAR,
        "Portfolio crosses $50,000 USD", 50_000, "portfolio_usd", ">",
        "Confirm conviction. Add 5% to put pyramid.", 0.85, [2, 11],
        "accumulation"))
    ms.append(Milestone(2, "Double Up $75K", MilestoneCategory.DOLLAR,
        "Portfolio crosses $75,000 USD", 75_000, "portfolio_usd", ">",
        "Scale gold/silver to 20% allocation. Tighten stops.", 0.70, [3, 12],
        "accumulation"))
    ms.append(Milestone(3, "Six Figures $100K", MilestoneCategory.DOLLAR,
        "Portfolio crosses $100,000 USD", 100_000, "portfolio_usd", ">",
        "Reduce put leverage 15%. Add GDX calls. Start BDC short.", 0.55, [4, 21],
        "accumulation"))
    ms.append(Milestone(4, "De-Risk Gate $150K", MilestoneCategory.DOLLAR,
        "Portfolio crosses $150,000 USD", 150_000, "portfolio_usd", ">",
        "PHASE TRANSITION: Close 30% puts. Rotate to gold/miners. Treasury 10%.", 0.40, [5, 31],
        "growth"))
    ms.append(Milestone(5, "Quarter Million $250K", MilestoneCategory.DOLLAR,
        "Portfolio crosses $250,000 USD", 250_000, "portfolio_usd", ">",
        "Add real estate shorts. Increase private credit exposure.", 0.30, [6],
        "growth"))
    ms.append(Milestone(6, "Half Million $500K", MilestoneCategory.DOLLAR,
        "Portfolio crosses $500,000 USD", 500_000, "portfolio_usd", ">",
        "Reduce options to 25% of book. Add income strategies.", 0.20, [7],
        "growth"))
    ms.append(Milestone(7, "Millionaire Gate $1M", MilestoneCategory.DOLLAR,
        "Portfolio crosses $1,000,000 USD", 1_000_000, "portfolio_usd", ">",
        "PHASE TRANSITION: Income mode. Max 15% options. 40% fixed income.", 0.10, [8],
        "rotation"))
    ms.append(Milestone(8, "Two Million $2M", MilestoneCategory.DOLLAR,
        "Portfolio crosses $2,000,000 USD", 2_000_000, "portfolio_usd", ">",
        "Add infrastructure plays. Diversify internationally.", 0.05, [9],
        "rotation"))
    ms.append(Milestone(9, "Five Million Gate $5M", MilestoneCategory.DOLLAR,
        "Portfolio crosses $5,000,000 USD", 5_000_000, "portfolio_usd", ">",
        "PHASE TRANSITION: Preservation mode. Max 5% options. 60% yield.", 0.02, [10],
        "preservation"))
    ms.append(Milestone(10, "Hundred Million $100M", MilestoneCategory.DOLLAR,
        "Portfolio crosses $100,000,000 USD", 100_000_000, "portfolio_usd", ">",
        "Endowment mode. Family office structure.", 0.001, [],
        "preservation"))

    # -- OIL MILESTONES (6) --
    ms.append(Milestone(11, "Oil Breaks $105", MilestoneCategory.OIL,
        "Oil price crosses $105/bbl", 105, "oil_price", ">",
        "Add 10% to energy puts. Double SPY put size.", 0.65, [12, 21, 41],
        "accumulation"))
    ms.append(Milestone(12, "Oil Spike $120", MilestoneCategory.OIL,
        "Oil price crosses $120/bbl (Hormuz disruption)", 120, "oil_price", ">",
        "Max out energy exposure. Buy XLE calls as hedge. Alert: recession imminent.", 0.45, [13, 22, 42],
        "accumulation"))
    ms.append(Milestone(13, "Oil Superspike $150", MilestoneCategory.OIL,
        "Oil price crosses $150/bbl (full blockade)", 150, "oil_price", ">",
        "Begin profit-taking on oil calls. Rotate to gold. SPY puts at max.", 0.25, [14, 3],
        "accumulation"))
    ms.append(Milestone(14, "Oil Reversal Below $90", MilestoneCategory.OIL,
        "Oil price drops below $90 (de-escalation)", 90, "oil_price", "<",
        "CLOSE all oil longs. Close energy puts. Rotate to equities.", 0.40, [35],
        "accumulation"))
    ms.append(Milestone(15, "Oil Contango Widen", MilestoneCategory.OIL,
        "Oil contango spread exceeds $5", 5, "oil_contango", ">",
        "Add calendar put spreads on oil ETFs. Storage play.", 0.50, [11],
        "accumulation"))
    ms.append(Milestone(16, "OPEC+ Emergency Cut", MilestoneCategory.OIL,
        "OPEC announces emergency production cut", 1, "opec_emergency", ">",
        "Oil to $130+. Max XLE calls. Double put pyramid on SPY.", 0.35, [12, 13],
        "accumulation"))

    # -- GOLD MILESTONES (5) --
    ms.append(Milestone(17, "Gold Breaks $3200", MilestoneCategory.GOLD,
        "Gold crosses $3,200/oz", 3200, "gold_price", ">",
        "Add 8% to GDX. Silver follows. Increase miners.", 0.70, [18, 2],
        "accumulation"))
    ms.append(Milestone(18, "Gold $3500 ATH", MilestoneCategory.GOLD,
        "Gold crosses $3,500/oz (all-time high territory)", 3500, "gold_price", ">",
        "Take 20% gold profit. Rotate to silver. GDX at full size.", 0.50, [19],
        "accumulation"))
    ms.append(Milestone(19, "Gold $4000 Blowoff", MilestoneCategory.GOLD,
        "Gold crosses $4,000/oz", 4000, "gold_price", ">",
        "Sell 40% gold. Silver still running. Physical considered.", 0.25, [20],
        "growth"))
    ms.append(Milestone(20, "Gold Reversal Below $2800", MilestoneCategory.GOLD,
        "Gold drops below $2,800 (risk-on)", 2800, "gold_price", "<",
        "Reduce gold to 5%. Miners to zero. Check if crisis fading.", 0.30, [35],
        "accumulation"))
    ms.append(Milestone(21, "Silver Outperforms Gold 2:1", MilestoneCategory.GOLD,
        "Silver/Gold ratio improves by 2x", 2.0, "silver_gold_ratio_change", ">",
        "Rotate gold to silver. Industrial demand + safe haven = double bid.", 0.40, [18],
        "accumulation"))

    # -- CREDIT / BDC MILESTONES (5) --
    ms.append(Milestone(22, "HY Spread Widens 500bp", MilestoneCategory.CREDIT,
        "High-yield credit spread exceeds 500bp", 500, "hy_spread_bp", ">",
        "Max BDC puts. Add HYG puts. Private credit arm at full.", 0.45, [23, 3],
        "accumulation"))
    ms.append(Milestone(23, "BDC NAV Discount 15%+", MilestoneCategory.CREDIT,
        "Average BDC trades at 15%+ NAV discount", 15, "bdc_nav_discount", ">",
        "Roll BDC puts deeper. Watch for non-accrual spike.", 0.40, [24],
        "accumulation"))
    ms.append(Milestone(24, "Non-Accrual Spike >5%", MilestoneCategory.CREDIT,
        "BDC non-accrual rate exceeds 5%", 5, "bdc_nonaccrual_pct", ">",
        "Max private credit shorts. Dividend cuts coming. Add FSK/TCPC puts.", 0.30, [25],
        "accumulation"))
    ms.append(Milestone(25, "PE Firm Revenue Drop >20%", MilestoneCategory.CREDIT,
        "BX/KKR/APO revenue drops 20%+", 20, "pe_revenue_drop_pct", ">",
        "Add PE firm puts. Redemption wave starting. Carry trade unwinding.", 0.25, [4],
        "accumulation"))
    ms.append(Milestone(26, "Credit Normalization", MilestoneCategory.CREDIT,
        "HY spread drops below 350bp", 350, "hy_spread_bp", "<",
        "Close all credit shorts. Rotate to equity recovery plays.", 0.35, [35],
        "accumulation"))

    # -- CRYPTO MILESTONES (5) --
    ms.append(Milestone(27, "BTC Breaks Below $70K", MilestoneCategory.CRYPTO,
        "BTC price drops below $70,000", 70_000, "btc_price", "<",
        "Add BITO puts. Crypto contagion starting. Watch stablecoins.", 0.55, [28],
        "accumulation"))
    ms.append(Milestone(28, "BTC Below $50K", MilestoneCategory.CRYPTO,
        "BTC price drops below $50,000 (crypto winter)", 50_000, "btc_price", "<",
        "Max BITO puts. Stablecoin depeg watch. DeFi liquidation cascade.", 0.30, [29, 36],
        "accumulation"))
    ms.append(Milestone(29, "Stablecoin Depeg >2%", MilestoneCategory.CRYPTO,
        "Major stablecoin depegs by 2%+", 2, "stablecoin_depeg_pct", ">",
        "EMERGENCY: Close all crypto longs. Max puts. Cash position.", 0.15, [30],
        "accumulation"))
    ms.append(Milestone(30, "BTC Recovery Above $90K", MilestoneCategory.CRYPTO,
        "BTC recovers above $90,000", 90_000, "btc_price", ">",
        "Close crypto puts. Consider DCA back into ETH/BTC.", 0.40, [35],
        "accumulation"))
    ms.append(Milestone(31, "ETH Below $1500", MilestoneCategory.CRYPTO,
        "ETH drops below $1,500", 1500, "eth_price", "<",
        "DeFi cascade confirmed. Add AAVE/UNI puts. Yield farming frozen.", 0.35, [36],
        "accumulation"))

    # -- MACRO / VIX MILESTONES (5) --
    ms.append(Milestone(32, "VIX Breaks 30", MilestoneCategory.MACRO,
        "VIX crosses above 30", 30, "vix", ">",
        "Full crisis mode. All puts at max vega. Gamma scalping active.", 0.50, [33, 22],
        "accumulation"))
    ms.append(Milestone(33, "VIX Spikes to 45+", MilestoneCategory.MACRO,
        "VIX crosses above 45 (panic)", 45, "vix", ">",
        "Begin taking profit on puts. IV too high for new entries. Sell vol.", 0.20, [34],
        "accumulation"))
    ms.append(Milestone(34, "Fed Emergency Cut", MilestoneCategory.MACRO,
        "Fed announces emergency rate cut", 1, "fed_emergency_cut", ">",
        "PIVOT: Close 50% puts. Add financials longs. Credit stabilizing.", 0.15, [35],
        "accumulation"))
    ms.append(Milestone(35, "VIX Returns Below 20", MilestoneCategory.MACRO,
        "VIX drops below 20 (calm restored)", 20, "vix", "<",
        "Crisis fading. Rotate to equities and income. Reduce options 50%.", 0.45, [4],
        "accumulation"))
    ms.append(Milestone(36, "DeFi TVL Drops 40%+", MilestoneCategory.DEFI,
        "Total DeFi TVL drops 40%+ from peak", 40, "defi_tvl_drop_pct", ">",
        "DeFi arm at max. AAVE/UNI puts. Yield farming short.", 0.30, [28],
        "accumulation"))

    # -- GEOPOLITICAL MILESTONES (5) --
    ms.append(Milestone(37, "Hormuz Strait Disruption", MilestoneCategory.GEOPOLITICAL,
        "Confirmed shipping disruption in Strait of Hormuz", 1, "hormuz_disrupted", ">",
        "Oil to $120+. Max energy calls + SPY puts. Gold spike.", 0.35, [12, 17, 32],
        "accumulation"))
    ms.append(Milestone(38, "Iran Sanctions Escalation", MilestoneCategory.GEOPOLITICAL,
        "New Iran sanctions round announced", 1, "iran_sanctions_new", ">",
        "Oil +10%. Add OXY/XLE calls. Tighten put levels.", 0.55, [11],
        "accumulation"))
    ms.append(Milestone(39, "BRICS Gold Settlement", MilestoneCategory.GEOPOLITICAL,
        "BRICS announces gold-backed settlement mechanism", 1, "brics_gold", ">",
        "Gold to $4000+. USD weakness. Max gold/silver/GDX.", 0.20, [19],
        "growth"))
    ms.append(Milestone(40, "US-Iran Diplomatic Breakthrough", MilestoneCategory.GEOPOLITICAL,
        "Diplomatic resolution to Iran tensions", 1, "iran_diplomacy", ">",
        "Oil crashes to $80. CLOSE ALL oil longs/puts. Rotate to equities.", 0.25, [14, 35],
        "accumulation"))
    ms.append(Milestone(41, "Taiwan Strait Incident", MilestoneCategory.GEOPOLITICAL,
        "Military incident in Taiwan Strait", 1, "taiwan_incident", ">",
        "QQQ puts max. TSMC supply chain shock. Gold spike. BTC uncertain.", 0.10, [32, 17],
        "accumulation"))

    # -- EQUITY MILESTONES (4) --
    ms.append(Milestone(42, "SPY Breaks Below 520", MilestoneCategory.EQUITY,
        "SPY drops below 520 (bear market)", 520, "spy_price", "<",
        "Take 25% put profit. Roll remaining deeper. Add IWM puts.", 0.45, [43],
        "accumulation"))
    ms.append(Milestone(43, "SPY Below 480 (Crash)", MilestoneCategory.EQUITY,
        "SPY drops below 480 (-15% from high)", 480, "spy_price", "<",
        "Take 50% put profit. Start adding SPY longs for recovery.", 0.25, [4],
        "accumulation"))
    ms.append(Milestone(44, "QQQ Below 400", MilestoneCategory.EQUITY,
        "QQQ drops below 400 (tech crash)", 400, "qqq_price", "<",
        "Close QQQ puts. Add TQQQ for recovery bounce. Watch earnings.", 0.20, [43],
        "accumulation"))
    ms.append(Milestone(45, "XLF Below 35", MilestoneCategory.EQUITY,
        "XLF drops below 35 (financial stress)", 35, "xlf_price", "<",
        "Banking stress confirmed. Max BDC puts. Regional bank cascade.", 0.30, [22, 24],
        "accumulation"))

    # -- PHASE MILESTONES (5) --
    ms.append(Milestone(46, "First Profitable Week", MilestoneCategory.PHASE,
        "Weekly P&L positive for first time", 0, "weekly_pnl", ">",
        "System validated. Increase position sizes 10%.", 0.90, [1],
        "accumulation"))
    ms.append(Milestone(47, "First 10x Trade", MilestoneCategory.PHASE,
        "Any single position returns 10x+", 10, "max_return_multiple", ">",
        "Take 80% profit. Reinvest 20% into next opportunity.", 0.15, [3],
        "accumulation"))
    ms.append(Milestone(48, "30-Day Streak Positive", MilestoneCategory.PHASE,
        "Portfolio positive for 30 consecutive days", 30, "positive_streak_days", ">",
        "System proven. Scale all positions 25%. Add new arms.", 0.20, [5],
        "growth"))
    ms.append(Milestone(49, "Max Drawdown Hit 25%", MilestoneCategory.PHASE,
        "Portfolio drawdown exceeds 25%", 25, "drawdown_pct", ">",
        "EMERGENCY: Cut all positions 50%. Review thesis. Cash heavy.", 0.35, [],
        "accumulation"))
    ms.append(Milestone(50, "System Running 90 Days", MilestoneCategory.PHASE,
        "War room operational for 90 days", 90, "days_running", ">",
        "Full review. Rebalance all arms. Update all milestones.", 0.95, [4],
        "accumulation"))

    return ms


MILESTONES = _build_milestones()


def check_milestones(state: dict[str, float]) -> list[Milestone]:
    """Check which milestones are newly triggered given current state."""
    newly_triggered = []
    for m in MILESTONES:
        if m.triggered:
            continue
        val = state.get(m.threshold_field)
        if val is None:
            continue
        if m.threshold_op == ">" and val > m.threshold_value:
            m.triggered = True
            m.triggered_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            newly_triggered.append(m)
        elif m.threshold_op == "<" and val < m.threshold_value:
            m.triggered = True
            m.triggered_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            newly_triggered.append(m)
    return newly_triggered


def get_spiderweb_chains(milestone_id: int) -> list[list[Milestone]]:
    """Trace all causal chains from a milestone through leads_to."""
    by_id = {m.id: m for m in MILESTONES}
    chains: list[list[Milestone]] = []

    def _trace(current_id: int, path: list[Milestone], visited: set[int]) -> None:
        if current_id in visited:
            return
        m = by_id.get(current_id)
        if m is None:
            return
        visited.add(current_id)
        new_path = path + [m]
        if not m.leads_to:
            chains.append(new_path)
            return
        for next_id in m.leads_to:
            _trace(next_id, new_path, visited.copy())

    _trace(milestone_id, [], set())
    return chains


# ============================================================================
# PART 5: 12-INDICATOR COMPOSITE MODEL
# ============================================================================

@dataclass
class IndicatorState:
    """Current reading of the 12-indicator model -- LIVE March 19 evening."""
    oil_price: float = 95.0
    gold_price: float = 4861.0
    vix: float = 25.0
    hy_spread_bp: float = 550.0
    bdc_nav_discount: float = 15.0
    bdc_nonaccrual_pct: float = 3.5
    defi_tvl_change_pct: float = -20.0
    stablecoin_depeg_pct: float = 0.1
    btc_price: float = 68000.0
    fed_funds_rate: float = 4.5
    dxy: float = 103.0
    spy_price: float = 665.0


def compute_composite_score(ind: IndicatorState) -> dict[str, Any]:
    """
    Compute the composite crisis score (0-100) from 12 indicators.
    Higher = more crisis = more aggressive positioning.
    """
    scores = {}

    # 1. Oil price (higher = more crisis)
    if ind.oil_price > 120:
        scores["oil"] = 100
    elif ind.oil_price > 105:
        scores["oil"] = 70 + (ind.oil_price - 105) * 2
    elif ind.oil_price > 95:
        scores["oil"] = 40 + (ind.oil_price - 95) * 3
    else:
        scores["oil"] = max(0, ind.oil_price - 70)

    # 2. Gold price (higher = more flight to safety)
    if ind.gold_price > 3500:
        scores["gold"] = 100
    elif ind.gold_price > 3200:
        scores["gold"] = 70 + (ind.gold_price - 3200) * 0.1
    elif ind.gold_price > 3000:
        scores["gold"] = 40 + (ind.gold_price - 3000) * 0.15
    else:
        scores["gold"] = max(0, (ind.gold_price - 2500) * 0.08)

    # 3. VIX (higher = more fear)
    if ind.vix > 40:
        scores["vix"] = 100
    elif ind.vix > 30:
        scores["vix"] = 70 + (ind.vix - 30) * 3
    elif ind.vix > 22:
        scores["vix"] = 30 + (ind.vix - 22) * 5
    else:
        scores["vix"] = max(0, (ind.vix - 15) * 4.3)

    # 4. HY spread (wider = more stress)
    if ind.hy_spread_bp > 600:
        scores["hy_spread"] = 100
    elif ind.hy_spread_bp > 450:
        scores["hy_spread"] = 60 + (ind.hy_spread_bp - 450) * 0.267
    elif ind.hy_spread_bp > 350:
        scores["hy_spread"] = 20 + (ind.hy_spread_bp - 350) * 0.4
    else:
        scores["hy_spread"] = max(0, (ind.hy_spread_bp - 250) * 0.2)

    # 5. BDC NAV discount (higher = more stress)
    if ind.bdc_nav_discount > 20:
        scores["bdc_nav"] = 100
    elif ind.bdc_nav_discount > 12:
        scores["bdc_nav"] = 60 + (ind.bdc_nav_discount - 12) * 5
    elif ind.bdc_nav_discount > 5:
        scores["bdc_nav"] = 10 + (ind.bdc_nav_discount - 5) * 7.14
    else:
        scores["bdc_nav"] = max(0, ind.bdc_nav_discount * 2)

    # 6. BDC non-accrual (higher = more stress)
    if ind.bdc_nonaccrual_pct > 6:
        scores["bdc_nonaccrual"] = 100
    elif ind.bdc_nonaccrual_pct > 3:
        scores["bdc_nonaccrual"] = 50 + (ind.bdc_nonaccrual_pct - 3) * 16.67
    else:
        scores["bdc_nonaccrual"] = ind.bdc_nonaccrual_pct * 16.67

    # 7. DeFi TVL change (more negative = more stress)
    if ind.defi_tvl_change_pct < -40:
        scores["defi_tvl"] = 100
    elif ind.defi_tvl_change_pct < -20:
        scores["defi_tvl"] = 50 + abs(ind.defi_tvl_change_pct + 20) * 2.5
    elif ind.defi_tvl_change_pct < 0:
        scores["defi_tvl"] = abs(ind.defi_tvl_change_pct) * 2.5
    else:
        scores["defi_tvl"] = 0

    # 8. Stablecoin depeg (higher = more crisis)
    if ind.stablecoin_depeg_pct > 3:
        scores["stablecoin"] = 100
    elif ind.stablecoin_depeg_pct > 1:
        scores["stablecoin"] = 50 + (ind.stablecoin_depeg_pct - 1) * 25
    else:
        scores["stablecoin"] = ind.stablecoin_depeg_pct * 50

    # 9. BTC price (lower = more crypto stress)
    if ind.btc_price < 50000:
        scores["btc"] = 100
    elif ind.btc_price < 70000:
        scores["btc"] = 50 + (70000 - ind.btc_price) / 400
    elif ind.btc_price < 85000:
        scores["btc"] = (85000 - ind.btc_price) / 300
    else:
        scores["btc"] = 0

    # 10. Fed funds rate (higher = more tightening stress)
    if ind.fed_funds_rate > 5.5:
        scores["fed_rate"] = 100
    elif ind.fed_funds_rate > 4.5:
        scores["fed_rate"] = 50 + (ind.fed_funds_rate - 4.5) * 50
    elif ind.fed_funds_rate > 3.5:
        scores["fed_rate"] = (ind.fed_funds_rate - 3.5) * 50
    else:
        scores["fed_rate"] = 0

    # 11. DXY (higher = stronger dollar = EM stress)
    if ind.dxy > 110:
        scores["dxy"] = 100
    elif ind.dxy > 105:
        scores["dxy"] = 50 + (ind.dxy - 105) * 10
    elif ind.dxy > 100:
        scores["dxy"] = (ind.dxy - 100) * 10
    else:
        scores["dxy"] = 0

    # 12. SPY price (lower = more equity stress)
    if ind.spy_price < 480:
        scores["spy"] = 100
    elif ind.spy_price < 520:
        scores["spy"] = 50 + (520 - ind.spy_price) * 1.25
    elif ind.spy_price < 560:
        scores["spy"] = (560 - ind.spy_price) * 1.25
    else:
        scores["spy"] = 0

    # Weighted composite
    weights = {
        "oil": 0.15, "gold": 0.10, "vix": 0.12, "hy_spread": 0.10,
        "bdc_nav": 0.08, "bdc_nonaccrual": 0.07, "defi_tvl": 0.06,
        "stablecoin": 0.05, "btc": 0.07, "fed_rate": 0.06,
        "dxy": 0.06, "spy": 0.08,
    }

    composite = sum(scores[k] * weights[k] for k in scores)

    return {
        "composite_score": round(composite, 1),
        "individual_scores": {k: round(v, 1) for k, v in scores.items()},
        "regime": ("CRISIS" if composite > 70 else
                   "ELEVATED" if composite > 50 else
                   "WATCH" if composite > 30 else "CALM"),
        "confidence": round(min(composite / 100, 1.0), 2),
    }


# ============================================================================
# PART 6: 5-ARM POSITION MANAGEMENT
# ============================================================================

class ArmType(Enum):
    IRAN_OIL = "iran_oil"
    BDC_NONACCRUAL = "bdc_nonaccrual"
    CRYPTO_METALS = "crypto_metals"
    DEFI_YIELD = "defi_yield"
    TRADFI_ROTATE = "tradfi_rotate"


@dataclass
class Position:
    """A single position in the portfolio."""
    arm: ArmType
    symbol: str
    position_type: str   # "put", "call", "long", "short"
    quantity: int
    entry_price: float
    current_price: float
    strike: Optional[float] = None
    expiry: Optional[str] = None
    greeks: Optional[GreeksResult] = None
    account: str = "IBKR"

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price * (100 if "put" in self.position_type or "call" in self.position_type else 1)

    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity * (100 if "put" in self.position_type or "call" in self.position_type else 1)

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class ArmAllocation:
    """Allocation rules for each arm by phase."""
    arm: ArmType
    name: str
    target_pct: float          # target % of portfolio
    max_pct: float             # hard cap
    instruments: list[str]     # typical instruments
    entry_conditions: str      # when to enter
    exit_conditions: str       # when to exit


def get_arm_allocations(phase: str) -> list[ArmAllocation]:
    """Get arm allocations for the current phase."""
    if phase == "accumulation":
        return [
            ArmAllocation(ArmType.IRAN_OIL, "Iran/Oil Crisis", 0.30, 0.40,
                ["SPY puts", "QQQ puts", "XLE calls", "USO calls"],
                "Oil >$100, VIX >22, Hormuz tension", "Oil <$90 or VIX <18"),
            ArmAllocation(ArmType.BDC_NONACCRUAL, "Private Credit/BDC", 0.25, 0.35,
                ["FSK puts", "TCPC puts", "HYG puts", "BX puts"],
                "HY spread >400bp, NAV discount >10%", "HY spread <300bp"),
            ArmAllocation(ArmType.CRYPTO_METALS, "Crypto & Metals", 0.20, 0.30,
                ["GLD calls", "SLV calls", "BITO puts", "GDX calls"],
                "Gold >$3000, BTC <$80K", "Gold <$2800, BTC >$90K"),
            ArmAllocation(ArmType.DEFI_YIELD, "DeFi Yield Farm", 0.15, 0.25,
                ["AAVE puts", "UNI puts", "DeFi shorts"],
                "DeFi TVL drop >15%, yield compression", "DeFi TVL recovery"),
            ArmAllocation(ArmType.TRADFI_ROTATE, "TradFi Rotate", 0.10, 0.20,
                ["Treasury ETFs", "Dividend ETFs", "Income plays"],
                "Composite <40, recovery signals", "Crisis re-escalation"),
        ]
    elif phase == "growth":
        return [
            ArmAllocation(ArmType.IRAN_OIL, "Iran/Oil (Reduced)", 0.20, 0.25,
                ["SPY puts (smaller)", "XLE calls"], "Confirmed escalation only", "Any de-escalation"),
            ArmAllocation(ArmType.BDC_NONACCRUAL, "Private Credit", 0.15, 0.20,
                ["BDC puts (selective)", "HYG puts"], "Non-accrual spike only", "Spreads normalize"),
            ArmAllocation(ArmType.CRYPTO_METALS, "Metals (Dominant)", 0.30, 0.40,
                ["GLD", "SLV", "GDX", "Silver miners"], "Gold/silver trend", "Reversal below SMA50"),
            ArmAllocation(ArmType.DEFI_YIELD, "DeFi (Reduced)", 0.10, 0.15,
                ["Selective DeFi shorts"], "Major stress only", "Any stability"),
            ArmAllocation(ArmType.TRADFI_ROTATE, "TradFi Income", 0.25, 0.35,
                ["Treasuries", "Dividend aristocrats", "REITs"], "Yield opportunity", "Rate spike"),
        ]
    else:  # rotation / preservation
        return [
            ArmAllocation(ArmType.IRAN_OIL, "Oil (Minimal)", 0.05, 0.10,
                ["Energy ETF hedges only"], "Black swan only", "Immediately"),
            ArmAllocation(ArmType.BDC_NONACCRUAL, "Credit (Minimal)", 0.05, 0.10,
                ["Credit hedges only"], "Systemic risk only", "Immediately"),
            ArmAllocation(ArmType.CRYPTO_METALS, "Metals Preservation", 0.25, 0.30,
                ["Physical gold allocation", "GLD", "SLV"], "Inflation hedge", "Deflation risk"),
            ArmAllocation(ArmType.DEFI_YIELD, "DeFi (Zero)", 0.00, 0.05,
                [], "Never in preservation", "N/A"),
            ArmAllocation(ArmType.TRADFI_ROTATE, "TradFi Income (Core)", 0.65, 0.80,
                ["Treasuries", "Munis", "IG bonds", "Dividend ETFs"],
                "Always in preservation", "Never exit entirely"),
        ]


# Current positions -- VERIFIED from IBKR TWS port 7497, March 20 2026
# All avgCost and mktVal pulled live via ib_insync portfolio()
CURRENT_POSITIONS = [
    # === 9 REAL POSITIONS from IBKR account U24346218 ===
    Position(ArmType.BDC_NONACCRUAL, "ARCC", "put", 1, 0.25, 0.26,
             strike=17.0, expiry="2026-04-17", account="IBKR"),
    Position(ArmType.BDC_NONACCRUAL, "PFF", "put", 1, 0.17, 0.03,
             strike=29.0, expiry="2026-04-17", account="IBKR"),
    Position(ArmType.TRADFI_ROTATE, "LQD", "put", 1, 0.63, 0.66,
             strike=106.0, expiry="2026-05-15", account="IBKR"),
    Position(ArmType.TRADFI_ROTATE, "EMB", "put", 1, 0.48, 0.82,
             strike=90.0, expiry="2026-05-15", account="IBKR"),
    Position(ArmType.BDC_NONACCRUAL, "MAIN", "put", 1, 0.73, 0.45,
             strike=49.7, expiry="2026-04-17", account="IBKR"),
    Position(ArmType.TRADFI_ROTATE, "JNK", "put", 1, 0.35, 0.37,
             strike=92.0, expiry="2026-04-17", account="IBKR"),
    Position(ArmType.BDC_NONACCRUAL, "BKLN", "put", 3, 0.40, 0.21,
             strike=20.0, expiry="2026-06-18", account="IBKR"),
    Position(ArmType.TRADFI_ROTATE, "HYG", "put", 1, 0.80, 0.74,
             strike=77.0, expiry="2026-06-18", account="IBKR"),
    Position(ArmType.IRAN_OIL, "XLF", "put", 1, 0.75, 0.69,
             strike=46.0, expiry="2026-05-01", account="IBKR"),
]

# Account balances -- VERIFIED sources noted, unverified marked
ACCOUNTS = {
    # IBKR: live from TWS port 7497, Mar 20 2026
    "IBKR": {"balance_usd": 185.56, "type": "paper_trading",
             "note": "TWS port 7497, option_mkt_val=$464.34, uPnL=-$71.14"},
    # NDAX: last confirmed Mar 18 liquidation, no live API pull (NDAX_LOGIN not set)
    "NDAX": {"balance_cad": 4492.04, "type": "crypto_liquidated",
             "note": "unverified — last known from Mar 18 sell"},
    # Moomoo: Tier 1 alongside IBKR — REAL mode, FUTUCA
    "Moomoo": {"balance_usd": 365.15, "type": "brokerage_live",
               "note": "Tier 1 — FUTUCA firm, REAL mode, trade PIN set"},
    # WealthSimple: no API — manual check only
    "WealthSimple": {"balance_cad": 0.0, "type": "tfsa",
                     "note": "unverified — no API integration"},
    # EQ Bank: no API — manual check only
    "EQ_Bank": {"balance_cad": 0.0, "type": "savings",
                "note": "unverified — no API integration"},
}


def get_portfolio_value_usd() -> float:
    """Calculate total portfolio value in USD."""
    total = 0.0
    for name, acct in ACCOUNTS.items():
        if "balance_usd" in acct:
            total += acct["balance_usd"]
        elif "balance_cad" in acct:
            total += acct["balance_cad"] * CAD_TO_USD
    # Add position values
    for pos in CURRENT_POSITIONS:
        total += pos.market_value
    return round(total, 2)


def get_current_phase() -> str:
    """Determine current portfolio phase."""
    val = get_portfolio_value_usd()
    for phase, (lo, hi) in PHASE_THRESHOLDS.items():
        if lo <= val < hi:
            return phase
    return "preservation"


# ============================================================================
# PART 7: SCENARIO ENGINE
# ============================================================================

SCENARIOS = {
    "hormuz_closure": {
        "name": "Hormuz Strait Closure",
        "description": "Full naval blockade of Strait of Hormuz by Iran",
        "oil_price": 155.0, "gold_price": 5800.0, "vix": 42.0,
        "spy_price": 580.0, "btc_price": 52000.0, "hy_spread_bp": 700.0,
        "probability": 0.15,
        "drift_override": {"oil": 1.80, "gold": 1.20, "spy": -0.55, "btc": -0.80},
    },
    "iran_deescalation": {
        "name": "Iran De-Escalation",
        "description": "Diplomatic resolution, sanctions eased",
        "oil_price": 78.0, "gold_price": 3800.0, "vix": 16.0,
        "spy_price": 700.0, "btc_price": 95000.0, "hy_spread_bp": 300.0,
        "probability": 0.15,
        "drift_override": {"oil": -0.30, "gold": -0.25, "spy": 0.25, "btc": 0.50},
    },
    "credit_cascade": {
        "name": "Private Credit Cascade",
        "description": "BDC non-accruals spike, PE fund redemptions, HY spread blowout",
        "oil_price": 100.0, "gold_price": 5200.0, "vix": 38.0,
        "spy_price": 550.0, "btc_price": 55000.0, "hy_spread_bp": 800.0,
        "probability": 0.20,
        "drift_override": {"xlf": -0.65, "xlre": -0.50, "spy": -0.45},
    },
    "defi_collapse": {
        "name": "DeFi Systemic Collapse",
        "description": "Major stablecoin depegs, DeFi TVL -50%, liquidation cascade",
        "oil_price": 92.0, "gold_price": 5000.0, "vix": 35.0,
        "spy_price": 600.0, "btc_price": 38000.0, "hy_spread_bp": 600.0,
        "probability": 0.10,
        "drift_override": {"btc": -1.20, "eth": -1.50, "xrp": -1.30},
    },
    "soft_landing": {
        "name": "Soft Landing (Base Case)",
        "description": "Gradual normalization, mild recession, oil stable",
        "oil_price": 85.0, "gold_price": 4000.0, "vix": 17.0,
        "spy_price": 680.0, "btc_price": 85000.0, "hy_spread_bp": 350.0,
        "probability": 0.20,
        "drift_override": {"oil": -0.10, "gold": -0.15, "spy": 0.12},
    },
    "black_swan": {
        "name": "Black Swan (Multi-Front Crisis)",
        "description": "Hormuz + credit cascade + DeFi collapse simultaneously",
        "oil_price": 200.0, "gold_price": 6500.0, "vix": 60.0,
        "spy_price": 420.0, "btc_price": 28000.0, "hy_spread_bp": 1000.0,
        "probability": 0.05,
        "drift_override": {"oil": 2.50, "gold": 2.00, "spy": -0.75, "btc": -1.50,
                           "xlf": -0.85, "xlre": -0.60, "eth": -1.80},
    },
    "gold_supercycle": {
        "name": "Gold Supercycle",
        "description": "BRICS gold settlement + central bank buying + USD weakness",
        "oil_price": 105.0, "gold_price": 7000.0, "vix": 28.0,
        "spy_price": 620.0, "btc_price": 60000.0, "hy_spread_bp": 500.0,
        "probability": 0.10,
        "drift_override": {"gold": 2.00, "silver": 1.80, "gdx": 2.50},
    },
}


def run_scenario_mc(scenario_name: str, n_paths: int = 50_000) -> MCResult:
    """Run Monte Carlo with scenario-specific parameters."""
    sc = SCENARIOS.get(scenario_name)
    if sc is None:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")

    # Override spots
    spots = dict(SPOT_PRICES)
    for k in ["oil_price", "gold_price", "spy_price", "btc_price"]:
        asset = k.replace("_price", "")
        if k in sc and asset in spots:
            spots[asset] = sc[k]

    # Override drifts
    drifts = dict(CRISIS_DRIFTS)
    for asset, drift in sc.get("drift_override", {}).items():
        if asset in drifts:
            drifts[asset] = drift

    return run_monte_carlo(n_paths=n_paths, spot_prices=spots, drifts=drifts)


# ============================================================================
# PART 8: TWICE-DAILY MANDATE GENERATOR
# ============================================================================

@dataclass
class DailyMandate:
    """Twice-daily action mandate."""
    timestamp: str
    session: str             # "morning" or "evening"
    phase: str
    composite_score: float
    regime: str
    # Actions per arm
    arm_actions: dict[str, str]
    # Key metrics
    portfolio_value_usd: float
    mc_summary: str
    # Triggered milestones
    new_milestones: list[str]
    # Risk alerts
    risk_alerts: list[str]
    # Checklist
    checklist: list[str]


def generate_mandate(
    indicators: Optional[IndicatorState] = None,
    run_mc: bool = True,
) -> DailyMandate:
    """Generate the twice-daily trading mandate."""
    now = datetime.now()
    session = "morning" if now.hour < 14 else "evening"
    ind = indicators or IndicatorState()

    # Composite score
    comp = compute_composite_score(ind)
    phase = get_current_phase()
    pf_val = get_portfolio_value_usd()

    # Check milestones
    state = {
        "portfolio_usd": pf_val,
        "oil_price": ind.oil_price,
        "gold_price": ind.gold_price,
        "vix": ind.vix,
        "hy_spread_bp": ind.hy_spread_bp,
        "bdc_nav_discount": ind.bdc_nav_discount,
        "bdc_nonaccrual_pct": ind.bdc_nonaccrual_pct,
        "btc_price": ind.btc_price,
        "spy_price": ind.spy_price,
        "eth_price": SPOT_PRICES["eth"],
        "xlf_price": SPOT_PRICES["xlf"],
        "qqq_price": SPOT_PRICES["qqq"],
    }
    new_ms = check_milestones(state)
    new_ms_names = [f"#{m.id} {m.name}" for m in new_ms]

    # Monte Carlo summary
    mc_summary = "MC not run this session"
    if run_mc:
        mc = run_monte_carlo(n_paths=10_000, horizon_days=90)  # quick run for mandate
        mc_summary = (f"Mean ${mc.portfolio_mean:,.0f} | "
                      f"P5 ${mc.portfolio_p5:,.0f} | P95 ${mc.portfolio_p95:,.0f} | "
                      f"VaR95 ${mc.var_95:,.0f}")

    # Arm actions based on regime
    regime = comp["regime"]
    allocations = get_arm_allocations(phase)
    arm_actions = {}
    for alloc in allocations:
        if regime == "CRISIS":
            if alloc.arm in (ArmType.IRAN_OIL, ArmType.BDC_NONACCRUAL):
                arm_actions[alloc.name] = f"FULL SIZE ({alloc.max_pct*100:.0f}%)"
            elif alloc.arm == ArmType.CRYPTO_METALS:
                arm_actions[alloc.name] = f"GOLD MAX, CRYPTO PUTS ({alloc.target_pct*100:.0f}%)"
            elif alloc.arm == ArmType.DEFI_YIELD:
                arm_actions[alloc.name] = f"DeFi shorts active ({alloc.target_pct*100:.0f}%)"
            else:
                arm_actions[alloc.name] = f"Minimal ({alloc.target_pct*100:.0f}%)"
        elif regime == "ELEVATED":
            arm_actions[alloc.name] = f"Target size ({alloc.target_pct*100:.0f}%)"
        elif regime == "WATCH":
            arm_actions[alloc.name] = f"Half size ({alloc.target_pct*50:.0f}%)"
        else:
            arm_actions[alloc.name] = f"Minimal or flat ({max(alloc.target_pct*25, 0)*100:.0f}%)"

    # Risk alerts
    alerts = []
    if comp["composite_score"] > 80:
        alerts.append("CRITICAL: Composite >80 -- max aggression but watch for reversal")
    if ind.vix > 35:
        alerts.append("VIX >35: IV expensive. Consider selling premium or reducing entries.")
    if ind.oil_price > 120:
        alerts.append("Oil >$120: Geopolitical premium extreme. Take partial profit on energy.")
    if ind.hy_spread_bp > 600:
        alerts.append("HY spread >600bp: Credit stress severe. BDC puts at max.")
    if ind.btc_price < 60000:
        alerts.append("BTC <$60K: Crypto winter. Max BITO puts. Watch stablecoins.")
    if pf_val < STARTING_CAPITAL_CAD * CAD_TO_USD * 0.75:
        alerts.append("DRAWDOWN >25%: Cut all positions 50%. Review thesis.")

    # Checklist
    checklist = []
    if session == "morning":
        checklist = [
            "[ ] Check overnight oil/gold/BTC prices",
            "[ ] Review pre-market SPY/QQQ futures",
            "[ ] Check VIX level and HY spread",
            "[ ] Review any geopolitical headlines",
            "[ ] Run Monte Carlo (full 100K if time permits)",
            "[ ] Check expiring options (roll if <5 DTE)",
            "[ ] Execute any new entries from mandate",
            "[ ] Log intel update",
        ]
    else:
        checklist = [
            "[ ] Review day's P&L across all accounts",
            "[ ] Check for milestone triggers",
            "[ ] Review any after-hours news",
            "[ ] Update indicator readings",
            "[ ] Plan tomorrow's entries",
            "[ ] Log closing intel update",
        ]

    return DailyMandate(
        timestamp=now.strftime("%Y-%m-%d %H:%M"),
        session=session,
        phase=phase,
        composite_score=comp["composite_score"],
        regime=regime,
        arm_actions=arm_actions,
        portfolio_value_usd=pf_val,
        mc_summary=mc_summary,
        new_milestones=new_ms_names,
        risk_alerts=alerts,
        checklist=checklist,
    )


# ============================================================================
# PART 9: STATE PERSISTENCE
# ============================================================================

STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "war_engine"


def _ensure_state_dir() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def save_mandate(mandate: DailyMandate) -> Path:
    """Save mandate to JSON log."""
    d = _ensure_state_dir()
    fname = f"mandate_{mandate.timestamp.replace(':', '').replace(' ', '_')}.json"
    path = d / fname
    path.write_text(json.dumps(asdict(mandate), indent=2, default=str), encoding="utf-8")
    return path


def save_mc_result(mc: MCResult, label: str = "") -> Path:
    """Save MC result to JSON."""
    d = _ensure_state_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"mc_{label}_{ts}.json" if label else f"mc_{ts}.json"
    path = d / fname
    path.write_text(json.dumps(asdict(mc), indent=2, default=str), encoding="utf-8")
    return path


def load_milestone_state() -> None:
    """Load triggered milestone state from disk."""
    path = _ensure_state_dir() / "milestones.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id = {m.id: m for m in MILESTONES}
        for item in data:
            m = by_id.get(item.get("id"))
            if m:
                m.triggered = item.get("triggered", False)
                m.triggered_date = item.get("triggered_date")


def save_milestone_state() -> Path:
    """Save milestone state to disk."""
    path = _ensure_state_dir() / "milestones.json"
    data = [{"id": m.id, "name": m.name, "triggered": m.triggered,
             "triggered_date": m.triggered_date} for m in MILESTONES]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ============================================================================
# PART 10: RENDERING ENGINE (ASCII-SAFE)
# ============================================================================

def _bar(value: float, max_val: float, width: int = 30) -> str:
    """ASCII bar chart."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    filled = max(0, min(filled, width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _divider(char: str = "=", width: int = 72) -> str:
    return char * width


def render_dashboard() -> str:
    """Render the main dashboard."""
    lines = []
    lines.append(_divider())
    lines.append("  AAC WAR ROOM ENGINE v2.0 -- FORWARD MONTE CARLO + SPIDERWEB")
    lines.append(_divider())
    lines.append(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Phase: {get_current_phase().upper()}")
    lines.append(f"  Portfolio: ${get_portfolio_value_usd():,.2f} USD")
    lines.append("")

    # Indicator summary
    ind = IndicatorState()
    comp = compute_composite_score(ind)
    lines.append(f"  Composite Score: {comp['composite_score']}/100  [{comp['regime']}]")
    lines.append(f"  {_bar(comp['composite_score'], 100, 40)} {comp['composite_score']:.0f}")
    lines.append("")

    # Top indicators
    lines.append("  TOP INDICATORS:")
    sorted_ind = sorted(comp["individual_scores"].items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_ind[:6]:
        lines.append(f"    {name:18s} {_bar(score, 100, 20)} {score:.0f}")
    lines.append("")

    # Milestone status
    triggered = [m for m in MILESTONES if m.triggered]
    pending = [m for m in MILESTONES if not m.triggered]
    lines.append(f"  MILESTONES: {len(triggered)}/50 triggered, {len(pending)} pending")
    if triggered:
        lines.append("  Recent triggers:")
        for m in triggered[-3:]:
            lines.append(f"    #{m.id:02d} {m.name} ({m.triggered_date})")
    lines.append("")

    # Position summary
    total_mv = sum(p.market_value for p in CURRENT_POSITIONS)
    total_pnl = sum(p.pnl for p in CURRENT_POSITIONS)
    lines.append(f"  POSITIONS: {len(CURRENT_POSITIONS)} open")
    lines.append(f"  Market Value: ${total_mv:,.2f}")
    lines.append(f"  Unrealized P&L: ${total_pnl:+,.2f}")
    lines.append("")

    # Phase thresholds
    pf = get_portfolio_value_usd()
    lines.append("  PHASE PROGRESS:")
    for phase, (lo, hi) in PHASE_THRESHOLDS.items():
        if lo <= pf < hi:
            pct = ((pf - lo) / (hi - lo)) * 100 if hi > lo else 0
            lines.append(f"  >> {phase.upper()}: ${lo:,.0f} -> ${hi:,.0f}")
            lines.append(f"     {_bar(pct, 100, 30)} {pct:.1f}%")
        else:
            lines.append(f"     {phase}: ${lo:,.0f} -> ${hi:,.0f}")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


def render_mc_result(mc: MCResult) -> str:
    """Render Monte Carlo results."""
    lines = []
    lines.append(_divider())
    lines.append(f"  MONTE CARLO SIMULATION -- {mc.n_paths:,} PATHS x {mc.horizon_days} DAYS")
    lines.append(_divider())
    lines.append(f"  Runtime: {mc.runtime_ms:.0f}ms")
    lines.append("")
    lines.append("  PORTFOLIO DISTRIBUTION:")
    lines.append(f"    Mean:    ${mc.portfolio_mean:>12,.2f}")
    lines.append(f"    Median:  ${mc.portfolio_median:>12,.2f}")
    lines.append(f"    P5:      ${mc.portfolio_p5:>12,.2f}  (bear case)")
    lines.append(f"    P95:     ${mc.portfolio_p95:>12,.2f}  (bull case)")
    lines.append(f"    VaR 95%: ${mc.var_95:>12,.2f}  (max loss)")
    lines.append(f"    CVaR 95%:${mc.cvar_95:>12,.2f}  (avg tail loss)")
    lines.append("")

    lines.append("  ASSET PRICE PROJECTIONS (90-day):")
    lines.append(f"  {'Asset':>8s} {'Current':>10s} {'Mean':>10s} {'P5':>10s} {'P95':>10s}")
    lines.append(f"  {'-'*8:>8s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")
    for asset in ASSETS:
        curr = SPOT_PRICES[asset]
        mean = mc.asset_means[asset]
        p5 = mc.asset_p5[asset]
        p95 = mc.asset_p95[asset]
        lines.append(f"  {asset:>8s} ${curr:>9,.1f} ${mean:>9,.1f} ${p5:>9,.1f} ${p95:>9,.1f}")
    lines.append("")

    lines.append("  SCENARIO PROBABILITIES (calibrated Mar 19):")
    lines.append(f"    Oil >$120:          {mc.prob_oil_above_120*100:>6.1f}%")
    lines.append(f"    Gold >$5500:        {mc.prob_gold_above_3500*100:>6.1f}%")
    lines.append(f"    SPY <$600:          {mc.prob_spy_below_500*100:>6.1f}%")
    lines.append(f"    BTC <$55K:          {mc.prob_btc_below_60k*100:>6.1f}%")
    lines.append(f"    Portfolio >$150K:   {mc.prob_portfolio_above_150k*100:>6.1f}%")
    lines.append(f"    Portfolio >$1M:     {mc.prob_portfolio_above_1m*100:>6.1f}%")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


def render_milestones() -> str:
    """Render milestone spiderweb status."""
    lines = []
    lines.append(_divider())
    lines.append("  50-MILESTONE SPIDERWEB SYSTEM")
    lines.append(_divider())

    triggered = [m for m in MILESTONES if m.triggered]
    pending = [m for m in MILESTONES if not m.triggered]
    lines.append(f"  Status: {len(triggered)} triggered / {len(pending)} pending / 50 total")
    lines.append("")

    # Group by category
    by_cat: dict[str, list[Milestone]] = {}
    for m in MILESTONES:
        cat = m.category.value
        by_cat.setdefault(cat, []).append(m)

    for cat, ms in sorted(by_cat.items()):
        t_count = sum(1 for m in ms if m.triggered)
        lines.append(f"  [{cat.upper()}] ({t_count}/{len(ms)} triggered)")
        for m in ms:
            status = "[X]" if m.triggered else "[ ]"
            conf = f"conf:{m.confidence*100:.0f}%"
            chain = f"-> {m.leads_to}" if m.leads_to else ""
            lines.append(f"    {status} #{m.id:02d} {m.name:30s} {conf:10s} {chain}")
            if m.triggered:
                lines.append(f"         Triggered: {m.triggered_date}")
                lines.append(f"         Action: {m.strategy_action[:60]}")
        lines.append("")

    # Show most impactful chains
    lines.append("  TOP CAUSAL CHAINS (longest spiderweb paths):")
    all_chains = []
    for m in MILESTONES:
        if not m.triggered:
            chains = get_spiderweb_chains(m.id)
            all_chains.extend(chains)
    all_chains.sort(key=len, reverse=True)
    for chain in all_chains[:5]:
        path = " -> ".join(f"#{m.id}" for m in chain)
        lines.append(f"    {path} (length {len(chain)})")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


def render_positions() -> str:
    """Render all positions."""
    lines = []
    lines.append(_divider())
    lines.append("  POSITION MANAGEMENT -- ALL ACCOUNTS")
    lines.append(_divider())
    lines.append("")

    # Positions table
    lines.append(f"  {'Symbol':8s} {'Type':6s} {'Qty':>4s} {'Entry':>8s} {'Current':>8s} "
                 f"{'P&L':>10s} {'P&L%':>7s} {'Expiry':10s} {'Arm':20s}")
    lines.append(f"  {'-'*8} {'-'*6} {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*10} {'-'*20}")
    total_pnl = 0.0
    for p in CURRENT_POSITIONS:
        pnl = p.pnl
        total_pnl += pnl
        lines.append(
            f"  {p.symbol:8s} {p.position_type:6s} {p.quantity:>4d} "
            f"${p.entry_price:>6.2f} ${p.current_price:>6.2f} "
            f"${pnl:>+9.2f} {p.pnl_pct:>+6.1f}% "
            f"{p.expiry or 'N/A':10s} {p.arm.value:20s}"
        )
    lines.append(f"  {'-'*97}")
    lines.append(f"  {'TOTAL':8s} {'':6s} {len(CURRENT_POSITIONS):>4d} "
                 f"{'':>8s} {'':>8s} ${total_pnl:>+9.2f}")
    lines.append("")

    # Account summary
    lines.append("  ACCOUNT BALANCES:")
    for name, acct in ACCOUNTS.items():
        if "balance_usd" in acct:
            lines.append(f"    {name:15s} ${acct['balance_usd']:>10,.2f} USD  ({acct['type']})")
        else:
            lines.append(f"    {name:15s} ${acct['balance_cad']:>10,.2f} CAD  ({acct['type']})")
    lines.append(f"    {'TOTAL (USD)':15s} ${get_portfolio_value_usd():>10,.2f} USD")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


def render_greeks() -> str:
    """Render Greeks for all option positions (puts and calls)."""
    lines = []
    lines.append(_divider())
    lines.append("  BLACK-SCHOLES GREEKS -- ALL OPTION POSITIONS")
    lines.append(_divider())
    lines.append("")
    lines.append(f"  {'Symbol':8s} {'Strike':>8s} {'Expiry':10s} {'Price':>8s} "
                 f"{'Delta':>8s} {'Gamma':>8s} {'Vega':>8s} {'Theta':>8s} {'Score':>6s}")
    lines.append(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for pos in CURRENT_POSITIONS:
        if pos.strike and pos.expiry:
            # Calculate time to expiry
            exp_date = datetime.strptime(pos.expiry, "%Y-%m-%d")
            dte = max((exp_date - datetime.now()).days, 1)
            T = dte / 365.0

            # Underlying spot prices (LIVE March 19 evening)
            spot_map = {
                "ARCC": 19.0, "PFF": 31.0, "LQD": 103.0, "EMB": 87.0,
                "MAIN": 46.0, "JNK": 85.0, "KRE": 45.0, "IWM": 198.0,
                "SPY": 665.0, "QQQ": 450.0, "XLF": 35.0, "XLRE": 22.0,
                "GLD": 486.0, "SLV": 78.0,
            }
            spot = spot_map.get(pos.symbol, pos.strike * 1.1)
            sigma = 0.35  # approximate IV for these instruments

            if "call" in pos.position_type:
                g = bs_call(spot, pos.strike, T, RISK_FREE_RATE, sigma)
            else:
                g = bs_put(spot, pos.strike, T, RISK_FREE_RATE, sigma)
            pos.greeks = g

            lines.append(
                f"  {pos.symbol:8s} ${pos.strike:>6.0f} {pos.expiry:10s} "
                f"${g.price:>6.2f} {g.delta:>+7.3f} {g.gamma:>8.4f} "
                f"{g.vega:>8.4f} {g.theta:>8.4f} {g.greek_score():>5.1f}"
            )
    lines.append("")
    lines.append("  GreekScore = 0.35*|Delta| + 0.25*Gamma + 0.25*Vega + 0.15*(1-|Theta|)")
    lines.append("  Score >70 = optimal entry, <50 = avoid")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


def render_arms() -> str:
    """Render 5-arm breakdown."""
    lines = []
    lines.append(_divider())
    lines.append("  5-ARM STRATEGY BREAKDOWN")
    lines.append(_divider())
    lines.append("")

    phase = get_current_phase()
    allocations = get_arm_allocations(phase)

    # Count positions per arm
    arm_positions: dict[str, list[Position]] = {}
    for p in CURRENT_POSITIONS:
        arm_positions.setdefault(p.arm.value, []).append(p)

    for alloc in allocations:
        positions = arm_positions.get(alloc.arm.value, [])
        pos_value = sum(p.market_value for p in positions)
        pos_pnl = sum(p.pnl for p in positions)

        lines.append(f"  [{alloc.arm.value.upper()}] {alloc.name}")
        lines.append(f"    Target: {alloc.target_pct*100:.0f}% | Max: {alloc.max_pct*100:.0f}% | "
                     f"Positions: {len(positions)} | Value: ${pos_value:,.2f} | P&L: ${pos_pnl:+,.2f}")
        lines.append(f"    Instruments: {', '.join(alloc.instruments[:4])}")
        lines.append(f"    Entry: {alloc.entry_conditions[:60]}")
        lines.append(f"    Exit:  {alloc.exit_conditions[:60]}")
        if positions:
            for p in positions:
                lines.append(f"      - {p.symbol} {p.position_type} x{p.quantity} "
                             f"@ ${p.entry_price:.2f} -> ${p.current_price:.2f} "
                             f"(${p.pnl:+.2f})")
        lines.append("")

    lines.append(_divider())
    return "\n".join(lines)


def render_indicators() -> str:
    """Render 12-indicator model."""
    lines = []
    lines.append(_divider())
    lines.append("  12-INDICATOR COMPOSITE MODEL")
    lines.append(_divider())
    lines.append("")

    ind = IndicatorState()
    comp = compute_composite_score(ind)

    lines.append(f"  COMPOSITE: {comp['composite_score']:.1f}/100  [{comp['regime']}]")
    lines.append(f"  {_bar(comp['composite_score'], 100, 50)}")
    lines.append("")

    lines.append(f"  {'Indicator':18s} {'Score':>6s} {'Bar':32s} {'Weight':>7s}")
    lines.append(f"  {'-'*18} {'-'*6} {'-'*32} {'-'*7}")

    weights = {
        "oil": 0.15, "gold": 0.10, "vix": 0.12, "hy_spread": 0.10,
        "bdc_nav": 0.08, "bdc_nonaccrual": 0.07, "defi_tvl": 0.06,
        "stablecoin": 0.05, "btc": 0.07, "fed_rate": 0.06,
        "dxy": 0.06, "spy": 0.08,
    }

    for name, score in sorted(comp["individual_scores"].items(),
                                key=lambda x: x[1], reverse=True):
        w = weights.get(name, 0.0)
        lines.append(f"  {name:18s} {score:>5.1f} {_bar(score, 100, 30)} {w*100:>5.1f}%")

    lines.append("")
    lines.append("  REGIME THRESHOLDS:")
    lines.append("    CALM:     0-30   | WATCH:  30-50  | ELEVATED: 50-70 | CRISIS: 70+")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


def render_mandate(mandate: DailyMandate) -> str:
    """Render the daily mandate."""
    lines = []
    lines.append(_divider())
    lines.append(f"  TWICE-DAILY MANDATE -- {mandate.session.upper()} SESSION")
    lines.append(_divider())
    lines.append(f"  Time: {mandate.timestamp}")
    lines.append(f"  Phase: {mandate.phase.upper()}")
    lines.append(f"  Composite: {mandate.composite_score:.1f}/100 [{mandate.regime}]")
    lines.append(f"  Portfolio: ${mandate.portfolio_value_usd:,.2f} USD")
    lines.append("")

    lines.append("  MONTE CARLO:")
    lines.append(f"    {mandate.mc_summary}")
    lines.append("")

    lines.append("  ARM ACTIONS:")
    for arm, action in mandate.arm_actions.items():
        lines.append(f"    {arm:30s} {action}")
    lines.append("")

    if mandate.new_milestones:
        lines.append("  *** NEW MILESTONES TRIGGERED ***")
        for ms in mandate.new_milestones:
            lines.append(f"    >> {ms}")
        lines.append("")

    if mandate.risk_alerts:
        lines.append("  RISK ALERTS:")
        for alert in mandate.risk_alerts:
            lines.append(f"    ! {alert}")
        lines.append("")

    lines.append("  CHECKLIST:")
    for item in mandate.checklist:
        lines.append(f"    {item}")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


def render_scenario(name: str, mc: MCResult) -> str:
    """Render a scenario analysis."""
    sc = SCENARIOS[name]
    lines = []
    lines.append(_divider())
    lines.append(f"  SCENARIO: {sc['name'].upper()}")
    lines.append(_divider())
    lines.append(f"  {sc['description']}")
    lines.append(f"  Probability: {sc['probability']*100:.0f}%")
    lines.append("")
    lines.append("  SCENARIO PARAMETERS:")
    for k, v in sc.items():
        if k not in ("name", "description", "probability", "drift_override"):
            lines.append(f"    {k:20s} {v}")
    lines.append("")
    lines.append(render_mc_result(mc))
    return "\n".join(lines)


def render_phase() -> str:
    """Render phase status and transition rules."""
    lines = []
    lines.append(_divider())
    lines.append("  PHASE TRANSITION ENGINE")
    lines.append(_divider())
    lines.append("")

    current = get_current_phase()
    pf = get_portfolio_value_usd()

    for phase, (lo, hi) in PHASE_THRESHOLDS.items():
        marker = " >> " if phase == current else "    "
        if lo <= pf < hi:
            pct = ((pf - lo) / (hi - lo)) * 100
            lines.append(f"{marker}{phase.upper():15s} ${lo:>12,.0f} -> ${hi:>12,.0f}  "
                         f"{_bar(pct, 100, 20)} {pct:.1f}%")
        elif pf >= hi:
            lines.append(f"{marker}{phase.upper():15s} ${lo:>12,.0f} -> ${hi:>12,.0f}  "
                         f"[COMPLETED]")
        else:
            lines.append(f"{marker}{phase.upper():15s} ${lo:>12,.0f} -> ${hi:>12,.0f}  "
                         f"[LOCKED]")

    lines.append("")
    lines.append(f"  Current: {current.upper()}")
    lines.append(f"  Portfolio: ${pf:,.2f} USD")
    lines.append("")

    # Phase rules
    rules = {
        "accumulation": [
            "Max 50% in options",
            "Put pyramid: add on every 10% gain",
            "5 arms active, Iran/Oil dominant",
            "Risk cap: 3% per trade, 25% max drawdown",
        ],
        "growth": [
            "Reduce options to 35%",
            "Gold/silver/miners become dominant",
            "Income strategies begin (TradFi arm grows)",
            "Take profits on 10x+ trades immediately",
        ],
        "rotation": [
            "Max 15% in options",
            "40% fixed income, 25% metals",
            "Dividend and interest income focus",
            "Only opportunistic put entries",
        ],
        "preservation": [
            "Max 5% in options",
            "60% yield instruments",
            "Physical gold allocation considered",
            "Family office structure planning",
        ],
    }

    lines.append(f"  RULES FOR {current.upper()}:")
    for rule in rules.get(current, []):
        lines.append(f"    - {rule}")
    lines.append("")
    lines.append(_divider())
    return "\n".join(lines)


# ============================================================================
# PART 11: CLI
# ============================================================================

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AAC War Room Engine v2.0 -- Forward Monte Carlo + Spiderweb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--monte-carlo", "-mc", action="store_true",
                        help="Run full 100K-path Monte Carlo simulation")
    parser.add_argument("--milestones", "-ms", action="store_true",
                        help="Show 50-milestone spiderweb status")
    parser.add_argument("--greeks", "-g", action="store_true",
                        help="Show Black-Scholes Greeks for all positions")
    parser.add_argument("--mandate", "-m", action="store_true",
                        help="Generate twice-daily trading mandate")
    parser.add_argument("--positions", "-p", action="store_true",
                        help="Show all positions across accounts")
    parser.add_argument("--arms", "-a", action="store_true",
                        help="Show 5-arm strategy breakdown")
    parser.add_argument("--indicators", "-i", action="store_true",
                        help="Show 12-indicator composite model")
    parser.add_argument("--scenario", "-s", type=str, default="",
                        help="Run named scenario (hormuz_closure, iran_deescalation, "
                             "credit_cascade, defi_collapse, soft_landing, black_swan, "
                             "gold_supercycle)")
    parser.add_argument("--phase", action="store_true",
                        help="Show phase transition status")
    parser.add_argument("--paths", type=int, default=MC_PATHS,
                        help="Number of MC paths (default: 100000)")
    parser.add_argument("--horizon", type=int, default=MC_HORIZON_DAYS,
                        help="MC horizon in days (default: 90)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible MC")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--save", action="store_true",
                        help="Save results to data/war_engine/")

    args = parser.parse_args()

    # Load persisted state
    load_milestone_state()

    if args.monte_carlo:
        print("Running Monte Carlo simulation...")
        mc = run_monte_carlo(
            n_paths=args.paths,
            horizon_days=args.horizon,
            seed=args.seed,
        )
        if args.json:
            print(json.dumps(asdict(mc), indent=2, default=str))
        else:
            print(render_mc_result(mc))
        if args.save:
            path = save_mc_result(mc, "full")
            print(f"Saved to {path}")

    elif args.milestones:
        # Run milestone checks with current indicator state
        ind = IndicatorState()
        pf_val = get_portfolio_value_usd()
        state = {
            "portfolio_usd": pf_val,
            "oil_price": ind.oil_price,
            "gold_price": ind.gold_price,
            "vix": ind.vix,
            "hy_spread_bp": ind.hy_spread_bp,
            "bdc_nav_discount": ind.bdc_nav_discount,
            "bdc_nonaccrual_pct": ind.bdc_nonaccrual_pct,
            "btc_price": ind.btc_price,
            "spy_price": ind.spy_price,
            "eth_price": SPOT_PRICES["eth"],
            "xlf_price": SPOT_PRICES["xlf"],
            "qqq_price": SPOT_PRICES["qqq"],
        }
        new_ms = check_milestones(state)
        if new_ms:
            print(f"*** {len(new_ms)} NEW MILESTONES TRIGGERED ***")
            for m in new_ms:
                print(f"  #{m.id:02d} {m.name} -> ACTION: {m.strategy_action[:72]}")
            print("")
        if args.json:
            data = [{"id": m.id, "name": m.name, "category": m.category.value,
                     "triggered": m.triggered, "triggered_date": m.triggered_date,
                     "confidence": m.confidence, "leads_to": m.leads_to,
                     "action": m.strategy_action}
                    for m in MILESTONES]
            print(json.dumps(data, indent=2))
        else:
            print(render_milestones())
        if args.save:
            path = save_milestone_state()
            print(f"Milestone state saved to {path}")

    elif args.greeks:
        if args.json:
            results = []
            for pos in CURRENT_POSITIONS:
                if pos.strike and pos.expiry:
                    exp_date = datetime.strptime(pos.expiry, "%Y-%m-%d")
                    dte = max((exp_date - datetime.now()).days, 1)
                    T = dte / 365.0
                    spot_map = {
                        "ARCC": 19.0, "PFF": 31.0, "LQD": 103.0, "EMB": 87.0,
                        "MAIN": 46.0, "JNK": 85.0, "KRE": 45.0, "IWM": 198.0,
                        "SPY": 665.0, "QQQ": 450.0, "XLF": 35.0, "XLRE": 22.0,
                        "GLD": 486.0, "SLV": 78.0,
                    }
                    spot = spot_map.get(pos.symbol, pos.strike * 1.1)
                    if "call" in pos.position_type:
                        g = bs_call(spot, pos.strike, T, RISK_FREE_RATE, 0.35)
                    else:
                        g = bs_put(spot, pos.strike, T, RISK_FREE_RATE, 0.35)
                    results.append({"symbol": pos.symbol, **asdict(g)})
            print(json.dumps(results, indent=2))
        else:
            print(render_greeks())

    elif args.mandate:
        mandate = generate_mandate(run_mc=True)
        if args.json:
            print(json.dumps(asdict(mandate), indent=2, default=str))
        else:
            print(render_mandate(mandate))
        if args.save:
            path = save_mandate(mandate)
            print(f"Saved to {path}")
        save_milestone_state()

    elif args.positions:
        if args.json:
            data = [{"symbol": p.symbol, "type": p.position_type,
                     "qty": p.quantity, "entry": p.entry_price,
                     "current": p.current_price, "pnl": p.pnl,
                     "pnl_pct": p.pnl_pct, "arm": p.arm.value,
                     "expiry": p.expiry, "account": p.account}
                    for p in CURRENT_POSITIONS]
            print(json.dumps(data, indent=2))
        else:
            print(render_positions())

    elif args.arms:
        if args.json:
            phase = get_current_phase()
            allocs = get_arm_allocations(phase)
            data = [{"arm": a.arm.value, "name": a.name,
                     "target_pct": a.target_pct, "max_pct": a.max_pct,
                     "instruments": a.instruments}
                    for a in allocs]
            print(json.dumps(data, indent=2))
        else:
            print(render_arms())

    elif args.indicators:
        if args.json:
            ind = IndicatorState()
            comp = compute_composite_score(ind)
            print(json.dumps(comp, indent=2))
        else:
            print(render_indicators())

    elif args.scenario:
        name = args.scenario.lower().replace("-", "_")
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        print(f"Running scenario: {SCENARIOS[name]['name']}...")
        mc = run_scenario_mc(name, n_paths=min(args.paths, 50_000))
        if args.json:
            print(json.dumps({"scenario": SCENARIOS[name], "mc": asdict(mc)},
                             indent=2, default=str))
        else:
            print(render_scenario(name, mc))
        if args.save:
            path = save_mc_result(mc, name)
            print(f"Saved to {path}")

    elif args.phase:
        if args.json:
            data = {"phase": get_current_phase(),
                    "portfolio_usd": get_portfolio_value_usd(),
                    "thresholds": {k: list(v) for k, v in PHASE_THRESHOLDS.items()}}
            print(json.dumps(data, indent=2))
        else:
            print(render_phase())

    else:
        # Default: dashboard
        print(render_dashboard())


if __name__ == "__main__":
    main()


# ============================================================================
# PART 12: Integration wrapper for unified_component_integrator
# ============================================================================

class WarRoomEngine:
    """Thin wrapper exposing War Room Engine to the orchestrator/integrator."""

    def __init__(self):
        self.indicators = IndicatorState()

    def get_mandate(self) -> DailyMandate:
        return generate_mandate(indicators=self.indicators)

    def get_phase(self) -> str:
        return get_current_phase()

    def get_portfolio_value(self) -> float:
        return get_portfolio_value_usd()

    def render(self) -> str:
        return render_dashboard()

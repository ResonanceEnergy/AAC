"""TurboQuant Integration Layer — All 12 AAC Data Sources.

Wires every AAC data pipeline into TurboQuant similarity indices
for pattern matching, anomaly detection, and precedent retrieval.

Indices:
    1. OptionsScanIndex      — Matrix Maximizer cycle outputs
    2. MonteCarloSummaryIndex — War Room MC simulation fingerprints
    3. CorrelationIndex      — 11x11 asset correlation regime tracking
    4. MarketStateIndex      — (delegates to turboquant_engine)
    5. ScenarioIndex         — 43 black-swan scenario probability vectors
    6. GreeksIndex           — Portfolio Greeks aggregation snapshots
    7. MLFeatureIndex        — ML training feature vectors
    8. PricePatternIndex     — Multi-asset return pattern vectors
    9. SentimentIndex        — Reddit sentiment vectors
   10. PolymarketIndex       — Thesis-market edge vectors
   11. StrategyIndex         — Options strategy Greeks vectors
   12. PortfolioIndex        — Cross-platform balance snapshots

Usage:
    from strategies.turboquant_integrations import IntegrationHub
    hub = IntegrationHub()
    hub.record_options_scan(recommendations)
    hub.record_monte_carlo(mc_result)
    similar = hub.search_scenario(current_scenario_vec, top_k=5)
    hub.save_all()
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from strategies.turboquant_engine import (
    MonteCarloCompressor,
    TurboQuantIndex,
    TurboQuantizer,
    encode_market_state,
    get_market_index,
    record_market_state,
    save_market_index,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "turboquant"

# Matrix Maximizer assets (core.py order)
MM_ASSETS = ["SPY", "QQQ", "USO", "BITO", "TLT", "JETS", "KRE", "HYG", "XLY", "ZIM", "XLE"]

# War Room assets
WR_ASSETS = ["oil", "gold", "silver", "gdx", "spy", "qqq", "xlf", "xlre", "eth", "xrp", "btc"]

# 43 scenario codes (scenario_engine.py)
SCENARIO_CODES = [
    # Core 15
    "HORMUZ", "DEBT_CRISIS", "EU_BANKS", "SUPERCYCLE", "CRE_COLLAPSE",
    "THAILAND_SHOCK", "PAKISTAN_COLLAPSE", "DEFI_CASCADE", "AI_BUBBLE",
    "EM_FX_CRISIS", "FOOD_CRISIS", "CLIMATE_SHOCK", "JAPAN_CRISIS",
    "ELECTION_CHAOS", "PANDEMIC_V2",
    # Geopolitical 5
    "US_WITHDRAWAL", "IRAN_DEAL", "IRAN_NUCLEAR", "ELITE_EXPOSURE",
    "PETRODOLLAR_SPIRAL",
    # Western Hemisphere 23
    "HEMISPHERE_PIVOT", "NATO_EXIT", "EUROPE_ABANDON", "CANADA_DECLINE",
    "GREENLAND_ACQ", "PANAMA_RECLAIM", "LATAM_LOCKIN", "ARCTIC_EXPAND",
    "BORDER_MILITARY", "VENEZUELA_REGIME", "LITHIUM_TRIANGLE", "CUBA_EMBARGO",
    "BRAZIL_ARGENTINA", "NORTH_BORDER", "MIDEAST_REDEPLOY", "ENERGY_HEMISPHERE",
    "STARLINK_DOMINANCE", "FUSION_ROLLOUT", "RARE_EARTH_FORTRESS",
    "MIGRATION_SECURITY", "FORTRESS_2100", "ELITE_CAPITAL", "NUCLEAR_AMERICAS",
]

# Polymarket thesis stages
THESIS_STAGES = [
    "iran_escalation", "us_withdrawal", "gulf_shift",
    "gold_reprice", "usd_collapse", "crypto_contagion",
]


# ============================================================================
# ENCODER FUNCTIONS — Convert each data source to vectors
# ============================================================================

def encode_options_scan(recommendations: list[dict]) -> np.ndarray:
    """Encode a Matrix Maximizer scan cycle into a 64-dim vector.

    Aggregates top recommendations into a fixed-size fingerprint:
    - Per-ticker stats (top 11 tickers × 4 fields = 44)
    - Aggregate stats (20 fields)

    Args:
        recommendations: list of PutRecommendation dicts from scanner

    Returns:
        64-dim float32 vector
    """
    vec = np.zeros(64, dtype=np.float32)
    if not recommendations:
        return vec

    # Per-ticker slots (11 tickers × 4 fields = 44)
    ticker_map = {t: i for i, t in enumerate(MM_ASSETS)}
    for rec in recommendations:
        ticker = rec.get("ticker", "")
        idx = ticker_map.get(ticker)
        if idx is None:
            continue
        base = idx * 4
        vec[base] = rec.get("composite_score", 0.0) / 100.0
        vec[base + 1] = rec.get("delta_score", 0.0) / 100.0
        greeks = rec.get("greeks", {})
        vec[base + 2] = abs(greeks.get("delta", 0.0))
        vec[base + 3] = greeks.get("iv", greeks.get("sigma", 0.0))

    # Aggregate stats (indices 44-63)
    n = len(recommendations)
    scores = [r.get("composite_score", 0.0) for r in recommendations]
    ivs = [r.get("greeks", {}).get("sigma", r.get("greeks", {}).get("iv", 0.0))
           for r in recommendations]
    dtes = [r.get("contract", {}).get("dte", 0) for r in recommendations]
    costs = [r.get("total_cost", 0.0) for r in recommendations]

    vec[44] = n / 50.0  # normalized count
    vec[45] = np.mean(scores) / 100.0 if scores else 0.0
    vec[46] = np.std(scores) / 100.0 if len(scores) > 1 else 0.0
    vec[47] = np.max(scores) / 100.0 if scores else 0.0
    vec[48] = np.min(scores) / 100.0 if scores else 0.0
    vec[49] = np.mean(ivs) if ivs else 0.0
    vec[50] = np.std(ivs) if len(ivs) > 1 else 0.0
    vec[51] = np.max(ivs) if ivs else 0.0
    vec[52] = np.mean(dtes) / 90.0 if dtes else 0.0
    vec[53] = np.std(dtes) / 90.0 if len(dtes) > 1 else 0.0
    vec[54] = np.sum(costs) / 10000.0  # total capital deployed
    vec[55] = np.mean(costs) / 1000.0 if costs else 0.0

    # Count by mandate level
    mandates = [r.get("mandate", "") for r in recommendations]
    vec[56] = mandates.count("defensive") / max(n, 1)
    vec[57] = mandates.count("standard") / max(n, 1)
    vec[58] = mandates.count("aggressive") / max(n, 1)
    vec[59] = mandates.count("max_conviction") / max(n, 1)

    # Risk metrics
    risks = [r.get("risk_pct", 0.0) for r in recommendations]
    vec[60] = np.mean(risks) if risks else 0.0
    vec[61] = np.max(risks) if risks else 0.0
    vec[62] = np.sum(risks)  # total portfolio risk %
    vec[63] = np.median(scores) / 100.0 if scores else 0.0

    return vec


def encode_mc_summary(mc_result: dict) -> np.ndarray:
    """Encode War Room MC result into a 32-dim summary fingerprint.

    Captures the statistical signature of the simulation without storing
    all 100K paths. For path-level compression use MonteCarloCompressor.

    Args:
        mc_result: MCResult dict with asset_means, VaR, etc.

    Returns:
        32-dim float32 vector
    """
    vec = np.zeros(32, dtype=np.float32)

    # Per-asset mean returns (11)
    for i, asset in enumerate(WR_ASSETS):
        mean_key = f"{asset}_mean" if isinstance(mc_result.get("asset_means"), dict) else None
        if isinstance(mc_result.get("asset_means"), dict):
            vec[i] = mc_result["asset_means"].get(asset, 0.0) / 100.0
        elif hasattr(mc_result, "asset_means") and isinstance(mc_result.asset_means, dict):
            vec[i] = mc_result.asset_means.get(asset, 0.0) / 100.0

    # VaR spread per asset — p95-p5 normalized (11)
    for i, asset in enumerate(WR_ASSETS):
        p95 = 0.0
        p5 = 0.0
        for attr in ["asset_p95", "asset_p5"]:
            d = mc_result.get(attr, {}) if isinstance(mc_result, dict) else getattr(mc_result, attr, {})
            if isinstance(d, dict):
                if attr.endswith("p95"):
                    p95 = d.get(asset, 0.0)
                else:
                    p5 = d.get(asset, 0.0)
        spot = {"oil": 95, "gold": 4861, "silver": 78, "gdx": 95,
                "spy": 665, "qqq": 450, "xlf": 35, "xlre": 22,
                "eth": 3800, "xrp": 2.5, "btc": 68000}.get(asset, 100)
        vec[11 + i] = (p95 - p5) / max(spot, 1) if spot else 0.0

    # Portfolio-level stats (10)
    def _get(key: str, default: float = 0.0) -> float:
        if isinstance(mc_result, dict):
            return mc_result.get(key, default)
        return getattr(mc_result, key, default)

    capital = 32486.4  # CAD → USD
    vec[22] = _get("pf_mean", 0.0) / capital
    vec[23] = _get("pf_median", 0.0) / capital
    vec[24] = _get("var_95", 0.0) / capital
    vec[25] = _get("cvar_95", 0.0) / capital
    vec[26] = _get("prob_oil_120", 0.0)
    vec[27] = _get("prob_gold_5500", 0.0)
    vec[28] = _get("prob_spy_600", 0.0)
    vec[29] = _get("prob_btc_55k", 0.0)
    vec[30] = _get("prob_pf_150k", 0.0)
    vec[31] = _get("prob_pf_1m", 0.0)

    return vec


def encode_correlation_matrix(corr_matrix: np.ndarray,
                              vols: Optional[dict[str, float]] = None) -> np.ndarray:
    """Encode 11x11 correlation matrix + vols into a 66-dim vector.

    Upper triangle (55) + asset vols (11) = 66 dims.

    Args:
        corr_matrix: 11x11 symmetric matrix (assets in WR_ASSETS order)
        vols: optional dict of implied/realized vols per asset

    Returns:
        66-dim float32 vector
    """
    vec = np.zeros(66, dtype=np.float32)
    corr = np.asarray(corr_matrix, dtype=np.float32)

    # Upper triangle (excluding diagonal) = 11*10/2 = 55
    idx = 0
    for i in range(11):
        for j in range(i + 1, 11):
            vec[idx] = corr[i, j]
            idx += 1

    # Asset vols (11)
    if vols:
        for i, asset in enumerate(WR_ASSETS):
            vec[55 + i] = vols.get(asset, 0.0)

    return vec


def encode_scenario_state(scenarios: list[dict]) -> np.ndarray:
    """Encode 43 scenario probabilities into a 43-dim vector.

    Args:
        scenarios: list of ScenarioState dicts with 'code' and 'probability'

    Returns:
        43-dim float32 vector (probability per scenario)
    """
    vec = np.zeros(43, dtype=np.float32)
    code_map = {c: i for i, c in enumerate(SCENARIO_CODES)}

    for s in scenarios:
        code = s.get("code", "")
        idx = code_map.get(code)
        if idx is not None:
            vec[idx] = s.get("probability", 0.0)

    return vec


def encode_portfolio_greeks(snapshot: dict) -> np.ndarray:
    """Encode PortfolioRiskSnapshot into a 16-dim vector.

    Args:
        snapshot: PortfolioRiskSnapshot dict or dataclass.__dict__

    Returns:
        16-dim float32 vector
    """
    vec = np.zeros(16, dtype=np.float32)

    def _g(k: str) -> float:
        if isinstance(snapshot, dict):
            return float(snapshot.get(k, 0.0))
        return float(getattr(snapshot, k, 0.0))

    # Raw Greeks (normalized)
    vec[0] = _g("total_delta") / 100.0
    vec[1] = _g("total_gamma") / 50.0
    vec[2] = _g("total_theta") / 500.0
    vec[3] = _g("total_vega") / 200.0
    vec[4] = _g("total_vanna") / 100.0
    vec[5] = _g("total_charm") / 100.0

    # Dollar measures
    vec[6] = _g("dollar_delta") / 50000.0
    vec[7] = _g("gamma_risk_1pct") / 10000.0
    vec[8] = _g("daily_theta") / 500.0
    vec[9] = _g("vega_risk_1pct") / 10000.0

    # Limits ratios (proximity to limits)
    dl = _g("delta_limit") or 100.0
    gl = _g("gamma_limit") or 50.0
    tl = abs(_g("theta_limit")) or 500.0
    vl = _g("vega_limit") or 200.0
    vec[10] = abs(_g("total_delta")) / dl
    vec[11] = abs(_g("total_gamma")) / gl
    vec[12] = abs(_g("total_theta")) / tl
    vec[13] = abs(_g("total_vega")) / vl

    # Risk level encoding
    risk_map = {"LOW": 0.0, "MODERATE": 0.25, "ELEVATED": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}
    rl = snapshot.get("risk_level", "LOW") if isinstance(snapshot, dict) else getattr(snapshot, "risk_level", "LOW")
    vec[14] = risk_map.get(str(rl), 0.0)
    vec[15] = float(snapshot.get("num_positions", 0) if isinstance(snapshot, dict)
                    else getattr(snapshot, "num_positions", 0)) / 50.0

    return vec


def encode_ml_features(feature_dict: dict) -> np.ndarray:
    """Encode ML training features into a 32-dim vector.

    Takes the feature importance dict or raw feature values and produces
    a fixed-size fingerprint for similarity matching.

    Args:
        feature_dict: dict of feature_name → value (raw features)
                      OR ModelPerformance.feature_importance dict

    Returns:
        32-dim float32 vector
    """
    # Standard feature order (covers all 4 categories from pipeline)
    FEATURE_KEYS = [
        # Volatility (8)
        "realized_vol", "parkinson_vol", "volume_sma", "volume_ratio",
        "volume_surge", "fear_ratio_10", "autocorr_5", "corr_change",
        # Momentum (8)
        "momentum_5", "momentum_20", "consec_up", "rolling_corr_20",
        "rolling_corr_60", "spy_return_1d", "qqq_return_1d", "kre_return_1d",
        # Macro (8)
        "vix", "hy_spread_bps", "ig_spread_bps", "yield_curve_10_2",
        "oil_price", "gold_price", "dollar_index", "breakeven_inflation",
        # Sentiment/Risk (8)
        "fear_greed", "safe_haven_bid", "breadth_adv_dec", "new_highs_52w",
        "new_lows_52w", "private_credit_redemption_pct", "core_pce", "gdp_growth",
    ]

    vec = np.zeros(32, dtype=np.float32)
    for i, key in enumerate(FEATURE_KEYS):
        val = feature_dict.get(key, 0.0)
        if val is None:
            val = 0.0
        vec[i] = float(val)

    # Z-score normalize each field based on rough scale factors
    scales = np.array([
        0.3, 0.3, 1e6, 3.0, 3.0, 0.5, 0.3, 0.3,
        10.0, 20.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0,
        40.0, 500.0, 200.0, 2.0, 100.0, 5000.0, 110.0, 3.0,
        50.0, 1.0, 1.0, 200.0, 200.0, 5.0, 5.0, 3.0,
    ], dtype=np.float32)
    vec = vec / np.maximum(scales, 1e-8)

    return vec


def encode_price_pattern(returns_dict: dict[str, list[float]],
                         window: int = 20) -> np.ndarray:
    """Encode multi-asset return patterns into a 64-dim vector.

    Takes the last `window` returns for each of 11 war room assets and
    produces statistical summaries: mean, std, skew, min, max, autocorr.

    Args:
        returns_dict: asset_name → list of recent pct returns
        window: lookback period (default 20 trading days)

    Returns:
        64-dim float32 vector
    """
    vec = np.zeros(64, dtype=np.float32)

    for i, asset in enumerate(WR_ASSETS):
        rets = returns_dict.get(asset, [])
        if not rets:
            continue
        r = np.array(rets[-window:], dtype=np.float32)
        base = i * 5
        if base + 4 >= 64:
            break
        vec[base] = np.mean(r)
        vec[base + 1] = np.std(r) if len(r) > 1 else 0.0
        vec[base + 2] = float(np.min(r))
        vec[base + 3] = float(np.max(r))
        # Autocorrelation lag-1
        if len(r) > 2:
            r_demeaned = r - np.mean(r)
            denom = np.sum(r_demeaned ** 2)
            if denom > 1e-10:
                vec[base + 4] = float(np.sum(r_demeaned[:-1] * r_demeaned[1:]) / denom)

    # Cross-asset stats (indices 55-63)
    all_means = vec[::5][:11]  # every 5th starting at 0
    all_stds = vec[1::5][:11]
    nonzero_means = all_means[all_means != 0]
    if len(nonzero_means) > 0:
        vec[55] = np.mean(nonzero_means)
        vec[56] = np.std(nonzero_means) if len(nonzero_means) > 1 else 0.0
        vec[57] = np.min(nonzero_means)
        vec[58] = np.max(nonzero_means)
    nonzero_stds = all_stds[all_stds != 0]
    if len(nonzero_stds) > 0:
        vec[59] = np.mean(nonzero_stds)
        vec[60] = np.max(nonzero_stds)

    # Dispersion: max-min of means (breadth indicator)
    if len(nonzero_means) > 1:
        vec[61] = float(np.max(nonzero_means) - np.min(nonzero_means))

    # Count of negative-mean assets
    vec[62] = float(np.sum(nonzero_means < 0)) / 11.0
    # Count of high-vol assets (std > 2%)
    vec[63] = float(np.sum(nonzero_stds > 0.02)) / 11.0

    return vec


def encode_sentiment(sentiment_list: list[dict]) -> np.ndarray:
    """Encode Reddit sentiment data into a 32-dim vector.

    Takes top sentiment signals and produces a fixed fingerprint.

    Args:
        sentiment_list: list of dicts with symbol, sentiment_score, comments

    Returns:
        32-dim float32 vector
    """
    vec = np.zeros(32, dtype=np.float32)
    if not sentiment_list:
        return vec

    # Sort by abs sentiment score, take top 10
    sorted_sents = sorted(sentiment_list,
                          key=lambda x: abs(x.get("sentiment_score", 0.0)),
                          reverse=True)[:10]

    # Per-stock fields (10 stocks × 2 fields = 20)
    for i, s in enumerate(sorted_sents):
        vec[i * 2] = s.get("sentiment_score", 0.0)
        vec[i * 2 + 1] = min(s.get("comments", s.get("no_of_comments", 0)) / 1000.0, 1.0)

    # Aggregate stats (indices 20-31)
    all_scores = [s.get("sentiment_score", 0.0) for s in sentiment_list]
    all_comments = [s.get("comments", s.get("no_of_comments", 0)) for s in sentiment_list]

    vec[20] = len(sentiment_list) / 50.0  # normalized count
    vec[21] = np.mean(all_scores) if all_scores else 0.0
    vec[22] = np.std(all_scores) if len(all_scores) > 1 else 0.0
    vec[23] = np.max(all_scores) if all_scores else 0.0
    vec[24] = np.min(all_scores) if all_scores else 0.0

    bullish = sum(1 for s in all_scores if s > 0.15)
    bearish = sum(1 for s in all_scores if s < -0.15)
    n = max(len(all_scores), 1)
    vec[25] = bullish / n
    vec[26] = bearish / n
    vec[27] = (bullish - bearish) / n  # net sentiment ratio

    vec[28] = np.mean(all_comments) / 1000.0 if all_comments else 0.0
    vec[29] = np.max(all_comments) / 5000.0 if all_comments else 0.0
    vec[30] = np.sum(all_comments) / 50000.0  # total discussion volume
    vec[31] = np.median(all_scores) if all_scores else 0.0

    return vec


def encode_polymarket(thesis_matches: list[dict]) -> np.ndarray:
    """Encode Polymarket thesis-market matches into a 32-dim vector.

    Args:
        thesis_matches: list of ThesisMarketMatch or BlackSwanOpportunity dicts

    Returns:
        32-dim float32 vector
    """
    vec = np.zeros(32, dtype=np.float32)
    if not thesis_matches:
        return vec

    # Per-thesis-stage aggregation (6 stages × 4 fields = 24)
    stage_map = {s: i for i, s in enumerate(THESIS_STAGES)}
    for m in thesis_matches:
        stage = m.get("stage", "").lower().replace(" ", "_")
        # Try to match stage keywords
        stage_idx = stage_map.get(stage)
        if stage_idx is None:
            # Fuzzy match by keywords
            for key, idx in stage_map.items():
                if key.split("_")[0] in stage:
                    stage_idx = idx
                    break
        if stage_idx is None:
            continue
        base = stage_idx * 4
        vec[base] = max(vec[base], m.get("edge", 0.0))
        vec[base + 1] = max(vec[base + 1], m.get("thesis_probability", 0.0))
        vec[base + 2] = m.get("market_price", 0.0)
        vec[base + 3] = min(m.get("volume_24h", 0.0) / 100000.0, 1.0)

    # Aggregate (indices 24-31)
    edges = [m.get("edge", 0.0) for m in thesis_matches]
    prices = [m.get("market_price", 0.0) for m in thesis_matches]

    vec[24] = len(thesis_matches) / 50.0
    vec[25] = np.mean(edges) if edges else 0.0
    vec[26] = np.max(edges) if edges else 0.0
    vec[27] = np.mean(prices) if prices else 0.0

    # Tier counts
    tiers = [m.get("tier", "") for m in thesis_matches]
    n = max(len(thesis_matches), 1)
    vec[28] = tiers.count("deep_value") / n
    vec[29] = tiers.count("value") / n
    vec[30] = tiers.count("momentum") / n

    # Average kelly fraction
    kellys = [m.get("kelly_fraction", 0.0) for m in thesis_matches if m.get("kelly_fraction")]
    vec[31] = np.mean(kellys) if kellys else 0.0

    return vec


def encode_strategy(strategy: dict) -> np.ndarray:
    """Encode an options strategy into a 32-dim vector.

    Handles single puts, spreads, or multi-leg structures.

    Args:
        strategy: dict with Greeks, cost, strikes, DTE, etc.

    Returns:
        32-dim float32 vector
    """
    vec = np.zeros(32, dtype=np.float32)

    # Strategy type encoding (one-hot-ish)
    stype = str(strategy.get("strategy", strategy.get("type", ""))).lower()
    vec[0] = 1.0 if "put" in stype else 0.0
    vec[1] = 1.0 if "call" in stype else 0.0
    vec[2] = 1.0 if "spread" in stype else 0.0
    vec[3] = 1.0 if "straddle" in stype or "strangle" in stype else 0.0

    # Core strategy metrics
    vec[4] = strategy.get("net_debit", strategy.get("total_cost", 0.0)) / 1000.0
    vec[5] = strategy.get("max_profit", 0.0) / 5000.0
    vec[6] = strategy.get("max_loss", strategy.get("total_cost", 0.0)) / 1000.0
    vec[7] = strategy.get("reward_risk_ratio", 0.0) / 10.0
    vec[8] = strategy.get("breakeven", 0.0) / 1000.0

    # Greeks
    for i, g in enumerate(["delta", "gamma", "vega", "theta", "rho"]):
        # Check net_ prefix first (spreads), then direct
        val = strategy.get(f"net_{g}", strategy.get(g, 0.0))
        if val is None:
            val = 0.0
        vec[9 + i] = float(val)

    # Moneyness
    vec[14] = strategy.get("otm_pct", 0.0)
    vec[15] = strategy.get("iv", strategy.get("sigma", 0.0))

    # DTE
    vec[16] = strategy.get("dte", 0) / 365.0

    # Strikes (normalized to spot)
    spot = strategy.get("spot", strategy.get("underlying_price", 100.0)) or 100.0
    vec[17] = strategy.get("strike", strategy.get("K_long", 0.0)) / spot
    vec[18] = strategy.get("strike_short", strategy.get("K_short", 0.0)) / spot

    # Scoring
    vec[19] = strategy.get("composite_score", 0.0) / 100.0
    vec[20] = strategy.get("delta_score", 0.0) / 100.0
    vec[21] = strategy.get("liquidity_score", 0.0) / 100.0
    vec[22] = strategy.get("edge_score", 0.0) / 100.0

    # Volume/OI
    vec[23] = min(strategy.get("volume", 0) / 10000.0, 1.0)
    vec[24] = min(strategy.get("open_interest", 0) / 50000.0, 1.0)

    # Second-order Greeks
    vec[25] = strategy.get("vanna", 0.0)
    vec[26] = strategy.get("charm", 0.0)
    vec[27] = strategy.get("vomma", 0.0)

    # Mandate encoding
    mandate_map = {"defensive": 0.25, "standard": 0.5, "aggressive": 0.75, "max_conviction": 1.0}
    vec[28] = mandate_map.get(str(strategy.get("mandate", "")).lower(), 0.0)

    # Contracts and risk
    vec[29] = strategy.get("contracts", 1) / 10.0
    vec[30] = strategy.get("risk_pct", 0.0)
    vec[31] = strategy.get("extrinsic", 0.0) / 100.0

    return vec


def encode_portfolio_balance(snapshot: dict) -> np.ndarray:
    """Encode cross-platform balance snapshot into 16-dim vector.

    Args:
        snapshot: balance_snapshot.json dict

    Returns:
        16-dim float32 vector
    """
    vec = np.zeros(16, dtype=np.float32)

    # Per-platform net liquidation (6 platforms)
    platforms = ["ibkr", "moomoo", "polymarket", "metamask", "ndax", "wealthsimple"]
    for i, platform in enumerate(platforms):
        pdata = snapshot.get(platform, {})
        if isinstance(pdata, dict):
            vec[i] = pdata.get("net_liquidation", 0.0) / 50000.0

    # Summary metrics
    summary = snapshot.get("_summary", {})
    vec[6] = summary.get("total_usd_approx", 0.0) / 100000.0
    vec[7] = summary.get("total_cad_component", 0.0) / 100000.0
    vec[8] = summary.get("cad_usd_rate", 0.72)

    # Position count per platform
    for i, platform in enumerate(platforms[:5]):
        pdata = snapshot.get(platform, {})
        if isinstance(pdata, dict):
            positions = pdata.get("positions", [])
            vec[9 + i] = len(positions) / 20.0

    # Status encoding (14-15)
    active_count = 0
    for p in platforms:
        pdata = snapshot.get(p, {})
        if isinstance(pdata, dict) and pdata.get("status") == "ok":
            active_count += 1
    vec[14] = active_count / len(platforms)

    # Total position count
    total_positions = 0
    for p in platforms:
        pdata = snapshot.get(p, {})
        if isinstance(pdata, dict):
            total_positions += len(pdata.get("positions", []))
    vec[15] = total_positions / 50.0

    return vec


# ============================================================================
# INTEGRATION HUB — Manages all 12 TurboQuant indices
# ============================================================================

@dataclass
class IndexConfig:
    """Configuration for a single TurboQuant index."""
    name: str
    dimension: int
    bit_width: int = 3
    mode: str = "prod"
    filename: str = ""

    def __post_init__(self):
        if not self.filename:
            self.filename = f"{self.name}_index.json"


# Index configurations for all 12 data sources
INDEX_CONFIGS = {
    "options_scan":     IndexConfig("options_scan", 64),
    "mc_summary":       IndexConfig("mc_summary", 32),
    "correlation":      IndexConfig("correlation", 66),
    "market_state":     IndexConfig("market_state", 32),   # delegates to engine
    "scenario":         IndexConfig("scenario", 43),
    "greeks":           IndexConfig("greeks", 16),
    "ml_features":      IndexConfig("ml_features", 32),
    "price_pattern":    IndexConfig("price_pattern", 64),
    "sentiment":        IndexConfig("sentiment", 32),
    "polymarket":       IndexConfig("polymarket", 32),
    "strategy":         IndexConfig("strategy", 32),
    "portfolio":        IndexConfig("portfolio", 16),
}


class IntegrationHub:
    """Central hub managing all 12 TurboQuant similarity indices.

    Each record_*() method encodes domain data into a vector, stores it
    in the appropriate TurboQuantIndex, and optionally returns similar
    historical entries.

    Usage:
        hub = IntegrationHub()
        result = hub.record_options_scan(recommendations)
        similar = hub.search_options_scan(query_recs, top_k=3)
        hub.save_all()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._indices: dict[str, TurboQuantIndex] = {}
        self._mc_compressor = MonteCarloCompressor(bit_width=3)

    def _get_index(self, name: str) -> TurboQuantIndex:
        """Get or lazily create/load an index by name."""
        if name in self._indices:
            return self._indices[name]

        if name == "market_state":
            self._indices[name] = get_market_index()
            return self._indices[name]

        cfg = INDEX_CONFIGS.get(name)
        if cfg is None:
            raise ValueError(f"Unknown index: {name}")

        path = self._data_dir / cfg.filename
        if path.exists():
            try:
                idx = TurboQuantIndex.load(path)
                logger.info("Loaded %s index (%d entries)", name, len(idx.entries))
            except Exception as e:
                logger.warning("Failed to load %s: %s, creating new", name, e)
                idx = TurboQuantIndex(cfg.dimension, cfg.bit_width, cfg.mode)
        else:
            idx = TurboQuantIndex(cfg.dimension, cfg.bit_width, cfg.mode)

        self._indices[name] = idx
        return idx

    def _timestamp(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    # ------------------------------------------------------------------
    # 1. OPTIONS SCAN
    # ------------------------------------------------------------------
    def record_options_scan(self, recommendations: list[dict],
                            cycle_id: str = "", **meta: Any) -> dict:
        """Record a Matrix Maximizer scan cycle.

        Args:
            recommendations: list of PutRecommendation dicts
            cycle_id: optional cycle identifier
            **meta: extra metadata (regime, vix, etc.)

        Returns:
            dict with entry_id and similar_scans
        """
        vec = encode_options_scan(recommendations)
        idx = self._get_index("options_scan")
        ts = self._timestamp()
        entry_id = idx.add(ts, meta.get("regime", "scan"), vec,
                           cycle_id=cycle_id, n_recs=len(recommendations), **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_scans": similar[:5]}

    # ------------------------------------------------------------------
    # 2. MONTE CARLO SUMMARY
    # ------------------------------------------------------------------
    def record_monte_carlo(self, mc_result: Any, **meta: Any) -> dict:
        """Record War Room MC simulation fingerprint.

        For full path compression, use compress_mc_paths() separately.
        """
        result_dict = mc_result if isinstance(mc_result, dict) else mc_result.__dict__
        vec = encode_mc_summary(result_dict)
        idx = self._get_index("mc_summary")
        ts = self._timestamp()
        entry_id = idx.add(ts, meta.get("regime", "mc_run"), vec,
                           n_paths=result_dict.get("n_paths", 0),
                           horizon=result_dict.get("horizon_days", 0), **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_runs": similar[:5]}

    def compress_mc_paths(self, paths: np.ndarray) -> dict:
        """Full path-level compression via MonteCarloCompressor.

        Args:
            paths: shape (n_paths, n_assets, n_days)

        Returns:
            compressed dict with stats
        """
        return self._mc_compressor.compress_paths(paths)

    # ------------------------------------------------------------------
    # 3. CORRELATION MATRIX
    # ------------------------------------------------------------------
    def record_correlation(self, corr_matrix: np.ndarray,
                           vols: Optional[dict[str, float]] = None,
                           **meta: Any) -> dict:
        """Record asset correlation regime snapshot."""
        vec = encode_correlation_matrix(corr_matrix, vols)
        idx = self._get_index("correlation")
        ts = self._timestamp()
        entry_id = idx.add(ts, meta.get("regime", "corr"), vec, **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_regimes": similar[:5]}

    # ------------------------------------------------------------------
    # 4. MARKET STATE (delegates to turboquant_engine)
    # ------------------------------------------------------------------
    def record_market_state(self, snapshot: dict,
                            regime: Optional[dict] = None,
                            **meta: Any) -> dict:
        """Record market state via the core engine."""
        return record_market_state(snapshot, regime, **meta)

    # ------------------------------------------------------------------
    # 5. SCENARIO ENGINE
    # ------------------------------------------------------------------
    def record_scenarios(self, scenarios: list[dict], **meta: Any) -> dict:
        """Record 43-scenario probability snapshot."""
        vec = encode_scenario_state(scenarios)
        idx = self._get_index("scenario")
        ts = self._timestamp()

        # Which scenarios are active?
        active = [s["code"] for s in scenarios
                  if s.get("probability", 0) > 0.25 or s.get("status", "") in ("ACTIVE", "ESCALATING", "PEAK")]
        entry_id = idx.add(ts, meta.get("regime", "scenario"), vec,
                           active_scenarios=",".join(active[:10]), **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_scenarios": similar[:5]}

    # ------------------------------------------------------------------
    # 6. PORTFOLIO GREEKS
    # ------------------------------------------------------------------
    def record_greeks(self, snapshot: Any, **meta: Any) -> dict:
        """Record portfolio Greeks aggregation snapshot."""
        snap_dict = snapshot if isinstance(snapshot, dict) else snapshot.__dict__
        vec = encode_portfolio_greeks(snap_dict)
        idx = self._get_index("greeks")
        ts = self._timestamp()
        risk_level = snap_dict.get("risk_level", "UNKNOWN")
        entry_id = idx.add(ts, str(risk_level), vec,
                           num_positions=snap_dict.get("num_positions", 0), **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_risk_states": similar[:5]}

    # ------------------------------------------------------------------
    # 7. ML FEATURES
    # ------------------------------------------------------------------
    def record_ml_features(self, features: dict, model_type: str = "",
                           **meta: Any) -> dict:
        """Record ML training feature vector."""
        vec = encode_ml_features(features)
        idx = self._get_index("ml_features")
        ts = self._timestamp()
        entry_id = idx.add(ts, meta.get("regime", "ml"), vec,
                           model_type=model_type, **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_feature_states": similar[:5]}

    # ------------------------------------------------------------------
    # 8. PRICE PATTERNS
    # ------------------------------------------------------------------
    def record_price_pattern(self, returns_dict: dict[str, list[float]],
                             window: int = 20, **meta: Any) -> dict:
        """Record multi-asset return pattern snapshot."""
        vec = encode_price_pattern(returns_dict, window)
        idx = self._get_index("price_pattern")
        ts = self._timestamp()
        entry_id = idx.add(ts, meta.get("regime", "pattern"), vec,
                           window=window, **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_patterns": similar[:5]}

    # ------------------------------------------------------------------
    # 9. REDDIT SENTIMENT
    # ------------------------------------------------------------------
    def record_sentiment(self, sentiment_list: list[dict],
                         **meta: Any) -> dict:
        """Record Reddit sentiment snapshot."""
        vec = encode_sentiment(sentiment_list)
        idx = self._get_index("sentiment")
        ts = self._timestamp()

        scores = [s.get("sentiment_score", 0.0) for s in sentiment_list]
        net = "bullish" if np.mean(scores) > 0.05 else "bearish" if np.mean(scores) < -0.05 else "neutral"
        entry_id = idx.add(ts, net, vec,
                           n_stocks=len(sentiment_list), **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_sentiment": similar[:5]}

    # ------------------------------------------------------------------
    # 10. POLYMARKET
    # ------------------------------------------------------------------
    def record_polymarket(self, matches: list[dict], **meta: Any) -> dict:
        """Record Polymarket thesis-market edge snapshot."""
        vec = encode_polymarket(matches)
        idx = self._get_index("polymarket")
        ts = self._timestamp()
        entry_id = idx.add(ts, meta.get("regime", "poly"), vec,
                           n_matches=len(matches), **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_poly": similar[:5]}

    # ------------------------------------------------------------------
    # 11. OPTIONS STRATEGY
    # ------------------------------------------------------------------
    def record_strategy(self, strategy: dict, **meta: Any) -> dict:
        """Record an options strategy for similarity retrieval."""
        vec = encode_strategy(strategy)
        idx = self._get_index("strategy")
        ts = self._timestamp()
        ticker = strategy.get("ticker", strategy.get("symbol", ""))
        stype = strategy.get("strategy", strategy.get("type", ""))
        entry_id = idx.add(ts, meta.get("regime", "strat"), vec,
                           ticker=ticker, strategy_type=stype, **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_strategies": similar[:5]}

    # ------------------------------------------------------------------
    # 12. PORTFOLIO BALANCE
    # ------------------------------------------------------------------
    def record_portfolio(self, snapshot: dict, **meta: Any) -> dict:
        """Record cross-platform balance snapshot."""
        vec = encode_portfolio_balance(snapshot)
        idx = self._get_index("portfolio")
        ts = self._timestamp()
        total = snapshot.get("_summary", {}).get("total_usd_approx", 0.0)
        entry_id = idx.add(ts, meta.get("regime", "balance"), vec,
                           total_usd=total, **meta)
        similar = idx.search(vec, top_k=5)
        similar = [s for s in similar if s["timestamp"] != ts]
        return {"entry_id": entry_id, "similar_portfolio_states": similar[:5]}

    # ------------------------------------------------------------------
    # SEARCH METHODS (query without recording)
    # ------------------------------------------------------------------
    def search_options_scan(self, recs: list[dict], top_k: int = 5) -> list[dict]:
        return self._get_index("options_scan").search(encode_options_scan(recs), top_k)

    def search_mc_summary(self, mc_result: Any, top_k: int = 5) -> list[dict]:
        d = mc_result if isinstance(mc_result, dict) else mc_result.__dict__
        return self._get_index("mc_summary").search(encode_mc_summary(d), top_k)

    def search_correlation(self, corr: np.ndarray,
                           vols: Optional[dict] = None, top_k: int = 5) -> list[dict]:
        return self._get_index("correlation").search(encode_correlation_matrix(corr, vols), top_k)

    def search_scenario(self, scenarios: list[dict], top_k: int = 5) -> list[dict]:
        return self._get_index("scenario").search(encode_scenario_state(scenarios), top_k)

    def search_greeks(self, snapshot: Any, top_k: int = 5) -> list[dict]:
        d = snapshot if isinstance(snapshot, dict) else snapshot.__dict__
        return self._get_index("greeks").search(encode_portfolio_greeks(d), top_k)

    def search_price_pattern(self, returns: dict[str, list[float]],
                             window: int = 20, top_k: int = 5) -> list[dict]:
        return self._get_index("price_pattern").search(encode_price_pattern(returns, window), top_k)

    def search_sentiment(self, sents: list[dict], top_k: int = 5) -> list[dict]:
        return self._get_index("sentiment").search(encode_sentiment(sents), top_k)

    def search_polymarket(self, matches: list[dict], top_k: int = 5) -> list[dict]:
        return self._get_index("polymarket").search(encode_polymarket(matches), top_k)

    def search_strategy(self, strategy: dict, top_k: int = 5) -> list[dict]:
        return self._get_index("strategy").search(encode_strategy(strategy), top_k)

    def search_portfolio(self, snapshot: dict, top_k: int = 5) -> list[dict]:
        return self._get_index("portfolio").search(encode_portfolio_balance(snapshot), top_k)

    # ------------------------------------------------------------------
    # PERSISTENCE AND STATS
    # ------------------------------------------------------------------
    def save_all(self) -> dict[str, str]:
        """Save all loaded indices to disk."""
        results = {}
        for name, idx in self._indices.items():
            if name == "market_state":
                save_market_index()
                results[name] = "saved (via engine)"
                continue
            cfg = INDEX_CONFIGS.get(name)
            if cfg is None:
                continue
            path = self._data_dir / cfg.filename
            idx.save(path)
            results[name] = f"saved ({len(idx.entries)} entries)"
            logger.info("Saved %s: %d entries", name, len(idx.entries))
        return results

    def load_all(self) -> dict[str, int]:
        """Pre-load all indices from disk."""
        counts = {}
        for name in INDEX_CONFIGS:
            idx = self._get_index(name)
            counts[name] = len(idx.entries)
        return counts

    def stats(self) -> dict[str, Any]:
        """Get statistics across all loaded indices."""
        result = {}
        for name, idx in self._indices.items():
            s = idx.stats
            result[name] = {
                "entries": s.get("num_entries", 0),
                "dimension": s.get("dimension", 0),
                "bit_width": s.get("bit_width", 0),
                "size_bytes": s.get("total_size_bytes", 0),
            }
        return result

    def full_report(self) -> dict[str, Any]:
        """Comprehensive report for dashboard display."""
        self.load_all()
        total_entries = 0
        total_bytes = 0
        original_bytes = 0
        index_reports = {}

        for name, cfg in INDEX_CONFIGS.items():
            idx = self._indices.get(name)
            if idx is None:
                continue
            n = len(idx.entries)
            total_entries += n
            s = idx.stats
            idx_bytes = s.get("total_size_bytes", 0)
            total_bytes += idx_bytes
            # Estimate original size (float32 per dim per entry)
            orig = n * cfg.dimension * 4
            original_bytes += orig

            index_reports[name] = {
                "entries": n,
                "dimension": cfg.dimension,
                "bit_width": cfg.bit_width,
                "compressed_bytes": idx_bytes,
                "original_bytes": orig,
                "compression_ratio": orig / max(idx_bytes, 1) if idx_bytes > 0 else 0.0,
            }

        return {
            "total_entries": total_entries,
            "total_compressed_bytes": total_bytes,
            "total_original_bytes": original_bytes,
            "overall_compression_ratio": original_bytes / max(total_bytes, 1),
            "num_active_indices": sum(1 for v in index_reports.values() if v["entries"] > 0),
            "indices": index_reports,
            "algorithm": "TurboQuant (arXiv:2504.19874, ICLR 2026)",
        }


# ============================================================================
# INGEST FUNCTIONS — Standalone helpers to load existing data files
# ============================================================================

def ingest_balance_snapshot(hub: IntegrationHub,
                           path: Optional[str] = None) -> Optional[dict]:
    """Ingest the latest balance_snapshot.json."""
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "balance_snapshot.json")
    if not os.path.exists(path):
        logger.warning("balance_snapshot.json not found at %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        snapshot = json.load(f)
    return hub.record_portfolio(snapshot)


def ingest_matrix_maximizer_cycles(hub: IntegrationHub,
                                   data_dir: Optional[str] = None,
                                   max_files: int = 50) -> int:
    """Ingest historical Matrix Maximizer cycle JSON files.

    Args:
        hub: IntegrationHub instance
        data_dir: path to data/matrix_maximizer/ (auto-detected if None)
        max_files: limit on files to process

    Returns:
        Number of cycles ingested
    """
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "data" / "matrix_maximizer")
    if not os.path.isdir(data_dir):
        logger.warning("Matrix maximizer data dir not found: %s", data_dir)
        return 0

    count = 0
    for fn in sorted(os.listdir(data_dir))[:max_files]:
        if not fn.endswith(".json"):
            continue
        fpath = os.path.join(data_dir, fn)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            recs = data if isinstance(data, list) else data.get("recommendations", [])
            if recs:
                hub.record_options_scan(recs, cycle_id=fn)
                count += 1
        except Exception as e:
            logger.warning("Failed to ingest %s: %s", fn, e)

    logger.info("Ingested %d Matrix Maximizer cycles", count)
    return count


def ingest_polymarket_scenarios(hub: IntegrationHub,
                                path: Optional[str] = None) -> Optional[dict]:
    """Ingest polymarket_scenario_bets.json."""
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "polymarket_scenario_bets.json")
    if not os.path.exists(path):
        logger.warning("polymarket_scenario_bets.json not found at %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    bets = data if isinstance(data, list) else data.get("bets", data.get("matches", []))
    return hub.record_polymarket(bets)


# ============================================================================
# CLI DEMO
# ============================================================================

def _demo_integration():
    """Demo all 12 integration points with synthetic data."""
    print("=" * 70)
    print("  TurboQuant Integration Hub — 12-Index Demo")
    print("=" * 70)

    hub = IntegrationHub()
    rng = np.random.RandomState(42)

    # 1. Options Scan
    print("\n[1/12] Options Scan Index (64-dim)")
    recs = [
        {"ticker": "SPY", "composite_score": 85, "delta_score": 70,
         "greeks": {"delta": -0.35, "sigma": 0.28}, "total_cost": 500,
         "mandate": "aggressive", "risk_pct": 0.05,
         "contract": {"dte": 45}},
        {"ticker": "QQQ", "composite_score": 72, "delta_score": 65,
         "greeks": {"delta": -0.25, "sigma": 0.32}, "total_cost": 300,
         "mandate": "standard", "risk_pct": 0.03,
         "contract": {"dte": 30}},
    ]
    r = hub.record_options_scan(recs, cycle_id="demo_001")
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_scans'])} similar found")

    # 2. Monte Carlo Summary
    print("\n[2/12] Monte Carlo Summary Index (32-dim)")
    mc = {
        "n_paths": 100000, "horizon_days": 90,
        "asset_means": {a: rng.uniform(-10, 10) for a in WR_ASSETS},
        "asset_p95": {a: rng.uniform(50, 200) for a in WR_ASSETS},
        "asset_p5": {a: rng.uniform(10, 80) for a in WR_ASSETS},
        "pf_mean": 55000, "pf_median": 48000,
        "var_95": -8000, "cvar_95": -12000,
        "prob_oil_120": 0.35, "prob_gold_5500": 0.22,
        "prob_spy_600": 0.18, "prob_btc_55k": 0.42,
        "prob_pf_150k": 0.08, "prob_pf_1m": 0.002,
    }
    r = hub.record_monte_carlo(mc)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_runs'])} similar found")

    # 3. Correlation Matrix
    print("\n[3/12] Correlation Matrix Index (66-dim)")
    corr = np.eye(11, dtype=np.float32)
    for i in range(11):
        for j in range(i + 1, 11):
            c = rng.uniform(-0.3, 0.8)
            corr[i, j] = corr[j, i] = c
    vols = {a: rng.uniform(0.15, 1.2) for a in WR_ASSETS}
    r = hub.record_correlation(corr, vols)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_regimes'])} similar found")

    # 4. Market State (delegates to engine)
    print("\n[4/12] Market State Index (32-dim) [engine]")
    snap = {"vix": 25, "hy_spread_bps": 450, "oil_price": 95, "fear_greed": 28}
    regime = {"primary_regime": "credit_stress", "regime_confidence": 0.7}
    r = hub.record_market_state(snap, regime)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_states'])} similar found")

    # 5. Scenario Engine
    print("\n[5/12] Scenario Engine Index (43-dim)")
    scenarios = [
        {"code": "HORMUZ", "probability": 0.65, "status": "ESCALATING"},
        {"code": "DEBT_CRISIS", "probability": 0.30, "status": "ACTIVE"},
        {"code": "SUPERCYCLE", "probability": 0.45, "status": "ACTIVE"},
        {"code": "AI_BUBBLE", "probability": 0.20, "status": "EMERGING"},
    ]
    r = hub.record_scenarios(scenarios)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_scenarios'])} similar found")

    # 6. Portfolio Greeks
    print("\n[6/12] Portfolio Greeks Index (16-dim)")
    greeks_snap = {
        "total_delta": -45.2, "total_gamma": 12.5, "total_theta": -180.0,
        "total_vega": 95.0, "total_vanna": 3.2, "total_charm": -1.5,
        "dollar_delta": -22000, "gamma_risk_1pct": 3500,
        "daily_theta": -180, "vega_risk_1pct": 4800,
        "delta_limit": 100, "gamma_limit": 50,
        "theta_limit": -500, "vega_limit": 200,
        "risk_level": "ELEVATED", "num_positions": 12,
    }
    r = hub.record_greeks(greeks_snap)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_risk_states'])} similar found")

    # 7. ML Features
    print("\n[7/12] ML Feature Index (32-dim)")
    features = {
        "realized_vol": 0.22, "parkinson_vol": 0.25, "volume_sma": 5e6,
        "volume_ratio": 1.3, "momentum_5": -2.5, "momentum_20": -8.0,
        "vix": 25, "hy_spread_bps": 450, "fear_greed": 28,
        "oil_price": 95, "gold_price": 4861,
    }
    r = hub.record_ml_features(features, model_type="RETURN_PREDICTOR")
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_feature_states'])} similar found")

    # 8. Price Patterns
    print("\n[8/12] Price Pattern Index (64-dim)")
    returns = {a: (rng.randn(20) * 0.02).tolist() for a in WR_ASSETS}
    r = hub.record_price_pattern(returns)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_patterns'])} similar found")

    # 9. Reddit Sentiment
    print("\n[9/12] Sentiment Index (32-dim)")
    sentiments = [
        {"symbol": "AAPL", "sentiment_score": 0.45, "no_of_comments": 250},
        {"symbol": "TSLA", "sentiment_score": -0.30, "no_of_comments": 180},
        {"symbol": "GME", "sentiment_score": 0.80, "no_of_comments": 500},
        {"symbol": "NVDA", "sentiment_score": 0.55, "no_of_comments": 320},
    ]
    r = hub.record_sentiment(sentiments)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_sentiment'])} similar found")

    # 10. Polymarket
    print("\n[10/12] Polymarket Index (32-dim)")
    poly_matches = [
        {"stage": "iran_escalation", "edge": 0.15, "thesis_probability": 0.35,
         "market_price": 0.20, "volume_24h": 50000, "tier": "deep_value"},
        {"stage": "gold_reprice", "edge": 0.10, "thesis_probability": 0.45,
         "market_price": 0.35, "volume_24h": 80000, "tier": "value"},
    ]
    r = hub.record_polymarket(poly_matches)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_poly'])} similar found")

    # 11. Options Strategy
    print("\n[11/12] Strategy Index (32-dim)")
    strat = {
        "strategy": "bear_put_spread", "ticker": "SPY",
        "net_debit": 250, "max_profit": 750, "max_loss": 250,
        "reward_risk_ratio": 3.0, "breakeven": 645,
        "net_delta": -0.30, "net_gamma": 0.02,
        "net_vega": 0.15, "net_theta": -0.05,
        "iv": 0.28, "dte": 45, "spot": 665,
        "K_long": 660, "K_short": 650,
        "composite_score": 78, "volume": 5000, "open_interest": 25000,
        "mandate": "aggressive", "contracts": 2, "risk_pct": 0.025,
    }
    r = hub.record_strategy(strat)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_strategies'])} similar found")

    # 12. Portfolio Balance
    print("\n[12/12] Portfolio Balance Index (16-dim)")
    balance = {
        "ibkr": {"net_liquidation": 10736.88, "status": "ok",
                 "positions": [{"symbol": "SPY"}, {"symbol": "QQQ"}]},
        "moomoo": {"net_liquidation": 2609.26, "status": "ok",
                   "positions": [{"symbol": "XLE"}]},
        "polymarket": {"net_liquidation": 535.74, "status": "ok", "positions": []},
        "_summary": {"total_usd_approx": 13882, "total_cad_component": 0,
                     "cad_usd_rate": 0.72},
    }
    r = hub.record_portfolio(balance)
    print(f"  Recorded: entry #{r['entry_id']}, {len(r['similar_portfolio_states'])} similar found")

    # Full report
    print("\n" + "=" * 70)
    print("  INTEGRATION HUB REPORT")
    print("=" * 70)
    report = hub.full_report()
    print(f"  Total entries:   {report['total_entries']}")
    print(f"  Active indices:  {report['num_active_indices']}/12")
    print(f"  Compressed:      {report['total_compressed_bytes']:,} bytes")
    print(f"  Original:        {report['total_original_bytes']:,} bytes")
    print(f"  Overall ratio:   {report['overall_compression_ratio']:.1f}x")
    print()
    for name, info in report["indices"].items():
        if info["entries"] > 0:
            print(f"  {name:20s}  dim={info['dimension']:3d}  "
                  f"entries={info['entries']:3d}  "
                  f"ratio={info['compression_ratio']:.1f}x")

    print("\n" + "=" * 70)
    print("  All 12 indices operational. Use IntegrationHub in production:")
    print("    from strategies.turboquant_integrations import IntegrationHub")
    print("    hub = IntegrationHub()")
    print("    hub.record_options_scan(recs)")
    print("    hub.save_all()")
    print("=" * 70)


if __name__ == "__main__":
    _demo_integration()

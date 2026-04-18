"""
OpenClaw Skill → Python Runtime Bridge
========================================

Closes the gap between the 93 OpenClaw skill manifests and the actual AAC
Python runtime.  Each skill is mapped to:

    1. A concrete Python callable (function / async method)
    2. The agent that owns it
    3. A cadence / timing rule (on-demand, scheduled, event-driven)
    4. Input extraction logic (natural-language → kwargs)
    5. Output formatting (Python object → Markdown / JSON for chat)

The bridge is consumed by the OpenClaw Gateway Bridge:
    inbound message → intent classifier → skill resolver → THIS BRIDGE → runtime → formatted response

Architecture
~~~~~~~~~~~~
::

    OpenClawGatewayBridge
        ↓  route_to_agent()
    AZSupremeOpenClawHandler
        ↓  resolve_skill()
    SkillRuntimeBridge          ← THIS MODULE
        ↓  execute(skill_id, params)
    strategies.regime_engine    (or war_room_engine, flow_signals, …)
        ↓
    Formatted response string → sent back to channel
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


# ─── Enums ──────────────────────────────────────────────────────────────────

class SkillCadence(Enum):
    """When a skill should execute."""
    ON_DEMAND = "on_demand"          # User asks → execute immediately
    SCHEDULED = "scheduled"          # Cron-driven (e.g. morning briefing)
    EVENT_DRIVEN = "event_driven"    # Fired by a system event (regime flip)
    CONTINUOUS = "continuous"        # Running in a loop (auto engine)


class SkillAgent(Enum):
    """Which AAC agent owns the skill."""
    AZ_SUPREME = "AZ-SUPREME"
    AX_HELIX = "AX-HELIX"
    BIGBRAIN = "BIGBRAIN-INTELLIGENCE"
    TRADING = "TRADING-EXECUTION"
    RISK = "RISK-MANAGEMENT"
    ACCOUNTING = "CENTRAL-ACCOUNTING"
    CRYPTO = "CRYPTO-INTELLIGENCE"
    INFRA = "SHARED-INFRASTRUCTURE"
    DOCTRINE = "DOCTRINE-ENGINE"
    WAR_ROOM = "WAR-ROOM"


# ─── Skill Binding ──────────────────────────────────────────────────────────

@dataclass
class SkillBinding:
    """Maps one OpenClaw skill to its Python runtime target."""
    skill_id: str
    agent: SkillAgent
    cadence: SkillCadence
    description: str
    # The async callable that does the real work.
    # Signature: async (params: dict[str, Any]) -> dict[str, Any]
    handler: Optional[Callable[..., Coroutine[Any, Any, dict[str, Any]]]] = None
    # Cron expression (only for SCHEDULED cadence)
    cron: Optional[str] = None
    # Minimum seconds between invocations (rate limit)
    cooldown_sec: float = 0.0
    # Last execution timestamp (for cooldown enforcement)
    _last_run: Optional[float] = field(default=None, repr=False)
    # Whether the handler has been lazily resolved
    _resolved: bool = field(default=False, repr=False)


# ─── Lazy Import Helpers ────────────────────────────────────────────────────
# We don't import heavy modules at bridge load time.  Each handler is a thin
# async wrapper that imports on first call, avoiding import-time side-effects
# and keeping startup fast.

async def _regime_evaluate(params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate macro regime via 9 formulas."""
    from strategies.regime_engine import RegimeEngine, MacroSnapshot
    engine = RegimeEngine()
    # If caller supplied snapshot fields, build a MacroSnapshot; otherwise
    # pull live data first.
    snap_fields = params.get("snapshot")
    if snap_fields and isinstance(snap_fields, dict):
        snap = MacroSnapshot(**snap_fields)
    else:
        from strategies.war_room_live_feeds import update_all_live_data
        feeds = await update_all_live_data(timeout_sec=20.0)
        snap = MacroSnapshot(
            hy_spread_bps=getattr(feeds, "hy_spread_bp_live", 300),
            vix=getattr(feeds, "ibkr_vix", 20.0),
            oil_price=getattr(feeds, "oil_price_wti", 80.0),
            fed_rate=getattr(feeds, "fed_rate", 5.0),
            fear_greed=getattr(feeds, "fear_greed_value", 50),
        )
    state = engine.evaluate(snap)
    return {
        "primary_regime": str(getattr(state, "primary_regime", "UNCERTAIN")),
        "secondary_regime": str(getattr(state, "secondary_regime", None)),
        "regime_confidence": getattr(state, "regime_confidence", 0.0),
        "vol_shock_readiness": getattr(state, "vol_shock_readiness", 0),
        "armed_formulas": [str(f) for f in getattr(state, "armed_formulas", [])],
        "bear_signals": getattr(state, "bear_signals", 0),
        "bull_signals": getattr(state, "bull_signals", 0),
    }


async def _live_scan(params: dict[str, Any]) -> dict[str, Any]:
    """Fetch all live feeds (11 sources)."""
    from strategies.war_room_live_feeds import update_all_live_data
    feeds = params.get("feeds")
    include = feeds if isinstance(feeds, list) else None
    result = await update_all_live_data(
        include_feeds=include,
        force_refresh=params.get("force", False),
        timeout_sec=float(params.get("timeout", 30)),
    )
    # Return a serializable dict of the dataclass
    out: dict[str, Any] = {}
    for attr in (
        "btc_price", "eth_price", "xrp_price", "total_market_cap_usd",
        "btc_dominance", "put_call_ratio", "market_tone",
        "fear_greed_value", "gold_price_oz", "oil_price_wti",
        "fed_rate", "dxy_index", "hy_spread_bp_live",
        "ibkr_net_liquidation", "ibkr_vix", "news_severity_score",
        "stablecoin_depeg_pct",
    ):
        val = getattr(result, attr, None)
        if val is not None:
            out[attr] = val
    out["errors"] = getattr(result, "errors", [])
    return out


async def _monte_carlo(params: dict[str, Any]) -> dict[str, Any]:
    """Run Monte Carlo simulation (100K paths, 11 assets)."""
    from strategies.war_room_engine import run_monte_carlo
    mc = run_monte_carlo(
        n_paths=int(params.get("paths", 100_000)),
        horizon_days=int(params.get("horizon", 90)),
        seed=params.get("seed"),
    )
    return {
        "var_95": getattr(mc, "var_95", None),
        "cvar_95": getattr(mc, "cvar_95", None),
        "prob_portfolio_above_tier1": getattr(mc, "prob_portfolio_above_tier1", None),
        "prob_portfolio_above_tier2": getattr(mc, "prob_portfolio_above_tier2", None),
        "runtime_ms": getattr(mc, "runtime_ms", None),
        "n_paths": getattr(mc, "n_paths", 100_000),
    }


async def _greeks_put(params: dict[str, Any]) -> dict[str, Any]:
    """Black-Scholes put pricing + Greeks."""
    from strategies.war_room_engine import bs_put
    g = bs_put(
        S=float(params["spot"]),
        K=float(params["strike"]),
        T=float(params.get("dte", 30)) / 365.0,
        r=float(params.get("rate", 0.05)),
        sigma=float(params.get("iv", 0.30)),
        q=float(params.get("div_yield", 0.0)),
    )
    return {
        "price": g.price,
        "delta": g.delta,
        "gamma": g.gamma,
        "vega": g.vega,
        "theta": g.theta,
        "vomma": getattr(g, "vomma", None),
        "moneyness": getattr(g, "moneyness", None),
    }


async def _greeks_call(params: dict[str, Any]) -> dict[str, Any]:
    """Black-Scholes call pricing + Greeks."""
    from strategies.war_room_engine import bs_call
    g = bs_call(
        S=float(params["spot"]),
        K=float(params["strike"]),
        T=float(params.get("dte", 30)) / 365.0,
        r=float(params.get("rate", 0.05)),
        sigma=float(params.get("iv", 0.30)),
        q=float(params.get("div_yield", 0.0)),
    )
    return {
        "price": g.price,
        "delta": g.delta,
        "gamma": g.gamma,
        "vega": g.vega,
        "theta": g.theta,
        "vomma": getattr(g, "vomma", None),
        "moneyness": getattr(g, "moneyness", None),
    }


async def _flow_analysis(params: dict[str, Any]) -> dict[str, Any]:
    """Analyze options flow via Unusual Whales."""
    from strategies.options_intelligence.flow_signals import FlowSignalEngine
    engine = FlowSignalEngine()
    tickers = params.get("tickers")
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(",")]
    convictions = await engine.analyze_flow(uw_client=None, tickers=tickers)
    return {
        "convictions": [
            {
                "ticker": getattr(c, "ticker", ""),
                "conviction": getattr(c, "conviction", 0.0),
                "direction": str(getattr(c, "direction", "")),
                "put_call_ratio": getattr(c, "put_call_ratio", None),
                "sweep_count": getattr(c, "sweep_count", 0),
            }
            for c in (convictions or [])[:20]
        ],
        "count": len(convictions or []),
    }


async def _ai_score_trade(params: dict[str, Any]) -> dict[str, Any]:
    """Score a trade setup via AI (LLM + heuristic fallback)."""
    from strategies.options_intelligence.ai_scorer import AITradeScorer, TradeSetup
    scorer = AITradeScorer()
    setup = TradeSetup(
        ticker=params.get("ticker", "SPY"),
        direction=params.get("direction", "put"),
        strike=float(params.get("strike", 0)),
        expiry=params.get("expiry", ""),
        dte=int(params.get("dte", 30)),
        premium=float(params.get("premium", 0)),
    )
    score = await scorer.score_trade(setup)
    return {
        "score": getattr(score, "score", 0),
        "confidence_level": getattr(score, "confidence_level", "low"),
        "reasoning": getattr(score, "reasoning", ""),
        "suggested_adjustments": getattr(score, "suggested_adjustments", []),
    }


async def _fibonacci(params: dict[str, Any]) -> dict[str, Any]:
    """Calculate Fibonacci retracement / extension / harmonic detection."""
    from strategies.golden_ratio_finance import FibonacciCalculator
    calc = FibonacciCalculator()
    action = params.get("action", "retracement")
    high = float(params.get("high", 100))
    low = float(params.get("low", 80))
    if action == "extension":
        result = calc.extension(high=high, low=low)
    elif action == "harmonics":
        prices = params.get("prices", [])
        patterns = calc.detect_harmonics(prices)
        return {
            "patterns": [
                {
                    "name": getattr(p, "name", ""),
                    "confidence": getattr(p, "confidence", 0.0),
                    "prz": getattr(p, "prz", None),
                }
                for p in (patterns or [])
            ]
        }
    else:
        result = calc.retracement(high=high, low=low)
    return {
        "levels": getattr(result, "levels", {}),
        "direction": getattr(result, "direction", "up"),
    }


async def _risk_check(params: dict[str, Any]) -> dict[str, Any]:
    """Pre-trade risk gate check."""
    from strategies.paper_trading.risk_manager import RiskManager
    mgr = RiskManager()
    allowed, reason = mgr.pre_trade_check(
        account_equity=float(params.get("equity", 10000)),
        account_positions=params.get("positions", {}),
        symbol=params.get("symbol", ""),
        side=params.get("side", "buy"),
        quantity=float(params.get("quantity", 1)),
        price=float(params.get("price", 0)),
        strategy=params.get("strategy", "manual"),
    )
    return {"allowed": allowed, "reason": reason}


async def _regime_detect(params: dict[str, Any]) -> dict[str, Any]:
    """Lightweight regime classification for paper trading."""
    from strategies.paper_trading.regime_detector import RegimeDetector
    detector = RegimeDetector()
    prices = params.get("prices", {})
    regime = detector.update_prices(prices)
    guidance = detector.get_guidance()
    return {
        "regime": str(regime),
        "guidance": guidance,
    }


async def _portfolio_status(params: dict[str, Any]) -> dict[str, Any]:
    """Portfolio value + P&L from CentralAccounting."""
    from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
    engine = FinancialAnalysisEngine()
    value = await engine.calculate_portfolio_value()
    pnl = await engine.calculate_daily_pnl()
    risk = await engine.update_risk_metrics()
    return {
        "portfolio_value": value,
        "daily_pnl": pnl,
        "max_drawdown_pct": getattr(risk, "max_drawdown_pct", None),
        "daily_loss_pct": getattr(risk, "daily_loss_pct", None),
        "capital_utilization": getattr(risk, "capital_utilization", None),
    }


async def _enterprise_status(params: dict[str, Any]) -> dict[str, Any]:
    """Enterprise-wide division health."""
    from divisions.enterprise import Enterprise
    ent = Enterprise()
    return await ent.status()


async def _morning_briefing(params: dict[str, Any]) -> dict[str, Any]:
    """Generate daily Helix briefing."""
    from divisions.trading.warroom.storm_lifeboat.helix_news import HelixNewsGenerator
    gen = HelixNewsGenerator()
    briefing = gen.generate()
    return {
        "headline": getattr(briefing, "headline", ""),
        "regime": str(getattr(briefing, "regime", "")),
        "mandate": str(getattr(briefing, "mandate", "")),
        "summary": getattr(briefing, "summary", ""),
        "formatted": gen.format_terminal(briefing),
    }


async def _metrics_snapshot(params: dict[str, Any]) -> dict[str, Any]:
    """Paper trading performance metrics."""
    from strategies.paper_trading.metrics import MetricsTracker
    tracker = MetricsTracker()
    # If trades provided, record them
    trades = params.get("trades", [])
    if trades:
        tracker.record_trades([float(t) for t in trades])
    snap = tracker.compute()
    return {
        "sharpe_ratio": getattr(snap, "sharpe_ratio", None),
        "sortino_ratio": getattr(snap, "sortino_ratio", None),
        "profit_factor": getattr(snap, "profit_factor", None),
        "win_rate": getattr(snap, "win_rate", None),
        "expectancy": getattr(snap, "expectancy", None),
        "max_consecutive_losses": getattr(snap, "max_consecutive_losses", None),
    }


async def _stock_forecast(params: dict[str, Any]) -> dict[str, Any]:
    """Regime-aware sector trade opportunities."""
    from strategies.stock_forecaster import StockForecaster
    forecaster = StockForecaster()
    sector = params.get("sector", "CREDIT")
    vix = float(params.get("vix", 20.0))
    opps = forecaster.forecast_sector(
        industry=sector,
        regime=None,  # will use current cached regime
        prices=params.get("prices", {}),
        vix=vix,
    )
    return {
        "opportunities": [
            {
                "rank": getattr(o, "rank", 0),
                "ticker": getattr(o, "ticker", ""),
                "expression": str(getattr(o, "expression_type", "")),
                "roi_score": getattr(o, "roi_score", 0),
                "risk_score": getattr(o, "risk_score", 0),
                "composite": getattr(o, "composite_score", 0),
            }
            for o in (opps or [])[:10]
        ],
        "sector": sector,
    }


async def _market_intel(params: dict[str, Any]) -> dict[str, Any]:
    """Market intelligence model sentiment snapshot."""
    from strategies.market_intelligence_model import MarketIntelligenceModel
    model = MarketIntelligenceModel()
    state = model.get_sentiment_state()
    recs = model.get_recommendations()
    return {
        "sentiment": state,
        "recommendations": [
            {
                "sector": getattr(r, "sector", ""),
                "direction": str(getattr(r, "direction", "")),
                "confidence": getattr(r, "confidence", 0.0),
            }
            for r in (recs or [])[:10]
        ],
    }


async def _dashboard_status(params: dict[str, Any]) -> dict[str, Any]:
    """Master monitoring dashboard status."""
    from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
    dash = AACMasterMonitoringDashboard()
    return await dash.get_system_status()


async def _crypto_prices(params: dict[str, Any]) -> dict[str, Any]:
    """CoinGecko price fetch."""
    from shared.data_sources import CoinGeckoClient
    client = CoinGeckoClient()
    await client.connect()
    try:
        coins = params.get("coins", ["bitcoin", "ethereum", "ripple"])
        if isinstance(coins, str):
            coins = [c.strip() for c in coins.split(",")]
        ticks = await client.get_prices_batch(coins)
        return {
            "prices": {
                getattr(t, "symbol", ""): getattr(t, "price", 0.0)
                for t in (ticks or [])
            }
        }
    finally:
        await client.disconnect()


async def _halt_trading(params: dict[str, Any]) -> dict[str, Any]:
    """Emergency halt — kill switch."""
    from strategies.paper_trading.risk_manager import RiskManager
    mgr = RiskManager()
    reason = params.get("reason", "openclaw_kill_switch")
    mgr.halt(reason)
    return {"halted": True, "reason": reason}


async def _resume_trading(params: dict[str, Any]) -> dict[str, Any]:
    """Resume after halt."""
    from strategies.paper_trading.risk_manager import RiskManager
    mgr = RiskManager()
    mgr.resume()
    return {"halted": False}


async def _milestone_check(params: dict[str, Any]) -> dict[str, Any]:
    """Check 50-milestone spiderweb status."""
    # Milestones are evaluated inside war_room_engine / auto
    from strategies.war_room_auto import WarRoomAutoEngine
    engine = WarRoomAutoEngine()
    regime = engine.get_regime_state()
    return {
        "regime": str(regime),
        "note": "Milestones evaluated during auto-evolve cycle",
    }


async def _noop(params: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for skills not yet wired to runtime."""
    return {"status": "skill_registered", "note": "Runtime handler pending"}


# ─── Skill Registry ────────────────────────────────────────────────────────
# Maps every skill_id to its binding.  Skills are grouped by agent + cadence
# so the bridge can schedule them correctly.

_REGISTRY: dict[str, SkillBinding] = {}


def _r(
    skill_id: str,
    agent: SkillAgent,
    cadence: SkillCadence,
    desc: str,
    handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]],
    cron: Optional[str] = None,
    cooldown: float = 0.0,
) -> None:
    """Register a skill binding."""
    _REGISTRY[skill_id] = SkillBinding(
        skill_id=skill_id,
        agent=agent,
        cadence=cadence,
        description=desc,
        handler=handler,
        cron=cron,
        cooldown_sec=cooldown,
    )


# ─── CORE AAC (10) ─────────────────────────────────────────────────────────
_r("bw-market-intelligence",  SkillAgent.BIGBRAIN,    SkillCadence.ON_DEMAND,    "3-theater market scan",               _market_intel)
_r("bw-trading-signals",      SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Flow-based trading signals",          _flow_analysis)
_r("bw-portfolio-dashboard",  SkillAgent.ACCOUNTING,  SkillCadence.ON_DEMAND,    "Portfolio value + P&L",               _portfolio_status)
_r("bw-risk-monitor",         SkillAgent.RISK,        SkillCadence.CONTINUOUS,   "Doctrine state + risk exposure",      _risk_check,          cooldown=60)
_r("bw-crypto-intel",         SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "CryptoIntelligence prices + metrics", _crypto_prices)
_r("bw-az-supreme-command",   SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Executive command interface",         _enterprise_status)
_r("bw-doctrine-status",      SkillAgent.DOCTRINE,    SkillCadence.ON_DEMAND,    "Doctrine engine state machine",       _regime_evaluate)
_r("bw-morning-briefing",     SkillAgent.AZ_SUPREME,  SkillCadence.SCHEDULED,   "Automated morning briefing",          _morning_briefing,    cron="0 7 * * 1-5")
_r("bw-agent-roster",         SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Division + agent directory",          _enterprise_status)
_r("bw-strategy-explorer",    SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "50 arbitrage strategies",             _stock_forecast)

# ─── TRADING & MARKETS (7) ─────────────────────────────────────────────────
_r("bw-digital-arbitrage",    SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Cross-exchange arbitrage scan",       _flow_analysis)
_r("bw-arbitrage-scanner",    SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Multi-venue arb detection",           _flow_analysis)
_r("bw-day-trading",          SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Intraday momentum signals",           _flow_analysis)
_r("bw-options-trading",      SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "Options Greeks + chain analysis",     _greeks_put)
_r("bw-calls-puts-flow",      SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Options flow: sweeps + blocks",       _flow_analysis)
_r("bw-hedging-strategies",   SkillAgent.RISK,        SkillCadence.ON_DEMAND,    "Portfolio hedging recommendations",   _risk_check)
_r("bw-currency-trading",     SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Forex pair analysis",                 _noop)

# ─── CRYPTO & DEFI (7) ─────────────────────────────────────────────────────
_r("bw-bitcoin-intel",        SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Bitcoin on-chain + macro",            _crypto_prices)
_r("bw-ethereum-defi",        SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "ETH DeFi protocols + gas",            _crypto_prices)
_r("bw-xrp-ripple",           SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "XRP cross-border analysis",           _crypto_prices)
_r("bw-stablecoins",          SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Stablecoin peg + yield",              _crypto_prices)
_r("bw-meme-coins",           SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Meme coin social velocity",           _crypto_prices)
_r("bw-liberty-coin",         SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Liberty coin tracking",               _crypto_prices)
_r("bw-x-tokens",             SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "X token ecosystem",                   _crypto_prices)

# ─── FINANCE & BANKING (3) ─────────────────────────────────────────────────
_r("bw-banking-intel",        SkillAgent.ACCOUNTING,  SkillCadence.ON_DEMAND,    "Banking + international finance",     _noop)
_r("bw-accounting-engine",    SkillAgent.ACCOUNTING,  SkillCadence.ON_DEMAND,    "Financial accounting + reporting",    _portfolio_status)
_r("bw-regulations",          SkillAgent.DOCTRINE,    SkillCadence.ON_DEMAND,    "Regulatory compliance",               _noop)

# ─── WEALTH BUILDING (3) ───────────────────────────────────────────────────
_r("bw-money-mastery",        SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Cash flow planning",                  _noop)
_r("bw-wealth-building",      SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Generational wealth strategies",      _noop)
_r("bw-superstonk-dd",        SkillAgent.BIGBRAIN,    SkillCadence.ON_DEMAND,    "SuperStonk due diligence",            _noop)

# ─── ADVANCED ANALYSIS (3) ─────────────────────────────────────────────────
_r("bw-crash-indicators",     SkillAgent.RISK,        SkillCadence.EVENT_DRIVEN, "2007/2008 crash pattern detection",   _regime_evaluate)
_r("bw-golden-ratio-finance", SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "Fibonacci harmonics + golden ratio",  _fibonacci)
_r("bw-jonny-bravo-course",   SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Jonny Bravo methodology",             _noop)

# ─── OPENCLAW POWER-UPS (2) ────────────────────────────────────────────────
_r("bw-polymarket-autopilot", SkillAgent.TRADING,     SkillCadence.CONTINUOUS,   "Prediction market paper trading",     _noop)
_r("bw-second-brain",         SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Knowledge capture",                   _noop)

# ─── QUANTITATIVE & PRICING (5) ────────────────────────────────────────────
_r("bw-black-scholes",        SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "Black-Scholes options pricing",       _greeks_put)
_r("bw-security-hardening",   SkillAgent.INFRA,       SkillCadence.ON_DEMAND,    "Security + CVE monitoring",           _noop)
_r("bw-skill-scanner",        SkillAgent.INFRA,       SkillCadence.ON_DEMAND,    "ClawHub skill security scan",         _noop)
_r("bw-flash-loans",          SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "DeFi flash loan arbitrage",           _noop)
_r("bw-dca-grid",             SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "DCA ladders + grid trading",          _noop)

# ─── AI STRATEGIES (5) ─────────────────────────────────────────────────────
_r("bw-trinity-scanner",      SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Trinity/Panic/2B reversal scan",      _flow_analysis)
_r("bw-backtester",           SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "Multi-strategy backtesting",          _metrics_snapshot)
_r("bw-trade-journal",        SkillAgent.ACCOUNTING,  SkillCadence.ON_DEMAND,    "Automated trade journaling",          _metrics_snapshot)
_r("bw-api-cost-guard",       SkillAgent.INFRA,       SkillCadence.CONTINUOUS,   "API cost monitoring",                 _noop,                cooldown=300)
_r("bw-graduated-mode",       SkillAgent.DOCTRINE,    SkillCadence.ON_DEMAND,    "Graduated trading permissions",       _noop)

# ─── DEFI & YIELD (5) ──────────────────────────────────────────────────────
_r("bw-yield-optimizer",      SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "DeFi yield optimization",             _crypto_prices)
_r("bw-onchain-forensics",    SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "On-chain wallet forensics",           _noop)
_r("bw-sentiment-engine",     SkillAgent.BIGBRAIN,    SkillCadence.ON_DEMAND,    "NLP sentiment analysis",              _market_intel)
_r("bw-sec-monitor",          SkillAgent.BIGBRAIN,    SkillCadence.SCHEDULED,   "SEC filings + insider tracking",      _noop,                cron="0 18 * * 1-5")
_r("bw-earnings-engine",      SkillAgent.BIGBRAIN,    SkillCadence.SCHEDULED,   "Earnings calendar tracking",          _noop,                cron="0 6 * * 1-5")

# ─── SECURITY & INFRASTRUCTURE (5) ─────────────────────────────────────────
_r("bw-scam-detector",        SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Crypto scam detection",               _noop)
_r("bw-websocket-feeds",      SkillAgent.INFRA,       SkillCadence.CONTINUOUS,   "Real-time WebSocket price feeds",     _live_scan,           cooldown=5)
_r("bw-kelly-criterion",      SkillAgent.RISK,        SkillCadence.ON_DEMAND,    "Kelly Criterion position sizing",     _risk_check)
_r("bw-var-calculator",       SkillAgent.RISK,        SkillCadence.ON_DEMAND,    "Value at Risk calculator",            _monte_carlo)
_r("bw-tax-harvester",        SkillAgent.ACCOUNTING,  SkillCadence.ON_DEMAND,    "Tax-loss harvesting",                 _noop)

# ─── FINANCIAL PLANNING (5) ────────────────────────────────────────────────
_r("bw-rebalance-alerts",     SkillAgent.RISK,        SkillCadence.EVENT_DRIVEN, "Portfolio drift + rebalancing",       _portfolio_status)
_r("bw-market-commentary",    SkillAgent.BIGBRAIN,    SkillCadence.SCHEDULED,   "AI market commentary",                _morning_briefing,    cron="0 8 * * 1-5")
_r("bw-compliance-engine",    SkillAgent.DOCTRINE,    SkillCadence.ON_DEMAND,    "Compliance documentation",            _noop)
_r("bw-wallet-manager",       SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Multi-chain wallet management",       _noop)
_r("bw-prediction-markets",   SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Prediction market intelligence",      _noop)

# ─── ADVANCED OPS (5) ──────────────────────────────────────────────────────
_r("bw-ccxt-exchange",        SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "CCXT multi-exchange integration",     _noop)
_r("bw-milestone-tracker",    SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "50-milestone spiderweb tracker",      _milestone_check)
_r("bw-estate-planner",       SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Estate planning coordination",        _noop)
_r("bw-referral-network",     SkillAgent.AZ_SUPREME,  SkillCadence.ON_DEMAND,    "Professional referral network",       _noop)
_r("bw-insider-tracker",      SkillAgent.BIGBRAIN,    SkillCadence.SCHEDULED,   "SEC Form 4 insider tracking",         _noop,                cron="0 17 * * 1-5")

# ─── OPTIONS DEEP DIVE (7) ─────────────────────────────────────────────────
_r("bw-gamma-exposure",       SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "Dealer GEX + flip levels",            _flow_analysis)
_r("bw-wheel-strategy",       SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "Wheel strategy (CSP -> CC)",          _greeks_put)
_r("bw-zero-dte",             SkillAgent.TRADING,     SkillCadence.ON_DEMAND,    "0DTE gamma engine",                   _flow_analysis)
_r("bw-vol-arb",              SkillAgent.WAR_ROOM,    SkillCadence.ON_DEMAND,    "Volatility arbitrage + regime",       _regime_evaluate)
_r("bw-iv-crush",             SkillAgent.TRADING,     SkillCadence.EVENT_DRIVEN, "IV crush + earnings",                 _greeks_put)
_r("bw-greeks-portfolio",     SkillAgent.RISK,        SkillCadence.ON_DEMAND,    "Portfolio-level Greeks and risk",      _greeks_put)
_r("bw-options-strategy-engine", SkillAgent.WAR_ROOM, SkillCadence.ON_DEMAND,    "20+ strategy builder + scanner",      _stock_forecast)

# ─── CRYPTO DEEP DIVE (6) ──────────────────────────────────────────────────
_r("bw-onchain-metrics",      SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "MVRV, SOPR, NUPL, NVT",              _crypto_prices)
_r("bw-mev-protect",          SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "MEV protection + Flashbots",          _noop)
_r("bw-defi-yield",           SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "DeFi yield + IL calculator",          _crypto_prices)
_r("bw-whale-tracker",        SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Whale wallet tracking",               _crypto_prices)
_r("bw-funding-rates",        SkillAgent.CRYPTO,      SkillCadence.ON_DEMAND,    "Funding rate + OI divergence",        _crypto_prices)
_r("bw-liquidation-watch",    SkillAgent.CRYPTO,      SkillCadence.EVENT_DRIVEN, "Liquidation cascades + dominance",    _crypto_prices)

# ─── SYSTEM SKILLS (implicit, not in manifest) ─────────────────────────────
_r("_sys-live-scan",          SkillAgent.WAR_ROOM,    SkillCadence.SCHEDULED,   "5-min live feed scan",                _live_scan,           cron="*/5 * * * *", cooldown=60)
_r("_sys-regime-check",       SkillAgent.WAR_ROOM,    SkillCadence.SCHEDULED,   "5-min regime evaluation",             _regime_evaluate,     cron="*/5 * * * *", cooldown=60)
_r("_sys-monte-carlo",        SkillAgent.WAR_ROOM,    SkillCadence.SCHEDULED,   "Twice-daily MC simulation",           _monte_carlo,         cron="0 7,16 * * 1-5", cooldown=3600)
_r("_sys-halt",               SkillAgent.RISK,        SkillCadence.ON_DEMAND,    "Emergency halt — kill switch",        _halt_trading)
_r("_sys-resume",             SkillAgent.RISK,        SkillCadence.ON_DEMAND,    "Resume after halt",                   _resume_trading)
_r("_sys-dashboard",          SkillAgent.INFRA,       SkillCadence.ON_DEMAND,    "Master monitoring dashboard",         _dashboard_status)


# ─── Bridge Class ───────────────────────────────────────────────────────────

class SkillRuntimeBridge:
    """
    The execution bridge between OpenClaw skill invocations and AAC's Python
    runtime.  Called by the gateway bridge / AZ SUPREME handler when a skill
    needs to execute.

    Usage::

        bridge = SkillRuntimeBridge()
        result = await bridge.execute("bw-market-intelligence", {"theater": "B"})
        # result → {"sentiment": {...}, "recommendations": [...]}
    """

    def __init__(self) -> None:
        self._registry = dict(_REGISTRY)
        self._execution_log: list[dict[str, Any]] = []

    # ── Public API ──────────────────────────────────────────────────────

    async def execute(
        self,
        skill_id: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a skill and return the result dict.

        Raises ``KeyError`` if skill_id is not registered.
        Enforces cooldown between invocations.
        """
        binding = self._registry.get(skill_id)
        if binding is None:
            raise KeyError(f"Unknown skill: {skill_id}")
        if binding.handler is None:
            return {"error": f"Skill {skill_id} has no handler"}

        # Cooldown enforcement
        now = time.monotonic()
        if binding.cooldown_sec and binding._last_run is not None:
            elapsed = now - binding._last_run
            if elapsed < binding.cooldown_sec:
                return {
                    "throttled": True,
                    "retry_after_sec": round(binding.cooldown_sec - elapsed, 1),
                }

        # Execute
        t0 = time.monotonic()
        try:
            result = await binding.handler(params or {})
        except Exception as exc:
            logger.error("skill.execute.error", extra={
                "skill": skill_id, "error": str(exc),
            })
            result = {"error": str(exc)}
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        binding._last_run = time.monotonic()

        # Log
        entry = {
            "skill_id": skill_id,
            "agent": binding.agent.value,
            "cadence": binding.cadence.value,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now().isoformat(),
            "ok": "error" not in result,
        }
        self._execution_log.append(entry)
        if len(self._execution_log) > 500:
            self._execution_log = self._execution_log[-250:]

        logger.info("skill.executed", extra=entry)
        return result

    def resolve(self, skill_id: str) -> Optional[SkillBinding]:
        """Look up a skill binding by ID."""
        return self._registry.get(skill_id)

    def resolve_by_text(self, text: str) -> Optional[SkillBinding]:
        """Fuzzy-resolve a skill from natural-language text.

        Matches against skill_id and description keywords.
        """
        lower = text.lower()
        best: Optional[SkillBinding] = None
        best_score = 0
        for binding in self._registry.values():
            score = 0
            # Exact skill ID match
            if binding.skill_id in lower:
                score += 10
            # Keyword overlap
            desc_words = set(binding.description.lower().split())
            query_words = set(lower.split())
            overlap = len(desc_words & query_words)
            score += overlap
            if score > best_score:
                best_score = score
                best = binding
        return best if best_score >= 2 else None

    def list_skills(
        self,
        agent: Optional[SkillAgent] = None,
        cadence: Optional[SkillCadence] = None,
    ) -> list[SkillBinding]:
        """List skills, optionally filtered by agent or cadence."""
        out = list(self._registry.values())
        if agent:
            out = [b for b in out if b.agent == agent]
        if cadence:
            out = [b for b in out if b.cadence == cadence]
        return out

    def get_scheduled_skills(self) -> list[SkillBinding]:
        """Return all SCHEDULED skills with their cron expressions."""
        return [
            b for b in self._registry.values()
            if b.cadence == SkillCadence.SCHEDULED and b.cron
        ]

    def get_event_driven_skills(self) -> list[SkillBinding]:
        """Return all EVENT_DRIVEN skills (triggered by system events)."""
        return [
            b for b in self._registry.values()
            if b.cadence == SkillCadence.EVENT_DRIVEN
        ]

    def get_execution_log(self, last_n: int = 20) -> list[dict[str, Any]]:
        """Return recent execution log entries."""
        return self._execution_log[-last_n:]

    @property
    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        bindings = list(self._registry.values())
        wired = sum(1 for b in bindings if b.handler is not _noop)
        return {
            "total_skills": len(bindings),
            "wired_to_runtime": wired,
            "pending_noop": len(bindings) - wired,
            "by_agent": {
                a.value: sum(1 for b in bindings if b.agent == a)
                for a in SkillAgent
            },
            "by_cadence": {
                c.value: sum(1 for b in bindings if b.cadence == c)
                for c in SkillCadence
            },
            "scheduled_crons": [
                {"skill": b.skill_id, "cron": b.cron}
                for b in bindings
                if b.cadence == SkillCadence.SCHEDULED and b.cron
            ],
        }


# ─── Module-level singleton ────────────────────────────────────────────────

_bridge_instance: Optional[SkillRuntimeBridge] = None


def get_skill_bridge() -> SkillRuntimeBridge:
    """Get or create the singleton SkillRuntimeBridge."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = SkillRuntimeBridge()
    return _bridge_instance

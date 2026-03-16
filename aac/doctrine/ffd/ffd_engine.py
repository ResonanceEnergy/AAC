"""
FFD — Future Financial Doctrine Engine
=======================================

Pack 11: Future Financial Doctrine (FFD)
Monitors the global monetary transition across three tracks:
  Track 1 — Decentralized protocols (BTC, XRP, FLR, ETH, SOL)
  Track 2 — Private digital money (stablecoins, tokenized assets)
  Track 3 — Sovereign digital money (CBDCs, programmable fiat)

Integrates with the existing DoctrineEngine as a department adapter
following the same pattern as CryptoIntelligenceDoctrineAdapter.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("FFDEngine")


# ═══════════════════════════════════════════════════════════════════════════
# FFD ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class FFDTrack(Enum):
    """The three tracks of the global monetary transition."""
    DECENTRALIZED = "Track1_Decentralized"
    PRIVATE_DIGITAL = "Track2_PrivateDigital"
    SOVEREIGN_DIGITAL = "Track3_SovereignDigital"


class EvidenceLevel(Enum):
    """Research maturity levels for FFD findings."""
    E0_CONCEPTUAL = 0           # Thesis only
    E1_OBSERVED = 1             # Patterns observed, data gathered
    E2_BACKTESTED = 2           # Historical data validated
    E3_PAPER_TRADED = 3         # Paper trade confirmation
    E4_LIVE_SMALL = 4           # Small live allocation
    E5_LIVE_PRODUCTION = 5      # Full production deployment


class TransitionPhase(Enum):
    """AAC's strategic positioning phases per FFD Overview."""
    PHASE_1_INTELLIGENCE = "Intelligence_Gathering"   # Now
    PHASE_2_STRATEGY = "Strategy_Development"          # Next
    PHASE_3_DEPLOYMENT = "Full_Deployment"             # When evidence >= E3


class StablecoinHealth(Enum):
    """Stablecoin peg health status."""
    HEALTHY = "HEALTHY"           # Deviation < 0.5%
    STRESSED = "STRESSED"         # Deviation 0.5% - 2%
    CRITICAL = "CRITICAL"         # Deviation 2% - 5%
    DEPEGGED = "DEPEGGED"         # Deviation > 5%


# ═══════════════════════════════════════════════════════════════════════════
# FFD DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StablecoinPegStatus:
    """Real-time peg monitoring for a stablecoin."""
    symbol: str                     # USDT, USDC, RLUSD, etc.
    current_price: float            # Actual price vs peg
    peg_target: float               # Usually 1.0 for USD
    deviation_pct: float            # Abs percentage from peg
    health: StablecoinHealth
    issuer: str
    market_cap_usd: float
    volume_24h_usd: float
    last_checked: datetime = field(default_factory=datetime.now)


@dataclass
class CBDCSignal:
    """Intelligence signal from CBDC/sovereign digital money developments."""
    jurisdiction: str               # "EU", "China", "India", etc.
    event_type: str                 # "pilot_launch", "legislation", "ban", etc.
    description: str
    market_impact_estimate: str     # "low", "medium", "high", "critical"
    timestamp: datetime = field(default_factory=datetime.now)
    evidence_level: EvidenceLevel = EvidenceLevel.E0_CONCEPTUAL


@dataclass
class RegulatoryEvent:
    """Regulatory catalyst relevant to FFD strategies."""
    jurisdiction: str
    legislation: str                # "GENIUS_Act", "MiCAR", "CSRC_Ban", etc.
    status: str                     # "proposed", "committee", "passed", "enforced"
    impact_tracks: List[FFDTrack] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FFDMetrics:
    """Aggregated FFD metrics for doctrine compliance."""
    # Track 1 — Decentralized
    btc_hash_ribbon_signal: str = "neutral"     # "buy", "sell", "neutral"
    xrp_escrow_release_pct: float = 0.0         # % of monthly release
    flr_staking_yield: float = 0.0              # Current Flare staking APY
    cross_chain_settlement_score: float = 0.0   # 0-100 composite

    # Track 2 — Private Digital
    stablecoin_peg_health: float = 100.0        # 0-100 composite across monitored stablecoins
    stablecoin_concentration: float = 0.0       # % in largest single stablecoin
    defi_yield_sustainability: float = 0.0      # 0-100 sustainability score
    rwa_tvl_growth: float = 0.0                 # 30-day RWA TVL change %

    # Track 3 — Sovereign Digital
    cbdc_pilot_count: int = 0                   # Active CBDC pilots globally
    regulatory_shock_score: float = 0.0         # 0-100 based on recent events
    capital_flight_signal: float = 0.0          # 0-100 based on on-chain flows

    # Cross-track
    monetary_transition_index: float = 0.0      # 0-100 composite of all tracks

    # Expanded metrics (Pack 11 deep-dive expansion)
    etf_net_inflow_daily: float = 0.0           # Daily net inflows across all crypto ETFs (USD)
    memecoin_sentiment_index: float = 0.0       # 0-100 memecoin social momentum
    political_token_risk: float = 0.0           # 0-100 risk from politically-affiliated tokens
    xrp_etf_probability: float = 0.0            # 0-100 market-implied XRP ETF approval odds
    sol_network_uptime: float = 100.0           # 0-100 Solana uptime (100 = no outages)
    eth_l2_tvl_ratio: float = 0.0               # L2 TVL as % of L1 TVL
    defi_tvl_total: float = 0.0                 # Total DeFi TVL across all chains (USD)
    x_platform_sentiment: float = 50.0          # 0-100 aggregate financial sentiment from X
    halving_cycle_position: float = 0.0         # 0-1 normalized position in BTC halving cycle
    regulatory_convergence_score: float = 0.0   # 0-100 major-economy regulatory alignment

    # Expanded metrics (50-topic research integration)
    pqc_migration_urgency: float = 0.0          # 0-100 post-quantum crypto migration urgency
    brics_dedollarization_index: float = 0.0    # 0-100 BRICS de-dollarization progress
    cbdc_retail_active_count: int = 0            # Active retail CBDCs globally
    gold_btc_ratio: float = 0.0                 # Gold price / BTC price (oz/$)
    usd_reserve_share: float = 57.0             # USD share of global reserves (%)
    l2_fee_index: float = 0.0                   # Average L2 tx fee in cents
    mev_extraction_daily: float = 0.0           # Est daily MEV extraction (USD)
    restaking_tvl: float = 0.0                  # EigenLayer + restaking TVL (USD)
    tokenized_treasury_aum: float = 0.0         # Tokenized gov securities AUM (USD)
    prediction_market_volume: float = 0.0       # Daily prediction market volume (USD)
    global_debt_gdp_ratio: float = 0.0          # Global debt-to-GDP percentage
    micar_compliance_score: float = 0.0         # 0-100 AAC MiCAR readiness

    # Master Plan metrics (FFD-11 strategic execution tracking)
    portfolio_nominal_usd: float = 0.0          # Total portfolio value (USD nominal)
    portfolio_purchasing_power: float = 0.0     # Portfolio value adjusted for CPI/M2
    portfolio_gold_ratio: float = 0.0           # Portfolio value / gold oz (real wealth)
    stablecoin_yield_monthly: float = 0.0       # Monthly yield from stablecoin deployments (USD)
    arb_profit_monthly: float = 0.0             # Monthly cross-exchange arb profits (USD)
    strategy_count_live: int = 0                # Number of strategies at E4+ (live)
    strategy_count_paper: int = 0               # Number of strategies at E2-E3 (paper)
    capital_injected_total: float = 0.0         # Total capital injected since inception (USD)
    portfolio_drawdown_pct: float = 0.0         # Current drawdown from portfolio ATH (%)
    cycle_phase: str = "expansion"              # "accumulation", "expansion", "peak", "correction"

    # No Limits Framework metrics (FFD-12)
    leverage_utilization_pct: float = 0.0       # Current leverage used vs max allowed (%)
    funding_rate_income_monthly: float = 0.0    # Monthly income from funding rate strategies (USD)
    options_premium_income_monthly: float = 0.0  # Monthly income from options selling (USD)
    defi_yield_monthly: float = 0.0             # Monthly DeFi yield income (USD)
    active_account_count: int = 4               # Number of active trading accounts
    active_jurisdiction: str = "Canada"         # Current operating jurisdiction
    uruguay_relocation_days: int = 0            # Days until Uruguay relocation (countdown)

    # Integrated asset metrics (asset audit integration)
    prediction_market_pnl: float = 0.0          # Cumulative prediction market P&L (USD) — PlanktonXD
    options_flow_signal_count: int = 0           # Unusual Whales options flow signals received today
    gold_silver_ratio: float = 0.0              # Gold price / Silver price (oz/oz)
    silver_price_oz: float = 0.0                # Silver spot price per oz (USD)
    vce_signal_active: bool = False             # PU Prime VCE signal currently active
    superstonk_ftd_alert_count: int = 0          # SuperStonk FTD cycle alerts pending

    evidence_level: EvidenceLevel = EvidenceLevel.E0_CONCEPTUAL
    phase: TransitionPhase = TransitionPhase.PHASE_1_INTELLIGENCE

    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# FFD DOCTRINE PACK DEFINITION (Pack 11)
# ═══════════════════════════════════════════════════════════════════════════

FFD_DOCTRINE_PACK: Dict[str, Any] = {
    "name": "Future Financial Doctrine — Monetary Transition Intelligence",
    "owner": "FFD",
    "key_metrics": [
        "stablecoin_peg_health",
        "monetary_transition_index",
        "regulatory_shock_score",
        "capital_flight_signal",
        "cross_chain_settlement_score",
        "defi_yield_sustainability",
        "etf_net_inflow_daily",
        "defi_tvl_total",
        "x_platform_sentiment",
        "halving_cycle_position",
        "regulatory_convergence_score",
        "political_token_risk",
        "pqc_migration_urgency",
        "brics_dedollarization_index",
        "gold_btc_ratio",
        "usd_reserve_share",
        "tokenized_treasury_aum",
        "global_debt_gdp_ratio",
        "micar_compliance_score",
        "portfolio_nominal_usd",
        "portfolio_purchasing_power",
        "portfolio_gold_ratio",
        "stablecoin_yield_monthly",
        "arb_profit_monthly",
        "portfolio_drawdown_pct",
        "cycle_phase",
        "prediction_market_pnl",
        "options_flow_signal_count",
        "gold_silver_ratio",
        "silver_price_oz",
        "vce_signal_active",
        "superstonk_ftd_alert_count",
    ],
    "required_metrics": [
        {
            "metric": "stablecoin_peg_health",
            "thresholds": {"good": ">90", "warning": "70-90", "critical": "<70"}
        },
        {
            "metric": "monetary_transition_index",
            "thresholds": {"good": ">20", "warning": "10-20", "critical": "<10"}
        },
        {
            "metric": "regulatory_shock_score",
            "thresholds": {"good": "<30", "warning": "30-60", "critical": ">60"}
        },
        {
            "metric": "capital_flight_signal",
            "thresholds": {"good": "<25", "warning": "25-50", "critical": ">50"}
        },
        {
            "metric": "cross_chain_settlement_score",
            "thresholds": {"good": ">80", "warning": "60-80", "critical": "<60"}
        },
        {
            "metric": "defi_yield_sustainability",
            "thresholds": {"good": ">70", "warning": "40-70", "critical": "<40"}
        },
    ],
    "failure_modes": [
        "stablecoin_systemic_depeg",
        "cbdc_launch_disruption",
        "regulatory_shock_cascade",
        "capital_flight_acceleration",
        "defi_yield_collapse",
        "quantum_cryptographic_threat",
        "flash_loan_exploit_cascade",
        "political_token_conflict_of_interest",
        "solana_network_outage",
        "etf_redemption_spiral",
        "ai_flash_crash_amplification",
        "cross_chain_bridge_hack",
        "brics_payment_system_fragmentation",
        "mev_sandwich_attack_losses",
        "restaking_cascade_slashing",
        "synthetic_stablecoin_depeg",
        "l2_sequencer_centralization_failure",
        "tokenized_treasury_liquidity_crisis",
    ],
    "strategies": {
        "track_1_decentralized": [
            "btc_halving_cycle_positioning",
            "xrp_regulatory_catalyst_trading",
            "xrp_escrow_release_trading",
            "flare_ftso_delegation_yield",
            "cross_l1_settlement_arbitrage",
            "etf_inflow_momentum_trading",
            "xrp_etf_catalyst_positioning",
            "sol_network_recovery_trading",
            "eth_l2_migration_flow_tracking",
            "memecoin_sentiment_momentum",
        ],
        "track_2_private_digital": [
            "stablecoin_peg_deviation_capture",
            "stablecoin_basis_trade",
            "cross_venue_stablecoin_arbitrage",
            "regulatory_catalyst_trading",
            "stablecoin_flow_signals",
            "rwa_tokenization_arb",
            "defi_yield_harvesting",
            "flash_loan_arbitrage",
            "institutional_flow_tracking",
            "political_token_risk_hedging",
        ],
        "track_3_sovereign_digital": [
            "cbdc_launch_window_trading",
            "cbdc_interest_rate_arbitrage",
            "capital_flight_detection",
            "privacy_premium_trading",
            "cross_cbdc_fx_arbitrage",
        ],
        "cross_track_expanded": [
            "x_sentiment_signal_trading",
            "social_momentum_event_driven",
            "geopolitical_bloc_divergence_arb",
            "tokenized_securities_early_mover",
            "ai_market_consensus_contrarian",
            "gold_btc_ratio_mean_reversion",
            "brics_dedollarization_hedge",
            "pqc_migration_risk_monitoring",
            "l2_fee_arbitrage_venue_selection",
            "mev_protected_execution",
            "restaking_yield_optimization",
            "prediction_market_signal_integration",
            "debt_cycle_hard_asset_allocation",
        ],
    },
    "allocation_guidance": {
        "decentralized": {"min_pct": 40, "max_pct": 50},
        "private_digital": {"min_pct": 20, "max_pct": 30},
        "legacy_to_new_arbitrage": {"min_pct": 20, "max_pct": 30},
        "cbdc_hedging": {"min_pct": 5, "max_pct": 10},
    },
    # FFD-12: No Limits Framework — 50 new strategies (complement existing)
    "no_limits_strategies": {
        "leveraged_crypto_derivatives": [
            "btc_perp_momentum_scalping",
            "eth_perp_basis_trade",
            "btc_options_event_straddle",
            "funding_rate_arbitrage",
            "btc_covered_call_income",
            "btc_put_selling_bull",
            "altcoin_perp_breakout",
            "btc_gamma_scalping",
            "crypto_vol_smile_arbitrage",
            "cross_exchange_perp_arbitrage",
        ],
        "leveraged_equity": [
            "mstr_call_options_btc_proxy",
            "coin_earnings_strangle",
            "leveraged_etf_momentum_swing",
            "micro_futures_scalping",
            "spy_0dte_morning_momentum",
            "sector_rotation_options",
            "cad_usd_weakness_hedge",
            "miner_options_earnings_play",
        ],
        "defi_leverage_yield": [
            "aave_recursive_lending",
            "gmx_onchain_perps",
            "dydx_perps_governance",
            "pendle_yield_tokenization",
            "eigenlayer_restaking_optimization",
            "ethena_susde_yield",
            "uniswap_concentrated_lp",
            "morpho_optimized_lending",
            "flash_loan_arbitrage_automated",
        ],
        "volatility_harvesting": [
            "vix_term_structure_trading",
            "crypto_iv_rv_vol_trade",
            "weekend_gap_exploitation",
            "earnings_iv_crush_systematic",
            "halving_cycle_vol_enhancement",
            "fx_vol_brics_events",
            "crypto_weekend_vol_premium",
            "correlation_breakdown_alpha",
        ],
        "cross_jurisdiction_arbitrage": [
            "cad_usd_crypto_premium_arb",
            "tfsa_uruguay_tax_double_shield",
            "zona_franca_corporate_structure",
            "multi_exchange_stablecoin_rate_arb",
            "ndax_liquidity_latency_arb",
            "el_salvador_btc_residency_arb",
            "crypto_gold_fiat_rotation",
        ],
        "speed_automation": [
            "ai_signal_multi_model_consensus",
            "automated_grid_trading",
            "copy_trading_network_intel",
            "mev_protected_execution_capture",
            "cross_dex_routing_optimization",
            "automated_yield_farming_rotation",
            "vol_adjusted_dca",
            "capital_cockpit_dashboard",
        ],
    },
    # Integrated assets discovered during asset audit
    "integrated_assets": {
        "planktonxd_prediction_harvester": {
            "module": "strategies/planktonxd_prediction_harvester.py",
            "class": "PlanktonXDPredictionHarvester",
            "role": "Prediction market harvesting — Polymarket deep OTM + spread making",
            "tier": 2,
            "revenue_stream": 5,
        },
        "puprime_vce_module": {
            "module": "modules/aac_puprime_vce/",
            "class": "VCEStrategy",
            "role": "Volatility Compression→Expansion on XAUUSD/EURUSD/BTCUSD via MT5",
            "tier": 2,
            "revenue_stream": 7,
        },
        "unusual_whales_client": {
            "module": "integrations/unusual_whales_client.py",
            "class": "UnusualWhalesClient",
            "role": "Options flow, dark pool, Congress trades intelligence",
            "tier": 2,
            "revenue_stream": 6,
        },
        "superstonk_dd_engine": {
            "module": "reddit/reddit_scraper.py",
            "class": None,
            "role": "FTD cycle T+35, short interest, dark pool volume from r/Superstonk",
            "tier": 2,
            "revenue_stream": None,
        },
        "jonny_bravo_division": {
            "module": "agent_jonny_bravo_division/jonny_bravo_agent.py",
            "class": "JonnyBravoAgent",
            "role": "Education + methodology (supply/demand, order flow, fibonacci)",
            "tier": 3,
            "revenue_stream": None,
        },
        "silver_precious_metals": {
            "module": "modules/aac_puprime_vce/",
            "class": None,
            "role": "Silver (XAGUSD/PSLV) alongside gold — 2% portfolio target at $1M",
            "tier": 2,
            "revenue_stream": 7,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# STABLECOIN MONITOR
# ═══════════════════════════════════════════════════════════════════════════

# Reference stablecoins to monitor
MONITORED_STABLECOINS: Dict[str, Dict[str, Any]] = {
    "USDT": {"issuer": "Tether", "peg": 1.0, "tier": "critical"},
    "USDC": {"issuer": "Circle", "peg": 1.0, "tier": "primary"},
    "RLUSD": {"issuer": "Ripple", "peg": 1.0, "tier": "emerging"},
    "EURC": {"issuer": "Circle", "peg": 1.0, "tier": "emerging"},    # EUR peg
    "BUSD": {"issuer": "Paxos", "peg": 1.0, "tier": "legacy"},
    "USD1": {"issuer": "WLFI", "peg": 1.0, "tier": "political"},
    "PYUSD": {"issuer": "PayPal/Paxos", "peg": 1.0, "tier": "emerging"},
    "USDe": {"issuer": "Ethena", "peg": 1.0, "tier": "synthetic"},
    "FDUSD": {"issuer": "FirstDigital", "peg": 1.0, "tier": "caution"},
}

# Kill switch thresholds
DEPEG_ALERT_THRESHOLD = 0.5     # Alert at 0.5% deviation
DEPEG_EXIT_THRESHOLD = 2.0      # Auto-exit at 2% deviation
DEPEG_HALT_THRESHOLD = 5.0      # Kill switch at 5% deviation (Tether-specific)


# ═══════════════════════════════════════════════════════════════════════════
# SEED CAPITAL — REAL ACCOUNTS (FFD-11 Master Plan v3.0 — No Limits Edition)
# ═══════════════════════════════════════════════════════════════════════════

SEED_CAPITAL: Dict[str, Dict[str, Any]] = {
    "NDAX": {
        "initial_usd": 3800.0,
        "currency": "CAD",
        "connector": "NDAXConnector",
        "role": "PRIMARY crypto execution — BTC/ETH/XRP spot",
        "priority": "P0",
        "tax_status": "taxable",
        "max_leverage": 1.0,
        "jurisdiction": "Canada",
    },
    "IBKR": {
        "initial_usd": 1000.0,
        "currency": "CAD",
        "connector": "IBKRConnector",
        "role": "Options, futures, equities, FX, leveraged ETFs",
        "priority": "P0",
        "tax_status": "taxable",
        "max_leverage": 5.0,
        "jurisdiction": "Canada",
    },
    "Moomoo": {
        "initial_usd": 1000.0,
        "currency": "CAD",
        "connector": "MoomooConnector",
        "role": "Crypto-adjacent equities — MSTR, COIN, RIOT",
        "priority": "P1",
        "tax_status": "taxable",
        "max_leverage": 3.0,
        "jurisdiction": "Canada",
    },
    "TFSA_Wealthsimple": {
        "initial_usd": 3000.0,
        "currency": "CAD",
        "connector": None,
        "role": "TAX-FREE high-growth — BTC/ETH ETFs (BTCC/ETHX)",
        "priority": "P1",
        "tax_status": "tax_free",
        "max_leverage": 1.0,
        "jurisdiction": "Canada",
    },
    "Deribit": {
        "initial_usd": 0.0,
        "currency": "BTC",
        "connector": None,
        "role": "BTC/ETH options + perpetual futures — primary derivatives",
        "priority": "P0",
        "tax_status": "taxable",
        "max_leverage": 10.0,
        "jurisdiction": "Netherlands",
    },
    "OKX": {
        "initial_usd": 0.0,
        "currency": "USD",
        "connector": None,
        "role": "Perps, options, earn, grid trading",
        "priority": "P0",
        "tax_status": "taxable",
        "max_leverage": 10.0,
        "jurisdiction": "Seychelles",
    },
    "Bybit": {
        "initial_usd": 0.0,
        "currency": "USD",
        "connector": None,
        "role": "Perps, copy trading, earn products",
        "priority": "P1",
        "tax_status": "taxable",
        "max_leverage": 10.0,
        "jurisdiction": "Dubai",
    },
}

TOTAL_SEED_CAPITAL: float = sum(float(a["initial_usd"]) for a in SEED_CAPITAL.values())  # $8,800

# Performance-based milestones (M1-M7)
MILESTONES = {
    "M1_IGNITION": 15_000,
    "M2_TRACTION": 25_000,
    "M3_VELOCITY": 50_000,
    "M4_MOMENTUM": 100_000,
    "M5_ACCELERATION": 250_000,
    "M6_ORBIT": 500_000,
    "M7_DESTINATION": 1_000_000,
}


# ═══════════════════════════════════════════════════════════════════════════
# JURISDICTION CONFIGURATION — No Limits Framework (FFD-12)
# ═══════════════════════════════════════════════════════════════════════════

JURISDICTION_CONFIG = {
    "Canada": {
        "status": "current",
        "capital_gains_inclusion": 0.50,  # 50% of gains taxable
        "tfsa_available": True,
        "notes": "Primary until Uruguay relocation (~8 months)",
    },
    "Uruguay": {
        "status": "target_primary",
        "capital_gains_inclusion": 0.0,  # Territorial taxation — foreign-sourced = 0%
        "zona_franca_available": True,
        "regulator": "BCU",
        "regulation": "Circular 2377 (VASP)",
        "dtc_with_canada": True,
        "notes": "PRIMARY base after relocation — 0% tax on global trading income",
    },
    "El_Salvador": {
        "status": "backup",
        "capital_gains_btc": 0.0,  # reportedly still 0% on BTC transactions
        "btc_legal_tender": False,  # RESCINDED Feb 2025 (IMF $1.4B loan conditions)
        "residency_btc_threshold": 3.0,  # ₿3 for permanent residency
        "notes": "Backup jurisdiction only — BTC legal tender rescinded Feb 2025",
    },
}

# Leverage limits per account type (hard constraints from Section 10)
LEVERAGE_LIMITS = {
    "crypto_spot": 1.0,
    "crypto_perps": 10.0,
    "equity_options": None,  # Defined risk (premium = max loss)
    "micro_futures": 5.0,
    "fx_micro": 3.0,
    "defi_recursive": 3.0,
    "leveraged_etf": 3.0,  # Built-in, no additional margin
}

# Hard risk rules (never violated)
RISK_HARD_RULES = {
    "max_risk_per_trade_pct": 2.0,
    "max_single_exchange_pct": 40.0,
    "min_exchange_distribution": 3,
    "portfolio_drawdown_halt_pct": 25.0,
    "daily_loss_halt_pct": 5.0,
    "weekly_loss_halt_pct": 10.0,
    "isolated_margin_only": True,
    "naked_options_allowed": False,
}


class StablecoinMonitor:
    """Monitors stablecoin peg health across all tracked stablecoins."""

    def __init__(self):
        self.statuses: Dict[str, StablecoinPegStatus] = {}
        self._alert_history: List[Dict[str, Any]] = []

    def update_price(self, symbol: str, price: float, market_cap: float = 0.0, volume: float = 0.0):
        """Update price for a monitored stablecoin and evaluate health."""
        config = MONITORED_STABLECOINS.get(symbol)
        if not config:
            logger.warning(f"Unknown stablecoin: {symbol}")
            return

        peg = float(config["peg"])
        deviation = abs(price - peg) / peg * 100.0

        if deviation < DEPEG_ALERT_THRESHOLD:
            health = StablecoinHealth.HEALTHY
        elif deviation < DEPEG_EXIT_THRESHOLD:
            health = StablecoinHealth.STRESSED
        elif deviation < DEPEG_HALT_THRESHOLD:
            health = StablecoinHealth.CRITICAL
        else:
            health = StablecoinHealth.DEPEGGED

        self.statuses[symbol] = StablecoinPegStatus(
            symbol=symbol,
            current_price=price,
            peg_target=peg,
            deviation_pct=deviation,
            health=health,
            issuer=str(config["issuer"]),
            market_cap_usd=market_cap,
            volume_24h_usd=volume,
        )

        if health != StablecoinHealth.HEALTHY:
            alert = {
                "symbol": symbol,
                "deviation_pct": deviation,
                "health": health.value,
                "timestamp": datetime.now().isoformat(),
            }
            self._alert_history.append(alert)
            logger.warning(f"FFD ALERT: {symbol} peg deviation {deviation:.2f}% — {health.value}")

    def get_composite_health(self) -> float:
        """Composite peg health score (0-100). 100 = all stable."""
        if not self.statuses:
            return 100.0  # No data = assume healthy (no news is good news)

        scores = []
        for status in self.statuses.values():
            if status.health == StablecoinHealth.HEALTHY:
                scores.append(100.0)
            elif status.health == StablecoinHealth.STRESSED:
                scores.append(70.0)
            elif status.health == StablecoinHealth.CRITICAL:
                scores.append(30.0)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores)

    def should_halt(self) -> bool:
        """Check if Tether depeg triggers system-wide halt."""
        usdt = self.statuses.get("USDT")
        if usdt and usdt.health == StablecoinHealth.DEPEGGED:
            logger.critical("FFD KILL SWITCH: USDT depegged >5% — recommending HALT")
            return True
        # Also halt if 3+ stablecoins critical simultaneously
        critical_count = sum(
            1 for s in self.statuses.values()
            if s.health in (StablecoinHealth.CRITICAL, StablecoinHealth.DEPEGGED)
        )
        if critical_count >= 3:
            logger.critical(f"FFD KILL SWITCH: {critical_count} stablecoins critical — recommending HALT")
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# FFD ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class FFDEngine:
    """
    Future Financial Doctrine Engine.

    Monitors the global monetary transition and provides intelligence,
    strategy signals, and risk alerts across all three tracks.
    """

    def __init__(self):
        self.stablecoin_monitor = StablecoinMonitor()
        self.metrics = FFDMetrics()
        self.cbdc_signals: List[CBDCSignal] = []
        self.regulatory_events: List[RegulatoryEvent] = []
        self._initialized = False
        # Seed capital tracking — per-account balances
        self.account_balances: Dict[str, float] = {
            name: acct["initial_usd"] for name, acct in SEED_CAPITAL.items()
        }
        self.metrics.capital_injected_total = TOTAL_SEED_CAPITAL
        self.metrics.portfolio_nominal_usd = TOTAL_SEED_CAPITAL

    def initialize(self) -> bool:
        """Initialize the FFD engine with baseline data."""
        try:
            self._load_regulatory_baseline()
            self._initialized = True
            logger.info("FFD Engine initialized — Phase 1: Intelligence Gathering")
            return True
        except Exception as e:
            logger.error(f"FFD initialization failed: {e}")
            return False

    def _load_regulatory_baseline(self):
        """Load known regulatory events as baseline intelligence."""
        baseline_events = [
            RegulatoryEvent(
                jurisdiction="US",
                legislation="GENIUS_Act",
                status="enforced",
                impact_tracks=[FFDTrack.PRIVATE_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="EU",
                legislation="MiCAR",
                status="enforced",
                impact_tracks=[FFDTrack.PRIVATE_DIGITAL, FFDTrack.SOVEREIGN_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="US",
                legislation="Strategic_Crypto_Reserve",
                status="enforced",
                impact_tracks=[FFDTrack.DECENTRALIZED],
            ),
            RegulatoryEvent(
                jurisdiction="China",
                legislation="Stablecoin_Ban",
                status="enforced",
                impact_tracks=[FFDTrack.PRIVATE_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="Hong_Kong",
                legislation="Stablecoin_Bill",
                status="enforced",
                impact_tracks=[FFDTrack.PRIVATE_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="Japan",
                legislation="Yen_Stablecoin_Framework",
                status="enforced",
                impact_tracks=[FFDTrack.PRIVATE_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="EU",
                legislation="Digital_Euro_Legislation",
                status="proposed",
                impact_tracks=[FFDTrack.SOVEREIGN_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="US",
                legislation="DTCC_Tokenized_Equities_NoAction",
                status="enforced",
                impact_tracks=[FFDTrack.PRIVATE_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="US",
                legislation="FedNow_Launch",
                status="enforced",
                impact_tracks=[FFDTrack.SOVEREIGN_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="India",
                legislation="Digital_Rupee_Pilot",
                status="enforced",
                impact_tracks=[FFDTrack.SOVEREIGN_DIGITAL],
            ),
            RegulatoryEvent(
                jurisdiction="Brazil",
                legislation="DREX_CBDC_Pilot",
                status="committee",
                impact_tracks=[FFDTrack.SOVEREIGN_DIGITAL],
            ),
        ]
        self.regulatory_events.extend(baseline_events)

    def add_cbdc_signal(self, jurisdiction: str, event_type: str, description: str,
                        impact: str = "medium"):
        """Record a CBDC/sovereign digital money intelligence signal."""
        signal = CBDCSignal(
            jurisdiction=jurisdiction,
            event_type=event_type,
            description=description,
            market_impact_estimate=impact,
        )
        self.cbdc_signals.append(signal)
        logger.info(f"FFD CBDC Signal: [{jurisdiction}] {event_type} — {impact} impact")

    def add_regulatory_event(self, jurisdiction: str, legislation: str, status: str,
                             tracks: Optional[List[FFDTrack]] = None):
        """Record a regulatory event affecting FFD strategies."""
        event = RegulatoryEvent(
            jurisdiction=jurisdiction,
            legislation=legislation,
            status=status,
            impact_tracks=tracks or [],
        )
        self.regulatory_events.append(event)
        logger.info(f"FFD Regulatory Event: [{jurisdiction}] {legislation} — {status}")

    def update_stablecoin_prices(self, prices: Dict[str, float]):
        """Batch update stablecoin prices. Keys are symbols (USDT, USDC, etc.)."""
        for symbol, price in prices.items():
            self.stablecoin_monitor.update_price(symbol, price)
        self.metrics.stablecoin_peg_health = self.stablecoin_monitor.get_composite_health()

    def compute_regulatory_shock_score(self) -> float:
        """
        Score 0-100 based on recent regulatory events.
        Higher = more regulatory turbulence.
        """
        recent = [
            e for e in self.regulatory_events
            if (datetime.now() - e.timestamp).days <= 30
        ]
        if not recent:
            return 0.0

        impact_weights = {"proposed": 5, "committee": 15, "passed": 30, "enforced": 10}
        score = sum(impact_weights.get(e.status, 5) for e in recent)
        return min(score, 100.0)

    def compute_monetary_transition_index(self) -> float:
        """
        Composite index (0-100) measuring how far the monetary transition
        has progressed. Higher = further along.
        """
        components = [
            self.metrics.stablecoin_peg_health * 0.2,
            min(self.metrics.cbdc_pilot_count * 2, 100) * 0.15,
            self.metrics.cross_chain_settlement_score * 0.2,
            self.metrics.defi_yield_sustainability * 0.15,
            self.metrics.rwa_tvl_growth * 0.1,
            (100 - self.metrics.regulatory_shock_score) * 0.1,
            (100 - self.metrics.capital_flight_signal) * 0.1,
        ]
        return sum(components)

    def get_metrics(self) -> Dict[str, float]:
        """Return FFD metrics in doctrine-compatible format."""
        self.metrics.regulatory_shock_score = self.compute_regulatory_shock_score()
        self.metrics.monetary_transition_index = self.compute_monetary_transition_index()

        return {
            "stablecoin_peg_health": self.metrics.stablecoin_peg_health,
            "monetary_transition_index": self.metrics.monetary_transition_index,
            "regulatory_shock_score": self.metrics.regulatory_shock_score,
            "capital_flight_signal": self.metrics.capital_flight_signal,
            "cross_chain_settlement_score": self.metrics.cross_chain_settlement_score,
            "defi_yield_sustainability": self.metrics.defi_yield_sustainability,
            "stablecoin_concentration": self.metrics.stablecoin_concentration,
            "cbdc_pilot_count": float(self.metrics.cbdc_pilot_count),
            "rwa_tvl_growth": self.metrics.rwa_tvl_growth,
            # Expanded metrics
            "etf_net_inflow_daily": self.metrics.etf_net_inflow_daily,
            "memecoin_sentiment_index": self.metrics.memecoin_sentiment_index,
            "political_token_risk": self.metrics.political_token_risk,
            "xrp_etf_probability": self.metrics.xrp_etf_probability,
            "sol_network_uptime": self.metrics.sol_network_uptime,
            "eth_l2_tvl_ratio": self.metrics.eth_l2_tvl_ratio,
            "defi_tvl_total": self.metrics.defi_tvl_total,
            "x_platform_sentiment": self.metrics.x_platform_sentiment,
            "halving_cycle_position": self.metrics.halving_cycle_position,
            "regulatory_convergence_score": self.metrics.regulatory_convergence_score,
            # 50-topic research metrics
            "pqc_migration_urgency": self.metrics.pqc_migration_urgency,
            "brics_dedollarization_index": self.metrics.brics_dedollarization_index,
            "cbdc_retail_active_count": float(self.metrics.cbdc_retail_active_count),
            "gold_btc_ratio": self.metrics.gold_btc_ratio,
            "usd_reserve_share": self.metrics.usd_reserve_share,
            "l2_fee_index": self.metrics.l2_fee_index,
            "mev_extraction_daily": self.metrics.mev_extraction_daily,
            "restaking_tvl": self.metrics.restaking_tvl,
            "tokenized_treasury_aum": self.metrics.tokenized_treasury_aum,
            "prediction_market_volume": self.metrics.prediction_market_volume,
            "global_debt_gdp_ratio": self.metrics.global_debt_gdp_ratio,
            "micar_compliance_score": self.metrics.micar_compliance_score,
            # Master Plan metrics
            "portfolio_nominal_usd": self.metrics.portfolio_nominal_usd,
            "portfolio_purchasing_power": self.metrics.portfolio_purchasing_power,
            "portfolio_gold_ratio": self.metrics.portfolio_gold_ratio,
            "stablecoin_yield_monthly": self.metrics.stablecoin_yield_monthly,
            "arb_profit_monthly": self.metrics.arb_profit_monthly,
            "strategy_count_live": float(self.metrics.strategy_count_live),
            "strategy_count_paper": float(self.metrics.strategy_count_paper),
            "capital_injected_total": self.metrics.capital_injected_total,
            "portfolio_drawdown_pct": self.metrics.portfolio_drawdown_pct,
            "cycle_phase": self.metrics.cycle_phase,
            # Integrated asset metrics
            "prediction_market_pnl": self.metrics.prediction_market_pnl,
            "options_flow_signal_count": float(self.metrics.options_flow_signal_count),
            "gold_silver_ratio": self.metrics.gold_silver_ratio,
            "silver_price_oz": self.metrics.silver_price_oz,
            "vce_signal_active": float(self.metrics.vce_signal_active),
            "superstonk_ftd_alert_count": float(self.metrics.superstonk_ftd_alert_count),
        }

    def get_active_strategies(self) -> Any:
        """Return currently active FFD strategies per track."""
        return FFD_DOCTRINE_PACK["strategies"]

    def get_allocation_guidance(self) -> Any:
        """Return recommended allocation targets per track."""
        return FFD_DOCTRINE_PACK["allocation_guidance"]

    def get_status_report(self) -> Dict[str, Any]:
        """Full FFD status report."""
        metrics = self.get_metrics()
        return {
            "engine": "FFDEngine",
            "phase": self.metrics.phase.value,
            "evidence_level": self.metrics.evidence_level.name,
            "initialized": self._initialized,
            "metrics": metrics,
            "stablecoin_statuses": {
                sym: {
                    "price": s.current_price,
                    "deviation_pct": round(s.deviation_pct, 4),
                    "health": s.health.value,
                }
                for sym, s in self.stablecoin_monitor.statuses.items()
            },
            "cbdc_signals_count": len(self.cbdc_signals),
            "regulatory_events_count": len(self.regulatory_events),
            "halt_recommended": self.stablecoin_monitor.should_halt(),
            "allocation_guidance": self.get_allocation_guidance(),
            "active_strategies": self.get_active_strategies(),
        }

    def should_trigger_state_change(self) -> Optional[str]:
        """
        Check if any FFD condition should trigger a BarrenWuffet state change.
        Returns state name or None.
        """
        if self.stablecoin_monitor.should_halt():
            return "HALT"

        if self.metrics.regulatory_shock_score > 60:
            return "SAFE_MODE"

        if self.metrics.capital_flight_signal > 50:
            return "CAUTION"

        if self.metrics.stablecoin_peg_health < 70:
            return "CAUTION"

        return None

    def compute_cycle_phase(self, halving_position: float) -> str:
        """
        Determine BTC halving cycle phase from normalized position (0.0-1.0).
        0.0 = halving day, 1.0 = next halving day.
        """
        if halving_position < 0.15:
            return "accumulation"      # 0-6 months post-halving: recovery begins
        elif halving_position < 0.50:
            return "expansion"         # 6-24 months: main bull run
        elif halving_position < 0.70:
            return "peak"              # 24-33 months: cycle top territory
        else:
            return "correction"        # 33-48 months: bear market / pre-halving

    def update_halving_position(self, position: float):
        """Update halving cycle position and derive cycle phase."""
        self.metrics.halving_cycle_position = max(0.0, min(1.0, position))
        self.metrics.cycle_phase = self.compute_cycle_phase(position)

    def update_portfolio(self, nominal_usd: float, gold_price_oz: float = 0.0,
                         cpi_adjustment: float = 1.0, silver_price_oz: float = 0.0):
        """Update portfolio tracking metrics."""
        self.metrics.portfolio_nominal_usd = nominal_usd
        if cpi_adjustment > 0:
            self.metrics.portfolio_purchasing_power = nominal_usd / cpi_adjustment
        if gold_price_oz > 0:
            self.metrics.portfolio_gold_ratio = nominal_usd / gold_price_oz
        if silver_price_oz > 0:
            self.metrics.silver_price_oz = silver_price_oz
        if gold_price_oz > 0 and silver_price_oz > 0:
            self.metrics.gold_silver_ratio = gold_price_oz / silver_price_oz

    def get_allocation_targets(self) -> Any:
        """
        Return dynamic allocation targets adjusted for cycle phase.
        In peak/correction phases, shift toward stablecoins and yield.
        In accumulation/expansion phases, shift toward spot appreciation.
        """
        base = FFD_DOCTRINE_PACK["allocation_guidance"]
        phase = self.metrics.cycle_phase

        if phase == "accumulation":
            return {
                "decentralized": {"min_pct": 50, "max_pct": 60},
                "private_digital": {"min_pct": 20, "max_pct": 25},
                "legacy_to_new_arbitrage": {"min_pct": 15, "max_pct": 20},
                "cbdc_hedging": {"min_pct": 5, "max_pct": 10},
            }
        elif phase == "expansion":
            return {
                "decentralized": {"min_pct": 45, "max_pct": 55},
                "private_digital": {"min_pct": 20, "max_pct": 25},
                "legacy_to_new_arbitrage": {"min_pct": 20, "max_pct": 25},
                "cbdc_hedging": {"min_pct": 5, "max_pct": 10},
            }
        elif phase == "peak":
            return {
                "decentralized": {"min_pct": 25, "max_pct": 35},
                "private_digital": {"min_pct": 35, "max_pct": 45},
                "legacy_to_new_arbitrage": {"min_pct": 20, "max_pct": 25},
                "cbdc_hedging": {"min_pct": 5, "max_pct": 10},
            }
        elif phase == "correction":
            return {
                "decentralized": {"min_pct": 30, "max_pct": 40},
                "private_digital": {"min_pct": 35, "max_pct": 45},
                "legacy_to_new_arbitrage": {"min_pct": 15, "max_pct": 25},
                "cbdc_hedging": {"min_pct": 5, "max_pct": 10},
            }
        return base

    def update_account_balance(self, account: str, balance_usd: float):
        """Update a specific account balance and recalculate portfolio total."""
        if account not in SEED_CAPITAL:
            logger.warning(f"Unknown account: {account}")
            return
        self.account_balances[account] = balance_usd
        self.metrics.portfolio_nominal_usd = sum(self.account_balances.values())

    def get_current_milestone(self) -> str:
        """Return the current milestone based on portfolio value."""
        total = self.metrics.portfolio_nominal_usd
        current = "PRE_M1"
        for name, target in MILESTONES.items():
            if total >= target:
                current = name
            else:
                break
        return current

    def get_next_milestone(self) -> Dict[str, Any]:
        """Return next milestone target and distance."""
        total = self.metrics.portfolio_nominal_usd
        for name, target in MILESTONES.items():
            if total < target:
                return {
                    "name": name,
                    "target": target,
                    "current": total,
                    "remaining": target - total,
                    "multiplier": target / total if total > 0 else float("inf"),
                }
        return {"name": "BEYOND_M7", "target": 0, "current": total, "remaining": 0, "multiplier": 1.0}

    def get_account_summary(self) -> Dict[str, Any]:
        """Return summary of all accounts with balances and roles."""
        summary = {}
        for name, config in SEED_CAPITAL.items():
            summary[name] = {
                "initial": config["initial_usd"],
                "current": self.account_balances.get(name, 0.0),
                "pnl": self.account_balances.get(name, 0.0) - float(config["initial_usd"]),
                "connector": config["connector"],
                "role": config["role"],
                "tax_status": config["tax_status"],
                "max_leverage": config.get("max_leverage", 1.0),
                "jurisdiction": config.get("jurisdiction", "Canada"),
            }
        summary["_total"] = {
            "initial": TOTAL_SEED_CAPITAL,
            "current": self.metrics.portfolio_nominal_usd,
            "pnl": self.metrics.portfolio_nominal_usd - TOTAL_SEED_CAPITAL,
            "milestone": self.get_current_milestone(),
            "next_milestone": self.get_next_milestone(),
        }
        return summary

    def get_jurisdiction_config(self, jurisdiction: str = "current") -> Dict[str, Any]:
        """Return configuration for a specific jurisdiction or current operating jurisdiction."""
        if jurisdiction == "current":
            jurisdiction = self.metrics.active_jurisdiction
        return JURISDICTION_CONFIG.get(jurisdiction, {})

    def get_leverage_limit(self, account_type: str) -> float:
        """Return max leverage allowed for a given account type."""
        limit = LEVERAGE_LIMITS.get(account_type)
        return limit if limit is not None else 1.0

    def check_risk_hard_rules(self, trade_risk_pct: float, exchange_pct: float) -> Dict[str, bool]:
        """Validate a proposed trade against hard risk rules."""
        return {
            "trade_risk_ok": trade_risk_pct <= RISK_HARD_RULES["max_risk_per_trade_pct"],
            "exchange_concentration_ok": exchange_pct <= RISK_HARD_RULES["max_single_exchange_pct"],
            "all_ok": (
                trade_risk_pct <= RISK_HARD_RULES["max_risk_per_trade_pct"]
                and exchange_pct <= RISK_HARD_RULES["max_single_exchange_pct"]
            ),
        }

    def get_no_limits_strategy_count(self) -> int:
        """Return total count of no-limits strategies defined in doctrine."""
        nl: Dict[str, list] = FFD_DOCTRINE_PACK.get("no_limits_strategies", {})
        return sum(len(v) for v in nl.values())  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════

_ffd_engine: Optional[FFDEngine] = None


def get_ffd_engine() -> FFDEngine:
    """Get or create the FFD engine singleton."""
    global _ffd_engine
    if _ffd_engine is None:
        _ffd_engine = FFDEngine()
        _ffd_engine.initialize()
    return _ffd_engine

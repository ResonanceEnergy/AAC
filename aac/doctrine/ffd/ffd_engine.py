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
    evidence_level: EvidenceLevel = EvidenceLevel.E0_CONCEPTUAL
    phase: TransitionPhase = TransitionPhase.PHASE_1_INTELLIGENCE

    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# FFD DOCTRINE PACK DEFINITION (Pack 11)
# ═══════════════════════════════════════════════════════════════════════════

FFD_DOCTRINE_PACK = {
    "name": "Future Financial Doctrine — Monetary Transition Intelligence",
    "owner": "FFD",
    "key_metrics": [
        "stablecoin_peg_health",
        "monetary_transition_index",
        "regulatory_shock_score",
        "capital_flight_signal",
        "cross_chain_settlement_score",
        "defi_yield_sustainability",
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
    ],
    "strategies": {
        "track_1_decentralized": [
            "btc_halving_cycle_positioning",
            "xrp_regulatory_catalyst_trading",
            "xrp_escrow_release_trading",
            "flare_ftso_delegation_yield",
            "cross_l1_settlement_arbitrage",
        ],
        "track_2_private_digital": [
            "stablecoin_peg_deviation_capture",
            "stablecoin_basis_trade",
            "cross_venue_stablecoin_arbitrage",
            "regulatory_catalyst_trading",
            "stablecoin_flow_signals",
            "rwa_tokenization_arb",
        ],
        "track_3_sovereign_digital": [
            "cbdc_launch_window_trading",
            "cbdc_interest_rate_arbitrage",
            "capital_flight_detection",
            "privacy_premium_trading",
            "cross_cbdc_fx_arbitrage",
        ],
    },
    "allocation_guidance": {
        "decentralized": {"min_pct": 40, "max_pct": 50},
        "private_digital": {"min_pct": 20, "max_pct": 30},
        "legacy_to_new_arbitrage": {"min_pct": 20, "max_pct": 30},
        "cbdc_hedging": {"min_pct": 5, "max_pct": 10},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# STABLECOIN MONITOR
# ═══════════════════════════════════════════════════════════════════════════

# Reference stablecoins to monitor
MONITORED_STABLECOINS = {
    "USDT": {"issuer": "Tether", "peg": 1.0, "tier": "critical"},
    "USDC": {"issuer": "Circle", "peg": 1.0, "tier": "primary"},
    "RLUSD": {"issuer": "Ripple", "peg": 1.0, "tier": "emerging"},
    "EURC": {"issuer": "Circle", "peg": 1.0, "tier": "emerging"},    # EUR peg
    "BUSD": {"issuer": "Paxos", "peg": 1.0, "tier": "legacy"},
}

# Kill switch thresholds
DEPEG_ALERT_THRESHOLD = 0.5     # Alert at 0.5% deviation
DEPEG_EXIT_THRESHOLD = 2.0      # Auto-exit at 2% deviation
DEPEG_HALT_THRESHOLD = 5.0      # Kill switch at 5% deviation (Tether-specific)


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

        peg = config["peg"]
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
            issuer=config["issuer"],
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
        }

    def get_active_strategies(self) -> Dict[str, List[str]]:
        """Return currently active FFD strategies per track."""
        return FFD_DOCTRINE_PACK["strategies"]

    def get_allocation_guidance(self) -> Dict[str, Dict[str, int]]:
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

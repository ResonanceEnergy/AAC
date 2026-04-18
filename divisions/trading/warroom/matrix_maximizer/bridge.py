"""
MATRIX MAXIMIZER — NCC / NCL / AAC Integration Bridge
========================================================
Deep integration layer connecting the options engine to all Resonance Energy pillars.

NCC (Command)  → Doctrine mode gates, governance directives, risk multiplier
NCL (Brain)    → Market intelligence feeds, forecast data, signal enrichment
AAC (Bank)     → Regime engine state, stock forecaster output, order execution
BRS (Bravo)    → Pattern recognition signals (optional)

Data Flow:
    NCC governance → Bridge → risk multiplier → RiskManager
    NCL intelligence → Bridge → scenario weight adjustments → MonteCarloEngine
    AAC RegimeEngine → Bridge → regime state → Scanner config
    MATRIX MAXIMIZER output → Bridge → CrossPillarHub → NCL/NCC sync
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from integrations.cross_pillar_hub import CrossPillarHub, CrossPillarState
from strategies.matrix_maximizer.core import (
    Asset,
    MatrixConfig,
    Scenario,
    ScenarioWeights,
)
from strategies.matrix_maximizer.risk import CircuitBreaker, RiskSnapshot

logger = logging.getLogger(__name__)


@dataclass
class RegimeContext:
    """Regime engine state relevant to MATRIX MAXIMIZER decisions."""
    regime: str = "uncertain"           # e.g. "vol_shock_armed", "risk_off"
    confidence: float = 0.0             # 0-100
    war_active: bool = False
    hormuz_blocked: bool = False
    oil_price: float = 96.5
    vix: float = 22.0
    hy_spread_bps: float = 350.0
    treasury_10y_2y: float = -0.1
    fear_greed: float = 35.0
    active_formulas: List[str] = field(default_factory=list)

    @property
    def is_crisis(self) -> bool:
        return self.regime in ("vol_shock_active", "credit_stress", "liquidity_crunch")

    @property
    def is_armed(self) -> bool:
        return self.regime in ("vol_shock_armed", "policy_delay_trap")


@dataclass
class PillarSync:
    """Data package pushed to other pillars after each cycle."""
    timestamp: str
    circuit_breaker: str
    mandate_level: str
    portfolio_delta: float
    exposure_pct: float
    top_picks: List[Dict[str, Any]]
    regime_context: Dict[str, Any]
    risk_checks_passed: int
    risk_checks_failed: int


class PillarBridge:
    """Integration bridge connecting MATRIX MAXIMIZER to the Resonance Energy ecosystem.

    Usage:
        bridge = PillarBridge(config)
        regime = bridge.get_regime_context()
        weights = bridge.adjust_scenario_weights(base_weights, regime)
        # ... run cycle ...
        bridge.push_results(snapshot, picks)
    """

    def __init__(self, config: MatrixConfig) -> None:
        self.config = config
        self._hub: Optional[CrossPillarHub] = None
        self._state: Optional[CrossPillarState] = None
        self._regime_cache: Optional[RegimeContext] = None
        self._init_hub()

    def _init_hub(self) -> None:
        """Initialize CrossPillarHub safely."""
        try:
            self._hub = CrossPillarHub()
            self._state = self._hub.state
            logger.info("PillarBridge connected — doctrine: %s", self._state.doctrine_mode)
        except Exception as exc:
            logger.warning("CrossPillarHub unavailable, running standalone: %s", exc)
            self._hub = None

    # ── NCC Governance ─────────────────────────────────────────────────

    def should_trade(self) -> bool:
        """Check NCC governance: is trading allowed?"""
        if self._hub is None:
            return True  # Standalone mode — no governance block
        return self._hub.should_trade()

    def get_risk_multiplier(self) -> float:
        """Get NCC doctrine risk multiplier (0.0 to 1.0)."""
        if self._hub is None:
            return 1.0
        return self._hub.get_risk_multiplier()

    def get_doctrine_mode(self) -> str:
        """Current NCC doctrine mode."""
        if self._state:
            return self._state.doctrine_mode
        return "NORMAL"

    # ── Regime Engine Integration ──────────────────────────────────────

    def get_regime_context(self) -> RegimeContext:
        """Pull regime state from AAC's RegimeEngine output files.

        Reads the latest regime snapshot from data/regime_state.json
        which is written by the RegimeEngine during its evaluation cycle.
        """
        ctx = RegimeContext()
        regime_file = Path("data/regime_state.json")

        if regime_file.exists():
            try:
                data = json.loads(regime_file.read_text(encoding="utf-8"))
                ctx.regime = data.get("regime", "uncertain")
                ctx.confidence = data.get("confidence", 0.0)
                ctx.war_active = data.get("war_active", False)
                ctx.hormuz_blocked = data.get("hormuz_blocked", False)
                ctx.oil_price = data.get("oil_price", 96.5)
                ctx.vix = data.get("vix", 22.0)
                ctx.hy_spread_bps = data.get("hy_spread_bps", 350.0)
                ctx.treasury_10y_2y = data.get("treasury_10y_2y", -0.1)
                ctx.fear_greed = data.get("fear_greed", 35.0)
                ctx.active_formulas = data.get("active_formulas", [])
                logger.info("Regime loaded: %s (conf=%.0f)", ctx.regime, ctx.confidence)
            except Exception as exc:
                logger.warning("Failed to load regime_state.json: %s", exc)

        self._regime_cache = ctx
        return ctx

    def adjust_scenario_weights(
        self, base_weights: ScenarioWeights, regime: RegimeContext
    ) -> ScenarioWeights:
        """Adjust Monte Carlo scenario weights based on regime + NCC governance.

        Rules:
            - VOL_SHOCK_ACTIVE / CREDIT_STRESS → heavier BEAR weight
            - RISK_ON → heavier BULL weight
            - NCC CAUTION → shift 10% more to BEAR
            - War active + Hormuz blocked → oil adjustment
        """
        weights = ScenarioWeights(
            base=base_weights.base,
            bear=base_weights.bear,
            bull=base_weights.bull,
        )

        # Regime adjustments
        if regime.is_crisis:
            shift = 0.15
            weights.bear = min(0.80, weights.bear + shift)
            weights.base = max(0.10, weights.base - shift * 0.6)
            weights.bull = max(0.05, weights.bull - shift * 0.4)
            logger.info("Crisis regime — BEAR weight boosted to %.0f%%", weights.bear * 100)

        elif regime.is_armed:
            shift = 0.08
            weights.bear = min(0.65, weights.bear + shift)
            weights.base = max(0.20, weights.base - shift * 0.5)
            weights.bull = max(0.05, weights.bull - shift * 0.5)

        elif regime.regime == "risk_on":
            shift = 0.10
            weights.bull = min(0.40, weights.bull + shift)
            weights.bear = max(0.20, weights.bear - shift)

        # NCC governance overlay
        doctrine = self.get_doctrine_mode()
        if doctrine == "CAUTION":
            weights.bear = min(0.75, weights.bear + 0.10)
            weights.bull = max(0.05, weights.bull - 0.10)
        elif doctrine in ("SAFE_MODE", "HALT"):
            weights.bear = 0.70
            weights.base = 0.25
            weights.bull = 0.05

        # Oil adjustment — apply when oil is at extreme levels OR hormuz blocked
        if regime.hormuz_blocked or regime.oil_price > 95 or regime.oil_price < 85:
            weights = weights.adjust_for_oil(regime.oil_price)

        # VIX adjustment
        weights = weights.adjust_for_vix(regime.vix)

        # Normalize
        total = weights.base + weights.bear + weights.bull
        if total > 0:
            weights.base /= total
            weights.bear /= total
            weights.bull /= total

        return weights

    # ── NCL Intelligence Feed ─────────────────────────────────────────

    def get_ncl_signals(self) -> Dict[str, Any]:
        """Read NCL intelligence for signal enrichment.

        Returns signals like sector rotation, sentiment shifts, etc.
        that can augment MATRIX MAXIMIZER's scanner scoring.
        """
        signals: Dict[str, Any] = {"available": False}

        ncl_intel_file = Path("data/pillar_state/ncl_intelligence.json")
        if ncl_intel_file.exists():
            try:
                data = json.loads(ncl_intel_file.read_text(encoding="utf-8"))
                signals = {
                    "available": True,
                    "sector_rotation": data.get("sector_rotation", {}),
                    "sentiment": data.get("sentiment", {}),
                    "momentum_signals": data.get("momentum_signals", []),
                    "timestamp": data.get("timestamp", ""),
                }
            except Exception as exc:
                logger.debug("NCL intelligence unavailable: %s", exc)

        return signals

    # ── Stock Forecaster Integration ──────────────────────────────────

    def get_forecaster_rankings(self) -> List[Dict[str, Any]]:
        """Read latest stock forecaster trade opportunities.

        Returns ranked trade opportunities from the StockForecaster
        that can be cross-referenced with MATRIX MAXIMIZER's scanner output.
        """
        rankings: List[Dict[str, Any]] = []

        forecast_file = Path("data/stock_forecaster_output.json")
        if forecast_file.exists():
            try:
                data = json.loads(forecast_file.read_text(encoding="utf-8"))
                rankings = data.get("opportunities", [])
                logger.info("Loaded %d forecaster trade opportunities", len(rankings))
            except Exception as exc:
                logger.debug("Stock forecaster output unavailable: %s", exc)

        return rankings

    # ── Push Results Back ──────────────────────────────────────────────

    def push_results(
        self,
        snapshot: RiskSnapshot,
        top_picks: List[Dict[str, Any]],
        regime: RegimeContext,
    ) -> bool:
        """Push MATRIX MAXIMIZER results to all connected pillars.

        Writes to:
            - data/pillar_state/matrix_maximizer_sync.json (for NCC/NCL)
            - data/matrix_maximizer_latest.json (for dashboard)
        """
        sync = PillarSync(
            timestamp=datetime.utcnow().isoformat(),
            circuit_breaker=snapshot.circuit_breaker.value,
            mandate_level=snapshot.mandate.level.value,
            portfolio_delta=snapshot.portfolio_delta,
            exposure_pct=snapshot.exposure_pct,
            top_picks=top_picks[:10],
            regime_context={
                "regime": regime.regime,
                "confidence": regime.confidence,
                "war_active": regime.war_active,
                "hormuz_blocked": regime.hormuz_blocked,
                "oil_price": regime.oil_price,
                "vix": regime.vix,
            },
            risk_checks_passed=snapshot.passed,
            risk_checks_failed=snapshot.failed,
        )

        payload = {
            "source": "MATRIX_MAXIMIZER",
            "timestamp": sync.timestamp,
            "circuit_breaker": sync.circuit_breaker,
            "mandate": sync.mandate_level,
            "portfolio_delta": sync.portfolio_delta,
            "exposure_pct": sync.exposure_pct,
            "top_picks": sync.top_picks,
            "regime": sync.regime_context,
            "risk": {
                "passed": sync.risk_checks_passed,
                "failed": sync.risk_checks_failed,
            },
        }

        success = True

        # Write to pillar state directory
        try:
            state_dir = Path("data/pillar_state")
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "matrix_maximizer_sync.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write pillar sync: %s", exc)
            success = False

        # Write dashboard-ready output
        try:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "matrix_maximizer_latest.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write dashboard output: %s", exc)
            success = False

        # Push to NCL if hub is available
        if self._hub:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._hub.push_intelligence_to_ncl(payload))
                else:
                    loop.run_until_complete(self._hub.push_intelligence_to_ncl(payload))
            except Exception as exc:
                logger.debug("NCL push skipped: %s", exc)

        if success:
            logger.info(
                "Results pushed — circuit=%s mandate=%s picks=%d",
                sync.circuit_breaker, sync.mandate_level, len(sync.top_picks),
            )

        return success

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all pillar integrations."""
        return {
            "ncc": {
                "connected": self._hub is not None,
                "doctrine_mode": self.get_doctrine_mode(),
                "risk_multiplier": self.get_risk_multiplier(),
                "should_trade": self.should_trade(),
            },
            "ncl": {
                "signals_available": self.get_ncl_signals().get("available", False),
            },
            "regime_engine": {
                "loaded": self._regime_cache is not None,
                "regime": self._regime_cache.regime if self._regime_cache else "unknown",
                "confidence": self._regime_cache.confidence if self._regime_cache else 0,
            },
            "forecaster": {
                "rankings_available": len(self.get_forecaster_rankings()) > 0,
            },
        }

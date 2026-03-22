"""
Storm Lifeboat Matrix — 15-Scenario Tracking Engine
=====================================================
Monitors all 15 crisis scenarios with indicator-based status tracking,
probability updates, and cross-scenario contagion modeling.

Each scenario tracks:
- Current status (DORMANT → EMERGING → ACTIVE → ESCALATING → PEAK → RECEDING)
- Firing indicators (from external data or manual input)
- Dynamic probability adjustment based on indicator count
- Cross-contagion: active scenarios can raise probability of related ones

Contagion rules:
- HORMUZ active → raises SUPERCYCLE, FOOD_CRISIS
- DEBT_CRISIS active → raises EU_BANKS, JAPAN_CRISIS, CRE_COLLAPSE
- DEFI_CASCADE active → no equity contagion (crypto-isolated)
- MONETARY_RESET active → raises all scenarios 5-10%
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.storm_lifeboat.core import (
    ScenarioDefinition,
    ScenarioState,
    ScenarioStatus,
    SCENARIOS,
    SCENARIO_MAP,
)

logger = logging.getLogger(__name__)

# Contagion map: if scenario X is ACTIVE+, what other scenarios get probability boosts
_CONTAGION_MAP: Dict[str, Dict[str, float]] = {
    "HORMUZ": {"SUPERCYCLE": 0.10, "FOOD_CRISIS": 0.08, "EM_FX_CRISIS": 0.05},
    "DEBT_CRISIS": {"EU_BANKS": 0.12, "JAPAN_CRISIS": 0.10, "CRE_COLLAPSE": 0.08,
                     "EM_FX_CRISIS": 0.06},
    "TAIWAN": {"HORMUZ": 0.05, "AI_BUBBLE": 0.15, "EM_FX_CRISIS": 0.08},
    "EU_BANKS": {"DEBT_CRISIS": 0.08, "CRE_COLLAPSE": 0.10, "JAPAN_CRISIS": 0.06},
    "DEFI_CASCADE": {},  # crypto-isolated
    "SUPERCYCLE": {"HORMUZ": 0.05, "FOOD_CRISIS": 0.06, "MONETARY_RESET": 0.08},
    "CRE_COLLAPSE": {"EU_BANKS": 0.06, "DEBT_CRISIS": 0.05},
    "AI_BUBBLE": {"DEFI_CASCADE": 0.05},
    "EM_FX_CRISIS": {"EU_BANKS": 0.05, "DEBT_CRISIS": 0.04},
    "FOOD_CRISIS": {"EM_FX_CRISIS": 0.06, "ELECTION_CHAOS": 0.04},
    "CLIMATE_SHOCK": {"FOOD_CRISIS": 0.10, "EM_FX_CRISIS": 0.05},
    "MONETARY_RESET": {s.code: 0.05 for s in SCENARIOS if s.code != "MONETARY_RESET"},
    "JAPAN_CRISIS": {"EU_BANKS": 0.08, "DEBT_CRISIS": 0.06, "DEFI_CASCADE": 0.04},
    "ELECTION_CHAOS": {"DEBT_CRISIS": 0.04, "AI_BUBBLE": 0.03},
    "PANDEMIC_V2": {"FOOD_CRISIS": 0.06, "EM_FX_CRISIS": 0.05, "CRE_COLLAPSE": 0.04},
}


def _indicators_to_status(n_firing: int, total: int) -> ScenarioStatus:
    """Map indicator firing ratio to scenario status."""
    if total == 0:
        return ScenarioStatus.DORMANT
    ratio = n_firing / total
    if ratio >= 0.90:
        return ScenarioStatus.PEAK
    if ratio >= 0.70:
        return ScenarioStatus.ESCALATING
    if ratio >= 0.50:
        return ScenarioStatus.ACTIVE
    if ratio >= 0.25:
        return ScenarioStatus.EMERGING
    return ScenarioStatus.DORMANT


class ScenarioEngine:
    """Tracks and updates all 15 crisis scenarios.

    Maintains a state vector of scenario probabilities and statuses,
    applies contagion effects, and generates scenario-weighted inputs
    for the Monte Carlo engine.
    """

    def __init__(self) -> None:
        self._states: Dict[str, ScenarioState] = {}
        now = datetime.utcnow()
        for sc in SCENARIOS:
            self._states[sc.code] = ScenarioState(
                code=sc.code,
                status=ScenarioStatus.DORMANT,
                probability=sc.probability,
                indicators_firing=[],
                last_updated=now,
            )

    @property
    def states(self) -> Dict[str, ScenarioState]:
        return dict(self._states)

    def update_indicators(
        self,
        code: str,
        firing_indicators: List[str],
    ) -> ScenarioState:
        """Update which indicators are firing for a scenario.

        Automatically adjusts status and probability.
        """
        if code not in self._states:
            raise ValueError(f"Unknown scenario code: {code}")

        defn = SCENARIO_MAP[code]
        total_indicators = len(defn.trigger_indicators)

        self._states[code].indicators_firing = firing_indicators
        self._states[code].status = _indicators_to_status(len(firing_indicators), total_indicators)
        self._states[code].last_updated = datetime.utcnow()

        # Adjust probability based on indicator ratio
        ratio = len(firing_indicators) / max(total_indicators, 1)
        base_prob = defn.probability
        # Scale probability: at 0 indicators = base_prob * 0.5, at all indicators = min(0.95, base_prob * 2)
        adjusted = base_prob * (0.5 + ratio * 1.5)
        self._states[code].probability = min(0.95, max(0.01, adjusted))

        return self._states[code]

    def apply_contagion(self) -> Dict[str, float]:
        """Apply cross-scenario contagion effects.

        Active/escalating/peak scenarios boost related scenarios' probabilities.
        Returns dict of probability adjustments applied.
        """
        adjustments: Dict[str, float] = {}
        active_statuses = {
            ScenarioStatus.ACTIVE,
            ScenarioStatus.ESCALATING,
            ScenarioStatus.PEAK,
        }

        for code, state in self._states.items():
            if state.status not in active_statuses:
                continue
            contagion = _CONTAGION_MAP.get(code, {})
            # Scale contagion by how active the source is
            intensity = {
                ScenarioStatus.ACTIVE: 0.5,
                ScenarioStatus.ESCALATING: 0.8,
                ScenarioStatus.PEAK: 1.0,
            }.get(state.status, 0.0)

            for target, boost in contagion.items():
                if target in self._states:
                    actual_boost = boost * intensity
                    self._states[target].probability = min(
                        0.95,
                        self._states[target].probability + actual_boost,
                    )
                    adjustments[f"{code}->{target}"] = actual_boost

        return adjustments

    def get_active_scenarios(self) -> List[ScenarioDefinition]:
        """Return ScenarioDefinitions for scenarios that are EMERGING or beyond.

        Updates each definition's probability to match the current tracked state.
        """
        active = []
        for sc in SCENARIOS:
            state = self._states.get(sc.code)
            if state and state.status != ScenarioStatus.DORMANT:
                # Create a copy with updated probability
                updated = ScenarioDefinition(
                    id=sc.id,
                    name=sc.name,
                    code=sc.code,
                    description=sc.description,
                    trigger_indicators=sc.trigger_indicators,
                    probability=state.probability,
                    impact_severity=sc.impact_severity,
                    beneficiary_assets=sc.beneficiary_assets,
                    victim_assets=sc.victim_assets,
                    oil_sensitivity=sc.oil_sensitivity,
                    status=state.status,
                )
                active.append(updated)
        return active

    def get_risk_heatmap(self) -> List[Dict[str, Any]]:
        """Generate a risk heatmap of all scenarios for dashboard display."""
        heatmap = []
        for sc in SCENARIOS:
            state = self._states[sc.code]
            heatmap.append({
                "id": sc.id,
                "name": sc.name,
                "code": sc.code,
                "status": state.status.value,
                "probability": round(state.probability, 3),
                "severity": sc.impact_severity,
                "risk_score": round(state.probability * sc.impact_severity, 3),
                "indicators_firing": len(state.indicators_firing),
                "indicators_total": len(sc.trigger_indicators),
            })
        # Sort by risk score descending
        heatmap.sort(key=lambda x: x["risk_score"], reverse=True)
        return heatmap

    def save_state(self, path: str) -> None:
        """Persist current scenario states to JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for code, state in self._states.items():
            data[code] = {
                "status": state.status.value,
                "probability": state.probability,
                "indicators_firing": state.indicators_firing,
                "last_updated": state.last_updated.isoformat(),
            }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Scenario state saved to %s", out_path)

    def load_state(self, path: str) -> None:
        """Restore scenario states from JSON."""
        out_path = Path(path)
        if not out_path.exists():
            logger.warning("No saved state at %s", out_path)
            return
        data = json.loads(out_path.read_text())
        for code, vals in data.items():
            if code in self._states:
                self._states[code].status = ScenarioStatus(vals["status"])
                self._states[code].probability = vals["probability"]
                self._states[code].indicators_firing = vals.get("indicators_firing", [])
                self._states[code].last_updated = datetime.fromisoformat(vals["last_updated"])
        logger.info("Scenario state loaded from %s", out_path)

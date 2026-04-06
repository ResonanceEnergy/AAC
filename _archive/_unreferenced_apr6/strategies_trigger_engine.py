"""
Rocket Ship — Ignition Trigger Engine
=======================================
Determines the current system phase (LIFE_BOAT / IGNITING / ROCKET / ORBIT)
by combining:

  1. Gulf Yuan Oil Trigger  (real-time geopolitical signal — manual input)
  2. Indicator green count  (≥10 of 15 = threshold met)
  3. Moon cycle position    (Moon 13 new moon = default ignition date)

Priority order:
  a. CONFIRMED Gulf trigger  → ignite immediately regardless of moon/indicators
  b. ≥10 indicators GREEN    → ignite on NEXT NEW MOON (disciplined entry)
  c. Default Moon 13         → ignite on March 19, 2027 new moon

Phase state machine:
    LIFE_BOAT ──► IGNITING ──► ROCKET ──► ORBIT
                     ▲
                 (deploy on
                  new moon)

The engine also calculates:
  - Time-to-ignition (days)
  - Probability score (0-1) based on indicator + trigger state
  - Recommended immediate actions
  - Alert level (MONITOR / WATCH / STANDBY / LAUNCH)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from strategies.rocket_ship.lunar_cycles import NEW_MOON_DATES, RocketLunarEngine

from strategies.rocket_ship.core import (
    INDICATORS_REQUIRED_FOR_IGNITION,
    LIFEBOAT_INCEPTION,
    ROCKET_MOON_START,
    STARTING_CAPITAL_USD,
    SystemPhase,
    TriggerStatus,
)
from strategies.rocket_ship.indicators import IndicatorEngine

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ALERT LEVELS
# ═══════════════════════════════════════════════════════════════════════════

ALERT_MONITOR  = "MONITOR"   # <5 green, trigger watching — no action
ALERT_WATCH    = "WATCH"     # 5-9 green OR trigger emerging — prepare
ALERT_STANDBY  = "STANDBY"   # 10+ green, trigger NOT yet confirmed — bridge prepped
ALERT_LAUNCH   = "LAUNCH"    # Trigger CONFIRMED or Moon 13+ with 10+ green


@dataclass
class TriggerReport:
    """Full ignition status report."""
    # Core state
    phase: SystemPhase
    alert_level: str
    is_ignited: bool

    # Timing
    today: date
    days_to_default_ignition: int    # Days to Moon 13 (if not yet)
    days_in_rocket: int              # Days elapsed in rocket phase (if ignited)
    ignition_date: Optional[date]    # Actual ignition date (None if Life Boat)

    # Signals
    green_count: int
    indicators_met: bool             # green_count >= 10
    gulf_trigger_status: str         # TriggerStatus value
    gulf_triggered: bool

    # Probability score (0.0-1.0) — composite of indicator + trigger + lunar
    ignition_probability: float      # Probability ignition fires by Moon 13

    # Actions
    immediate_actions: List[str]
    reason: str

    @property
    def status_line(self) -> str:
        phase_labels = {
            SystemPhase.LIFE_BOAT: "🛥  LIFE BOAT — Survival mode",
            SystemPhase.IGNITING:  "⚡  IGNITING  — Deploying Rocket positions",
            SystemPhase.ROCKET:    "🚀  ROCKET    — Full yield deployment",
            SystemPhase.ORBIT:     "🌍  ORBIT     — 2030 stabilization",
        }
        return phase_labels.get(self.phase, self.phase.value)


# ═══════════════════════════════════════════════════════════════════════════
# TRIGGER ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TriggerEngine:
    """
    Central decision engine for Rocket Ship phase management.

    Reads indicators and Gulf trigger state to determine current phase,
    compute time-to-ignition, and generate action priority list.
    """

    def __init__(
        self,
        indicator_engine: Optional[IndicatorEngine] = None,
        lunar_engine: Optional[RocketLunarEngine] = None,
        inception: date = LIFEBOAT_INCEPTION,
    ) -> None:
        self.indicators = indicator_engine or IndicatorEngine()
        self.lunar = lunar_engine or RocketLunarEngine(inception)
        self.inception = inception
        self._ignition_date: Optional[date] = None

    @property
    def _default_ignition_date(self) -> date:
        """Moon 13 new moon = default ignition."""
        return NEW_MOON_DATES[ROCKET_MOON_START - 1]  # index 12 = Moon 13

    def _next_new_moon(self, after: date) -> date:
        """Return the next new moon date after `after`."""
        for nm in NEW_MOON_DATES:
            if nm > after:
                return nm
        # Extrapolate
        from strategies.rocket_ship.core import SYNODIC_MONTH_DAYS
        last = NEW_MOON_DATES[-1]
        while last <= after:
            last += timedelta(days=SYNODIC_MONTH_DAYS)
        return last

    def _compute_probability(self, green: int, trigger: TriggerStatus) -> float:
        """
        Heuristic probability of ignition occurring by Moon 13.

        Components:
          - Indicator score: green/15 × 0.50 weight
          - Trigger score:   WATCHING=0, EMERGING=0.30, CONFIRMED=1.0 × 0.40 weight
          - Time decay:      0.10 weight — closer to Moon 13 = higher base
        """
        ind_score = (green / 15) * 0.50
        trigger_weights = {
            TriggerStatus.WATCHING:  0.00,
            TriggerStatus.EMERGING:  0.30,
            TriggerStatus.CONFIRMED: 1.00,
        }
        trig_score = trigger_weights.get(trigger, 0.0) * 0.40
        today = date.today()
        total_lifeboat_days = (self._default_ignition_date - self.inception).days
        elapsed = (today - self.inception).days
        time_score = min(1.0, elapsed / max(total_lifeboat_days, 1)) * 0.10
        return min(1.0, ind_score + trig_score + time_score)

    def _build_actions(
        self,
        phase: SystemPhase,
        alert: str,
        green: int,
        trigger: TriggerStatus,
        moon_state: any,
    ) -> List[str]:
        """Generate prioritized action list for current state."""
        actions: List[str] = []

        if phase == SystemPhase.LIFE_BOAT:
            actions.append("HOLD: 40% BTC+gold, 30% stables (USDC/USDT), 20% XRP, 10% cash")
            if alert == ALERT_WATCH:
                actions.append("PREPARE: Set up Ledger/Trezor + practice XRP → Flare bridge with $50 test")
                actions.append("PREPARE: Begin Panama lawyer consultation (Kraemer & Kraemer or Agroup)")
                actions.append("PREPARE: Gather apostilled documents (RCMP check, birth cert, passport)")
            if alert == ALERT_STANDBY:
                actions.append("STANDBY: Pre-bridge XRP → Flare (FXRP) — ready to deploy on ignition")
                actions.append("STANDBY: Open Panama bank account (Towerbank / Banistmo)")
                actions.append("STANDBY: Confirm Panama June trip flights and lawyer booking")
            if trigger == TriggerStatus.EMERGING:
                actions.append("WATCH: Monitor Gulf yuan oil news daily (Reuters, S&P Global, Bloomberg)")
                actions.append("WATCH: Track mBridge monthly volume report for energy settlement spike")

        elif phase == SystemPhase.IGNITING:
            actions.append("IGNITE: Bridge native XRP → FXRP on Flare (no sell, 1:1 ratio)")
            actions.append("IGNITE: Deploy FXRP to Morpho lending vault — target 5-8% APY")
            actions.append("IGNITE: Bridge wXRP to Solana via Hex Trust/LayerZero for LP yield")
            actions.append("IGNITE: Begin BRICS/Unit exposure via PAXG + gold-pegged stables")
            actions.append("GEO: Book Panama flights immediately — ops base launch")

        elif phase == SystemPhase.ROCKET:
            if moon_state and moon_state.phase.value == "new":
                actions.append("ROCKET NEW MOON: Re-read all 15 indicators — update statuses")
                actions.append("ROCKET NEW MOON: Review geo-plan task completions")
            if moon_state and moon_state.in_any_phi_window:
                actions.append("★ PHI WINDOW: Bridge + deploy on peak coherence day")
            actions.append("ROCKET: Maintain 25% XRP/Flare, 20% SOL, 15% ETH, 15% BTC, 15% BRICS")
            actions.append("ROCKET: Compound yields quarterly — reinvest Morpho + SP DEX returns")
            actions.append("GEO: Execute geo-plan tasks for current moon (see geo_plan module)")

        elif phase == SystemPhase.ORBIT:
            actions.append("ORBIT: Rebalance to 2030 allocations — review all 5 residencies")
            actions.append("ORBIT: Maintain hardware wallet custody for all crypto")

        return actions

    def evaluate(self, today: Optional[date] = None) -> TriggerReport:
        """Run full trigger evaluation for the given date."""
        target = today or date.today()
        moon_state = self.lunar.get_current_state(target)
        green = self.indicators.green_count()
        indicators_met = green >= INDICATORS_REQUIRED_FOR_IGNITION
        gulf_status = self.indicators.gulf_trigger.status
        gulf_confirmed = gulf_status == TriggerStatus.CONFIRMED

        # Determine current phase
        default_ignition = self._default_ignition_date
        days_to_default = max(0, (default_ignition - target).days)

        if self._ignition_date is not None:
            # Already ignited
            days_in_rocket = (target - self._ignition_date).days
            from strategies.rocket_ship.lunar_cycles import NEW_MOON_DATES as NMD

            from strategies.rocket_ship.core import ROCKET_MOON_END
            orbit_start = NMD[ROCKET_MOON_END - 1]
            if target >= orbit_start:
                phase = SystemPhase.ORBIT
            else:
                phase = SystemPhase.ROCKET
        elif gulf_confirmed:
            # Gulf trigger fires ignition on next new moon
            self._ignition_date = self._next_new_moon(target)
            phase = SystemPhase.IGNITING
            days_in_rocket = 0
        elif indicators_met and target >= default_ignition:
            self._ignition_date = default_ignition
            phase = SystemPhase.ROCKET
            days_in_rocket = (target - default_ignition).days
        elif target >= default_ignition:
            # Past Moon 13 — in rocket phase regardless (default launch)
            self._ignition_date = default_ignition
            phase = SystemPhase.ROCKET
            days_in_rocket = (target - default_ignition).days
        else:
            phase = SystemPhase.LIFE_BOAT
            days_in_rocket = 0

        # Alert level
        if gulf_confirmed or (indicators_met and phase != SystemPhase.LIFE_BOAT):
            alert = ALERT_LAUNCH
        elif indicators_met:
            alert = ALERT_STANDBY
        elif green >= 5 or gulf_status == TriggerStatus.EMERGING:
            alert = ALERT_WATCH
        else:
            alert = ALERT_MONITOR

        # Probability
        prob = self._compute_probability(green, gulf_status)

        # Ignition reason
        if gulf_confirmed:
            reason = "GULF YUAN OIL CONFIRMED — Rocket launches on next new moon"
        elif phase in (SystemPhase.ROCKET, SystemPhase.ORBIT):
            reason = f"Rocket active — ignited {self._ignition_date}"
        elif indicators_met:
            reason = f"{green}/15 indicators GREEN — standby for Moon 13 ignition"
        else:
            reason = f"{green}/15 GREEN (need {INDICATORS_REQUIRED_FOR_IGNITION}) — Life Boat holds"

        actions = self._build_actions(phase, alert, green, gulf_status, moon_state)

        return TriggerReport(
            phase=phase,
            alert_level=alert,
            is_ignited=phase in (SystemPhase.ROCKET, SystemPhase.ORBIT, SystemPhase.IGNITING),
            today=target,
            days_to_default_ignition=days_to_default,
            days_in_rocket=days_in_rocket,
            ignition_date=self._ignition_date,
            green_count=green,
            indicators_met=indicators_met,
            gulf_trigger_status=gulf_status.value,
            gulf_triggered=gulf_confirmed,
            ignition_probability=prob,
            immediate_actions=actions,
            reason=reason,
        )

    def format_dashboard(self, today: Optional[date] = None) -> str:
        """Full ignition status dashboard."""
        r = self.evaluate(today)

        bar_len = 15
        green_bars = int(r.green_count / 15 * bar_len)
        bar = "█" * green_bars + "░" * (bar_len - green_bars)
        prob_pct = int(r.ignition_probability * 100)

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════════╗",
            "║          ROCKET SHIP — IGNITION TRIGGER ENGINE                         ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║  {r.status_line:<72}║",
            f"║  Alert: {r.alert_level:<10}  │  Probability: {prob_pct:>3}%  │  {r.today}          ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║  Indicators: [{bar}] {r.green_count:>2}/15 GREEN                      ║",
            f"║  Gulf Trigger: [{r.gulf_trigger_status.upper():^12}]  {'IGNITION ACTIVE' if r.gulf_triggered else '':^20}    ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
        ]

        if not r.is_ignited:
            lines.append(
                f"║  Days to Moon 13 (default ignition): {r.days_to_default_ignition:>4}"
                f"  ({self._default_ignition_date})         ║"
            )
        else:
            lines.append(
                f"║  Ignited: {r.ignition_date}  │  Days in Rocket window: {r.days_in_rocket:>4}              ║"
            )

        lines += [
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║  Reason: {r.reason[:63]:<63}║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            "║  IMMEDIATE ACTIONS:                                                     ║",
        ]
        for i, action in enumerate(r.immediate_actions[:5], 1):
            lines.append(f"║  {i}. {action[:69]:<69}║")

        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

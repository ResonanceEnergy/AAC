"""
AAC Strategic Doctrine Engine
==============================

Integrates Sun Tzu's Art of War and Robert Greene's 48 Laws of Power
into the AAC trading system's decision-making architecture.

Maps ancient strategic principles to modern quantitative trading:
- Terrain analysis → Market regime detection
- Force assessment → Portfolio strength evaluation
- Timing mastery → Entry/exit optimization
- Deception → Order flow concealment
- Power dynamics → Market presence management

Doctrine Pack 9: "Strategic Warfare" (Art of War / Sun Tzu)
Doctrine Pack 10: "Power Dynamics" (48 Laws of Power / Greene)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("StrategicDoctrine")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGIC ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class MarketTerrain(Enum):
    """Sun Tzu's Nine Terrains mapped to market conditions.

    "The art of war recognizes nine varieties of ground."
    """
    DISPERSIVE = "dispersive"        # Home market, low conviction — avoid trading
    FRONTIER = "frontier"            # Shallow position, early entry — stay nimble
    CONTENTIOUS = "contentious"      # Contested levels (support/resistance) — speed wins
    OPEN = "open"                    # Trending, no obstacles — ride momentum
    INTERSECTING = "intersecting"    # Multi-exchange convergence — build alliances
    SERIOUS = "serious"              # Deep in position — commit fully, no retreat
    DIFFICULT = "difficult"          # Low liquidity, rough conditions — move carefully
    HEMMED_IN = "hemmed_in"          # Limited exits, gap risk — prepare escape routes
    DEATH = "death"                  # Fight or die — all-in or cut losses immediately


class StrategicPosture(Enum):
    """Overall strategic positioning derived from Sun Tzu's five essentials."""
    AGGRESSIVE = "aggressive"        # "He will win who knows when to fight"
    DEFENSIVE = "defensive"          # "Stand on the defensive against a superior"
    OPPORTUNISTIC = "opportunistic"  # "In the midst of chaos, there is opportunity"
    PATIENT = "patient"              # "The wise warrior avoids the battle"
    DECEPTIVE = "deceptive"          # "All warfare is based on deception"


class PowerLaw(Enum):
    """Key 48 Laws of Power mapped to trading operations."""
    CONCEAL_INTENTIONS = "conceal_intentions"          # Law 3: Don't telegraph trades
    SAY_LESS = "say_less_than_necessary"               # Law 4: Minimize market footprint
    GUARD_REPUTATION = "guard_reputation"              # Law 5: Maintain exchange trust
    COURT_ATTENTION = "court_attention"                 # Law 6: Strategic visibility
    MAKE_OTHERS_COME = "make_others_come_to_you"       # Law 8: Passive limit orders
    ACTIONS_NOT_ARGUMENTS = "win_through_actions"      # Law 9: P&L speaks
    AVOID_UNLUCKY = "avoid_the_unhappy"                # Law 10: Avoid correlated losers
    SELECTIVE_HONESTY = "use_selective_honesty"         # Law 12: Build counterparty trust
    CRUSH_TOTALLY = "crush_totally"                     # Law 15: Exploit weakness completely
    USE_ABSENCE = "use_absence"                        # Law 16: Strategic withdrawal
    KEEP_OTHERS_SUSPENSE = "keep_others_in_suspense"   # Law 17: Unpredictable execution
    CONCENTRATE_FORCES = "concentrate_forces"          # Law 23: Focus capital on best setups
    PLAY_PERFECT_COURTIER = "play_perfect_courtier"    # Law 24: Adapt to exchange rules
    ENTER_BOLDLY = "enter_action_with_boldness"        # Law 28: Decisive entries
    PLAN_ALL_THE_WAY = "plan_all_the_way"              # Law 29: Full trade lifecycle
    DESPISE_FREE_LUNCH = "despise_the_free_lunch"      # Law 40: No free alpha exists
    KNOW_WHEN_TO_STOP = "know_when_to_stop"            # Law 47: Don't overshoot targets


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGIC DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TerrainAssessment:
    """Sun Tzu terrain analysis applied to current market conditions."""
    terrain: MarketTerrain
    volatility_regime: str             # "low", "normal", "high", "extreme"
    liquidity_depth: float             # 0.0 (desert) to 1.0 (ocean)
    trend_strength: float              # -1.0 (strong down) to 1.0 (strong up)
    support_resistance_proximity: float  # 0.0 (at level) to 1.0 (far away)
    time_of_day_advantage: float       # 0.0 (worst) to 1.0 (optimal session)
    assessment_confidence: float       # 0.0 to 1.0
    recommended_posture: StrategicPosture = StrategicPosture.PATIENT
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ForceAssessment:
    """Sun Tzu force comparison — "know yourself, know your enemy"."""
    our_capital_strength: float        # Available capital as fraction of desired
    our_position_diversity: float      # Portfolio diversification 0-1
    our_alpha_edge: float              # Measured alpha vs market, 0-1
    our_execution_speed: float         # Latency advantage 0-1
    market_adversary_strength: float   # Opposing flow intensity 0-1
    market_regime_favorability: float  # How favorable conditions are 0-1
    force_ratio: float = 0.0          # Our strength / adversary strength
    recommended_posture: StrategicPosture = StrategicPosture.PATIENT
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.market_adversary_strength > 0:
            self.force_ratio = (
                (self.our_capital_strength + self.our_alpha_edge + self.our_execution_speed)
                / (3 * self.market_adversary_strength)
            )
        # Sun Tzu: "If ten times the enemy's strength, surround them;
        #           if five times, attack; if double, divide; if equal, engage;
        #           if fewer, retreat; if outmatched, avoid."
        if self.force_ratio >= 5.0:
            self.recommended_posture = StrategicPosture.AGGRESSIVE
        elif self.force_ratio >= 2.0:
            self.recommended_posture = StrategicPosture.AGGRESSIVE
        elif self.force_ratio >= 1.0:
            self.recommended_posture = StrategicPosture.OPPORTUNISTIC
        elif self.force_ratio >= 0.5:
            self.recommended_posture = StrategicPosture.DEFENSIVE
        else:
            self.recommended_posture = StrategicPosture.PATIENT


@dataclass
class PowerAssessment:
    """48 Laws assessment of our market positioning."""
    market_footprint_stealth: float    # How hidden our order flow is, 0-1
    exchange_reputation_score: float   # Standing with venues, 0-1
    alpha_source_uniqueness: float     # How differentiated our edge is, 0-1
    strategy_unpredictability: float   # How hard to front-run, 0-1
    capital_concentration_ratio: float  # Focused vs scattered, 0-1
    active_laws: List[PowerLaw] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategicDirective:
    """Combined strategic recommendation from both doctrines."""
    terrain: TerrainAssessment
    force: ForceAssessment
    power: PowerAssessment
    overall_posture: StrategicPosture
    position_size_modifier: float      # 0.0 (no trade) to 2.0 (max conviction)
    urgency: float                     # 0.0 (wait) to 1.0 (act now)
    active_principles: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGIC DOCTRINE ENGINE
# ═══════════════════════════════════════════════════════════════════════════


class StrategicDoctrineEngine:
    """
    Core engine applying Art of War and 48 Laws of Power to trading decisions.

    Sun Tzu: "The supreme art of war is to subdue the enemy without fighting."
    → The supreme art of trading is to capture alpha without market impact.

    Greene: "The best deceptions are those that seem to give the other
    person a choice."
    → The best trades are those the market doesn't see coming.
    """

    def __init__(self):
        self.terrain_history: List[TerrainAssessment] = []
        self.force_history: List[ForceAssessment] = []
        self.power_history: List[PowerAssessment] = []
        self.directive_history: List[StrategicDirective] = []
        self.active_posture: StrategicPosture = StrategicPosture.PATIENT

        # Sun Tzu's five constant factors
        self._moral_law = 1.0    # Team alignment / system coherence
        self._heaven = 1.0       # Timing / macro conditions
        self._earth = 1.0        # Terrain / market microstructure
        self._commander = 1.0    # Decision quality / algorithm performance
        self._method = 1.0       # Execution discipline / system reliability

    # ─── TERRAIN ANALYSIS (Art of War, Ch. 11) ───────────────────────────

    def assess_terrain(
        self,
        volatility: float,
        liquidity: float,
        trend: float,
        sr_proximity: float,
        session_quality: float,
    ) -> TerrainAssessment:
        """
        Map current market conditions to Sun Tzu's Nine Terrains.

        "On dispersive ground, therefore, fight not.
         On facile ground, halt not.
         On contentious ground, attack not.
         On death ground, fight."
        """
        # Classify volatility regime
        if volatility < 0.1:
            vol_regime = "low"
        elif volatility < 0.25:
            vol_regime = "normal"
        elif volatility < 0.5:
            vol_regime = "high"
        else:
            vol_regime = "extreme"

        # Determine terrain type
        terrain = self._classify_terrain(volatility, liquidity, trend, sr_proximity)

        # Determine recommended posture from terrain
        posture = self._terrain_to_posture(terrain, vol_regime)

        confidence = min(1.0, (liquidity + session_quality + (1.0 - volatility)) / 3.0)

        assessment = TerrainAssessment(
            terrain=terrain,
            volatility_regime=vol_regime,
            liquidity_depth=liquidity,
            trend_strength=trend,
            support_resistance_proximity=sr_proximity,
            time_of_day_advantage=session_quality,
            assessment_confidence=confidence,
            recommended_posture=posture,
        )

        self.terrain_history.append(assessment)
        if len(self.terrain_history) > 100:
            self.terrain_history = self.terrain_history[-100:]

        return assessment

    def _classify_terrain(
        self, volatility: float, liquidity: float, trend: float, sr_proximity: float
    ) -> MarketTerrain:
        """Classify market conditions into one of Sun Tzu's Nine Terrains."""
        abs_trend = abs(trend)

        # Death ground: extreme volatility + low liquidity
        if volatility > 0.5 and liquidity < 0.2:
            return MarketTerrain.DEATH

        # Hemmed in: at support/resistance with low liquidity
        if sr_proximity < 0.15 and liquidity < 0.4:
            return MarketTerrain.HEMMED_IN

        # Difficult: low liquidity terrain
        if liquidity < 0.3:
            return MarketTerrain.DIFFICULT

        # Contentious: near support/resistance levels with decent liquidity
        if sr_proximity < 0.2 and liquidity >= 0.4:
            return MarketTerrain.CONTENTIOUS

        # Intersecting: high liquidity, multiple venues, moderate volatility
        if liquidity > 0.7 and 0.1 <= volatility <= 0.3:
            return MarketTerrain.INTERSECTING

        # Open: strong trend, good liquidity
        if abs_trend > 0.5 and liquidity > 0.5:
            return MarketTerrain.OPEN

        # Serious: moderate trend, committed position territory
        if abs_trend > 0.3 and liquidity > 0.4:
            return MarketTerrain.SERIOUS

        # Frontier: early trend, shallow
        if abs_trend > 0.1:
            return MarketTerrain.FRONTIER

        # Default: dispersive — no clear edge
        return MarketTerrain.DISPERSIVE

    def _terrain_to_posture(
        self, terrain: MarketTerrain, vol_regime: str
    ) -> StrategicPosture:
        """Sun Tzu: terrain dictates posture."""
        posture_map = {
            MarketTerrain.DISPERSIVE: StrategicPosture.PATIENT,
            MarketTerrain.FRONTIER: StrategicPosture.OPPORTUNISTIC,
            MarketTerrain.CONTENTIOUS: StrategicPosture.AGGRESSIVE,
            MarketTerrain.OPEN: StrategicPosture.AGGRESSIVE,
            MarketTerrain.INTERSECTING: StrategicPosture.OPPORTUNISTIC,
            MarketTerrain.SERIOUS: StrategicPosture.AGGRESSIVE,
            MarketTerrain.DIFFICULT: StrategicPosture.DEFENSIVE,
            MarketTerrain.HEMMED_IN: StrategicPosture.DECEPTIVE,
            MarketTerrain.DEATH: StrategicPosture.AGGRESSIVE,  # fight or die
        }
        posture = posture_map.get(terrain, StrategicPosture.PATIENT)

        # Extreme volatility forces caution regardless of terrain
        if vol_regime == "extreme" and terrain != MarketTerrain.DEATH:
            posture = StrategicPosture.DEFENSIVE

        return posture

    # ─── FORCE ASSESSMENT (Art of War, Ch. 3) ────────────────────────────

    def assess_force(
        self,
        available_capital_ratio: float,
        position_diversity: float,
        measured_alpha: float,
        execution_speed_advantage: float,
        opposing_flow_intensity: float,
        regime_favorability: float,
    ) -> ForceAssessment:
        """
        "Know your enemy and know yourself; in a hundred battles
        you will never be in peril."

        Compares our trading capabilities against market adversaries.
        """
        assessment = ForceAssessment(
            our_capital_strength=available_capital_ratio,
            our_position_diversity=position_diversity,
            our_alpha_edge=measured_alpha,
            our_execution_speed=execution_speed_advantage,
            market_adversary_strength=max(0.01, opposing_flow_intensity),
            market_regime_favorability=regime_favorability,
        )

        self.force_history.append(assessment)
        if len(self.force_history) > 100:
            self.force_history = self.force_history[-100:]

        return assessment

    # ─── POWER DYNAMICS (48 Laws) ────────────────────────────────────────

    def assess_power(
        self,
        order_flow_visibility: float,
        exchange_reputation: float,
        alpha_uniqueness: float,
        execution_predictability: float,
        capital_focus: float,
    ) -> PowerAssessment:
        """
        Apply 48 Laws of Power to determine which laws are currently active
        and what our market power position is.
        """
        active_laws: List[PowerLaw] = []

        # Law 3: Conceal your intentions — if our flow is hidden
        if order_flow_visibility < 0.3:
            active_laws.append(PowerLaw.CONCEAL_INTENTIONS)

        # Law 4: Say less than necessary — small market footprint
        if order_flow_visibility < 0.2:
            active_laws.append(PowerLaw.SAY_LESS)

        # Law 5: Guard reputation — maintain exchange relationships
        if exchange_reputation > 0.8:
            active_laws.append(PowerLaw.GUARD_REPUTATION)

        # Law 8: Make others come to you — use passive orders
        if capital_focus > 0.6:
            active_laws.append(PowerLaw.MAKE_OTHERS_COME)

        # Law 10: Avoid the unhappy — avoid correlated losers
        # (Evaluated at strategy selection level)
        active_laws.append(PowerLaw.AVOID_UNLUCKY)

        # Law 15: Crush totally — exploit weakness when found
        if alpha_uniqueness > 0.7:
            active_laws.append(PowerLaw.CRUSH_TOTALLY)

        # Law 17: Keep in suspense — unpredictable execution
        if execution_predictability < 0.3:
            active_laws.append(PowerLaw.KEEP_OTHERS_SUSPENSE)

        # Law 23: Concentrate forces — focused capital deployment
        if capital_focus > 0.7:
            active_laws.append(PowerLaw.CONCENTRATE_FORCES)

        # Law 28: Enter with boldness — decisive entries
        if alpha_uniqueness > 0.6 and capital_focus > 0.5:
            active_laws.append(PowerLaw.ENTER_BOLDLY)

        # Law 29: Plan all the way to the end
        active_laws.append(PowerLaw.PLAN_ALL_THE_WAY)

        # Law 40: Despise the free lunch — reject too-good-to-be-true setups
        active_laws.append(PowerLaw.DESPISE_FREE_LUNCH)

        # Law 47: Know when to stop — don't overshoot
        active_laws.append(PowerLaw.KNOW_WHEN_TO_STOP)

        assessment = PowerAssessment(
            market_footprint_stealth=1.0 - order_flow_visibility,
            exchange_reputation_score=exchange_reputation,
            alpha_source_uniqueness=alpha_uniqueness,
            strategy_unpredictability=1.0 - execution_predictability,
            capital_concentration_ratio=capital_focus,
            active_laws=active_laws,
        )

        self.power_history.append(assessment)
        if len(self.power_history) > 100:
            self.power_history = self.power_history[-100:]

        return assessment

    # ─── COMBINED STRATEGIC DIRECTIVE ────────────────────────────────────

    def generate_directive(
        self,
        terrain: TerrainAssessment,
        force: ForceAssessment,
        power: PowerAssessment,
    ) -> StrategicDirective:
        """
        Synthesize terrain, force, and power assessments into a unified directive.

        Sun Tzu: "Strategy without tactics is the slowest route to victory.
        Tactics without strategy is the noise before defeat."
        """
        # Determine overall posture by voting across assessments
        postures = [terrain.recommended_posture, force.recommended_posture]
        posture = self._resolve_posture(postures, terrain, force, power)

        # Calculate position size modifier
        size_mod = self._calculate_size_modifier(posture, terrain, force, power)

        # Calculate urgency
        urgency = self._calculate_urgency(terrain, force)

        # Collect active principles
        principles: List[str] = []
        warnings: List[str] = []

        # Art of War principles
        self._apply_art_of_war_principles(terrain, force, principles, warnings)

        # 48 Laws principles
        self._apply_power_law_principles(power, principles, warnings)

        directive = StrategicDirective(
            terrain=terrain,
            force=force,
            power=power,
            overall_posture=posture,
            position_size_modifier=size_mod,
            urgency=urgency,
            active_principles=principles,
            warnings=warnings,
        )

        self.directive_history.append(directive)
        if len(self.directive_history) > 100:
            self.directive_history = self.directive_history[-100:]

        self.active_posture = posture
        logger.info(
            f"Strategic Directive: posture={posture.value}, "
            f"size_mod={size_mod:.2f}, urgency={urgency:.2f}, "
            f"terrain={terrain.terrain.value}, "
            f"active_principles={len(principles)}"
        )

        return directive

    def _resolve_posture(
        self,
        postures: List[StrategicPosture],
        terrain: TerrainAssessment,
        force: ForceAssessment,
        power: PowerAssessment,
    ) -> StrategicPosture:
        """Resolve conflicting posture recommendations."""
        # Death ground overrides everything — fight or die
        if terrain.terrain == MarketTerrain.DEATH:
            return StrategicPosture.AGGRESSIVE

        # If we're outmatched, defense regardless of terrain
        if force.force_ratio < 0.3:
            return StrategicPosture.PATIENT

        # If stealth is high and we have edge, be deceptive
        if power.market_footprint_stealth > 0.8 and force.our_alpha_edge > 0.6:
            return StrategicPosture.DECEPTIVE

        # Majority vote with terrain tiebreaker
        from collections import Counter
        counts = Counter(postures)
        most_common = counts.most_common(1)[0]
        if most_common[1] > 1:
            return most_common[0]

        # Terrain has final say (Sun Tzu: "terrain determines strategy")
        return terrain.recommended_posture

    def _calculate_size_modifier(
        self,
        posture: StrategicPosture,
        terrain: TerrainAssessment,
        force: ForceAssessment,
        power: PowerAssessment,
    ) -> float:
        """
        Calculate position size modifier based on strategic assessment.

        Sun Tzu: "The general who wins makes many calculations in his temple
        before the battle is fought."
        """
        base_modifiers = {
            StrategicPosture.AGGRESSIVE: 1.5,
            StrategicPosture.OPPORTUNISTIC: 1.0,
            StrategicPosture.DEFENSIVE: 0.5,
            StrategicPosture.PATIENT: 0.25,
            StrategicPosture.DECEPTIVE: 0.8,
        }
        modifier = base_modifiers.get(posture, 1.0)

        # Terrain adjustments
        terrain_mod = {
            MarketTerrain.DISPERSIVE: 0.3,
            MarketTerrain.FRONTIER: 0.6,
            MarketTerrain.CONTENTIOUS: 0.8,
            MarketTerrain.OPEN: 1.2,
            MarketTerrain.INTERSECTING: 1.0,
            MarketTerrain.SERIOUS: 1.3,
            MarketTerrain.DIFFICULT: 0.4,
            MarketTerrain.HEMMED_IN: 0.5,
            MarketTerrain.DEATH: 1.5,    # all-in or cut
        }
        modifier *= terrain_mod.get(terrain.terrain, 1.0)

        # Force ratio adjustment (Law 23: Concentrate forces)
        if force.force_ratio >= 2.0:
            modifier *= 1.2    # superiority = larger size
        elif force.force_ratio < 0.5:
            modifier *= 0.5    # outmatched = reduce exposure

        # Confidence dampening
        modifier *= terrain.assessment_confidence

        # Law 40: Despise the free lunch — if setup looks too perfect, reduce
        if (force.our_alpha_edge > 0.95 and
                terrain.liquidity_depth > 0.95 and
                force.market_regime_favorability > 0.95):
            modifier *= 0.7
            logger.warning(
                "Law 40 active: Setup appears too perfect, reducing size"
            )

        return round(max(0.0, min(2.0, modifier)), 3)

    def _calculate_urgency(
        self, terrain: TerrainAssessment, force: ForceAssessment
    ) -> float:
        """
        Calculate action urgency.

        Sun Tzu: "Let your rapidity be that of the wind."
        """
        urgency = 0.5  # baseline

        # Death ground = maximum urgency
        if terrain.terrain == MarketTerrain.DEATH:
            return 1.0

        # Hemmed in = high urgency (escape before trapped)
        if terrain.terrain == MarketTerrain.HEMMED_IN:
            urgency = 0.9

        # Strong trend + good liquidity = act quickly
        if terrain.terrain == MarketTerrain.OPEN:
            urgency = max(urgency, 0.7 + abs(terrain.trend_strength) * 0.3)

        # Contentious ground — speed wins
        if terrain.terrain == MarketTerrain.CONTENTIOUS:
            urgency = max(urgency, 0.8)

        # Force superiority increases urgency (strike while strong)
        if force.force_ratio > 2.0:
            urgency = min(1.0, urgency + 0.15)

        # Dispersive ground = no urgency
        if terrain.terrain == MarketTerrain.DISPERSIVE:
            urgency = 0.1

        return round(urgency, 3)

    def _apply_art_of_war_principles(
        self,
        terrain: TerrainAssessment,
        force: ForceAssessment,
        principles: List[str],
        warnings: List[str],
    ):
        """Map active Art of War principles to the current situation."""
        # Chapter 1: Laying Plans — always active
        principles.append(
            "SunTzu.LayingPlans: 'The general who wins makes many "
            "calculations before battle'"
        )

        # Chapter 3: Attack by Stratagem
        if force.force_ratio >= 5.0:
            principles.append(
                "SunTzu.Stratagem: 'If ten times the enemy strength, surround them' "
                "→ Overwhelm with size"
            )
        elif force.force_ratio < 0.5:
            principles.append(
                "SunTzu.Stratagem: 'If fewer, retreat; if outmatched, avoid' "
                "→ Reduce exposure & wait"
            )
            warnings.append("STRATEGIC RETREAT: Force ratio unfavorable")

        # Chapter 4: Tactical Dispositions
        if terrain.terrain in (MarketTerrain.DIFFICULT, MarketTerrain.HEMMED_IN):
            principles.append(
                "SunTzu.Disposition: 'Security against defeat implies "
                "defensive tactics' → Protect capital"
            )
            warnings.append(
                "DEFENSIVE POSTURE: Difficult terrain requires capital preservation"
            )

        # Chapter 6: Weak Points and Strong (exploit regime favorability)
        if force.market_regime_favorability > 0.7:
            principles.append(
                "SunTzu.WeakPoints: 'Attack where he is unprepared, appear "
                "where you are not expected' → Exploit favorable regime"
            )

        # Chapter 7: Maneuvering
        if terrain.terrain == MarketTerrain.OPEN and terrain.trend_strength > 0.5:
            principles.append(
                "SunTzu.Maneuvering: 'Let your rapidity be that of the wind' "
                "→ Ride momentum"
            )

        # Chapter 9: The Army on the March
        if terrain.volatility_regime == "extreme":
            principles.append(
                "SunTzu.ArmyOnMarch: 'In difficult ground, press on; in "
                "encircled ground, devise stratagems' → Rapid risk reduction"
            )
            warnings.append("EXTREME VOLATILITY: Heightened risk management active")

        # Chapter 11: The Nine Situations
        if terrain.terrain == MarketTerrain.DEATH:
            principles.append(
                "SunTzu.NineSituations: 'On death ground, fight' "
                "→ Cut losses or go all-in, no half measures"
            )
            warnings.append("DEATH GROUND: Take decisive action immediately")

        # Chapter 13: The Use of Spies (intelligence gathering)
        if terrain.assessment_confidence < 0.5:
            principles.append(
                "SunTzu.Spies: 'What enables the wise sovereign to strike and "
                "conquer is foreknowledge' → Gather more intelligence before acting"
            )
            warnings.append(
                "LOW CONFIDENCE: Increase reconnaissance before committing capital"
            )

    def _apply_power_law_principles(
        self,
        power: PowerAssessment,
        principles: List[str],
        warnings: List[str],
    ):
        """Map active 48 Laws principles to the current situation."""
        for law in power.active_laws:
            if law == PowerLaw.CONCEAL_INTENTIONS:
                principles.append(
                    "Law3.ConcealIntentions: Use iceberg orders, split execution "
                    "across time and venues"
                )
            elif law == PowerLaw.SAY_LESS:
                principles.append(
                    "Law4.SayLess: Minimize market footprint, avoid large "
                    "visible orders"
                )
            elif law == PowerLaw.GUARD_REPUTATION:
                principles.append(
                    "Law5.GuardReputation: Maintain fill rates and order "
                    "discipline with exchanges"
                )
            elif law == PowerLaw.MAKE_OTHERS_COME:
                principles.append(
                    "Law8.MakeOthersCome: Prefer passive limit orders over "
                    "aggressive market orders"
                )
            elif law == PowerLaw.CRUSH_TOTALLY:
                principles.append(
                    "Law15.CrushTotally: When edge is confirmed, exploit fully "
                    "before it decays"
                )
            elif law == PowerLaw.KEEP_OTHERS_SUSPENSE:
                principles.append(
                    "Law17.Suspense: Vary execution timing and sizing to avoid "
                    "pattern detection"
                )
            elif law == PowerLaw.CONCENTRATE_FORCES:
                principles.append(
                    "Law23.ConcentrateForces: Focus capital on highest-conviction "
                    "setups, avoid dilution"
                )
            elif law == PowerLaw.ENTER_BOLDLY:
                principles.append(
                    "Law28.EnterBoldly: When entering, commit decisively — "
                    "hesitation reveals weakness"
                )
            elif law == PowerLaw.PLAN_ALL_THE_WAY:
                principles.append(
                    "Law29.PlanAllTheWay: Define entry, target, stop-loss, and "
                    "exit plan before execution"
                )
            elif law == PowerLaw.DESPISE_FREE_LUNCH:
                principles.append(
                    "Law40.DespiseFree: If a setup appears risk-free, hidden "
                    "risk exists — reduce size"
                )
            elif law == PowerLaw.KNOW_WHEN_TO_STOP:
                principles.append(
                    "Law47.KnowWhenToStop: Take profits at targets, do not "
                    "let greed extend exposure"
                )

        # Warnings from power assessment
        if power.market_footprint_stealth < 0.3:
            warnings.append(
                "POWER WARNING: Market footprint too visible — "
                "being front-run risk elevated"
            )
        if power.exchange_reputation_score < 0.5:
            warnings.append(
                "POWER WARNING: Exchange reputation degraded — "
                "may face reduced priority or limits"
            )

    # ─── STRATEGY FILTER (Art of War, Ch. 3) ─────────────────────────────

    def filter_strategies(
        self,
        directive: StrategicDirective,
        strategies: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank strategies based on current strategic directive.

        Sun Tzu: "He who knows when to fight and when not to fight will win."
        """
        filtered = []

        for strategy in strategies:
            score = self._score_strategy_alignment(directive, strategy)
            if score > 0.3:  # Minimum alignment threshold
                strategy_copy = dict(strategy)
                strategy_copy["strategic_alignment_score"] = score
                strategy_copy["strategic_posture"] = directive.overall_posture.value
                strategy_copy["terrain"] = directive.terrain.terrain.value
                filtered.append(strategy_copy)

        # Sort by alignment score descending
        filtered.sort(key=lambda s: s["strategic_alignment_score"], reverse=True)

        # Law 23: Concentrate forces — limit active strategies
        if directive.overall_posture == StrategicPosture.DEFENSIVE:
            filtered = filtered[:3]  # Minimal strategy count when defensive
        elif directive.overall_posture == StrategicPosture.PATIENT:
            filtered = filtered[:2]  # Very few when waiting
        elif directive.overall_posture == StrategicPosture.AGGRESSIVE:
            filtered = filtered[:10]  # More when attacking
        else:
            filtered = filtered[:7]  # Normal

        return filtered

    def _score_strategy_alignment(
        self, directive: StrategicDirective, strategy: Dict[str, Any]
    ) -> float:
        """Score how well a strategy aligns with the current directive."""
        score = 0.5  # baseline

        terrain = directive.terrain.terrain
        posture = directive.overall_posture

        # Strategy type alignment with terrain
        strategy_cat = strategy.get("category", "").lower()

        # Momentum strategies love open terrain
        if "momentum" in strategy_cat or "trend" in strategy_cat:
            if terrain == MarketTerrain.OPEN:
                score += 0.3
            elif terrain == MarketTerrain.DISPERSIVE:
                score -= 0.3

        # Mean reversion strategies suit contentious/frontier terrain
        if "mean_reversion" in strategy_cat or "statistical" in strategy_cat:
            if terrain in (MarketTerrain.CONTENTIOUS, MarketTerrain.FRONTIER):
                score += 0.2
            elif terrain == MarketTerrain.OPEN:
                score -= 0.2

        # Volatility strategies thrive in difficult/death terrain
        if "volatility" in strategy_cat or "vol" in strategy_cat:
            if terrain in (MarketTerrain.DIFFICULT, MarketTerrain.DEATH):
                score += 0.3

        # Market making needs intersecting terrain (multi-venue)
        if "market_making" in strategy_cat:
            if terrain == MarketTerrain.INTERSECTING:
                score += 0.3
            elif terrain in (MarketTerrain.DEATH, MarketTerrain.HEMMED_IN):
                score -= 0.4  # Never market-make on death ground

        # Arbitrage works on intersecting, open, and contentious
        if "arbitrage" in strategy_cat:
            if terrain in (
                MarketTerrain.INTERSECTING,
                MarketTerrain.OPEN,
                MarketTerrain.CONTENTIOUS,
            ):
                score += 0.2

        # Posture alignment
        confidence = strategy.get("confidence", 0.5)
        if posture == StrategicPosture.AGGRESSIVE and confidence > 0.7:
            score += 0.15
        elif posture == StrategicPosture.DEFENSIVE and confidence < 0.5:
            score += 0.1
        elif posture == StrategicPosture.PATIENT:
            score -= 0.1  # Reduce all scores when patient

        # Force ratio bonus
        if directive.force.force_ratio > 1.5:
            score += 0.1
        elif directive.force.force_ratio < 0.5:
            score -= 0.15

        return max(0.0, min(1.0, score))

    # ─── EXECUTION STYLE (48 Laws Application) ──────────────────────────

    def get_execution_style(
        self, directive: StrategicDirective
    ) -> Dict[str, Any]:
        """
        Determine how orders should be executed based on power laws.

        Returns execution parameters to pass to the order execution engine.
        """
        style: Dict[str, Any] = {
            "use_iceberg_orders": False,
            "split_across_venues": False,
            "randomize_timing": False,
            "prefer_passive": False,
            "max_visible_size_pct": 100,
            "time_slicing": False,
            "time_slice_count": 1,
        }

        active = {law for law in directive.power.active_laws}

        # Law 3: Conceal intentions — iceberg orders
        if PowerLaw.CONCEAL_INTENTIONS in active:
            style["use_iceberg_orders"] = True
            style["max_visible_size_pct"] = 20

        # Law 4: Say less — split across venues
        if PowerLaw.SAY_LESS in active:
            style["split_across_venues"] = True
            style["max_visible_size_pct"] = min(
                style["max_visible_size_pct"], 15
            )

        # Law 8: Make others come to you — passive limit orders
        if PowerLaw.MAKE_OTHERS_COME in active:
            style["prefer_passive"] = True

        # Law 17: Keep in suspense — randomize timing
        if PowerLaw.KEEP_OTHERS_SUSPENSE in active:
            style["randomize_timing"] = True
            style["time_slicing"] = True
            style["time_slice_count"] = 5

        # Override for death ground — speed over stealth
        if directive.terrain.terrain == MarketTerrain.DEATH:
            style["use_iceberg_orders"] = False
            style["randomize_timing"] = False
            style["prefer_passive"] = False
            style["max_visible_size_pct"] = 100

        return style

    # ─── RISK OVERLAY (Combined Doctrine) ────────────────────────────────

    def get_risk_overlay(
        self, directive: StrategicDirective
    ) -> Dict[str, Any]:
        """
        Generate risk management overlay based on strategic assessment.

        Sun Tzu: "The art of war teaches us to rely not on the likelihood of
        the enemy's not coming, but on our own readiness to receive him."
        """
        overlay: Dict[str, Any] = {
            "max_position_size_multiplier": directive.position_size_modifier,
            "stop_loss_tightness": "normal",
            "take_profit_aggression": "normal",
            "max_concurrent_positions": 10,
            "correlation_limit": 0.7,
            "max_sector_exposure_pct": 30.0,
        }

        posture = directive.overall_posture
        terrain = directive.terrain.terrain

        # Defensive posture tightens everything
        if posture == StrategicPosture.DEFENSIVE:
            overlay["stop_loss_tightness"] = "tight"
            overlay["max_concurrent_positions"] = 5
            overlay["correlation_limit"] = 0.5
            overlay["max_sector_exposure_pct"] = 20.0

        # Patient posture — minimal exposure
        elif posture == StrategicPosture.PATIENT:
            overlay["stop_loss_tightness"] = "very_tight"
            overlay["max_concurrent_positions"] = 3
            overlay["correlation_limit"] = 0.3
            overlay["max_sector_exposure_pct"] = 15.0

        # Aggressive posture — wider stops, more positions
        elif posture == StrategicPosture.AGGRESSIVE:
            overlay["stop_loss_tightness"] = "wide"
            overlay["take_profit_aggression"] = "aggressive"
            overlay["max_concurrent_positions"] = 15
            overlay["max_sector_exposure_pct"] = 40.0

        # Death ground override
        if terrain == MarketTerrain.DEATH:
            overlay["stop_loss_tightness"] = "immediate"
            overlay["max_concurrent_positions"] = 2
            overlay["take_profit_aggression"] = "immediate"

        # Hemmed-in ground — prepare escape
        if terrain == MarketTerrain.HEMMED_IN:
            overlay["stop_loss_tightness"] = "tight"
            overlay["max_concurrent_positions"] = 4

        return overlay

    # ─── METRICS FOR DOCTRINE PACKS ──────────────────────────────────────

    def get_doctrine_metrics(self) -> Dict[str, float]:
        """Return metrics for Doctrine Packs 9 and 10."""
        # Get latest assessments or defaults
        latest_terrain = (
            self.terrain_history[-1]
            if self.terrain_history
            else TerrainAssessment(
                terrain=MarketTerrain.DISPERSIVE,
                volatility_regime="normal",
                liquidity_depth=0.7,
                trend_strength=0.0,
                support_resistance_proximity=0.5,
                time_of_day_advantage=0.7,
                assessment_confidence=0.75,
            )
        )
        latest_force = (
            self.force_history[-1]
            if self.force_history
            else ForceAssessment(
                our_capital_strength=0.7,
                our_position_diversity=0.7,
                our_alpha_edge=0.7,
                our_execution_speed=0.7,
                market_adversary_strength=0.5,
                market_regime_favorability=0.7,
            )
        )
        latest_power = (
            self.power_history[-1]
            if self.power_history
            else PowerAssessment(
                market_footprint_stealth=0.75,
                exchange_reputation_score=0.85,
                alpha_source_uniqueness=0.65,
                strategy_unpredictability=0.7,
                capital_concentration_ratio=0.5,
            )
        )

        return {
            # Pack 9 metrics: Strategic Warfare
            "terrain_favorability": (
                latest_terrain.liquidity_depth * 0.4
                + latest_terrain.time_of_day_advantage * 0.3
                + (1.0 - abs(latest_terrain.trend_strength - 0.5)) * 0.3
            ),
            "force_ratio": latest_force.force_ratio,
            "strategic_confidence": latest_terrain.assessment_confidence,
            "posture_alignment": 1.0 if self.active_posture != StrategicPosture.PATIENT else 0.8,
            # Pack 10 metrics: Power Dynamics
            "market_stealth_score": latest_power.market_footprint_stealth,
            "exchange_reputation": latest_power.exchange_reputation_score,
            "alpha_uniqueness": latest_power.alpha_source_uniqueness,
            "execution_unpredictability": latest_power.strategy_unpredictability,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL ACCESSOR
# ═══════════════════════════════════════════════════════════════════════════

_strategic_engine: Optional[StrategicDoctrineEngine] = None


def get_strategic_doctrine_engine() -> StrategicDoctrineEngine:
    """Get or create the singleton StrategicDoctrineEngine."""
    global _strategic_engine
    if _strategic_engine is None:
        _strategic_engine = StrategicDoctrineEngine()
    return _strategic_engine


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY-AWARE DOCTRINE ENGINE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyDirective:
    """Per-strategy directive issued by the doctrine engine."""
    strategy_name: str
    allowed: bool = True
    position_size_pct: float = 1.0     # fraction of normal allocation
    bias: str = "neutral"              # "aggressive" | "defensive" | "neutral"
    max_positions: int = 10
    notes: str = ""


class StrategyAwareDoctrine:
    """
    Doctrine engine that synthesises:
    1. All 7 active strategy signals (War Room, Lifeboat, etc.)
    2. Manual input from owner (override commands)
    3. Market regime from MarketIntelligenceModel
    4. NCL BRAIN intelligence (relayed back)

    Output: Dynamic position sizing, sector allocation, risk posture
    per strategy -- NOT just a single multiplier.
    """

    STRATEGY_DOCTRINE_MAP: Dict[str, Dict[str, str]] = {
        "war_room": {"regime": "crisis", "bias": "defensive"},
        "storm_lifeboat": {"regime": "lunar", "bias": "cyclical"},
        "capital_engine": {"regime": "commodity", "bias": "rotation"},
        "matrix_maximizer": {"regime": "options", "bias": "premium"},
        "exploitation_matrix": {"regime": "blackswan", "bias": "conviction"},
        "polymarket": {"regime": "event", "bias": "probability"},
        "blackswan_authority": {"regime": "expert", "bias": "consensus"},
    }

    def __init__(self):
        self._base_engine = get_strategic_doctrine_engine()
        self._manual_overrides: Dict[str, Any] = {}
        self._ncl_intel: Dict[str, Any] = {}
        self._last_regime: str = "normal"

    def set_manual_override(self, strategy_name: str, override: Dict[str, Any]) -> None:
        """Owner manual override for a specific strategy."""
        self._manual_overrides[strategy_name] = override
        logger.info("Manual override set for %s: %s", strategy_name, override)

    def clear_manual_override(self, strategy_name: str) -> None:
        """Remove manual override."""
        self._manual_overrides.pop(strategy_name, None)

    def update_ncl_intel(self, intel: Dict[str, Any]) -> None:
        """Receive intelligence from NCL BRAIN."""
        self._ncl_intel = intel

    def update_regime(self, regime: str) -> None:
        """Update market regime from MarketIntelligenceModel."""
        self._last_regime = regime

    def generate_composite_directive(
        self,
        strategy_signals: Dict[str, Dict[str, Any]],
    ) -> Dict[str, StrategyDirective]:
        """
        Synthesise all inputs into per-strategy directives.

        Args:
            strategy_signals: {strategy_key: {signal_data}} from each active strategy

        Returns:
            {strategy_key: StrategyDirective} with per-strategy sizing/bias/limits
        """
        directives: Dict[str, StrategyDirective] = {}

        for strat_key, doc_map in self.STRATEGY_DOCTRINE_MAP.items():
            signal = strategy_signals.get(strat_key, {})
            manual = self._manual_overrides.get(strat_key, {})

            # Start with defaults
            directive = StrategyDirective(strategy_name=strat_key)

            # --- Regime alignment ---
            expected_regime = doc_map["regime"]
            regime_match = self._regime_alignment(expected_regime, self._last_regime)
            directive.position_size_pct = regime_match

            # --- Strategy-specific bias ---
            doc_bias = doc_map["bias"]
            if doc_bias == "defensive" and self._last_regime in ("crisis", "crash"):
                directive.position_size_pct = min(1.5, directive.position_size_pct * 1.3)
                directive.bias = "aggressive"
                directive.notes = "Crisis regime favours defensive strategies"
            elif doc_bias == "premium" and self._last_regime in ("normal", "bull"):
                directive.position_size_pct *= 1.2
                directive.bias = "aggressive"
            else:
                directive.bias = "neutral"

            # --- NCL Intelligence adjustment ---
            ncl_caution = self._ncl_intel.get("caution_level", 0)
            if ncl_caution > 0.7:
                directive.position_size_pct *= 0.5
                directive.max_positions = 3
                directive.notes += " | NCL BRAIN: HIGH CAUTION"
            elif ncl_caution > 0.4:
                directive.position_size_pct *= 0.8
                directive.max_positions = 7

            # --- Signal strength from strategy itself ---
            sig_confidence = signal.get("confidence", 0.5)
            directive.position_size_pct *= (0.5 + sig_confidence * 0.5)

            # --- Manual override (final authority) ---
            if manual:
                if "allowed" in manual:
                    directive.allowed = bool(manual["allowed"])
                if "position_size_pct" in manual:
                    directive.position_size_pct = float(manual["position_size_pct"])
                if "bias" in manual:
                    directive.bias = str(manual["bias"])
                if "max_positions" in manual:
                    directive.max_positions = int(manual["max_positions"])
                directive.notes += " | MANUAL OVERRIDE ACTIVE"

            # Clamp
            directive.position_size_pct = round(max(0.0, min(2.0, directive.position_size_pct)), 3)
            directives[strat_key] = directive

        return directives

    def _regime_alignment(self, expected: str, current: str) -> float:
        """
        Score how well the current regime aligns with a strategy's expected regime.
        Returns 0.3 (poor) to 1.5 (perfect match).
        """
        alignment_matrix: Dict[str, Dict[str, float]] = {
            "crisis":    {"crisis": 1.5, "crash": 1.3, "bear": 1.0, "normal": 0.5, "bull": 0.3},
            "lunar":     {"crisis": 0.8, "crash": 0.6, "bear": 0.8, "normal": 1.0, "bull": 1.0},
            "commodity": {"crisis": 1.2, "crash": 0.8, "bear": 0.9, "normal": 1.0, "bull": 0.7},
            "options":   {"crisis": 1.3, "crash": 1.5, "bear": 1.2, "normal": 1.0, "bull": 0.8},
            "blackswan": {"crisis": 1.5, "crash": 1.5, "bear": 1.0, "normal": 0.4, "bull": 0.3},
            "event":     {"crisis": 1.0, "crash": 1.0, "bear": 0.8, "normal": 0.8, "bull": 0.8},
            "expert":    {"crisis": 1.2, "crash": 1.0, "bear": 0.9, "normal": 0.7, "bull": 0.6},
        }
        row = alignment_matrix.get(expected, {})
        return row.get(current, 0.7)

    def get_doctrine_state(self) -> Dict[str, Any]:
        """Return current doctrine state for display/relay."""
        return {
            "regime": self._last_regime,
            "manual_overrides": list(self._manual_overrides.keys()),
            "ncl_caution": self._ncl_intel.get("caution_level", 0),
            "strategy_count": len(self.STRATEGY_DOCTRINE_MAP),
        }


# Module-level accessor
_strategy_aware_doctrine: Optional[StrategyAwareDoctrine] = None


def get_strategy_aware_doctrine() -> StrategyAwareDoctrine:
    """Get or create the singleton StrategyAwareDoctrine."""
    global _strategy_aware_doctrine
    if _strategy_aware_doctrine is None:
        _strategy_aware_doctrine = StrategyAwareDoctrine()
    return _strategy_aware_doctrine

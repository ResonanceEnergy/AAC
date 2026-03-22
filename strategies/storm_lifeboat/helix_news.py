"""
Storm Lifeboat Matrix — Helix News Briefing Generator
=======================================================
Generates daily intelligence briefings combining all Storm Lifeboat
subsystems into a single actionable report.

Briefing structure:
    1. HEADLINE — One-line summary of the day's posture
    2. REGIME STATUS — Current volatility regime and mandate level
    3. LUNAR POSITION — 13-moon phase, phi windows, sizing multiplier
    4. COHERENCE READING — PlanckPhire score with interpretation
    5. SCENARIO HEATMAP — Top active scenarios ranked by risk score
    6. PORTFOLIO OUTLOOK — MC simulation key metrics
    7. TOP TRADES — Recommended actions for today
    8. RISK ALERTS — Critical warnings

Output formats: plain text (terminal), JSON (storage/API)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.storm_lifeboat.core import (
    Asset,
    HelixBriefing,
    MandateLevel,
    MoonPhase,
    PortfolioForecast,
    ScenarioState,
    VolRegime,
)

logger = logging.getLogger(__name__)

# Mandate action guidance
_MANDATE_GUIDANCE = {
    MandateLevel.OBSERVE: "STAND DOWN — Monitor only, no new positions",
    MandateLevel.DEFENSIVE: "HEDGE MODE — Protect existing positions, buy protective puts",
    MandateLevel.STANDARD: "ACTIVE — Execute standard bearish playbook, 1x position sizing",
    MandateLevel.AGGRESSIVE: "PRESS IT — High conviction entries, 1.5x sizing, scale into weakness",
    MandateLevel.MAX_CONVICTION: "ALL IN — Maximum position sizing, tail risk confirmed, deploy reserves",
}

# Regime descriptions
_REGIME_DESCRIPTIONS = {
    VolRegime.CALM: "Markets orderly, risk-on dominant. Puts are cheap — accumulate hedges.",
    VolRegime.ELEVATED: "Stress building. Protect downside, trim risk-on.",
    VolRegime.CRISIS: "Active crisis. Execute bearish playbook, expect dislocations.",
    VolRegime.PANIC: "Panic selling underway. Maximum divergence — protect capital, harvest vol.",
}


def _generate_headline(
    regime: VolRegime,
    mandate: MandateLevel,
    coherence: float,
    n_active_scenarios: int,
) -> str:
    """Generate a one-line headline for the briefing."""
    coherence_word = "HIGH" if coherence > 0.7 else "LOW" if coherence < 0.3 else "MODERATE"
    return (
        f"STORM LIFEBOAT | Regime: {regime.value.upper()} | "
        f"Mandate: {mandate.value.upper()} | "
        f"Coherence: {coherence_word} ({coherence:.2f}) | "
        f"{n_active_scenarios} scenarios active"
    )


def _generate_top_trades(
    mandate: MandateLevel,
    forecast: Optional[PortfolioForecast],
    moon_phase: MoonPhase,
    position_multiplier: float,
) -> List[str]:
    """Generate top trade recommendations based on current state."""
    trades: List[str] = []

    if mandate == MandateLevel.OBSERVE:
        trades.append("No new positions — review thesis, run scenario updates")
        return trades

    # Phase-specific guidance
    phase_action = {
        MoonPhase.NEW: "Reset cycle — close losers, reassess scenario weights",
        MoonPhase.WAXING: f"Accumulation phase — scale in at {position_multiplier:.1f}x sizing",
        MoonPhase.FULL: f"Execution phase — max conviction entries at {position_multiplier:.1f}x",
        MoonPhase.WANING: "Protection phase — roll winners, trim losers, buy hedges",
    }
    trades.append(phase_action[moon_phase])

    if forecast:
        # Find highest-conviction short candidates (highest P(down 10%))
        short_candidates = sorted(
            forecast.asset_forecasts.values(),
            key=lambda f: f.prob_down_10,
            reverse=True,
        )[:3]

        for fc in short_candidates:
            if fc.prob_down_10 > 0.3:
                trades.append(
                    f"PUT {fc.asset.value}: P(10%% down)={fc.prob_down_10:.0%}, "
                    f"VaR95={fc.var_95:.1%}, expected={fc.expected_return_pct:+.1f}%%"
                )

        # Find highest-conviction long candidates (highest P(up 10%))
        long_candidates = sorted(
            forecast.asset_forecasts.values(),
            key=lambda f: f.prob_up_10,
            reverse=True,
        )[:2]

        for fc in long_candidates:
            if fc.prob_up_10 > 0.25:
                trades.append(
                    f"LONG {fc.asset.value}: P(10%% up)={fc.prob_up_10:.0%}, "
                    f"expected={fc.expected_return_pct:+.1f}%%"
                )

    return trades


def _generate_risk_alert(
    forecast: Optional[PortfolioForecast],
    coherence: float,
    n_escalating: int,
) -> Optional[str]:
    """Generate critical risk alert if warranted."""
    alerts = []

    if forecast and forecast.portfolio_cvar_95 > 0.25:
        alerts.append(f"Portfolio CVaR95 = {forecast.portfolio_cvar_95:.1%} — EXTREME risk")

    if n_escalating >= 3:
        alerts.append(f"{n_escalating} scenarios ESCALATING — contagion risk elevated")

    if coherence > 0.85:
        alerts.append("Coherence > 0.85 — strong trend but reversal risk at extremes")

    if coherence < 0.15:
        alerts.append("Coherence < 0.15 — maximum chaos, expect whipsaws, reduce sizing")

    if not alerts:
        return None
    return " | ".join(alerts)


class HelixNewsGenerator:
    """Generates daily Helix News briefings from all subsystem inputs."""

    def generate(
        self,
        forecast: Optional[PortfolioForecast] = None,
        scenario_states: Optional[Dict[str, ScenarioState]] = None,
        coherence_score: float = 0.5,
        moon_phase: MoonPhase = MoonPhase.NEW,
        position_multiplier: float = 1.0,
        regime: VolRegime = VolRegime.CRISIS,
        mandate: Optional[MandateLevel] = None,
    ) -> HelixBriefing:
        """Generate a complete daily briefing.

        Args:
            forecast: Latest MC simulation results
            scenario_states: Current scenario tracking states
            coherence_score: Latest PlanckPhire coherence
            moon_phase: Current 13-moon phase
            position_multiplier: Current phi-adjusted sizing
            regime: Current volatility regime
            mandate: Override mandate (derived from forecast if None)

        Returns:
            HelixBriefing dataclass
        """
        mandate = mandate or (forecast.mandate if forecast else MandateLevel.OBSERVE)

        # Build active scenario list
        active_states: List[ScenarioState] = []
        n_escalating = 0
        if scenario_states:
            from strategies.storm_lifeboat.core import ScenarioStatus
            for state in scenario_states.values():
                if state.status != ScenarioStatus.DORMANT:
                    active_states.append(state)
                if state.status in (ScenarioStatus.ESCALATING, ScenarioStatus.PEAK):
                    n_escalating += 1
        # Sort by probability descending
        active_states.sort(key=lambda s: s.probability, reverse=True)

        headline = _generate_headline(regime, mandate, coherence_score, len(active_states))
        top_trades = _generate_top_trades(mandate, forecast, moon_phase, position_multiplier)
        risk_alert = _generate_risk_alert(forecast, coherence_score, n_escalating)

        # Portfolio summary
        if forecast:
            portfolio_summary = (
                f"MC {forecast.n_simulations:,} paths / {forecast.horizon_days}d: "
                f"Expected return {forecast.weighted_return_pct:+.1f}%, "
                f"VaR95 {forecast.portfolio_var_95:.1%}, "
                f"CVaR95 {forecast.portfolio_cvar_95:.1%}"
            )
        else:
            portfolio_summary = "No simulation run — execute MC to generate portfolio metrics"

        return HelixBriefing(
            date=date.today(),
            headline=headline,
            regime=regime,
            mandate=mandate,
            moon_phase=moon_phase,
            coherence_score=coherence_score,
            active_scenarios=active_states,
            top_trades=top_trades,
            portfolio_summary=portfolio_summary,
            risk_alert=risk_alert,
        )

    def format_terminal(self, briefing: HelixBriefing) -> str:
        """Format briefing for terminal display (ASCII-safe)."""
        lines = [
            "",
            "=" * 78,
            f"  HELIX NEWS DAILY BRIEFING — {briefing.date.isoformat()}",
            "=" * 78,
            "",
            f"  {briefing.headline}",
            "",
            "-" * 78,
            f"  REGIME: {briefing.regime.value.upper()}",
            f"  {_REGIME_DESCRIPTIONS[briefing.regime]}",
            "",
            f"  MANDATE: {briefing.mandate.value.upper()}",
            f"  {_MANDATE_GUIDANCE[briefing.mandate]}",
            "",
            f"  MOON: {briefing.moon_phase.value.upper()} | Coherence: {briefing.coherence_score:.2f}",
            "",
            "-" * 78,
            "  ACTIVE SCENARIOS:",
        ]

        if briefing.active_scenarios:
            for sc in briefing.active_scenarios[:8]:
                lines.append(
                    f"    [{sc.status.value.upper():12s}] {sc.code:20s} "
                    f"P={sc.probability:.0%}  "
                    f"Indicators: {len(sc.indicators_firing)} firing"
                )
        else:
            lines.append("    No scenarios active — all dormant")

        lines.extend([
            "",
            "-" * 78,
            "  PORTFOLIO:",
            f"    {briefing.portfolio_summary}",
            "",
            "-" * 78,
            "  TOP TRADES:",
        ])

        for i, trade in enumerate(briefing.top_trades, 1):
            lines.append(f"    {i}. {trade}")

        if briefing.risk_alert:
            lines.extend([
                "",
                "-" * 78,
                f"  !! RISK ALERT: {briefing.risk_alert}",
            ])

        lines.extend([
            "",
            "=" * 78,
            "",
        ])

        return "\n".join(lines)

    def save_json(self, briefing: HelixBriefing, directory: str = "data/storm_lifeboat") -> str:
        """Save briefing to JSON file. Returns the filepath."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"helix_briefing_{briefing.date.isoformat()}.json"
        filepath = out_dir / filename

        data = {
            "date": briefing.date.isoformat(),
            "headline": briefing.headline,
            "regime": briefing.regime.value,
            "mandate": briefing.mandate.value,
            "moon_phase": briefing.moon_phase.value,
            "coherence_score": briefing.coherence_score,
            "top_trades": briefing.top_trades,
            "portfolio_summary": briefing.portfolio_summary,
            "risk_alert": briefing.risk_alert,
            "active_scenarios": [
                {
                    "code": sc.code,
                    "status": sc.status.value,
                    "probability": sc.probability,
                    "indicators_firing": sc.indicators_firing,
                }
                for sc in briefing.active_scenarios
            ],
        }

        filepath.write_text(json.dumps(data, indent=2))
        logger.info("Helix briefing saved to %s", filepath)
        return str(filepath)

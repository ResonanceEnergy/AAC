#!/usr/bin/env python3
"""
Storm Lifeboat Matrix v9.0 — CLI Runner
=========================================
Unified entry point for all Storm Lifeboat subsystems.

Usage:
    python -m strategies.storm_lifeboat.runner              # Full briefing
    python -m strategies.storm_lifeboat.runner --monte-carlo # Run MC simulation
    python -m strategies.storm_lifeboat.runner --scenarios    # Scenario heatmap
    python -m strategies.storm_lifeboat.runner --lunar        # Lunar position
    python -m strategies.storm_lifeboat.runner --coherence    # Coherence score
    python -m strategies.storm_lifeboat.runner --put SPY      # Put payoff sim
    python -m strategies.storm_lifeboat.runner --briefing     # Helix News briefing
    python -m strategies.storm_lifeboat.runner --all          # Everything
    python -m strategies.storm_lifeboat.runner --json         # JSON output
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date

# Windows cp1252 stdout fix
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print()


def cmd_monte_carlo(args: argparse.Namespace) -> dict:
    """Run Monte Carlo simulation."""
    from strategies.storm_lifeboat.core import StormConfig, VolRegime
    from strategies.storm_lifeboat.monte_carlo import StormMonteCarloEngine
    from strategies.storm_lifeboat.scenario_engine import ScenarioEngine

    config = StormConfig(
        n_simulations=args.paths,
        horizon_days=args.horizon,
        regime=VolRegime(args.regime),
    )
    engine = StormMonteCarloEngine(config)
    scenario_eng = ScenarioEngine()
    active = scenario_eng.get_active_scenarios()

    t0 = time.perf_counter()
    live_snap = getattr(args, '_live_snapshot', None)
    forecast = engine.simulate(
        prices=live_snap.prices if live_snap else None,
        vix=args.vix,
        active_scenarios=active,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if not args.json:
        _print_header(f"MONTE CARLO — {config.n_simulations:,} paths / {config.horizon_days}d")
        print(f"  Regime: {forecast.regime.value.upper()}")
        print(f"  Mandate: {forecast.mandate.value.upper()}")
        print(f"  Runtime: {elapsed:.0f}ms")
        print(f"  Portfolio VaR95: {forecast.portfolio_var_95:.2%}")
        print(f"  Portfolio CVaR95: {forecast.portfolio_cvar_95:.2%}")
        print(f"  Weighted Return: {forecast.weighted_return_pct:+.2f}%")
        print()
        print(f"  {'Asset':>8s}  {'Current':>10s}  {'Mean':>10s}  {'P5':>10s}  "
              f"{'P95':>10s}  {'E[R]':>8s}  {'P(↓10%)':>8s}  {'VaR95':>8s}")
        print("  " + "-" * 76)
        for asset, fc in sorted(forecast.asset_forecasts.items(), key=lambda x: x[1].expected_return_pct):
            print(f"  {asset.value:>8s}  {fc.current_price:>10.2f}  {fc.mean_price:>10.2f}  "
                  f"{fc.pct_5:>10.2f}  {fc.pct_95:>10.2f}  {fc.expected_return_pct:>+7.1f}%  "
                  f"{fc.prob_down_10:>7.0%}  {fc.var_95:>7.1%}")
        print()

    return {"elapsed_ms": elapsed, "forecast": forecast}


def cmd_scenarios(args: argparse.Namespace) -> dict:
    """Display scenario heatmap."""
    from strategies.storm_lifeboat.scenario_engine import ScenarioEngine

    engine = ScenarioEngine()

    # If live data available, fire indicators from news/data
    live_snap = getattr(args, '_live_snapshot', None)
    if live_snap and live_snap.firing_indicators:
        for code, indicators in live_snap.firing_indicators.items():
            engine.update_indicators(code, indicators)

    heatmap = engine.get_risk_heatmap()

    if not args.json:
        _print_header("43-SCENARIO RISK HEATMAP")
        print(f"  {'#':>2s}  {'Code':>20s}  {'Status':>12s}  {'Prob':>6s}  "
              f"{'Severity':>8s}  {'Risk':>6s}  {'Indicators':>10s}")
        print("  " + "-" * 70)
        for sc in heatmap:
            print(f"  {sc['id']:>2d}  {sc['code']:>20s}  {sc['status']:>12s}  "
                  f"{sc['probability']:>5.0%}  {sc['severity']:>8.2f}  "
                  f"{sc['risk_score']:>5.3f}  {sc['indicators_firing']}/{sc['indicators_total']}")
        print()

    return {"heatmap": heatmap}


def cmd_lunar(args: argparse.Namespace) -> dict:
    """Display 13-moon lunar position."""
    from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine

    engine = LunarPhiEngine()
    pos = engine.get_position()
    next_phi = engine.get_next_phi_window()

    if not args.json:
        _print_header("13-MOON PHI CYCLE")
        print(f"  {engine.format_display(pos)}")
        print()
        print(f"  Cycle Start:      {pos.cycle_start.isoformat()}")
        print(f"  Day in Cycle:     {pos.day_in_cycle}/364")
        print(f"  Moon:             {pos.moon_number}/13 — {pos.moon_name}")
        print(f"  Day in Moon:      {pos.day_in_moon}/28")
        print(f"  Phase:            {pos.phase.value.upper()}")
        print(f"  Phi Window:       {'YES' if pos.in_phi_window else 'no'}")
        print(f"  Phi Coherence:    {pos.phi_coherence:.4f}")
        print(f"  Position Sizing:  {pos.position_multiplier:.2f}x")
        print(f"  Next Phi Window:  {next_phi.isoformat()}")
        print()

    return {
        "moon": pos.moon_number,
        "moon_name": pos.moon_name,
        "day": pos.day_in_moon,
        "phase": pos.phase.value,
        "phi_window": pos.in_phi_window,
        "phi_coherence": pos.phi_coherence,
        "multiplier": pos.position_multiplier,
        "next_phi": next_phi.isoformat(),
    }


def cmd_coherence(args: argparse.Namespace) -> dict:
    """Run PlanckPhire coherence analysis."""
    from strategies.storm_lifeboat.coherence import CoherenceEngine
    from strategies.storm_lifeboat.core import VolRegime
    from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine
    from strategies.storm_lifeboat.scenario_engine import ScenarioEngine

    lunar = LunarPhiEngine()
    pos = lunar.get_position()
    scenario_eng = ScenarioEngine()
    active = scenario_eng.get_active_scenarios()
    regime = VolRegime(args.regime)

    coherence_eng = CoherenceEngine()
    result = coherence_eng.analyze(
        active_scenarios=active,
        moon_phase=pos.phase,
        lunar_phi_coherence=pos.phi_coherence,
        current_regime=regime,
    )

    if not args.json:
        _print_header("PLANCKPHIRE COHERENCE ANALYSIS")
        print(f"  Overall Score:      {result.overall_score:.4f}")
        print(f"  Harmonic Ratio:     {result.harmonic_ratio:.4f}")
        print(f"  Scenario Alignment: {result.scenario_alignment:.4f}")
        print(f"  Lunar Alignment:    {result.lunar_alignment:.4f}")
        print(f"  Regime Stability:   {result.regime_stability:.4f}")
        print(f"  Dominant Frequency: {result.dominant_frequency}")
        print(f"  Confidence:         {result.confidence:.0%}")
        print()

    return {
        "overall": result.overall_score,
        "harmonic": result.harmonic_ratio,
        "scenario_alignment": result.scenario_alignment,
        "lunar_alignment": result.lunar_alignment,
        "regime_stability": result.regime_stability,
        "frequency": result.dominant_frequency,
        "confidence": result.confidence,
    }


def cmd_put(args: argparse.Namespace) -> dict:
    """Simulate put option payoff."""
    from strategies.storm_lifeboat.core import Asset, StormConfig, VolRegime
    from strategies.storm_lifeboat.monte_carlo import StormMonteCarloEngine

    asset = Asset(args.put.upper())
    config = StormConfig(
        n_simulations=args.paths,
        horizon_days=args.horizon,
        regime=VolRegime(args.regime),
    )
    engine = StormMonteCarloEngine(config)
    result = engine.simulate_put_payoff(
        asset=asset,
        strike_pct_otm=args.otm / 100.0,
        premium_per_share=args.premium,
        n_contracts=args.contracts,
    )

    if not args.json:
        _print_header(f"PUT PAYOFF SIMULATION — {asset.value}")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k:>25s}: {v:>12.2f}")
            else:
                print(f"  {k:>25s}: {v}")
        print()

    return result


def cmd_briefing(args: argparse.Namespace) -> dict:
    """Generate full Helix News briefing."""
    from strategies.storm_lifeboat.coherence import CoherenceEngine
    from strategies.storm_lifeboat.core import StormConfig, VolRegime
    from strategies.storm_lifeboat.helix_news import HelixNewsGenerator
    from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine
    from strategies.storm_lifeboat.monte_carlo import StormMonteCarloEngine
    from strategies.storm_lifeboat.scenario_engine import ScenarioEngine

    regime = VolRegime(args.regime)
    config = StormConfig(
        n_simulations=args.paths,
        horizon_days=args.horizon,
        regime=regime,
    )

    # Run all subsystems
    lunar = LunarPhiEngine()
    pos = lunar.get_position()

    scenario_eng = ScenarioEngine()

    # Fire indicators from live feed if available
    live_snap = getattr(args, '_live_snapshot', None)
    if live_snap and live_snap.firing_indicators:
        for code, indicators in live_snap.firing_indicators.items():
            scenario_eng.update_indicators(code, indicators)

    active = scenario_eng.get_active_scenarios()

    mc = StormMonteCarloEngine(config)
    forecast = mc.simulate(
        prices=live_snap.prices if live_snap else None,
        vix=args.vix,
        active_scenarios=active,
        moon_phase=pos.phase,
    )

    coherence_eng = CoherenceEngine()
    coherence = coherence_eng.analyze(
        active_scenarios=active,
        moon_phase=pos.phase,
        lunar_phi_coherence=pos.phi_coherence,
        current_regime=regime,
    )

    # Generate briefing
    helix = HelixNewsGenerator()
    briefing = helix.generate(
        forecast=forecast,
        scenario_states=scenario_eng.states,
        coherence_score=coherence.overall_score,
        moon_phase=pos.phase,
        position_multiplier=pos.position_multiplier,
        regime=regime,
        mandate=forecast.mandate,
    )

    if not args.json:
        print(helix.format_terminal(briefing))
    else:
        filepath = helix.save_json(briefing)
        print(json.dumps({"saved": filepath}, indent=2))

    return {"briefing_date": briefing.date.isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Storm Lifeboat Matrix v9.0 — Crisis Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--monte-carlo", action="store_true", help="Run MC simulation")
    parser.add_argument("--scenarios", action="store_true", help="Scenario heatmap")
    parser.add_argument("--lunar", action="store_true", help="Lunar phi position")
    parser.add_argument("--coherence", action="store_true", help="PlanckPhire coherence")
    parser.add_argument("--put", type=str, metavar="ASSET", help="Put payoff sim for asset")
    parser.add_argument("--briefing", action="store_true", help="Full Helix News briefing")
    parser.add_argument("--capital-engine", action="store_true",
                        help="Run one cycle of the gold-oil-silver see-saw Capital Engine")
    parser.add_argument("--capital-loop", action="store_true",
                        help="Run the Capital Engine in continuous hourly mode")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--paths", type=int, default=100_000, help="MC paths (default: 100000)")
    parser.add_argument("--horizon", type=int, default=90, help="MC horizon days (default: 90)")
    parser.add_argument("--regime", type=str, default="crisis",
                        choices=["calm", "elevated", "crisis", "panic"],
                        help="Volatility regime (default: crisis)")
    parser.add_argument("--vix", type=float, default=25.0, help="VIX level (default: 25)")
    parser.add_argument("--otm", type=float, default=10.0,
                        help="Put strike %% OTM (default: 10)")
    parser.add_argument("--premium", type=float, default=5.0,
                        help="Put premium per share (default: 5.0)")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Number of put contracts (default: 1)")
    parser.add_argument("--live", action="store_true",
                        help="Fetch live prices/VIX/sentiment from all data sources")
    parser.add_argument("--ibkr", action="store_true",
                        help="Also fetch from IBKR TWS (requires TWS connected)")
    parser.add_argument("--moomoo", action="store_true",
                        help="Also fetch from Moomoo OpenD (requires OpenD running)")

    args = parser.parse_args()

    # Default to briefing if no command specified
    if not any([args.monte_carlo, args.scenarios, args.lunar,
                args.coherence, args.put, args.briefing, args.all,
                args.capital_engine, args.capital_loop]):
        args.briefing = True

    # ── Live data feed ──
    if args.live:
        from strategies.storm_lifeboat.live_feed import get_live_snapshot
        _print_header("LIVE DATA FEED — fetching from all sources")
        snap = get_live_snapshot(
            include_ibkr=args.ibkr,
            include_moomoo=args.moomoo,
        )
        # Override VIX and regime from live data
        args.vix = snap.vix
        args.regime = snap.regime.value
        # Stash snapshot on args for sub-commands to use
        args._live_snapshot = snap

        if not args.json:
            print(f"  VIX:            {snap.vix:.1f} ({snap.regime.value.upper()})")
            print(f"  Fear & Greed:   {snap.fear_greed} ({snap.fear_greed_label})")
            print(f"  P/C Ratio:      {snap.put_call_ratio:.2f}")
            print(f"  Market Tone:    {snap.market_tone}")
            print(f"  UW Signals:     {snap.options_flow_signal_count}")
            print(f"  Top Flow:       {', '.join(snap.top_flow_tickers[:8]) or 'N/A'}")
            n_live = sum(
                1 for a in snap.prices
                if snap.prices[a] != __import__('strategies.storm_lifeboat.core', fromlist=['DEFAULT_PRICES']).DEFAULT_PRICES.get(a)
            )
            print(f"  Live Prices:    {n_live}/{len(snap.prices)}")
            if snap.firing_indicators:
                print(f"  Indicators:     {', '.join(snap.firing_indicators.keys())}")
            if snap.trend_interest:
                top_term = max(snap.trend_interest, key=snap.trend_interest.get)
                print(f"  Top Trend:      {top_term} ({snap.trend_interest[top_term]})")
            print(f"  Sources OK:     {', '.join(snap.sources_ok)}")
            if snap.sources_failed:
                print(f"  Sources FAIL:   {', '.join(snap.sources_failed)}")
            if snap.ibkr_account_value:
                print(f"  IBKR Equity:    ${snap.ibkr_account_value:,.2f}")
            print()
    else:
        args._live_snapshot = None

    results = {}

    if args.all or args.lunar:
        results["lunar"] = cmd_lunar(args)

    if args.all or args.scenarios:
        results["scenarios"] = cmd_scenarios(args)

    if args.all or args.coherence:
        results["coherence"] = cmd_coherence(args)

    if args.all or args.monte_carlo:
        results["monte_carlo"] = cmd_monte_carlo(args)

    if args.put:
        results["put"] = cmd_put(args)

    if args.all or args.briefing:
        results["briefing"] = cmd_briefing(args)

    # Capital Engine (single cycle or continuous loop)
    if args.capital_engine or args.capital_loop:
        import asyncio as _asyncio

        from strategies.storm_lifeboat.capital_engine import LifeboatCapitalEngine
        engine = LifeboatCapitalEngine()
        if args.capital_loop:
            _print_header("CAPITAL ENGINE — CONTINUOUS HOURLY LOOP")
            print("  Press Ctrl+C to stop.")
            print()
            _asyncio.run(engine.run_forever())
        else:
            _print_header("CAPITAL ENGINE — SINGLE CYCLE")
            report = _asyncio.run(engine.run_hourly())
            engine._print_cycle_summary(report)
            results["capital_engine"] = {
                "phase": report.phase.value,
                "portfolio": report.portfolio_value,
                "signals": len(report.signals),
                "stops": len(report.stops_triggered),
            }

    if args.json and len(results) > 1:
        # Consolidate JSON output (skip non-serializable forecast objects)
        clean = {}
        for k, v in results.items():
            if k == "monte_carlo":
                clean[k] = {"elapsed_ms": v.get("elapsed_ms")}
            else:
                clean[k] = v
        print(json.dumps(clean, indent=2, default=str))


if __name__ == "__main__":
    main()

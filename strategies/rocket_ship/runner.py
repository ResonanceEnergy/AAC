#!/usr/bin/env python3
"""
Rocket Ship Module — CLI Runner
=================================
Unified entry point for all Rocket Ship subsystems.

Usage:
    python -m strategies.rocket_ship.runner              # Full briefing
    python -m strategies.rocket_ship.runner --indicators # Indicator dashboard
    python -m strategies.rocket_ship.runner --lunar      # Lunar position
    python -m strategies.rocket_ship.runner --trigger    # Ignition trigger report
    python -m strategies.rocket_ship.runner --allocation # Allocation plan
    python -m strategies.rocket_ship.runner --geo        # Geo plan tracker
    python -m strategies.rocket_ship.runner --geo PANAMA # Filter by base
    python -m strategies.rocket_ship.runner --all        # Everything above
    python -m strategies.rocket_ship.runner --json       # JSON output
"""
from __future__ import annotations

import argparse
import io
import json
import sys
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


def cmd_indicators(args: argparse.Namespace) -> dict:
    """Show all 15 de-dollarization indicators + Gulf Yuan Trigger."""
    from strategies.rocket_ship.indicators import IndicatorEngine

    engine = IndicatorEngine()
    engine.load_state()

    if not args.json:
        _print_header("DE-DOLLARIZATION INDICATOR DASHBOARD")
        print(engine.format_dashboard())
        print()

    green  = engine.green_count()
    status = engine.check_ignition()
    return {
        "green_count": green,
        "total": len(engine.indicators),
        "ignition_ready": status,
        "gulf_trigger": engine.gulf_trigger.status,
    }


def cmd_lunar(args: argparse.Namespace) -> dict:
    """Display current moon position in the Rocket Ship lunar cycle."""
    from strategies.rocket_ship.lunar_cycles import RocketLunarEngine

    engine = RocketLunarEngine()
    state  = engine.get_current_state()

    if not args.json:
        _print_header("ROCKET SHIP — LUNAR CYCLE")
        print(engine.format_dashboard())
        print()

    return {
        "moon_number":          state.moon_number,
        "moon_name":            state.moon_name,
        "new_moon_date":        state.new_moon_date.isoformat(),
        "day_in_moon":          state.day_in_moon,
        "days_to_rocket_start": state.days_to_rocket_start,
        "in_phi_window_1":      state.in_phi_window_1,
        "in_phi_window_2":      state.in_phi_window_2,
        "is_rocket_phase":      state.is_rocket_phase,
        "milestone":            state.milestone,
    }


def cmd_trigger(args: argparse.Namespace) -> dict:
    """Evaluate ignition triggers and show current phase."""
    from strategies.rocket_ship.trigger_engine import TriggerEngine

    engine = TriggerEngine()
    report = engine.evaluate()

    if not args.json:
        _print_header("IGNITION TRIGGER REPORT")
        print(engine.format_dashboard())
        print()

    return {
        "phase":             report.phase.value,
        "alert_level":       report.alert_level,
        "is_ignited":        report.is_ignited,
        "ignition_prob":     report.ignition_probability,
        "green_count":       report.green_count,
        "gulf_trigger":      report.gulf_trigger_status,
        "days_to_ignition":  report.days_to_default_ignition,
        "actions":           report.immediate_actions,
        "reason":            report.reason,
    }


def cmd_allocation(args: argparse.Namespace) -> dict:
    """Show portfolio allocation plan for current or specified phase."""
    from strategies.rocket_ship.allocation import AllocationEngine
    from strategies.rocket_ship.core import SystemPhase
    from strategies.rocket_ship.trigger_engine import TriggerEngine

    alloc_engine = AllocationEngine()

    if args.phase:
        phase = SystemPhase(args.phase.upper())
    else:
        trigger = TriggerEngine()
        report  = trigger.evaluate()
        phase   = report.phase

    # total_usd can be overridden via --capital flag
    total_usd = getattr(args, "capital", None)

    if not args.json:
        _print_header(f"ROCKET SHIP — ALLOCATION ({phase.value})")
        if total_usd:
            print(alloc_engine.format_dashboard(phase, float(total_usd)))
        else:
            print(alloc_engine.format_dashboard(phase))
        print()

    plan = alloc_engine.compute(phase, float(total_usd) if total_usd else 32486.0)
    return {
        "phase":       phase.value,
        "total_usd":   plan.total_usd,
        "buckets": {
            label: {"pct": pct, "usd": usd}
            for label, (pct, usd) in plan.buckets.items()
        },
        "yield_low":   plan.estimated_portfolio_yield_low,
        "yield_high":  plan.estimated_portfolio_yield_high,
        "annual_yield_low_usd":  plan.annual_yield_usd_low,
        "annual_yield_high_usd": plan.annual_yield_usd_high,
    }


def cmd_geo(args: argparse.Namespace) -> dict:
    """Show geo plan task tracker."""
    from strategies.rocket_ship.geo_plan import GeoPlanEngine
    from strategies.rocket_ship.lunar_cycles import RocketLunarEngine

    geo    = GeoPlanEngine()
    geo.load_state()

    lunar  = RocketLunarEngine()
    state  = lunar.get_current_state()

    base_filter = getattr(args, "geo_base", None)

    if not args.json:
        _print_header("GEO PLAN TRACKER — Panama / Paraguay / UAE")
        print(geo.format_dashboard(current_moon=state.moon_number, base_filter=base_filter))
        print()

    total    = len(geo.ALL_TASKS)
    complete = len(geo.complete_tasks())
    upcoming = geo.tasks_for_moon(state.moon_number)
    return {
        "current_moon":       state.moon_number,
        "total_tasks":        total,
        "complete_tasks":     complete,
        "incomplete_tasks":   total - complete,
        "upcoming_this_moon": [{"id": t.id, "title": t.title} for t in upcoming],
    }


def cmd_full_briefing(args: argparse.Namespace) -> dict:
    """Run all dashboards in sequence."""
    _print_header("ROCKET SHIP — FULL MISSION BRIEFING")
    print(f"  Date: {date.today().isoformat()}")
    print()

    results: dict = {}
    results["indicators"] = cmd_indicators(args)
    results["lunar"]      = cmd_lunar(args)
    results["trigger"]    = cmd_trigger(args)
    results["allocation"] = cmd_allocation(args)
    results["geo"]        = cmd_geo(args)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m strategies.rocket_ship.runner",
        description="Rocket Ship Module — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--indicators", action="store_true",
                        help="Show indicator dashboard")
    parser.add_argument("--lunar",      action="store_true",
                        help="Show lunar cycle position")
    parser.add_argument("--trigger",    action="store_true",
                        help="Show ignition trigger report")
    parser.add_argument("--allocation", action="store_true",
                        help="Show allocation plan")
    parser.add_argument("--geo",        nargs="?", const="ALL", metavar="BASE",
                        help="Show geo plan  [PANAMA|PARAGUAY|UAE|CANADA]")
    parser.add_argument("--all",        action="store_true",
                        help="Run full briefing (all dashboards)")
    parser.add_argument("--phase",      default=None,
                        choices=["LIFE_BOAT", "IGNITING", "ROCKET", "ORBIT"],
                        help="Override phase for allocation display")
    parser.add_argument("--capital",    type=float, default=None,
                        help="Override portfolio size in USD")
    parser.add_argument("--json",       action="store_true",
                        help="Output all results as JSON")
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Attach geo_base from --geo argument
    geo_base_raw = getattr(args, "geo", None)
    args.geo_base = geo_base_raw if (geo_base_raw and geo_base_raw != "ALL") else None

    any_flag = args.indicators or args.lunar or args.trigger or args.allocation or args.geo or args.all

    if args.all or not any_flag:
        result = cmd_full_briefing(args)
    else:
        result = {}
        if args.indicators:
            result["indicators"] = cmd_indicators(args)
        if args.lunar:
            result["lunar"] = cmd_lunar(args)
        if args.trigger:
            result["trigger"] = cmd_trigger(args)
        if args.allocation:
            result["allocation"] = cmd_allocation(args)
        if args.geo:
            result["geo"] = cmd_geo(args)

    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

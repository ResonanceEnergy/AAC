#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AAC WEEKLY CYCLE & COMPOUND TRACKER                                         ║
║  Track the $8,800 → $1,000,000 journey over 26 weeks at 20% weekly           ║
║                                                                              ║
║  Weekly Schedule:                                                            ║
║    MON → Deploy capital into new weekly positions (puts, crypto rotation)     ║
║    TUE → Monitor, adjust stops, roll positions if needed                     ║
║    WED → Mid-week assessment — add/trim based on crisis vector changes       ║
║    THU → Harvest mode — close winners, tighten stops on remaining            ║
║    FRI → Close all weekly positions, compound profits, plan next week        ║
║                                                                              ║
║  Usage:                                                                      ║
║    python weekly_tracker.py                  # Show current week status        ║
║    python weekly_tracker.py --compound       # Run compound calculation        ║
║    python weekly_tracker.py --plan           # Show next week's plan           ║
║    python weekly_tracker.py --history        # Show all weeks history          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_project_path

logger = logging.getLogger("WEEKLY_TRACKER")

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL = 8_800.0
TARGET = 1_000_000.0
WEEKLY_GROWTH_RATE = 0.20       # 20% weekly target
WEEKS_TO_TARGET = 26            # 26 weeks at 20% = $1M+
MAX_WEEKLY_DRAWDOWN = 0.10      # 10% max weekly drawdown
CASH_RESERVE_PCT = 0.17         # 17% always in cash

DATA_FILE = get_project_path("data", "weekly_compound_tracker.json")


class DayOfWeek(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class WeekPhase(Enum):
    """What phase of the weekly cycle we're in."""
    DEPLOY = "DEPLOY"           # Monday — open new positions
    MONITOR = "MONITOR"         # Tuesday — adjust and monitor
    ASSESS = "ASSESS"           # Wednesday — mid-week review
    HARVEST = "HARVEST"         # Thursday — close winners
    CLOSE_COMPOUND = "CLOSE"    # Friday — close all, compound


@dataclass
class WeeklyAllocation:
    """Capital allocation for one week."""
    total_capital: float = 0.0
    cash_reserve: float = 0.0            # 17% cash
    options_allocation: float = 0.0       # 45% to options (puts/calls)
    crypto_allocation: float = 0.0        # 35% to crypto rotation
    arb_allocation: float = 0.0           # 3% to arbitrage

    @classmethod
    def from_capital(cls, capital: float) -> "WeeklyAllocation":
        return cls(
            total_capital=capital,
            cash_reserve=capital * CASH_RESERVE_PCT,
            options_allocation=capital * 0.45,
            crypto_allocation=capital * 0.35,
            arb_allocation=capital * 0.03,
        )


@dataclass
class WeekRecord:
    """Record of one week's performance."""
    week_number: int
    start_date: str
    end_date: str
    starting_capital: float
    ending_capital: float = 0.0
    target_capital: float = 0.0         # What we needed to hit 20%
    pnl: float = 0.0
    pnl_pct: float = 0.0
    options_pnl: float = 0.0
    crypto_pnl: float = 0.0
    arb_pnl: float = 0.0
    trades_executed: int = 0
    signals_generated: int = 0
    crisis_severity_avg: float = 0.0
    best_trade: str = ""
    worst_trade: str = ""
    notes: str = ""
    status: str = "planned"             # planned, active, completed, missed

    @property
    def on_target(self) -> bool:
        return self.pnl_pct >= WEEKLY_GROWTH_RATE * 100

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["on_target"] = self.on_target
        return d


@dataclass
class CompoundTracker:
    """Tracks the full journey from $8,800 to $1,000,000."""
    start_date: str = ""
    current_week: int = 0
    current_capital: float = STARTING_CAPITAL
    peak_capital: float = STARTING_CAPITAL
    max_drawdown: float = 0.0
    weeks: List[WeekRecord] = field(default_factory=list)

    def current_allocation(self) -> WeeklyAllocation:
        return WeeklyAllocation.from_capital(self.current_capital)

    def progress_pct(self) -> float:
        if self.current_capital <= 0:
            return 0.0
        return (self.current_capital / TARGET) * 100

    def weeks_remaining(self) -> int:
        if self.current_capital <= 0:
            return 999
        if self.current_capital >= TARGET:
            return 0
        import math
        return max(0, math.ceil(
            math.log(TARGET / self.current_capital) / math.log(1 + WEEKLY_GROWTH_RATE)
        ))

    def compound_schedule(self) -> List[Dict[str, Any]]:
        """Generate the full 26-week compound schedule from current position."""
        schedule = []
        capital = self.current_capital
        start = datetime.now()
        for week in range(1, WEEKS_TO_TARGET + 1):
            week_start = start + timedelta(weeks=week - 1)
            target = capital * (1 + WEEKLY_GROWTH_RATE)
            alloc = WeeklyAllocation.from_capital(capital)
            schedule.append({
                "week": self.current_week + week,
                "start": week_start.strftime("%Y-%m-%d"),
                "starting_capital": round(capital, 2),
                "target": round(target, 2),
                "growth_needed": round(target - capital, 2),
                "options_alloc": round(alloc.options_allocation, 2),
                "crypto_alloc": round(alloc.crypto_allocation, 2),
                "cash_reserve": round(alloc.cash_reserve, 2),
            })
            capital = target
        return schedule

    def save(self):
        """Persist tracker to disk."""
        data = {
            "start_date": self.start_date,
            "current_week": self.current_week,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "max_drawdown": self.max_drawdown,
            "weeks": [w.to_dict() for w in self.weeks],
        }
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls) -> "CompoundTracker":
        """Load tracker from disk or create new."""
        if DATA_FILE.exists():
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            tracker = cls(
                start_date=data.get("start_date", datetime.now().strftime("%Y-%m-%d")),
                current_week=data.get("current_week", 0),
                current_capital=data.get("current_capital", STARTING_CAPITAL),
                peak_capital=data.get("peak_capital", STARTING_CAPITAL),
                max_drawdown=data.get("max_drawdown", 0),
            )
            for w in data.get("weeks", []):
                # Remove computed property if present
                w.pop("on_target", None)
                tracker.weeks.append(WeekRecord(**w))
            return tracker

        # New tracker
        tracker = cls(start_date=datetime.now().strftime("%Y-%m-%d"))
        tracker.save()
        return tracker

    def record_week(self, pnl: float, options_pnl: float = 0, crypto_pnl: float = 0,
                    arb_pnl: float = 0, trades: int = 0, signals: int = 0,
                    crisis_severity: float = 0, best_trade: str = "",
                    worst_trade: str = "", notes: str = ""):
        """Record a completed week and compound."""
        self.current_week += 1
        start_cap = self.current_capital
        target_cap = start_cap * (1 + WEEKLY_GROWTH_RATE)

        week = WeekRecord(
            week_number=self.current_week,
            start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            starting_capital=start_cap,
            ending_capital=start_cap + pnl,
            target_capital=target_cap,
            pnl=pnl,
            pnl_pct=(pnl / start_cap * 100) if start_cap > 0 else 0,
            options_pnl=options_pnl,
            crypto_pnl=crypto_pnl,
            arb_pnl=arb_pnl,
            trades_executed=trades,
            signals_generated=signals,
            crisis_severity_avg=crisis_severity,
            best_trade=best_trade,
            worst_trade=worst_trade,
            notes=notes,
            status="completed",
        )

        self.weeks.append(week)
        self.current_capital = start_cap + pnl

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.save()
        return week


def get_current_phase() -> WeekPhase:
    """Determine what phase of the weekly cycle we're in."""
    day = datetime.now().weekday()
    phase_map = {
        0: WeekPhase.DEPLOY,
        1: WeekPhase.MONITOR,
        2: WeekPhase.ASSESS,
        3: WeekPhase.HARVEST,
        4: WeekPhase.CLOSE_COMPOUND,
    }
    return phase_map.get(day, WeekPhase.MONITOR)


def print_status(tracker: CompoundTracker):
    """Print comprehensive status dashboard."""
    phase = get_current_phase()
    alloc = tracker.current_allocation()

    print("\n" + "═" * 70)
    print("  BARREN WUFFET — WEEKLY COMPOUND TRACKER")
    print("  $8,800 → $1,000,000 MISSION")
    print("═" * 70)

    print(f"\n  Week:     {tracker.current_week} of ~{tracker.current_week + tracker.weeks_remaining()}")
    print(f"  Capital:  ${tracker.current_capital:>12,.2f}")
    print(f"  Target:   ${TARGET:>12,.0f}")
    print(f"  Progress: {tracker.progress_pct():>11.4f}%")
    print(f"  Phase:    {phase.value} ({datetime.now().strftime('%A')})")
    print(f"  Peak:     ${tracker.peak_capital:>12,.2f}")
    print(f"  Max DD:   {tracker.max_drawdown * 100:>11.2f}%")
    print(f"  Weeks to target: {tracker.weeks_remaining()}")

    print("\n  CURRENT ALLOCATION:")
    print(f"    Options (45%):  ${alloc.options_allocation:>10,.2f}")
    print(f"    Crypto (35%):   ${alloc.crypto_allocation:>10,.2f}")
    print(f"    Arb (3%):       ${alloc.arb_allocation:>10,.2f}")
    print(f"    Cash (17%):     ${alloc.cash_reserve:>10,.2f}")

    # This week's target
    target = tracker.current_capital * (1 + WEEKLY_GROWTH_RATE)
    needed = target - tracker.current_capital
    print(f"\n  THIS WEEK'S TARGET:")
    print(f"    Starting:  ${tracker.current_capital:>10,.2f}")
    print(f"    Target:    ${target:>10,.2f}")
    print(f"    Needed:    ${needed:>10,.2f} (+{WEEKLY_GROWTH_RATE*100:.0f}%)")

    # Weekly cycle phases
    print(f"\n  WEEKLY CYCLE:")
    phases = ["MON: Deploy", "TUE: Monitor", "WED: Assess", "THU: Harvest", "FRI: Close+Compound"]
    today = datetime.now().weekday()
    for i, p in enumerate(phases):
        marker = " →" if i == today else "  "
        print(f"    {marker} {p}")

    if tracker.weeks:
        print(f"\n  RECENT WEEKS:")
        for w in tracker.weeks[-5:]:
            on = "✅" if w.on_target else "❌"
            print(f"    Week {w.week_number:>2}: ${w.starting_capital:>10,.2f} → "
                  f"${w.ending_capital:>10,.2f} ({w.pnl_pct:>+6.1f}%) {on}")

    print("\n" + "═" * 70)


def print_schedule(tracker: CompoundTracker):
    """Print the full compound schedule."""
    schedule = tracker.compound_schedule()

    print("\n" + "═" * 70)
    print("  26-WEEK COMPOUND SCHEDULE")
    print("  Target: 20% weekly growth")
    print("═" * 70)
    print(f"  {'Week':>4}  {'Start':>10}  {'Capital':>12}  {'Target':>12}  {'Growth':>10}  {'Options':>10}  {'Crypto':>10}")
    print("  " + "-" * 66)

    for s in schedule:
        marker = " " if s["starting_capital"] < TARGET else "🎯"
        print(f"  {s['week']:>4}  {s['start']:>10}  ${s['starting_capital']:>10,.0f}  "
              f"${s['target']:>10,.0f}  ${s['growth_needed']:>8,.0f}  "
              f"${s['options_alloc']:>8,.0f}  ${s['crypto_alloc']:>8,.0f}  {marker}")

        if s["starting_capital"] >= TARGET:
            print(f"\n  🎯 TARGET REACHED AT WEEK {s['week']}!")
            break

    print("\n" + "═" * 70)


def print_history(tracker: CompoundTracker):
    """Print full week-by-week history."""
    if not tracker.weeks:
        print("\n  No weeks recorded yet. Run full_activation.py to start trading.\n")
        return

    print("\n" + "═" * 70)
    print("  WEEK-BY-WEEK HISTORY")
    print("═" * 70)

    for w in tracker.weeks:
        on = "✅" if w.on_target else "❌"
        print(f"\n  Week {w.week_number} ({w.start_date} → {w.end_date}) {on}")
        print(f"    Capital: ${w.starting_capital:,.2f} → ${w.ending_capital:,.2f}")
        print(f"    P&L: ${w.pnl:+,.2f} ({w.pnl_pct:+.1f}%) | Target: ${w.target_capital:,.2f}")
        print(f"    Options: ${w.options_pnl:+,.2f} | Crypto: ${w.crypto_pnl:+,.2f} | Arb: ${w.arb_pnl:+,.2f}")
        print(f"    Trades: {w.trades_executed} | Signals: {w.signals_generated}")
        if w.best_trade:
            print(f"    Best: {w.best_trade}")
        if w.worst_trade:
            print(f"    Worst: {w.worst_trade}")
        if w.notes:
            print(f"    Notes: {w.notes}")

    print("\n" + "═" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AAC Weekly Compound Tracker")
    parser.add_argument("--compound", action="store_true", help="Show compound schedule")
    parser.add_argument("--plan", action="store_true", help="Show next week's plan")
    parser.add_argument("--history", action="store_true", help="Show all weeks history")
    args = parser.parse_args()

    tracker = CompoundTracker.load()

    if args.compound:
        print_schedule(tracker)
    elif args.history:
        print_history(tracker)
    elif args.plan:
        print_schedule(tracker)
    else:
        print_status(tracker)

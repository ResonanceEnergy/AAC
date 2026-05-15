#!/usr/bin/env python3
"""
ninety_day_war_room — STRIPPED (2026-05-14)
===========================================================================
This module previously contained:
  - Hardcoded ACCOUNTS / POSITIONS dicts (stale March 2026 values)
  - Sin-wave fake indicator interpolation (build_historical_model /
    build_forward_model / DailyIndicatorState)
  - Confirmation-biased mandate generator (generate_daily_mandate /
    DailyMandate / _risk_level)
  - CAD_USD constant, CASH_INJECTION, scenario projections

All of that has been retired — it was a scripted narrative, not an
algorithm. Real edge lives in:
  - config.account_balances.Balances        ← portfolio balances
  - strategies.war_room_live_feeds          ← real indicator feeds
  - strategies.regime_engine                ← F10–F14 scoring + COT veto
  - strategies.war_room_council_feeds       ← council aggregation

The legacy file is preserved at _archive/ninety_day_war_room_legacy.py
for reference only. Do not import from it.

What remains here (the parts with real value):
  - MacroEvent / MACRO_CALENDAR             ← hand-curated event calendar
  - get_events_for_date / get_events_in_range
  - log_intel_update / get_intel_for_date / get_recent_intel
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

WAR_START_DATE = date(2026, 2, 28)
DATA_DIR = Path("data/war_room")


# ===========================================================================
# MACRO CALENDAR — hand-curated 180-day event list
# ===========================================================================

class EventImpact(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    THESIS = "THESIS"


@dataclass
class MacroEvent:
    date: date
    name: str
    category: str
    impact: EventImpact
    thesis_relevance: str
    action_items: List[str] = field(default_factory=list)
    actual_result: str = ""


MACRO_CALENDAR: List[MacroEvent] = [
    # ====== DECEMBER 2025 ======
    MacroEvent(date(2025, 12, 18), "FOMC Decision Dec", "fed", EventImpact.CRITICAL,
               "Rate path sets stage for Q1 2026 credit conditions",
               ["Monitor dot plot", "Check HY spread reaction"]),
    MacroEvent(date(2025, 12, 20), "PCE Inflation Nov", "economic", EventImpact.HIGH,
               "Inflation trajectory affects Fed ability to cut in crisis",
               ["If high -> stagflation thesis strengthens"]),
    MacroEvent(date(2025, 12, 25), "Christmas — Markets Closed", "calendar", EventImpact.LOW,
               "Low liquidity window. Geopolitical events during closure amplified."),

    # ====== JANUARY 2026 ======
    MacroEvent(date(2026, 1, 3), "Jobs Report Dec", "economic", EventImpact.HIGH,
               "Labor market health. Strong = rates stay high = credit stress builds",
               ["Compare NFP vs expectations", "Watch wage growth"]),
    MacroEvent(date(2026, 1, 10), "CPI Dec", "economic", EventImpact.HIGH,
               "Inflation reading. >3% = stagflation risk climbs",
               ["Core CPI trend", "Shelter component"]),
    MacroEvent(date(2026, 1, 15), "Q4 Earnings Season Begins", "earnings", EventImpact.HIGH,
               "Banks report first. Credit losses, loan reserves = thesis signals",
               ["JPM, BAC, C, WFC — watch loan loss provisions"]),
    MacroEvent(date(2026, 1, 20), "Trump Inauguration Day", "geopolitical", EventImpact.CRITICAL,
               "New admin policy direction. Iran stance critical for thesis.",
               ["Watch executive orders on Iran sanctions", "ME troop deployment signals"]),
    MacroEvent(date(2026, 1, 29), "FOMC Decision Jan", "fed", EventImpact.CRITICAL,
               "First Fed decision under new geopolitical reality",
               ["Rate decision + statement tone", "Treasury market reaction"]),
    MacroEvent(date(2026, 1, 31), "GDP Q4 Advance", "economic", EventImpact.HIGH,
               "Growth vs inflation trade-off",
               ["If GDP slowing + inflation rising = stagflation setup"]),

    # ====== FEBRUARY 2026 ======
    MacroEvent(date(2026, 2, 7), "Jobs Report Jan", "economic", EventImpact.HIGH,
               "First labor read of 2026",
               ["Unemployment rate trend"]),
    MacroEvent(date(2026, 2, 12), "CPI Jan", "economic", EventImpact.HIGH,
               "Inflation persistence check",
               ["Energy component — oil trend feeding through?"]),
    MacroEvent(date(2026, 2, 14), "Retail Sales Jan", "economic", EventImpact.MEDIUM,
               "Consumer health. Weakening = recession signal."),
    MacroEvent(date(2026, 2, 20), "OPEC+ Meeting", "geopolitical", EventImpact.HIGH,
               "Oil supply decision. Cuts = price support = thesis acceleration",
               ["Production quota changes", "Saudi signaling"]),
    MacroEvent(date(2026, 2, 28), "US-IRAN WAR BEGINS", "thesis", EventImpact.CRITICAL,
               "THE CATALYST. US strikes on Iran. War day 0. Thesis activated.",
               ["ALL SYSTEMS ACTIVE", "All 25 pressure indicators live",
                "Begin daily monitoring cadence"]),

    # ====== MARCH 2026 ======
    MacroEvent(date(2026, 3, 1), "War Day 1 — Market Reaction", "thesis", EventImpact.CRITICAL,
               "First trading day after war announcement. Oil gap up, equities gap down.",
               ["Oil open price", "VIX open", "Gold spot", "Credit spread widening"]),
    MacroEvent(date(2026, 3, 5), "OPEC Emergency Meeting (rumored)", "thesis", EventImpact.HIGH,
               "OPEC response to ME war. Supply cuts = oil $100+",
               ["Any emergency session announcement", "Saudi statement"]),
    MacroEvent(date(2026, 3, 7), "Jobs Report Feb", "economic", EventImpact.MEDIUM,
               "War impact not yet in data. Baseline read.",
               ["Watch revisions to Jan data"]),
    MacroEvent(date(2026, 3, 10), "CPI Feb", "economic", EventImpact.HIGH,
               "Oil spike starting to feed into headline CPI?",
               ["Energy component spike", "Headline vs core divergence"]),
    MacroEvent(date(2026, 3, 12), "Iran Hormuz Partial Closure", "thesis", EventImpact.CRITICAL,
               "Day 12: Hormuz disruption begins. Oil $95+ confirmed.",
               ["Insurance premium spikes", "Tanker rerouting", "Yuan passage proposal"]),
    MacroEvent(date(2026, 3, 18), "LIVE TRADES EXECUTED", "thesis", EventImpact.CRITICAL,
               "8 puts deployed on IBKR. $910 total across credit/equity verticals.",
               ["ARCC, PFF, LQD, EMB, MAIN, JNK, KRE, IWM puts live"]),
    MacroEvent(date(2026, 3, 18), "NDAX Liquidated", "thesis", EventImpact.HIGH,
               "Sold all crypto (XRP+ETH) for $4,492 CAD. Cash ready for deployment.",
               ["CAD cash available at NDAX"]),
    MacroEvent(date(2026, 3, 19), "FOMC Decision Mar", "fed", EventImpact.CRITICAL,
               "Fed faces impossible choice: cut (fuel inflation) or hold (strangle credit)",
               ["Rate decision", "Dot plot revision", "Press conference tone on war"]),

    # ====== APRIL 2026 ======
    MacroEvent(date(2026, 4, 2), "JOLTS Feb", "economic", EventImpact.MEDIUM,
               "Job openings. War impact starting to show?"),
    MacroEvent(date(2026, 4, 4), "Jobs Report Mar", "economic", EventImpact.HIGH,
               "FIRST report with war-period data. Key read on economic impact.",
               ["Energy sector hiring surge?", "Defense hiring?", "Consumer-facing layoffs?"]),
    MacroEvent(date(2026, 4, 5), "OPEC+ Regular Meeting", "geopolitical", EventImpact.HIGH,
               "Formal OPEC meeting during active ME war. Supply decisions critical.",
               ["Production cuts deepen?", "Saudi volume signals"]),
    MacroEvent(date(2026, 4, 10), "CPI Mar", "economic", EventImpact.CRITICAL,
               "FIRST CPI with full month of war. Oil spike in data. Headline >4%?",
               ["Energy component", "Headline vs core spread"]),
    MacroEvent(date(2026, 4, 11), "Q1 Earnings Begin — Banks", "earnings", EventImpact.CRITICAL,
               "Bank earnings with war impact. Loan loss reserves, trading revenue.",
               ["JPM, C, BAC, WFC loan loss provisions", "Trading desk war profits",
                "Credit card delinquencies", "CRE write-downs"]),
    MacroEvent(date(2026, 4, 15), "Tax Day — Forced Selling", "economic", EventImpact.MEDIUM,
               "Tax-related selling pressure. Liquidity stress possible."),
    MacroEvent(date(2026, 4, 17), "ECB Decision", "economic", EventImpact.MEDIUM,
               "Europe caught between war, energy, and recession"),
    MacroEvent(date(2026, 4, 20), "Q1 Earnings — Tech Week", "earnings", EventImpact.HIGH,
               "AAPL, MSFT, GOOG, AMZN. War + AI bubble deflation?",
               ["Revenue guidance cuts?", "Supply chain disruption mentions"]),
    MacroEvent(date(2026, 4, 24), "Durable Goods Mar", "economic", EventImpact.MEDIUM,
               "Capital investment. War uncertainty = pullback?"),
    MacroEvent(date(2026, 4, 30), "GDP Q1 Advance", "economic", EventImpact.CRITICAL,
               "First GDP with full war quarter. Negative = recession.",
               ["If GDP <1% = recession fear", "If negative = panic"]),

    # ====== MAY 2026 ======
    MacroEvent(date(2026, 5, 2), "Jobs Report Apr", "economic", EventImpact.HIGH,
               "Second war-month employment data. Trend emerging."),
    MacroEvent(date(2026, 5, 6), "FOMC Decision May", "fed", EventImpact.CRITICAL,
               "Fed faces full reality of war + inflation + slowing growth.",
               ["Emergency cut?", "Financial stability language",
                "Treasury market functioning", "Swap line activations?"]),
    MacroEvent(date(2026, 5, 13), "CPI Apr", "economic", EventImpact.CRITICAL,
               "Second full war-month inflation. Oil pass-through complete.",
               ["Headline CPI trend", "Core persistence"]),
    MacroEvent(date(2026, 5, 15), "Retail Sales Apr", "economic", EventImpact.HIGH,
               "Consumer breaking under gas prices + fear?",
               ["Real spending decline", "Gas station spending surge"]),
    MacroEvent(date(2026, 5, 22), "PMI Flash May", "economic", EventImpact.MEDIUM,
               "Business sentiment 3 months into war"),
    MacroEvent(date(2026, 5, 29), "GDP Q1 Second Estimate", "economic", EventImpact.HIGH,
               "Revised GDP with more data. Confirms or denies recession."),

    # ====== JUNE 2026 ======
    MacroEvent(date(2026, 6, 5), "OPEC+ Meeting", "geopolitical", EventImpact.HIGH,
               "Oil supply decision during prolonged war",
               ["By now oil trajectory set", "OPEC power dynamics shifted"]),
    MacroEvent(date(2026, 6, 6), "Jobs Report May", "economic", EventImpact.HIGH,
               "Third war-month labor data. Structural damage visible?"),
    MacroEvent(date(2026, 6, 10), "CPI May", "economic", EventImpact.CRITICAL,
               "Third war-month CPI. Inflation entrenched or peaking?"),
    MacroEvent(date(2026, 6, 16), "FOMC Decision Jun", "fed", EventImpact.CRITICAL,
               "Fed 3 months into war. Dot plot for 2026-2027.",
               ["Rate path for rest of year", "Updated economic projections"]),
]


def get_events_for_date(target: date) -> List[MacroEvent]:
    """Get all macro events for a specific date."""
    return [e for e in MACRO_CALENDAR if e.date == target]


def get_events_in_range(start: date, end: date) -> List[MacroEvent]:
    """Get all events in a date range."""
    return sorted(
        [e for e in MACRO_CALENDAR if start <= e.date <= end],
        key=lambda e: e.date,
    )


# ===========================================================================
# INTEL UPDATE LOG
# ===========================================================================

def _war_day(d: date) -> int:
    """Calculate war day number (-1 for pre-war)."""
    if d < WAR_START_DATE:
        return -1
    return (d - WAR_START_DATE).days


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_intel_update(note: str, indicators: Optional[Dict] = None) -> None:
    """Append an intelligence update to the JSONL log."""
    ensure_data_dir()
    ts = datetime.now()
    entry = {
        "timestamp": ts.isoformat(),
        "war_day": _war_day(ts.date()),
        "note": note,
        "indicators": indicators or {},
    }
    log_file = DATA_DIR / "intel_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_intel_for_date(target: date) -> List[Dict]:
    """Retrieve all intel updates for a specific date."""
    log_file = DATA_DIR / "intel_log.jsonl"
    if not log_file.exists():
        return []
    results = []
    target_str = target.isoformat()
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["timestamp"].startswith(target_str):
                results.append(entry)
    return results


def get_recent_intel(n: int = 10) -> List[Dict]:
    """Get the N most recent intel updates."""
    log_file = DATA_DIR / "intel_log.jsonl"
    if not log_file.exists():
        return []
    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries[-n:]


__all__ = [
    "EventImpact",
    "MacroEvent",
    "MACRO_CALENDAR",
    "WAR_START_DATE",
    "DATA_DIR",
    "get_events_for_date",
    "get_events_in_range",
    "log_intel_update",
    "get_intel_for_date",
    "get_recent_intel",
    "ensure_data_dir",
]

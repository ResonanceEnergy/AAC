#!/usr/bin/env python3
"""
90-DAY WAR ROOM  --  AAC Supreme Command Center
===========================================================================
Day-by-day model: 90 days back (Dec 19 2025 -> Mar 19 2026) + 90 days forward
(Mar 19 -> Jun 17 2026).

Every day has: thesis pressure, indicator states, macro events, correlation
snapshots, portfolio actions, capital deployment mandates, and confidence
scores.

THESIS: Iran war -> US pullback -> Gulf yuan conversion -> gold reprice -> USD collapse

PORTFOLIO ACCOUNTS (as of March 19, 2026):
  IBKR          $920 USD   (8 put positions ~ $760 in options)
  NDAX          $4,400 CAD (~$3,080 USD at 0.70)
  Moomoo        $500 USD
  WealthSimple  $4,200 CAD TFSA (~$2,940 USD)
  EQ Bank       $100 CAD   (~$70 USD)
  INJECTION     $35,000 (incoming)

Usage:
  python -m strategies.ninety_day_war_room                    # Full dashboard
  python -m strategies.ninety_day_war_room --day 2026-04-15   # Query specific day
  python -m strategies.ninety_day_war_room --week 3           # Week 3 mandate
  python -m strategies.ninety_day_war_room --backtest          # Historical 90-day model
  python -m strategies.ninety_day_war_room --forward           # Forward 90-day projection
  python -m strategies.ninety_day_war_room --portfolio         # All accounts & positions
  python -m strategies.ninety_day_war_room --correlations      # Live correlation matrix
  python -m strategies.ninety_day_war_room --mandate           # Today's mandate
  python -m strategies.ninety_day_war_room --update "..."      # Log intel update
  python -m strategies.ninety_day_war_room --calendar          # Macro event calendar
  python -m strategies.ninety_day_war_room --range 2026-04-01 2026-04-14  # Date range view
"""

import io
import json
import math
import os
import sys
import logging

# Windows cp1252 fix  --  ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, "encoding") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "encoding") and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("war_room")

# ===========================================================================
# CONSTANTS
# ===========================================================================

WAR_START_DATE = date(2026, 2, 28)  # US-Iran war start
MODEL_START = date(2025, 12, 19)     # 90 days before today
MODEL_TODAY = date(2026, 3, 19)      # anchor date
MODEL_END = date(2026, 6, 17)        # 90 days forward
CAD_USD = 0.70                        # CAD/USD conversion rate
DATA_DIR = Path("data/war_room")

# ===========================================================================
# PART 1  --  PORTFOLIO STATE (all accounts, all positions)
# ===========================================================================

class AccountType(Enum):
    BROKERAGE = "brokerage"
    TFSA = "tfsa"
    CRYPTO = "crypto"
    BANK = "bank"


class Currency(Enum):
    USD = "USD"
    CAD = "CAD"


@dataclass
class Account:
    name: str
    account_type: AccountType
    currency: Currency
    balance: float
    available_cash: float
    in_positions: float
    platform: str
    notes: str = ""

    @property
    def balance_usd(self) -> float:
        return self.balance if self.currency == Currency.USD else self.balance * CAD_USD


@dataclass
class Position:
    symbol: str
    direction: str          # long_put, long_call, long_stock, short
    quantity: int
    entry_price: float
    current_price: float
    account: str
    entry_date: str
    expiry: str = ""
    strike: float = 0.0
    thesis_vertical: str = ""  # oil, credit, equities, gold, etc.
    notes: str = ""

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity * 100

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity * 100


# Current portfolio snapshot
ACCOUNTS: Dict[str, Account] = {
    "ibkr": Account(
        name="Interactive Brokers",
        account_type=AccountType.BROKERAGE,
        currency=Currency.USD,
        balance=920.0,
        available_cash=160.0,
        in_positions=760.0,
        platform="IBKR TWS",
        notes="Account U24346218. Port 7496 LIVE. 8 put positions deployed.",
    ),
    "ndax": Account(
        name="NDAX",
        account_type=AccountType.CRYPTO,
        currency=Currency.CAD,
        balance=4400.0,
        available_cash=4400.0,
        in_positions=0.0,
        platform="NDAX",
        notes="LIQUIDATED all crypto Mar 18. Pure CAD cash sitting.",
    ),
    "moomoo": Account(
        name="Moomoo (Futu)",
        account_type=AccountType.BROKERAGE,
        currency=Currency.USD,
        balance=500.0,
        available_cash=500.0,
        in_positions=0.0,
        platform="OpenD",
        notes="FUTUCA firm. $365.15 real + paper accounts. Options approval pending.",
    ),
    "wealthsimple": Account(
        name="WealthSimple TFSA",
        account_type=AccountType.TFSA,
        currency=Currency.CAD,
        balance=4200.0,
        available_cash=4200.0,
        in_positions=0.0,
        platform="WealthSimple",
        notes="Tax-free. Limited to Canadian-listed ETFs/stocks + some US ETFs.",
    ),
    "eq_bank": Account(
        name="EQ Bank",
        account_type=AccountType.BANK,
        currency=Currency.CAD,
        balance=100.0,
        available_cash=100.0,
        in_positions=0.0,
        platform="EQ Bank",
        notes="High-interest savings. Emergency / transfer buffer.",
    ),
}

CASH_INJECTION = 35_000.0  # Incoming capital injection

# Current IBKR positions (as of Mar 19 2026)
POSITIONS: List[Position] = [
    Position("ARCC", "long_put", 1, 0.25, 0.28, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=17.0, thesis_vertical="credit",
             notes="BDC private credit exposure. Ares Capital."),
    Position("PFF", "long_put", 1, 0.40, 0.42, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=29.0, thesis_vertical="credit",
             notes="Preferred securities. Quasi-bond that collapses in credit stress."),
    Position("LQD", "long_put", 1, 0.64, 0.67, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=106.0, thesis_vertical="credit",
             notes="Investment grade corporate bonds. Spread widening play."),
    Position("EMB", "long_put", 1, 0.75, 0.78, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=90.0, thesis_vertical="credit",
             notes="EM bonds. Dollar strength + oil shock = EM debt blowup."),
    Position("MAIN", "long_put", 1, 0.85, 0.88, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=50.0, thesis_vertical="credit",
             notes="Main Street Capital BDC. Private credit canary."),
    Position("JNK", "long_put", 1, 0.80, 0.83, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=92.0, thesis_vertical="credit",
             notes="SPDR High Yield. Core junk bond put."),
    Position("KRE", "long_put", 1, 1.45, 1.50, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=58.0, thesis_vertical="equities",
             notes="Regional banks. CRE + credit stress ground zero."),
    Position("IWM", "long_put", 1, 3.96, 4.05, "ibkr", "2026-03-18",
             expiry="2026-06-20", strike=230.0, thesis_vertical="equities",
             notes="Russell 2000. Small caps most leveraged + credit sensitive."),
]


def get_total_portfolio() -> Dict[str, Any]:
    """Calculate total portfolio value across all accounts."""
    total_usd = sum(a.balance_usd for a in ACCOUNTS.values())
    total_cad = sum(a.balance for a in ACCOUNTS.values() if a.currency == Currency.CAD)
    total_cash_usd = sum(
        a.available_cash * (1.0 if a.currency == Currency.USD else CAD_USD)
        for a in ACCOUNTS.values()
    )
    positions_value = sum(p.current_price * p.quantity * 100 for p in POSITIONS)
    unrealized = sum(p.unrealized_pnl for p in POSITIONS)

    return {
        "total_usd_equivalent": total_usd,
        "total_cad_accounts": total_cad,
        "available_cash_usd": total_cash_usd,
        "positions_value": positions_value,
        "unrealized_pnl": unrealized,
        "num_positions": len(POSITIONS),
        "cash_injection_pending": CASH_INJECTION,
        "post_injection_total": total_usd + CASH_INJECTION,
        "accounts": {k: asdict(v) for k, v in ACCOUNTS.items()},
        "positions": [asdict(p) for p in POSITIONS],
    }


# ===========================================================================
# PART 2  --  MACRO EVENT CALENDAR
# Known dates that create volatility/opportunity windows
# ===========================================================================

class EventImpact(Enum):
    CRITICAL = "CRITICAL"    # Market-moving, must prepare 24h before
    HIGH = "HIGH"            # Significant vol, position adjustments needed
    MEDIUM = "MEDIUM"        # Moderate impact, monitor
    LOW = "LOW"              # Background noise, note only
    THESIS = "THESIS"        # Directly relevant to black swan thesis


@dataclass
class MacroEvent:
    date: date
    name: str
    category: str          # fed, earnings, geopolitical, economic, thesis
    impact: EventImpact
    thesis_relevance: str  # How it connects to our thesis
    action_items: List[str] = field(default_factory=list)
    actual_result: str = ""  # filled in after event happens


# Complete 180-day macro calendar (Dec 19 2025 -> Jun 17 2026)
MACRO_CALENDAR: List[MacroEvent] = [
    # ====== DECEMBER 2025 ======
    MacroEvent(date(2025, 12, 18), "FOMC Decision Dec", "fed", EventImpact.CRITICAL,
               "Rate path sets stage for Q1 2026 credit conditions",
               ["Monitor dot plot", "Check HY spread reaction"]),
    MacroEvent(date(2025, 12, 20), "PCE Inflation Nov", "economic", EventImpact.HIGH,
               "Inflation trajectory affects Fed ability to cut in crisis",
               ["If high -> stagflation thesis strengthens"]),
    MacroEvent(date(2025, 12, 25), "Christmas  --  Markets Closed", "calendar", EventImpact.LOW,
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
               ["JPM, BAC, C, WFC  --  watch loan loss provisions"]),
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
               ["Energy component  --  oil trend feeding through?"]),
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
    MacroEvent(date(2026, 3, 1), "War Day 1  --  Market Reaction", "thesis", EventImpact.CRITICAL,
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
               ["Rate decision", "Dot plot revision", "Press conference tone on war",
                "Any mention of 'financial stability'"]),
    MacroEvent(date(2026, 3, 19), "WAR DAY 19  --  TODAY", "thesis", EventImpact.CRITICAL,
               "Current anchor date. Pressure cooker at 44%. Phase 1 BUILDING.",
               ["Run full pressure cooker scan", "Authority consensus check",
                "All correlation updates"]),

    # ====== FORWARD CALENDAR: MARCH 20 -> JUNE 17 ======

    # Week 1: March 20-26
    MacroEvent(date(2026, 3, 20), "War Day 20  --  Fed Reaction Day", "thesis", EventImpact.CRITICAL,
               "Markets digest FOMC + war. Credit spreads critical to watch.",
               ["HY spread movement", "VIX trend", "Gold spot", "DXY direction"]),
    MacroEvent(date(2026, 3, 21), "PMI Flash Mar", "economic", EventImpact.MEDIUM,
               "First hint of war impact on business sentiment"),
    MacroEvent(date(2026, 3, 24), "War enters Week 4", "thesis", EventImpact.HIGH,
               "4 weeks of war. Historical inflection  --  either de-escalation or entrenchment.",
               ["Monitor ceasefire rumors", "Trump rhetoric shifts",
                "Casualty reports", "Gulf state statements"]),
    MacroEvent(date(2026, 3, 26), "GDP Q4 Final", "economic", EventImpact.MEDIUM,
               "Backward-looking but revisions signal data quality"),
    MacroEvent(date(2026, 3, 28), "PCE Feb", "economic", EventImpact.HIGH,
               "Fed's preferred inflation gauge. War not fully reflected yet.",
               ["Core PCE trend", "If rising = Fed trapped"]),

    # Week 2-3: March 30 -> April 11
    MacroEvent(date(2026, 4, 1), "War Day 32  --  Month 2 Begins", "thesis", EventImpact.HIGH,
               "War enters month 2. Each week = more thesis pressure.",
               ["Pressure cooker should be >50%", "Scale up if confirming"]),
    MacroEvent(date(2026, 4, 2), "JOLTS Feb", "economic", EventImpact.MEDIUM,
               "Job openings. War impact starting to show?"),
    MacroEvent(date(2026, 4, 4), "Jobs Report Mar", "economic", EventImpact.HIGH,
               "FIRST report with war-period data. Key read on economic impact.",
               ["Energy sector hiring surge?", "Defense hiring?",
                "Consumer-facing layoffs?"]),
    MacroEvent(date(2026, 4, 5), "OPEC+ Regular Meeting", "geopolitical", EventImpact.HIGH,
               "Formal OPEC meeting during active ME war. Supply decisions critical.",
               ["Production cuts deepen?", "Saudi volume signals"]),
    MacroEvent(date(2026, 4, 10), "CPI Mar", "economic", EventImpact.CRITICAL,
               "FIRST CPI with full month of war. Oil spike in data. Headline >4%?",
               ["Energy component", "Headline vs core spread",
                "If >4% headline = stagflation narrative dominates"]),
    MacroEvent(date(2026, 4, 11), "Q1 Earnings Begin  --  Banks", "earnings", EventImpact.CRITICAL,
               "Bank earnings with war impact. Loan loss reserves, trading revenue.",
               ["JPM, C, BAC, WFC loan loss provisions",
                "Trading desk war profits", "Credit card delinquencies",
                "CRE write-downs", "Private credit exposure disclosures"]),

    # Week 4-6: April 14 -> May 1
    MacroEvent(date(2026, 4, 14), "War Day 45  --  Critical Checkpoint", "thesis", EventImpact.CRITICAL,
               "45 days. If no ceasefire, thesis is CONFIRMED accelerating.",
               ["Review all 25 indicators", "Adjust position sizing",
                "If thesis stalling, increase hedges to 15%"]),
    MacroEvent(date(2026, 4, 15), "Tax Day  --  Forced Selling", "economic", EventImpact.MEDIUM,
               "Tax-related selling pressure. Liquidity stress possible."),
    MacroEvent(date(2026, 4, 17), "ECB Decision", "economic", EventImpact.MEDIUM,
               "Europe caught between war, energy, and recession"),
    MacroEvent(date(2026, 4, 20), "Q1 Earnings  --  Tech Week", "earnings", EventImpact.HIGH,
               "AAPL, MSFT, GOOG, AMZN. War + AI bubble deflation?",
               ["Revenue guidance cuts?", "Supply chain disruption mentions",
                "Capital spending pullbacks", "Cloud growth deceleration"]),
    MacroEvent(date(2026, 4, 24), "Durable Goods Mar", "economic", EventImpact.MEDIUM,
               "Capital investment. War uncertainty = pullback?"),
    MacroEvent(date(2026, 4, 30), "GDP Q1 Advance", "economic", EventImpact.CRITICAL,
               "CRITICAL: First GDP with full war quarter. Negative = recession.",
               ["If GDP <1% = recession fear", "If negative = panic",
                "Fed boxed: can't cut (inflation) or hike (growth)",
                "This is the F4 Policy Delay Trap trigger"]),

    # Week 7-9: May 1 -> May 22
    MacroEvent(date(2026, 5, 1), "War Day 62  --  2 Months", "thesis", EventImpact.CRITICAL,
               "2 full months of war. Thesis should be >60% confirmed or abandoned.",
               ["COMPLETE PORTFOLIO REVIEW", "Re-allocate across 8 verticals",
                "Take partial profits on 500%+ winners",
                "Double down on strongest thesis vectors"]),
    MacroEvent(date(2026, 5, 2), "Jobs Report Apr", "economic", EventImpact.HIGH,
               "Second war-month employment data. Trend emerging."),
    MacroEvent(date(2026, 5, 6), "FOMC Decision May", "fed", EventImpact.CRITICAL,
               "Fed faces full reality of war + inflation + slowing growth.",
               ["Emergency cut?", "Financial stability language",
                "Treasury market functioning", "Swap line activations?"]),
    MacroEvent(date(2026, 5, 13), "CPI Apr", "economic", EventImpact.CRITICAL,
               "Second full war-month inflation. Oil pass-through complete.",
               ["Headline CPI trend", "Core persistence",
                "If >5% headline = 1970s stagflation comparison dominates"]),
    MacroEvent(date(2026, 5, 15), "Retail Sales Apr", "economic", EventImpact.HIGH,
               "Consumer breaking under gas prices + fear?",
               ["Real spending decline", "Gas station spending surge"]),
    MacroEvent(date(2026, 5, 19), "War Day 80  --  Endgame Window Opens", "thesis", EventImpact.CRITICAL,
               "80 days. If Hormuz still disrupted, gold >$3,500, DXY weakening...",
               ["This is where FULL_CRISIS phase should activate",
                "Begin transitioning from options to hard assets",
                "Scale gold/dollar positions to maximum"]),

    # Week 10-13: May 22 -> June 17
    MacroEvent(date(2026, 5, 22), "PMI Flash May", "economic", EventImpact.MEDIUM,
               "Business sentiment 3 months into war"),
    MacroEvent(date(2026, 5, 28), "War Day 89  --  3 Month Mark", "thesis", EventImpact.CRITICAL,
               "Quarter point. This is where geopolitical resolutions happen or don't.",
               ["Ceasefire negotiations active?", "UN involvement?",
                "China mediation status", "Gulf state alignment final"]),
    MacroEvent(date(2026, 5, 29), "GDP Q1 Second Estimate", "economic", EventImpact.HIGH,
               "Revised GDP with more data. Confirms or denies recession."),
    MacroEvent(date(2026, 6, 5), "OPEC+ Meeting", "geopolitical", EventImpact.HIGH,
               "Oil supply decision during prolonged war",
               ["By now oil trajectory set", "OPEC power dynamics shifted"]),
    MacroEvent(date(2026, 6, 6), "Jobs Report May", "economic", EventImpact.HIGH,
               "Third war-month labor data. Structural damage visible?"),
    MacroEvent(date(2026, 6, 10), "CPI May", "economic", EventImpact.CRITICAL,
               "Third war-month CPI. Inflation entrenched or peaking?",
               ["If still rising = Fed completely trapped",
                "If peaking = possible ceasefire priced in"]),
    MacroEvent(date(2026, 6, 16), "FOMC Decision Jun", "fed", EventImpact.CRITICAL,
               "Fed 3 months into war. Dot plot for 2026-2027.",
               ["Rate path for rest of year", "Updated economic projections",
                "Balance sheet decisions", "Financial stability assessment"]),
    MacroEvent(date(2026, 6, 17), "WAR DAY 109  --  90-DAY MODEL ENDPOINT", "thesis", EventImpact.CRITICAL,
               "End of projection window. Full thesis evaluation.",
               ["COMPLETE REVIEW", "Calculate total P&L",
                "Next 90-day plan if thesis active",
                "Begin hard asset transition if confirmed"]),
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
# PART 3  --  INDICATOR PROGRESSION MODEL
# Models how 7 key indicators evolve day-by-day across 180 days
# ===========================================================================

@dataclass
class DailyIndicatorState:
    """Snapshot of all key indicators for a single day."""
    date: date
    war_day: int                # -1 means pre-war
    oil_wti: float              # WTI crude $/barrel
    gold_spot: float            # Gold $/oz
    vix: float                  # VIX index
    dxy: float                  # Dollar index
    btc_usd: float              # Bitcoin USD
    hy_spread_bps: int          # HY OAS spread basis points
    spy_price: float            # SPY ETF price
    thesis_pressure_pct: float  # 0-100 composite pressure score
    phase: str                  # PRECURSOR through BLACK_SWAN_EVENT
    confidence: float           # Model confidence 0-1
    notes: str = ""


def _war_day(d: date) -> int:
    """Calculate war day number (-1 for pre-war)."""
    if d < WAR_START_DATE:
        return -1
    return (d - WAR_START_DATE).days


def _interpolate(start: float, end: float, progress: float,
                 noise_pct: float = 0.02) -> float:
    """Interpolate between two values with optional variance."""
    base = start + (end - start) * progress
    # Deterministic pseudo-noise based on progress
    noise = math.sin(progress * 47.3) * noise_pct * base
    return round(base + noise, 2)


def build_historical_model() -> List[DailyIndicatorState]:
    """Build day-by-day model for the past 90 days (Dec 19 2025 -> Mar 19 2026).

    This reconstructs the indicator progression based on known events:
    - Dec-Jan: Calm pre-war, building tensions
    - Late Feb: War starts, immediate market reaction
    - Mar: War escalation, pressure building
    """
    days = []
    total_days = (MODEL_TODAY - MODEL_START).days

    # Phase 1: Pre-war calm (Dec 19 -> Feb 27)  --  70 days
    # Phase 2: War shock (Feb 28 -> Mar 7)  --  7 days
    # Phase 3: Escalation (Mar 8 -> Mar 19)  --  12 days

    for i in range(total_days + 1):
        d = MODEL_START + timedelta(days=i)
        wd = _war_day(d)
        progress = i / total_days

        if d < date(2026, 2, 28):
            # PRE-WAR: Calm with undercurrents
            pre_progress = i / 70.0
            oil = _interpolate(70.0, 78.0, pre_progress, 0.01)
            gold = _interpolate(2650.0, 2920.0, pre_progress, 0.005)
            vix = _interpolate(14.0, 18.0, pre_progress, 0.03)
            dxy = _interpolate(106.5, 105.0, pre_progress, 0.005)
            btc = _interpolate(97000.0, 85000.0, pre_progress, 0.02)
            hy_spread = int(_interpolate(320, 370, pre_progress, 0.02))
            spy = _interpolate(598.0, 575.0, pre_progress, 0.008)
            pressure = _interpolate(8.0, 22.0, pre_progress, 0.05)
            phase = "PRECURSOR"
            conf = 0.3 + pre_progress * 0.1
            notes = ""

            # Key pre-war events
            if d == date(2026, 1, 20):
                notes = "Trump inauguration. Iran policy direction TBD."
                pressure += 3
            elif d == date(2026, 2, 20):
                notes = "OPEC+ meeting. Tensions rising."
                oil += 2
                pressure += 2

        elif d <= date(2026, 3, 7):
            # WAR SHOCK: Immediate market dislocation (days 0-7)
            shock_day = (d - WAR_START_DATE).days
            shock_progress = shock_day / 7.0
            oil = _interpolate(78.0, 92.0, shock_progress, 0.03)
            gold = _interpolate(2920.0, 3020.0, shock_progress, 0.01)
            vix = _interpolate(18.0, 28.0, shock_progress, 0.05)
            dxy = _interpolate(105.0, 106.5, shock_progress, 0.01)  # initial dollar strength
            btc = _interpolate(85000.0, 76000.0, shock_progress, 0.04)
            hy_spread = int(_interpolate(370, 450, shock_progress, 0.03))
            spy = _interpolate(575.0, 548.0, shock_progress, 0.02)
            pressure = _interpolate(22.0, 35.0, shock_progress, 0.03)
            phase = "BUILDING"
            conf = 0.5 + shock_progress * 0.1
            notes = f"War Day {shock_day}. Initial shock wave."

            if shock_day == 0:
                notes = "WAR BEGINS. US strikes Iran. Oil gaps up. Markets in shock."
            elif shock_day == 1:
                notes = "Day 1: Full risk-off. VIX spike. Gold safe haven bid."

        else:
            # ESCALATION: Mar 8-19 (days 8-19 of war)
            esc_day = (d - date(2026, 3, 8)).days
            esc_progress = esc_day / 12.0
            oil = _interpolate(92.0, 88.0, esc_progress, 0.02)  # settles after shock
            gold = _interpolate(3020.0, 3050.0, esc_progress, 0.005)
            vix = _interpolate(28.0, 22.0, esc_progress, 0.04)  # vol compression
            dxy = _interpolate(106.5, 104.0, esc_progress, 0.008)
            btc = _interpolate(76000.0, 74500.0, esc_progress, 0.03)
            hy_spread = int(_interpolate(450, 420, esc_progress, 0.02))
            spy = _interpolate(548.0, 562.0, esc_progress, 0.01)  # dead cat bounce
            pressure = _interpolate(35.0, 44.0, esc_progress, 0.02)
            phase = "BUILDING"
            conf = 0.55 + esc_progress * 0.05

            if d == date(2026, 3, 12):
                notes = "Hormuz partial closure. Insurance premiums spike. Yuan passage proposed."
                oil += 5
                pressure += 3
            elif d == date(2026, 3, 18):
                notes = "LIVE TRADES DEPLOYED. 8 puts on IBKR. NDAX liquidated."
            elif d == date(2026, 3, 19):
                notes = "TODAY. Fed decision. Pressure 44%. Phase 1 BUILDING."
            else:
                notes = f"War Day {_war_day(d)}. Markets digesting."

        # Weekend adjustment (no trading but geopolitical events continue)
        if d.weekday() >= 5:
            vix = round(vix * 0.98, 2)  # VIX doesn't update weekends but futures gap
            notes = (notes + " [Weekend  --  monitor geopolitical developments]").strip()

        days.append(DailyIndicatorState(
            date=d, war_day=wd, oil_wti=oil, gold_spot=gold,
            vix=vix, dxy=dxy, btc_usd=btc, hy_spread_bps=hy_spread,
            spy_price=spy, thesis_pressure_pct=round(pressure, 1),
            phase=phase, confidence=round(conf, 2), notes=notes,
        ))

    return days


def build_forward_model() -> List[DailyIndicatorState]:
    """Build day-by-day projection for next 90 days (Mar 20 -> Jun 17 2026).

    Three scenario tracks modeled simultaneously  --  returns the EXPECTED
    (probability-weighted) path. Individual scenarios can be queried separately.

    Scenario weights:
      - THESIS_FAILS (45%):  Ceasefire, oil drops, markets recover
      - MODERATE_CRISIS (30%): Partial thesis, oil $120, spreads +200bps
      - MAJOR_CRISIS (20%):  Hormuz closed, oil $180, credit crisis
      - BLACK_SWAN (5%):     Full thesis  --  gold $5K+, DXY collapse
    """
    days = []
    total_days = (MODEL_END - MODEL_TODAY).days

    # Starting values (from today Mar 19)
    s = {
        "oil": 88.0, "gold": 3050.0, "vix": 22.0, "dxy": 104.0,
        "btc": 74500.0, "hy_spread": 420, "spy": 562.0, "pressure": 44.0,
    }

    # Scenario endpoints at day 90
    scenarios = {
        "fails": {
            "prob": 0.45, "oil": 65.0, "gold": 2800.0, "vix": 14.0,
            "dxy": 108.0, "btc": 90000.0, "hy_spread": 300, "spy": 610.0,
            "pressure": 15.0,
        },
        "moderate": {
            "prob": 0.30, "oil": 120.0, "gold": 3500.0, "vix": 32.0,
            "dxy": 100.0, "btc": 60000.0, "hy_spread": 600, "spy": 490.0,
            "pressure": 65.0,
        },
        "major": {
            "prob": 0.20, "oil": 180.0, "gold": 5000.0, "vix": 50.0,
            "dxy": 92.0, "btc": 45000.0, "hy_spread": 900, "spy": 420.0,
            "pressure": 85.0,
        },
        "blackswan": {
            "prob": 0.05, "oil": 220.0, "gold": 8000.0, "vix": 80.0,
            "dxy": 82.0, "btc": 30000.0, "hy_spread": 1500, "spy": 350.0,
            "pressure": 98.0,
        },
    }

    for i in range(1, total_days + 1):
        d = MODEL_TODAY + timedelta(days=i)
        progress = i / total_days
        wd = _war_day(d)

        # Probability-weighted expected values
        expected = {}
        for key in ["oil", "gold", "vix", "dxy", "btc", "hy_spread", "spy", "pressure"]:
            weighted = sum(
                sc["prob"] * _interpolate(s[key], sc[key], progress, 0.015)
                for sc in scenarios.values()
            )
            expected[key] = round(weighted, 2)

        # Phase determination from pressure
        pressure = expected["pressure"]
        if pressure < 25:
            phase = "PRECURSOR"
        elif pressure < 45:
            phase = "BUILDING"
        elif pressure < 65:
            phase = "ACCELERATION"
        elif pressure < 80:
            phase = "CRISIS_ONSET"
        elif pressure < 95:
            phase = "FULL_CRISIS"
        else:
            phase = "BLACK_SWAN_EVENT"

        # Confidence decays with distance from today
        conf = max(0.15, 0.60 - progress * 0.50)

        # Event-based notes
        notes = ""
        events = get_events_for_date(d)
        if events:
            notes = " | ".join(e.name for e in events)

        # Key inflection points
        if wd == 30:
            notes += " MONTH 1 COMPLETE  --  full review mandatory."
        elif wd == 45:
            notes += " 45-DAY CHECKPOINT  --  thesis confirm/deny pivot."
        elif wd == 60:
            notes += " 2 MONTHS  --  structural positions only from here."
        elif wd == 90:
            notes += " 3 MONTHS  --  geopolitical resolution window."

        days.append(DailyIndicatorState(
            date=d, war_day=wd,
            oil_wti=expected["oil"],
            gold_spot=expected["gold"],
            vix=expected["vix"],
            dxy=expected["dxy"],
            btc_usd=expected["btc"],
            hy_spread_bps=int(expected["hy_spread"]),
            spy_price=expected["spy"],
            thesis_pressure_pct=round(pressure, 1),
            phase=phase,
            confidence=round(conf, 2),
            notes=notes.strip(),
        ))

    return days


# ===========================================================================
# PART 4  --  CORRELATION MATRIX
# How thesis indicators move together
# ===========================================================================

# Historical correlation coefficients (estimated from data + thesis logic)
# Values: -1.0 (inverse) to +1.0 (correlated)
CORRELATION_MATRIX = {
    # In a thesis-confirming environment:
    ("oil", "gold"):       0.65,   # Both rise on war/inflation
    ("oil", "vix"):        0.72,   # Oil shock = volatility spike
    ("oil", "dxy"):       -0.45,   # Oil up + petrodollar stress = DXY down
    ("oil", "btc"):       -0.30,   # Oil shock = risk-off, BTC dumps initially
    ("oil", "hy_spread"):  0.78,   # Oil shock = credit stress
    ("oil", "spy"):       -0.68,   # Oil shock = equities down

    ("gold", "vix"):       0.40,   # Gold rises in fear
    ("gold", "dxy"):      -0.80,   # Gold up = dollar down (structural inverse)
    ("gold", "btc"):       0.15,   # Weak positive  --  both "alternatives"
    ("gold", "hy_spread"): 0.35,   # Credit stress = gold bid
    ("gold", "spy"):      -0.55,   # Gold up = equities under pressure

    ("vix", "dxy"):       -0.20,   # Complex  --  initial dollar strength in panic
    ("vix", "btc"):       -0.60,   # VIX up = BTC down (risk-off)
    ("vix", "hy_spread"):  0.85,   # VIX and credit spreads highly correlated
    ("vix", "spy"):       -0.92,   # VIX inverse to SPY (near-perfect)

    ("dxy", "btc"):        0.10,   # Weak  --  BTC loosely correlated with risk
    ("dxy", "hy_spread"): -0.30,   # Dollar strength + credit stress = complex
    ("dxy", "spy"):        0.25,   # Weak positive in normal times

    ("btc", "hy_spread"): -0.45,   # Credit stress = BTC down
    ("btc", "spy"):        0.55,   # BTC tracks equities in risk-off

    ("hy_spread", "spy"): -0.80,   # Spreads widen = equities fall
}


def get_correlation(a: str, b: str) -> float:
    """Get correlation coefficient between two indicators."""
    if a == b:
        return 1.0
    return CORRELATION_MATRIX.get((a, b), CORRELATION_MATRIX.get((b, a), 0.0))


def render_correlation_matrix() -> str:
    """Render the full correlation matrix as text."""
    indicators = ["oil", "gold", "vix", "dxy", "btc", "hy_spread", "spy"]
    labels = {"oil": "OIL", "gold": "GOLD", "vix": "VIX", "dxy": "DXY",
              "btc": "BTC", "hy_spread": "HY_SPR", "spy": "SPY"}

    lines = [
        "  CORRELATION MATRIX  --  Thesis Environment",
        "  " + "=" * 65,
        "",
        "  {:>8}".format("") + "".join(f"  {labels[i]:>7}" for i in indicators),
    ]

    for row in indicators:
        vals = []
        for col in indicators:
            c = get_correlation(row, col)
            if c == 1.0:
                vals.append("    1.00")
            elif c >= 0.7:
                vals.append(f"  {c:>+5.2f}+")
            elif c <= -0.7:
                vals.append(f"  {c:>+5.2f}-")
            else:
                vals.append(f"  {c:>+5.2f} ")
        lines.append(f"  {labels[row]:>8}" + "".join(vals))

    lines.append("")
    lines.append("  KEY: + = strong positive (>0.7)  - = strong negative (<-0.7)")
    lines.append("")
    lines.append("  CONFIDENT ASSUMPTIONS (|corr| > 0.7):")
    lines.append("  * VIX <-> SPY: -0.92  -> VIX spikes = SPY tanks (near certain)")
    lines.append("  * VIX <-> HY:  +0.85  -> VIX spikes = credit spreads blow out")
    lines.append("  * HY <-> SPY:  -0.80  -> Credit stress = equity selloff")
    lines.append("  * GOLD <-> DXY: -0.80 -> Gold rises = dollar weakens")
    lines.append("  * OIL <-> HY:  +0.78  -> Oil shock = credit stress")
    lines.append("  * OIL <-> VIX: +0.72  -> Oil shock = volatility spike")
    lines.append("")
    lines.append("  RANDOM FACTORS (|corr| < 0.3):")
    lines.append("  * BTC <-> DXY: +0.10  -> BTC decoupled from dollar (unreliable)")
    lines.append("  * BTC <-> GOLD: +0.15 -> BTC is NOT digital gold in a crisis")
    lines.append("  * VIX <-> DXY: -0.20  -> Dollar direction in panic is uncertain")

    return "\n".join(lines)


# ===========================================================================
# PART 5  --  CAPITAL DEPLOYMENT ENGINE
# Day-by-day capital allocation across all accounts
# ===========================================================================

@dataclass
class DailyMandate:
    """Complete daily action mandate."""
    date: date
    war_day: int
    phase: str

    # Intelligence
    pressure_score: float
    key_indicators: Dict[str, float]
    events_today: List[str]
    correlation_alerts: List[str]

    # Portfolio actions
    account_actions: Dict[str, List[str]]  # account_name -> actions
    capital_to_deploy: float
    cumulative_deployed: float

    # Risk management
    risk_level: str                        # GREEN, YELLOW, ORANGE, RED, BLACK
    stop_losses: List[str]
    hedge_adjustments: List[str]

    # Intel requirements
    morning_checklist: List[str]
    evening_checklist: List[str]
    thesis_confirmation_signals: List[str]
    thesis_denial_signals: List[str]


def _risk_level(pressure: float, vix: float) -> str:
    """Determine operational risk level."""
    if pressure >= 80 or vix >= 50:
        return "BLACK"
    elif pressure >= 65 or vix >= 35:
        return "RED"
    elif pressure >= 45 or vix >= 25:
        return "ORANGE"
    elif pressure >= 25 or vix >= 20:
        return "YELLOW"
    return "GREEN"


def generate_daily_mandate(target_date: date,
                           indicator_state: Optional[DailyIndicatorState] = None,
                           ) -> DailyMandate:
    """Generate the complete daily mandate for a given date."""
    # Find or model the indicator state for this date
    if indicator_state is None:
        # Build model and find the date
        if target_date <= MODEL_TODAY:
            model = build_historical_model()
        else:
            model = build_forward_model()
        matches = [d for d in model if d.date == target_date]
        if matches:
            indicator_state = matches[0]
        else:
            # Fallback to today's known state
            indicator_state = DailyIndicatorState(
                date=target_date, war_day=_war_day(target_date),
                oil_wti=88.0, gold_spot=3050.0, vix=22.0, dxy=104.0,
                btc_usd=74500.0, hy_spread_bps=420, spy_price=562.0,
                thesis_pressure_pct=44.0, phase="BUILDING",
                confidence=0.5,
            )

    st = indicator_state
    wd = st.war_day
    events = get_events_for_date(target_date)
    event_names = [e.name for e in events]

    # -- Capital deployment logic --
    total_available = sum(a.available_cash * (1 if a.currency == Currency.USD else CAD_USD)
                         for a in ACCOUNTS.values())
    injection_remaining = CASH_INJECTION
    days_from_today = (target_date - MODEL_TODAY).days

    # Deploy injection in tranches over 30 days
    if 0 <= days_from_today <= 30:
        daily_deploy = (CASH_INJECTION * 0.70) / 30  # 70% over 30 days
    elif 30 < days_from_today <= 60:
        daily_deploy = (CASH_INJECTION * 0.20) / 30  # 20% over next 30
    elif days_from_today > 60:
        daily_deploy = (CASH_INJECTION * 0.10) / 30  # 10% final 30 days
    else:
        daily_deploy = 0

    cumulative = daily_deploy * max(0, days_from_today)

    # -- Account-specific actions --
    account_actions = {}

    # IBKR  --  primary options venue
    ibkr_actions = []
    if wd >= 0 and days_from_today >= 0:
        if st.thesis_pressure_pct > 50:
            ibkr_actions.append("SCALE existing puts  --  add to winners (credit vertical)")
        if st.oil_wti > 95 and days_from_today <= 14:
            ibkr_actions.append("ENTER oil vertical: USO/XLE calls")
        if st.gold_spot > 3100:
            ibkr_actions.append("ENTER gold vertical: GLD/GDX calls")
        if st.vix > 25:
            ibkr_actions.append("ENTER vol vertical: UVXY calls")
        if st.vix < 18 and st.thesis_pressure_pct > 40:
            ibkr_actions.append("CHEAP VOL: Buy UVXY/VIX calls while suppressed")
        if days_from_today % 7 == 0 and days_from_today > 0:
            ibkr_actions.append("WEEKLY REVIEW: Check all option Greeks, roll if needed")

        # Position-specific management
        if st.hy_spread_bps > 500:
            ibkr_actions.append("CREDIT PUTS WINNING: Take 30% partial profit on HYG/JNK")
        if st.spy_price < 520:
            ibkr_actions.append("IWM/KRE PUTS WINNING: Consider rolling to deeper strikes")
    if not ibkr_actions:
        ibkr_actions.append("MONITOR  --  no action triggers met")
    account_actions["IBKR ($920)"] = ibkr_actions

    # NDAX  --  CAD cash, crypto re-entry or conversion
    ndax_actions = []
    if st.btc_usd < 60000:
        ndax_actions.append("BTC ACCUMULATION ZONE: Consider small BTC position ($500 CAD)")
    if st.btc_usd < 45000:
        ndax_actions.append("BTC DEEP VALUE: Larger allocation ($1,500 CAD)")
    if st.thesis_pressure_pct > 70:
        ndax_actions.append("CONVERT CAD->USD for options capital (use for IBKR deposit)")
    if days_from_today == 0:
        ndax_actions.append("HOLD as CAD cash. Crypto is risk-off. Wait for opportunity.")
    if not ndax_actions:
        ndax_actions.append("HOLD CAD cash  --  no crypto entry signals")
    account_actions["NDAX ($4,400 CAD)"] = ndax_actions

    # Moomoo  --  options if approved, else stock
    moo_actions = []
    if days_from_today < 7:
        moo_actions.append("CHECK options approval status (applied Mar 15)")
    if days_from_today >= 7:
        moo_actions.append("If options approved: deploy $300 in credit/equity puts")
    moo_actions.append("Research US-listed gold miners (GDX, GDXJ) for equity positions")
    account_actions["Moomoo ($500)"] = moo_actions

    # WealthSimple TFSA  --  tax-free, Canadian-listed
    ws_actions = []
    ws_actions.append("TAX-FREE GROWTH: Focus on Canadian-listed ETFs")
    if st.gold_spot > 3000:
        ws_actions.append("BUY CGL.TO (CI Gold Bullion ETF)  --  gold in TFSA, tax-free gains")
    if st.oil_wti > 85:
        ws_actions.append("BUY HUC.TO (Horizons Crude Oil ETF) or XEG.TO (iShares Energy)")
    if st.thesis_pressure_pct > 60:
        ws_actions.append("BUY HGD.TO (Horizons Gold Bear) or HSD.TO (S&P Bear)")
    if days_from_today % 14 == 0 and days_from_today > 0:
        ws_actions.append("BI-WEEKLY TFSA REVIEW: Rebalance thesis-aligned positions")
    account_actions["WealthSimple TFSA ($4,200 CAD)"] = ws_actions

    # EQ Bank  --  buffer
    account_actions["EQ Bank ($100 CAD)"] = ["HOLD as emergency buffer. Earn interest."]

    # Cash injection deployment
    if days_from_today >= 0:
        inject_actions = []
        pct = min(100, max(0, days_from_today / 30 * 100))
        inject_actions.append(f"Injection deployment: ~{pct:.0f}% of Phase 1 (70% in first 30d)")
        if days_from_today <= 3:
            inject_actions.append("IMMEDIATE: $5,000 to IBKR for options")
            inject_actions.append("IMMEDIATE: $3,000 to WealthSimple for CGL.TO/XEG.TO")
            inject_actions.append("RESERVE: $10,000 as dry powder for Phase 2/3 triggers")
        elif days_from_today <= 7:
            inject_actions.append("Deploy $3,000 more to IBKR: oil + gold verticals")
        elif days_from_today <= 14:
            inject_actions.append("Evaluate: deploy another $5,000 based on thesis confirmation")
        elif days_from_today <= 30:
            inject_actions.append("Remaining Phase 1: spread across best-performing verticals")
        elif days_from_today <= 60:
            inject_actions.append("Phase 2 (20%): $7,000 into confirmed thesis vectors only")
        else:
            inject_actions.append("Phase 3 (10%): Final $3,500 into highest-conviction plays")
        account_actions["$35K INJECTION"] = inject_actions

    # -- Correlation alerts --
    corr_alerts = []
    if st.vix > 25 and st.hy_spread_bps > 450:
        corr_alerts.append("VIX+HY ALIGNED (corr 0.85): Credit stress confirmed by vol")
    if st.gold_spot > 3100 and st.dxy < 103:
        corr_alerts.append("GOLD UP + DXY DOWN ALIGNED (corr -0.80): Dollar erosion thesis live")
    if st.oil_wti > 95 and st.vix > 25:
        corr_alerts.append("OIL+VIX ALIGNED (corr 0.72): Oil shock driving broad fear")
    if st.btc_usd > 80000 and st.spy_price > 575:
        corr_alerts.append("BTC+SPY BOTH UP: Risk-on environment  --  THESIS WEAKENING")
    if st.oil_wti < 75 and st.vix < 16:
        corr_alerts.append("OIL+VIX BOTH LOW: De-escalation priced in  --  increase hedges")

    # -- Risk management --
    rl = _risk_level(st.thesis_pressure_pct, st.vix)
    stop_losses = []
    hedges = []
    if rl in ("RED", "BLACK"):
        stop_losses.append("Set trailing stops at 30% on all option positions")
        stop_losses.append("If any put >500% gain: take 50% off table")
        hedges.append("HEDGE: Buy SPY calls (5% of portfolio) as anti-thesis insurance")
    if rl == "BLACK":
        stop_losses.append("PROTECT GAINS: Move to 60% hard assets (gold, CAD cash)")
        hedges.append("EXIT financial assets systematically over 5 trading days")

    # -- Checklists --
    morning = [
        f"Check oil price (target: {'UP' if st.thesis_pressure_pct > 40 else 'FLAT'})",
        "Check gold spot (compare to yesterday)",
        "Check VIX level (action if >25 or <16)",
        "Check DXY (action if <100 or >108)",
        "Scan NewsAPI/YouTube for authority updates",
        "Review overnight geopolitical developments",
    ]
    if events:
        morning.insert(0, f"TODAY'S EVENTS: {', '.join(event_names)}")

    evening = [
        "Log day's indicator values to war_room data",
        "Update pressure cooker scores if new intel",
        "Review all open positions P&L",
        "Check authority YouTube feeds for new content",
        "Prepare tomorrow's mandate adjustments",
    ]

    confirm_signals = [
        "Oil sustained >$100",
        "HY spreads >500bps for 3+ consecutive days",
        "Gold >$3,200 with momentum",
        "DXY breakdown below 100",
        "Multiple Gulf states announce yuan oil trades",
        "US troop withdrawal from any ME base",
    ]

    deny_signals = [
        "Ceasefire announced and holding",
        "Oil back below $70",
        "VIX back below 15 for 5+ days",
        "DXY rallying above 108",
        "Iran capitulation / regime change",
        "SPY making new all-time highs",
    ]

    return DailyMandate(
        date=target_date,
        war_day=wd,
        phase=st.phase,
        pressure_score=st.thesis_pressure_pct,
        key_indicators={
            "OIL_WTI": st.oil_wti,
            "GOLD": st.gold_spot,
            "VIX": st.vix,
            "DXY": st.dxy,
            "BTC": st.btc_usd,
            "HY_SPREAD": st.hy_spread_bps,
            "SPY": st.spy_price,
        },
        events_today=event_names,
        correlation_alerts=corr_alerts,
        account_actions=account_actions,
        capital_to_deploy=round(daily_deploy, 2),
        cumulative_deployed=round(cumulative, 2),
        risk_level=rl,
        stop_losses=stop_losses,
        hedge_adjustments=hedges,
        morning_checklist=morning,
        evening_checklist=evening,
        thesis_confirmation_signals=confirm_signals,
        thesis_denial_signals=deny_signals,
    )


# ===========================================================================
# PART 6  --  INTEL UPDATE SYSTEM
# Log and retrieve intelligence updates (twice daily)
# ===========================================================================

def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_intel_update(note: str, indicators: Optional[Dict] = None):
    """Log an intelligence update with timestamp."""
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

    print(f"  Intel logged at {ts.strftime('%Y-%m-%d %H:%M')} (War Day {entry['war_day']})")


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


# ===========================================================================
# PART 7  --  SCENARIO PROJECTIONS
# What happens at each scenario track day-by-day
# ===========================================================================

def project_portfolio_value(day_offset: int, scenario: str = "expected") -> Dict[str, Any]:
    """Project portfolio value at a given day offset under a scenario.

    Accounts for:
    - Existing positions (8 puts) with scenario-dependent returns
    - Cash injection deployment schedule
    - New positions entered per deployment schedule
    - Currency conversion (CAD accounts)
    """
    # Current portfolio
    existing_options = 760.0   # 8 puts cost basis
    ibkr_cash = 160.0
    ndax_cad = 4400.0
    moomoo_usd = 500.0
    ws_cad = 4200.0
    eq_cad = 100.0

    # Option multipliers by scenario at various time horizons
    option_mults = {
        "fails":     {30: 0.50, 60: 0.20, 90: 0.10},   # Options decay to near-zero
        "moderate":  {30: 2.50, 60: 5.00, 90: 8.00},    # Solid returns
        "major":     {30: 5.00, 60: 12.00, 90: 18.00},  # Major crisis payoff
        "blackswan": {30: 10.0, 60: 25.00, 90: 40.00},  # Life-changing returns
        "expected":  {30: 1.80, 60: 3.50, 90: 5.20},    # Probability-weighted
    }

    # Nearest bucket
    if day_offset <= 30:
        bucket = 30
    elif day_offset <= 60:
        bucket = 60
    else:
        bucket = 90

    mult = option_mults.get(scenario, option_mults["expected"]).get(bucket, 1.0)
    # Linear interpolation within bucket
    if bucket == 30:
        frac = day_offset / 30
        mult = 1.0 + (mult - 1.0) * frac
    elif bucket == 60:
        prev_mult = option_mults.get(scenario, option_mults["expected"])[30]
        frac = (day_offset - 30) / 30
        mult = prev_mult + (mult - prev_mult) * frac
    else:
        prev_mult = option_mults.get(scenario, option_mults["expected"])[60]
        frac = (day_offset - 60) / 30
        mult = prev_mult + (mult - prev_mult) * frac

    existing_value = existing_options * mult

    # Injection deployed by this day
    if day_offset <= 30:
        injected = CASH_INJECTION * 0.70 * (day_offset / 30)
    elif day_offset <= 60:
        injected = CASH_INJECTION * 0.70 + CASH_INJECTION * 0.20 * ((day_offset - 30) / 30)
    else:
        injected = CASH_INJECTION * 0.90 + CASH_INJECTION * 0.10 * ((day_offset - 60) / 30)

    # New positions return (assume similar mult but from later entry)
    new_positions_value = injected * 0.70 * max(1.0, mult * 0.6)  # 70% of injection into positions
    injection_cash = injected * 0.30  # 30% kept as dry powder

    # Non-position accounts (grow at savings rate or hold)
    cad_accounts = (ndax_cad + ws_cad + eq_cad) * CAD_USD

    total = (
        existing_value +
        ibkr_cash +
        moomoo_usd +
        cad_accounts +
        new_positions_value +
        injection_cash +
        (CASH_INJECTION - injected)  # undeployed injection
    )

    return {
        "day_offset": day_offset,
        "scenario": scenario,
        "existing_puts_value": round(existing_value, 2),
        "existing_puts_mult": f"{mult:.1f}x",
        "injection_deployed": round(injected, 2),
        "new_positions_value": round(new_positions_value, 2),
        "cad_accounts_usd": round(cad_accounts, 2),
        "cash_reserves": round(ibkr_cash + moomoo_usd + injection_cash + (CASH_INJECTION - injected), 2),
        "total_portfolio_usd": round(total, 2),
        "total_return_pct": round((total / (sum(a.balance_usd for a in ACCOUNTS.values()) + CASH_INJECTION) - 1) * 100, 1),
    }


# ===========================================================================
# PART 8  --  RENDERING ENGINE
# ===========================================================================

def render_dashboard() -> str:
    """Render the complete War Room dashboard."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("=  90-DAY WAR ROOM -- AAC SUPREME COMMAND CENTER                              =")
    lines.append("=" * 80)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  War Day: {_war_day(MODEL_TODAY)}  |  Model: {MODEL_START} -> {MODEL_END}")
    lines.append(f"  Thesis: Iran war -> US pullback -> Gulf yuan -> gold reprice -> USD collapse")
    lines.append("")

    # -- PORTFOLIO SUMMARY --
    port = get_total_portfolio()
    lines.append("  === PORTFOLIO STATE =========================================")
    lines.append(f"  Current Total (USD equiv):  ${port['total_usd_equivalent']:>12,.2f}")
    lines.append(f"  Cash Injection Pending:     ${CASH_INJECTION:>12,.2f}")
    lines.append(f"  POST-INJECTION TOTAL:       ${port['post_injection_total']:>12,.2f}")
    lines.append(f"  In Positions:               ${port['positions_value']:>12,.2f}  ({port['num_positions']} puts)")
    lines.append(f"  Unrealized P&L:             ${port['unrealized_pnl']:>12,.2f}")
    lines.append("")

    for name, acct in ACCOUNTS.items():
        tag = f"${acct.balance:,.2f} {acct.currency.value}"
        usd = f"(~${acct.balance_usd:,.2f} USD)" if acct.currency == Currency.CAD else ""
        lines.append(f"  {acct.name:<25} {tag:>15}  {usd}")
    lines.append("")

    # -- TODAY'S MANDATE --
    mandate = generate_daily_mandate(MODEL_TODAY)
    lines.append("  === TODAY'S MANDATE (Mar 19, 2026) =============================")
    lines.append(f"  Phase: {mandate.phase}  |  Pressure: {mandate.pressure_score}%  |  Risk: {mandate.risk_level}")
    lines.append("")

    lines.append("  KEY INDICATORS:")
    for k, v in mandate.key_indicators.items():
        lines.append(f"    {k:<12} {v:>12,.2f}")
    lines.append("")

    if mandate.events_today:
        lines.append("  TODAY'S EVENTS:")
        for e in mandate.events_today:
            lines.append(f"    [*] {e}")
        lines.append("")

    if mandate.correlation_alerts:
        lines.append("  CORRELATION ALERTS:")
        for a in mandate.correlation_alerts:
            lines.append(f"    >> {a}")
        lines.append("")

    lines.append("  ACCOUNT ACTIONS:")
    for acct, actions in mandate.account_actions.items():
        lines.append(f"    +-- {acct}")
        for action in actions:
            lines.append(f"    |   * {action}")
        lines.append(f"    +----------------------------")
    lines.append("")

    lines.append("  MORNING CHECKLIST:")
    for item in mandate.morning_checklist:
        lines.append(f"    [ ] {item}")
    lines.append("")
    lines.append("  EVENING CHECKLIST:")
    for item in mandate.evening_checklist:
        lines.append(f"    [ ] {item}")
    lines.append("")

    # -- SCENARIO PROJECTIONS --
    lines.append("  === 90-DAY SCENARIO PROJECTIONS ================================")
    for scenario in ["fails", "moderate", "major", "blackswan", "expected"]:
        lines.append(f"")
        lines.append(f"  +-- {scenario.upper()}")
        for day in [30, 60, 90]:
            proj = project_portfolio_value(day, scenario)
            lines.append(
                f"  |   Day {day:>2}: ${proj['total_portfolio_usd']:>12,.2f}  "
                f"({proj['total_return_pct']:>+7.1f}%)  "
                f"Puts: {proj['existing_puts_mult']}"
            )
        lines.append(f"  +------------------------------------------------------")
    lines.append("")

    # -- CORRELATION MATRIX --
    lines.append(render_correlation_matrix())

    # -- FORWARD CALENDAR PREVIEW (next 30 days) --
    lines.append("  === NEXT 30 DAYS -- EVENT CALENDAR =============================")
    upcoming = get_events_in_range(MODEL_TODAY, MODEL_TODAY + timedelta(days=30))
    for ev in upcoming:
        wd = _war_day(ev.date)
        impact_tag = {"CRITICAL": "!!!!", "HIGH": "!!! ", "MEDIUM": "!!  ",
                      "LOW": "!   ", "THESIS": "****"}.get(ev.impact.value, "    ")
        lines.append(
            f"  {ev.date.strftime('%b %d')} (WD{wd:>3}) [{impact_tag}] {ev.name}"
        )
        if ev.thesis_relevance:
            lines.append(f"                        -> {ev.thesis_relevance[:75]}")
    lines.append("")

    # -- CONFIDENCE ASSUMPTIONS --
    lines.append("  === CONFIDENT ASSUMPTIONS =====================================")
    lines.append("  HIGH CONFIDENCE (>70% probability):")
    lines.append("    - War continues for >30 days (already at day 19, no ceasefire signal)")
    lines.append("    - Oil stays above $80 while Hormuz disrupted")
    lines.append("    - VIX spikes >30 on any major escalation (corr 0.72 with oil)")
    lines.append("    - Credit spreads widen further if oil >$100 (corr 0.78)")
    lines.append("    - Fed holds rates at March meeting (inflation + growth uncertainty)")
    lines.append("    - Gold continues uptrend in war environment ($3,050 -> $3,200+)")
    lines.append("")
    lines.append("  MODERATE CONFIDENCE (40-70% probability):")
    lines.append("    - Hormuz disruption escalates to full closure in next 30 days")
    lines.append("    - Oil reaches $120+ by May if Hormuz closes")
    lines.append("    - Q1 earnings show credit deterioration in bank results")
    lines.append("    - GDP Q1 comes in under 1% (war + uncertainty drag)")
    lines.append("    - At least 1 Gulf state announces yuan oil trade by June")
    lines.append("")
    lines.append("  LOW CONFIDENCE / RANDOM FACTORS:")
    lines.append("    - Ceasefire timing -- could be tomorrow or 6 months (unknown)")
    lines.append("    - Trump rhetoric direction -- unpredictable by nature")
    lines.append("    - BTC behavior in war -- historical precedent doesn't exist")
    lines.append("    - China mediation timing -- opaque decision-making")
    lines.append("    - Private credit fund gate timing -- off-balance-sheet unknown")
    lines.append("    - Netanyahu status -- the ultimate unknown unknown")
    lines.append("")

    # -- 13-WEEK ROADMAP --
    lines.append("  === 13-WEEK STRATEGIC ROADMAP ==================================")
    weeks = [
        ("Week 1 (Mar 19-25)", "FOUNDATION",
         "Deploy initial injection ($8K to IBKR + $3K to WS). Fill oil+gold gaps. "
         "Monitor Fed reaction. Daily pressure cooker scans."),
        ("Week 2 (Mar 26-Apr 1)", "EXPAND",
         "Add oil vertical (USO/XLE). Add gold vertical (CGL.TO in TFSA). "
         "Evaluate Moomoo options approval. War Day 30+ zone."),
        ("Week 3 (Apr 2-8)", "CONFIRM",
         "First war-month data lands (jobs, CPI). If confirming -> scale up. "
         "If denying -> increase hedges. OPEC meeting critical."),
        ("Week 4 (Apr 9-15)", "EARNINGS",
         "Bank earnings. Credit loss provisions = thesis signal. "
         "Deploy another $5K based on data. Tax day selling pressure."),
        ("Week 5 (Apr 16-22)", "TECH EARNINGS",
         "Tech earnings + war impact guidance. AI bubble deflation? "
         "QQQ puts if earnings disappoint. 45-day war checkpoint."),
        ("Week 6 (Apr 23-29)", "GDP WEEK",
         "Q1 GDP advance = CRITICAL number. If <1% -> full thesis acceleration. "
         "If negative -> panic positions. Deploy Phase 2 injection."),
        ("Week 7 (Apr 30-May 6)", "2 MONTHS",
         "Fed May meeting. 2 months of war. Scale confirmed vectors. "
         "Take partial profits on >500% winners. Rebalance across 8 verticals."),
        ("Week 8 (May 7-13)", "INFLATION",
         "Second war-month CPI. If >5% headline -> stagflation narrative. "
         "Add dollar death vertical if DXY breaking down."),
        ("Week 9 (May 14-20)", "ENDGAME PREP",
         "80-day mark. Begin hard asset transition if thesis >70%. "
         "Convert winning options to gold-backed positions."),
        ("Week 10 (May 21-27)", "3 MONTHS",
         "Quarter-point evaluation. Ceasefire or entrenchment. "
         "Final Phase 3 injection deployment."),
        ("Week 11 (May 28-Jun 3)", "STRUCTURAL",
         "Move from event-driven to structural positions. "
         "Long-dated gold, dollar shorts, real assets."),
        ("Week 12 (Jun 4-10)", "HARVEST",
         "Q2 data landing. Third war-month CPI. "
         "Systematic profit-taking on overvalued positions."),
        ("Week 13 (Jun 11-17)", "REGROUP",
         "Fed June meeting. Full 90-day evaluation. "
         "Plan next 90 days or transition to endgame."),
    ]

    for week_name, phase, description in weeks:
        lines.append(f"  {week_name}  --  [{phase}]")
        lines.append(f"    {description}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("=  END OF WAR ROOM BRIEFING -- UPDATE TWICE DAILY WITH LATEST INTEL         =")
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def render_day_detail(target: date) -> str:
    """Render detailed view for a specific day."""
    mandate = generate_daily_mandate(target)
    events = get_events_for_date(target)
    intel = get_intel_for_date(target)
    wd = _war_day(target)

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  WAR ROOM -- DAY DETAIL: {target.strftime('%A, %B %d, %Y')}")
    lines.append(f"  War Day: {wd}  |  Phase: {mandate.phase}  |  Risk: {mandate.risk_level}")
    lines.append("=" * 80)
    lines.append("")

    lines.append("  INDICATORS:")
    for k, v in mandate.key_indicators.items():
        lines.append(f"    {k:<12} {v:>12,.2f}")
    lines.append(f"    {'PRESSURE':<12} {mandate.pressure_score:>11.1f}%")
    lines.append("")

    if events:
        lines.append("  MACRO EVENTS:")
        for ev in events:
            lines.append(f"    [{ev.impact.value:<8}] {ev.name}")
            lines.append(f"              {ev.thesis_relevance}")
            for ai in ev.action_items:
                lines.append(f"              - {ai}")
        lines.append("")

    if mandate.correlation_alerts:
        lines.append("  CORRELATION ALERTS:")
        for a in mandate.correlation_alerts:
            lines.append(f"    >> {a}")
        lines.append("")

    lines.append("  ACCOUNT ACTIONS:")
    for acct, actions in mandate.account_actions.items():
        lines.append(f"    +-- {acct}")
        for action in actions:
            lines.append(f"    |   * {action}")
        lines.append(f"    +----------------------------")
    lines.append("")

    lines.append(f"  CAPITAL: Deploy ${mandate.capital_to_deploy:,.2f} today  |  "
                 f"Cumulative: ${mandate.cumulative_deployed:,.2f}")
    lines.append("")

    if mandate.stop_losses:
        lines.append("  STOP LOSSES / RISK:")
        for s in mandate.stop_losses:
            lines.append(f"    !! {s}")
    if mandate.hedge_adjustments:
        lines.append("  HEDGE ADJUSTMENTS:")
        for h in mandate.hedge_adjustments:
            lines.append(f"    [HEDGE] {h}")
    lines.append("")

    lines.append("  MORNING CHECKLIST:")
    for item in mandate.morning_checklist:
        lines.append(f"    [ ] {item}")
    lines.append("")
    lines.append("  EVENING CHECKLIST:")
    for item in mandate.evening_checklist:
        lines.append(f"    [ ] {item}")
    lines.append("")

    # Scenario projection for this day
    day_offset = (target - MODEL_TODAY).days
    if day_offset > 0:
        lines.append("  SCENARIO PROJECTIONS AT THIS DATE:")
        for sc in ["fails", "moderate", "major", "blackswan", "expected"]:
            proj = project_portfolio_value(day_offset, sc)
            lines.append(
                f"    {sc:<12}: ${proj['total_portfolio_usd']:>12,.2f}  "
                f"({proj['total_return_pct']:>+7.1f}%)  "
                f"Puts: {proj['existing_puts_mult']}"
            )
        lines.append("")

    if intel:
        lines.append("  INTEL UPDATES LOGGED:")
        for entry in intel:
            ts = entry["timestamp"].split("T")[1][:5]
            lines.append(f"    [{ts}] {entry['note']}")
        lines.append("")

    lines.append("  THESIS CONFIRMATION SIGNALS:")
    for s in mandate.thesis_confirmation_signals:
        lines.append(f"    [Y] {s}")
    lines.append("")
    lines.append("  THESIS DENIAL SIGNALS:")
    for s in mandate.thesis_denial_signals:
        lines.append(f"    [N] {s}")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def render_backtest_summary() -> str:
    """Render the 90-day historical model summary."""
    model = build_historical_model()
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  HISTORICAL MODEL: Dec 19 2025 -> Mar 19 2026 (90 days)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'DATE':<12} {'WD':>3} {'OIL':>7} {'GOLD':>8} {'VIX':>6} "
                 f"{'DXY':>6} {'BTC':>9} {'HY_SP':>6} {'SPY':>7} {'PRESS':>6} {'PHASE':<12}")
    lines.append("  " + "-" * 78)

    # Show weekly snapshots
    for i, day in enumerate(model):
        if i % 7 == 0 or day.notes:  # Weekly + notable days
            lines.append(
                f"  {day.date.isoformat():<12} {day.war_day:>3} "
                f"${day.oil_wti:>5.1f} ${day.gold_spot:>6.0f} "
                f"{day.vix:>5.1f} {day.dxy:>5.1f} "
                f"${day.btc_usd:>7.0f} {day.hy_spread_bps:>5} "
                f"${day.spy_price:>5.1f} {day.thesis_pressure_pct:>5.1f}% "
                f"{day.phase:<12}"
            )
            if day.notes and i % 7 != 0:
                lines.append(f"  {'':>12}     -> {day.notes[:65]}")

    lines.append("")
    lines.append("  KEY OBSERVATIONS:")
    lines.append("  - Pre-war (Dec-Feb): Slow pressure build, gold rising, BTC declining")
    lines.append("  - War shock (Feb 28): Oil +18%, VIX +56%, SPY -4.7% in 7 days")
    lines.append("  - Escalation (Mar 8-19): Dead cat bounce in equities, pressure still rising")
    lines.append("  - Hormuz event (Mar 12): Partial closure confirmed, oil +$5 intraday")
    lines.append("  - Current (Mar 19): 44% pressure, Phase 1 BUILDING, 8 puts deployed")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def render_forward_summary() -> str:
    """Render the 90-day forward projection summary."""
    model = build_forward_model()
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  FORWARD PROJECTION: Mar 20 -> Jun 17 2026 (90 days)")
    lines.append("  [Probability-weighted expected path]")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'DATE':<12} {'WD':>3} {'OIL':>7} {'GOLD':>8} {'VIX':>6} "
                 f"{'DXY':>6} {'BTC':>9} {'HY_SP':>6} {'SPY':>7} {'PRESS':>6} {'PHASE':<12}")
    lines.append("  " + "-" * 78)

    for i, day in enumerate(model):
        if i % 7 == 0 or day.notes:
            lines.append(
                f"  {day.date.isoformat():<12} {day.war_day:>3} "
                f"${day.oil_wti:>5.1f} ${day.gold_spot:>6.0f} "
                f"{day.vix:>5.1f} {day.dxy:>5.1f} "
                f"${day.btc_usd:>7.0f} {day.hy_spread_bps:>5} "
                f"${day.spy_price:>5.1f} {day.thesis_pressure_pct:>5.1f}% "
                f"{day.phase:<12}"
            )
            if day.notes:
                # Truncate long event strings
                note = day.notes[:70]
                lines.append(f"  {'':>12}     -> {note}")

    lines.append("")
    lines.append("  CRITICAL INFLECTION POINTS:")
    lines.append("  - Day 30 (Apr 1):  Month 2 starts. If no ceasefire -> thesis accelerating")
    lines.append("  - Day 45 (Apr 14): Checkpoint. Q1 earnings + war fatigue. Confirm or deny.")
    lines.append("  - Day 60 (May 1):  Fed May meeting. GDP Q1 data. Binary risk event.")
    lines.append("  - Day 80 (May 19): Endgame window opens. Transition to hard assets if >70%")
    lines.append("  - Day 90 (Jun 17): Model endpoint. Fed Jun meeting. Full evaluation.")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def render_portfolio() -> str:
    """Render detailed portfolio view."""
    port = get_total_portfolio()
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  PORTFOLIO  --  ALL ACCOUNTS & POSITIONS")
    lines.append("=" * 80)
    lines.append("")

    # Accounts
    lines.append("  ACCOUNTS:")
    total = 0
    for name, acct in ACCOUNTS.items():
        usd = acct.balance_usd
        total += usd
        currency_note = f" (~${usd:,.2f} USD)" if acct.currency == Currency.CAD else ""
        lines.append(
            f"    {acct.name:<25} ${acct.balance:>10,.2f} {acct.currency.value}"
            f"  Cash: ${acct.available_cash:,.2f}  In positions: ${acct.in_positions:,.2f}"
            f"{currency_note}"
        )
    lines.append(f"    {'-' * 60}")
    lines.append(f"    {'TOTAL (USD equiv)':<25} ${total:>10,.2f}")
    lines.append(f"    {'+ INJECTION':<25} ${CASH_INJECTION:>10,.2f}")
    lines.append(f"    {'= DEPLOYABLE':<25} ${total + CASH_INJECTION:>10,.2f}")
    lines.append("")

    # Positions
    lines.append("  OPEN POSITIONS (IBKR):")
    total_cost = 0
    total_value = 0
    for p in POSITIONS:
        cost = p.cost_basis
        value = p.current_price * p.quantity * 100
        pnl = p.unrealized_pnl
        total_cost += cost
        total_value += value
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0
        lines.append(
            f"    {p.symbol:<6} {p.direction:<10} {p.quantity}x "
            f"${p.strike:.0f} {p.expiry}  "
            f"Entry: ${p.entry_price:.2f}  Now: ${p.current_price:.2f}  "
            f"P&L: ${pnl:>+7.2f} ({pnl_pct:>+5.1f}%)"
        )
        lines.append(f"           [{p.thesis_vertical}] {p.notes}")
    lines.append(f"    {'-' * 60}")
    lines.append(f"    Cost Basis: ${total_cost:>8,.2f}  |  "
                 f"Market Value: ${total_value:>8,.2f}  |  "
                 f"Unrealized: ${total_value - total_cost:>+8,.2f}")
    lines.append("")

    # Vertical exposure
    lines.append("  VERTICAL EXPOSURE:")
    vertical_exposure: Dict[str, float] = {}
    for p in POSITIONS:
        v = p.thesis_vertical or "other"
        vertical_exposure[v] = vertical_exposure.get(v, 0) + p.cost_basis
    for v, cost in sorted(vertical_exposure.items(), key=lambda x: -x[1]):
        pct = cost / total_cost * 100 if total_cost > 0 else 0
        bar = "#" * int(pct / 2)
        lines.append(f"    {v:<12} ${cost:>7,.2f} ({pct:>5.1f}%) {bar}")
    lines.append("")

    # Gaps
    lines.append("  GAPS (verticals with ZERO exposure):")
    active = set(p.thesis_vertical for p in POSITIONS)
    all_verticals = ["oil", "gold", "dollar", "vol", "prediction", "hedge"]
    for v in all_verticals:
        if v not in active:
            lines.append(f"    !! {v.upper()} -- NO POSITIONS. Action required.")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def render_week_mandate(week_num: int) -> str:
    """Render all 7 daily mandates for a given week (1-13)."""
    if not 1 <= week_num <= 13:
        return "  Week must be 1-13"

    start = MODEL_TODAY + timedelta(days=(week_num - 1) * 7)
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  WEEK {week_num} MANDATE: {start.strftime('%b %d')} -> "
                 f"{(start + timedelta(days=6)).strftime('%b %d, %Y')}")
    lines.append("=" * 80)

    for i in range(7):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            lines.append(f"\n  {d.strftime('%a %b %d')}  --  [WEEKEND] Monitor geopolitical only.\n")
            continue

        mandate = generate_daily_mandate(d)
        lines.append(f"\n  +- {d.strftime('%A %b %d')} (WD{mandate.war_day}) "
                     f"[{mandate.phase}] [Risk: {mandate.risk_level}]")
        lines.append(f"  |  Pressure: {mandate.pressure_score}%  |  "
                     f"Deploy: ${mandate.capital_to_deploy:,.2f}")

        if mandate.events_today:
            lines.append(f"  |  EVENTS: {', '.join(mandate.events_today)}")

        for acct, actions in mandate.account_actions.items():
            for action in actions[:2]:  # Top 2 per account
                lines.append(f"  |  [{acct.split('(')[0].strip()[:10]}] {action}")

        lines.append(f"  +--------------------------------------------")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="90-Day War Room  --  AAC Supreme Command Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m strategies.ninety_day_war_room                    # Full dashboard\n"
            "  python -m strategies.ninety_day_war_room --day 2026-04-15   # Query specific day\n"
            "  python -m strategies.ninety_day_war_room --week 3           # Week 3 mandate\n"
            "  python -m strategies.ninety_day_war_room --backtest         # Historical model\n"
            "  python -m strategies.ninety_day_war_room --forward          # Forward projection\n"
            "  python -m strategies.ninety_day_war_room --portfolio        # Account details\n"
            "  python -m strategies.ninety_day_war_room --correlations     # Correlation matrix\n"
            "  python -m strategies.ninety_day_war_room --mandate          # Today's mandate\n"
            "  python -m strategies.ninety_day_war_room --calendar         # Event calendar\n"
            "  python -m strategies.ninety_day_war_room --update 'intel'   # Log intel\n"
            "  python -m strategies.ninety_day_war_room --range 2026-04-01 2026-04-14\n"
        ),
    )
    parser.add_argument("--day", "-d", type=str, help="Query specific date (YYYY-MM-DD)")
    parser.add_argument("--week", "-w", type=int, help="Show week N mandate (1-13)")
    parser.add_argument("--backtest", "-b", action="store_true", help="Historical 90-day model")
    parser.add_argument("--forward", "-f", action="store_true", help="Forward 90-day projection")
    parser.add_argument("--portfolio", "-p", action="store_true", help="Portfolio detail")
    parser.add_argument("--correlations", "-c", action="store_true", help="Correlation matrix")
    parser.add_argument("--mandate", "-m", action="store_true", help="Today's mandate only")
    parser.add_argument("--calendar", action="store_true", help="Full event calendar")
    parser.add_argument("--update", "-u", type=str, help="Log intel update")
    parser.add_argument("--range", "-r", nargs=2, type=str, help="Date range view (start end)")
    parser.add_argument("--scenario", type=str, default="expected",
                        help="Scenario for projections (fails, moderate, major, blackswan, expected)")
    parser.add_argument("--json", action="store_true", help="Output as JSON where applicable")
    args = parser.parse_args()

    # Handle specific commands
    if args.update:
        log_intel_update(args.update)
        return

    if args.day:
        target = date.fromisoformat(args.day)
        if args.json:
            mandate = generate_daily_mandate(target)
            print(json.dumps(asdict(mandate), indent=2, default=str))
        else:
            print(render_day_detail(target))
        return

    if args.week:
        print(render_week_mandate(args.week))
        return

    if args.backtest:
        print(render_backtest_summary())
        return

    if args.forward:
        print(render_forward_summary())
        return

    if args.portfolio:
        print(render_portfolio())
        return

    if args.correlations:
        print(render_correlation_matrix())
        return

    if args.mandate:
        mandate = generate_daily_mandate(MODEL_TODAY)
        if args.json:
            print(json.dumps(asdict(mandate), indent=2, default=str))
        else:
            print(render_day_detail(MODEL_TODAY))
        return

    if args.calendar:
        lines = [
            "",
            "=" * 80,
            "  MACRO EVENT CALENDAR  --  Full 180-Day Window",
            "=" * 80,
            "",
        ]
        for ev in sorted(MACRO_CALENDAR, key=lambda e: e.date):
            wd = _war_day(ev.date)
            past = " [PAST]" if ev.date < MODEL_TODAY else ""
            impact_tag = {"CRITICAL": "!!!!",  "HIGH": "!!! ",
                          "MEDIUM": "!!  ", "LOW": "!   ",
                          "THESIS": "****"}.get(ev.impact.value, "    ")
            lines.append(
                f"  {ev.date.strftime('%Y-%m-%d %a')} WD{wd:>3} "
                f"[{impact_tag}] {ev.name}{past}"
            )
            lines.append(f"    -> {ev.thesis_relevance[:75]}")
            for ai in ev.action_items[:3]:
                lines.append(f"    * {ai}")
            lines.append("")
        print("\n".join(lines))
        return

    if args.range:
        start = date.fromisoformat(args.range[0])
        end = date.fromisoformat(args.range[1])
        lines = [
            "",
            "=" * 80,
            f"  WAR ROOM  --  DATE RANGE: {start} -> {end}",
            "=" * 80,
        ]
        current = start
        while current <= end:
            if current.weekday() < 5:  # Weekdays only
                mandate = generate_daily_mandate(current)
                lines.append(
                    f"\n  {current.strftime('%a %b %d')} WD{mandate.war_day:>3} "
                    f"[{mandate.phase}] P:{mandate.pressure_score}% "
                    f"Risk:{mandate.risk_level}"
                )
                if mandate.events_today:
                    lines.append(f"    EVENTS: {', '.join(mandate.events_today)}")
                for acct, actions in mandate.account_actions.items():
                    if actions and actions[0] != "MONITOR  --  no action triggers met":
                        lines.append(f"    [{acct.split('(')[0].strip()[:12]}] {actions[0][:60]}")
            current += timedelta(days=1)
        lines.append("\n" + "=" * 80)
        print("\n".join(lines))
        return

    # Default: full dashboard
    print(render_dashboard())


if __name__ == "__main__":
    main()

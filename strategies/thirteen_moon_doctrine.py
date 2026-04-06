"""
13-Moon Doctrine Timeline — March 2026 to April 2027
======================================================
Master compounding calendar anchored to the March 3, 2026 Total Lunar
Eclipse. Integrates:

    1. Lunar cycles (13 synodic months, ~29.53 days each)
    2. Astrology overlays (eclipses, equinoxes, solstices, planetary ingresses)
    3. Golden-mean phi coherence markers (Planck-length scaled to timeline)
    4. Quarterly earnings / financial reporting events
    5. World news / geopolitical catalysts
    6. Doctrine action mandates per cycle

Phi Coherence Overlay
---------------------
Dan Winter PlanckPhire principle: successive phi powers applied to the
synodic lunar interval create fractal resonance nodes. Starting from the
March 3 eclipse anchor, each phi^n interval marks a phase-coherence peak
where volatility amplification is highest.

Usage
-----
    from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine
    doctrine = ThirteenMoonDoctrine()
    doctrine.print_timeline()
    doctrine.get_events_with_lead_time(days_ahead=14)
    doctrine.export_html("data/storyboard/thirteen_moon_storyboard.html")
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2  # 1.6180339887
SYNODIC_MONTH = 29.53         # Average synodic month in days
MOON_ZERO_DATE = date(2026, 3, 3)  # Total Lunar Eclipse — epoch anchor


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AstrologyEvent:
    date: date
    name: str
    category: str          # eclipse, equinox, solstice, ingress, conjunction, opposition, retrograde, station, aspect, full_moon, lunar
    impact: str            # HIGH, MEDIUM, LOW
    description: str = ""
    volatility_mult: float = 1.0
    zodiac_sign: str = ""  # Aries, Taurus, etc. — sign context for the event


@dataclass
class PhiCoherenceMarker:
    date: date
    phi_power: int         # n in phi^n
    phi_value: float       # actual phi^n value
    days_from_anchor: float
    label: str
    resonance_strength: float  # 0-1, decays with higher powers


@dataclass
class FinancialEvent:
    date: date
    name: str
    category: str          # earnings, fed, economic, ipo, policy
    companies: List[str] = field(default_factory=list)
    impact: str = "MEDIUM"
    description: str = ""


@dataclass
class WorldEvent:
    date: date
    name: str
    category: str          # geopolitical, trade, summit, deadline, military
    impact: str = "MEDIUM"
    description: str = ""


@dataclass
class DoctrineAction:
    mandate: str           # ACCUMULATE, DEPLOY, HOLD, ROTATE, REBALANCE, EXIT
    description: str
    conviction: float      # 0-1
    targets: List[str] = field(default_factory=list)  # SLV, XLE, BITO, etc.


@dataclass
class MoonCycle:
    moon_number: int               # 0-13
    start_date: date
    end_date: date
    lunar_phase_name: str          # Full Moon name (Pink Moon, Blue Moon, etc.)
    fire_peak_date: Optional[date] = None
    new_moon_date: Optional[date] = None
    astrology_events: List[AstrologyEvent] = field(default_factory=list)
    phi_markers: List[PhiCoherenceMarker] = field(default_factory=list)
    financial_events: List[FinancialEvent] = field(default_factory=list)
    world_events: List[WorldEvent] = field(default_factory=list)
    aac_events: List[AACEvent] = field(default_factory=list)
    doctrine_action: Optional[DoctrineAction] = None


@dataclass
class AACEvent:
    """AAC system-specific event (trades, war room, scenarios, milestones, etc.)."""
    date: date
    name: str
    layer: str             # trade, war_room, scenario, milestone, seesaw, strategy, options_lifecycle
    category: str          # More specific: live_trade, put_expiry, fomc, capital_phase, etc.
    impact: str = "MEDIUM"
    description: str = ""
    assets: List[str] = field(default_factory=list)
    conviction: float = 0.0
    thesis_relevance: str = ""


@dataclass
class ScheduledAlert:
    """An upcoming event with lead-time context."""
    event_date: date
    event_name: str
    event_type: str        # astrology, financial, world, phi, doctrine, aac
    moon_number: int
    days_until: int
    lead_time_action: str  # What to do N days before
    priority: str          # CRITICAL, HIGH, MEDIUM, LOW


# ═══════════════════════════════════════════════════════════════════════════
# TIMELINE DATA — 13 CYCLES FROM MARCH 3, 2026
# ═══════════════════════════════════════════════════════════════════════════

def _build_astrology_events() -> List[AstrologyEvent]:
    """All significant astrology events March 2026 - April 2027."""
    return [
        # Eclipses
        AstrologyEvent(date(2026, 3, 3), "Total Lunar Eclipse (Virgo)", "eclipse", "HIGH",
                       "Moon 0 anchor. South Node Virgo — karmic release, purification, perfectionism purge. "
                       "Conjunct South Node, sextile Jupiter in Cancer for emotional support. Maximum disruption signal.",
                       1.40, "Virgo"),
        AstrologyEvent(date(2026, 3, 29), "Partial Solar Eclipse", "eclipse", "MEDIUM",
                       "Solar-lunar tension. Aries influence — action/initiation energy. Validate positions.", 1.20, "Aries"),
        AstrologyEvent(date(2026, 8, 12), "Total Solar Eclipse", "eclipse", "HIGH",
                       "Near perigee. Leo themes — identity, power, creative expression. "
                       "New Leo-Aquarius eclipse axis begins (identity/power vs. collective innovation). Path of totality = sentiment shift.",
                       1.40, "Leo"),
        AstrologyEvent(date(2026, 8, 28), "Partial Lunar Eclipse", "eclipse", "MEDIUM",
                       "Pisces — dissolution, compassion. Follow-on lunar disruption from Aug 12 solar. "
                       "Virgo-Pisces eclipse axis closing.", 1.20, "Pisces"),
        AstrologyEvent(date(2027, 2, 6), "Annular Solar Eclipse", "eclipse", "MEDIUM",
                       "Ring of fire. Aquarius — collective innovation. Pre-spring reset.", 1.25, "Aquarius"),
        AstrologyEvent(date(2027, 2, 20), "Total Lunar Eclipse", "eclipse", "HIGH",
                       "Leo — expression, creativity. Same day as Saturn-Neptune conjunction at 0° Aries. "
                       "Historic convergence of eclipse + 36-year cycle reset.", 1.40, "Leo"),

        # Equinoxes & Solstices
        AstrologyEvent(date(2026, 3, 20), "Vernal Equinox — Sun enters Aries", "equinox", "HIGH",
                       "New astrological year. Cardinal fire — action, initiation, courage. "
                       "Spring equinox thesis renewal. Mars co-ruler activates.", 1.25, "Aries"),
        AstrologyEvent(date(2026, 6, 21), "Summer Solstice — Sun enters Cancer", "solstice", "HIGH",
                       "Emotional security, nurturing, home themes. Maximum solar energy. "
                       "Peak conviction window opens. Cancer = protective/defensive positioning.", 1.25, "Cancer"),
        AstrologyEvent(date(2026, 9, 22), "Autumnal Equinox — Sun enters Libra", "equinox", "HIGH",
                       "Balance, relationships, justice. Harvest cycle. Profit-taking window. "
                       "Cardinal air — diplomatic/trade resolutions or escalations.", 1.25, "Libra"),
        AstrologyEvent(date(2026, 12, 21), "Winter Solstice — Sun enters Capricorn", "solstice", "HIGH",
                       "Structure, ambition, authority. Shortest day — maximum introspection. "
                       "Year-end rebalance. Capricorn = institutional/systemic themes.", 1.25, "Capricorn"),
        AstrologyEvent(date(2027, 3, 20), "Vernal Equinox 2027 — Sun enters Aries", "equinox", "HIGH",
                       "Full-year review. $10M milestone assessment. New cycle begins.", 1.25, "Aries"),

        # ── RETROGRADES ──────────────────────────────────────────────────
        AstrologyEvent(date(2026, 2, 25), "Mercury Retrograde in Pisces Begins", "retrograde", "MEDIUM",
                       "Communication breakdowns, travel delays, contract confusion. Pisces = especially foggy. "
                       "Rough ride through Moon 0. Review, don't initiate. Ends March 20.", 1.15, "Pisces"),
        AstrologyEvent(date(2026, 3, 20), "Mercury Retrograde Ends (Stations Direct)", "station", "MEDIUM",
                       "Mercury stations direct in Pisces. Clarity returns on Spring Equinox — powerful alignment. "
                       "Green light for new initiatives.", 1.10, "Pisces"),
        AstrologyEvent(date(2026, 12, 13), "Jupiter Retrograde in Leo Begins", "retrograde", "MEDIUM",
                       "Creative expansion internalized. Re-evaluate risk appetite and position sizing. "
                       "Ends April 13, 2027. Optimism contracts — contrarian buy signal for conviction trades.", 1.10, "Leo"),

        # ── PLANETARY INGRESSES ──────────────────────────────────────────
        AstrologyEvent(date(2026, 3, 2), "Mars enters Pisces", "ingress", "MEDIUM",
                       "Intuitive action, dissolution of old conflicts. Mars in Pisces = action through surrender. "
                       "Not aggressive — spiritual warrior energy. Favors behind-the-scenes positioning.", 1.10, "Pisces"),
        AstrologyEvent(date(2026, 4, 25), "Uranus enters Gemini", "ingress", "HIGH",
                       "MAJOR: 7-year tech/comms/information revolution begins (until 2032-33). "
                       "Last time Uranus in Gemini: 1942-49 (WWII tech boom, computers, nuclear). "
                       "Expect: AI disruption, comms infrastructure upheaval, media fragmentation. "
                       "Ties to big-tech April earnings cracks (AI ROI skepticism).", 1.30, "Gemini"),
        AstrologyEvent(date(2026, 6, 29), "Jupiter enters Leo", "ingress", "HIGH",
                       "MAJOR: Creative expansion, leadership, visibility boost (until July 25, 2027). "
                       "Jupiter in Leo = bullish for 'shiny' things: gold, silver, entertainment, energy. "
                       "Risk appetite expands. Gap-up moves amplified at Moon 3/4. "
                       "Favors bold, high-conviction positions.", 1.25, "Leo"),
        AstrologyEvent(date(2027, 2, 13), "Saturn enters Aries", "ingress", "HIGH",
                       "MAJOR: 2.5-year cycle of new structures, discipline, boundaries in cardinal fire. "
                       "End of easy-money era reinforced. Rates trap + private credit gates harden. "
                       "Bearish for credit puts (OBDC/ARCC book prints harder). "
                       "New leadership structures emerge from old order collapse.", 1.30, "Aries"),

        # ── CONJUNCTIONS & ASPECTS ───────────────────────────────────────
        AstrologyEvent(date(2026, 3, 7), "Venus-Saturn Conjunction", "conjunction", "MEDIUM",
                       "Reality check on money, love, values. Saturn restricts Venus pleasures. "
                       "Financial sobriety. Good for disciplined entries, not impulse trades.", 1.10, "Pisces"),
        AstrologyEvent(date(2026, 3, 10), "Jupiter Stations Direct in Cancer", "station", "MEDIUM",
                       "Jupiter direct after months retrograde. Expansion resumes in emotional/protective sign. "
                       "Bullish for real estate, food commodities, family-oriented sectors. "
                       "Influence carries into Moon 1 deployment window.", 1.10, "Cancer"),
        AstrologyEvent(date(2026, 7, 7), "Jupiter-Neptune Square", "aspect", "MEDIUM",
                       "Speculation vs reality tension. Bubble/deflation signal. "
                       "Neptune dissolves Jupiter's optimism — watch for crypto/meme-stock traps. "
                       "Risk of over-leveraging on false signals.", 1.15, "Leo/Aries"),
        AstrologyEvent(date(2026, 8, 27), "Saturn Opposition", "opposition", "MEDIUM",
                       "Structural stress test. Old systems challenged. "
                       "Authority figures face scrutiny. Fed/institutional credibility questioned.", 1.15, ""),
        AstrologyEvent(date(2026, 11, 14), "Mars-Jupiter Alignment", "conjunction", "MEDIUM",
                       "Aggressive energy + expansion. War/trade escalation potential. "
                       "Bullish energy if channeled — dangerous if reckless.", 1.20, "Scorpio/Leo"),
        AstrologyEvent(date(2026, 11, 16), "Mars-Jupiter Exact Conjunction", "conjunction", "MEDIUM",
                       "Peak aggressive/expansive energy. Military/trade action window. "
                       "Mid-November volatility cluster with Supermoon.", 1.20, ""),
        AstrologyEvent(date(2027, 2, 20), "Saturn-Neptune Conjunction at 0° Aries", "conjunction", "HIGH",
                       "ONCE-IN-36-YEARS. The single most significant astrology marker in the entire doctrine. "
                       "0° Aries = absolute beginning of zodiac wheel — structural reset button. "
                       "Saturn (reality, restriction, discipline) + Neptune (dissolution, dreams, illusions). "
                       "Last conjunction 1989: Fall of Berlin Wall, end of Cold War, birth of modern fiat system. "
                       "Previous: 1953 (consumer credit boom), 1917 (Russian Revolution), 1881 (industrial trusts). "
                       "2027: Petrodollar death spiral visibility. Private credit crunch. De-dollarization acceleration. "
                       "Silver becomes bridge asset — Neptune's dream metal meets Saturn's real metal. "
                       "Japan carry unwind. Rate regime change. Era-defining dissolution of old financial structures + "
                       "new foundations built. This is the capstone of the entire 13-moon march.",
                       1.40, "Aries"),

        # ── NEPTUNE ──────────────────────────────────────────────────────
        AstrologyEvent(date(2026, 1, 26), "Neptune enters Aries", "ingress", "HIGH",
                       "Neptune's first Aries ingress since 1861. Inspired action, spiritual pioneering, "
                       "collective dreams channeled into new beginnings. Dissolves old institutional illusions. "
                       "Influence carries through entire doctrine timeline.", 1.20, "Aries"),

        # ── PLUTO ────────────────────────────────────────────────────────
        AstrologyEvent(date(2026, 9, 2), "Pluto Sextile Neptune", "aspect", "MEDIUM",
                       "Deep systemic transformation harmonizes with collective dreams. "
                       "Generational infrastructure change. Supports thesis of old order dissolution.", 1.10, "Aquarius/Aries"),

        # ── NODE SHIFTS ──────────────────────────────────────────────────
        AstrologyEvent(date(2026, 7, 1), "North Node shifts to Aquarius / South Node to Leo", "ingress", "HIGH",
                       "Eclipse axis migration: Virgo-Pisces closes, Leo-Aquarius opens. "
                       "Collective focus shifts from personal perfection to collective innovation. "
                       "New eclipse series in Leo-Aquarius begins Aug 12 solar eclipse. "
                       "Themes: individual identity vs. group consciousness, tech disruption.", 1.20, "Aquarius/Leo"),

        # ── FULL MOONS (FIRE PEAKS) ─────────────────────────────────────
        AstrologyEvent(date(2026, 4, 1), "Pink Moon (Full) in Libra", "full_moon", "MEDIUM",
                       "Fire Peak. Balance/relationships emphasis. Deploy capital. Gap-up risk window. "
                       "Empirical: Full moons = higher vol, lower avg equity returns (Yuan/Zheng/Zhu 2006, Lucey 2010). "
                       "Silver shows ~4× higher returns in new-moon vs full-moon reversals.", 1.15, "Libra"),
        AstrologyEvent(date(2026, 5, 1), "Flower Moon (Full) in Scorpio", "full_moon", "MEDIUM",
                       "Scorpio intensity — transformation, power, secrets revealed. "
                       "Blue Moon buildup. Deep-value entries.", 1.10, "Scorpio"),
        AstrologyEvent(date(2026, 5, 31), "Blue Moon Fire Peak in Sagittarius", "full_moon", "HIGH",
                       "Rare second full moon. Sagittarius — expansion, truth-seeking, risk-taking. "
                       "Highest-probability blowup window visibility. Fire Peak amplifier.", 1.25, "Sagittarius"),
        AstrologyEvent(date(2026, 6, 29), "Buck Moon (Full) in Capricorn", "full_moon", "MEDIUM",
                       "Capricorn — structure, ambition. Solstice amplifier. Post-solstice vol spike.", 1.15, "Capricorn"),
        AstrologyEvent(date(2026, 7, 29), "Sturgeon Moon (Full) in Aquarius", "full_moon", "MEDIUM",
                       "Aquarius — innovation, collective. Eclipse lead-in. Pre-position for Aug 12.", 1.10, "Aquarius"),
        AstrologyEvent(date(2026, 8, 27), "Corn Moon (Full) in Pisces", "full_moon", "MEDIUM",
                       "Pisces — dissolution, compassion. Post-eclipse assessment. Lunar eclipse region.", 1.10, "Pisces"),
        AstrologyEvent(date(2026, 9, 26), "Harvest Moon (Full) in Aries", "full_moon", "MEDIUM",
                       "Aries — action, initiation. Equinox amplifier. Profit harvest window.", 1.15, "Aries"),
        AstrologyEvent(date(2026, 10, 25), "Hunter Moon (Full) in Taurus", "full_moon", "MEDIUM",
                       "Taurus — stability, values, material security. Mid-Q3 volatility check.", 1.10, "Taurus"),
        AstrologyEvent(date(2026, 11, 24), "Beaver Supermoon (Full) in Gemini", "full_moon", "HIGH",
                       "SUPERMOON — closest approach. Gemini — communication, duality. "
                       "Maximum tidal/vol amplification. Mars-Jupiter conjunction amplifier.", 1.25, "Gemini"),
        AstrologyEvent(date(2026, 12, 24), "Cold Moon (Full) in Cancer", "full_moon", "MEDIUM",
                       "Cancer — home, security, emotional. Solstice aligned. Year-end positioning.", 1.15, "Cancer"),
        AstrologyEvent(date(2027, 1, 22), "Wolf Moon (Full) in Leo", "full_moon", "MEDIUM",
                       "Leo — expression, creativity, leadership. New year opening.", 1.10, "Leo"),
        AstrologyEvent(date(2027, 2, 20), "Snow Moon (Full) in Virgo", "full_moon", "HIGH",
                       "Virgo — discernment, purification. SAME DAY as Saturn-Neptune conjunction. "
                       "Full circle from Moon 0 Virgo eclipse. Historic convergence.", 1.35, "Virgo"),
        AstrologyEvent(date(2027, 3, 22), "Worm Moon (Full) in Libra", "full_moon", "MEDIUM",
                       "Libra — balance. Equinox amplifier. Final doctrine review cycle.", 1.15, "Libra"),
    ]


def _build_phi_coherence_markers() -> List[PhiCoherenceMarker]:
    """Golden-mean scaled coherence points from March 3 anchor.

    Each phi^n interval (measured in days from the synodic month base)
    marks a fractal resonance node. Lower powers = stronger resonance.

    The timeline spans phi^0 through phi^13, each representing a
    progressively higher harmonic that resonates with the base lunar
    cycle through the golden ratio.
    """
    markers = []
    base_interval = SYNODIC_MONTH  # ~29.53 days

    for n in range(14):  # phi^0 through phi^13
        phi_n = PHI ** n
        days_from_anchor = base_interval * phi_n
        marker_date = MOON_ZERO_DATE + timedelta(days=int(days_from_anchor))

        # Resonance decays with higher harmonics
        resonance = math.exp(-0.15 * n)

        # Date range (3-5 day window around the exact point)
        window_start = marker_date - timedelta(days=2)
        window_end = marker_date + timedelta(days=2)

        markers.append(PhiCoherenceMarker(
            date=marker_date,
            phi_power=n,
            phi_value=round(phi_n, 4),
            days_from_anchor=round(days_from_anchor, 1),
            label=f"phi^{n} resonance ({window_start.strftime('%b %d')}-{window_end.strftime('%b %d')})",
            resonance_strength=round(resonance, 3),
        ))

    return markers


def _build_financial_events() -> List[FinancialEvent]:
    """Major financial events March 2026 - April 2027."""
    return [
        # Q4 2025 wrap-up
        FinancialEvent(date(2026, 3, 5), "Q4 2025 Final Reports Wrap", "earnings", [], "MEDIUM",
                       "Last stragglers of Q4 2025 reporting season."),
        # Q1 2026 Earnings Season (April-May)
        FinancialEvent(date(2026, 4, 14), "Q1 Banks Earnings Start", "earnings",
                       ["JPM", "WFC", "C", "GS", "MS"], "HIGH",
                       "Major banks kick off Q1 2026 earnings."),
        FinancialEvent(date(2026, 4, 22), "Tesla Q1 Earnings", "earnings",
                       ["TSLA"], "HIGH", "EV bellwether. Vol catalyst."),
        FinancialEvent(date(2026, 4, 24), "Amazon Q1 Earnings", "earnings",
                       ["AMZN"], "HIGH", "Consumer + cloud barometer."),
        FinancialEvent(date(2026, 4, 28), "Microsoft Q1 Earnings", "earnings",
                       ["MSFT"], "HIGH", "Enterprise + AI capex indicator."),
        FinancialEvent(date(2026, 4, 29), "Meta Q1 Earnings", "earnings",
                       ["META"], "HIGH", "Ad spend + AI infrastructure."),
        FinancialEvent(date(2026, 4, 30), "Apple Q1 Earnings", "earnings",
                       ["AAPL"], "HIGH", "Consumer hardware + services."),
        FinancialEvent(date(2026, 5, 1), "Google Q1 Earnings", "earnings",
                       ["GOOGL"], "HIGH", "Search + cloud + AI."),
        # Fed Meetings 2026
        FinancialEvent(date(2026, 3, 18), "FOMC March Meeting", "fed", ["FED"], "HIGH",
                       "Rate decision + dot plot. Crisis-era forward guidance."),
        FinancialEvent(date(2026, 5, 6), "FOMC May Meeting", "fed", ["FED"], "HIGH",
                       "Post-Q1 earnings assessment."),
        FinancialEvent(date(2026, 6, 17), "FOMC June Meeting", "fed", ["FED"], "HIGH",
                       "Mid-year. SEP update. Dot plot refresh."),
        FinancialEvent(date(2026, 7, 29), "FOMC July Meeting", "fed", ["FED"], "HIGH",
                       "Summer assessment. QT/rate pivot signals."),
        FinancialEvent(date(2026, 9, 16), "FOMC September Meeting", "fed", ["FED"], "HIGH",
                       "Election proximity. Policy pivot window."),
        FinancialEvent(date(2026, 11, 4), "FOMC November Meeting", "fed", ["FED"], "HIGH",
                       "Post-election. Lame duck Fed."),
        FinancialEvent(date(2026, 12, 16), "FOMC December Meeting", "fed", ["FED"], "HIGH",
                       "Year-end. Full projection update."),
        # Q2 2026 Earnings Season (July-August)
        FinancialEvent(date(2026, 7, 14), "Q2 Banks Earnings Start", "earnings",
                       ["JPM", "WFC", "C", "GS"], "HIGH",
                       "Q2 bank earnings — credit quality in focus."),
        FinancialEvent(date(2026, 7, 22), "Big Tech Q2 Earnings Week", "earnings",
                       ["TSLA", "AMZN", "MSFT", "META", "AAPL", "GOOGL"], "HIGH",
                       "Mega-cap tech earnings cluster."),
        # Q3 2026 Earnings Season (October)
        FinancialEvent(date(2026, 10, 13), "Q3 Banks Earnings Start", "earnings",
                       ["JPM", "WFC", "C", "GS"], "HIGH",
                       "Q3 bank earnings — recession signals."),
        FinancialEvent(date(2026, 10, 22), "Big Tech Q3 Earnings Week", "earnings",
                       ["TSLA", "AMZN", "MSFT", "META", "AAPL", "GOOGL"], "HIGH",
                       "Mega-cap Q3 cluster."),
        # Q4 2026 Earnings (January 2027)
        FinancialEvent(date(2027, 1, 13), "Q4 Banks Earnings Start", "earnings",
                       ["JPM", "WFC", "C", "GS"], "HIGH",
                       "Q4 2026 bank earnings."),
        FinancialEvent(date(2027, 1, 22), "Big Tech Q4 Earnings Week", "earnings",
                       ["TSLA", "AMZN", "MSFT", "META", "AAPL", "GOOGL"], "HIGH",
                       "Year-end mega-cap results."),
        # Economic Data Points
        FinancialEvent(date(2026, 3, 28), "Q4 2025 GDP Final", "economic", [], "MEDIUM",
                       "Final GDP revision."),
        FinancialEvent(date(2026, 4, 10), "March CPI Report", "economic", [], "HIGH",
                       "Inflation data — rate path implications."),
        FinancialEvent(date(2026, 6, 26), "Q1 2026 GDP Final", "economic", [], "HIGH",
                       "First half growth assessment."),
        FinancialEvent(date(2026, 9, 25), "Q2 2026 GDP Final", "economic", [], "HIGH",
                       "Mid-year growth. Recession indicator."),
        FinancialEvent(date(2026, 12, 23), "Q3 2026 GDP Final", "economic", [], "MEDIUM",
                       "Year-end growth data."),
        FinancialEvent(date(2027, 3, 26), "Q4 2026 GDP Final", "economic", [], "HIGH",
                       "Full-year 2026 economic assessment."),
        # Major IPOs / Events
        FinancialEvent(date(2026, 4, 1), "April Tariff Deadline", "policy", [], "HIGH",
                       "Trade policy deadline. Market-moving."),
        FinancialEvent(date(2026, 4, 15), "Tax Day (US)", "policy", [], "LOW",
                       "Capital flows / selling pressure."),
        FinancialEvent(date(2026, 6, 19), "Quad Witching (June)", "options", [], "HIGH",
                       "Options/futures expiration. Max vol."),
        FinancialEvent(date(2026, 9, 18), "Quad Witching (September)", "options", [], "HIGH",
                       "Options/futures expiration."),
        FinancialEvent(date(2026, 12, 18), "Quad Witching (December)", "options", [], "HIGH",
                       "Year-end options expiration."),
        FinancialEvent(date(2027, 3, 19), "Quad Witching (March 2027)", "options", [], "HIGH",
                       "Q1 2027 options/futures expiration. Post-Saturn-Neptune."),

        # ── APRIL 17 OPTIONS EXPIRY (IBKR PUTS) ─────────────────────────
        FinancialEvent(date(2026, 4, 17), "April 17 Options Expiry (IBKR Puts)", "options",
                       ["ARCC", "PFF", "MAIN", "JNK"], "CRITICAL",
                       "ARCC $17P, PFF $29P, MAIN $50P, JNK $92P all expire. PFF let expire (worthless). "
                       "ARCC hold. MAIN roll to May/Jun. JNK hold/roll — closest to ITM."),

        # ── MONTHLY CPI RELEASES ─────────────────────────────────────────
        FinancialEvent(date(2026, 5, 13), "April CPI Report", "economic", [], "HIGH",
                       "Oil-embedded inflation first print. Hormuz impact visible."),
        FinancialEvent(date(2026, 6, 11), "May CPI Report", "economic", [], "HIGH",
                       "Inflation trajectory — rates trap confirmation."),
        FinancialEvent(date(2026, 7, 15), "June CPI Report", "economic", [], "HIGH",
                       "Mid-year inflation. Solstice-aligned."),
        FinancialEvent(date(2026, 8, 12), "July CPI Report", "economic", [], "HIGH",
                       "Eclipse-day CPI release. Maximum vol catalyst."),
        FinancialEvent(date(2026, 9, 10), "August CPI Report", "economic", [], "HIGH",
                       "Post-eclipse inflation assessment."),
        FinancialEvent(date(2026, 10, 14), "September CPI Report", "economic", [], "HIGH",
                       "Q3 inflation trajectory. Election proximity."),
        FinancialEvent(date(2026, 11, 12), "October CPI Report", "economic", [], "HIGH",
                       "Post-election inflation. Policy repricing."),
        FinancialEvent(date(2026, 12, 10), "November CPI Report", "economic", [], "MEDIUM",
                       "Year-end inflation. Fed December input."),
        FinancialEvent(date(2027, 1, 14), "December CPI Report", "economic", [], "HIGH",
                       "Full-year 2026 inflation assessment."),
        FinancialEvent(date(2027, 2, 12), "January 2027 CPI Report", "economic", [], "HIGH",
                       "New year inflation baseline. Saturn-Neptune approach."),
        FinancialEvent(date(2027, 3, 12), "February 2027 CPI Report", "economic", [], "HIGH",
                       "Post-conjunction inflation. New regime data."),

        # ── MONTHLY NON-FARM PAYROLLS ─────────────────────────────────────
        FinancialEvent(date(2026, 4, 3), "March NFP Report", "economic", [], "HIGH",
                       "Jobs data. Recession signal or resilience."),
        FinancialEvent(date(2026, 5, 1), "April NFP Report", "economic", [], "HIGH",
                       "Post-tariff employment impact."),
        FinancialEvent(date(2026, 6, 5), "May NFP Report", "economic", [], "HIGH",
                       "Labor market under oil shock stress."),
        FinancialEvent(date(2026, 7, 2), "June NFP Report", "economic", [], "HIGH",
                       "Mid-year employment. Solstice context."),
        FinancialEvent(date(2026, 8, 7), "July NFP Report", "economic", [], "HIGH",
                       "Summer jobs. Pre-eclipse."),
        FinancialEvent(date(2026, 9, 4), "August NFP Report", "economic", [], "HIGH",
                       "Post-eclipse labor data."),
        FinancialEvent(date(2026, 10, 2), "September NFP Report", "economic", [], "HIGH",
                       "Q3 employment. Election proximity."),
        FinancialEvent(date(2026, 11, 6), "October NFP Report", "economic", [], "HIGH",
                       "Pre-election employment."),
        FinancialEvent(date(2026, 12, 4), "November NFP Report", "economic", [], "HIGH",
                       "Post-election jobs. Fed December input."),
        FinancialEvent(date(2027, 1, 9), "December NFP Report", "economic", [], "HIGH",
                       "Year-end employment. Recession verdict."),
        FinancialEvent(date(2027, 2, 5), "January 2027 NFP Report", "economic", [], "HIGH",
                       "New year jobs baseline."),
        FinancialEvent(date(2027, 3, 5), "February 2027 NFP Report", "economic", [], "HIGH",
                       "Post-conjunction labor market."),

        # ── PCE INFLATION (Fed's Preferred Measure) ──────────────────────
        FinancialEvent(date(2026, 4, 30), "March PCE Inflation", "economic", [], "HIGH",
                       "Fed's preferred inflation gauge. Core PCE critical for rate path."),
        FinancialEvent(date(2026, 5, 29), "April PCE Inflation", "economic", [], "HIGH",
                       "Post-tariff PCE. Oil shock feeding through."),
        FinancialEvent(date(2026, 6, 26), "May PCE Inflation", "economic", [], "HIGH",
                       "Blue Moon window PCE. Rates trap data."),
        FinancialEvent(date(2026, 7, 31), "June PCE Inflation", "economic", [], "HIGH",
                       "Mid-year PCE. Solstice aftermath."),
        FinancialEvent(date(2026, 8, 28), "July PCE Inflation", "economic", [], "HIGH",
                       "Eclipse-week PCE. Post-Jackson Hole."),
        FinancialEvent(date(2026, 9, 25), "August PCE Inflation", "economic", [], "HIGH",
                       "Equinox PCE. Harvest Moon data."),
        FinancialEvent(date(2026, 10, 30), "September PCE Inflation", "economic", [], "HIGH",
                       "Q3 PCE. Election proximity."),
        FinancialEvent(date(2026, 11, 25), "October PCE Inflation", "economic", [], "MEDIUM",
                       "Post-election PCE."),
        FinancialEvent(date(2026, 12, 23), "November PCE Inflation", "economic", [], "MEDIUM",
                       "Year-end PCE. Solstice-aligned."),

        # ── ISM PMI (Manufacturing Pulse) ────────────────────────────────
        FinancialEvent(date(2026, 4, 1), "ISM Manufacturing PMI (March)", "economic", [], "HIGH",
                       "Below 50 = contraction. Oil/supply-chain impact."),
        FinancialEvent(date(2026, 5, 1), "ISM Manufacturing PMI (April)", "economic", [], "HIGH",
                       "Tariff + oil-shock impact on manufacturing."),
        FinancialEvent(date(2026, 6, 1), "ISM Manufacturing PMI (May)", "economic", [], "MEDIUM",
                       "Supply chain disruption assessment."),
        FinancialEvent(date(2026, 7, 1), "ISM Manufacturing PMI (June)", "economic", [], "HIGH",
                       "Mid-year manufacturing health."),
        FinancialEvent(date(2026, 9, 1), "ISM Manufacturing PMI (August)", "economic", [], "HIGH",
                       "Post-eclipse manufacturing data."),
        FinancialEvent(date(2026, 10, 1), "ISM Manufacturing PMI (September)", "economic", [], "HIGH",
                       "Q3 contraction check."),
        FinancialEvent(date(2026, 12, 1), "ISM Manufacturing PMI (November)", "economic", [], "MEDIUM",
                       "Year-end manufacturing."),
        FinancialEvent(date(2027, 1, 5), "ISM Manufacturing PMI (December)", "economic", [], "HIGH",
                       "Full-year manufacturing verdict."),
        FinancialEvent(date(2027, 3, 2), "ISM Manufacturing PMI (February)", "economic", [], "HIGH",
                       "Post-conjunction manufacturing."),

        # ── NVIDIA EARNINGS (AI Bellwether) ──────────────────────────────
        FinancialEvent(date(2026, 5, 28), "NVIDIA Q1 FY2027 Earnings", "earnings",
                       ["NVDA"], "CRITICAL",
                       "AI capex reality check. $50B+ quarterly revenue expectations. "
                       "Uranus-in-Gemini tech disruption overlay."),
        FinancialEvent(date(2026, 8, 27), "NVIDIA Q2 FY2027 Earnings", "earnings",
                       ["NVDA"], "CRITICAL",
                       "Eclipse-week NVIDIA. Jackson Hole same day. Maximum vol."),
        FinancialEvent(date(2026, 11, 19), "NVIDIA Q3 FY2027 Earnings", "earnings",
                       ["NVDA"], "HIGH",
                       "Pre-Supermoon NVIDIA. AI spending trajectory."),
        FinancialEvent(date(2027, 2, 26), "NVIDIA Q4 FY2027 Earnings", "earnings",
                       ["NVDA"], "HIGH",
                       "Post-conjunction NVIDIA. New regime spending."),

        # ── BDC EARNINGS (Private Credit Thesis) ─────────────────────────
        FinancialEvent(date(2026, 5, 1), "ARCC Q1 Earnings", "earnings",
                       ["ARCC"], "HIGH",
                       "Private credit health. NAV discount, non-accruals, dividend coverage."),
        FinancialEvent(date(2026, 5, 7), "MAIN Q1 Earnings", "earnings",
                       ["MAIN"], "HIGH",
                       "BDC supplemental dividend. Internal management advantage."),
        FinancialEvent(date(2026, 5, 8), "OBDC Q1 Earnings", "earnings",
                       ["OBDC"], "HIGH",
                       "Blue Owl BDC. NAV compression + gate risk. Our OWL put anchor."),
        FinancialEvent(date(2026, 8, 6), "ARCC Q2 Earnings", "earnings",
                       ["ARCC"], "HIGH",
                       "Mid-year private credit. Recession stress visible."),
        FinancialEvent(date(2026, 8, 7), "MAIN Q2 Earnings", "earnings",
                       ["MAIN"], "HIGH",
                       "Summer BDC health check."),
        FinancialEvent(date(2026, 11, 4), "ARCC Q3 Earnings", "earnings",
                       ["ARCC"], "HIGH",
                       "Post-election BDC stress."),
        FinancialEvent(date(2026, 11, 6), "MAIN Q3 Earnings", "earnings",
                       ["MAIN"], "HIGH",
                       "Q3 BDC. Saturn Aries approach."),
        FinancialEvent(date(2027, 2, 12), "ARCC Q4 Earnings", "earnings",
                       ["ARCC"], "HIGH",
                       "Full-year BDC. Conjunction proximity."),

        # ── REGIONAL BANK EARNINGS (KRE Thesis) ──────────────────────────
        FinancialEvent(date(2026, 4, 16), "Regional Banks Q1 Earnings", "earnings",
                       ["KEY", "HBAN", "CFG", "ZION", "FHN"], "HIGH",
                       "CRE exposure, deposit flight, NIM compression. KRE put thesis."),
        FinancialEvent(date(2026, 7, 15), "Regional Banks Q2 Earnings", "earnings",
                       ["KEY", "HBAN", "CFG", "ZION", "FHN"], "HIGH",
                       "Mid-year regional bank health. CRE writedowns."),
        FinancialEvent(date(2026, 10, 15), "Regional Banks Q3 Earnings", "earnings",
                       ["KEY", "HBAN", "CFG", "ZION", "FHN"], "HIGH",
                       "Q3 regional banks. Election uncertainty stress."),
        FinancialEvent(date(2027, 1, 15), "Regional Banks Q4 Earnings", "earnings",
                       ["KEY", "HBAN", "CFG", "ZION", "FHN"], "HIGH",
                       "Full-year regional bank assessment."),

        # ── ECB RATE DECISIONS ───────────────────────────────────────────
        FinancialEvent(date(2026, 4, 17), "ECB Rate Decision (April)", "fed",
                       ["ECB"], "HIGH",
                       "European rate path. EUR/USD impact on commodities."),
        FinancialEvent(date(2026, 6, 5), "ECB Rate Decision (June)", "fed",
                       ["ECB"], "HIGH",
                       "Mid-year European monetary policy."),
        FinancialEvent(date(2026, 9, 11), "ECB Rate Decision (September)", "fed",
                       ["ECB"], "HIGH",
                       "Post-summer European assessment."),
        FinancialEvent(date(2026, 12, 18), "ECB Rate Decision (December)", "fed",
                       ["ECB"], "HIGH",
                       "Year-end European rates. Divergence from Fed."),

        # ── BOJ RATE DECISIONS (Yen Carry Unwind Thesis) ─────────────────
        FinancialEvent(date(2026, 4, 24), "BOJ Rate Decision (April)", "fed",
                       ["BOJ"], "HIGH",
                       "Japan rate normalization. Yen carry unwind risk. "
                       "USD/JPY key level for Treasury selling thesis."),
        FinancialEvent(date(2026, 7, 31), "BOJ Rate Decision (July)", "fed",
                       ["BOJ"], "HIGH",
                       "Summer BOJ. YCC adjustment potential. Carry unwind catalyst."),
        FinancialEvent(date(2026, 10, 31), "BOJ Rate Decision (October)", "fed",
                       ["BOJ"], "HIGH",
                       "Q4 BOJ. Yen intervention risk."),
        FinancialEvent(date(2027, 1, 29), "BOJ Rate Decision (January)", "fed",
                       ["BOJ"], "HIGH",
                       "New year BOJ. Saturn-Neptune yen dissolution thesis."),

        # ── TREASURY REFUNDING ANNOUNCEMENTS ─────────────────────────────
        FinancialEvent(date(2026, 5, 4), "Treasury Quarterly Refunding (May)", "policy",
                       [], "HIGH",
                       "Issuance size. Duration mix. Demand signals. Interest expense trajectory."),
        FinancialEvent(date(2026, 8, 3), "Treasury Quarterly Refunding (August)", "policy",
                       [], "HIGH",
                       "Summer refunding. Debt ceiling proximity assessment."),
        FinancialEvent(date(2026, 11, 2), "Treasury Quarterly Refunding (November)", "policy",
                       [], "HIGH",
                       "Post-election refunding. Fiscal policy repricing."),
        FinancialEvent(date(2027, 2, 3), "Treasury Quarterly Refunding (February)", "policy",
                       [], "HIGH",
                       "Conjunction-era refunding. Structural debt assessment."),

        # ── FOMC MINUTES RELEASES ────────────────────────────────────────
        FinancialEvent(date(2026, 4, 8), "FOMC March Minutes Released", "fed",
                       ["FED"], "MEDIUM",
                       "Detailed discussion behind March decision. Rate path clues."),
        FinancialEvent(date(2026, 5, 27), "FOMC May Minutes Released", "fed",
                       ["FED"], "MEDIUM",
                       "Post-Q1 earnings assessment details."),
        FinancialEvent(date(2026, 7, 8), "FOMC June Minutes Released", "fed",
                       ["FED"], "MEDIUM",
                       "SEP details. Dot plot discussion nuances."),
        FinancialEvent(date(2026, 8, 19), "FOMC July Minutes Released", "fed",
                       ["FED"], "MEDIUM",
                       "Summer policy deliberation details."),
        FinancialEvent(date(2026, 10, 7), "FOMC September Minutes Released", "fed",
                       ["FED"], "MEDIUM",
                       "Election-proximity policy discussion."),
        FinancialEvent(date(2026, 11, 25), "FOMC November Minutes Released", "fed",
                       ["FED"], "MEDIUM",
                       "Post-election Fed deliberation."),
        FinancialEvent(date(2027, 1, 7), "FOMC December Minutes Released", "fed",
                       ["FED"], "MEDIUM",
                       "Year-end projection discussion."),

        # ── MONTHLY OPTIONS EXPIRY (OPEX) ────────────────────────────────
        FinancialEvent(date(2026, 5, 15), "May Monthly OPEX", "options", [], "MEDIUM",
                       "Monthly options expiry. Gamma/delta hedging flows."),
        FinancialEvent(date(2026, 7, 17), "July Monthly OPEX", "options", [], "MEDIUM",
                       "Post-Q2 earnings OPEX."),
        FinancialEvent(date(2026, 8, 21), "August Monthly OPEX", "options", [], "MEDIUM",
                       "Post-eclipse OPEX. Vol reset."),
        FinancialEvent(date(2026, 10, 16), "October Monthly OPEX", "options", [], "MEDIUM",
                       "Pre-election OPEX."),
        FinancialEvent(date(2026, 11, 20), "November Monthly OPEX", "options", [], "MEDIUM",
                       "Post-election OPEX. Position unwinding."),
        FinancialEvent(date(2027, 1, 16), "January 2027 Monthly OPEX", "options", [], "MEDIUM",
                       "New year OPEX."),
        FinancialEvent(date(2027, 2, 19), "February 2027 Monthly OPEX", "options", [], "MEDIUM",
                       "Pre-conjunction OPEX."),
    ]


def _build_world_events() -> List[WorldEvent]:
    """Major world news / geopolitical events March 2026 - April 2027."""
    return [
        # Hormuz / Iran
        WorldEvent(date(2026, 4, 6), "Iran Hormuz Deadline Extension", "geopolitical", "HIGH",
                   "Dual chokepoint (Hormuz + Russian Baltic) deadline. Oil shock catalyst."),
        WorldEvent(date(2026, 3, 15), "OPEC+ Output Decision", "trade", "HIGH",
                   "Production quota decision. Oil supply/demand rebalance."),
        # IMF / World Bank
        WorldEvent(date(2026, 4, 13), "IMF/World Bank Spring Meetings Start", "summit", "HIGH",
                   "Global economic assessment. SDR allocation signals."),
        WorldEvent(date(2026, 4, 19), "IMF/World Bank Spring Meetings End", "summit", "MEDIUM",
                   "Closing communique. Policy prescriptions."),
        WorldEvent(date(2026, 10, 12), "IMF/World Bank Annual Meetings", "summit", "HIGH",
                   "Fall assessment. Global growth outlook."),
        # BRICS
        WorldEvent(date(2026, 5, 15), "BRICS Finance Ministers Meeting", "geopolitical", "MEDIUM",
                   "De-dollarization acceleration signals."),
        WorldEvent(date(2026, 8, 20), "BRICS Summit Prep (India)", "geopolitical", "MEDIUM",
                   "India presidency. Expansion + currency announcements."),
        WorldEvent(date(2026, 10, 5), "BRICS Summit (India)", "geopolitical", "HIGH",
                   "Annual summit. New members. Payment system updates."),
        # US Politics
        WorldEvent(date(2026, 11, 3), "US Midterm Elections", "geopolitical", "HIGH",
                   "Congressional control shifts. Policy uncertainty peak."),
        WorldEvent(date(2026, 11, 10), "US Midterm Fallout Week", "geopolitical", "MEDIUM",
                   "Market digestion of results. Policy repricing."),
        # Trade
        WorldEvent(date(2026, 6, 12), "OPEC+ Mid-Year Review", "trade", "HIGH",
                   "Summer production assessment. Hormuz context."),
        WorldEvent(date(2026, 10, 18), "WTO Ministerial Conference", "trade", "MEDIUM",
                   "Trade rules. Tariff escalation/de-escalation."),
        # Fed / Macro
        WorldEvent(date(2026, 8, 27), "Jackson Hole Symposium", "economic", "HIGH",
                   "Fed Chair keynote. Policy signal. Markets hang on every word."),
        # China / Taiwan
        WorldEvent(date(2026, 5, 20), "Taiwan Inauguration Anniversary", "geopolitical", "MEDIUM",
                   "Cross-strait tension assessment window."),
        WorldEvent(date(2026, 10, 1), "China National Day", "geopolitical", "MEDIUM",
                   "Military parade. Taiwan rhetoric."),
        # Energy
        WorldEvent(date(2026, 7, 1), "EU Energy Policy Review", "trade", "MEDIUM",
                   "European energy transition update. LNG/renewables."),
        WorldEvent(date(2026, 12, 1), "COP31 Climate Summit", "summit", "MEDIUM",
                   "Climate commitments. ESG/energy sector impact."),
        # Crypto/Digital
        WorldEvent(date(2026, 4, 14), "Bitcoin Halving +2yr Anniversary", "economic", "MEDIUM",
                   "Cycle analysis. Post-halving accumulation phase."),
        WorldEvent(date(2027, 1, 20), "US Presidential Inauguration", "geopolitical", "HIGH",
                   "New administration. Day-1 executive orders."),
        WorldEvent(date(2027, 3, 1), "BRICS Currency System Target", "geopolitical", "HIGH",
                   "Target date for BRICS alternative payment system launch."),

        # ── MAJOR SUMMITS ────────────────────────────────────────────────
        WorldEvent(date(2026, 1, 19), "World Economic Forum (Davos) Start", "summit", "HIGH",
                   "Global elite gathering. Narrative setting for 2026. ESG, AI, trade themes."),
        WorldEvent(date(2026, 6, 23), "G7 Summit (Canada)", "summit", "HIGH",
                   "G7 leaders. Trade war, sanctions, energy security. Dollar-system defense."),
        WorldEvent(date(2026, 7, 9), "NATO Summit", "summit", "HIGH",
                   "Defense spending, Ukraine status, Indo-Pacific. Military-industrial catalyst."),
        WorldEvent(date(2026, 9, 15), "UN General Assembly (UNGA) Opens", "summit", "HIGH",
                   "World leaders converge NYC. Speeches = geopolitical signal density peak."),
        WorldEvent(date(2026, 11, 15), "G20 Leaders Summit (South Africa)", "summit", "HIGH",
                   "BRICS-host G20. De-dollarization narratives + Global South agenda."),
        WorldEvent(date(2026, 11, 20), "APEC Leaders Summit", "summit", "MEDIUM",
                   "Asia-Pacific trade. Pacific Rim supply chain signals."),
        WorldEvent(date(2027, 1, 18), "World Economic Forum 2027 (Davos)", "summit", "HIGH",
                   "Post-inauguration Davos. Saturn-Neptune approaching. New world order narratives."),

        # ── OPEC / ENERGY EXPANDED ───────────────────────────────────────
        WorldEvent(date(2026, 5, 25), "OPEC+ Meeting (May)", "trade", "HIGH",
                   "Production adjustment post-Hormuz disruption. Supply weaponization."),
        WorldEvent(date(2026, 8, 1), "OPEC+ Meeting (August)", "trade", "HIGH",
                   "Summer oil supply review. Eclipse-week energy context."),
        WorldEvent(date(2026, 9, 5), "OPEC+ Meeting (September)", "trade", "HIGH",
                   "Q4 production targets. Winter supply planning."),
        WorldEvent(date(2026, 11, 26), "OPEC+ Meeting (November)", "trade", "HIGH",
                   "Post-election OPEC. US energy policy direction."),
        WorldEvent(date(2027, 3, 10), "OPEC+ Meeting (March 2027)", "trade", "MEDIUM",
                   "New administration OPEC dynamics."),

        # ── IRAN / NUCLEAR ───────────────────────────────────────────────
        WorldEvent(date(2026, 4, 20), "Iran Nuclear Talks Resumption Window", "geopolitical", "HIGH",
                   "Potential diplomacy window. De-escalation or acceleration. Oil $120+ bear case."),
        WorldEvent(date(2026, 6, 1), "Iran-Saudi Normalization Checkpoint", "geopolitical", "MEDIUM",
                   "China-brokered rapprochement assessment. BRICS alignment."),
        WorldEvent(date(2026, 9, 1), "Iran Uranium Enrichment Review (IAEA)", "geopolitical", "HIGH",
                   "Enrichment level assessment. Military action threshold."),

        # ── CHINA / TAIWAN ───────────────────────────────────────────────
        WorldEvent(date(2026, 3, 5), "China Two Sessions (NPC/CPPCC)", "geopolitical", "HIGH",
                   "Annual legislative session. GDP target, military budget, Taiwan rhetoric."),
        WorldEvent(date(2026, 7, 1), "CPC Anniversary (July 1)", "geopolitical", "MEDIUM",
                   "Communist Party founding anniversary. Nationalist rhetoric peak."),
        WorldEvent(date(2026, 8, 1), "PLA Anniversary (August 1)", "military", "MEDIUM",
                   "People's Liberation Army founding. Military exercises around Taiwan heightened."),
        WorldEvent(date(2026, 12, 15), "Taiwan Strait Winter Navigation Check", "geopolitical", "MEDIUM",
                   "Winter strait tensions. Semiconductor supply chain assessment."),

        # ── RUSSIA / UKRAINE ─────────────────────────────────────────────
        WorldEvent(date(2026, 5, 9), "Russia Victory Day", "military", "MEDIUM",
                   "May 9 parade. War status signals. Escalation rhetoric."),
        WorldEvent(date(2026, 7, 15), "NATO Ukraine Assessment", "military", "HIGH",
                   "Mid-year Ukraine conflict status. Aid package decisions."),
        WorldEvent(date(2026, 11, 1), "Russia-Ukraine Winter Offensive Assessment", "military", "HIGH",
                   "Winter campaign. Energy weaponization (gas, oil, nuclear)."),

        # ── JAPAN / YEN CARRY ────────────────────────────────────────────
        WorldEvent(date(2026, 4, 1), "Japan New Fiscal Year Start", "economic", "MEDIUM",
                   "FY2026 begins. Ministry of Finance positioning. Yen carry unwind risk."),
        WorldEvent(date(2026, 7, 20), "Japan Upper House Election", "geopolitical", "HIGH",
                   "Diet elections. BOJ policy continuity at stake. Yen direction."),

        # ── EUROPE ───────────────────────────────────────────────────────
        WorldEvent(date(2026, 3, 1), "EU Carbon Border Adjustment Mechanism Phase-In", "trade", "MEDIUM",
                   "CBAM implementation. Trade friction. ESG cost passthrough."),
        WorldEvent(date(2026, 6, 1), "EU Digital Markets Act Enforcement", "trade", "MEDIUM",
                   "Big-tech regulation. FAANG compliance costs. Uranus-Gemini alignment."),
        WorldEvent(date(2026, 10, 20), "EU Autumn Economic Forecasts", "economic", "MEDIUM",
                   "European recession assessment. EUR/USD impact."),

        # ── CRYPTO / DIGITAL ASSETS ──────────────────────────────────────
        WorldEvent(date(2026, 6, 30), "SEC Crypto Regulation Deadline (Spot ETH ETF)", "trade", "HIGH",
                   "Regulatory clarity window. BTC/ETH institutional flows."),
        WorldEvent(date(2026, 9, 15), "MiCA Full Implementation (EU)", "trade", "MEDIUM",
                   "Markets in Crypto-Assets regulation live. Compliance impact."),
        WorldEvent(date(2027, 1, 3), "BTC Halving Cycle Year 3 Start", "economic", "MEDIUM",
                   "Bitcoin halving cycle analysis. Year 3 historically bullish."),

        # ── KEY ECONOMIC DATES ───────────────────────────────────────────
        WorldEvent(date(2026, 4, 2), "US Tariff Reciprocal Deadline (Day 2)", "trade", "HIGH",
                   "April tariff implementation. Trade war escalation catalyst."),
        WorldEvent(date(2026, 10, 1), "US Government Fiscal Year Start", "economic", "MEDIUM",
                   "FY2027 begins. Budget/shutdown risk."),
        WorldEvent(date(2026, 12, 31), "US Debt Ceiling Reinstatement", "economic", "HIGH",
                   "Debt limit reinstates. Treasury general account drawdown. "
                   "Sovereign debt crisis thesis P=20%."),
    ]


def _build_doctrine_actions() -> Dict[int, DoctrineAction]:
    """Doctrine mandates for each moon cycle with expanded implications."""
    return {
        0: DoctrineAction(
            "ACCUMULATE",
            "Karmic release and discernment phase. Hold existing book while monitoring primes. "
            "Total Lunar Eclipse in Virgo = South Node purification purge. Mercury retrograde in Pisces "
            "(Feb 25–Mar 20) — rough ride, review don't initiate. Venus-Saturn conjunction (Mar 7-8) = "
            "financial sobriety. Mars in Pisces = intuitive action. Eclipses amplify karmic/release themes: "
            "releasing fiat exposure and positioning in hard assets.",
            0.6, ["CASH", "EXISTING_PUTS"]),
        1: DoctrineAction(
            "DEPLOY",
            "Deploy $38k seed on Pink Moon volatility. Gap-up risk window for silver/oil debit spreads. "
            "Libra emphasis on balance. Uranus enters Gemini Apr 25 = major tech/comms disruption (AI ROI cracks). "
            "Jupiter direct in Cancer influence carries = expansion resumes. "
            "SPLIT: 50% silver debit spreads ($19k), 25% oil ($9.5k), 15-20% credit put reinforcement + gold, 5% cash. "
            "Empirical: Full moons = higher vol, silver ~4× returns in new-moon vs full-moon reversals.",
            0.85, ["SLV", "XLE", "GLD", "KRE", "JNK"]),
        2: DoctrineAction(
            "HOLD",
            "Hold and monitor. Conditional double-down on 8% breaches if primes strong. "
            "Scorpio full moon intensity — transformation, power. Building to Blue Moon. "
            "Mars-Neptune/Mars-Saturn aspects in April carry. Ongoing Uranus in Gemini effects. "
            "New Moon accumulation phase: statistically higher returns, lower volatility.",
            0.6, ["SLV", "XLE", "GLD"]),
        3: DoctrineAction(
            "EXIT_ROTATE",
            "Main put-book exit / rotate profits to accelerators. Highest-probability blowup window. "
            "Blue Moon Fire Peak in Sagittarius (May 31) — expansion, truth-seeking, risk-taking. "
            "Summer Solstice (Jun 21) Sun enters Cancer = emotional security. "
            "This is the 'super fuck' window: inflation + private credit gates + supply chokes converge. "
            "Sell 50-70% of printed positions at Blue Moon, rotate into silver/oil for compounding.",
            0.9, ["ROTATE_PUTS_TO_CALLS", "SLV_CALLS", "GDX"]),
        4: DoctrineAction(
            "REBALANCE",
            "Solstice rebalance. Scale silver/oil on any gap-up. Buck Moon in Capricorn = structure/ambition. "
            "Jupiter enters Leo Jun 29 = creative expansion, leadership, visibility. Bullish for silver/gold "
            "as 'shiny' monetary metals and for energy (XLE) as visible power. "
            "Risk appetite expands — amplifies gap-up moves. Q2 earnings vol catalyst.",
            0.75, ["SLV", "XLE", "GLD", "BITO"]),
        5: DoctrineAction(
            "ACCUMULATE",
            "Eclipse preparation. Position for Aug 12 total solar eclipse (Leo themes). "
            "Sturgeon Moon in Aquarius = innovation, collective. New Leo-Aquarius eclipse axis begins. "
            "Neptune re-enters Aries influence ongoing. Hold through vol. "
            "New-moon accumulation: statistically higher returns, lower vol — ideal for drawdown patience.",
            0.65, ["SLV", "GLD", "CASH"]),
        6: DoctrineAction(
            "HOLD",
            "Volatility check. Minor rotation. Corn Moon in Pisces = dissolution, compassion. "
            "Autumnal Equinox approach (Sep 22) Sun enters Libra = balance. "
            "Post-eclipse dust settles. North Node now in Aquarius = collective focus shift.",
            0.5, ["EXISTING_POSITIONS"]),
        7: DoctrineAction(
            "REBALANCE",
            "Major rebalance / profit rotation. Harvest Moon in Aries = action, initiation. "
            "Equinox amplifier. Q3 earnings. Pluto sextile Neptune supports systemic transformation. "
            "Cardinal resets (equinox) = high sentiment vol overlapping Fire Peaks for compounding.",
            0.8, ["ROTATE", "TRIM_WINNERS", "ADD_LAGGARDS"]),
        8: DoctrineAction(
            "HOLD",
            "Hunter Moon in Taurus = stability, values. Election proximity — reduce risk. "
            "Hold through noise. Mars-Jupiter alignments mid-November coming. "
            "WTO Ministerial + BRICS Summit risk clusters.",
            0.5, ["EXISTING_POSITIONS", "HEDGE_VIX"]),
        9: DoctrineAction(
            "HOLD",
            "Beaver Supermoon in Gemini = communication, duality. Max tidal/vol amplification. "
            "Mars-Jupiter exact conjunction Nov 16. Fed December meeting approaching. "
            "Jupiter retrograde begins Dec 13 (Leo) — creative expansion internalized. "
            "Winter Solstice Dec 21 Sun enters Capricorn = structure/authority themes.",
            0.55, ["EXISTING_POSITIONS"]),
        10: DoctrineAction(
            "ACCUMULATE",
            "Cold Moon in Cancer = home, security. Year-end. New year accumulation phase. "
            "Q4 earnings lead-in. Solstice-aligned positioning. "
            "Jupiter retrograde ongoing — contrarian buy signal for conviction trades.",
            0.65, ["SLV", "GLD", "QUALITY_LONGS"]),
        11: DoctrineAction(
            "HOLD",
            "Wolf Moon in Leo = expression, creativity. Saturn enters Aries Feb 13 = "
            "MAJOR 2.5-year cycle of new structures/discipline. End of easy money era reinforced. "
            "Rates trap + private credit gates harden. Bearish for credit puts (OBDC/ARCC prints harder). "
            "Hold through geopolitical reset transition.",
            0.5, ["EXISTING_POSITIONS"]),
        12: DoctrineAction(
            "HOLD",
            "Snow Moon in Virgo = discernment, purification. Full circle from Moon 0. "
            "SATURN-NEPTUNE CONJUNCTION AT 0° ARIES — the single most significant event in the entire doctrine. "
            "36-year cycle starter at zodiac zero point. Last: 1989 Berlin Wall/Cold War end. "
            "2027: Petrodollar death spiral visibility, private credit crunch, de-dollarization, "
            "silver as bridge asset (Neptune's dream metal + Saturn's real metal). "
            "Final rotation out of credit puts. Full scaling into silver/oil accelerators. "
            "This is the moment the fireworks become visible to everyone.",
            0.6, ["EXISTING_POSITIONS", "HEDGE", "SLV", "GLD"]),
        13: DoctrineAction(
            "REVIEW",
            "$10M milestone review. Final rotation. Worm Moon in Libra = balance. "
            "Spring Equinox 2027 = new cycle. Saturn-Neptune afterglow + equinox creates final review window. "
            "Structural regime change (new monetary order) cements silver/gold as cycle winners. "
            "Full-year doctrine assessment. The powder-keg cluster is resolved — harvest or re-seed.",
            0.7, ["FULL_REVIEW", "CONSOLIDATE"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MOON BRIEFINGS — Rich narrative per cycle
# ═══════════════════════════════════════════════════════════════════════════

MOON_BRIEFINGS: Dict[int, Dict[str, str]] = {
    0: {
        "theme": "Karmic Release & Purification",
        "lunar": "Total Lunar Eclipse in Virgo (South Node) — release perfectionism, purge old patterns. "
                 "Conjunct South Node, sextile Jupiter in Cancer for emotional support.",
        "astro_highlights": "Mar 20 Spring Equinox (Sun enters Aries — new astrological year). "
                           "Mars in Pisces (Mar 2 — intuitive action, dissolution). "
                           "Mercury retrograde in Pisces (Feb 25–Mar 20 — rough ride, review don't initiate). "
                           "Venus-Saturn conjunction (Mar 7-8 — financial sobriety).",
        "market_implication": "Eclipse windows amplify karmic/release themes — perfect for releasing fiat exposure "
                             "and positioning in hard assets. Mercury Rx = contract confusion, travel delays. "
                             "Do NOT initiate new trades until Mar 20 station direct.",
        "empirical": "Eclipses documented as sentiment reset catalysts. Full moon Fire Peaks = higher vol, "
                     "lower avg equity returns (Yuan/Zheng/Zhu 2006, Lucey 2010).",
    },
    1: {
        "theme": "Pink Moon Deployment — Seed Capital",
        "lunar": "Pink Moon Fire Peak (Apr 1-2, Libra — balance/relationships). New Moon ~Apr 17.",
        "astro_highlights": "Apr 25 Uranus enters Gemini — MAJOR tech/comms/innovation shift (lasts 2032-33). "
                           "Jupiter direct in Cancer influence carries. Mercury direct = green light.",
        "market_implication": "Deploy $38k seed on Pink Moon volatility. Gap-up risk window for silver/oil debit spreads. "
                             "Uranus Gemini = AI disruption ties to big-tech April earnings cracks (AI ROI skepticism). "
                             "Split: 50% silver ($19k), 25% oil ($9.5k), 15-20% credit puts + gold, 5% cash.",
        "empirical": "Silver shows ~4x higher returns in new-moon periods vs full-moon reversals. "
                     "Full moons create liquidity for debit spreads and gap-up torque on silver (dual monetary + industrial).",
    },
    2: {
        "theme": "Hold & Monitor — Scorpio Transformation",
        "lunar": "Full Moon May 1 (Scorpio — transformation, power). Building to Blue Moon. New Moon ~May 16-17.",
        "astro_highlights": "Ongoing Uranus in Gemini effects. Mars aspects to outer planets (Mars-Neptune, Mars-Saturn).",
        "market_implication": "Hold and monitor. Conditional double-down on 8% breaches if primes strong. "
                             "New Moon accumulation: statistically higher returns, lower volatility — ideal for holding.",
        "empirical": "New Moon phases = statistically higher returns and lower volatility. "
                     "Ideal for holding through drawdowns and monitoring 8% stops without forced action.",
    },
    3: {
        "theme": "SUPER FUCK Window — Blue Moon Exit/Rotate",
        "lunar": "Blue Moon Fire Peak (May 31, Sagittarius — expansion, truth-seeking).",
        "astro_highlights": "Jun 21 Summer Solstice (Sun enters Cancer — emotional security, nurturing). "
                           "Sagittarius Fire Peak = risk-taking amplifier.",
        "market_implication": "Main put-book exit / rotate profits to accelerators. HIGHEST-PROBABILITY BLOWUP WINDOW. "
                             "Inflation + private credit gates + supply chokes converge. "
                             "Sell 50-70% of printed positions at Blue Moon (May 31) and Solstice (Jun 21). "
                             "Rotate into silver/oil for compounding.",
        "empirical": "Dual chokepoint black swan + April big-tech earnings cracks accelerate powder-keg "
                     "into June 'super fuck' window.",
    },
    4: {
        "theme": "Solstice Rebalance — Jupiter Leo Expansion",
        "lunar": "Full Moon Jun 29 (Capricorn — structure, ambition). Solstice amplifier.",
        "astro_highlights": "Jupiter enters Leo (Jun 29 — Jul 25, 2027) — creative expansion, leadership, visibility. "
                           "Bullish for silver/gold as 'shiny' metals and for energy (XLE) as visible power.",
        "market_implication": "Rebalance. Scale silver/oil on gap-ups. Jupiter Leo = risk appetite expands. "
                             "Gap-up moves amplified. Q2 earnings vol catalyst.",
        "empirical": "Jupiter ingress into fire signs historically correlates with commodity bull runs "
                     "and increased trader confidence.",
    },
    5: {
        "theme": "Eclipse Preparation — Leo-Aquarius Axis",
        "lunar": "Full Moon Jul 29 (Aquarius — innovation, collective).",
        "astro_highlights": "Aug 12-13 Total Solar Eclipse (near perigee, Leo themes). "
                           "Neptune in Aries influence ongoing. New Leo-Aquarius eclipse axis begins.",
        "market_implication": "Accumulate. Position for Aug 12 total solar eclipse. Hold through vol. "
                             "New eclipse axis = identity/power vs. collective innovation theme for 2+ years.",
        "empirical": "Total solar eclipses near perigee = maximum gravitational/tidal influence. "
                     "Documented sentiment shift along path of totality.",
    },
    6: {
        "theme": "Post-Eclipse Assessment — Equinox Approach",
        "lunar": "Full Moon Aug 27 (Pisces — dissolution, compassion; lunar eclipse visibility in some regions).",
        "astro_highlights": "Sep 22 Autumnal Equinox (Sun enters Libra — balance, relationships). "
                           "Pluto sextile Neptune — deep systemic transformation.",
        "market_implication": "Volatility check. Minor rotation if needed. Post-eclipse dust settles. "
                             "North Node now in Aquarius = collective focus shift.",
        "empirical": "Equinox windows create cardinal resets — high sentiment volatility "
                     "overlapping Fire Peaks for compounding rotations.",
    },
    7: {
        "theme": "Harvest Moon — Major Profit Rotation",
        "lunar": "Full Moon Sep 26 (Aries — action, initiation; Equinox amplifier).",
        "astro_highlights": "Sep 22 Equinox. Pluto aspects (sextile/trine outer planets).",
        "market_implication": "Major rebalance / profit rotation. Q3 earnings. "
                             "Aries Fire Peak + Equinox = maximum action/rotation energy.",
        "empirical": "Autumn equinox historically marks seasonal volatility increase. "
                     "Harvest moon (closest to equinox) amplifies overnight risk sentiment.",
    },
    8: {
        "theme": "Hunter Moon — Election Hedge",
        "lunar": "Full Moon Oct 25 (Taurus — stability, values).",
        "astro_highlights": "Mars-Jupiter alignments building mid-November.",
        "market_implication": "Hold. Election proximity = reduce risk. Taurus = focus on material security. "
                             "BRICS Summit + WTO risks cluster.",
        "empirical": "US midterm elections historically create 2-4 weeks of elevated implied volatility "
                     "followed by post-election rally.",
    },
    9: {
        "theme": "Supermoon Volatility — Mars-Jupiter Conjunction",
        "lunar": "Full Moon Nov 24 (Gemini — communication, duality; SUPERMOON = closest approach).",
        "astro_highlights": "Dec 21 Winter Solstice (Sun enters Capricorn). "
                           "Jupiter retrograde Dec 13, 2026 – Apr 13, 2027. "
                           "Mars-Jupiter exact conjunction Nov 16.",
        "market_implication": "Supermoon = maximum tidal/vol amplification. Mars-Jupiter = aggressive expansion energy. "
                             "Fed December meeting. Jupiter Rx = optimism contracts — contrarian buy signal.",
        "empirical": "Supermoon perigee events show measurable increase in market volatility. "
                     "Mars-Jupiter conjunctions historically correlate with military/trade escalation.",
    },
    10: {
        "theme": "Cold Moon — Year-End Accumulation",
        "lunar": "Full Moon Dec 24 (Cancer — home, security).",
        "astro_highlights": "Dec 21 Solstice. Jupiter retrograde ongoing.",
        "market_implication": "Year-end accumulation. Q4 earnings lead-in. "
                             "Jupiter Rx = contrarian positioning for conviction trades.",
        "empirical": "Year-end tax-loss selling creates opportunity. "
                     "Cancer moon = protective/defensive sentiment favors safe-haven metals.",
    },
    11: {
        "theme": "Wolf Moon — Saturn Aries New Order",
        "lunar": "Full Moon Jan 22 (Leo — expression, creativity).",
        "astro_highlights": "Feb 13 Saturn enters Aries — MAJOR 2.5-year cycle of structure/initiative. "
                           "End of easy-money era reinforced.",
        "market_implication": "Saturn Aries = new authority structures. Rates trap + private credit gates harden. "
                             "Bearish for credit puts (OBDC/ARCC book prints hardest). "
                             "Hold through geopolitical reset.",
        "empirical": "Saturn sign changes historically mark 2-3 year macro regime shifts. "
                     "Saturn in cardinal fire = aggressive structural discipline.",
    },
    12: {
        "theme": "SATURN-NEPTUNE CONJUNCTION — 36-Year Reset",
        "lunar": "Snow Moon in Virgo — discernment, purification. Full circle from Moon 0.",
        "astro_highlights": "Feb 20 Saturn-Neptune Conjunction at 0 Aries — THE event. "
                           "Annular Solar Eclipse Feb 6. Total Lunar Eclipse Feb 20 (same day!).",
        "market_implication": "ONCE-IN-36-YEARS at zodiac zero point. Dissolution of old financial structures. "
                             "Petrodollar death spiral visibility. Private credit crunch. De-dollarization. "
                             "Silver = bridge asset (Neptune's dream metal + Saturn's real metal). "
                             "Final rotation out of credit puts. Full scaling into silver/oil accelerators.",
        "empirical": "Last Saturn-Neptune conjunction (1989 Capricorn): Fall of Berlin Wall, end of Cold War, "
                     "birth of modern fiat system. Previous: 1953 (consumer credit boom), 1917 (Russian Revolution), "
                     "1881 (industrial trusts). Each time: old empires dissolve, new power structures emerge.",
    },
    13: {
        "theme": "$10M Milestone Review — New Cycle Dawns",
        "lunar": "Worm Moon in Libra — balance. Equinox amplifier.",
        "astro_highlights": "Mar 20 Spring Equinox (Sun enters Aries). "
                           "Saturn-Neptune afterglow. Jupiter direct Apr 13.",
        "market_implication": "$10M milestone review. Saturn-Neptune afterglow + Equinox = final review window. "
                             "Structural regime change cements silver/gold as cycle winners. "
                             "Harvest or re-seed for next doctrine cycle.",
        "empirical": "Spring equinox 2027 marks completion of full 13-moon compounding cycle. "
                     "New Saturn-Neptune 36-year era begins — position for structural winners.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SATURN-NEPTUNE CONJUNCTION DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════

SATURN_NEPTUNE_DEEPDIVE: Dict[str, Any] = {
    "date": "2027-02-20",
    "degree": "0°00' Aries",
    "cycle_years": 36,
    "significance": "Single most significant astrology marker in the entire 13-moon doctrine. "
                    "0° Aries = absolute beginning of the zodiac wheel — structural reset button for collective reality.",
    "saturn_themes": "Reality, restriction, discipline, long-term structures, authority, boundaries",
    "neptune_themes": "Dissolution, dreams, illusions, spirituality, collective vision, compassion",
    "blend": "Dreams meeting reality — collective fantasies (AI hype, endless fiat debt, petrodollar supremacy, "
             "private-credit liquidity illusion) forced to confront hard limits.",
    "historical_parallels": [
        {"year": 1989, "sign": "Capricorn", "events": "Fall of Berlin Wall, end of Cold War, collapse of Soviet communism, "
         "birth of modern globalized fiat system, start of 'peace dividend' era. "
         "Dissolution of old world order → imposition of new structures (unipolar US dominance, globalization, debt-based growth)."},
        {"year": 1953, "sign": "Libra", "events": "End of Korean War, start of consumer credit boom, suburban expansion."},
        {"year": 1917, "sign": "Leo", "events": "Russian Revolution, birth of communism, WWI collapse of empires."},
        {"year": 1881, "sign": "Taurus", "events": "End of Reconstruction, rise of industrial trusts, Gilded Age begins."},
    ],
    "2027_predictions": {
        "petrodollar": "Final visibility of end of 50-year fiat experiment. Accelerated de-dollarization, "
                       "yuan/gold-backed alternatives, Treasury sales (Japan's $1.2T holdings flashpoint).",
        "private_credit": "Saturn restricts easy liquidity; Neptune dissolves 'evergreen' redemption promises. "
                          "Gates, NAV markdowns, forced asset sales become mainstream — OBDC/ARCC/HYG/JNK puts print hardest.",
        "rates_trap": "Higher oil from chokepoints feeds CPI; Fed structurally trapped. "
                      "Saturn-Neptune forces structural resolution (higher-for-longer or regime change).",
        "silver": "Bridge asset — Neptune's 'dream metal' (monetary safe-haven) meets Saturn's 'real metal' "
                  "(industrial necessity from sulfuric-acid/copper/cobalt choke). Ultimate catalyst for silver's dual role.",
        "japan_carry": "Neptune dissolves yen's safe-haven status; Saturn enforces new boundaries "
                       "on Treasury holdings and rates.",
    },
    "doctrine_action": "Moon 12: Final rotation out of credit puts (sell 50-70% on peak visibility). "
                       "Full scaling into silver/oil accelerators. Moon 13: $10M review window. "
                       "The powder-keg cluster is already lit. The conjunction is the moment the fireworks become visible to everyone.",
}


# ═══════════════════════════════════════════════════════════════════════════
# AGE OF AQUARIUS DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════

AGE_OF_AQUARIUS: Dict[str, Any] = {
    "phenomenon": "Precession of the Equinoxes — Earth's axial wobble shifts the vernal equinox backward "
                  "through the zodiac at ~1 degree every 72 years.",
    "great_year": "~25,772-25,800 years (Platonic Year). Each age ~2,150-2,160 years.",
    "current_transition": "Late Pisces → early Aquarius cusp. Gradual, not instantaneous. "
                          "Estimates range from ~2000 (already underway) to ~2150 CE (IAU constellation boundary).",
    "sign_qualities": {
        "element": "Fixed Air (intellect, ideas, communication, objectivity)",
        "rulers": "Saturn (structure, discipline) + Uranus (revolution, innovation)",
        "key_phrase": "I know — knowledge, awakening, intellectual freedom",
        "opposite": "Leo — individual creativity + heart-centered leadership tempers Aquarian detachment",
    },
    "core_themes": [
        "Innovation, rationality, science, technology",
        "Humanitarianism, equality, networks, community ('we' over 'me')",
        "Freedom, rebellion against outdated hierarchies",
        "Collective consciousness, detachment from old emotional/faith-based paradigms",
        "Decentralization — power shifts from hierarchies (Pisces) to networks (Aquarius)",
    ],
    "shadow_themes": [
        "Not utopia — transition is chaotic; old structures dissolve before new ones stabilize",
        "Surveillance, groupthink, cold rationalism without heart",
        "Loss of individuality, tech overreach, AI displacement",
        "Polarization and 'awakening pain' before collective harmony",
    ],
    "element_shift": "From Water (Pisces: emotion, illusion, compassion, sacrifice) to "
                     "Air (Aquarius: intellect, ideas, communication, objectivity). "
                     "This is why the world feels more mental, tech-driven, and less purely spiritual.",
    "amplifying_transits_2026": [
        "Pluto in Aquarius (since 2024) — long-term structural transformation",
        "Uranus in Gemini (April 2026) — tech/comms disruption overlay",
        "Saturn-Neptune conjunction at 0 Aries (Feb 2027) — 36-year cycle reset",
    ],
    "cultural_history": {
        "1967": "Hair musical 'Age of Aquarius' — counterculture, civil rights, spiritual awakening",
        "previous_shift": "End of Aries (~0 CE) → Pisces: rise of Christianity, faith, sacrifice, hierarchy",
        "current": "Pisces dissolving (endless debt, centralized authority, fiat illusion) → "
                   "Aquarius forming (networks, transparency, resource-backed value, collective power)",
    },
    "doctrine_alignment": {
        "powder_keg": "Resource-backed reset (oil, silver/gold, copper/cobalt shortages) + "
                      "de-dollarization = classic Aquarian tech + collective resource redistribution",
        "silver": "Silver embodies Aquarius: innovation/industrial (Saturn) + monetary awakening (Uranus). "
                  "Ultimate bridge asset for the age.",
        "private_credit": "April 2026 earnings (Moon 1) = 'cracks showing'. June 'super fuck' visibility "
                          "(Moon 3/4) = dissolution of Piscean illusions into Aquarian transparency.",
        "personal_arc": "Lead-up (now to Feb 2027) = preparation phase for second half of life. "
                        "Saturn-Neptune at 0 Aries = structural 'new me' reset.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SACRED GEOMETRY — PER-MOON OVERLAY
# ═══════════════════════════════════════════════════════════════════════════

SACRED_GEOMETRY_OVERLAY: Dict[int, Dict[str, Any]] = {
    0: {
        "geometry": "Seed of Life",
        "description": "7 overlapping circles — foundation of creation. New beginning.",
        "frequency_hz": 396,
        "frequency_name": "Liberation (Solfeggio UT)",
        "platonic_solid": None,
        "angle_sum": None,
        "vertices": 7,
        "phi_link": "7-fold symmetry resonates at root Solfeggio base tone",
        "correlation": "Eclipse anchor = seed planted. Release old fiat illusions (petrodollar). "
                       "Accumulation phase matches new-moon energy.",
    },
    1: {
        "geometry": "Flower of Life",
        "description": "19 overlapping circles, 6-fold symmetry. Contains all Platonic solids in 2D.",
        "frequency_hz": 528,
        "frequency_name": "DNA Repair / Love (Solfeggio MI)",
        "platonic_solid": "All (embedded)",
        "angle_sum": None,
        "vertices": 19,
        "phi_link": "Circle radius ratios follow phi. Harmonic f = v/lambda scaled by overlaps.",
        "correlation": "HIGH — Pink Moon vol spike + big-tech earnings cracks = ideal LEAPS entry. "
                       "Phi^1 node (~Apr 12-15) resonates with deployment torque. 528 Hz = heart-centered compounding.",
    },
    2: {
        "geometry": "Merkaba",
        "description": "Star tetrahedron (two interlocking tetrahedra). Rotation creates toroidal field.",
        "frequency_hz": 639,
        "frequency_name": "Connecting Relationships (Solfeggio FA)",
        "platonic_solid": "Dual Tetrahedra",
        "angle_sum": 1440,
        "vertices": 8,
        "phi_link": "Rotational harmonics = 528 x phi multiples",
        "correlation": "Accumulation. Hold through drawdowns. Merkaba 'spin' mirrors conditional "
                       "double-down discipline. Balance male/female energy.",
    },
    3: {
        "geometry": "Dodecahedron + Icosahedron",
        "description": "12 pentagonal faces (phi-dominant) + 20 triangular faces. Dual pair.",
        "frequency_hz": 741,
        "frequency_name": "Awakening Intuition (Solfeggio SOL)",
        "platonic_solid": "Dodecahedron (6,480 deg) + Icosahedron (3,600 deg)",
        "angle_sum": 10080,
        "vertices": 32,
        "phi_link": "Pentagon diagonals = phi. Strongest phi-geometry in Platonic set.",
        "correlation": "VERY HIGH — Main put-book exit/rotation. Blue Moon + Solstice = maximum cascade "
                       "visibility. Dodecahedron 12 faces = Moon 3 payoff. 741 Hz = intuition awakening for rotation.",
    },
    4: {
        "geometry": "Sri Yantra",
        "description": "9 interlocking triangles (5 down, 4 up) around bindu. 43 sub-triangles. "
                       "Golden ratio in every proportion.",
        "frequency_hz": 852,
        "frequency_name": "Returning to Spiritual Order (Solfeggio LA)",
        "platonic_solid": None,
        "angle_sum": None,
        "vertices": 43,
        "phi_link": "Side ratios = phi^n. Energy flow modeled as standing waves.",
        "correlation": "Rebalance/scale. Solstice amplifier + Jupiter Leo = creative expansion "
                       "for silver compounding. Integration of masculine/feminine market forces.",
    },
    5: {
        "geometry": "Flower of Life + Solar Eclipse",
        "description": "19-circle expansion overlaid with eclipse reset node.",
        "frequency_hz": 528,
        "frequency_name": "DNA Repair / Love (Solfeggio MI) — Eclipse Overlay",
        "platonic_solid": "All (embedded)",
        "angle_sum": None,
        "vertices": 19,
        "phi_link": "Eclipse amplifies 528 Hz resonance — total reset frequency.",
        "correlation": "Accumulation. Eclipse = major reset node for private credit visibility. "
                       "Flower of Life expansion mirrors portfolio compound growth.",
    },
    6: {
        "geometry": "Octahedron",
        "description": "8 triangular faces, 6 vertices. Air element. Balance and equilibrium.",
        "frequency_hz": 528,
        "frequency_name": "DNA Repair / Love (Solfeggio MI)",
        "platonic_solid": "Octahedron",
        "angle_sum": 1440,
        "vertices": 6,
        "phi_link": "V = a^3 * sqrt(2)/3. Dual of Cube. Air element = Aquarian resonance.",
        "correlation": "Balance phase. Phi^6 coherence window. Incremental compounding. "
                       "Octahedron air element echoes Aquarian Age transition.",
    },
    7: {
        "geometry": "Cube (Hexahedron)",
        "description": "6 square faces, 8 vertices, 12 edges. Earth element. Grounding.",
        "frequency_hz": 417,
        "frequency_name": "Facilitating Change (Solfeggio RE)",
        "platonic_solid": "Cube",
        "angle_sum": 2160,
        "vertices": 8,
        "phi_link": "Euler: 8-12+6=2. 2,160 deg = one astrological age length in years.",
        "correlation": "Harvest Moon grounding. Cube angle sum (2,160) = one zodiac age duration. "
                       "Structural consolidation of gains. Autumnal equinox rebalance prep.",
    },
    8: {
        "geometry": "Tetrahedron",
        "description": "4 triangular faces, 4 vertices, 6 edges. Fire element. Simplest Platonic solid.",
        "frequency_hz": 396,
        "frequency_name": "Liberation from Fear (Solfeggio UT)",
        "platonic_solid": "Tetrahedron",
        "angle_sum": 720,
        "vertices": 4,
        "phi_link": "V = a^3 * sqrt(2)/12. Freq 396 = 528 x 0.75.",
        "correlation": "Hunter Moon. Fire element = conviction testing. Liberation from fear "
                       "of drawdowns. Hold or scale based on thesis confirmation.",
    },
    9: {
        "geometry": "Metatron's Cube",
        "description": "13 circles + 78 lines. Contains every Platonic solid. Divine structure.",
        "frequency_hz": 528,
        "frequency_name": "DNA Repair / Love (Solfeggio MI) — Divine Structure",
        "platonic_solid": "All 5 (embedded)",
        "angle_sum": None,
        "vertices": 13,
        "phi_link": "13 circles = 13 moon cycle. Full Platonic containment = complete geometry.",
        "correlation": "Beaver Supermoon. 13 circles mirror 13-moon doctrine. Metatron's Cube "
                       "= divine template for the full compounding path. Peak coherence reference.",
    },
    10: {
        "geometry": "Icosahedron",
        "description": "20 triangular faces, 12 vertices, 30 edges. Water element. Flow.",
        "frequency_hz": 741,
        "frequency_name": "Awakening Intuition (Solfeggio SOL)",
        "platonic_solid": "Icosahedron",
        "angle_sum": 3600,
        "vertices": 12,
        "phi_link": "Dual of Dodecahedron. 12 vertices = 12 zodiac signs.",
        "correlation": "Cold Moon. Water element bridges Pisces-to-Aquarius transition. "
                       "12 vertices = zodiac completion. Prepare for Saturn ingress.",
    },
    11: {
        "geometry": "Dodecahedron",
        "description": "12 pentagonal faces, 20 vertices, 30 edges. Cosmos/Ether element.",
        "frequency_hz": 639,
        "frequency_name": "Connecting Relationships (Solfeggio FA)",
        "platonic_solid": "Dodecahedron",
        "angle_sum": 6480,
        "vertices": 20,
        "phi_link": "Pentagon diagonals = phi. Cosmos element = quintessence. Highest phi-geometry.",
        "correlation": "Wolf Moon. Saturn Aries ingress prep. Dodecahedron ether = transcendent "
                       "structure. Prepare for structural shift from old order to new.",
    },
    12: {
        "geometry": "Metatron's Cube + Merkaba",
        "description": "Divine structure (13 circles) + light body activation (dual tetrahedra). "
                       "Combined = full ascension geometry.",
        "frequency_hz": 639,
        "frequency_name": "Unity (Solfeggio FA) + 528 Hz (Love)",
        "platonic_solid": "All 5 + Dual Tetrahedra",
        "angle_sum": None,
        "vertices": 21,
        "phi_link": "Metatron 13 + Merkaba 8 = 21 (Fibonacci). Full phi cascade.",
        "correlation": "PEAK — Saturn-Neptune conjunction. Old illusions dissolve (petrodollar, "
                       "endless credit). New foundations form. $10M milestone rotation. "
                       "Combined geometry = full ascension from old monetary system.",
    },
    13: {
        "geometry": "Full Platonic Set + Flower of Life",
        "description": "All 5 Platonic solids nested within Flower of Life. Completion. "
                       "432 Hz universal harmony.",
        "frequency_hz": 432,
        "frequency_name": "Universal Harmony (Ancient Tuning A=432)",
        "platonic_solid": "All 5 (nested in Flower of Life)",
        "angle_sum": 14400,
        "vertices": None,
        "phi_link": "432 = sum of all Platonic angle sums modulo phi harmonic. Completion tone.",
        "correlation": "$10M review. New 36-year cycle begins with silver/gold as bridge assets. "
                       "Full Platonic completion = doctrine fulfilled. Worm Moon = rebirth into "
                       "Age of Aquarius with real capital and independence.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DALIO BIG CYCLE FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

DALIO_BIG_CYCLE: Dict[str, Any] = {
    "framework": "Meta-cycle of empires, monetary systems, political orders, and world orders. "
                 "Rise → Peak → Decline → Reset over 50-100+ years. Studied 500 years of history "
                 "(Dutch, British, American empires, Chinese dynasties).",
    "five_forces": [
        "Debt/Money/Credit Cycle (central — long-term ~75-100 years)",
        "Internal Order/Disorder Cycle (wealth gaps, populism, civil conflict)",
        "External Order/Disorder Cycle (geopolitics, wars, reserve currency status)",
        "Productivity & Innovation Cycle (technology, education, competitiveness)",
        "Acts of Nature / Exogenous Shocks (pandemics, climate, wars as catalysts)",
    ],
    "stages": {
        "1_rise": "Strong leadership, sound money, productive debt. Growth strong, inflation low.",
        "2_bubble": "Debt rises faster than income/productivity. Financial wealth inflates vs real wealth. "
                    "Wealth gaps widen, populism rises.",
        "3_top": "Central banks keep stimulating. Feels good short-term. Unsustainable.",
        "4_decline": "Debt service unsustainable. Central banks print → currency devaluation + inflation. "
                     "Real returns negative. Internal conflict rises. External pressures mount.",
        "5_breakdown": "Debt defaults, currency crises, political upheaval, sometimes war. "
                       "Old order collapses; new monetary/political/geopolitical orders emerge.",
    },
    "current_position_2026": "Late Stage 4 → entering Stage 5. US debt/GDP extreme, interest expense "
                             "rivaling defense spending. Widening wealth gaps, rising rival (China), "
                             "printing to manage debt → inflation + currency weakness. "
                             "Dalio: 2026-2029 danger zone.",
    "bridgewater_portfolio_2026": {
        "aum": "$27.42B",
        "holdings": 1040,
        "turnover": "22%",
        "top_positions": "SPY 11.08%, IVV 10.45%, NVDA 2.63%, LRCX 1.90%, CRM 1.87%",
        "note": "Heavy US equity beta (SPY+IVV ~21.5%) + AI/semiconductor tilt. "
                "Gold/silver mining (Newmont) small but present — nod to hard-asset pivot.",
    },
    "doctrine_alignment": {
        "petrodollar": "#18 — Classic late-cycle reserve currency decline (excessive debt, printing, devaluation)",
        "hormuz": "#1 — External disorder + resource wars over shrinking pie",
        "private_credit": "#2 — Late-cycle debt burdens + money printing → liquidity stress + stagflation",
        "silver_gold_oil": "Hard assets outperform financial assets (promises) in late/downwave. "
                           "Silver's dual role (monetary + industrial) = maximum leverage.",
        "lunar_timing": "Fire Peaks give tactical entry/rotation during increasing volatility.",
        "saturn_neptune": "Feb 2027 = the 'reset' phase Dalio describes — dissolution of illusions, "
                          "emergence of new structures.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# ACCOUNT BALANCES — Auto-loaded from central config (data/account_balances.json)
# Update via:  python -m config.account_balances set <key> --balance <val>
# ═══════════════════════════════════════════════════════════════════════════

def _build_exchange_inventory() -> Dict[str, Any]:
    """Build EXCHANGE_INVENTORY from central config + static platform metadata."""
    try:
        from config.account_balances import Balances
        _accts = Balances.all_accounts()
        _fx = Balances.cad_usd()
        _total_usd = Balances.total_portfolio_usd()
    except Exception:
        _accts = {}
        _fx = 0.70
        _total_usd = 0.0

    def _bal(key: str) -> float:
        return float(_accts.get(key, {}).get("total_assets", 0)
                     or _accts.get(key, {}).get("balance", 0))

    usd_confirmed = _bal("ibkr") + _bal("moomoo") + _bal("polymarket")
    cad_total = _bal("wealthsimple") + _bal("ndax") + _bal("eq_bank")

    return {
        "last_scan": _accts.get("ibkr", {}).get("verified", "unknown"),
        "platforms": {
            "ibkr": {
                "label": "Interactive Brokers",
                "account": _accts.get("ibkr", {}).get("account_id", "U24346218"),
                "mode": "LIVE",
                "port": "auto-detect (7496/7497)",
                "currency": _accts.get("ibkr", {}).get("currency", "USD"),
                "estimated_balance": _bal("ibkr"),
                "note": _accts.get("ibkr", {}).get("note", ""),
            },
            "moomoo": {
                "label": "Moomoo (Futu Canada)",
                "mode": "REAL",
                "security_firm": "FUTUCA",
                "currency": _accts.get("moomoo", {}).get("currency", "USD"),
                "scanned_balance": _bal("moomoo"),
                "note": _accts.get("moomoo", {}).get("note", ""),
            },
            "ndax": {
                "label": "NDAX (National Digital Asset Exchange)",
                "currency": "CAD",
                "estimated_balance": _bal("ndax"),
                "status": "LIQUIDATED",
                "note": _accts.get("ndax", {}).get("note", ""),
            },
            "polymarket": {
                "label": "Polymarket (CLOB)",
                "chain": "Polygon (137)",
                "sig_type": 1,
                "funder": "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8",
                "currency": "USDC",
                "scanned_balance": _bal("polymarket"),
                "note": _accts.get("polymarket", {}).get("note", ""),
            },
            "metamask": {
                "label": "MetaMask / Polygon Wallet",
                "eoa": "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8",
                "chains": ["Polygon", "Ethereum"],
                "tokens_tracked": ["MATIC", "ETH", "USDC.e", "USDC"],
                "scanned_balance": 0,
                "note": "Funder address. ETH mainnet shows $0. Polygon RPC check pending.",
            },
            "eqbank": {
                "label": "EQ Bank",
                "currency": "CAD",
                "estimated_balance": _bal("eq_bank"),
                "note": _accts.get("eq_bank", {}).get("note", ""),
            },
            "wealthsimple": {
                "label": "Wealthsimple (via SnapTrade)",
                "accounts": ["TFSA", "RRSP"],
                "currency": "CAD",
                "estimated_balance": _bal("wealthsimple"),
                "note": _accts.get("wealthsimple", {}).get("note", ""),
            },
        },
        "totals": {
            "usd_confirmed": round(usd_confirmed, 2),
            "cad_estimated": round(cad_total, 2),
            "grand_total_usd": round(_total_usd, 2),
            "cad_usd_rate": _fx,
            "note": "Auto-loaded from data/account_balances.json",
        },
        "scanner_script": "_check_all_balances.py",
        "snaptrade_setup": "_setup_snaptrade.py",
    }


ACCOUNT_BALANCES: Dict[str, Any] = _build_exchange_inventory()


def _build_capital_deployment() -> Dict[str, Any]:
    """Build live capital deployment state from balances + current positions."""
    try:
        from config.account_balances import Balances
        from strategies.war_room_engine import CURRENT_POSITIONS

        accts = Balances.all_accounts()
        fx = float(Balances.cad_usd() or 0.72)
        injection = Balances._load().get("injection", {})
    except Exception:
        return {
            "injection_total": "$0",
            "schedule": "70% first 30 days, 20% next 30, 10% final 30",
            "accounts": [],
            "total_positions": 0,
        }

    label_map = {
        "ibkr": ("IBKR", "Primary options venue"),
        "moomoo": ("Moomoo", "Equity + LEAPS calls"),
        "wealthsimple": ("WealthSimple TFSA", "Tax-advantaged options / holdings"),
        "ndax": ("NDAX", "Crypto cash / exchange reserve"),
        "eq_bank": ("EQ Bank", "Emergency buffer"),
        "polymarket": ("Polymarket", "Event-driven binary exposure"),
    }

    positions_by_account: Dict[str, int] = {}
    for pos in CURRENT_POSITIONS:
        positions_by_account[pos.account] = positions_by_account.get(pos.account, 0) + 1

    accounts = []
    for key in ["ibkr", "moomoo", "wealthsimple", "ndax", "eq_bank", "polymarket"]:
        acct = accts.get(key, {})
        if not acct:
            continue
        label, role = label_map[key]
        amount = float(acct.get("total_assets", acct.get("balance", 0)) or 0)
        currency = acct.get("currency", "USD")
        usd_amount = amount * fx if currency == "CAD" else amount
        cad_amount = amount if currency == "CAD" else (amount / fx if fx else 0.0)
        accounts.append({
            "name": label,
            "balance": f"${usd_amount:,.2f} USD | C${cad_amount:,.2f} CAD",
            "balance_native": f"${amount:,.2f} {currency}",
            "balance_usd": round(usd_amount, 2),
            "balance_cad": round(cad_amount, 2),
            "role": role,
            "positions": positions_by_account.get(label.replace(" TFSA", ""), positions_by_account.get(label, len(acct.get("positions", []) or []))),
        })

    return {
        "injection_total": f"${float(injection.get('total', 0) or 0):,.0f}",
        "schedule": "70% first 30 days, 20% next 30, 10% final 30",
        "cad_usd_rate": fx,
        "accounts": accounts,
        "total_positions": len(CURRENT_POSITIONS),
    }


def _refresh_live_doctrine_state() -> None:
    """Refresh doctrine account-dependent globals so exports reflect latest balances."""
    global ACCOUNT_BALANCES, WAR_ROOM_DOCTRINE
    ACCOUNT_BALANCES = _build_exchange_inventory()
    if isinstance(WAR_ROOM_DOCTRINE, dict):
        WAR_ROOM_DOCTRINE["capital_deployment"] = _build_capital_deployment()


# ═══════════════════════════════════════════════════════════════════════════
# LEAPS PLAYBOOK — $38K DEPLOYMENT (Moon 1 Pink Moon Entry)
# ═══════════════════════════════════════════════════════════════════════════

LEAPS_PLAYBOOK: Dict[str, Any] = {
    "total_capital": 38000,
    "conviction": "100% — dual-chokepoint black swan, acid/copper/cobalt choke, Japan yen unwind, "
                  "rates trap, private credit stress, April big-tech cracks → June super-fuck.",
    "strategy": "Buy and hold LEAPS. Ride the wave. Rotate only at major Fire Peak nodes.",
    "entry_window": "Monday March 30 / Tuesday April 1, 2026 (Pink Moon Fire Peak)",
    "accounts": "Split ~$10k-$12k across 7 accounts (IBKR, Moomoo, NDAX, Polymarket, MetaMask, EQ Bank, Wealthsimple)",
    "existing_book": "HOLD: XLE Jan 2027 $85 Call x26 (WS). GLD Mar 2027 $515 Call x1 (WS). "
                     "OWL Jan 2027 $5P x10 (Moomoo) + OWL Jun $8P x5 (WS). "
                     "Apr 17 puts ALL EXPIRED WORTHLESS — see ROLL_DISCIPLINE.",
    "positions": {
        "silver_leaps": {
            "allocation_pct": 55,
            "amount": 20900,
            "ticker": "SLV",
            "strike": "Jan 2027 65C or Dec 2026 70C",
            "contracts": "30-40",
            "premium_est": "$4.50-$7.00/contract",
            "thesis": "Dual-leverage: monetary fear + industrial pivot from acid/copper choke. "
                      "Biggest gap-jumper. Silver embodies Aquarius (innovation + monetary awakening).",
        },
        "oil_leaps": {
            "allocation_pct": 25,
            "amount": 9500,
            "ticker": "XLE",
            "strike": "Jan 2027 75C or Dec 2026 80C",
            "contracts": "20-28",
            "premium_est": "$3.00-$5.00/contract",
            "thesis": "Direct chokepoint exposure (Hormuz + Russian Baltic). "
                      "Complements existing XLE Jan 2027 $85 Call.",
        },
        "gold_leaps": {
            "allocation_pct": 10,
            "amount": 3800,
            "ticker": "GLD",
            "strike": "Jan 2027 410C or Dec 2026 415C",
            "contracts": "3-5",
            "premium_est": "$8.00-$12.00/contract",
            "thesis": "Safe-haven ballast. Gold at $414 already moving. "
                      "Saturn's 'real metal' meets Neptune's 'dream metal'.",
        },
        "credit_put_leaps": {
            "allocation_pct": 10,
            "amount": 3800,
            "ticker": "JNK/XLF",
            "strike": "Jan 2027 90P or 45P",
            "contracts": "8-12",
            "premium_est": "$3.00-$5.00/contract",
            "thesis": "Reinforcement on private credit/bank pain. "
                      "Neptune dissolves 'evergreen' redemption illusions.",
        },
    },
    "exit_rotation_dates": {
        "moon_3_blue_moon": {
            "date": "2026-05-31",
            "action": "Sell 50-70% of printed LEAPS (especially credit puts). "
                      "Reinvest into additional silver/oil LEAPS.",
        },
        "moon_4_solstice": {
            "date": "2026-06-21",
            "action": "Sell another 20-30% of remaining printed positions. "
                      "Full rebalance into silver/oil LEAPS. Target $1M-$3M range.",
        },
        "moon_7_equinox": {
            "date": "2026-09-22",
            "action": "Sell 40-60% of printed positions. Reinvest into silver/oil LEAPS.",
        },
        "moon_12_conjunction": {
            "date": "2027-02-20",
            "action": "Sell 50-70% of all printed LEAPS on reset visibility. "
                      "Final rotation into silver/gold for new 36-year cycle.",
        },
        "moon_13_milestone": {
            "date": "2027-03-21",
            "action": "Sell remaining if $10M+ achieved. Full review. "
                      "New 36-year cycle begins.",
        },
    },
    "risk_rules": [
        "Max loss = premium paid (~$38k if all expire worthless)",
        "Never sell on big jumps — ride the wave",
        "Conditional double-down on 8% breach only if prime movers strong",
        "Hold existing book (XLE call + credit puts) untouched",
        "If SLV/XLE gap up sharply pre-entry, shift strikes higher",
        "Enter limit orders at or better than mid-price",
    ],
    "daily_scrape_march_28": {
        "silver_spot": "$67.80-$68.55/oz (SLV $63.44, +4.39%)",
        "xle": "$62.56 (+1.69%)",
        "gld": "$414.70 (+3.51%)",
        "brent": "$108-$111",
        "wti": "$96-$98",
        "hormuz": "Traffic ~90% down. Iran 'safe-passage' tolls. Russian Baltic ports offline.",
        "obdc": "Trading at ~22-25% discount to NAV (~$14.81). Redemption pressures continue.",
        "thesis_status": "All prime movers intact and accelerating.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO DOCTRINE — Positions, Outlooks & Moon-Phase Alignment
# ═══════════════════════════════════════════════════════════════════════════

CRYPTO_DOCTRINE: Dict[str, Any] = {
    "thesis": "Crypto is a liquidity amplifier. It front-runs TradFi in both directions. "
              "Bitcoin leads, alts follow with leverage. The 13-Moon cycle captures crypto's "
              "reflexive nature: monetary fear (gold/silver) spills into BTC as 'digital gold', "
              "then cascades into alt-season. Saturn-Neptune conjunction (Moon 12) = institutional "
              "crypto adoption trigger. Neptune dissolves fiat illusion → digital assets rise.",
    "regime_current": "Accumulation → Reflexive Melt-Up (Moon 1-4), "
                      "Leverage Risk (Moon 5-7), Rotation (Moon 8-11), "
                      "Institutional Adoption (Moon 12-13)",
    "data_sources": {
        "primary": "CoinGecko Pro (API key active)",
        "exchanges": ["NDAX (CAD, LIQUIDATED)", "Binance (public endpoints)", "Kraken (reference)"],
        "on_chain": ["Polymarket (Polygon/CLOB)", "MetaMask (Polygon/Ethereum)"],
        "sentiment": "Alternative.me Fear & Greed Index",
    },
    "positions": {
        "ndax": {
            "status": "LIQUIDATED",
            "date": "2026-03-18",
            "proceeds_cad": 4492.04,
            "sold": ["XRP (all)", "ETH (all)"],
            "reason": "Capital redeployed to LEAPS. Crypto exposure via options (SLV/GLD proxy for "
                      "monetary metal thesis). Will re-enter when BTC dominance drops below 50% "
                      "and funding rates normalize.",
        },
        "polymarket": {
            "status": "ACTIVE",
            "chain": "Polygon (137)",
            "protocol": "CLOB (Central Limit Order Book)",
            "sig_type": 1,
            "funder": "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8",
            "currency": "USDC",
            "strategy": "Event-driven binary options on geopolitical + macro outcomes. "
                        "Prediction markets = information extraction. Complement War Room signals.",
        },
        "metamask": {
            "status": "MONITORING",
            "eoa": "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8",
            "chains": ["Polygon", "Ethereum"],
            "tokens": ["MATIC", "ETH", "USDC.e", "USDC"],
            "note": "Funder wallet. Min balances for gas. ETH mainnet ~$0.",
        },
    },
    "watchlist": {
        "btc": {
            "ticker": "BTC",
            "current_price": "~$87,000",
            "thesis": "Digital gold narrative strengthens with Saturn-Neptune. Store of value "
                      "confirmed if BTC/Gold ratio stabilizes above 20. Halving cycle (Apr 2024) "
                      "supply shock ongoing. Next target: $100k (Moon 3-4), $150k (Moon 7-9).",
            "moon_affinity": "Moon 5 (Sturgeon) — peak reflexive melt if liquidity expanding",
            "indicators": ["BTC dominance", "Funding rate", "Exchange inflows", "M2 global growth"],
            "sacred_geometry": "Metatron's Cube — 13 circles = 13 moons. BTC's finite supply "
                               "(21M) mirrors sacred geometric completeness.",
        },
        "eth": {
            "ticker": "ETH",
            "current_price": "~$2,050",
            "thesis": "DeFi + L2 scaling thesis. ETH underperforms BTC in risk-off but leads "
                      "in alt-season. Staking yield (~4%) provides floor. Neptune's 'dissolve old "
                      "structures' = DeFi eats TradFi. Target: $4k (Moon 4), $6k (Moon 8).",
            "moon_affinity": "Moon 4 (Strawberry) — Solstice rebalance, DeFi summer",
            "indicators": ["ETH/BTC ratio", "Gas fees", "TVL", "L2 adoption"],
            "sacred_geometry": "Flower of Life — Ethereum's interconnected smart contracts "
                               "mirror the Flower's overlapping circles of creation.",
        },
        "xrp": {
            "ticker": "XRP",
            "current_price": "~$2.30",
            "thesis": "Payment rail + institutional settlement. SEC clarity resolved. "
                      "NDAX position liquidated but watching for re-entry. Metal X (Proton/XPR) "
                      "integration = zero-fee DEX arbitrage. Target: $5 (Moon 6), $10 (Moon 10).",
            "moon_affinity": "Moon 6 (Corn) — Harvest preparation, institutional flows",
            "indicators": ["ODL volume", "XRPL on-demand liquidity", "Exchange listings"],
            "sacred_geometry": "Vesica Piscis — two-circle overlap = bridge between fiat and crypto.",
        },
        "sol": {
            "ticker": "SOL",
            "current_price": "~$138",
            "thesis": "High-throughput L1 competitor. Outperforms in risk-on. "
                      "Memecoin/DePIN narrative driver. Fragile in risk-off (leverage flush). "
                      "Target: $250 (Moon 4), $400 (Moon 8).",
            "moon_affinity": "Moon 3 (Flower) — Blue Moon expansion energy",
            "indicators": ["SOL/ETH ratio", "DEX volume", "Validator count"],
            "sacred_geometry": "Seed of Life — rapid growth from simple origins.",
        },
    },
    "strategies_active": {
        "crypto_arb": {
            "name": "NDAX Cross-Exchange Spread Scanner",
            "status": "PAUSED (NDAX liquidated)",
            "pairs": ["XRP/CAD", "BTC/CAD", "ETH/CAD", "SOL/CAD"],
            "edge_target_bps": 25,
            "script": "strategies/run_crypto_arb.py",
        },
        "metalx_arb": {
            "name": "Metal X ↔ CEX Arbitrage",
            "status": "MONITORING",
            "pairs": ["BTC/XMD", "ETH/XMD", "XPR/XMD"],
            "edge": "Zero-fee DEX vs CEX maker/taker fee asymmetry",
            "script": "strategies/metalx_arb_strategy.py",
        },
        "xmd_treasury": {
            "name": "XMD Stablecoin Treasury",
            "status": "MONITORING",
            "instrument": "XMD/USD peg arbitrage + lending yield",
            "script": "strategies/xmd_treasury_strategy.py",
        },
        "crypto_forecast": {
            "name": "Crypto Regime Forecaster",
            "status": "ACTIVE",
            "formulas": ["C1 Liquidity Melt", "C2 Leverage Flush", "C3 Decoupling",
                         "C4 Vol Trap", "C5 Exchange Inflow", "C6 BTC Dominance",
                         "C7 Funding Reversion", "C8 Correlation Break"],
            "script": "strategies/crypto_forecaster.py",
        },
    },
    "moon_phase_outlook": {
        0: {"regime": "Accumulation", "action": "Set watchlist alerts. CoinGecko Pro scanning.",
            "risk": "LOW", "note": "Seed Moon — plant crypto seeds. Fear & Greed check."},
        1: {"regime": "Entry Window", "action": "LEAPS entry (indirect crypto via SLV/GLD). "
            "Polymarket event bets on macro catalysts.",
            "risk": "MEDIUM", "note": "Worm Moon — capital deployed. Crypto on standby."},
        2: {"regime": "Reflexive Melt-Up", "action": "If BTC breaks ATH, consider spot accumulation. "
            "Run C1 formula. Check funding rates.",
            "risk": "MEDIUM", "note": "Pink Moon — liquidity expanding. Crypto amplifies."},
        3: {"regime": "Alt-Season Candidate", "action": "If BTC dominance dropping + ETH/BTC rising, "
            "small alt allocation. SOL/ETH ratio watch.",
            "risk": "MEDIUM-HIGH", "note": "Flower Moon — bloom energy. Alts flourish or flush."},
        4: {"regime": "DeFi Summer", "action": "ETH staking yield check. L2 adoption metrics. "
            "Solstice rebalance includes crypto allocation review.",
            "risk": "MEDIUM", "note": "Strawberry Moon — sweet yields. DeFi summer."},
        5: {"regime": "Peak Reflexive", "action": "Run C2 (Leverage Flush) check. If OI extreme + "
            "funding spiking, reduce any crypto exposure. VIX correlation check.",
            "risk": "HIGH", "note": "Sturgeon Moon — deep waters. Leverage risk peak."},
        6: {"regime": "Institutional Window", "action": "XRP ODL volume check. Metal X arb scan. "
            "Institutional settlement flows peak.",
            "risk": "MEDIUM", "note": "Corn Moon — harvest institutional flows."},
        7: {"regime": "Equinox Rebalance", "action": "Full crypto portfolio review. Run all C1-C8 "
            "formulas. Autumnal equinox = balance point. Dalio cycle check.",
            "risk": "MEDIUM", "note": "Harvest Moon — reap what was sown."},
        8: {"regime": "Risk-Off Watch", "action": "BTC dominance expanding = alt liquidation incoming. "
            "Run C6. Tighten stops. Exchange inflow monitoring.",
            "risk": "HIGH", "note": "Hunter's Moon — hunt for exits. Protect gains."},
        9: {"regime": "Vol Compression", "action": "C4 check. IV crushed → violent move incoming. "
            "Prepare for direction. Options cheap.",
            "risk": "MEDIUM", "note": "Beaver Moon — build positions for next move."},
        10: {"regime": "Correlation Watch", "action": "C8 (Correlation Breakout). If BTC/SPX "
             "decorrelating, crypto may run independently. DXY inverse check.",
             "risk": "MEDIUM", "note": "Cold Moon — markets cool. Decorrelation = opportunity."},
        11: {"regime": "Pre-Conjunction Prep", "action": "Position for Saturn-Neptune. If thesis "
             "intact, build BTC/ETH spot positions for institutional adoption wave.",
             "risk": "MEDIUM", "note": "Wolf Moon — pack formation. Smart money accumulating."},
        12: {"regime": "Institutional Adoption", "action": "Saturn-Neptune conjunction. "
             "Neptune dissolves fiat illusion → digital assets as new monetary layer. "
             "BTC as reserve asset narrative peaks. ETF flows watch.",
             "risk": "CRITICAL", "note": "Snow Moon — crystallization. Old fiat structure melts."},
        13: {"regime": "New Cycle", "action": "Full crypto review. If $10M+ portfolio, allocate "
             "5-10% to crypto (BTC 60%, ETH 25%, SOL/XRP 15%). New 36-year cycle.",
             "risk": "LOW", "note": "Worm Moon rebirth — new monetary paradigm emerges."},
    },
    "risk_framework": {
        "max_crypto_allocation": "10% of total portfolio (currently 0% — LEAPS-focused)",
        "re_entry_triggers": [
            "BTC dominance < 50% + Fear & Greed > 30",
            "Funding rates normalized (< 0.01% / 8h)",
            "M2 global growth positive for 2+ quarters",
            "VIX < 25 sustained (risk-on environment)",
        ],
        "stop_loss": "20% drawdown from entry on any crypto position",
        "correlation_alert": "If BTC/SPX 30-day correlation > 0.7, reduce crypto to 5% max",
    },
}


# ── WAR ROOM DOCTRINE ────────────────────────────────────────────────
# Unified view across war_room_engine.py (statistical) and ninety_day_war_room.py (operational).
# Compose, don't merge — both engines remain separate; this structures the storyboard overlay.
WAR_ROOM_DOCTRINE: Dict[str, Any] = {
    "thesis": "US-Iran conflict -> oil supply shock -> credit cascade -> gold reprice -> USD transition",
    "war_start": "2026-02-28",
    "model_window": {"start": "2025-12-19", "end": "2026-06-17", "days": 180},
    "engines": {
        "statistical": "strategies/war_room_engine.py (2,188 lines) -- MC simulation, Greeks, 50 milestones, 15-indicator composite",
        "operational": "strategies/ninety_day_war_room.py (1,643 lines) -- day-by-day model, macro calendar, capital deployment, intel logging",
    },
    # 15-indicator composite score (from war_room_engine)
    "composite_score": {
        "current": 57.7,
        "regime": "ELEVATED",
        "indicators": 15,
        "top_weights": [
            {"name": "Oil Price", "weight": 0.12, "current": 100.0},
            {"name": "X/Twitter Sentiment", "weight": 0.12, "current": 0.5},
            {"name": "VIX", "weight": 0.10, "current": 31.0},
            {"name": "Gold Price", "weight": 0.08, "current": 4524.0},
            {"name": "HY Spread", "weight": 0.08, "current": 600},
            {"name": "SPY Price", "weight": 0.07, "current": 634.0},
            {"name": "BDC NAV Discount", "weight": 0.07, "current": 16.0},
        ],
        "regimes": {
            "CALM": "Score <= 30 -- minimal positioning",
            "WATCH": "Score 30-50 -- half-size positions",
            "ELEVATED": "Score 50-70 -- target allocations active",
            "CRISIS": "Score > 70 -- maximum aggression, all arms deployed",
        },
    },
    # 5-arm allocation system (from war_room_engine)
    "five_arms": {
        "iran_oil": {"target_pct": 30, "max_pct": 40, "instruments": "SPY/QQQ puts + XLE/USO calls", "status": "ACTIVE"},
        "bdc_credit": {"target_pct": 25, "max_pct": 35, "instruments": "FSK/TCPC/HYG/BX/BKLN puts", "status": "ACTIVE"},
        "crypto_metals": {"target_pct": 20, "max_pct": 30, "instruments": "GLD/SLV calls + BITO puts + GDX", "status": "ACTIVE"},
        "defi_yield": {"target_pct": 15, "max_pct": 25, "instruments": "AAVE/UNI puts + DeFi shorts", "status": "WATCHING"},
        "tradfi_rotate": {"target_pct": 10, "max_pct": 20, "instruments": "Treasuries + dividends + income", "status": "SEEDING"},
    },
    # 50-milestone spiderweb (from war_room_engine)
    "milestones": {
        "total": 50,
        "categories": {
            "dollar": {"count": 10, "example": "First $50K -> $100M endowment"},
            "oil": {"count": 6, "example": "Oil $105 -> $150 superspike"},
            "gold": {"count": 5, "example": "Gold $4800 -> $5500 blowoff"},
            "credit": {"count": 5, "example": "HY 500bp -> PE revenue drop 20%"},
            "crypto": {"count": 5, "example": "BTC <$60K -> stablecoin depeg"},
            "macro": {"count": 5, "example": "VIX 30 -> Fed emergency cut"},
            "geopolitical": {"count": 5, "example": "Hormuz -> BRICS gold settlement"},
            "equity": {"count": 4, "example": "SPY <520 -> XLF <42"},
            "phase": {"count": 5, "example": "First profitable week -> 90 days running"},
        },
        "key_gates": [
            {"id": 1, "name": "First $50K", "confidence": "85%", "phase": "accumulation"},
            {"id": 7, "name": "Millionaire Gate $1M", "confidence": "10%", "phase": "rotation"},
            {"id": 12, "name": "Oil Spike $120", "confidence": "45%", "phase": "accumulation"},
            {"id": 17, "name": "Gold Breaks $4800", "confidence": "55%", "phase": "accumulation"},
            {"id": 22, "name": "HY Spread 500bp", "confidence": "45%", "phase": "accumulation"},
            {"id": 32, "name": "VIX Breaks 30", "confidence": "50%", "phase": "accumulation"},
            {"id": 37, "name": "Hormuz Disruption", "confidence": "35%", "phase": "accumulation"},
        ],
    },
    # Monte Carlo engine (from war_room_engine)
    "monte_carlo": {
        "paths": 100000,
        "horizon_days": 90,
        "assets": "oil, gold, silver, gdx, spy, qqq, xlf, xlre, eth, xrp, btc",
        "method": "Multivariate GBM + Cholesky correlation",
        "starting_capital": "$45,120 CAD (~$31,584 USD)",
    },
    # Phase thresholds (from war_room_engine)
    "phases": {
        "accumulation": {"range": "$0-$150K", "strategy": "Max options, aggressive puts, build 5 arms"},
        "growth": {"range": "$150K-$1M", "strategy": "Reduce puts, scale metals, add income"},
        "rotation": {"range": "$1M-$5M", "strategy": "Income mode, max 15% options, 40% fixed income"},
        "preservation": {"range": "$5M-$100M", "strategy": "Endowment mode, 5% options, 60% yield"},
    },
    # 4 scenario tracks (from ninety_day_war_room)
    "scenario_tracks": {
        "fails": {"probability": "45%", "oil": "$65", "gold": "$2,800", "spy": "$610", "outcome": "Ceasefire, recovery"},
        "moderate": {"probability": "30%", "oil": "$120", "gold": "$3,500", "spy": "$490", "outcome": "Partial thesis, solid returns"},
        "major": {"probability": "20%", "oil": "$180", "gold": "$5,000", "spy": "$420", "outcome": "Full crisis, major payoff"},
        "blackswan": {"probability": "5%", "oil": "$220", "gold": "$8,000", "spy": "$350", "outcome": "Multi-front, life-changing"},
    },
    # Capital deployment (from ninety_day_war_room)
    "capital_deployment": _build_capital_deployment(),
    # Correlation highlights (strongest from both engines)
    "correlations": [
        {"pair": "VIX <-> SPY", "value": -0.92, "meaning": "VIX spikes = SPY tanks (near certain)"},
        {"pair": "VIX <-> HY Spread", "value": 0.85, "meaning": "Vol confirms credit stress"},
        {"pair": "HY Spread <-> SPY", "value": -0.80, "meaning": "Credit dominoes hit equities"},
        {"pair": "Gold <-> DXY", "value": -0.80, "meaning": "Gold up = dollar down"},
        {"pair": "Oil <-> HY Spread", "value": 0.78, "meaning": "Oil shock = credit stress"},
        {"pair": "Oil <-> VIX", "value": 0.72, "meaning": "Oil shock = vol spike"},
    ],
    # Risk framework
    "risk_framework": {
        "GREEN": "Pressure <30, VIX <20. Monitor only.",
        "YELLOW": "Pressure <50, VIX <25. Half-size positions.",
        "ORANGE": "Pressure 45-65, VIX 25-35. Target size active.",
        "RED": "Pressure 65-80, VIX 35-50. Full arms. Trailing stops 30%.",
        "BLACK": "Pressure >80, VIX >50. Max aggression, watch reversal.",
        "max_drawdown": "25% drawdown -> cut positions 50%, review thesis",
        "profit_taking": "500%+ gain -> take 50% off table",
    },
    # 13-week strategic roadmap (from ninety_day_war_room)
    "roadmap": [
        {"week": 1, "label": "FOUNDATION", "dates": "Mar 19-25", "focus": "Deploy injection, fill oil+gold gaps, daily pressure scans"},
        {"week": 2, "label": "EXPAND", "dates": "Mar 26-Apr 1", "focus": "Add oil/gold verticals (CGL.TO TFSA), Moomoo options approval, War Day 30+"},
        {"week": 3, "label": "CONFIRM", "dates": "Apr 2-8", "focus": "First war-month data (jobs, CPI). OPEC meeting. Confirm or deny thesis."},
        {"week": 4, "label": "EARNINGS", "dates": "Apr 9-15", "focus": "Bank earnings credit losses. Deploy $5K. Tax day selling pressure."},
        {"week": 5, "label": "TECH EARNINGS", "dates": "Apr 16-22", "focus": "Tech earnings + war impact. QQQ puts if disappoint. 45-day checkpoint."},
        {"week": 6, "label": "GDP WEEK", "dates": "Apr 23-29", "focus": "Q1 GDP advance -- CRITICAL. If <1% -> thesis acceleration. If negative -> panic."},
        {"week": 7, "label": "2 MONTHS", "dates": "Apr 30-May 6", "focus": "Fed May meeting. 2 months of war. Partial profits on 500%+ winners."},
        {"week": 8, "label": "INFLATION", "dates": "May 7-13", "focus": "2nd war-month CPI. If >5% headline -> stagflation narrative dominates."},
        {"week": 9, "label": "ENDGAME PREP", "dates": "May 14-20", "focus": "Day 80. Begin hard asset transition if thesis >70%."},
        {"week": 10, "label": "3 MONTHS", "dates": "May 21-27", "focus": "Quarter-point. Ceasefire or entrenchment. Final Phase 3 injection."},
        {"week": 11, "label": "STRUCTURAL", "dates": "May 28-Jun 3", "focus": "Long-dated gold, dollar shorts, real assets. Event-driven -> structural."},
        {"week": 12, "label": "HARVEST", "dates": "Jun 4-10", "focus": "Q2 data lands. 3rd war-month CPI. Systematic profit-taking."},
        {"week": 13, "label": "REGROUP", "dates": "Jun 11-17", "focus": "Fed June meeting. Full 90-day evaluation. Plan next 90 days or endgame."},
    ],
    # Per-moon war room overlay
    "per_moon_outlook": {
        0: {"regime": "PRECURSOR", "pressure": "8-15%", "risk": "GREEN", "action": "Pre-war. Build watchlist. Monitor Iran tensions. No positions yet."},
        1: {"regime": "BUILDING", "pressure": "15-35%", "risk": "YELLOW", "action": "War started Feb 28. Initial shock wave. Deploy first puts on credit/equity."},
        2: {"regime": "BUILDING", "pressure": "35-44%", "risk": "YELLOW", "action": "8 puts LIVE on IBKR. Hormuz partial closure. NDAX liquidated. Escalation phase."},
        3: {"regime": "ACCELERATION", "pressure": "44-55%", "risk": "ORANGE", "action": "$19.8K injection landing. Scale confirmed vectors. Bank earnings season."},
        4: {"regime": "ACCELERATION", "pressure": "50-65%", "risk": "ORANGE", "action": "GDP Q1 data. Fed May. War month 2 complete. Full portfolio review."},
        5: {"regime": "CRISIS_ONSET", "pressure": "55-75%", "risk": "RED", "action": "Day 80+. Endgame window. Hard asset transition begins if >70%."},
        6: {"regime": "FULL_CRISIS OR RESOLUTION", "pressure": "varies", "risk": "RED", "action": "90-day endpoint. Full thesis evaluation. Fed June. Next cycle planning."},
        7: {"regime": "POST-MODEL", "pressure": "varies", "risk": "varies", "action": "Beyond initial 90d model. Structural positions or wind-down based on outcome."},
        8: {"regime": "ROTATION", "pressure": "declining", "risk": "YELLOW", "action": "If thesis confirmed: rotate options to income + hard assets."},
        9: {"regime": "ROTATION", "pressure": "declining", "risk": "YELLOW", "action": "Preservation seeds. Long-dated gold, dollar positions."},
        10: {"regime": "PRESERVATION", "pressure": "low", "risk": "GREEN", "action": "Income mode if >$150K. Treasury + dividend focus."},
        11: {"regime": "PRESERVATION", "pressure": "low", "risk": "GREEN", "action": "Annual rebalance. Saturn-Neptune window approaching."},
        12: {"regime": "REVIEW", "pressure": "varies", "risk": "GREEN", "action": "Full year review. Next cycle thesis development."},
        13: {"regime": "RESET", "pressure": "baseline", "risk": "GREEN", "action": "New 13-Moon cycle. Lessons integrated. Fresh mandate."},
    },
    # Key macro calendar events (from ninety_day_war_room)
    "macro_calendar_highlights": [
        {"date": "2026-02-28", "event": "US-IRAN WAR BEGINS", "impact": "CRITICAL"},
        {"date": "2026-03-12", "event": "Hormuz Partial Closure", "impact": "CRITICAL"},
        {"date": "2026-03-18", "event": "8 Live Puts Deployed + NDAX Liquidated", "impact": "CRITICAL"},
        {"date": "2026-03-19", "event": "FOMC March + War Day 19", "impact": "CRITICAL"},
        {"date": "2026-04-10", "event": "CPI March (first war-month CPI)", "impact": "CRITICAL"},
        {"date": "2026-04-11", "event": "Q1 Bank Earnings Begin", "impact": "CRITICAL"},
        {"date": "2026-04-14", "event": "War Day 45 -- Critical Checkpoint", "impact": "CRITICAL"},
        {"date": "2026-04-30", "event": "GDP Q1 Advance (recession signal?)", "impact": "CRITICAL"},
        {"date": "2026-05-06", "event": "FOMC May (emergency cut?)", "impact": "CRITICAL"},
        {"date": "2026-05-19", "event": "War Day 80 -- Endgame Window", "impact": "CRITICAL"},
        {"date": "2026-06-16", "event": "FOMC June (3 months into war)", "impact": "CRITICAL"},
        {"date": "2026-06-17", "event": "War Day 109 -- 90-Day Model Endpoint", "impact": "CRITICAL"},
    ],
}


# ── INDICATOR DOCTRINE ────────────────────────────────────────────────
# Per-indicator threshold-based triggers, conviction levels, and moon-phase alignment.
INDICATOR_DOCTRINE: Dict[str, Any] = {
    "gold": {
        "current_spot": 4524,
        "thresholds": [
            {"level": 2500, "label": "Stagflation Confirmed", "action": "Accelerate GLD/GDX accumulation",
             "conviction": 0.8, "moon_affinity": "Moon 2 (Pink Moon)"},
            {"level": 3000, "label": "Parabolic Entry", "action": "GLD LEAPS 500%+ return zone",
             "conviction": 0.9, "moon_affinity": "Moon 5 (Sturgeon Moon)"},
            {"level": 3500, "label": "Central Bank Capitulation", "action": "Begin profit rotation to silver",
             "conviction": 0.95, "moon_affinity": "Moon 8 (Harvest Moon)"},
            {"level": 5000, "label": "Monetary Reset Signal", "action": "Full rotation. Legacy capital preservation",
             "conviction": 1.0, "moon_affinity": "Moon 12 (Snow Moon)"},
        ],
        "seesaw": "Inverse DXY. Positive oil/silver correlation. Credit stress accelerator.",
        "sacred_geometry": "Golden ratio ($2618, $4236) = Fibonacci extension targets.",
    },
    "oil": {
        "current_spot": 100,
        "thresholds": [
            {"level": 85, "label": "Supply Disruption Base", "action": "XLE calls active. Oil thesis confirmed",
             "conviction": 0.8, "moon_affinity": "Moon 1 (Worm Moon)"},
            {"level": 100, "label": "Hormuz Premium", "action": "XLE calls deep ITM. Scale profits",
             "conviction": 0.85, "moon_affinity": "Moon 3 (Flower Moon)"},
            {"level": 120, "label": "Crisis Premium", "action": "Energy sector parabolic. Consumer stress",
             "conviction": 0.9, "moon_affinity": "Moon 6 (Corn Moon)"},
            {"level": 150, "label": "Demand Destruction", "action": "Take XLE profits. Recession incoming",
             "conviction": 0.95, "moon_affinity": "Moon 9 (Hunter's Moon)"},
        ],
        "seesaw": "Positive gold/silver. Negative SPY. Credit stress lagger.",
        "sacred_geometry": "Vesica Piscis (75-150 range = creative destruction zone).",
    },
    "silver": {
        "current_spot": 65,
        "thresholds": [
            {"level": 30, "label": "Industrial Demand Floor", "action": "Accumulate SLV calls aggressively",
             "conviction": 0.8, "moon_affinity": "Moon 1 (Worm Moon)"},
            {"level": 35, "label": "Breakout Confirmed", "action": "SLV LEAPS printing. Add positions",
             "conviction": 0.85, "moon_affinity": "Moon 3 (Flower Moon)"},
            {"level": 50, "label": "Silver Squeeze 2.0", "action": "WSB momentum. Parabolic phase",
             "conviction": 0.9, "moon_affinity": "Moon 7 (Harvest Moon)"},
            {"level": 80, "label": "Monetary Metal Status", "action": "Silver=monetary asset. Peak profits",
             "conviction": 0.95, "moon_affinity": "Moon 11 (Wolf Moon)"},
            {"level": 100, "label": "Paradigm Shift", "action": "Silver gold ratio normalizing to 30:1",
             "conviction": 1.0, "moon_affinity": "Moon 12 (Snow Moon)"},
        ],
        "seesaw": "Positive gold/oil. Industrial + monetary dual demand. Lead indicator.",
        "sacred_geometry": "Seed of Life (6-fold: $30, $35, $50, $65, $80, $100).",
    },
    "vix": {
        "current_spot": 22,
        "thresholds": [
            {"level": 20, "label": "Complacency", "action": "Low vol = cheap hedges. Buy puts",
             "conviction": 0.6, "moon_affinity": "Any"},
            {"level": 25, "label": "ELEVATED Regime", "action": "Increase hedge allocation 20→30%",
             "conviction": 0.75, "moon_affinity": "Moon 2 (Pink Moon)"},
            {"level": 35, "label": "CRITICAL Regime", "action": "All seesaw amplifiers activate. Max hedge",
             "conviction": 0.9, "moon_affinity": "Moon 5 (Sturgeon Moon)"},
            {"level": 40, "label": "CRISIS Mode", "action": "War Room DEFCON 1. All puts printing",
             "conviction": 0.95, "moon_affinity": "Moon 8 (Harvest Moon)"},
            {"level": 50, "label": "Systemic Panic", "action": "Begin rotating puts to assets. Counter-trend buys",
             "conviction": 1.0, "moon_affinity": "Moon 10 (Beaver Moon)"},
        ],
        "seesaw": "Inverse SPY/QQQ. Leading indicator for credit spreads.",
        "sacred_geometry": "Fibonacci retracements of prior VIX spikes as targets.",
    },
    "spy": {
        "current_spot": 634,
        "thresholds": [
            {"level": 600, "label": "5% Correction", "action": "IWM puts printing. Monitor breadth",
             "conviction": 0.7, "moon_affinity": "Moon 3 (Flower Moon)"},
            {"level": 550, "label": "10% Correction", "action": "Bear market approaching. Add puts",
             "conviction": 0.8, "moon_affinity": "Moon 5 (Sturgeon Moon)"},
            {"level": 500, "label": "Bear Market", "action": "20% drawdown. Full thesis validation",
             "conviction": 0.9, "moon_affinity": "Moon 8 (Harvest Moon)"},
            {"level": 450, "label": "Panic Crash", "action": "Begin counter-trend longs. Covered calls on puts",
             "conviction": 0.95, "moon_affinity": "Moon 10 (Beaver Moon)"},
        ],
        "seesaw": "Inverse VIX/gold. Inverse HY OAS. Key regime indicator.",
        "sacred_geometry": "Golden ratio retracements from ATH as support levels.",
    },
    "dxy": {
        "current_spot": 104,
        "thresholds": [
            {"level": 108, "label": "Dollar Strength", "action": "EM stress. Gold headwind. Patience",
             "conviction": 0.7, "moon_affinity": "Moon 2 (Pink Moon)"},
            {"level": 110, "label": "Wrecking Ball", "action": "EM crisis. Yen carry unwind. Commodity pressure",
             "conviction": 0.85, "moon_affinity": "Moon 5 (Sturgeon Moon)"},
            {"level": 95, "label": "Dollar Weakness", "action": "Gold/silver parabolic. Commodities surge",
             "conviction": 0.85, "moon_affinity": "Moon 8 (Harvest Moon)"},
            {"level": 90, "label": "De-dollarization", "action": "BRICS thesis confirmed. Gold new reserve",
             "conviction": 0.95, "moon_affinity": "Moon 12 (Snow Moon)"},
        ],
        "seesaw": "Inverse gold/silver/oil. Positive US equities (paradoxically). BRICS inverse.",
        "sacred_geometry": "100 = unity point. Deviations = Flower of Life petal extremes.",
    },
    "hy_oas": {
        "current_spot": 380,
        "thresholds": [
            {"level": 400, "label": "Stress Building", "action": "JNK puts accumulate. BDC caution",
             "conviction": 0.7, "moon_affinity": "Moon 3 (Flower Moon)"},
            {"level": 500, "label": "Credit Stress", "action": "JNK/HYG puts deep ITM. BDC NAV pressure",
             "conviction": 0.85, "moon_affinity": "Moon 6 (Corn Moon)"},
            {"level": 600, "label": "Credit Crisis", "action": "Full credit dislocation. ARCC/MAIN earnings stress",
             "conviction": 0.9, "moon_affinity": "Moon 8 (Harvest Moon)"},
            {"level": 800, "label": "2008-level Spreads", "action": "Prepare for regime change. Counter-trend credit",
             "conviction": 0.95, "moon_affinity": "Moon 11 (Wolf Moon)"},
        ],
        "seesaw": "Positive VIX. Inverse SPY/IWM. Leading indicator for equity crash.",
        "sacred_geometry": "Sacred geometry of credit cycles: compression→expansion→crisis→reset.",
    },
}


# ── RISK MANAGEMENT DOCTRINE ─────────────────────────────────────────
# Per-capital-phase risk rules: position sizing, max drawdown, hedging, Greek targets.
RISK_MANAGEMENT_DOCTRINE: Dict[str, Any] = {
    "phases": {
        "seed_capital": {
            "portfolio_range": "$5K — $25K",
            "position_sizing": "1-2% per trade. Max $500 per position.",
            "max_drawdown": "30% of portfolio ($1.5K — $7.5K).",
            "hedge_ratio": "20% of capital in protective puts.",
            "greek_targets": {
                "delta": "Portfolio delta -0.10 to -0.30 (short bias via puts)",
                "gamma": "Positive gamma preferred. Long options only.",
                "theta": "Negative theta acceptable (long premium). Max -$50/day.",
                "vega": "Long vega. Benefit from vol expansion.",
            },
            "moon_phase": "Moon 0–2 (Worm → Pink)",
            "rules": [
                "No margin. Cash-secured only.",
                "Max 8-10 positions total.",
                "No single position >5% of portfolio.",
                "Close losers at 50% loss. Let winners run.",
                "Weekly War Room review mandatory.",
            ],
        },
        "growth_capital": {
            "portfolio_range": "$25K — $100K",
            "position_sizing": "2-3% per trade. Max $2K per position.",
            "max_drawdown": "20% of portfolio ($5K — $20K).",
            "hedge_ratio": "25% of capital in hedges (puts + VIX calls).",
            "greek_targets": {
                "delta": "Portfolio delta -0.20 to -0.50 (stronger short bias)",
                "gamma": "Positive gamma. Add gamma via weekly options.",
                "theta": "Max -$150/day. Roll positions at 50% profit.",
                "vega": "Long vega up to composite score 80. Then reduce.",
            },
            "moon_phase": "Moon 3–6 (Flower → Corn)",
            "rules": [
                "Consider limited margin for defined-risk spreads.",
                "Diversify across 12-15 positions minimum.",
                "Sector concentration max 30% per sector.",
                "Bi-weekly War Room review with P&L attribution.",
                "Begin LEAPS accumulation on dips.",
            ],
        },
        "acceleration_capital": {
            "portfolio_range": "$100K — $500K",
            "position_sizing": "1-2% per trade (larger base). Max $5K per position.",
            "max_drawdown": "15% of portfolio ($15K — $75K).",
            "hedge_ratio": "30% of capital in hedges.",
            "greek_targets": {
                "delta": "Portfolio delta -0.30 to -0.60 in CRITICAL regime",
                "gamma": "Balanced. Long gamma via LEAPS, short gamma via spreads.",
                "theta": "Neutral to positive. Sell premium against LEAPS.",
                "vega": "Reduce long vega as thesis matures. Sell vol on spikes.",
            },
            "moon_phase": "Moon 7–9 (Harvest → Hunter's)",
            "rules": [
                "Multi-broker: IBKR + Moomoo + additional.",
                "Tax-loss harvesting on losers quarterly.",
                "Professional tax consultation.",
                "Daily War Room with automated alerts.",
                "Begin exit planning for peak positions.",
            ],
        },
        "preservation_capital": {
            "portfolio_range": "$500K — $10M+",
            "position_sizing": "0.5-1% per trade. Max $10K per position.",
            "max_drawdown": "10% of portfolio ($50K — $1M). HARD STOP.",
            "hedge_ratio": "40% of capital in hedges and cash.",
            "greek_targets": {
                "delta": "Portfolio delta near 0 (market neutral)",
                "gamma": "Low gamma. Defined risk only.",
                "theta": "Positive theta. Income generation from covered writes.",
                "vega": "Short vega where possible. Sell vol.",
            },
            "moon_phase": "Moon 10–13 (Beaver → Worm rebirth)",
            "rules": [
                "Saturn-Neptune conjunction approaching. Rotate to preservation.",
                "Max 50% equity exposure. 20% precious metals. 30% cash/bonds.",
                "Estate planning and asset protection.",
                "Monthly War Room with external advisor review.",
                "New 36-year cycle positioning begins.",
            ],
        },
    },
    "universal_rules": [
        "NEVER risk more than 5% of portfolio on a single idea.",
        "ALWAYS have defined exits BEFORE entering a trade.",
        "Thesis invalidation = immediate close. No averaging losers.",
        "Composite score >80 = max conviction. <40 = reduce exposure.",
        "Options expiry = NEVER hold to expiry unless deep ITM and exercising.",
        "Seesaw amplifiers activate above Composite 60.",
        "Sacred geometry coherence bonus: +5% position size if moon aligns.",
        "War Room regime overrides individual trade decisions.",
        "IBKR margin call = liquidate smallest losers first.",
        "Record EVERY trade with timestamp, thesis, and conviction level.",
        # ── Apr 6 2026 Post-Mortem Rules (8 dead puts lesson) ──
        "MAX 20 contracts per position. OBDC x65 was untradeable at $0 bid.",
        "Roll trigger at 21 DTE, NOT 7. By 7 DTE theta has already killed value.",
        "Max 5% OTM for short-dated puts (≤3 months). Deep OTM = lotto tickets.",
        "Dead-put gate: if STC bid = $0, do NOT roll. Re-evaluate thesis first.",
        "Allocate 70% LEAPS / 30% directional puts. LEAPS compound, short puts decay.",
    ],
}


def _build_aac_events() -> List[AACEvent]:
    """All AAC system events: live trades, war room, scenarios, milestones, options lifecycle, seesaw, strategies."""
    events: List[AACEvent] = []

    # ── LIVE TRADES EXECUTED ──────────────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 18), "LIVE TRADING ACTIVATED", "trade", "system",
                 "CRITICAL", "DRY_RUN=false, PAPER_TRADING=false, LIVE_TRADING_ENABLED=true. 8 puts deployed on IBKR.",
                 ["ARCC", "PFF", "LQD", "EMB", "MAIN", "JNK", "KRE", "IWM"], 0.9,
                 "Black-swan thesis activation. First real capital at risk."),
        AACEvent(date(2026, 3, 18), "ARCC $17P @ $0.25", "trade", "live_trade",
                 "HIGH", "Private credit put. ARCC stress if credit contagion.", ["ARCC"], 0.7,
                 "Private credit exposure — BDC distress."),
        AACEvent(date(2026, 3, 18), "PFF $29P @ $0.40", "trade", "live_trade",
                 "HIGH", "Preferred stock ETF put. Rate sensitivity.", ["PFF"], 0.7,
                 "Fixed-income stress — preferred shares blowout."),
        AACEvent(date(2026, 3, 18), "LQD $106P @ $0.64", "trade", "live_trade",
                 "HIGH", "Investment-grade bond ETF put. Credit spreads widening.", ["LQD"], 0.8,
                 "IG credit stress — spread widening thesis."),
        AACEvent(date(2026, 3, 18), "EMB $90P @ $0.75", "trade", "live_trade",
                 "HIGH", "EM bond ETF put. Dollar strength + oil shock.", ["EMB"], 0.75,
                 "EM debt crisis — oil/dollar dual squeeze."),
        AACEvent(date(2026, 3, 18), "MAIN $50P @ $0.85", "trade", "live_trade",
                 "HIGH", "BDC put. Main Street Capital credit exposure.", ["MAIN"], 0.7,
                 "Middle-market lending stress."),
        AACEvent(date(2026, 3, 18), "JNK $92P @ $0.80", "trade", "live_trade",
                 "HIGH", "High-yield bond ETF put. Junk spread blowout.", ["JNK"], 0.85,
                 "HY credit crisis — contagion from IG."),
        AACEvent(date(2026, 3, 18), "KRE $58P @ $1.45", "trade", "live_trade",
                 "HIGH", "Regional bank ETF put. SVB-style contagion.", ["KRE"], 0.85,
                 "Regional bank stress — CRE + rate exposure."),
        AACEvent(date(2026, 3, 18), "IWM $230P @ $3.96", "trade", "live_trade",
                 "HIGH", "Russell 2000 put. Small-cap recession proxy.", ["IWM"], 0.8,
                 "Broad equity recession — small-cap most vulnerable."),
        AACEvent(date(2026, 3, 18), "NDAX Liquidated: $4,492 CAD", "trade", "liquidation",
                 "HIGH", "Sold all XRP + ETH on NDAX. Cash freed for redeployment.",
                 ["XRP", "ETH"], 0.0, "De-risk crypto. Reallocate to options thesis."),
    ])

    # ── APR 6 2026: POSITION POST-MORTEM & LESSONS ───────────────────────
    events.extend([
        AACEvent(date(2026, 4, 6), "8 PUTS EXPIRED WORTHLESS — POST-MORTEM", "trade", "expiry_loss",
                 "CRITICAL", "All Apr 17 puts confirmed $0 bid at 11 DTE. IBKR: ARCC/PFF/MAIN/JNK. "
                 "WS: ARCC x10/JNK x5/KRE x1/OBDC x65. Total premium lost: ~$1,850. "
                 "OBDC x65 was the worst — untradeable size, $975 loss on one position.",
                 ["ARCC", "PFF", "MAIN", "JNK", "KRE", "OBDC"], 0.0,
                 "Hard lesson: position sizing, roll timing, and strike selection all failed."),
        AACEvent(date(2026, 4, 6), "ROLL DISCIPLINE RULES ENCODED", "trade", "policy_change",
                 "HIGH", "5 new rules from post-mortem: max 20 contracts, roll at 21 DTE, "
                 "max 5% OTM for short-dated, dead-put gate ($0 bid = no roll), "
                 "70/30 LEAPS vs puts allocation. Encoded in war_room_engine.ROLL_DISCIPLINE.",
                 [], 0.95,
                 "Institutional memory. Never repeat the OBDC x65 mistake."),
        AACEvent(date(2026, 4, 6), "COGNITIVE ARCHITECTURE CONFIRMED", "trade", "system",
                 "MEDIUM", "AAC validated as cognitive architecture (not agentic). "
                 "Zero LLM in trading path. Structured data flow, explicit state, deterministic scoring. "
                 "Architecture > Agents.",
                 [], 0.9,
                 "System design philosophy locked in."),
    ])

    # ── OPTIONS LIFECYCLE (Active positions after Apr 6 cleanup) ─────────
    events.extend([
        AACEvent(date(2026, 4, 6), "Apr 17 Puts: EXPIRED WORTHLESS", "options_lifecycle", "expiry",
                 "HIGH", "IBKR: ARCC/PFF/MAIN/JNK. WS: ARCC x10/JNK x5/KRE x1/OBDC x65. "
                 "All $0 bid at 11 DTE. Let expire. Premium lost ~$1,850.",
                 ["ARCC", "PFF", "MAIN", "JNK", "KRE", "OBDC"], 0.0,
                 "Dead puts. Lesson encoded in ROLL_DISCIPLINE."),
        AACEvent(date(2026, 4, 20), "DTE 25: XLF May 1 Roll Decision", "options_lifecycle", "roll_window",
                 "HIGH", "XLF $46P May 1 approaching. Roll at 21 DTE per new rules.",
                 ["XLF"], 0.6,
                 "First test of new 21-DTE roll discipline."),
        AACEvent(date(2026, 4, 20), "DTE 59: LQD/EMB May 15 Check", "options_lifecycle", "dte_check",
                 "MEDIUM", "LQD $106P and EMB $90P. ~60 DTE. Monitor credit spreads.",
                 ["LQD", "EMB"], 0.5,
                 "May expiry positions — still have time value."),
        AACEvent(date(2026, 5, 6), "DTE 45: Jun Puts Theta Acceleration", "options_lifecycle", "theta_accel",
                 "HIGH", "Theta enters hyperbolic zone for BKLN/HYG/OWL Jun puts. Close winners > 50% profit.",
                 ["BKLN", "HYG", "OWL"], 0.7,
                 "Critical theta inflection — Jun 18 expiry positions."),
        AACEvent(date(2026, 5, 20), "DTE 31: Jun Puts One Month Out", "options_lifecycle", "dte_check",
                 "HIGH", "30 DTE checkpoint for BKLN/HYG/OWL. Roll or hold per ROLL_DISCIPLINE.",
                 ["BKLN", "HYG", "OWL"], 0.75,
                 "One month to Jun expiry — conviction test."),
        AACEvent(date(2026, 5, 28), "DTE 21: Jun Puts ROLL TRIGGER", "options_lifecycle", "management",
                 "CRITICAL", "21 DTE = mandatory roll evaluation per new rules. Check bid. If $0 → dead-put gate.",
                 ["BKLN", "HYG", "OWL"], 0.85,
                 "New 21-DTE discipline — first real test on Jun puts."),
        AACEvent(date(2026, 6, 4), "DTE 14: Jun Final Roll Window", "options_lifecycle", "final_roll",
                 "CRITICAL", "14 DTE. Last chance to roll BKLN/HYG/OWL without excessive slippage.",
                 ["BKLN", "HYG", "OWL"], 0.9,
                 "Final roll opportunity — gamma risk rising."),
        AACEvent(date(2026, 6, 11), "DTE 7: Jun Assignment Risk Zone", "options_lifecycle", "assignment_risk",
                 "CRITICAL", "7 DTE. Gamma dominates. If not rolled at 21 DTE, accept expiry outcome.",
                 ["BKLN", "HYG", "OWL"], 0.95,
                 "Assignment probability increasing."),
        AACEvent(date(2026, 6, 18), "OPTIONS EXPIRY: Jun Puts + Moomoo Calls", "options_lifecycle", "expiry",
                 "CRITICAL", "Jun 18 expiry. BKLN/HYG/OWL puts + XLE/SLV Jun calls. Manage by market close.",
                 ["BKLN", "HYG", "OWL", "XLE", "SLV"], 1.0,
                 "Jun expiry. Calls may have value — puts depend on thesis."),
    ])

    # ── WAR ROOM MACRO EVENTS (90-Day War Room + Extended) ───────────────
    events.extend([
        AACEvent(date(2026, 2, 28), "US-IRAN CONFLICT THESIS START", "war_room", "thesis",
                 "CRITICAL", "War Day 0. Hormuz disruption scenario activated.",
                 ["OIL", "XLE", "GOLD", "SILVER"], 0.95,
                 "Master thesis activation — oil shock cascade."),
        AACEvent(date(2026, 3, 1), "War Day 1: Market Reaction", "war_room", "crisis",
                 "CRITICAL", "Oil gap up, equities gap down. Thesis confirmed.",
                 ["OIL", "SPY", "QQQ"], 0.9, "First market reaction to conflict."),
        AACEvent(date(2026, 3, 5), "OPEC Emergency Meeting", "war_room", "supply",
                 "HIGH", "Emergency production cuts. Oil $100+ confirmed.",
                 ["OIL", "XLE"], 0.85, "Supply side confirmation."),
        AACEvent(date(2026, 3, 12), "Hormuz Partial Closure", "war_room", "thesis",
                 "CRITICAL", "Oil $95+ confirmed. Chokepoint disrupted.",
                 ["OIL", "XLE", "GOLD"], 0.95, "Physical supply disruption."),
        AACEvent(date(2026, 3, 24), "War Week 4: Critical Assessment", "war_room", "assessment",
                 "HIGH", "Thesis entering critical window. Monitor escalation.",
                 ["OIL", "GOLD", "SPY"], 0.8, "Escalation probability assessment."),
        AACEvent(date(2026, 4, 14), "War Day 45: Major Inflection", "war_room", "inflection",
                 "CRITICAL", "45-day critical checkpoint. Escalation vs de-escalation.",
                 ["OIL", "GOLD", "SILVER", "SPY"], 0.9,
                 "Major inflection — determines next phase."),
        AACEvent(date(2026, 5, 29), "War Day 90: Model End / Reassess", "war_room", "model_end",
                 "CRITICAL", "90-day cycle closes. Full thesis review. Rotate or extend.",
                 ["OIL", "GOLD", "SILVER", "SPY", "KRE"], 0.85,
                 "War room model termination. New cycle begins."),
        # ── Extended macro calendar (from ninety_day_war_room.py) ──
        AACEvent(date(2026, 3, 18), "8 Live Puts Deployed + NDAX Liquidated", "war_room", "execution",
                 "CRITICAL", "IBKR: ARCC/PFF/LQD/EMB/MAIN/JNK/KRE/IWM puts. NDAX sold all XRP+ETH -> $4,492 CAD.",
                 ["ARCC", "PFF", "LQD", "EMB", "MAIN", "JNK", "KRE", "IWM"], 0.95,
                 "War Room goes LIVE. Doctrine to execution."),
        AACEvent(date(2026, 3, 19), "FOMC March + War Day 19", "war_room", "fomc",
                 "CRITICAL", "First Fed meeting since war start. Watch for emergency guidance, dot plot shift.",
                 ["SPY", "QQQ", "GOLD", "TLT"], 0.9,
                 "Fed's first post-conflict posture."),
        AACEvent(date(2026, 4, 4), "Jobs Report April (War Month 1)", "war_room", "macro_data",
                 "HIGH", "First labor data reflecting war impact. Watch unemployment claims surge.",
                 ["SPY", "KRE", "XLF"], 0.8,
                 "Employment shock first signal."),
        AACEvent(date(2026, 4, 10), "CPI March: First War-Month Inflation", "war_room", "macro_data",
                 "CRITICAL", "Oil pass-through to headline CPI. If >5% -> stagflation narrative locks in.",
                 ["OIL", "GOLD", "SILVER", "TLT"], 0.9,
                 "War inflation arrives in data."),
        AACEvent(date(2026, 4, 11), "Q1 Bank Earnings Begin", "war_room", "earnings",
                 "CRITICAL", "JPM/WFC/C. Watch: loan loss provisions, credit reserves, HY exposure comments.",
                 ["KRE", "XLF", "JNK", "HYG"], 0.85,
                 "Credit system stress under microscope."),
        AACEvent(date(2026, 4, 30), "GDP Q1 Advance: Recession Signal?", "war_room", "macro_data",
                 "CRITICAL", "If <1% -> thesis accelerates. If negative -> panic selling. Binary trigger.",
                 ["SPY", "QQQ", "GOLD", "TLT"], 0.9,
                 "GDP determines next phase — recession or resilience."),
        AACEvent(date(2026, 5, 6), "FOMC May: Emergency Cut?", "war_room", "fomc",
                 "CRITICAL", "2 months into war. Fed facing stagflation dilemma. Cut = admit, hold = crush.",
                 ["SPY", "GOLD", "TLT", "KRE"], 0.85,
                 "Fed's crisis response defines Q3 trajectory."),
        AACEvent(date(2026, 5, 14), "CPI April: 2nd War-Month Inflation", "war_room", "macro_data",
                 "CRITICAL", "Second month of oil pass-through. If >5% headline -> stagflation confirmed.",
                 ["OIL", "GOLD", "SILVER"], 0.85,
                 "Stagflation narrative confirmation."),
        AACEvent(date(2026, 5, 19), "War Day 80: Endgame Window", "war_room", "inflection",
                 "CRITICAL", "Begin hard asset transition if composite >70%. Take profits on 500%+ winners.",
                 ["GOLD", "SILVER", "GDX", "SLV"], 0.8,
                 "Endgame window — options to hard assets."),
        AACEvent(date(2026, 6, 16), "FOMC June: 3 Months Into War", "war_room", "fomc",
                 "CRITICAL", "Quarter 3 of conflict. Full monetary policy reset expected.",
                 ["SPY", "GOLD", "TLT", "XLF"], 0.8,
                 "Fed's 3-month war assessment."),
        AACEvent(date(2026, 6, 17), "War Day 109: 90-Day Model Endpoint", "war_room", "model_end",
                 "CRITICAL", "ninety_day_war_room.py MODEL_END. Full evaluation. Plan next 90d or wind-down.",
                 ["FULL_REVIEW"], 0.85,
                 "Operational model terminates. New cycle or preservation."),
    ])

    # ── CAPITAL MILESTONES ────────────────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 18), "Current Capital: ~$49.5K USD across 7 platforms", "milestone", "capital",
                 "HIGH", "Confirmed via central config (data/account_balances.json): "
                 "IBKR $30,148 + Moomoo $2,609 + Polymarket $536 = $33,293 USD. "
                 "WS $18,638 + NDAX $4,492 = $23,130 CAD. Grand total ~$49,554 USD. "
                 "Update: python -m config.account_balances show",
                 [], 0.3, "Starting capital across all platforms."),
        AACEvent(date(2026, 4, 1), "$35K Cash Injection Expected", "milestone", "capital",
                 "CRITICAL", "Major capital injection. Deploy per Moon 1 (Pink Moon) doctrine.",
                 ["SLV", "XLE", "GLD", "KRE", "JNK"], 0.85,
                 "Seed capital for full deployment."),
        AACEvent(date(2026, 5, 15), "$150K Target: Phase 1 -> Phase 2", "milestone", "phase_transition",
                 "HIGH", "Accumulation -> Growth phase transition checkpoint.",
                 ["GOLD", "SILVER", "XLE"], 0.6,
                 "Phase 1 exit. Begin growth allocation."),
        AACEvent(date(2026, 9, 22), "$1M Target: Phase 2 -> Phase 3", "milestone", "phase_transition",
                 "HIGH", "Growth -> Rotation phase transition. Equinox aligned.",
                 ["ROTATE", "TRIM_WINNERS"], 0.5,
                 "Rotation phase. Preserve gains."),
        AACEvent(date(2027, 3, 20), "$10M Milestone Assessment", "milestone", "phase_transition",
                 "CRITICAL", "Full-year target. Vernal equinox 2027. Doctrine review.",
                 ["FULL_REVIEW", "CONSOLIDATE"], 0.7,
                 "Ultimate milestone. Doctrine success/failure assessment."),
    ])

    # ── SEESAW PHASE TRANSITIONS ─────────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 12), "SEESAW: Oil Spike Phase Active", "seesaw", "oil_spike",
                 "HIGH", "Oil > $90 trigger. Allocation: OIL 10%, XLE 20%, GOLD 25%, SILVER 30%, CASH 15%.",
                 ["OIL", "XLE", "GOLD", "SILVER"], 0.8,
                 "Oil spike phase — primary catalyst active."),
        AACEvent(date(2026, 4, 15), "SEESAW: Inflation Rotation Expected", "seesaw", "inflation_rotation",
                 "HIGH", "Oil embedded in CPI. Rotation: GOLD 30%, SILVER 35%, GDX 15%, CASH 20%.",
                 ["GOLD", "SILVER", "GDX"], 0.75,
                 "Inflation rotation — oil passes baton to gold."),
        AACEvent(date(2026, 5, 30), "SEESAW: Gold Breakout Window", "seesaw", "gold_breakout",
                 "HIGH", "Gold leading. Silver amplification approaching. Blue Moon Fire Peak.",
                 ["GOLD", "SILVER", "GDX"], 0.8,
                 "Gold breakout confirmation phase."),
        AACEvent(date(2026, 6, 15), "SEESAW: Silver Amplifier Phase", "seesaw", "silver_amplifier",
                 "CRITICAL", "Silver 2-3x outperformance vs gold. Maximum leverage window.",
                 ["SILVER", "SLV"], 0.9,
                 "Silver amplifier — highest return phase."),
        AACEvent(date(2026, 8, 15), "SEESAW: Recovery Phase Check", "seesaw", "recovery",
                 "HIGH", "Equities bottoming? Rotation: SILVER 25%, GOLD 20%, XLE 15%, QQQ 15%, CASH 25%.",
                 ["SPY", "QQQ", "SILVER", "GOLD"], 0.6,
                 "Recovery assessment — exit havens?"),
    ])

    # ── STORM LIFEBOAT SCENARIOS (Status Checkpoints) ────────────────────
    events.extend([
        AACEvent(date(2026, 3, 15), "SCENARIO: Hormuz Closure P=45%", "scenario", "geopolitical",
                 "CRITICAL", "Hormuz Strait Closure scenario. P=0.45, Severity=0.95. Watch: Oil>$120, USN carrier, Iran exercises.",
                 ["OIL", "GOLD", "GDX", "XLE", "SMR"], 0.95,
                 "Primary thesis scenario."),
        AACEvent(date(2026, 4, 15), "SCENARIO: US Debt Crisis P=20%", "scenario", "financial",
                 "HIGH", "Sovereign debt stress. Watch: 10Y>5.5%, CDS widening, failed auction, DXY collapse.",
                 ["GOLD", "SILVER", "GDX", "BTC"], 0.7,
                 "Secondary cascade from primary."),
        AACEvent(date(2026, 5, 15), "SCENARIO: Credit Contagion Check", "scenario", "credit",
                 "HIGH", "Banking/credit contagion assessment. SVB-style regional bank stress.",
                 ["KRE", "XLF", "HYG", "JNK"], 0.75,
                 "Credit contagion propagation check."),
        AACEvent(date(2026, 6, 15), "SCENARIO: China-Taiwan P=30%", "scenario", "geopolitical",
                 "HIGH", "Cross-strait tension check. P=0.30, Severity=0.88. Semiconductor supply chain.",
                 ["GOLD", "SILVER", "TLT", "SMH"], 0.65,
                 "Taiwan scenario — tech supply chain risk."),
        AACEvent(date(2026, 8, 15), "SCENARIO: All Scenarios Review", "scenario", "assessment",
                 "HIGH", "Mid-year review of all 43 scenario probabilities. Recalibrate.",
                 [], 0.7, "Full scenario reassessment."),
        AACEvent(date(2026, 11, 15), "SCENARIO: Post-Election Recalibrate", "scenario", "assessment",
                 "HIGH", "Post-midterm scenario adjustment. Policy uncertainty resolved.",
                 [], 0.6, "Electoral outcome repricing."),
    ])

    # ── KEY STRATEGY ACTIVATION WINDOWS ──────────────────────────────────
    events.extend([
        # FOMC-linked strategies (#5, #28, #29)
        AACEvent(date(2026, 3, 11), "STRAT: Pre-FOMC Drift Active (#5,#28,#29)", "strategy", "macro_event",
                 "HIGH", "FOMC Cycle & Pre-Announcement Drift + VIX/Equity Pair + Regime Switch Filter.",
                 ["VIX", "SPY"], 0.7, "Pre-FOMC strategies armed."),
        AACEvent(date(2026, 4, 29), "STRAT: Pre-FOMC May Setup", "strategy", "macro_event",
                 "HIGH", "Pre-FOMC drift strategies reactivated for May meeting.",
                 ["VIX", "SPY"], 0.7, "Pre-FOMC May window."),
        AACEvent(date(2026, 6, 10), "STRAT: Pre-FOMC June Setup", "strategy", "macro_event",
                 "HIGH", "SEP update meeting. Maximum signal quality.",
                 ["VIX", "SPY"], 0.8, "Pre-FOMC June window."),
        # Earnings IV strategies (#13, #47)
        AACEvent(date(2026, 4, 7), "STRAT: Earnings IV Run-Up (#13,#47)", "strategy", "event_driven",
                 "HIGH", "Q1 earnings IV expansion. Event Vega Calendars active.",
                 ["TSLA", "AMZN", "MSFT", "META", "AAPL", "GOOGL"], 0.75,
                 "Earnings IV run-up capture."),
        AACEvent(date(2026, 7, 7), "STRAT: Q2 Earnings IV Run-Up", "strategy", "event_driven",
                 "HIGH", "Q2 earnings IV expansion.",
                 ["TSLA", "AMZN", "MSFT", "META", "AAPL", "GOOGL"], 0.75,
                 "Q2 earnings IV capture."),
        AACEvent(date(2026, 10, 6), "STRAT: Q3 Earnings IV Run-Up", "strategy", "event_driven",
                 "HIGH", "Q3 earnings IV expansion.",
                 ["TSLA", "AMZN", "MSFT", "META", "AAPL", "GOOGL"], 0.75,
                 "Q3 earnings IV capture."),
        AACEvent(date(2027, 1, 6), "STRAT: Q4 Earnings IV Run-Up", "strategy", "event_driven",
                 "HIGH", "Q4 earnings IV expansion.",
                 ["TSLA", "AMZN", "MSFT", "META", "AAPL", "GOOGL"], 0.75,
                 "Q4 earnings IV capture."),
        # Turn-of-Month (#10, #25, #49)
        AACEvent(date(2026, 3, 27), "STRAT: TOM Overlay Mar/Apr (#10,#25)", "strategy", "calendar",
                 "MEDIUM", "Turn-of-Month anomaly. Last 3 days + first 3 days of month.",
                 ["SPY", "DIA"], 0.6, "Monthly TOM signal."),
        AACEvent(date(2026, 4, 28), "STRAT: TOM Overlay Apr/May", "strategy", "calendar",
                 "MEDIUM", "Turn-of-Month anomaly active.", ["SPY", "DIA"], 0.6, "Monthly TOM."),
        AACEvent(date(2026, 5, 28), "STRAT: TOM Overlay May/Jun", "strategy", "calendar",
                 "MEDIUM", "Turn-of-Month anomaly active.", ["SPY", "DIA"], 0.6, "Monthly TOM."),
        AACEvent(date(2026, 6, 26), "STRAT: TOM Overlay Jun/Jul", "strategy", "calendar",
                 "MEDIUM", "Turn-of-Month + Quad Witching combo.", ["SPY", "DIA"], 0.7, "TOM + Quad."),
        # VRP strategies (#6, #7, #21, #37, #38, #39, #40)
        AACEvent(date(2026, 3, 3), "STRAT: VRP Basket Active (#6,#37)", "strategy", "volatility",
                 "HIGH", "Cross-Asset Variance Risk Premium basket. Selling vol across assets.",
                 ["SPY", "QQQ", "GLD", "TLT"], 0.7,
                 "VRP harvest — continuous."),
        # Dispersion (#8, #42, #43)
        AACEvent(date(2026, 4, 10), "STRAT: Dispersion Pre-Earnings (#8,#42)", "strategy", "dispersion",
                 "HIGH", "Active Dispersion + IC-RC Gate activated pre-earnings season.",
                 ["SPY", "QQQ"], 0.7, "Correlation breakdown expected."),
        # Matrix Maximizer (#56)
        AACEvent(date(2026, 3, 3), "STRAT: Matrix Maximizer Geo Bear Puts (#56)", "strategy", "geopolitical",
                 "CRITICAL", "Geopolitical Bear Puts engine. Core thesis expression.",
                 ["KRE", "IWM", "JNK", "LQD", "EMB"], 0.9,
                 "Primary strategy — crisis puts."),
    ])

    # ── AUTOMATION & DAILY OPS MARKERS ────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 3), "AAC v3.1 Launch: Full Automation", "automation", "system",
                 "CRITICAL", "Pipeline: scan(30s) + execute(15s) + metrics + intelligence + capital engine loops.",
                 [], 0.9, "System fully operational."),
        AACEvent(date(2026, 3, 19), "FOMC Decision: First Live Test", "automation", "system",
                 "HIGH", "First FOMC with live trading. Automation stress test.",
                 [], 0.8, "Live system under real FOMC stress."),
        AACEvent(date(2026, 4, 1), "Q2 Strategy Rotation Check", "automation", "ops",
                 "MEDIUM", "Quarterly strategy roster review. Enable/disable strategies per macro.",
                 [], 0.5, "Quarterly ops review."),
        AACEvent(date(2026, 7, 1), "Q3 Strategy Rotation Check", "automation", "ops",
                 "MEDIUM", "Mid-year strategy roster review.", [], 0.5, "Quarterly ops review."),
        AACEvent(date(2026, 10, 1), "Q4 Strategy Rotation Check", "automation", "ops",
                 "MEDIUM", "Pre-election strategy review.", [], 0.5, "Quarterly ops review."),
        AACEvent(date(2027, 1, 1), "2027 Strategy Rotation Check", "automation", "ops",
                 "MEDIUM", "New year strategy activation review.", [], 0.5, "Annual ops review."),
    ])

    # ── WATCHDOG / CIRCUIT BREAKER CHECKPOINTS ────────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 29), "WATCHDOG: Eclipse Day System Stress Test", "automation", "health",
                 "HIGH", "Health checks during Partial Solar Eclipse vol. Circuit breaker readiness.",
                 [], 0.7, "System resilience under vol."),
        AACEvent(date(2026, 6, 19), "WATCHDOG: Quad Witching Stress Test", "automation", "health",
                 "HIGH", "Max options activity. System load test. Circuit breakers armed.",
                 [], 0.7, "Quad witching load test."),
        AACEvent(date(2026, 8, 12), "WATCHDOG: Total Solar Eclipse Vol Test", "automation", "health",
                 "HIGH", "Total solar eclipse + earnings. Maximum vol event.",
                 [], 0.8, "Eclipse vol stress test."),
    ])

    # ── LEAPS PLAYBOOK EVENTS ────────────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 30), "LEAPS ENTRY: SLV Jan2027 65C (55% / $20.9k)", "leaps", "execution",
                 "CRITICAL", "Pink Moon Fire Peak entry. 30-40 contracts @ $4.50-$7.00. "
                 "Silver dual-leverage: monetary fear + industrial pivot from acid/copper choke.",
                 ["SLV"], 1.0, "Core position. Ride the wave."),
        AACEvent(date(2026, 3, 30), "LEAPS ENTRY: XLE Jan2027 75C (25% / $9.5k)", "leaps", "execution",
                 "CRITICAL", "Pink Moon Fire Peak entry. 20-28 contracts @ $3.00-$5.00. "
                 "Direct Hormuz + Russian Baltic chokepoint exposure. Complements existing $85C.",
                 ["XLE"], 1.0, "Chokepoint amplifier. Hold."),
        AACEvent(date(2026, 4, 1), "LEAPS ENTRY: GLD Jan2027 410C (10% / $3.8k)", "leaps", "execution",
                 "HIGH", "Pink Moon entry day 2. 3-5 contracts @ $8.00-$12.00. "
                 "Safe-haven ballast. Gold at $414 already accelerating.",
                 ["GLD"], 0.9, "Saturn's real metal meets Neptune's dream metal."),
        AACEvent(date(2026, 4, 1), "LEAPS ENTRY: JNK Jan2027 90P (10% / $3.8k)", "leaps", "execution",
                 "HIGH", "Pink Moon entry day 2. 8-12 contracts @ $3.00-$5.00. "
                 "Private credit stress amplifier. Neptune dissolves evergreen illusions.",
                 ["JNK"], 0.9, "Credit-stress reinforcement."),
        AACEvent(date(2026, 5, 31), "LEAPS EXIT: Moon 3 Blue Moon Rotation", "leaps", "rotation",
                 "CRITICAL", "Sell 50-70% of printed LEAPS (esp. credit puts). "
                 "Reinvest proceeds into additional silver/oil LEAPS. Dodecahedron geometry payoff.",
                 ["SLV", "XLE", "GLD", "JNK"], 1.0, "First major rotation node."),
        AACEvent(date(2026, 6, 21), "LEAPS EXIT: Moon 4 Solstice Rebalance", "leaps", "rotation",
                 "HIGH", "Sell 20-30% remaining printed positions. Full rebalance into silver/oil. "
                 "Target $1M-$3M portfolio range. Sri Yantra integration.",
                 ["SLV", "XLE"], 0.9, "Second rotation. Scale into winners."),
        AACEvent(date(2026, 9, 22), "LEAPS EXIT: Moon 7 Autumnal Equinox", "leaps", "rotation",
                 "HIGH", "Sell 40-60% of printed positions. Reinvest into silver/oil. "
                 "Cube geometry grounding. Harvest consolidation.",
                 ["SLV", "XLE", "GLD", "JNK"], 0.9, "Major rebalance node."),
        AACEvent(date(2027, 2, 20), "LEAPS EXIT: Moon 12 Saturn-Neptune Rotation", "leaps", "rotation",
                 "CRITICAL", "Sell 50-70% of ALL printed LEAPS on Saturn-Neptune conjunction visibility. "
                 "Final rotation into silver/gold for new 36-year cycle. Metatron's Cube + Merkaba.",
                 ["SLV", "XLE", "GLD", "JNK"], 1.0, "Peak conjunction. Old order dissolves."),
        AACEvent(date(2027, 3, 21), "LEAPS REVIEW: Moon 13 $10M Milestone", "leaps", "milestone",
                 "CRITICAL", "Sell remaining if $10M+ achieved. Full Platonic completion. "
                 "New 36-year cycle begins with real capital and independence.",
                 ["SLV", "XLE", "GLD", "JNK"], 1.0, "$10M review. Worm Moon rebirth."),
    ])

    # ── CRYPTO DOCTRINE EVENTS ───────────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 18), "CRYPTO: NDAX Liquidated — $4,492 CAD Redeployed", "crypto", "execution",
                 "HIGH", "Sold all XRP + ETH on NDAX. Proceeds redeployed to LEAPS playbook. "
                 "Crypto exposure paused. Re-entry when BTC dominance < 50% + funding normalized.",
                 ["XRP", "ETH"], 0.85, "Capital rotation: crypto → LEAPS."),
        AACEvent(date(2026, 4, 15), "CRYPTO: BTC $100k Watch — Moon 2 Reflexive Melt Check", "crypto", "milestone",
                 "HIGH", "If BTC breaks $100k during Pink Moon, evaluate spot accumulation. "
                 "Run C1 (Liquidity Reflexive Melt) formula. Check funding rates + OI.",
                 ["BTC"], 0.8, "Reflexive melt-up candidate."),
        AACEvent(date(2026, 5, 15), "CRYPTO: Alt-Season Scanner — Moon 3 BTC Dominance Check", "crypto", "analysis",
                 "MEDIUM", "Check BTC dominance trend. If dropping below 50% with ETH/BTC rising, "
                 "small alt allocation (SOL, ETH). Run C6 formula.",
                 ["ETH", "SOL", "BTC"], 0.7, "Alt-season bloom or flush."),
        AACEvent(date(2026, 6, 21), "CRYPTO: DeFi Summer Review — Moon 4 Solstice", "crypto", "analysis",
                 "MEDIUM", "ETH staking yield check. L2 adoption metrics. DeFi TVL scan. "
                 "Solstice rebalance includes crypto allocation review.",
                 ["ETH"], 0.7, "DeFi summer assessment."),
        AACEvent(date(2026, 7, 15), "CRYPTO: Leverage Flush Alert — Moon 5 Sturgeon", "crypto", "risk",
                 "CRITICAL", "Peak reflexive risk zone. Run C2 (Leverage Fragility Flush). "
                 "If OI extreme + funding spiking, AVOID new crypto positions. VIX correlation.",
                 ["BTC", "ETH", "SOL"], 0.9, "Leverage flush risk peak."),
        AACEvent(date(2026, 9, 22), "CRYPTO: Full Portfolio Review — Moon 7 Equinox", "crypto", "analysis",
                 "HIGH", "Run all C1-C8 crypto formulas. Autumnal equinox = balance point. "
                 "Metal X arb scan. NDAX re-entry evaluation. Dalio cycle cross-reference.",
                 ["BTC", "ETH", "XRP", "SOL"], 0.85, "Equinox crypto rebalance."),
        AACEvent(date(2026, 11, 1), "CRYPTO: Exchange Inflow Watch — Moon 8-9 Risk-Off", "crypto", "risk",
                 "HIGH", "C5 (Exchange Inflow Spike) monitoring. BTC dominance expansion = "
                 "alt liquidation signal. Tighten stops. Hunter-Beaver transition.",
                 ["BTC"], 0.8, "Risk-off crypto watch."),
        AACEvent(date(2027, 1, 15), "CRYPTO: Pre-Conjunction Accumulation — Moon 11 Wolf", "crypto", "execution",
                 "HIGH", "If thesis intact + re-entry triggers met, build BTC/ETH spot positions "
                 "for Saturn-Neptune institutional adoption wave. Smart money accumulating.",
                 ["BTC", "ETH"], 0.85, "Pre-conjunction positioning."),
        AACEvent(date(2027, 2, 20), "CRYPTO: Saturn-Neptune Adoption Wave — Moon 12", "crypto", "milestone",
                 "CRITICAL", "Neptune dissolves fiat illusion → digital assets as new monetary layer. "
                 "BTC as reserve asset narrative peaks. ETF flows watch. Institutional adoption.",
                 ["BTC", "ETH"], 1.0, "Fiat dissolution. Crypto ascension."),
        AACEvent(date(2027, 3, 21), "CRYPTO: New Cycle Allocation — Moon 13", "crypto", "milestone",
                 "HIGH", "If $10M+ portfolio, allocate 5-10% to crypto. "
                 "BTC 60%, ETH 25%, SOL/XRP 15%. New 36-year monetary cycle begins.",
                 ["BTC", "ETH", "SOL", "XRP"], 0.9, "New paradigm crypto allocation."),
    ])

    # ── INDICATOR PRICE TARGET MILESTONES ────────────────────────────
    events.extend([
        AACEvent(date(2026, 4, 15), "INDICATOR TARGET: Gold $2500 Breakpoint", "indicator", "milestone",
                 "HIGH", "Gold sustaining above $2500 = stagflation confirmation. "
                 "Accelerate GLD/GDX accumulation. Golden ratio Fibonacci extension.",
                 ["GLD", "GDX"], 0.85, "Stagflation indicator confirmed."),
        AACEvent(date(2026, 5, 1), "INDICATOR TARGET: VIX 25 Regime Shift", "indicator", "milestone",
                 "HIGH", "VIX sustained above 25 = ELEVATED→CRITICAL regime shift. "
                 "Increase put hedge allocation. Theta decay accelerates.",
                 ["VIX", "UVXY"], 0.8, "Volatility regime shift."),
        AACEvent(date(2026, 6, 1), "INDICATOR TARGET: Oil $85+ Sustained", "indicator", "milestone",
                 "HIGH", "Oil above $85 = supply disruption confirmed. "
                 "XLE calls print. Hormuz/OPEC thesis validated.",
                 ["XLE", "OIL"], 0.85, "Energy thesis confirmed."),
        AACEvent(date(2026, 7, 1), "INDICATOR TARGET: Silver $35 Breakout", "indicator", "milestone",
                 "HIGH", "Silver above $35 = industrial+monetary demand. "
                 "SLV LEAPS printing. Silver/Gold ratio compressing.",
                 ["SLV"], 0.9, "Silver breakout signal."),
        AACEvent(date(2026, 8, 1), "INDICATOR TARGET: HY OAS 500+ Credit Stress", "indicator", "milestone",
                 "CRITICAL", "High-yield OAS above 500bp = credit stress. "
                 "JNK puts print. BDC exposure under pressure. Seesaw amplifier.",
                 ["JNK", "HYG"], 0.9, "Credit spread blowout."),
        AACEvent(date(2026, 9, 1), "INDICATOR TARGET: DXY 110+ Dollar Wrecking Ball", "indicator", "milestone",
                 "HIGH", "Dollar Index above 110 = EM/commodity pressure. "
                 "Gold inverse squeeze. Yen carry unwind accelerates.",
                 ["UUP", "GLD"], 0.8, "Dollar dominance peak."),
        AACEvent(date(2026, 10, 1), "INDICATOR TARGET: SPY $500 Bear Market Test", "indicator", "milestone",
                 "CRITICAL", "SPY at $500 = 12% drawdown from highs. "
                 "Bear market entry zone. IWM/KRE puts deep ITM.",
                 ["SPY", "IWM", "KRE"], 0.9, "Bear market confirmation."),
        AACEvent(date(2026, 11, 1), "INDICATOR TARGET: 10Y Yield 5%+ Stress", "indicator", "milestone",
                 "CRITICAL", "10Y above 5% = fiscal dominance. LQD/EMB puts printing. "
                 "Bond vigilantes force Fed hand. Seesaw maximum dislocation.",
                 ["LQD", "EMB", "TLT"], 0.95, "Yield breakout."),
        AACEvent(date(2026, 12, 1), "INDICATOR TARGET: Gold $3000 Parabolic", "indicator", "milestone",
                 "CRITICAL", "Gold at $3000 = monetary system stress. "
                 "GLD LEAPS 500%+ return. Central bank buying acceleration.",
                 ["GLD", "GDX", "SLV"], 0.95, "Gold parabolic."),
        AACEvent(date(2027, 1, 1), "INDICATOR TARGET: VIX 40+ Crisis Mode", "indicator", "milestone",
                 "CRITICAL", "VIX above 40 = systemic crisis. All puts printing. "
                 "Portfolio protection thesis fully validated. War Room DEFCON 1.",
                 ["VIX", "UVXY", "SPY"], 1.0, "Crisis volatility."),
    ])

    # ── PORTFOLIO VALUE CHECKPOINTS ──────────────────────────────────
    events.extend([
        AACEvent(date(2026, 4, 1), "PORTFOLIO CHECKPOINT: $10K Level", "portfolio", "milestone",
                 "HIGH", "Portfolio crosses $10K. First major validation. "
                 "Scale position sizing to 2% per trade. War Room cadence increase.",
                 [], 0.7, "First compounding milestone."),
        AACEvent(date(2026, 5, 15), "PORTFOLIO CHECKPOINT: $25K Level", "portfolio", "milestone",
                 "HIGH", "Portfolio crosses $25K. Day trading threshold. "
                 "Pattern Day Trader rules apply. Consider Moomoo margin.",
                 [], 0.75, "PDT threshold reached."),
        AACEvent(date(2026, 7, 1), "PORTFOLIO CHECKPOINT: $50K Level", "portfolio", "milestone",
                 "HIGH", "Portfolio crosses $50K. Real capital territory. "
                 "Scale into LEAPS. Consider second IBKR account.",
                 [], 0.8, "Real capital milestone."),
        AACEvent(date(2026, 9, 1), "PORTFOLIO CHECKPOINT: $100K Level", "portfolio", "milestone",
                 "CRITICAL", "Portfolio crosses $100K. Six-figure territory. "
                 "Risk management paramount. Diversify across brokers.",
                 [], 0.85, "Six-figure milestone."),
        AACEvent(date(2026, 12, 1), "PORTFOLIO CHECKPOINT: $500K Level", "portfolio", "milestone",
                 "CRITICAL", "Portfolio crosses $500K. Thesis compounding proven. "
                 "Consider tax optimization. Set up holding structures.",
                 [], 0.9, "Half-million milestone."),
        AACEvent(date(2027, 2, 1), "PORTFOLIO CHECKPOINT: $1M Level", "portfolio", "milestone",
                 "CRITICAL", "Portfolio crosses $1M. Independence threshold. "
                 "Saturn-Neptune conjunction approaching. Rotate to preservation.",
                 [], 0.95, "Millionaire milestone."),
    ])

    # ── MOOMOO POSITION LIFECYCLE ────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 4, 17), "MOOMOO: OWL Put Theta Decay Check (270 DTE)", "options_lifecycle",
                 "risk_management", "MEDIUM",
                 "OWL $5P Jan 15 2027 (10 contracts) — 9 months to expiry. Theta decay minimal. "
                 "Monitor BDC credit conditions. +50% profit ($0.30→$0.45). Let ride.",
                 ["OWL"], 0.7, "Long-dated put. Patience."),
        AACEvent(date(2026, 5, 18), "MOOMOO: SLV Jun-18 Call — 30 DTE Warning", "options_lifecycle",
                 "risk_management", "HIGH",
                 "SLV $70C Jun 18 2026 — 30 days to expiry. Currently +53% ($4.00→$6.10). "
                 "Take partial profits or roll to Sep $67.5C (already have 1 Sep contract).",
                 ["SLV"], 0.85, "30-DTE decision point."),
        AACEvent(date(2026, 6, 4), "MOOMOO: SLV Jun-18 Call — 14 DTE Critical", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "SLV $70C Jun 18 2026 — 2 weeks remain. Theta acceleration. "
                 "Close if OTM. Roll to Sep $67.5C. Do NOT hold to expiry OTM.",
                 ["SLV"], 0.9, "Theta cliff. Act now."),
        AACEvent(date(2026, 6, 11), "MOOMOO: XLE Jun-18 Call — 7 DTE Final", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "XLE $60C Jun 18 2026 — last week. Gamma spike. Currently +79% ($3.00→$5.37). "
                 "Close or exercise if ITM. DEEP ITM — likely exercise or close for profit.",
                 ["XLE"], 0.95, "Final week. Execute."),
        AACEvent(date(2026, 6, 18), "MOOMOO: XLE & SLV June Expirations", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "June 18 OPTIONS EXPIRY — XLE $60C and SLV $70C expire. "
                 "Also: WS OWL $8P (5 contracts) expires. Close all before 3pm ET.",
                 ["XLE", "SLV", "OWL"], 1.0, "Expiration day. Close all."),
        AACEvent(date(2026, 8, 18), "MOOMOO: SLV Sep-18 Call — 30 DTE Warning", "options_lifecycle",
                 "risk_management", "HIGH",
                 "SLV $67.5C Sep 18 2026 — 30 days remain. Currently +84% ($5.50→$10.12). "
                 "Silver thesis strong. Consider roll to December or take profit.",
                 ["SLV"], 0.85, "September SLV decision."),
        AACEvent(date(2026, 9, 18), "MOOMOO: SLV Sep-18 Expiration", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "SLV $67.5C Sep 18 2026 — EXPIRY. Close before 3pm ET.",
                 ["SLV"], 1.0, "SLV Sep expiry."),
        AACEvent(date(2026, 12, 15), "MOOMOO: OWL Jan-15 Put — 30 DTE Warning", "options_lifecycle",
                 "risk_management", "HIGH",
                 "OWL $5P Jan 15 2027 (10 contracts Moomoo) — 30 days to expiry. "
                 "BDC credit thesis check. Also: WS XLE $85C (26 contracts) expires same day.",
                 ["OWL", "XLE"], 0.85, "OWL put + XLE LEAPS final month."),
        AACEvent(date(2027, 1, 15), "MOOMOO + WS: OWL & XLE Jan-15 Expiration", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "OWL $5P Jan 15 2027 (10 Moomoo) + XLE $85C (26 WS) — EXPIRY. "
                 "Close all before 3pm ET. Two largest positions by contract count.",
                 ["OWL", "XLE"], 1.0, "OWL + XLE expiry."),
    ])

    # ── IBKR PUT LIFECYCLE (APR 17 EXPIRY) ──────────────────────────
    events.extend([
        AACEvent(date(2026, 3, 31), "IBKR PUTS: 17 DTE — Mid-Life Review", "options_lifecycle",
                 "risk_management", "HIGH",
                 "4 IBKR puts expire Apr 17 (17 DTE). Mid-life theta acceleration. "
                 "ARCC $17P (break-even), PFF $29P (-82% LOSS), MAIN $49.7P (-38% loss), "
                 "JNK $92P (break-even). ROLL DECISION REQUIRED. "
                 "Also: XLF $46P May 1 (32 DTE), LQD $106P May 15, EMB $90P May 15 (+71%), "
                 "BKLN $20P x3 Jun 18, HYG $77P Jun 18. Cash: $9,900.",
                 ["ARCC", "PFF", "MAIN", "JNK", "XLF", "LQD", "EMB", "BKLN", "HYG"], 0.9,
                 "Mid-life. Roll Apr 17 losers."),
        AACEvent(date(2026, 4, 3), "IBKR PUTS: 14 DTE — Theta Cliff Warning", "options_lifecycle",
                 "risk_management", "HIGH",
                 "14 DTE on Apr 17 puts. Theta accelerating. PFF $29P near worthless — close. "
                 "MAIN $49.7P underwater — roll to Jun if BDC thesis holds. "
                 "ARCC/JNK break-even — roll to Jun same strike.",
                 ["ARCC", "PFF", "MAIN", "JNK"], 0.9,
                 "Theta cliff. Close PFF, roll rest."),
        AACEvent(date(2026, 4, 10), "IBKR PUTS: 7 DTE — Final Week", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "7 DTE. Gamma spike zone. Close all Apr 17 positions. "
                 "Roll ARCC, JNK, MAIN to Jun/Jul. Let PFF expire. $9,900 cash available for new trades.",
                 ["ARCC", "PFF", "MAIN", "JNK"], 0.95,
                 "Final week. Roll or close."),
        AACEvent(date(2026, 4, 16), "IBKR PUTS: EXPIRY EVE — Close All Apr 17", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "Day before Apr 17 expiry. Close ALL remaining Apr 17 puts by 3:30pm ET. "
                 "Do NOT hold to expiry for exercise risk. Deploy rolls with $9,900 cash.",
                 ["ARCC", "PFF", "MAIN", "JNK"], 1.0,
                 "Expiry eve. Close & roll."),
    ])

    # ── WEALTHSIMPLE TFSA OPTIONS LIFECYCLE (APR 17 EXPIRY) ─────────
    events.extend([
        AACEvent(date(2026, 3, 31), "WS TFSA: 4 Options Expiring Apr 17 — Roll Review", "options_lifecycle",
                 "risk_management", "HIGH",
                 "WealthSimple TFSA: 4 positions expire Apr 17 (18 DTE). "
                 "OBDC $10P x65 (+17%, largest by contracts — $975 cost basis). "
                 "JNK $94P x5 (+36% PROFIT). KRE $60P x1 (+17%). ARCC $16P x10 (break-even). "
                 "Total WS Apr 17 exposure: ~$2,405 cost basis. All profitable — ROLL to Jun/Jul.",
                 ["OBDC", "JNK", "KRE", "ARCC"], 0.9,
                 "WS Apr 17 roll decisions."),
        AACEvent(date(2026, 4, 10), "WS TFSA: 7 DTE — Execute Rolls", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "7 DTE on WS TFSA Apr 17 puts. Execute all rolls NOW. "
                 "Priority: OBDC 65 contracts first (largest), JNK 5 contracts, ARCC 10 contracts, KRE 1 contract. "
                 "Roll to Jun 18 or Jul 17 at same or lower strikes.",
                 ["OBDC", "JNK", "KRE", "ARCC"], 0.95,
                 "Roll WS TFSA positions."),
        AACEvent(date(2026, 4, 16), "WS TFSA: EXPIRY EVE — Close All Apr 17", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "Day before Apr 17 expiry. Close or roll ALL remaining WS positions. "
                 "OBDC $10P (65 contracts) and ARCC $16P (10 contracts) are largest.",
                 ["OBDC", "JNK", "KRE", "ARCC"], 1.0,
                 "WS expiry eve."),
    ])

    # ── WS TFSA LEAPS LIFECYCLE ──────────────────────────────────────
    events.extend([
        AACEvent(date(2026, 12, 15), "WS TFSA: XLE LEAPS Jan-15 — 30 DTE", "options_lifecycle",
                 "risk_management", "HIGH",
                 "XLE $85C Jan 15 2027 — 26 contracts, WS TFSA. Currently +162% ($0.37→$0.97). "
                 "Massive winner. Consider taking 50% profit or rolling to Jun 2027.",
                 ["XLE"], 0.85, "XLE LEAPS decision."),
        AACEvent(date(2027, 2, 17), "WS TFSA: GLD LEAPS Mar-19 — 30 DTE", "options_lifecycle",
                 "risk_management", "HIGH",
                 "GLD $515C Mar 19 2027 — 1 contract, WS TFSA. Currently +25% ($19.40→$24.33). "
                 "Gold thesis check. Roll or close depending on $GLD level.",
                 ["GLD"], 0.85, "GLD LEAPS decision."),
        AACEvent(date(2027, 3, 19), "WS TFSA: GLD Mar-19 Expiration", "options_lifecycle",
                 "risk_management", "CRITICAL",
                 "GLD $515C Mar 19 2027 — EXPIRY. Close before 3pm ET.",
                 ["GLD"], 1.0, "GLD LEAPS expiry."),
    ])

    # ── COMPOSITE SCORE REGIME TRIGGERS ──────────────────────────────
    events.extend([
        AACEvent(date(2026, 4, 1), "WAR ROOM: Composite 60 — Accumulation Trigger", "war_room",
                 "regime", "HIGH",
                 "Composite score crosses 60/100. Thesis strengthening. "
                 "Increase position sizing to 3%. Add new seesaw amplifiers.",
                 [], 0.75, "Accumulation trigger."),
        AACEvent(date(2026, 5, 1), "WAR ROOM: Composite 70 — Conviction Trigger", "war_room",
                 "regime", "HIGH",
                 "Composite score crosses 70/100. Strong thesis confirmation. "
                 "Max position sizing. Deploy LEAPS capital.",
                 [], 0.8, "High conviction."),
        AACEvent(date(2026, 7, 1), "WAR ROOM: Composite 80 — Acceleration Trigger", "war_room",
                 "regime", "CRITICAL",
                 "Composite score crosses 80/100. Thesis in overdrive. "
                 "Deploy reserves. All seesaw amplifiers active.",
                 [], 0.9, "Acceleration mode."),
        AACEvent(date(2026, 9, 1), "WAR ROOM: Composite 90 — Maximum Conviction", "war_room",
                 "regime", "CRITICAL",
                 "Composite score crosses 90/100. Maximum thesis validation. "
                 "Full deployment. Start planning exit strategy.",
                 [], 0.95, "Peak conviction."),
    ])

    # ── MONTHLY DOCTRINE CHECK-INS ───────────────────────────────────
    for month_num, month_name in [(4, "April"), (5, "May"), (6, "June"),
                                   (7, "July"), (8, "August"), (9, "September"),
                                   (10, "October"), (11, "November"), (12, "December")]:
        events.append(AACEvent(
            date(2026, month_num, 1),
            f"MONTHLY DOCTRINE: {month_name} 2026 Review",
            "strategy", "review", "MEDIUM",
            f"{month_name} monthly review. Recalibrate all indicators. "
            "Update composite score. Review P&L. Adjust thesis probabilities. "
            "Check moon phase alignment. Sacred geometry coherence.",
            [], 0.6, f"Monthly doctrine cycle."))
    for month_num, month_name in [(1, "January"), (2, "February"), (3, "March")]:
        events.append(AACEvent(
            date(2027, month_num, 1),
            f"MONTHLY DOCTRINE: {month_name} 2027 Review",
            "strategy", "review", "MEDIUM",
            f"{month_name} 2027 monthly review. Saturn-Neptune era approach. "
            "Final cycle review. Position for new 36-year epoch.",
            [], 0.6, f"Monthly doctrine cycle."))

    return events


_MOON_CYCLE_DATA = [
    (0,  date(2026, 3, 3),  date(2026, 3, 31), "Worm Moon (Total Lunar Eclipse)",
     date(2026, 3, 3), date(2026, 3, 17)),
    (1,  date(2026, 4, 1),  date(2026, 4, 30), "Pink Moon",
     date(2026, 4, 1), date(2026, 4, 17)),
    (2,  date(2026, 4, 30), date(2026, 5, 29), "Flower Moon",
     date(2026, 5, 1), date(2026, 5, 16)),
    (3,  date(2026, 5, 30), date(2026, 6, 27), "Blue Moon",
     date(2026, 5, 31), date(2026, 6, 14)),
    (4,  date(2026, 6, 28), date(2026, 7, 27), "Buck Moon",
     date(2026, 6, 29), date(2026, 7, 14)),
    (5,  date(2026, 7, 28), date(2026, 8, 25), "Sturgeon Moon",
     date(2026, 7, 29), date(2026, 8, 12)),
    (6,  date(2026, 8, 26), date(2026, 9, 24), "Corn Moon",
     date(2026, 8, 27), date(2026, 9, 10)),
    (7,  date(2026, 9, 25), date(2026, 10, 23), "Harvest Moon",
     date(2026, 9, 26), date(2026, 10, 10)),
    (8,  date(2026, 10, 24), date(2026, 11, 22), "Hunter Moon",
     date(2026, 10, 25), date(2026, 11, 9)),
    (9,  date(2026, 11, 23), date(2026, 12, 22), "Beaver Supermoon",
     date(2026, 11, 24), date(2026, 12, 8)),
    (10, date(2026, 12, 23), date(2027, 1, 20), "Cold Moon",
     date(2026, 12, 24), date(2027, 1, 7)),
    (11, date(2027, 1, 21), date(2027, 2, 19), "Wolf Moon",
     date(2027, 1, 22), date(2027, 2, 6)),
    (12, date(2027, 2, 20), date(2027, 3, 20), "Snow Moon",
     date(2027, 2, 20), date(2027, 3, 8)),
    (13, date(2027, 3, 21), date(2027, 4, 19), "Worm Moon",
     date(2027, 3, 22), date(2027, 4, 5)),
]


# ═══════════════════════════════════════════════════════════════════════════
# LEAD TIME RULES
# ═══════════════════════════════════════════════════════════════════════════

LEAD_TIME_RULES: Dict[str, List[Tuple[int, str]]] = {
    "eclipse": [
        (14, "Review all positions. Close weak thesis trades."),
        (7, "Set hard stops. Reduce leverage. Cash up."),
        (3, "Final positioning. Maximum conviction entries only."),
        (0, "ECLIPSE DAY. Execute doctrine mandate."),
    ],
    "equinox": [
        (7, "Begin seasonal rotation assessment."),
        (3, "Finalize rebalance plan."),
        (0, "EQUINOX. Execute rebalance."),
    ],
    "solstice": [
        (7, "Begin seasonal rotation assessment."),
        (3, "Finalize rebalance plan."),
        (0, "SOLSTICE. Execute rebalance."),
    ],
    "ingress": [
        (7, "Research ingress implications. Historical pattern scan."),
        (3, "Position for regime change."),
        (0, "INGRESS. New astrological era begins."),
    ],
    "earnings": [
        (7, "Pre-earnings vol scan. IV rank assessment."),
        (3, "Set straddles/strangles if playing vol. Or exit risk."),
        (1, "Final check. Remove/reduce delta exposure around report."),
        (0, "EARNINGS. Post-report assessment within 24h."),
    ],
    "fed": [
        (7, "Pre-FOMC positioning. Reduce directional risk."),
        (3, "Tighten stops. Review rate-sensitive positions."),
        (1, "Day before FOMC. Minimal new risk."),
        (0, "FOMC DAY. Wait for statement + presser. React after 2:30pm ET."),
    ],
    "geopolitical": [
        (7, "Scenario analysis. Update probability weights."),
        (3, "Hedge tail risk. Check oil/gold positioning."),
        (0, "EVENT DAY. Monitor real-time. Execute contingency if triggered."),
    ],
    "phi": [
        (3, "Phi resonance window approaching. Prepare conviction entries."),
        (1, "Final preparation. Check coherence score."),
        (0, "PHI WINDOW. Peak coherence. Maximum signal quality."),
    ],
    "options": [
        (7, "Roll or close expiring positions."),
        (3, "Final roll window. Pin risk assessment."),
        (0, "EXPIRATION. All positions must be managed by close."),
    ],
    "full_moon": [
        (3, "Fire Peak approaching. Prepare deployment capital."),
        (1, "Pre-Fire Peak. Final sizing."),
        (0, "FIRE PEAK. Deploy per doctrine mandate."),
    ],
    "summit": [
        (5, "Pre-summit positioning. Monitor communique leaks."),
        (1, "Day before. Expect headline risk."),
        (0, "SUMMIT. Headline risk active."),
    ],
    "trade": [
        (5, "Monitor trade policy signals."),
        (1, "Final positioning."),
        (0, "TRADE EVENT. Watch oil/FX/tariff impact."),
    ],
    "economic": [
        (3, "Data release lead-in. Review consensus estimates."),
        (0, "DATA DAY. React to deviation from consensus."),
    ],
    "policy": [
        (5, "Policy deadline approaching."),
        (1, "Final day. Maximum uncertainty."),
        (0, "DEADLINE. Policy enacted or extended."),
    ],
    "live_trade": [
        (0, "TRADE EXECUTED. Monitor fill quality and slippage."),
    ],
    "dte_check": [
        (3, "DTE milestone approaching. Review P&L on all positions."),
        (0, "DTE CHECKPOINT. Assess each position: close, roll, or hold."),
    ],
    "roll_window": [
        (3, "Roll window opening. Prepare roll orders."),
        (0, "ROLL WINDOW. Execute rolls on losing positions."),
    ],
    "theta_accel": [
        (3, "Theta acceleration imminent. Identify positions to close."),
        (0, "THETA ACCELERATION. Close winners > 50% profit."),
    ],
    "management": [
        (3, "Management window approaching. Prepare final decisions."),
        (0, "MANAGEMENT WINDOW. Must decide: close, roll, or ride."),
    ],
    "final_roll": [
        (3, "Final roll window approaching."),
        (0, "FINAL ROLL. Last chance to roll without excessive slippage."),
    ],
    "assignment_risk": [
        (3, "Assignment risk zone approaching. Gamma dominates."),
        (0, "ASSIGNMENT ZONE. Close or accept assignment."),
    ],
    "expiry": [
        (7, "Expiry in one week. Roll or close all positions."),
        (3, "Expiry in 3 days. Final management."),
        (0, "EXPIRY DAY. All positions must be managed by close."),
    ],
    "thesis": [
        (7, "Major thesis event approaching. Full review."),
        (3, "Position sizing check. Maximum conviction only."),
        (0, "THESIS EVENT. Execute per doctrine."),
    ],
    "crisis": [
        (3, "Crisis event approaching. Hedge and protect."),
        (0, "CRISIS EVENT. Execute contingency plan."),
    ],
    "inflection": [
        (7, "Major inflection approaching. Scenario analysis."),
        (3, "Finalize positioning for inflection."),
        (0, "INFLECTION POINT. Reassess all thesis probabilities."),
    ],
    "model_end": [
        (14, "Model termination in 2 weeks. Begin exit planning."),
        (7, "Model end in 1 week. Finalize rotation plan."),
        (0, "MODEL END. Full thesis review. New cycle begins."),
    ],
    "capital": [
        (7, "Capital event approaching. Prepare deployment plan."),
        (0, "CAPITAL EVENT. Deploy per doctrine mandate."),
    ],
    "phase_transition": [
        (14, "Phase transition checkpoint in 2 weeks. Assess progress."),
        (7, "Phase transition in 1 week. Prepare allocation shift."),
        (0, "PHASE TRANSITION. Execute allocation change."),
    ],
    "oil_spike": [
        (3, "Oil spike phase active. Monitor oil price levels."),
        (0, "OIL SPIKE. Execute seesaw allocation."),
    ],
    "inflation_rotation": [
        (3, "Inflation rotation expected. Prepare gold/silver allocation."),
        (0, "INFLATION ROTATION. Shift from oil to precious metals."),
    ],
    "gold_breakout": [
        (3, "Gold breakout approaching. Increase gold allocation."),
        (0, "GOLD BREAKOUT. Silver amplification phase imminent."),
    ],
    "silver_amplifier": [
        (3, "Silver amplifier approaching. Maximum leverage window."),
        (0, "SILVER AMPLIFIER. Highest return phase active."),
    ],
    "recovery": [
        (3, "Recovery phase check. Monitor equity support levels."),
        (0, "RECOVERY CHECK. Assess equity bottoming signals."),
    ],
    "macro_event": [
        (5, "Pre-FOMC strategy window opening."),
        (1, "Strategies armed. Final calibration."),
        (0, "STRATEGY ACTIVE. Execution per model."),
    ],
    "event_driven": [
        (5, "Earnings IV expansion starting. Calendars/straddles."),
        (1, "IV peak approaching. Final entries."),
        (0, "EVENT DRIVEN. IV crush or continuation assessment."),
    ],
    "calendar": [
        (2, "Turn-of-Month anomaly window opening."),
        (0, "TOM ACTIVE. Execute calendar overlay."),
    ],
    "volatility": [
        (0, "VRP HARVEST. Continuous variance risk premium capture."),
    ],
    "dispersion": [
        (3, "Dispersion strategy activation approaching."),
        (0, "DISPERSION. Index vs single-stock vol spread active."),
    ],
    "system": [
        (1, "System event imminent. Pre-flight checks."),
        (0, "SYSTEM EVENT. Monitor all dashboards."),
    ],
    "ops": [
        (3, "Quarterly ops review approaching."),
        (0, "OPS REVIEW. Strategy roster update."),
    ],
    "health": [
        (1, "Stress test imminent. Verify circuit breakers."),
        (0, "STRESS TEST. Monitor system health."),
    ],
    "assessment": [
        (7, "Scenario review approaching. Gather data."),
        (0, "SCENARIO REVIEW. Update all probability weights."),
    ],
    "credit": [
        (5, "Credit contagion check approaching."),
        (0, "CREDIT CHECK. Assess spread widening, defaults."),
    ],
    "supply": [
        (3, "Supply event approaching. Monitor OPEC signals."),
        (0, "SUPPLY EVENT. Oil/energy impact assessment."),
    ],
    "liquidation": [
        (0, "LIQUIDATION COMPLETE. Cash available for redeployment."),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ThirteenMoonDoctrine:
    """Master 13-Moon compounding timeline with all overlays."""

    def __init__(self) -> None:
        _refresh_live_doctrine_state()
        self.astrology_events = _build_astrology_events()
        self.phi_markers = _build_phi_coherence_markers()
        self.financial_events = _build_financial_events()
        self.world_events = _build_world_events()
        self.doctrine_actions = _build_doctrine_actions()
        self.aac_events = _build_aac_events()
        self.moon_cycles = self._build_cycles()

    def _build_cycles(self) -> List[MoonCycle]:
        """Build all 14 moon cycles (0-13) with events attached."""
        cycles = []
        for data in _MOON_CYCLE_DATA:
            moon_num, start, end, name, fire_peak, new_moon = data
            cycle = MoonCycle(
                moon_number=moon_num,
                start_date=start,
                end_date=end,
                lunar_phase_name=name,
                fire_peak_date=fire_peak,
                new_moon_date=new_moon,
            )
            # Attach events that fall within this cycle's window
            for evt in self.astrology_events:
                if start <= evt.date <= end:
                    cycle.astrology_events.append(evt)
            for phi in self.phi_markers:
                if start <= phi.date <= end:
                    cycle.phi_markers.append(phi)
            for fin in self.financial_events:
                if start <= fin.date <= end:
                    cycle.financial_events.append(fin)
            for world in self.world_events:
                if start <= world.date <= end:
                    cycle.world_events.append(world)
            for aac in self.aac_events:
                if start <= aac.date <= end:
                    cycle.aac_events.append(aac)
            cycle.doctrine_action = self.doctrine_actions.get(moon_num)
            cycles.append(cycle)
        return cycles

    def get_current_moon(self, target: Optional[date] = None) -> Optional[MoonCycle]:
        """Get the moon cycle containing the target date."""
        target = target or date.today()
        for cycle in self.moon_cycles:
            if cycle.start_date <= target <= cycle.end_date:
                return cycle
        return None

    def get_events_with_lead_time(
        self, days_ahead: int = 14, target: Optional[date] = None,
    ) -> List[ScheduledAlert]:
        """Get all upcoming events within days_ahead, with lead-time actions."""
        target = target or date.today()
        horizon = target + timedelta(days=days_ahead)
        alerts: List[ScheduledAlert] = []

        # Find which moon we're in
        current_moon = self.get_current_moon(target)
        moon_num = current_moon.moon_number if current_moon else -1

        # Collect all events in the horizon
        for evt in self.astrology_events:
            if target <= evt.date <= horizon:
                days_until = (evt.date - target).days
                lead_action = self._get_lead_action(evt.category, days_until)
                alerts.append(ScheduledAlert(
                    event_date=evt.date, event_name=evt.name,
                    event_type="astrology", moon_number=moon_num,
                    days_until=days_until, lead_time_action=lead_action,
                    priority="CRITICAL" if evt.impact == "HIGH" else "HIGH",
                ))

        for phi in self.phi_markers:
            if target <= phi.date <= horizon:
                days_until = (phi.date - target).days
                lead_action = self._get_lead_action("phi", days_until)
                alerts.append(ScheduledAlert(
                    event_date=phi.date, event_name=phi.label,
                    event_type="phi", moon_number=moon_num,
                    days_until=days_until, lead_time_action=lead_action,
                    priority="HIGH" if phi.resonance_strength > 0.5 else "MEDIUM",
                ))

        for fin in self.financial_events:
            if target <= fin.date <= horizon:
                days_until = (fin.date - target).days
                lead_action = self._get_lead_action(fin.category, days_until)
                alerts.append(ScheduledAlert(
                    event_date=fin.date, event_name=fin.name,
                    event_type="financial", moon_number=moon_num,
                    days_until=days_until, lead_time_action=lead_action,
                    priority="CRITICAL" if fin.impact == "HIGH" else "MEDIUM",
                ))

        for world in self.world_events:
            if target <= world.date <= horizon:
                days_until = (world.date - target).days
                lead_action = self._get_lead_action(world.category, days_until)
                alerts.append(ScheduledAlert(
                    event_date=world.date, event_name=world.name,
                    event_type="world", moon_number=moon_num,
                    days_until=days_until, lead_time_action=lead_action,
                    priority="CRITICAL" if world.impact == "HIGH" else "MEDIUM",
                ))

        for aac in self.aac_events:
            if target <= aac.date <= horizon:
                days_until = (aac.date - target).days
                lead_action = self._get_lead_action(aac.category, days_until)
                alerts.append(ScheduledAlert(
                    event_date=aac.date, event_name=aac.name,
                    event_type="aac", moon_number=moon_num,
                    days_until=days_until, lead_time_action=lead_action,
                    priority="CRITICAL" if aac.impact in ("CRITICAL", "HIGH") else "MEDIUM",
                ))

        alerts.sort(key=lambda a: (a.event_date, a.priority))
        return alerts

    def _get_lead_action(self, category: str, days_until: int) -> str:
        """Look up the appropriate lead-time action for an event category."""
        rules = LEAD_TIME_RULES.get(category, LEAD_TIME_RULES.get("economic", []))
        action = "Monitor."
        for threshold, rule_action in rules:
            if days_until <= threshold:
                action = rule_action
                break
        return action

    def get_all_events_sorted(self) -> List[Dict[str, Any]]:
        """Get every event across all layers, sorted chronologically."""
        events = []
        for evt in self.astrology_events:
            events.append({"date": evt.date.isoformat(), "name": evt.name,
                           "type": "astrology", "category": evt.category,
                           "impact": evt.impact, "desc": evt.description})
        for phi in self.phi_markers:
            events.append({"date": phi.date.isoformat(), "name": phi.label,
                           "type": "phi", "category": "coherence",
                           "impact": "HIGH" if phi.resonance_strength > 0.5 else "MEDIUM",
                           "desc": f"Resonance: {phi.resonance_strength:.1%}"})
        for fin in self.financial_events:
            events.append({"date": fin.date.isoformat(), "name": fin.name,
                           "type": "financial", "category": fin.category,
                           "impact": fin.impact, "desc": fin.description,
                           "companies": fin.companies})
        for world in self.world_events:
            events.append({"date": world.date.isoformat(), "name": world.name,
                           "type": "world", "category": world.category,
                           "impact": world.impact, "desc": world.description})
        for aac in self.aac_events:
            events.append({"date": aac.date.isoformat(), "name": aac.name,
                           "type": "aac", "category": aac.category,
                           "impact": aac.impact, "desc": aac.description,
                           "layer": aac.layer, "assets": aac.assets,
                           "conviction": aac.conviction})
        events.sort(key=lambda e: e["date"])
        return events

    def export_json(self, path: str = "data/storyboard/thirteen_moon_timeline.json") -> str:
        """Export the full timeline as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _refresh_live_doctrine_state()

        data = {
            "generated": datetime.now().isoformat(),
            "anchor": MOON_ZERO_DATE.isoformat(),
            "phi": PHI,
            "synodic_month": SYNODIC_MONTH,
            "total_cycles": len(self.moon_cycles),
            "moon_cycles": [],
            "all_events": self.get_all_events_sorted(),
                "account_balances": ACCOUNT_BALANCES,
                "war_room_doctrine": WAR_ROOM_DOCTRINE,
        }

        for cycle in self.moon_cycles:
            c = {
                "moon_number": cycle.moon_number,
                "start_date": cycle.start_date.isoformat(),
                "end_date": cycle.end_date.isoformat(),
                "name": cycle.lunar_phase_name,
                "fire_peak": cycle.fire_peak_date.isoformat() if cycle.fire_peak_date else None,
                "new_moon": cycle.new_moon_date.isoformat() if cycle.new_moon_date else None,
                "astrology_count": len(cycle.astrology_events),
                "phi_marker_count": len(cycle.phi_markers),
                "financial_count": len(cycle.financial_events),
                "world_count": len(cycle.world_events),
                "doctrine": {
                    "mandate": cycle.doctrine_action.mandate,
                    "description": cycle.doctrine_action.description,
                    "conviction": cycle.doctrine_action.conviction,
                    "targets": cycle.doctrine_action.targets,
                } if cycle.doctrine_action else None,
            }
            data["moon_cycles"].append(c)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
        logger.info("Timeline JSON exported to %s", path)
        return path

    def print_timeline(self) -> None:
        """Print the full 13-Moon timeline to console (ASCII-safe)."""
        print("=" * 100)
        print("  13-MOON DOCTRINE TIMELINE -- March 2026 to April 2027")
        print("  Anchor: March 3, 2026 Total Lunar Eclipse (Virgo)")
        print("  Phi (golden mean): %.10f" % PHI)
        print("=" * 100)

        for cycle in self.moon_cycles:
            print()
            print("-" * 100)
            mandate = cycle.doctrine_action.mandate if cycle.doctrine_action else "N/A"
            conviction = cycle.doctrine_action.conviction if cycle.doctrine_action else 0
            print(
                f"  MOON {cycle.moon_number:>2} | {cycle.start_date} -> {cycle.end_date} | "
                f"{cycle.lunar_phase_name}"
            )
            print(f"  MANDATE: {mandate} (conviction: {conviction:.0%})")
            if cycle.doctrine_action:
                print(f"  ACTION: {cycle.doctrine_action.description}")
                if cycle.doctrine_action.targets:
                    print(f"  TARGETS: {', '.join(cycle.doctrine_action.targets)}")

            if cycle.fire_peak_date:
                print(f"  FIRE PEAK: {cycle.fire_peak_date}")
            if cycle.new_moon_date:
                print(f"  NEW MOON: {cycle.new_moon_date}")

            if cycle.astrology_events:
                print("  ASTROLOGY:")
                for evt in sorted(cycle.astrology_events, key=lambda e: e.date):
                    print(f"    [{evt.impact:>6}] {evt.date} -- {evt.name}")
                    if evt.description:
                        print(f"             {evt.description}")

            if cycle.phi_markers:
                print("  PHI COHERENCE:")
                for phi in cycle.phi_markers:
                    bar = "#" * int(phi.resonance_strength * 20)
                    print(
                        f"    phi^{phi.phi_power:<2} | {phi.date} | "
                        f"resonance: {phi.resonance_strength:.1%} [{bar}]"
                    )

            if cycle.financial_events:
                print("  FINANCIAL:")
                for fin in sorted(cycle.financial_events, key=lambda e: e.date):
                    companies = f" ({', '.join(fin.companies)})" if fin.companies else ""
                    print(f"    [{fin.impact:>6}] {fin.date} -- {fin.name}{companies}")

            if cycle.world_events:
                print("  WORLD NEWS:")
                for world in sorted(cycle.world_events, key=lambda e: e.date):
                    print(f"    [{world.impact:>6}] {world.date} -- {world.name}")
                    if world.description:
                        print(f"             {world.description}")

            if cycle.aac_events:
                print("  AAC EVENTS:")
                for aac in sorted(cycle.aac_events, key=lambda e: e.date):
                    layer_tag = aac.layer.upper()[:8]
                    print(f"    [{aac.impact:>8}] {aac.date} [{layer_tag:>8}] {aac.name}")
                    if aac.assets:
                        print(f"              Assets: {', '.join(aac.assets)}")
                    if aac.description:
                        print(f"              {aac.description}")

        print()
        print("=" * 100)
        print("  PHI OVERLAY SUMMARY")
        print("=" * 100)
        for phi in self.phi_markers:
            print(
                f"  phi^{phi.phi_power:<2} = {phi.phi_value:>10.4f} | "
                f"+{phi.days_from_anchor:>7.1f} days | {phi.date} | "
                f"resonance: {phi.resonance_strength:.1%}"
            )

    def print_upcoming(self, days_ahead: int = 14) -> None:
        """Print upcoming events with lead-time actions."""
        alerts = self.get_events_with_lead_time(days_ahead)
        today = date.today()
        moon = self.get_current_moon()

        print("=" * 90)
        if moon:
            mandate = moon.doctrine_action.mandate if moon.doctrine_action else "N/A"
            print(
                f"  CURRENT: Moon {moon.moon_number} ({moon.lunar_phase_name}) | "
                f"Mandate: {mandate}"
            )
        print(f"  UPCOMING EVENTS (next {days_ahead} days from {today})")
        print("=" * 90)

        if not alerts:
            print("  No events in the next %d days." % days_ahead)
            return

        for alert in alerts:
            priority_marker = {
                "CRITICAL": "!!!",
                "HIGH": " !!",
                "MEDIUM": "  !",
                "LOW": "   ",
            }.get(alert.priority, "   ")

            print(
                f"  {priority_marker} [{alert.event_type:>10}] "
                f"{alert.event_date} (in {alert.days_until:>2}d) -- {alert.event_name}"
            )
            print(f"       ACTION: {alert.lead_time_action}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="13-Moon Doctrine Timeline")
    parser.add_argument("--timeline", action="store_true", help="Print full timeline")
    parser.add_argument("--upcoming", type=int, nargs="?", const=14, default=None,
                        help="Show upcoming events (default: 14 days)")
    parser.add_argument("--export-json", action="store_true", help="Export timeline as JSON")
    parser.add_argument("--export-html", action="store_true", help="Export interactive HTML storyboard")
    parser.add_argument("--moon", type=int, default=None, help="Show specific moon cycle (0-13)")
    parser.add_argument("--date", type=str, default=None, help="Check date (YYYY-MM-DD)")
    args = parser.parse_args()

    doctrine = ThirteenMoonDoctrine()

    if args.date:
        target = date.fromisoformat(args.date)
        moon = doctrine.get_current_moon(target)
        if moon:
            print(
                f"Date {target} falls in Moon {moon.moon_number} ({moon.lunar_phase_name})"
            )
            if moon.doctrine_action:
                print(f"Mandate: {moon.doctrine_action.mandate}")
                print(f"Action: {moon.doctrine_action.description}")
        else:
            print(f"Date {target} is outside the 13-Moon doctrine window.")

    elif args.moon is not None:
        for cycle in doctrine.moon_cycles:
            if cycle.moon_number == args.moon:
                print(f"Moon {cycle.moon_number}: {cycle.lunar_phase_name}")
                print(f"  {cycle.start_date} -> {cycle.end_date}")
                if cycle.doctrine_action:
                    print(f"  Mandate: {cycle.doctrine_action.mandate}")
                    print(f"  Action: {cycle.doctrine_action.description}")
                print(f"  Events: {len(cycle.astrology_events)} astro, "
                      f"{len(cycle.phi_markers)} phi, {len(cycle.financial_events)} financial, "
                      f"{len(cycle.world_events)} world")
                break

    elif args.export_json:
        path = doctrine.export_json()
        print(f"Exported to {path}")

    elif args.export_html:
        from strategies.thirteen_moon_storyboard import export_interactive_storyboard
        path = export_interactive_storyboard(doctrine)
        print(f"Exported to {path}")

    elif args.upcoming is not None:
        doctrine.print_upcoming(args.upcoming)

    else:
        doctrine.print_timeline()

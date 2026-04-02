#!/usr/bin/env python3
"""
Black Swan Pressure Cooker — AAC v3.1
=======================================
Real-time thesis tracker: How right are we, how fast is it accelerating?

The Thesis (March 18, 2026):
    Iran wins → US pulls out of ME → sanctions lifted → Gulf converts to yuan
    → gold reprices to $10-15k → USD collapses → petroyuan dominates

This script scores REAL observable indicators against the thesis,
tracks acceleration, and generates a semi-daily monitoring checklist.

Usage:
    python strategies/black_swan_pressure_cooker.py                # Full dashboard
    python strategies/black_swan_pressure_cooker.py --update       # Log new observation
    python strategies/black_swan_pressure_cooker.py --score-only   # Quick score
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pressure_cooker")

# ═══════════════════════════════════════════════════════════════════════════
# THESIS INDICATORS — Tiered Observable Signals
# ═══════════════════════════════════════════════════════════════════════════

class TierLevel(Enum):
    """How close to thesis confirmation. Lower = earlier warning."""
    TIER_0 = "PRECURSOR"          # Background noise that shifts toward thesis
    TIER_1 = "RHETORICAL_SHIFT"   # Diplomatic/narrative signals (1-4 week lead)
    TIER_2 = "MILITARY_OPERATIONAL"  # Troop movements, base changes (2-8 week lead)
    TIER_3 = "FINANCIAL_PAYMENT"  # Yuan flows, CIPS spikes, sanctions (4-12 week lead)
    TIER_4 = "SYSTEMIC_CONFIRM"   # Full conversion, gold rerate, USD crash (8-18 weeks)


class IndicatorCategory(Enum):
    """What domain the indicator belongs to."""
    NETANYAHU_STATUS = "netanyahu"       # The Big Red Flag
    INFO_SUPPRESSION = "info_suppress"   # Govts hiding info = things worse than stated
    WAR_ESCALATION = "war_escalation"    # Active conflict indicators
    US_RETREAT = "us_retreat"            # Signs of US pullback/defeat
    YUAN_ADOPTION = "yuan_adoption"      # Petroyuan/CIPS/Gulf conversion
    GOLD_FLIGHT = "gold_flight"          # Safe haven flows
    CREDIT_STRESS = "credit_stress"      # Private credit / financial contagion
    TRUMP_RHETORIC = "trump_rhetoric"    # The man himself — what's he saying?
    CHINA_MEDIATION = "china_mediation"  # Beijing stepping in
    GULF_PIVOT = "gulf_pivot"            # Saudi/UAE/Qatar shifting allegiance
    OIL_DISRUPTION = "oil_disruption"    # Hormuz/supply shock
    DOMESTIC_PRESSURE = "domestic_us"    # US public turning against war
    RISK_OFF = "risk_off"                # Broad risk aversion (BTC, equities)
    INFLATION = "inflation"              # Inflation/stagflation pressure
    LIQUIDITY_LOCKUP = "liquidity_lock"  # Private credit gates, fund lockups
    CAPITAL_FLIGHT = "capital_flight"    # People/money leaving US system


@dataclass
class ThesisIndicator:
    """A single observable indicator that confirms or denies the thesis."""
    id: str
    name: str
    category: IndicatorCategory
    tier: TierLevel
    description: str
    bullish_threshold: str          # What confirms the thesis
    bearish_threshold: str          # What denies it
    current_score: float = 0.0     # -1.0 (thesis wrong) to +1.0 (thesis confirmed)
    last_observation: str = ""
    last_updated: str = ""
    weight: float = 1.0            # How important (1.0 = normal, 2.0 = critical)


# ═══════════════════════════════════════════════════════════════════════════
# THE INDICATOR REGISTRY — Everything we're watching
# ═══════════════════════════════════════════════════════════════════════════

INDICATORS: List[ThesisIndicator] = [
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # THE BIG RED FLAG — Netanyahu
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="bibi_alive",
        name="Netanyahu Proof of Life",
        category=IndicatorCategory.NETANYAHU_STATUS,
        tier=TierLevel.TIER_0,
        description=(
            "Is there VERIFIED live video/audio of Netanyahu in the last 48 hours? "
            "Not archive footage, not AI-generated. Real press conference, real "
            "handshake, real interaction with verifiable date markers."
        ),
        bullish_threshold="No verified live appearance in 72+ hours; AI/deepfake rumors increase",
        bearish_threshold="Clear live press conference with verifiable date/context",
        current_score=0.3,  # March 19: proof-of-life but rumors spiking
        last_observation=(
            "March 12-17: Office released clips mocking 'fake news' death rumors. "
            "BUT — frequency of appearances declining. Rumors are INCREASING not "
            "decreasing despite 'debunking'. That's suspicious."
        ),
        last_updated="2026-03-19",
        weight=2.5,  # This is THE trigger for mega black swan
    ),
    ThesisIndicator(
        id="bibi_rumor_intensity",
        name="Netanyahu Death Rumor Intensity",
        category=IndicatorCategory.NETANYAHU_STATUS,
        tier=TierLevel.TIER_0,
        description=(
            "How intense are the rumors? Are they growing despite 'debunking'? "
            "Are mainstream outlets picking them up? Is the denial getting louder?"
        ),
        bullish_threshold="MSM covering rumors; denial becomes primary messaging; Streisand effect",
        bearish_threshold="Rumors die naturally after proof-of-life; no MSM pickup",
        current_score=0.4,
        last_observation=(
            "Rumors GREW from March 12-18 despite office denials. AI deepfake "
            "angle being pushed hard = possible narrative preparation. "
            "Office dedicated resources to 'debunking' = taking it seriously."
        ),
        last_updated="2026-03-19",
        weight=2.0,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INFORMATION SUPPRESSION — The canary in the coal mine
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="info_suppress_canada",
        name="Canada War Video Suppression Warning",
        category=IndicatorCategory.INFO_SUPPRESSION,
        tier=TierLevel.TIER_1,
        description=(
            "Canada warning citizens in ME not to record or share war videos. "
            "Governments suppress information when reality is WORSE than the "
            "narrative they're selling. This is a classic escalation indicator."
        ),
        bullish_threshold="Multiple NATO countries issue similar warnings; social media scrubbing increases",
        bearish_threshold="Warning rescinded; explained as routine consular advisory",
        current_score=0.6,
        last_observation=(
            "March 19, 2026: MSN/Canadian news — 'Canadians in Middle East warned "
            "not to record or share war videos'. This is NOT routine. You don't "
            "suppress video sharing unless what's being recorded contradicts the "
            "official narrative. Key question: WHAT are they trying to hide?"
        ),
        last_updated="2026-03-19",
        weight=1.8,
    ),
    ThesisIndicator(
        id="info_suppress_western",
        name="Western Media Narrative Control",
        category=IndicatorCategory.INFO_SUPPRESSION,
        tier=TierLevel.TIER_1,
        description=(
            "Are Western media outlets restricting ME war coverage? "
            "Social media platform censorship of war content? "
            "Journalist access being limited?"
        ),
        bullish_threshold="Multiple platforms scrubbing content; journalist access revoked; embed restrictions",
        bearish_threshold="Free press access; live reporting from conflict zones",
        current_score=0.35,
        last_observation=(
            "Canada video warning is the first concrete NATO-country suppression signal. "
            "Watch for: US/UK/EU similar advisories, X/Meta content restrictions, "
            "journalist detention or 'safety' removals from theater."
        ),
        last_updated="2026-03-19",
        weight=1.5,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # WAR ESCALATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="war_duration",
        name="War Duration & Intensity",
        category=IndicatorCategory.WAR_ESCALATION,
        tier=TierLevel.TIER_0,
        description="How long has the US-Iran war been going? Longer = more drain = closer to thesis.",
        bullish_threshold="War exceeds 60 days with no clear US victory; escalation continues",
        bearish_threshold="Swift US victory (<30 days); Iran capitulates",
        current_score=0.5,
        last_observation=(
            "Day 19 (Feb 28 - March 19). Ongoing. No capitulation either side. "
            "US strikes continue. Iran retaliating on bases, Gulf, Israel. "
            "This is already longer than most predicted. Each week = more thesis pressure."
        ),
        last_updated="2026-03-19",
        weight=1.3,
    ),
    ThesisIndicator(
        id="hormuz_status",
        name="Strait of Hormuz Disruption",
        category=IndicatorCategory.OIL_DISRUPTION,
        tier=TierLevel.TIER_2,
        description="Is Hormuz blocked/restricted? 21% of global oil transits here.",
        bullish_threshold="Full or partial blockade; tanker rerouting; insurance premiums spike 5x+",
        bearish_threshold="Open transit; Iranian navy retreats; US secures strait",
        current_score=0.7,
        last_observation=(
            "Partial disruption ongoing. Iran proposed yuan-only passage for select "
            "countries. Insurance premiums elevated. This is ALREADY a yuan adoption "
            "vector — Iran is weaponizing access for currency conversion."
        ),
        last_updated="2026-03-19",
        weight=1.8,
    ),
    ThesisIndicator(
        id="iran_retaliation_effectiveness",
        name="Iranian Retaliation Success",
        category=IndicatorCategory.WAR_ESCALATION,
        tier=TierLevel.TIER_1,
        description="Are Iranian strikes hitting targets? Causing real damage?",
        bullish_threshold="Confirmed hits on US bases, Israeli infrastructure; casualties mounting",
        bearish_threshold="All intercepted; no real damage; purely symbolic",
        current_score=0.45,
        last_observation=(
            "Missiles hitting Gulf states, Israel, US bases. Some damage confirmed "
            "but full extent unclear (see: info suppression). Israeli economic hit "
            "estimated ~$3B/week. US base casualties being underreported?"
        ),
        last_updated="2026-03-19",
        weight=1.3,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # US RETREAT SIGNALS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="us_base_withdrawals",
        name="US Base Evacuation/Drawdown",
        category=IndicatorCategory.US_RETREAT,
        tier=TierLevel.TIER_2,
        description="Are US troops leaving ME bases? Al Udeid, Bahrain, UAE, Syria?",
        bullish_threshold="Troop numbers drop >50% in 30 days; 'transition' language",
        bearish_threshold="Troop buildup; reinforcements; permanent basing announcements",
        current_score=0.25,
        last_observation=(
            "Precautionary repositionings from Al Udeid, Syria Qasrak pullout (Feb 2026). "
            "Not full withdrawal yet. Watch for: 'conditions-based transition' language, "
            "family evacuation orders, equipment drawdowns."
        ),
        last_updated="2026-03-19",
        weight=1.5,
    ),
    ThesisIndicator(
        id="us_war_cost_pressure",
        name="US Domestic War Cost Pressure",
        category=IndicatorCategory.DOMESTIC_PRESSURE,
        tier=TierLevel.TIER_1,
        description="Is the US public/Congress turning against the war? Cost complaints?",
        bullish_threshold="Bipartisan calls to end war; cost estimates dominate news cycle; polling <40% support",
        bearish_threshold="Strong public support; Congress authorizes expanded operations",
        current_score=0.3,
        last_observation=(
            "Week 3: Cost stories starting to appear. No major anti-war movement yet. "
            "Watch for: congressional hearings on cost, polling data, protest coverage."
        ),
        last_updated="2026-03-19",
        weight=1.0,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRUMP RHETORIC — The Man Himself
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="trump_rhetoric",
        name="Trump Tone on Iran/ME",
        category=IndicatorCategory.TRUMP_RHETORIC,
        tier=TierLevel.TIER_1,
        description="What is Trump saying? 'Unconditional surrender' vs 'deal' vs 'peace'",
        bullish_threshold="Softening to 'deal'/'peace'/'enough'; apology-adjacent language",
        bearish_threshold="Escalation rhetoric; 'total victory'; new strike authorizations",
        current_score=-0.1,
        last_observation=(
            "As of March 18: Still 'unconditional surrender' rhetoric. Doubling down "
            "on strikes. BUT — watch for subtle shifts: if he stops tweeting about it "
            "entirely, that's a tell. Silence = recalculation."
        ),
        last_updated="2026-03-19",
        weight=2.0,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # YUAN ADOPTION / PETROYUAN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="hormuz_yuan_passage",
        name="Iran Hormuz Yuan-Only Passage",
        category=IndicatorCategory.YUAN_ADOPTION,
        tier=TierLevel.TIER_2,
        description=(
            "Iran's proposal: you can transit Hormuz if you pay in yuan. "
            "This is literally weaponizing geography for currency adoption."
        ),
        bullish_threshold="Proposal accepted by 3+ countries; China confirms routing",
        bearish_threshold="Proposal rejected universally; US Navy secures strait",
        current_score=0.55,
        last_observation=(
            "Proposal active and being negotiated. Limited acceptance. "
            "THIS IS ALREADY HAPPENING — even partial adoption is a foot in the door. "
            "Watch for: China confirming routing, India/Japan/Korea accepting terms."
        ),
        last_updated="2026-03-19",
        weight=1.8,
    ),
    ThesisIndicator(
        id="cips_volume",
        name="CIPS Transaction Volume Spike",
        category=IndicatorCategory.YUAN_ADOPTION,
        tier=TierLevel.TIER_3,
        description="China's SWIFT alternative. Volume spikes = yuan adoption accelerating.",
        bullish_threshold="CIPS monthly volume >$15T; new Gulf bank connections",
        bearish_threshold="Flat or declining volumes; no new institutional adoption",
        current_score=0.3,
        last_observation="Incremental growth. No public Gulf bank mega-announcements yet.",
        last_updated="2026-03-19",
        weight=1.3,
    ),
    ThesisIndicator(
        id="gulf_yuan_oil",
        name="Gulf States Yuan Oil Contracts",
        category=IndicatorCategory.GULF_PIVOT,
        tier=TierLevel.TIER_3,
        description="Saudi/UAE/Qatar pricing oil in yuan instead of USD.",
        bullish_threshold="Aramco announces >25% yuan contracts; official GCC statement",
        bearish_threshold="Aramco reaffirms 100% USD; Saudi doubles down on petrodollar",
        current_score=0.2,
        last_observation=(
            "Limited swaps exist. No full endorsement. Iran conflict creates pressure "
            "on Gulf states to hedge — every day the war continues, the incentive to "
            "diversify away from USD grows."
        ),
        last_updated="2026-03-19",
        weight=1.5,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GOLD & SAFE HAVEN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="gold_price",
        name="Gold Spot Price",
        category=IndicatorCategory.GOLD_FLIGHT,
        tier=TierLevel.TIER_2,
        description="Gold above $5k = risk-off flight already happening.",
        bullish_threshold="Gold >$7k; SGE premium over LBMA; central bank buying accelerates",
        bearish_threshold="Gold falls below $4k; risk-on returns; crypto absorbs safe haven",
        current_score=0.6,
        last_observation=(
            "Gold at $5,011 as of March 18. ABOVE $5k. This is already historic. "
            "The thesis says $10-12k under black swan. We're halfway up the ramp."
        ),
        last_updated="2026-03-19",
        weight=1.5,
    ),
    ThesisIndicator(
        id="sge_hk_volume",
        name="SGE HK Corridor Volume",
        category=IndicatorCategory.GOLD_FLIGHT,
        tier=TierLevel.TIER_3,
        description="Shanghai Gold Exchange via Hong Kong — the physical gold pipeline.",
        bullish_threshold="Daily volumes double; Saudi hub activation; >500t/month throughput",
        bearish_threshold="Volumes flat; no Saudi hub progress",
        current_score=0.35,
        last_observation="Volumes elevated but no explosive spike yet. Watch for Saudi hub news.",
        last_updated="2026-03-19",
        weight=1.3,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CREDIT STRESS (Existing positions)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="credit_spreads",
        name="HY Credit Spread Widening",
        category=IndicatorCategory.CREDIT_STRESS,
        tier=TierLevel.TIER_2,
        description="High yield spreads widening = credit contagion building.",
        bullish_threshold="HY spreads >500bps; fund gates/redemption caps multiply",
        bearish_threshold="Spreads tighten <300bps; fund flows normalize",
        current_score=0.5,
        last_observation=(
            "HY spreads at ~380bps. Private credit funds capping redemptions. "
            "JPMorgan marking down portfolios. This feeds directly into your 8 "
            "live put positions (LQD, KRE, IWM, JNK, EMB, etc.)"
        ),
        last_updated="2026-03-19",
        weight=1.5,
    ),
    ThesisIndicator(
        id="dollar_weakness",
        name="USD Index Weakness",
        category=IndicatorCategory.YUAN_ADOPTION,
        tier=TierLevel.TIER_2,
        description="Dollar weakening = confidence eroding in US financial supremacy.",
        bullish_threshold="DXY drops below 95; worst weekly decline in 2+ years",
        bearish_threshold="DXY strengthens above 105; flight TO dollar safety",
        current_score=0.4,
        last_observation=(
            "Dollar had its worst day in over a month recently. Weakening trend "
            "but not collapse. Each day of war = more dollar confidence erosion."
        ),
        last_updated="2026-03-19",
        weight=1.2,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHINA MEDIATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="china_mediation",
        name="China Mediation Moves",
        category=IndicatorCategory.CHINA_MEDIATION,
        tier=TierLevel.TIER_1,
        description="Beijing positioning as peacemaker = yuan leverage play.",
        bullish_threshold="Xi calls Trump; Beijing hosts emergency Gulf summit; public framework",
        bearish_threshold="China stays neutral; no mediation offers",
        current_score=0.2,
        last_observation="Hints of mediation but no formal announcement. Watch for summit.",
        last_updated="2026-03-19",
        weight=1.5,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RISK-OFF / CRYPTO DUMP (March 19 signal)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="btc_risk_off",
        name="Bitcoin Risk-Off Dump",
        category=IndicatorCategory.RISK_OFF,
        tier=TierLevel.TIER_1,
        description=(
            "BTC dumping on risk aversion = smart money fleeing to safety. "
            "Crypto is the canary — when BTC dumps on war news, it means "
            "real fear is entering the system, not just hedging."
        ),
        bullish_threshold="BTC drops >10% in a week; correlation with equities tightens; volume spikes on sell side",
        bearish_threshold="BTC recovers quickly; decouples from war news; acts as safe haven",
        current_score=0.6,
        last_observation=(
            "March 19: BTC fell 5.4% to ~$70,500 — biggest drop in 3 weeks. "
            "Triggered by Iran hitting Qatar LNG site. ETH/SOL down 6%. "
            "Risk-off across ALL asset classes. RIOT -4%, COIN -3.8%. "
            "Fed held rates = no relief valve. Oil rising = pressure stays."
        ),
        last_updated="2026-03-19",
        weight=1.5,
    ),
    ThesisIndicator(
        id="equities_risk_off",
        name="Equity Market Risk Aversion",
        category=IndicatorCategory.RISK_OFF,
        tier=TierLevel.TIER_1,
        description="Broad equity sell-off on war/crisis fear. VIX rising. Defensive rotation.",
        bullish_threshold="SPY down >5% from recent high; VIX >30; defensive sectors outperform",
        bearish_threshold="Markets shrug off war; new highs; VIX below 15",
        current_score=0.45,
        last_observation=(
            "March 19: Broad risk-off after Iran Qatar LNG strike. VIX at 21.48. "
            "Not panic yet but fear is building. Each escalation ratchets higher."
        ),
        last_updated="2026-03-19",
        weight=1.3,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE CREDIT LIQUIDITY LOCKUP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="private_credit_lockup",
        name="Private Credit Liquidity Lockup",
        category=IndicatorCategory.LIQUIDITY_LOCKUP,
        tier=TierLevel.TIER_2,
        description=(
            "Funds gating redemptions = money is TRAPPED. When investors "
            "can't get out, forced sellers appear elsewhere. Contagion vector."
        ),
        bullish_threshold="3+ major funds gate; redemption queues >30 days; secondary market discounts >15%",
        bearish_threshold="Gates lifted; redemptions normalize; no new restrictions",
        current_score=0.65,
        last_observation=(
            "Blackstone BCRED, Morgan Stanley PIF, BlackRock HLEND, Blue Owl ALL "
            "restricting withdrawals. JPMorgan marking down private credit portfolios. "
            "Partners Group warns default rate heading >5%. This is EXACTLY the "
            "credit freeze pattern from 2008 — it starts slow then accelerates."
        ),
        last_updated="2026-03-19",
        weight=2.0,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INFLATION / FED TRAPPED
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="inflation_soaring",
        name="Inflation / Stagflation Spiral",
        category=IndicatorCategory.INFLATION,
        tier=TierLevel.TIER_2,
        description=(
            "Core PCE 3.1%, GDP 0.7% = textbook stagflation. Oil above $95 "
            "feeds into EVERYTHING — transport, food, manufacturing. "
            "War makes it worse every single day."
        ),
        bullish_threshold="Core PCE >3.5%; GDP <0.5%; oil >$110; consumer sentiment collapses",
        bearish_threshold="PCE drops below 2.5%; GDP rebounds >2%; oil normalizes",
        current_score=0.7,
        last_observation=(
            "March 19: Core PCE 3.1%, GDP 0.7% — stagflation CONFIRMED. "
            "Oil surging after Iran hit Qatar LNG = energy costs accelerating. "
            "Fed held rates today = TRAPPED. Can't cut (inflation too hot), "
            "can't hike (economy too weak). This is the worst macro setup."
        ),
        last_updated="2026-03-19",
        weight=1.8,
    ),
    ThesisIndicator(
        id="fed_trapped",
        name="Fed Policy Paralysis",
        category=IndicatorCategory.INFLATION,
        tier=TierLevel.TIER_2,
        description=(
            "Fed can't cut (inflation), can't hike (recession). "
            "Second rate cut pushed to December. Markets get no relief."
        ),
        bullish_threshold="Fed skips 3+ meetings; 'higher for longer' language; rate cut expectations collapse",
        bearish_threshold="Fed cuts 50bps; dovish pivot; inflation drops fast",
        current_score=0.6,
        last_observation=(
            "March 19: Fed held rates steady. Second cut pushed to December. "
            "No dovish pivot. Oil rising = inflation expectations rising. "
            "BTC dumped AFTER FOMC — no risk-on relief. Markets realize: no help coming."
        ),
        last_updated="2026-03-19",
        weight=1.5,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # IRAN ESCALATION — Qatar LNG Hit
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="iran_qatar_lng",
        name="Iran Strikes Gulf Energy Infrastructure",
        category=IndicatorCategory.WAR_ESCALATION,
        tier=TierLevel.TIER_2,
        description=(
            "Iran hitting Qatar LNG + Dubai airport fuel tank = targeting "
            "Gulf state ENERGY infrastructure, not just military. This is "
            "economic warfare on US allies."
        ),
        bullish_threshold="More Gulf infrastructure hits; Aramco/ADNOC targeted; energy exports disrupted",
        bearish_threshold="Strikes limited to military; no economic infrastructure damage",
        current_score=0.75,
        last_observation=(
            "March 19: Iran drone struck fuel tank near Dubai Int'l Airport. "
            "Iran attacked key LNG site in Qatar. These are ECONOMIC targets — "
            "not military. Iran is telling Gulf states: your infrastructure "
            "is not safe while you host US bases. This is coercion toward the thesis."
        ),
        last_updated="2026-03-19",
        weight=2.0,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CAPITAL FLIGHT — US Citizenship Renunciation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ThesisIndicator(
        id="us_citizenship_renounce",
        name="US Citizenship Renunciation Fee Slashed",
        category=IndicatorCategory.CAPITAL_FLIGHT,
        tier=TierLevel.TIER_0,
        description=(
            "State Dept cut renunciation fee 80% ($2,350 to $450). "
            "They don't make it EASIER to leave unless demand is surging "
            "or they want to signal openness to exodus. Background noise "
            "but directionally telling."
        ),
        bullish_threshold="Renunciation applications surge >50% YoY; expat tax reform; capital controls hint",
        bearish_threshold="Purely administrative; no demand surge; routine policy",
        current_score=0.25,
        last_observation=(
            "March 19: State Dept slashed renunciation fee 80% to $450. "
            "Effective April 13. Fee was raised in 2015 when renunciations surged. "
            "Now cutting it = either demand overwhelming or signal. Minor indicator "
            "but adds to the mosaic of US system stress."
        ),
        last_updated="2026-03-19",
        weight=0.7,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# SCORING ENGINE — The Pressure Cooker
# ═══════════════════════════════════════════════════════════════════════════

class PressureCooker:
    """
    Scores the black swan thesis against reality.
    Produces a "pressure" reading: how much thesis-confirming pressure
    is building in the system.
    """

    def __init__(self, indicators: List[ThesisIndicator]):
        self.indicators = {i.id: i for i in indicators}
        self.history_file = Path(__file__).parent.parent / "data" / "pressure_cooker_history.jsonl"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def compute_pressure(self) -> Dict:
        """Compute the overall pressure reading."""
        total_weighted = 0.0
        total_weight = 0.0
        category_scores: Dict[str, List[float]] = {}
        tier_scores: Dict[str, List[float]] = {}

        for ind in self.indicators.values():
            weighted = ind.current_score * ind.weight
            total_weighted += weighted
            total_weight += ind.weight

            cat = ind.category.value
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(ind.current_score)

            tier = ind.tier.value
            if tier not in tier_scores:
                tier_scores[tier] = []
            tier_scores[tier].append(ind.current_score)

        overall = total_weighted / total_weight if total_weight > 0 else 0

        # Category averages
        cat_avgs = {k: sum(v) / len(v) for k, v in category_scores.items()}
        tier_avgs = {k: sum(v) / len(v) for k, v in tier_scores.items()}

        # How many indicators are bullish (>0.4)?
        bullish = sum(1 for i in self.indicators.values() if i.current_score > 0.4)
        neutral = sum(1 for i in self.indicators.values() if -0.2 <= i.current_score <= 0.4)
        bearish = sum(1 for i in self.indicators.values() if i.current_score < -0.2)
        total = len(self.indicators)

        # Thesis ratio: bullish / (bullish + bearish) — ignoring neutral
        sided = bullish + bearish
        ratio = bullish / sided if sided > 0 else 0.5

        # Probability estimate based on current pressure
        if overall >= 0.7:
            prob_desc = "HIGH (30-50%) — Multiple tiers confirming"
        elif overall >= 0.5:
            prob_desc = "ELEVATED (15-30%) — Tier 1-2 signals active"
        elif overall >= 0.3:
            prob_desc = "MODERATE (5-15%) — Precursors building"
        elif overall >= 0.1:
            prob_desc = "LOW (2-5%) — Thesis alive but unconfirmed"
        else:
            prob_desc = "MINIMAL (<2%) — Thesis not supported"

        return {
            "overall_pressure": overall,
            "prob_description": prob_desc,
            "bullish_count": bullish,
            "neutral_count": neutral,
            "bearish_count": bearish,
            "total_indicators": total,
            "thesis_ratio": ratio,
            "category_averages": cat_avgs,
            "tier_averages": tier_avgs,
            "timestamp": datetime.now().isoformat(),
        }

    def update_indicator(self, indicator_id: str, score: float, observation: str):
        """Update a single indicator with a new observation."""
        if indicator_id not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_id}")
        ind = self.indicators[indicator_id]
        old_score = ind.current_score
        ind.current_score = max(-1.0, min(1.0, score))
        ind.last_observation = observation
        ind.last_updated = datetime.now().strftime("%Y-%m-%d")

        # Log to history
        entry = {
            "ts": datetime.now().isoformat(),
            "id": indicator_id,
            "old": old_score,
            "new": score,
            "obs": observation,
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return old_score, ind.current_score

    def log_daily_ratio(self):
        """Log today's ratio to the trend file for daily tracking."""
        ratio_file = Path(__file__).parent.parent / "data" / "pressure_cooker_ratio_trend.jsonl"
        ratio_file.parent.mkdir(parents=True, exist_ok=True)
        pressure = self.compute_pressure()
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "war_day": (datetime.now() - datetime(2026, 2, 28)).days,
            "pressure": round(pressure["overall_pressure"], 4),
            "ratio": round(pressure["thesis_ratio"], 4),
            "bullish": pressure["bullish_count"],
            "neutral": pressure["neutral_count"],
            "bearish": pressure["bearish_count"],
            "total": pressure["total_indicators"],
        }
        with open(ratio_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def get_ratio_trend(self) -> List[Dict]:
        """Load the historical ratio trend."""
        ratio_file = Path(__file__).parent.parent / "data" / "pressure_cooker_ratio_trend.jsonl"
        if not ratio_file.exists():
            return []
        entries = []
        with open(ratio_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def generate_ratio_section(self) -> str:
        """Generate the ratio trend display for the dashboard."""
        pressure = self.compute_pressure()
        ratio = pressure["thesis_ratio"]
        bullish = pressure["bullish_count"]
        bearish = pressure["bearish_count"]
        neutral = pressure["neutral_count"]
        total = pressure["total_indicators"]

        lines = []
        lines.append("")
        lines.append("=" * 78)
        lines.append("  THESIS RATIO — Positive vs Negative Indicators")
        lines.append("=" * 78)
        lines.append("")

        # Big ratio display
        ratio_bar_len = 50
        bull_chars = int(ratio * ratio_bar_len)
        bear_chars = ratio_bar_len - bull_chars
        ratio_bar = "▓" * bull_chars + "░" * bear_chars

        lines.append(f"  FOR thesis  [{ratio_bar}]  AGAINST")
        lines.append(f"     {bullish:>2} bull                                             {bearish} bear")
        lines.append("")
        lines.append(f"  RATIO:  {bullish} FOR  /  {bearish} AGAINST  /  {neutral} UNDECIDED  =  {ratio:.0%} CONFIRMING")
        lines.append("")

        # Trend history
        trend = self.get_ratio_trend()
        if trend:
            lines.append("  DAILY TREND (watch this march toward 100%):")
            lines.append("  " + "-" * 74)
            # Show last 14 entries max
            for entry in trend[-14:]:
                day = entry.get("war_day", "?")
                date = entry.get("date", "?")
                r = entry.get("ratio", 0)
                p = entry.get("pressure", 0)
                b = entry.get("bullish", 0)
                bear = entry.get("bearish", 0)
                bar_w = 30
                filled = int(r * bar_w)
                bar = "█" * filled + "░" * (bar_w - filled)
                lines.append(
                    f"    Day {day:>3} ({date}) [{bar}] {r:.0%}  "
                    f"pressure={p:.0%}  {b}B/{bear}A"
                )

            # Trend direction
            if len(trend) >= 2:
                prev = trend[-2]["ratio"]
                curr = trend[-1]["ratio"]
                delta = curr - prev
                if delta > 0.02:
                    direction = "ACCELERATING toward thesis ▲▲"
                elif delta > 0:
                    direction = "Drifting toward thesis ▲"
                elif delta < -0.02:
                    direction = "Moving AWAY from thesis ▼▼"
                elif delta < 0:
                    direction = "Slight drift away ▼"
                else:
                    direction = "Holding steady ─"
                lines.append(f"\n    TREND: {direction} (delta {delta:+.1%})")
        else:
            lines.append("  No trend data yet — run daily to build history.")

        lines.append("")
        return "\n".join(lines)

    def get_biggest_movers(self, since_hours: int = 24) -> List[Dict]:
        """Check history for biggest score changes in last N hours."""
        movers = []
        cutoff = datetime.now() - timedelta(hours=since_hours)

        if not self.history_file.exists():
            return movers

        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["ts"])
                if ts >= cutoff:
                    delta = entry["new"] - entry["old"]
                    if abs(delta) > 0.05:
                        movers.append({
                            "id": entry["id"],
                            "delta": delta,
                            "new_score": entry["new"],
                            "observation": entry["obs"],
                            "time": entry["ts"],
                        })

        movers.sort(key=lambda m: abs(m["delta"]), reverse=True)
        return movers

    def generate_dashboard(self) -> str:
        """Generate the full pressure cooker dashboard."""
        pressure = self.compute_pressure()
        lines = []

        # ── Header ──
        lines.append("")
        lines.append("=" * 78)
        lines.append("  BLACK SWAN PRESSURE COOKER — AAC THESIS TRACKER")
        lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Day {(datetime.now() - datetime(2026, 2, 28)).days} of Iran War")
        lines.append("=" * 78)

        # ── Overall Pressure Gauge ──
        overall = pressure["overall_pressure"]
        bar_len = 40
        filled = int(overall * bar_len) if overall > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        lines.append("")
        lines.append(f"  THESIS PRESSURE: [{bar}] {overall:.0%}")
        lines.append(f"  PROBABILITY:     {pressure['prob_description']}")
        lines.append(f"  INDICATORS:      {pressure['bullish_count']} BULLISH / "
                      f"{pressure['neutral_count']} NEUTRAL / "
                      f"{pressure['bearish_count']} BEARISH "
                      f"(of {pressure['total_indicators']})")
        ratio = pressure["thesis_ratio"]
        lines.append(f"  THESIS RATIO:    {ratio:.0%} of sided indicators CONFIRM thesis")
        lines.append("")

        # ── THE BIG RED FLAG ──
        lines.append("  " + "━" * 74)
        lines.append("  🚨 THE BIG RED FLAG — NETANYAHU STATUS")
        lines.append("  " + "━" * 74)
        for ind in self.indicators.values():
            if ind.category == IndicatorCategory.NETANYAHU_STATUS:
                score_bar = self._mini_bar(ind.current_score)
                lines.append(f"    {score_bar} {ind.name}")
                lines.append(f"         {ind.last_observation[:120]}")
                lines.append("")

        # ── NEW SIGNAL: Info Suppression ──
        lines.append("  " + "━" * 74)
        lines.append("  🔇 INFORMATION SUPPRESSION — What are they hiding?")
        lines.append("  " + "━" * 74)
        for ind in self.indicators.values():
            if ind.category == IndicatorCategory.INFO_SUPPRESSION:
                score_bar = self._mini_bar(ind.current_score)
                lines.append(f"    {score_bar} {ind.name}")
                lines.append(f"         {ind.last_observation[:120]}")
                lines.append("")

        # ── Category Breakdown ──
        lines.append("  " + "━" * 74)
        lines.append("  CATEGORY PRESSURE:")
        lines.append("  " + "━" * 74)

        cat_display = {
            "netanyahu": "Netanyahu Status    ",
            "info_suppress": "Info Suppression    ",
            "war_escalation": "War Escalation      ",
            "oil_disruption": "Oil/Hormuz Disrupt. ",
            "us_retreat": "US Retreat Signals  ",
            "trump_rhetoric": "Trump Rhetoric      ",
            "yuan_adoption": "Yuan Adoption       ",
            "gold_flight": "Gold/Safe Haven     ",
            "credit_stress": "Credit Stress       ",
            "liquidity_lock": "Private Credit Lock ",
            "inflation": "Inflation/Stagflation",
            "risk_off": "Risk-Off / Crypto   ",
            "capital_flight": "Capital Flight      ",
            "china_mediation": "China Mediation     ",
            "gulf_pivot": "Gulf State Pivot    ",
            "domestic_us": "US Domestic Pressure",
        }

        for cat_key, cat_label in cat_display.items():
            avg = pressure["category_averages"].get(cat_key, 0)
            bar = self._mini_bar(avg)
            lines.append(f"    {bar} {cat_label}")

        # ── Tier Breakdown ──
        lines.append("")
        lines.append("  " + "━" * 74)
        lines.append("  TIER READINGS (Earlier = Longer Lead Time):")
        lines.append("  " + "━" * 74)

        tier_display = {
            "PRECURSOR": "Tier 0 PRECURSOR     ",
            "RHETORICAL_SHIFT": "Tier 1 RHETORIC      ",
            "MILITARY_OPERATIONAL": "Tier 2 MILITARY/OPS  ",
            "FINANCIAL_PAYMENT": "Tier 3 FINANCIAL     ",
            "SYSTEMIC_CONFIRM": "Tier 4 SYSTEMIC      ",
        }
        for tier_key, tier_label in tier_display.items():
            avg = pressure["tier_averages"].get(tier_key, 0)
            bar = self._mini_bar(avg)
            lines.append(f"    {bar} {tier_label}")

        # ── All Indicators Detail ──
        lines.append("")
        lines.append("  " + "━" * 74)
        lines.append("  ALL INDICATORS (sorted by score):")
        lines.append("  " + "━" * 74)

        sorted_inds = sorted(self.indicators.values(),
                             key=lambda i: i.current_score, reverse=True)
        for ind in sorted_inds:
            bar = self._mini_bar(ind.current_score)
            wt = f"x{ind.weight:.1f}" if ind.weight != 1.0 else "    "
            lines.append(f"    {bar} {wt} [{ind.tier.value[:4]}] {ind.name}")

        # ── Semi-Daily Monitoring Checklist ──
        lines.append("")
        lines.append("=" * 78)
        lines.append("  SEMI-DAILY MONITORING CHECKLIST (Check AM + PM)")
        lines.append("=" * 78)
        lines.append("")
        lines.append("  MORNING (Before market open):")
        lines.append("    [ ] Netanyahu: Any new live appearance in last 12 hours?")
        lines.append("    [ ] Trump Truth Social: Any Iran/ME posts? Tone shift?")
        lines.append("    [ ] Oil price: Brent above/below $100?")
        lines.append("    [ ] Gold price: Above/below $5,000?")
        lines.append("    [ ] VIX level: Rising or falling?")
        lines.append("    [ ] Overnight ME news: Strikes? Casualties? Base activity?")
        lines.append("    [ ] China/Gulf news: Any mediation or yuan announcements?")
        lines.append("    [ ] Info suppression: New country warnings? Social media censorship?")
        lines.append("")
        lines.append("  EVENING (After market close):")
        lines.append("    [ ] Put positions P&L: LQD/KRE/IWM/JNK/EMB/ARCC/PFF/MAIN")
        lines.append("    [ ] Credit spreads: HY/IG widening or tightening?")
        lines.append("    [ ] US troop movements: CENTCOM updates, base news")
        lines.append("    [ ] Hormuz tanker count: Reuters shipping data")
        lines.append("    [ ] CIPS/SWIFT: Any transaction volume news")
        lines.append("    [ ] Congressional activity: Any anti-war hearings/statements")
        lines.append("    [ ] SGE HK volumes: Any spike in physical delivery")
        lines.append("    [ ] Netanyahu rumor scan: Reddit/X/Telegram sentiment")
        lines.append("")

        # ── How Right Are You? ──
        lines.append("=" * 78)
        lines.append("  HOW RIGHT ARE YOU? — Thesis Scorecard")
        lines.append("=" * 78)
        lines.append("")

        confirmed = [
            "War ongoing 19+ days (predicted prolonged conflict)",
            "Gold above $5,000 (predicted safe haven flight)",
            "Hormuz partially disrupted (predicted oil chokepoint)",
            "Iran yuan-passage proposal active (predicted petroyuan catalyst)",
            "Private credit funds gating redemptions (predicted credit stress)",
            "Dollar weakening trend (predicted USD confidence erosion)",
            "HY spreads elevated at 380bps (predicted credit contagion)",
            "Oil above $95/bbl (predicted energy shock)",
            "Netanyahu rumors growing despite denials (predicted info warfare)",
            "Canada suppressing war video sharing (predicted info control)",
            "Bitcoin -5.4% risk-off dump on Iran escalation (predicted risk contagion)",
            "Iran targeting Gulf ECONOMIC infrastructure — Qatar LNG, Dubai airport",
            "Fed trapped — held rates, can't cut or hike (stagflation confirmed)",
            "Private credit lockup — 4 major funds gating redemptions simultaneously",
            "Core PCE 3.1% + GDP 0.7% = textbook stagflation (predicted)",
            "21 arrested in UAE for filming strikes (info suppression confirmed)",
        ]

        not_yet = [
            "Trump rhetoric shift (still 'unconditional surrender')",
            "Full US base withdrawal (only minor repositioning so far)",
            "Gulf states formal yuan announcement (only incremental swaps)",
            "CIPS mega-spike (gradual growth, no explosion yet)",
            "China formal mediation initiative (hints only)",
            "Gold above $7,000 (at $5,011 — 40% of the way)",
            "SGE HK volume explosion (elevated but no breakout)",
            "Netanyahu confirmed dead or incapacitated (alive but suspicious)",
            "VIX above 30 (at 21.48 — elevated but not panic)",
            "Aramco/ADNOC directly targeted by Iran (only Qatar/Dubai so far)",
        ]

        lines.append(f"  CONFIRMED ({len(confirmed)}/{len(confirmed)+len(not_yet)}):")
        for i, c in enumerate(confirmed, 1):
            lines.append(f"    ✓ {i:2d}. {c}")

        lines.append("")
        lines.append(f"  NOT YET ({len(not_yet)}/{len(confirmed)+len(not_yet)}):")
        for i, n in enumerate(not_yet, 1):
            lines.append(f"    ○ {i:2d}. {n}")

        pct_right = len(confirmed) / (len(confirmed) + len(not_yet)) * 100
        lines.append("")
        lines.append(f"  THESIS ACCURACY: {pct_right:.0f}% of indicators confirming")
        lines.append(f"  ACCELERATION:    Day 19 — faster than base case predicted")
        lines.append("")

        # ── Position Connection ──
        lines.append("=" * 78)
        lines.append("  YOUR LIVE POSITIONS vs THESIS")
        lines.append("=" * 78)
        lines.append("")
        lines.append("  8 Puts on IBKR (Mar 18, $910 deployed):")
        lines.append("    ARCC $17P  — Private credit BDC → credit_stress ✓")
        lines.append("    PFF  $29P  — Preferred shares → rate sensitivity ✓")
        lines.append("    LQD  $106P — Investment grade → credit contagion ✓")
        lines.append("    EMB  $90P  — EM bonds → dollar weakness + oil shock ✓")
        lines.append("    MAIN $50P  — Private credit BDC → credit collapse ✓")
        lines.append("    JNK  $92P  — Junk bonds → HY spread widening ✓")
        lines.append("    KRE  $58P  — Regional banks → CRE + loan losses ✓")
        lines.append("    IWM  $230P — Small caps → rate + credit sensitive ✓")
        lines.append("")
        lines.append("  ALL 8 POSITIONS ALIGN with thesis vectors. If black swan")
        lines.append("  accelerates, these puts are in the direct blast zone.")
        lines.append("")

        # ── Ratio Trend Section ──
        lines.append(self.generate_ratio_section())

        lines.append("=" * 78)

        return "\n".join(lines)

    def _get_authority_consensus(self) -> Optional[Dict]:
        """Fetch authority consensus from the authority monitor cache (non-blocking)."""
        try:
            from strategies.blackswan_authority_monitor import get_authority_consensus
            data = get_authority_consensus()
            if data and data.get("total_signals", 0) > 0:
                return {
                    "consensus": data.get("consensus", "UNKNOWN"),
                    "consensus_score": data.get("consensus_score", 0.0),
                    "agreement_pct": data.get("agreement_pct", 0.0),
                    "total_signals": data.get("total_signals", 0),
                    "total_authorities": data.get("total_authorities", 0),
                }
        except Exception:
            pass
        return None

    def get_crisis_data(self) -> Dict:
        """Return structured crisis center data for API/JSON consumers."""
        pressure = self.compute_pressure()
        war_day = (datetime.now() - datetime(2026, 2, 28)).days

        # Top 5 hottest indicators
        ranked = sorted(self.indicators.values(), key=lambda i: i.current_score, reverse=True)
        top5 = [
            {"id": i.id, "name": i.name, "score": i.current_score,
             "category": i.category.value, "tier": i.tier.value}
            for i in ranked[:5]
        ]

        # Bottom/contrarian indicators
        contrarian = [
            {"id": i.id, "name": i.name, "score": i.current_score}
            for i in ranked if i.current_score < 0
        ]

        trend = self.get_ratio_trend()
        prev_ratio = trend[-2]["ratio"] if len(trend) >= 2 else None
        curr_ratio = pressure["thesis_ratio"]
        if prev_ratio is not None:
            delta = curr_ratio - prev_ratio
            if delta > 0.02:
                direction = "ACCELERATING"
            elif delta > 0:
                direction = "DRIFTING_FOR"
            elif delta < -0.02:
                direction = "RETREATING"
            elif delta < 0:
                direction = "DRIFTING_AGAINST"
            else:
                direction = "STEADY"
        else:
            delta = 0.0
            direction = "FIRST_READING"

        return {
            "status": "ok",
            "war_day": war_day,
            "pressure": round(pressure["overall_pressure"], 4),
            "pressure_pct": f"{pressure['overall_pressure']:.0%}",
            "probability": pressure["prob_description"],
            "thesis_ratio": round(curr_ratio, 4),
            "ratio_pct": f"{curr_ratio:.0%}",
            "ratio_delta": round(delta, 4),
            "trend_direction": direction,
            "bullish": pressure["bullish_count"],
            "neutral": pressure["neutral_count"],
            "bearish": pressure["bearish_count"],
            "total": pressure["total_indicators"],
            "top5_indicators": top5,
            "contrarian_indicators": contrarian,
            "trend_entries": len(trend),
            "authority_consensus": self._get_authority_consensus(),
            "timestamp": datetime.now().isoformat(),
        }

    def get_crisis_center_section(self) -> str:
        """Generate a compact BLACKSWAN CRISIS CENTER section for matrix monitors."""
        data = self.get_crisis_data()
        p = data["pressure"]
        war_day = data["war_day"]

        # Pressure bar
        bar_len = 30
        filled = int(p * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Pressure level icon
        if p >= 0.7:
            lvl_icon = "🔴 CRITICAL"
        elif p >= 0.5:
            lvl_icon = "🟠 ELEVATED"
        elif p >= 0.3:
            lvl_icon = "🟡 MODERATE"
        else:
            lvl_icon = "🟢 LOW"

        # Trend arrow
        d = data["trend_direction"]
        arrow_map = {
            "ACCELERATING": "⬆⬆ ACCELERATING",
            "DRIFTING_FOR": "⬆ Drifting toward thesis",
            "STEADY": "➡ Steady",
            "DRIFTING_AGAINST": "⬇ Drifting away",
            "RETREATING": "⬇⬇ RETREATING",
            "FIRST_READING": "● First reading",
        }
        trend_str = arrow_map.get(d, d)

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  🦢 BLACKSWAN CRISIS CENTER — Day {} of Iran War".format(war_day))
        lines.append("=" * 60)
        lines.append(f"  Pressure: [{bar}] {data['pressure_pct']}  {lvl_icon}")
        lines.append(f"  Thesis:   {data['bullish']} FOR / {data['bearish']} AGAINST / "
                     f"{data['neutral']} UNDECIDED  →  {data['ratio_pct']} confirming")
        lines.append(f"  Trend:    {trend_str}  (delta {data['ratio_delta']:+.1%})")
        lines.append(f"  Prob:     {data['probability']}")
        lines.append(f"  ──────────────────────────────────────────────────────────")
        lines.append(f"  HOTTEST INDICATORS:")
        for i, ind in enumerate(data["top5_indicators"], 1):
            score_pct = f"+{ind['score']:.0%}"
            lines.append(f"    {i}. {score_pct:>5}  {ind['name']}")

        if data["contrarian_indicators"]:
            lines.append(f"  CONTRARIAN (against thesis):")
            for ind in data["contrarian_indicators"]:
                lines.append(f"    ⚠ {ind['score']:+.0%}  {ind['name']}")
        else:
            lines.append(f"  ✓ ZERO indicators pointing against thesis")

        # ── Authority Consensus (if available) ──
        auth = data.get("authority_consensus")
        if auth and auth.get("total_signals", 0) > 0:
            lines.append(f"  ──────────────────────────────────────────────────────────")
            lines.append(f"  AUTHORITY CONSENSUS: {auth['consensus']} "
                         f"(score {auth['consensus_score']:+.2f}, "
                         f"{auth['agreement_pct']:.0f}% agreement)")
            lines.append(f"  Experts: {auth['total_authorities']}/4 active, "
                         f"{auth['total_signals']} signals tracked")

        lines.append("=" * 60)
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _mini_bar(score: float) -> str:
        """Create a mini visual bar for a -1.0 to 1.0 score."""
        # Clamp
        score = max(-1.0, min(1.0, score))
        # 10-char bar centered at 5
        width = 10
        center = width // 2
        pos = int((score + 1) / 2 * width)
        pos = max(0, min(width - 1, pos))

        bar_chars = list("░" * width)
        bar_chars[pos] = "█"

        if score > 0.4:
            label = f"+{score:.0%}"
            color = "▲"
        elif score < -0.2:
            label = f"{score:.0%}"
            color = "▼"
        else:
            label = f"{score:+.0%}"
            color = "─"

        return f"[{''.join(bar_chars)}] {label:>5} {color}"


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API — importable by AAC Matrix Monitor and NCC Matrix Monitor
# ═══════════════════════════════════════════════════════════════════════════

def get_crisis_center() -> PressureCooker:
    """Return a ready-to-use PressureCooker instance with all indicators."""
    return PressureCooker(INDICATORS)


def get_crisis_data() -> Dict:
    """Quick data fetch for external monitors (AAC / NCC)."""
    return get_crisis_center().get_crisis_data()


def get_crisis_section() -> str:
    """Quick section render for embedding in text dashboards."""
    return get_crisis_center().get_crisis_center_section()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Black Swan Pressure Cooker")
    parser.add_argument('--score-only', action='store_true', help='Quick pressure score')
    parser.add_argument('--update', nargs=3, metavar=('ID', 'SCORE', 'OBS'),
                        help='Update indicator: ID SCORE "observation"')
    parser.add_argument('--list-ids', action='store_true', help='List all indicator IDs')
    parser.add_argument('--movers', type=int, default=0, metavar='HOURS',
                        help='Show biggest movers in last N hours')
    parser.add_argument('--log-ratio', action='store_true',
                        help='Log current ratio to daily trend file')
    parser.add_argument('--trend', action='store_true',
                        help='Show ratio trend only')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cooker = PressureCooker(INDICATORS)

    if args.list_ids:
        for ind in INDICATORS:
            print(f"  {ind.id:<30} [{ind.tier.value[:4]}] {ind.name}")
        return

    if args.update:
        ind_id, score_str, obs = args.update
        old, new = cooker.update_indicator(ind_id, float(score_str), obs)
        print(f"  Updated {ind_id}: {old:+.2f} -> {new:+.2f}")
        print(f"  Observation: {obs}")
        return

    if args.movers:
        movers = cooker.get_biggest_movers(args.movers)
        if not movers:
            print("  No significant movers in that period.")
        else:
            for m in movers:
                print(f"  {m['id']}: {m['delta']:+.2f} -> {m['new_score']:.2f} | {m['observation'][:60]}")
        return

    if args.log_ratio:
        entry = cooker.log_daily_ratio()
        print(f"  Logged: Day {entry['war_day']} | ratio={entry['ratio']:.0%} | "
              f"pressure={entry['pressure']:.0%} | {entry['bullish']}B/{entry['bearish']}A/{entry['neutral']}N")
        return

    if args.trend:
        print(cooker.generate_ratio_section())
        return

    if args.score_only:
        pressure = cooker.compute_pressure()
        print(f"\n  PRESSURE: {pressure['overall_pressure']:.0%}")
        print(f"  RATIO:    {pressure['thesis_ratio']:.0%} confirming")
        print(f"  {pressure['prob_description']}")
        print(f"  {pressure['bullish_count']} bullish / {pressure['neutral_count']} neutral / {pressure['bearish_count']} bearish\n")
        return

    # Full dashboard — also auto-log the ratio
    cooker.log_daily_ratio()
    print(cooker.generate_dashboard())


if __name__ == "__main__":
    main()

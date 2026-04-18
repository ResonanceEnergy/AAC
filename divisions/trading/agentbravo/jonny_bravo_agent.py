"""
Jonny Bravo Division — Trading Education & Methodology Agent
═══════════════════════════════════════════════════════════════

The Jonny Bravo Division handles trading education, methodology delivery,
and knowledge transfer within the AAC BARREN WUFFET ecosystem.

Responsibilities:
    - Trading course curriculum management
    - Chart pattern recognition & teaching
    - Supply/demand zone methodology
    - Risk management education
    - Trade journal analysis & coaching
    - Strategy backtesting education

Research Status:
    - Dan Winter golden ratio integration: PARTIAL (needs implementation)
    - Jonny Bravo course curriculum: PARTIAL (needs specifics)
    - See RESEARCH.md Section 9 for details
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────


class CourseLevel(Enum):
    """Trading education levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    MASTER = "master"


class Methodology(Enum):
    """Trading methodologies taught."""
    SUPPLY_DEMAND = "supply_demand"
    PRICE_ACTION = "price_action"
    ORDER_FLOW = "order_flow"
    FIBONACCI = "fibonacci"
    GOLDEN_RATIO = "golden_ratio"
    MARKET_STRUCTURE = "market_structure"
    VOLUME_PROFILE = "volume_profile"


# ── Data Models ────────────────────────────────────────────────────────────


@dataclass
class TradingLesson:
    """A single trading lesson or module."""
    id: str
    title: str
    methodology: Methodology
    level: CourseLevel
    content: str
    key_concepts: List[str] = field(default_factory=list)
    chart_patterns: List[str] = field(default_factory=list)
    risk_rules: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TradeJournalEntry:
    """Paper trade journal entry for education tracking."""
    trade_id: str
    pair: str
    direction: str  # "long" | "short"
    entry_reason: str
    methodology_used: Methodology
    risk_reward_ratio: float
    outcome: Optional[str] = None  # "win" | "loss" | "breakeven" | None
    lessons_learned: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ── Core Agent ─────────────────────────────────────────────────────────────


class JonnyBravoAgent:
    """
    Jonny Bravo Division Agent — Trading education & methodology.

    Manages course delivery, trade journaling, pattern libraries,
    and methodology scoring.
    """

    def __init__(self):
        self.lessons: Dict[str, TradingLesson] = {}
        self.journal: List[TradeJournalEntry] = []
        self.student_level: CourseLevel = CourseLevel.BEGINNER
        self._initialized = False
        logger.info("JonnyBravoAgent initialized — Trading education ready")

    async def initialize(self) -> bool:
        """Initialize lesson library and methodology database."""
        self._load_default_lessons()
        self._initialized = True
        logger.info(f"Loaded {len(self.lessons)} lessons")
        return True

    def _load_default_lessons(self):
        """Load default trading lesson catalog."""
        defaults = [
            TradingLesson(
                id="jb-001",
                title="Supply & Demand Zones — Foundation",
                methodology=Methodology.SUPPLY_DEMAND,
                level=CourseLevel.BEGINNER,
                content="Identify institutional supply/demand zones on higher timeframes.",
                key_concepts=["Zone identification", "Fresh vs tested zones",
                              "Institutional footprint", "Drop-base-rally"],
                chart_patterns=["DBR", "RBD", "DBD", "RBR"],
                risk_rules=["Never risk more than 1% per trade",
                             "Minimum 1:3 risk-reward ratio"],
            ),
            TradingLesson(
                id="jb-002",
                title="Price Action — Candlestick Mastery",
                methodology=Methodology.PRICE_ACTION,
                level=CourseLevel.BEGINNER,
                content="Read raw price action without indicators.",
                key_concepts=["Pin bars", "Engulfing patterns", "Inside bars",
                              "Momentum candles", "Rejection wicks"],
                chart_patterns=["Bullish engulfing", "Bearish pin bar",
                                "Morning star", "Evening star"],
                risk_rules=["Confirm with volume", "Wait for close"],
            ),
            TradingLesson(
                id="jb-003",
                title="Market Structure — Break of Structure",
                methodology=Methodology.MARKET_STRUCTURE,
                level=CourseLevel.INTERMEDIATE,
                content="Identify trend shifts via market structure breaks.",
                key_concepts=["Higher highs", "Higher lows", "BOS",
                              "CHoCH", "Liquidity sweeps"],
                chart_patterns=["BOS", "CHoCH", "Liquidity grab"],
                risk_rules=["Only trade with the trend after BOS confirmation"],
            ),
            TradingLesson(
                id="jb-004",
                title="Golden Ratio & Fibonacci Confluence",
                methodology=Methodology.GOLDEN_RATIO,
                level=CourseLevel.ADVANCED,
                content="Apply Dan Winter's golden ratio harmonics to financial markets.",
                key_concepts=["Phi (1.618)", "0.618 retracement", "1.618 extension",
                              "Nested Fibonacci clusters", "Harmonic convergence zones",
                              "Fractal self-similarity"],
                chart_patterns=["Gartley", "Butterfly", "Bat", "Crab",
                                "ABCD", "Cypher"],
                risk_rules=["Require 2+ Fibonacci level confluence",
                             "Confirm with volume at harmonic zones"],
            ),
            TradingLesson(
                id="jb-005",
                title="Order Flow & Volume Profile",
                methodology=Methodology.ORDER_FLOW,
                level=CourseLevel.MASTER,
                content="Read institutional order flow and volume profile for precision entries.",
                key_concepts=["POC (Point of Control)", "Value Area High/Low",
                              "Delta divergence", "Aggressive iceberg detection",
                              "Footprint charts"],
                chart_patterns=["Volume imbalance", "Delta flip",
                                "Absorption pattern", "Initiative activity"],
                risk_rules=["Trade with delta momentum",
                             "Avoid low-volume nodes for entries"],
            ),
        ]
        for lesson in defaults:
            self.lessons[lesson.id] = lesson

    # ── Public API ─────────────────────────────────────────────────────

    def get_lesson(self, lesson_id: str) -> Optional[TradingLesson]:
        """Retrieve a specific lesson by ID."""
        return self.lessons.get(lesson_id)

    def get_lessons_by_level(self, level: CourseLevel) -> List[TradingLesson]:
        """Get all lessons for a specific experience level."""
        return [lesson for lesson in self.lessons.values() if lesson.level == level]

    def get_lessons_by_methodology(self, method: Methodology) -> List[TradingLesson]:
        """Get all lessons using a specific methodology."""
        return [lesson for lesson in self.lessons.values() if lesson.methodology == method]

    def log_trade(self, entry: TradeJournalEntry) -> None:
        """Add a trade journal entry for education tracking."""
        self.journal.append(entry)
        logger.info(f"Trade journal: {entry.pair} {entry.direction} — {entry.methodology_used.value}")

    def get_journal_stats(self) -> Dict[str, Any]:
        """Get trading journal statistics."""
        if not self.journal:
            return {"total_trades": 0, "win_rate": 0.0, "avg_rr": 0.0}

        total = len(self.journal)
        wins = sum(1 for j in self.journal if j.outcome == "win")
        avg_rr = sum(j.risk_reward_ratio for j in self.journal) / total

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total * 100) if total > 0 else 0.0,
            "avg_risk_reward": round(avg_rr, 2),
            "methodologies_used": list({j.methodology_used.value for j in self.journal}),
        }

    def get_curriculum_overview(self) -> Dict[str, Any]:
        """Get full curriculum overview."""
        return {
            "total_lessons": len(self.lessons),
            "levels": {
                level.value: len(self.get_lessons_by_level(level))
                for level in CourseLevel
            },
            "methodologies": {
                method.value: len(self.get_lessons_by_methodology(method))
                for method in Methodology
            },
            "student_level": self.student_level.value,
        }

    def get_status(self) -> Dict[str, Any]:
        """Agent status for dashboard/doctrine queries."""
        return {
            "agent": "JonnyBravoAgent",
            "department": "Jonny Bravo Division",
            "initialized": self._initialized,
            "lessons_loaded": len(self.lessons),
            "journal_entries": len(self.journal),
            "student_level": self.student_level.value,
            "methodologies": [m.value for m in Methodology],
        }

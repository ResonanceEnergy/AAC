"""
AAC Bake-Off Module
===================
Strategy validation, gate progression, and safety state management.
"""

from .engine import (
    BakeoffEngine,
    SafetyState,
    Gate,
    Decision,
    MetricValue,
    ChecklistItem,
    GateValidation,
    CompositeScore,
    StrategyState,
)

__all__ = [
    "BakeoffEngine",
    "SafetyState",
    "Gate",
    "Decision",
    "MetricValue",
    "ChecklistItem",
    "GateValidation",
    "CompositeScore",
    "StrategyState",
]

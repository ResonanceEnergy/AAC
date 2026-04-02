"""
AAC Bake-Off Module
===================
Strategy validation, gate progression, and safety state management.
"""

from .engine import (
    BakeoffEngine,
    ChecklistItem,
    CompositeScore,
    Decision,
    Gate,
    GateValidation,
    MetricValue,
    SafetyState,
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

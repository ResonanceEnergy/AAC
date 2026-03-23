"""
Options Intelligence Engine — AAC v3.2
========================================
AI-powered options trading system that weaponizes Unusual Whales flow data
into actionable entry signals, dynamically expands the trading universe,
scores trades with AI, optimizes strike selection via IV skew analysis,
and learns from past fills.

Modules:
    flow_signals    — UW flow → conviction multipliers & entry triggers
    universe        — Dynamic universe expansion beyond 16 fixed symbols
    ai_scorer       — AI/LLM-powered trade thesis evaluation
    skew_optimizer  — IV skew analysis for optimal strike selection
    feedback        — Fill logging & parameter tuning feedback loop
"""

from strategies.options_intelligence.flow_signals import (
    FlowSignalEngine,
    FlowConviction,
    FlowEntry,
)
from strategies.options_intelligence.universe import (
    UniverseExpander,
    DynamicCandidate,
)
from strategies.options_intelligence.ai_scorer import (
    AITradeScorer,
    TradeScore,
)
from strategies.options_intelligence.skew_optimizer import (
    SkewOptimizer,
    SkewAnalysis,
    OptimalStrike,
)
from strategies.options_intelligence.feedback import (
    FeedbackLoop,
    FillRecord,
)
from strategies.options_intelligence.pipeline import (
    OptionsIntelligencePipeline,
    PipelineResult,
)

__all__ = [
    "FlowSignalEngine",
    "FlowConviction",
    "FlowEntry",
    "UniverseExpander",
    "DynamicCandidate",
    "AITradeScorer",
    "TradeScore",
    "SkewOptimizer",
    "SkewAnalysis",
    "OptimalStrike",
    "FeedbackLoop",
    "FillRecord",
    "OptionsIntelligencePipeline",
    "PipelineResult",
]

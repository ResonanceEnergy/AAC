"""
FFD — Future Financial Doctrine
================================

The monetary transition doctrine for AAC.
Monitors and positions across three tracks:

  Track 1: Decentralized protocols (BTC, XRP, FLR, ETH, SOL)
  Track 2: Private digital money (stablecoins, tokenized assets)
  Track 3: Sovereign digital money (CBDCs, programmable fiat)

Based on Lyn Alden's "Broken Money" thesis and real-time
regulatory/market intelligence.

Usage:
    from aac.doctrine.ffd import get_ffd_engine, FFDEngine
    engine = get_ffd_engine()
    report = engine.get_status_report()
"""

from .ffd_engine import (
    FFDEngine,
    FFDMetrics,
    FFDTrack,
    EvidenceLevel,
    TransitionPhase,
    StablecoinHealth,
    StablecoinMonitor,
    StablecoinPegStatus,
    CBDCSignal,
    RegulatoryEvent,
    FFD_DOCTRINE_PACK,
    MONITORED_STABLECOINS,
    get_ffd_engine,
)

__all__ = [
    "FFDEngine",
    "FFDMetrics",
    "FFDTrack",
    "EvidenceLevel",
    "TransitionPhase",
    "StablecoinHealth",
    "StablecoinMonitor",
    "StablecoinPegStatus",
    "CBDCSignal",
    "RegulatoryEvent",
    "FFD_DOCTRINE_PACK",
    "MONITORED_STABLECOINS",
    "get_ffd_engine",
]

"""
CryptoIntelligence Department — BARREN WUFFET
==============================================

Core crypto analysis, venue health monitoring, and scam detection.
"""
from __future__ import annotations

from .crypto_intelligence_engine import CryptoIntelligenceEngine

__all__ = [
    "CryptoIntelligenceEngine",
]


def get_scam_detector():
    """Lazy import for scam detection module."""
    from .scam_detection import ScamDetector
    return ScamDetector()

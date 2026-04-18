"""
aac_puprime_vce — Volatility Compression → Expansion (VCE) Strategy Module
===========================================================================
Drop-in module for AAC: backtests, generates signals, and journals trades
for XAUUSD, EURUSD, and BTCUSD using PU Prime / MT5 CSV data.

Hypothetical strategy for education and testing — not financial advice.
"""

__version__ = "0.1.0"


def module_manifest() -> dict:
    """Return module metadata so AAC can discover this module."""
    return {
        "name": "aac_puprime_vce",
        "version": __version__,
        "strategy": "VCE",
        "instruments": ["XAUUSD", "EURUSD", "BTCUSD"],
        "entrypoints": {
            "ingest": "modules.aac_puprime_vce.cli:ingest",
            "backtest": "modules.aac_puprime_vce.cli:backtest",
            "signals": "modules.aac_puprime_vce.cli:signals",
            "report": "modules.aac_puprime_vce.cli:report",
        },
    }

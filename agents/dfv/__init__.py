from __future__ import annotations

"""DFV — Roaring Kitty agent.

Persona, doctrine, decision engine, daily routines, and 24/7 daemon for the
Keith Gill (DeepFuckingValue / Roaring Kitty) operator on AAC.

Public surface:
    from agents.dfv import DFV, decide, brief, daemon
"""

from agents.dfv.decision_engine import DFV, decide
from agents.dfv.routines import brief, midday, eod, weekend_dd, retail_pulse
from agents.dfv import daemon  # noqa: F401  re-export module

__all__ = ["DFV", "decide", "brief", "midday", "eod", "weekend_dd", "retail_pulse", "daemon"]

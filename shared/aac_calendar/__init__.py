"""AAC Financial Calendar Aggregator.

Fuses three sources into one timeline:
  1. Hardcoded high-conviction events (FOMC, OPEX, refunding) from
     `divisions/trading/warroom/thirteen_moon.py::_build_financial_events`
  2. Live earnings via Finnhub (`integrations/finnhub_client.FinnhubClient`)
  3. FRED release-date macros (CPI, NFP, PCE) via `integrations/fred_client`

Public API:
    from shared.aac_calendar import upcoming, by_symbol, by_kind

    upcoming(days=14)              # everything in the next 14 days
    upcoming(days=7, watchlist_only=True)  # only events touching watchlist symbols
    by_symbol("NVDA", days=30)
    by_kind("fed", days=60)

CLI:
    python -m shared.aac_calendar next --days 14
    python -m shared.aac_calendar next --days 7 --watchlist
    python -m shared.aac_calendar symbol NVDA --days 30
    python -m shared.aac_calendar kind fed --days 90
"""

from __future__ import annotations

from .aggregator import (
    CalendarEvent,
    by_kind,
    by_symbol,
    upcoming,
)

__all__ = ["CalendarEvent", "upcoming", "by_symbol", "by_kind"]

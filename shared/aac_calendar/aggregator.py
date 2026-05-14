"""Unified calendar event model + aggregator across live + hardcoded sources."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import structlog

# Load .env so live API keys are available when run as subprocess / CLI.
# Explicit path: walk up from this file to find the repo root .env.
try:
    from pathlib import Path as _Path  # noqa: PLC0415

    from dotenv import load_dotenv  # noqa: PLC0415
    _env_path = _Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

_log = structlog.get_logger().bind(component="aac_calendar.aggregator")

# Event kinds (canonical taxonomy)
KIND_EARNINGS = "earnings"
KIND_FED = "fed"          # FOMC meetings, minutes, speeches
KIND_ECONOMIC = "economic"  # CPI, NFP, PCE, GDP, etc.
KIND_OPTIONS = "options"   # OPEX, quad witching
KIND_POLICY = "policy"     # Treasury refunding, tax day, tariff deadlines
KIND_IPO = "ipo"
KIND_OTHER = "other"

# Importance levels
IMP_CRITICAL = "CRITICAL"
IMP_HIGH = "HIGH"
IMP_MEDIUM = "MEDIUM"
IMP_LOW = "LOW"

VALID_KINDS = {
    KIND_EARNINGS, KIND_FED, KIND_ECONOMIC, KIND_OPTIONS,
    KIND_POLICY, KIND_IPO, KIND_OTHER,
}


@dataclass
class CalendarEvent:
    """Unified calendar event."""
    date: date
    title: str
    kind: str
    symbols: list[str] = field(default_factory=list)
    importance: str = IMP_MEDIUM
    notes: str = ""
    source: str = ""  # e.g. "thirteen_moon", "finnhub", "fred"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.date, str):
            self.date = datetime.strptime(self.date, "%Y-%m-%d").date()
        self.symbols = [s.upper() for s in self.symbols]
        if self.kind not in VALID_KINDS:
            self.kind = KIND_OTHER

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d

    @property
    def days_away(self) -> int:
        return (self.date - date.today()).days


# ── Source 1: hardcoded thirteen_moon events ─────────────────────────────────


def _load_hardcoded_events() -> list[CalendarEvent]:
    """Pull events from thirteen_moon._build_financial_events()."""
    try:
        from divisions.trading.warroom.thirteen_moon import (  # noqa: PLC0415
            _build_financial_events,
        )
    except ImportError:
        _log.warning("thirteen_moon_not_importable")
        return []

    raw = _build_financial_events()
    out: list[CalendarEvent] = []
    for ev in raw:
        out.append(CalendarEvent(
            date=ev.date,
            title=ev.name,
            kind=ev.category if ev.category in VALID_KINDS else KIND_OTHER,
            symbols=list(ev.companies),
            importance=ev.impact,
            notes=ev.description,
            source="thirteen_moon",
        ))
    return out


# ── Source 2: live Finnhub earnings ──────────────────────────────────────────


async def _load_finnhub_earnings(
    days: int,
    symbols: list[str] | None = None,
) -> list[CalendarEvent]:
    """Pull earnings calendar from Finnhub for the given window."""
    if not os.getenv("FINNHUB_API_KEY"):
        _log.info("finnhub_key_missing_skipping")
        return []

    try:
        from integrations.finnhub_client import FinnhubClient  # noqa: PLC0415
    except ImportError as e:
        _log.warning("finnhub_client_unavailable", error=str(e))
        return []

    today = date.today()
    end = today + timedelta(days=days)

    out: list[CalendarEvent] = []
    try:
        async with FinnhubClient() as client:
            if symbols:
                # Per-symbol query (avoids fetching the entire universe)
                for sym in symbols:
                    entries = await client.get_earnings_calendar(
                        from_date=today.isoformat(),
                        to_date=end.isoformat(),
                        symbol=sym,
                    )
                    for e in entries:
                        out.extend(_finnhub_to_events([e]))
            else:
                entries = await client.get_earnings_calendar(
                    from_date=today.isoformat(),
                    to_date=end.isoformat(),
                )
                out.extend(_finnhub_to_events(entries))
    except Exception as e:
        _log.warning("finnhub_fetch_failed", error=str(e))
    return out


def _finnhub_to_events(entries: Iterable[Any]) -> list[CalendarEvent]:
    out: list[CalendarEvent] = []
    for e in entries:
        try:
            d = datetime.strptime(e.date, "%Y-%m-%d").date()
        except (ValueError, AttributeError):
            continue
        importance = IMP_HIGH if e.symbol in _MEGA_CAPS else IMP_MEDIUM
        notes_parts = []
        if e.eps_estimate:
            notes_parts.append(f"EPS est ${e.eps_estimate:.2f}")
        if e.revenue_estimate:
            notes_parts.append(f"Rev est ${e.revenue_estimate / 1e9:.2f}B")
        out.append(CalendarEvent(
            date=d,
            title=f"{e.symbol} Q{e.quarter} {e.year} Earnings",
            kind=KIND_EARNINGS,
            symbols=[e.symbol],
            importance=importance,
            notes=" | ".join(notes_parts),
            source="finnhub",
            extra={
                "eps_estimate": e.eps_estimate,
                "revenue_estimate": e.revenue_estimate,
                "quarter": e.quarter,
                "year": e.year,
            },
        ))
    return out


_MEGA_CAPS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN",
    "TSLA", "BRK.B", "JPM", "V", "MA", "UNH", "WMT", "XOM",
}


# ── Source 3: FRED release schedule (CPI, NFP, PCE) ──────────────────────────
# FRED's /series/release endpoint returns release IDs. We use the well-known
# release IDs (10=CPI, 50=Employment Situation, 21=PCE) and call
# /releases/dates to get upcoming release datetimes.


_FRED_KEY_RELEASES = {
    10: ("CPI", IMP_HIGH),                 # Consumer Price Index
    50: ("Non-Farm Payrolls", IMP_HIGH),   # Employment Situation
    21: ("PCE Inflation", IMP_HIGH),       # Personal Income and Outlays
    53: ("GDP", IMP_HIGH),                 # Gross Domestic Product
    82: ("FOMC Meeting", IMP_CRITICAL),    # FOMC
}


async def _load_fred_releases(days: int) -> list[CalendarEvent]:
    """Pull upcoming FRED release dates for key indicators."""
    if not os.getenv("FRED_API_KEY"):
        _log.info("fred_key_missing_skipping")
        return []

    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        return []

    api_key = os.getenv("FRED_API_KEY")
    today = date.today()
    end = today + timedelta(days=days)
    base = "https://api.stlouisfed.org/fred/release/dates"

    out: list[CalendarEvent] = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for release_id, (name, importance) in _FRED_KEY_RELEASES.items():
                params = {
                    "release_id": release_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "realtime_start": today.isoformat(),
                    "realtime_end": end.isoformat(),
                    "include_release_dates_with_no_data": "true",
                }
                try:
                    async with session.get(base, params=params) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                except Exception as e:
                    _log.warning("fred_release_fetch_failed", release=name, error=str(e))
                    continue

                for rd in data.get("release_dates", []):
                    try:
                        d = datetime.strptime(rd["date"], "%Y-%m-%d").date()
                    except (ValueError, KeyError):
                        continue
                    if not (today <= d <= end):
                        continue
                    kind = KIND_FED if release_id == 82 else KIND_ECONOMIC
                    out.append(CalendarEvent(
                        date=d,
                        title=f"{name} Release",
                        kind=kind,
                        symbols=["FED"] if release_id == 82 else [],
                        importance=importance,
                        notes=f"FRED release_id={release_id}",
                        source="fred",
                        extra={"release_id": release_id},
                    ))
    except Exception as e:
        _log.warning("fred_session_failed", error=str(e))
    return out


# ── Aggregator ───────────────────────────────────────────────────────────────


def _watchlist_symbols() -> set[str]:
    """All symbols across vol_premium + war_room regimes."""
    try:
        from shared.watchlist import (  # noqa: PLC0415
            get_vol_premium_tickers,
            get_war_room_rules,
        )
    except ImportError:
        return set()

    symbols: set[str] = set(get_vol_premium_tickers())
    for regime in ("CRISIS", "ELEVATED", "WATCH", "CALM"):
        try:
            for rule in get_war_room_rules(regime):
                if rule and len(rule) > 0:
                    symbols.add(str(rule[0]).upper())
        except Exception:
            continue
    return symbols


def _dedupe(events: list[CalendarEvent]) -> list[CalendarEvent]:
    """Merge events that share (date, kind, primary symbol)."""
    seen: dict[tuple[str, str, str], CalendarEvent] = {}
    for ev in events:
        primary = ev.symbols[0] if ev.symbols else ev.title[:40]
        key = (ev.date.isoformat(), ev.kind, primary.upper())
        if key in seen:
            existing = seen[key]
            # Prefer hardcoded over live (richer context); merge symbols
            if existing.source == "thirteen_moon":
                merged_syms = sorted(set(existing.symbols) | set(ev.symbols))
                existing.symbols = merged_syms
            else:
                ev.symbols = sorted(set(existing.symbols) | set(ev.symbols))
                seen[key] = ev
        else:
            seen[key] = ev
    return list(seen.values())


async def _aggregate(
    days: int,
    *,
    watchlist_only: bool,
    use_finnhub: bool,
    use_fred: bool,
) -> list[CalendarEvent]:
    today = date.today()
    end = today + timedelta(days=days)

    watchlist = _watchlist_symbols()
    target_symbols = sorted(watchlist) if watchlist_only and watchlist else None

    tasks: list[Any] = []
    if use_finnhub:
        tasks.append(_load_finnhub_earnings(days, target_symbols))
    if use_fred:
        tasks.append(_load_fred_releases(days))

    live_results: list[list[CalendarEvent]] = []
    if tasks:
        live_results = await asyncio.gather(*tasks, return_exceptions=False)

    all_events: list[CalendarEvent] = list(_load_hardcoded_events())
    for batch in live_results:
        all_events.extend(batch)

    # Window filter
    all_events = [e for e in all_events if today <= e.date <= end]

    # Watchlist filter
    if watchlist_only and watchlist:
        def touches(ev: CalendarEvent) -> bool:
            if not ev.symbols:
                # Macro events with no symbol affect everything
                return ev.kind in (KIND_FED, KIND_ECONOMIC, KIND_POLICY, KIND_OPTIONS)
            return any(s in watchlist for s in ev.symbols)
        all_events = [e for e in all_events if touches(e)]

    all_events = _dedupe(all_events)
    all_events.sort(key=lambda e: (e.date, _imp_rank(e.importance)))
    return all_events


def _imp_rank(imp: str) -> int:
    return {IMP_CRITICAL: 0, IMP_HIGH: 1, IMP_MEDIUM: 2, IMP_LOW: 3}.get(imp, 4)


def upcoming(
    days: int = 14,
    *,
    watchlist_only: bool = False,
    use_finnhub: bool = True,
    use_fred: bool = True,
) -> list[CalendarEvent]:
    """Return all calendar events in the next `days` days, sorted by date.

    Args:
        days: lookahead window in days
        watchlist_only: only return events touching watchlist symbols
                        (macro events with no symbol are always included)
        use_finnhub: pull live earnings from Finnhub
        use_fred: pull live release dates from FRED
    """
    return asyncio.run(_aggregate(
        days,
        watchlist_only=watchlist_only,
        use_finnhub=use_finnhub,
        use_fred=use_fred,
    ))


def by_symbol(symbol: str, days: int = 30) -> list[CalendarEvent]:
    """Return events touching a specific symbol in the next `days` days."""
    sym = symbol.upper()
    events = upcoming(days, watchlist_only=False)
    out = []
    for ev in events:
        if sym in ev.symbols:
            out.append(ev)
        elif not ev.symbols and ev.kind in (KIND_FED, KIND_ECONOMIC, KIND_POLICY):
            # Macro affects every symbol
            out.append(ev)
    return out


def by_kind(kind: str, days: int = 30) -> list[CalendarEvent]:
    """Return events of a specific kind in the next `days` days."""
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Valid: {sorted(VALID_KINDS)}")
    return [e for e in upcoming(days) if e.kind == kind]

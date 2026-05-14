"""CLI for the AAC financial calendar aggregator.

Usage:
    python -m shared.aac_calendar next [--days N] [--watchlist] [--no-live]
    python -m shared.aac_calendar symbol NVDA [--days 30]
    python -m shared.aac_calendar kind fed [--days 90]
"""

from __future__ import annotations

import argparse
import json
import sys

from .aggregator import VALID_KINDS, by_kind, by_symbol, upcoming


def _print_events(events: list, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps([e.to_dict() for e in events], indent=2))
        return
    if not events:
        print("(no events)")
        return
    print(f"{'DATE':<12} {'AWAY':>5}  {'IMP':<8} {'KIND':<10} {'SYMBOLS':<24} TITLE")
    print("-" * 100)
    for e in events:
        syms = ",".join(e.symbols[:4]) or "-"
        if len(e.symbols) > 4:
            syms += f"+{len(e.symbols) - 4}"
        away = f"+{e.days_away}d"
        print(f"{e.date.isoformat():<12} {away:>5}  {e.importance:<8} {e.kind:<10} {syms:<24} {e.title}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aac_calendar", description="AAC financial calendar")
    sub = p.add_subparsers(dest="cmd", required=True)

    pn = sub.add_parser("next", help="Show upcoming events")
    pn.add_argument("--days", type=int, default=14)
    pn.add_argument("--watchlist", action="store_true",
                    help="Only events touching watchlist symbols (macros always included)")
    pn.add_argument("--no-live", action="store_true",
                    help="Skip Finnhub + FRED live calls; use hardcoded only")
    pn.add_argument("--json", action="store_true")

    ps = sub.add_parser("symbol", help="Events touching a specific ticker")
    ps.add_argument("symbol")
    ps.add_argument("--days", type=int, default=30)
    ps.add_argument("--json", action="store_true")

    pk = sub.add_parser("kind", help="Events of a specific kind")
    pk.add_argument("kind", choices=sorted(VALID_KINDS))
    pk.add_argument("--days", type=int, default=30)
    pk.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "next":
        events = upcoming(
            days=args.days,
            watchlist_only=args.watchlist,
            use_finnhub=not args.no_live,
            use_fred=not args.no_live,
        )
        _print_events(events, as_json=args.json)
        return 0

    if args.cmd == "symbol":
        events = by_symbol(args.symbol, days=args.days)
        _print_events(events, as_json=args.json)
        return 0

    if args.cmd == "kind":
        events = by_kind(args.kind, days=args.days)
        _print_events(events, as_json=args.json)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

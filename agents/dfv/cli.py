from __future__ import annotations

"""DFV CLI — `python -m agents.dfv <cmd>`.

Commands:
    brief        Run pre-market brief now and print summary.
    midday       Run midday check now.
    eod          Run end-of-day debrief now.
    weekend      Run weekend DD slot now.
    review       Review a free-text prompt through the seven gates.
    decide       Read a JSON proposal from stdin and run it through gates.
    daemon       Run the 24/7 cadence loop.
    thesis       List, get, set thesis records.
    status       Show open theses count, watchlist size, last decisions.
"""

import argparse
import json
import sys
from typing import Any

from agents.dfv.decision_engine import DFV, decide, review_prompt
from agents.dfv.routines import (
    asia_digest,
    asia_watch,
    brief,
    close_debrief,
    eod,
    midday,
    open_bell_prep,
    weekend_dd,
)


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cmd_brief(_args: argparse.Namespace) -> int:
    out = brief()
    print(f"\n=== DFV Brief — {out['generated_at']} ===")
    print(out["headline"])
    print()
    print(f"Equity:    ${(out['portfolio_summary']['total_equity_usd'] or 0):,.0f}")
    print(f"Cash:      ${(out['portfolio_summary']['cash_usd'] or 0):,.0f}")
    print(f"BP:        ${(out['portfolio_summary']['buying_power_usd'] or 0):,.0f}")
    print(f"Positions: {out['portfolio_summary']['open_positions']}")
    print()
    print(f"Saved → {out.get('saved_to', '?')}")
    return 0


def _cmd_midday(_args: argparse.Namespace) -> int:
    _print_json(midday())
    return 0


def _cmd_eod(_args: argparse.Namespace) -> int:
    _print_json(eod())
    return 0


def _cmd_weekend(_args: argparse.Namespace) -> int:
    _print_json(weekend_dd())
    return 0


def _cmd_asia_digest(_args: argparse.Namespace) -> int:
    _print_json(asia_digest())
    return 0


def _cmd_open_bell_prep(_args: argparse.Namespace) -> int:
    _print_json(open_bell_prep())
    return 0


def _cmd_close_debrief(_args: argparse.Namespace) -> int:
    _print_json(close_debrief())
    return 0


def _cmd_asia_watch(_args: argparse.Namespace) -> int:
    _print_json(asia_watch())
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    text = args.text or sys.stdin.read()
    d = review_prompt(text)
    _print_json(d.to_dict())
    return 0 if d.verdict in ("approved", "approved_with_notes") else 2


def _cmd_decide(_args: argparse.Namespace) -> int:
    proposal = json.loads(sys.stdin.read())
    d = decide(proposal)
    _print_json(d.to_dict())
    return 0 if d.verdict in ("approved", "approved_with_notes") else 2


def _cmd_daemon(_args: argparse.Namespace) -> int:
    from agents.dfv.daemon import run_forever
    run_forever()
    return 0


def _cmd_thesis(args: argparse.Namespace) -> int:
    dfv = DFV()
    if args.action == "list":
        rows = [
            {"symbol": s, "conviction": r.get("conviction"),
             "updated": r.get("updated"), "horizon": r.get("horizon")}
            for s, r in sorted(dfv.thesis.all().items())
        ]
        _print_json(rows)
    elif args.action == "get":
        _print_json(dfv.thesis.get(args.symbol) or {})
    elif args.action == "set":
        rec = dfv.thesis.set(
            args.symbol,
            thesis=args.thesis or "",
            conviction=int(args.conviction or 3),
            horizon=args.horizon or "months",
            catalysts=(args.catalysts or "").split("|") if args.catalysts else [],
            invalidation=args.invalidation or "",
            target={"raw": args.target or ""},
            sizing={"max_pct_book": float(args.max_pct or 0.03)},
        )
        _print_json(rec)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from agents.dfv.llm import ask
    text = args.text or sys.stdin.read()
    if not text.strip():
        print("usage: dfv ask <question>", file=sys.stderr)
        return 2
    res = ask(text, verbose=bool(args.verbose))
    print(res.answer)
    if args.show_tools and res.tool_calls:
        print("\n--- tools called ---")
        for tc in res.tool_calls:
            print(f"  {tc.get('name')}({json.dumps(tc.get('args'), default=str)[:120]})")
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    from agents.dfv.rag import search
    hits = search(args.query, k=int(args.k), kind=args.kind, symbol=args.symbol)
    _print_json(hits)
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    from agents.dfv import rag as dfv_rag
    out = {}
    if args.what in ("all", "theses"):
        out["theses_indexed"] = dfv_rag.reindex_all_theses()
    if args.what in ("all", "briefs"):
        out["briefs_indexed"] = dfv_rag.reindex_all_briefs()
    _print_json(out)
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    from agents.dfv.daemon import heartbeat_status
    dfv = DFV()
    print("=== DFV Status ===")
    print(f"Theses:     {len(dfv.thesis.all())}")
    print(f"Conviction: {len(dfv.conviction.all())} tiers on file")
    print(f"Watchlist:  {len(dfv.watchlist.all())}")
    stale = dfv.thesis.needs_review(30)
    print(f"Stale (≥30d): {len(stale)}  {stale[:10]}")
    print(f"Recent decisions: {len(dfv.decisions.tail(100))}")
    print(f"Autonomy: trade_execution = "
          f"{dfv.doctrine.get('autonomy', {}).get('trade_execution')}")
    hb = heartbeat_status()
    if hb["alive"]:
        print(f"Daemon:     ALIVE (pid={hb.get('pid')}, age={hb.get('age_seconds')}s, last_routine={hb.get('last_routine')})")
    else:
        reason = hb.get("reason") or f"stale (age={hb.get('age_seconds')}s)"
        print(f"Daemon:     DOWN ({reason})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("dfv", description="Roaring Kitty operator CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("brief").set_defaults(func=_cmd_brief)
    sub.add_parser("midday").set_defaults(func=_cmd_midday)
    sub.add_parser("eod").set_defaults(func=_cmd_eod)
    sub.add_parser("weekend").set_defaults(func=_cmd_weekend)
    sub.add_parser("asia_digest", help="04:00 ET — Asia overnight tape").set_defaults(func=_cmd_asia_digest)
    sub.add_parser("open_bell_prep", help="09:25 ET — last 5 before the open").set_defaults(func=_cmd_open_bell_prep)
    sub.add_parser("close_debrief", help="17:00 ET — close prints + stale theses").set_defaults(func=_cmd_close_debrief)
    sub.add_parser("asia_watch", help="22:00 ET — overnight risk watch").set_defaults(func=_cmd_asia_watch)

    rv = sub.add_parser("review", help="review free-text prompt through seven gates")
    rv.add_argument("text", nargs="?", help="prompt text (or stdin)")
    rv.set_defaults(func=_cmd_review)

    sub.add_parser("decide", help="JSON proposal on stdin → seven-gate verdict") \
        .set_defaults(func=_cmd_decide)
    sub.add_parser("daemon", help="run 24/7 cadence loop").set_defaults(func=_cmd_daemon)
    sub.add_parser("status").set_defaults(func=_cmd_status)

    ak = sub.add_parser("ask", help="Ask DFV a question — uses local LLM + tools")
    ak.add_argument("text", nargs="?", help="prompt text (or stdin)")
    ak.add_argument("--verbose", "-v", action="store_true")
    ak.add_argument("--show-tools", action="store_true", dest="show_tools")
    ak.set_defaults(func=_cmd_ask)

    rc = sub.add_parser("recall", help="Semantic search DFV memory")
    rc.add_argument("query")
    rc.add_argument("-k", type=int, default=5)
    rc.add_argument("--kind", choices=["brief", "thesis", "decision", "note"])
    rc.add_argument("--symbol")
    rc.set_defaults(func=_cmd_recall)

    rx = sub.add_parser("reindex", help="Rebuild DFV RAG index from disk")
    rx.add_argument("what", choices=["all", "theses", "briefs"], nargs="?", default="all")
    rx.set_defaults(func=_cmd_reindex)

    th = sub.add_parser("thesis")
    th.add_argument("action", choices=["list", "get", "set"])
    th.add_argument("symbol", nargs="?")
    th.add_argument("--thesis")
    th.add_argument("--conviction", type=int)
    th.add_argument("--horizon")
    th.add_argument("--catalysts", help="pipe-separated")
    th.add_argument("--invalidation")
    th.add_argument("--target")
    th.add_argument("--max-pct", type=float, dest="max_pct")
    th.set_defaults(func=_cmd_thesis)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

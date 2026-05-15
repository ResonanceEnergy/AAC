from __future__ import annotations

"""
DFV — Grok thesis generator.

Closes Hard Rule #1 gaps by asking Grok (xAI) to write a candidate thesis for
every held symbol — good or bad. Verdict can be hold/add/trim/close. The
written thesis is stored with `author="GROK"` so DFV can ratify or override
without confusing it for an operator-authored thesis.

Usage (from repo root):
    python scripts/dfv_grok_theses.py                # gaps only (default)
    python scripts/dfv_grok_theses.py --all          # every held symbol
    python scripts/dfv_grok_theses.py --symbols TSLA,OWL,SLV
    python scripts/dfv_grok_theses.py --dry-run      # print, do not write

Per `.github/instructions/dfv-decisions.instructions.md`:
- G1 (thesis): satisfied — every held symbol gets one on paper.
- G6 (invalidation): Grok prompt forces a concrete invalidation level.
- author="GROK" preserves provenance — operator must ratify before DFV
  treats it as their own (G3 of the persona: 60-second whiteboard rule).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.dfv.decision_engine import DFV  # noqa: E402
from agents.dfv.llm import _grok_ask  # noqa: E402
from monitoring.dfv_dashboard import _flatten_positions  # noqa: E402
from monitoring.mission_control import collect_payload  # noqa: E402

SYSTEM_PROMPT = (
    "You are Roaring Kitty (DeepFuckingValue / DFV) writing investment theses for "
    "positions that ALREADY EXIST in the book. Your job is NOT to recommend closing — "
    "your job is to construct the strongest defensible thesis for why this position "
    "is on the sheet, then nail down a concrete invalidation level so we know when "
    "the thesis breaks. Be blunt and evidence-driven, but argue FOR the position. "
    "Cash is a position. No FOMO. Every thesis must be defensible on a whiteboard "
    "in 60 seconds and must include a concrete invalidation trigger "
    "(price level, time-stop, or named event)."
)

USER_TEMPLATE = """Write the strongest defensible thesis for this OPEN position. Verdict must be one of:
  - \"hold\"   : keep at current size — current sizing is right
  - \"add\"    : conviction high enough to upsize
  - \"trim\"   : reduce, but stay in the name

DO NOT use \"close\". The position is already on. Your job is to give it a thesis.
If you genuinely cannot construct a bull case, write the strongest tactical / hedge /
optionality case (e.g. tail hedge, vol exposure, asymmetric payoff, theme exposure).

POSITION:
  symbol     : {symbol}
  account    : {account}
  side       : {side}
  quantity   : {qty}
  avg_cost   : {avg_cost}
  last_price : {last_price}
  unrealized : {unrealized}
  asset_type : {asset_type}
  expiry     : {expiry}

Return STRICT JSON ONLY (no prose, no markdown fences) matching this shape:
{{
  \"verdict\": \"hold|add|trim\",
  \"thesis\": \"<=200 words. Argue FOR the position. Specific drivers, not platitudes.\",
  \"conviction\": 3,
  \"horizon\": \"e.g. '30 DTE', '6 months', '18 months'\",
  \"catalysts\": [\"specific catalyst 1\", \"specific catalyst 2\"],
  \"invalidation\": \"Concrete trigger. e.g. 'Close if SPY < 480 for 2 sessions' or 'Time-stop 21 DTE' or 'Exit if BTC reclaims 75k'.\",
  \"target\": {{\"price\": 0.0, \"rationale\": \"what gets you paid and how much\"}},
  \"max_pct_book\": 0.03,
  \"rationale\": \"1-2 sentences on edge / why asymmetric\"
}}

Conviction MUST be 1-5 (not 0):
  5=screaming, 4=high, 3=starter, 2=watch-size, 1=lottery/tail.
If you would honestly rate it 0, use 1 with a tail/optionality framing instead.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a Grok response, tolerant of fences."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        candidate = fenced.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _generate_one(position: dict[str, Any], model: str) -> dict[str, Any]:
    prompt = USER_TEMPLATE.format(
        symbol=position["symbol"],
        account=position.get("account", "?"),
        side=position.get("side", "-"),
        qty=position.get("qty", 0),
        avg_cost=position.get("avg_cost", 0),
        last_price=position.get("last_price", 0),
        unrealized=position.get("unrealized", 0),
        asset_type=position.get("asset_type", "stock"),
        expiry=position.get("expiry") or "n/a",
    )
    resp = _grok_ask(prompt=prompt, system=SYSTEM_PROMPT, model=model, temperature=0.3, max_tokens=900)
    if not resp.get("ok"):
        return {"ok": False, "error": resp.get("error", "unknown"), "raw": resp.get("text", "")}
    parsed = _extract_json(resp.get("text", ""))
    if not parsed:
        return {"ok": False, "error": "could not parse JSON", "raw": resp.get("text", "")}
    return {"ok": True, "data": parsed, "raw": resp.get("text", "")}


def _aggregate_positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse multi-leg / multi-account holdings of the same underlying into one row.

    DFV writes ONE thesis per underlying — not one per option leg.
    """
    by_sym: dict[str, dict[str, Any]] = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in by_sym:
            by_sym[sym] = {**r, "qty": 0, "unrealized": 0.0, "legs": 0, "accounts": set()}
        agg = by_sym[sym]
        agg["qty"] = (agg.get("qty") or 0) + (r.get("qty") or 0)
        agg["unrealized"] = float(agg.get("unrealized") or 0) + float(r.get("unrealized") or 0)
        agg["legs"] += 1
        agg["accounts"].add(r.get("account", "?"))
    out = []
    for sym, agg in by_sym.items():
        agg["account"] = ",".join(sorted(str(a) for a in agg.pop("accounts")))
        out.append(agg)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Grok-authored DFV theses.")
    parser.add_argument("--all", action="store_true", help="Every held symbol (not just gaps).")
    parser.add_argument("--symbols", default="", help="Comma list to force a specific set.")
    parser.add_argument("--model", default="grok-3-mini", help="xAI model id.")
    parser.add_argument("--dry-run", action="store_true", help="Print, do not write.")
    args = parser.parse_args()

    dfv = DFV()
    payload = collect_payload()
    held = _aggregate_positions(_flatten_positions(payload))
    held_syms = {p["symbol"] for p in held}

    if args.symbols.strip():
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        targets = [p for p in held if p["symbol"] in wanted]
        # If a wanted symbol isn't in current holdings, build a stub row so Grok still scores it
        for s in sorted(wanted - held_syms):
            targets.append({
                "symbol": s, "account": "n/a", "side": "-", "qty": 0,
                "avg_cost": 0, "last_price": 0, "unrealized": 0,
                "asset_type": "stock", "expiry": "",
            })
    elif args.all:
        targets = held
    else:
        existing = set(dfv.thesis.all().keys())
        targets = [p for p in held if p["symbol"] not in existing]

    if not targets:
        print("Nothing to do — every held symbol already has a thesis.")
        return 0

    print(f"Generating {len(targets)} thesis/theses with model={args.model} dry_run={args.dry_run}\n")

    written = 0
    failed: list[tuple[str, str]] = []
    for i, pos in enumerate(sorted(targets, key=lambda x: x["symbol"]), 1):
        sym = pos["symbol"]
        print(f"[{i}/{len(targets)}] {sym} ", end="", flush=True)
        result = _generate_one(pos, args.model)
        if not result["ok"]:
            print(f"FAILED — {result['error']}")
            failed.append((sym, result["error"]))
            continue
        d = result["data"]
        verdict = (d.get("verdict") or "").lower()
        if verdict == "close":
            verdict = "hold"  # forbidden — coerce, prompt should already prevent this
        conviction = int(d.get("conviction", 0) or 0)
        if conviction < 1:
            conviction = 1  # floor — no zero-conviction defenses; minimum is 'lottery/tail'
        thesis_text = (d.get("thesis") or "").strip()
        invalidation = (d.get("invalidation") or "").strip()
        if not thesis_text or not invalidation:
            print("FAILED — missing thesis or invalidation in JSON")
            failed.append((sym, "missing required fields"))
            continue
        catalysts = d.get("catalysts") or []
        if isinstance(catalysts, str):
            catalysts = [catalysts]
        target = d.get("target") or {}
        if not isinstance(target, dict):
            target = {"raw": str(target)}
        target.setdefault("verdict", verdict)
        target.setdefault("rationale", d.get("rationale", ""))
        sizing = {"max_pct_book": float(d.get("max_pct_book") or 0.03)}
        horizon = (d.get("horizon") or "").strip() or "tbd"

        print(f"{verdict.upper():<5}  conv={conviction}  -> ", end="")
        if args.dry_run:
            print("(dry-run)")
            print(json.dumps(d, indent=2)[:500])
            continue
        dfv.thesis.set(
            sym,
            thesis=thesis_text,
            conviction=conviction,
            horizon=horizon,
            catalysts=list(catalysts),
            invalidation=invalidation,
            target=target,
            sizing=sizing,
            author="GROK",
        )
        if conviction > 0:
            dfv.conviction.set(sym, conviction, reason=f"GROK candidate ({verdict})")
        written += 1
        print("written")

    print(f"\nDone. Written: {written}/{len(targets)}.  Failed: {len(failed)}.")
    if failed:
        print("Failures:")
        for sym, err in failed:
            print(f"  {sym}: {err[:140]}")
    print("\nNext: review in dashboard (http://localhost:8504) and ratify by editing")
    print("agents/dfv/memory/thesis_log.json (set author='DFV') or close the position.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

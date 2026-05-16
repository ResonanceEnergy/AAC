# SPEC — DFV Paper-Twin Shadow Book

> **Status:** spec only (Saturday). No code this weekend.
> **Author:** DFV / Roaring Kitty (Prime Operator)
> **Date:** 2026-05-17
> **Why this exists:** I want to measure whether my seven gates actually
> add edge, or whether I'd be richer if I just took every signal raw.
> The only honest way to know is to keep a parallel paper book that
> records *every* proposal — including the ones the gates killed —
> and compare PnL after 30 days.

---

## 1. Hard rules this spec inherits

1. **No FOMO.** Paper-twin never bypasses doctrine; it observes it.
2. **Cash is a position.** Paper book starts with the same notional cash as live.
3. **Every paper trade carries the same thesis ref** as the live decision it shadows. No anonymous fills.
4. **One whiteboard, 60 seconds.** Every paper row must be explainable in one line.
5. **Never confuses live and paper.** Separate files, separate UI tabs, separate alerts.

---

## 2. What we record

A new ledger:

```
agents/dfv/memory/paper_twin.jsonl
```

One JSON line per **decision event** (regardless of `verdict`):

```json
{
  "ts": "2026-05-17T14:32:11Z",
  "decision_id": "dfv_20260517_143211_GME",
  "symbol": "GME",
  "side": "buy",
  "qty": 1,
  "instrument": {"type": "option", "expiry": "2026-06-20", "strike": 30, "right": "C"},
  "price_assumed": 1.45,
  "thesis_ref": "GME@2026-05",
  "conviction_tier": 2,
  "live_verdict": "veto",
  "gates_failed": ["G5_correlation"],
  "gates_passed": ["G1","G2","G3","G4","G6","G7"],
  "shadow_action": "buy_anyway",
  "rationale": "Veto recorded. Paper-twin executes per Tier-2 sizing to measure G5 efficacy.",
  "live_action": "none",
  "live_order_id": null,
  "shadow_order_id": "pt_20260517_143211_GME"
}
```

Closeout rows mirror the same schema with `event_type="close"` and `realized_pnl_usd`.

## 3. Sizing & execution model (paper side)

- **Cash starts equal** to live `total_cash_usd` at the moment the paper book is created (`agents/dfv/memory/paper_twin_start.json` snapshots it once).
- **Sizing** is **always the gate-prescribed size** the doctrine would have approved if all gates passed — even when live vetoed. The point is to measure the *opportunity cost of the veto*, not to fantasise about leverage.
- **Fills:** assume mid-price at the timestamp of the decision (use the same `payload["quotes"]` snapshot the live decision saw — do not re-quote later, that's hindsight bias).
- **Slippage:** deterministic 0.5% on options, 2 bps on equities. No randomness in operational code (rule).
- **Commissions:** flat $0.65/contract options, $0.005/share equities.
- **Mark-to-market:** daily at EOD using the same payload the EOD routine consumes.

## 4. Lifecycle hooks (where the code will plug in, next week)

| Hook | File | What it does |
|---|---|---|
| `record_decision()` | new `agents/dfv/paper_twin.py` | Append a row for every `Decision` produced by `decision_engine.decide()` |
| `eod_mark_to_market()` | called from `routines.eod()` | Walk open paper positions, write today's MTM into `paper_twin_marks.jsonl` |
| `monthly_report()` | new CLI `python -m agents.dfv twin_report` | Compute gate-efficacy table after 30 days |
| `_render_paper_twin()` | `monitoring/dfv_dashboard.py` | New tab "Paper twin (shadow)" with the gate-efficacy table |

## 5. The single number we care about

After 30 calendar days:

```
gate_efficacy[G_i] = realized_pnl_paper(vetoed_only_by_G_i)
```

- **Negative or near-zero** → the gate is doing its job (it killed losing trades).
- **Large positive** → the gate is costing me money; revisit the threshold.

This is the only metric that justifies the seven-gate complexity budget. If after 90 days every gate is positive on net, **delete the gate** (rule #4: explain on a whiteboard).

## 6. UI surface (next week, not this weekend)

New dashboard tab below the watchlist:

```
🪞 Paper twin — shadow book
  Live PnL:  -$1,240   |  Paper PnL:  +$340
  Delta:     +$1,580   (gates are SAVING money)

  Gate efficacy (30d, vetoed-only trades):
    G1 thesis      → +$0     (n=0 trades vetoed by this alone)
    G2 size        → -$120   (gate saved money)
    G3 dry powder  → -$80    (saved)
    G4 catalyst    → +$210   (cost money — review threshold)
    G5 correlation → -$1,540 (saved BIG)
    G6 invalidation→ -$50    (saved)
    G7 liquidity   → -$0     (n=0)
```

## 7. What's explicitly out of scope

- ❌ Real broker calls — paper-twin is **memory-only**, never hits an exchange.
- ❌ Backfill of historic decisions — start from "day 1 of the twin", no fake history.
- ❌ Different position sizing than what doctrine prescribes — that's a different experiment.
- ❌ Auto-graduation of "winning paper strategies" into live — **always** requires human OK, autonomy stays `human_in_loop`.
- ❌ Showing paper PnL in any total/equity field that might confuse the dashboard.

## 8. Risks / things that will go wrong

1. **Survivorship bias on price snapshots.** Mitigation: lock the quote at decision time.
2. **Sample size will be tiny.** 30 days × maybe 5 decisions = 150 rows. Don't draw strong conclusions before n ≥ 200 per gate. Build the table; resist over-interpreting.
3. **Storage growth.** JSONL is fine for a year; switch to SQLite if rows > 50k.
4. **Schema drift.** Version every row: `"schema_version": 1`. Migrate carefully.

## 9. Acceptance criteria (when code lands next week)

- [ ] `tests/test_dfv_paper_twin.py` covers: record, MTM, monthly report, gate-efficacy math.
- [ ] No live broker import path is reachable from `paper_twin.py`.
- [ ] Dashboard tab is hidden behind a feature flag (`doctrine.paper_twin.enabled`).
- [ ] First 7 days of data are flagged "warmup — do not interpret".
- [ ] Commit message marker: `PAPER-TWIN-V1`.

---

**One whiteboard summary:**

> *"Every gate I trip recorded. Every trade I'd have made recorded. After 30 days I can prove — with my own money's would-be PnL — which gates earn their keep and which are theater. If a gate fails this test repeatedly, it dies."*

— DFV

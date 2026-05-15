# DFV memory directory

This directory holds Roaring Kitty's persistent state.  All files are JSON / JSONL,
read by `agents/dfv/memory_store.py`.

| File | Purpose |
|---|---|
| `thesis_log.json` | Per-ticker investment thesis (mandatory before any position) |
| `conviction.json` | Conviction tier 1-5 per ticker, with history |
| `watchlist.json` | Tickers under DD but not yet in book |
| `decisions.jsonl` | Append-only log of every decision DFV evaluated |
| `postmortems.jsonl` | Closed-position write-ups |
| `briefs/` | Saved daily brief reports (timestamped JSON) |

These files are versioned in git so the desk's institutional memory survives reboots.

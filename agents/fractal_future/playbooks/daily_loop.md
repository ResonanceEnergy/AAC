# Daily Loop Playbook — FRACTAL FUTURE Swarm

## Trigger
Every day, automated or manual kick.

## Steps

### 1. Entropy Sentinel Check (5 min)
- Read current entropy state (market + system + human)
- If E4+: skip rest of loop, issue alert, stand down
- If E3: run abbreviated loop (research scan only, no new experiments)
- If E1–E2: full loop

### 2. Research Cell Scan (15 min)
- Scan pre-configured sources (nature feeds, market data, AI papers)
- Log any new pattern observations as E0 notes
- Tag with `ff-pattern`, source, timestamp

### 3. Signal Brief (AAC Integration) (10 min)
- Pull current regime + entropy regime
- Top 5 active signals from signal cards
- Fill daily_signal_brief.md template

### 4. Telemetry Log
- Entropy sentinel writes hourly snapshot summary
- Any valve activations logged

### 5. End-of-Day
- Research cell submits observation count to synthesis backlog
- Sentinel confirms all cells nominal or logs exceptions

## Abort Conditions
- Entropy E4+: full stop
- Sentinel failure: all cells pause
- Human override: manual stand-down command

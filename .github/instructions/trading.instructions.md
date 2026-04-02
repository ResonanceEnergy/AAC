---
applyTo: "TradingExecution/**,strategies/**,trading/**"
---

# Trading Code — Safety Guardrails

This code executes REAL TRADES with REAL MONEY. Extra caution required.

## Absolute Rules

- NEVER modify order placement logic without explicit user confirmation
- NEVER change DRY_RUN, PAPER_TRADING, or LIVE_TRADING_ENABLED defaults
- ALWAYS preserve existing safety checks (position limits, order validation, kill switches)
- ALWAYS use deterministic slippage/scoring in non-simulation paths (no `random.random()`)
- Test with `DRY_RUN=true` before any change that touches order flow

## Active Exchanges

- IBKR: Port 7496 (live), account U24346218, IB API via `ib_insync`
- Moomoo: OpenD, FUTUCA market, `moomoo-api` package, trade PIN required
- NDAX: LIQUIDATED — connector exists but no active positions

## Key Files

- `TradingExecution/ibkr_connector.py` — IBKR order execution
- `TradingExecution/moomoo_connector.py` — Moomoo order execution  
- `TradingExecution/ndax_connector.py` — NDAX (inactive, uses ccxt)
- `strategies/war_room_engine.py` — War Room thesis evaluation
- `strategies/war_room_live_feeds.py` — 11 async live feed fetchers
- `core/command_center.py` — Orchestration layer

## Before Modifying

1. Read the existing implementation fully
2. Check if a safety guard already exists before adding one
3. Run the test suite after changes
4. If changing order flow: document the change in `.context/05_decisions/`

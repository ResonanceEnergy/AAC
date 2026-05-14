---
applyTo: "agents/**,BigBrainIntelligence/**"
---

# Agent Code — Swarm & Research

Agents are autonomous workers that consume data, reason, and emit reports/recommendations. They are NOT trade-execution code (that lives in `TradingExecution/`).

## Available Tools (USE THESE)

| Capability | Module | Notes |
|---|---|---|
| RAG over AAC repo | `shared/aac_rag/` | Local Ollama + LanceDB (Phase 2) |
| Live market data | `shared/data_sources.py` | yfinance, FRED, CoinGecko |
| Options flow | `integrations/unusual_whales_client.py` | UW field parsing currently broken — verify before relying |
| Quotes / news | `integrations/finnhub_client.py` | 60 req/min |
| Watchlist | `config/watchlist.yaml` | canonical ticker universe |

## Agent Conventions

1. Every agent is an `async def run()` — no blocking IO at top level
2. Agents emit structured output (dict / pydantic model), never print directly to user
3. Agents are stateless across runs — persist state to `data/` or DB, not in-memory
4. Recommendations are **advisory only** by default — never auto-execute trades unless explicit `live_execute=True` flag and downstream `TradingExecution/` guard
5. Use `structlog` with bound context: `_log = structlog.get_logger().bind(agent="my_agent")`
6. Errors propagate — agents don't swallow exceptions; the orchestrator decides retry/skip

## Forbidden in Agent Code

- Direct order placement (route through `TradingExecution/` connectors)
- Web scraping when an internal API client exists (check the API inventory in `.github/copilot-instructions.md`)
- Hardcoded tickers (read from `config/watchlist.yaml`)
- Synchronous `time.sleep()` (use `await asyncio.sleep()`)

## Before Adding a New Agent

1. Search existing agents — likely something similar exists
2. Decide: research agent (read-only, advisory), monitor agent (continuous loop), or workflow agent (one-shot)
3. Place in `agents/` (workflows) or `BigBrainIntelligence/` (research/analysis)
4. Add a test under `tests/agents/` mocking all external IO
5. Register with the scheduler if it should run on a cadence

# System Map

Primary areas:
- `core/` — command center, orchestration, state management
- `shared/` — cross-cutting infrastructure, monitoring, data sources, frameworks
- `TradingExecution/` — exchange connectors and execution plumbing (IBKR port 7497 live, Moomoo OpenD degraded)
- `strategies/` — strategy algorithms and execution engine
  - `strategies/war_room/` — MWP/ICM 5-stage pipeline (Scan → Evaluate → Plan → Execute → Report)
  - `strategies/war_room_engine.py` — 15-indicator composite scoring, Monte Carlo, 5 arms
  - `strategies/war_room_live_feeds.py` — 12 async API fetchers, live data pipeline
  - `strategies/thirteen_moon_doctrine.py` — 14-cycle temporal overlay (Mar 2026 → Apr 2027), 184 events, 6 layers
  - `strategies/thirteen_moon_storyboard.py` — interactive HTML timeline export
  - `strategies/paper_trading/` — engine, 5 strategies, optimizer, risk, regime detector
- `BigBrainIntelligence/` — research and analysis agents
- `CryptoIntelligence/` — venue health, crypto analytics, on-chain systems
- `CentralAccounting/` — accounting, P&L, database access
- `integrations/` — external service clients and bridges
- `monitoring/` — dashboards and continuous monitoring loops
- `agents/` — higher-level agent workflows and simulations
- `divisions/` — enterprise division wiring (11 divisions including paper trading)
- `data/storyboard/` — 13-Moon HTML storyboard + JSON export

Important architectural patterns:
- Async-first design with `asyncio` and `aiohttp`
- Base-class extension model in connectors, data sources, and agents
- A large portion of "gaps" in scans are correct abstract method placeholders
- Real defects tend to be in incomplete callback wiring, placeholder menu actions, and simulation shortcuts bleeding into operational code

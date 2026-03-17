# System Map

Primary areas:
- `core/` — command center, orchestration, state management
- `shared/` — cross-cutting infrastructure, monitoring, data sources, frameworks
- `TradingExecution/` — exchange connectors and execution plumbing
- `strategies/` — strategy algorithms and execution engine
- `BigBrainIntelligence/` — research and analysis agents
- `CryptoIntelligence/` — venue health, crypto analytics, on-chain systems
- `CentralAccounting/` — accounting, P&L, database access
- `integrations/` — external service clients and bridges
- `monitoring/` — dashboards and continuous monitoring loops
- `agents/` — higher-level agent workflows and simulations

Important architectural patterns:
- Async-first design with `asyncio` and `aiohttp`
- Base-class extension model in connectors, data sources, and agents
- A large portion of "gaps" in scans are correct abstract method placeholders
- Real defects tend to be in incomplete callback wiring, placeholder menu actions, and simulation shortcuts bleeding into operational code

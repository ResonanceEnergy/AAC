---
# AAC OpenClaw Agent Configuration
# Defines the multi-agent topology for the OpenClaw Gateway
# Reference: https://docs.openclaw.ai/concepts/agents
---

# AAC Agent Registry for OpenClaw

## Primary Agent — AZ SUPREME

The default agent for all inbound messages. Acts as the gateway to the
entire AAC ecosystem.

```yaml
name: AZ SUPREME
emoji: 👑
role: Supreme Executive Command
model: gpt-4o
provider: openai
session: main
always_active: true
```

## Secondary Agent — AX HELIX

Executive Operations and Integration. Handles tactical and operational
queries delegated by AZ SUPREME.

```yaml
name: AX HELIX
emoji: ⚙️
role: Executive Operations
model: gpt-4o
provider: openai
session: ax-helix
```

## Department Agents

### BigBrainIntelligence
```yaml
name: BigBrain Research
emoji: 🧠
role: Market Intelligence & Research
model: gpt-4o
provider: openai
session: bigbrain
capabilities:
  - Theater B (Attention & Narrative)
  - Theater C (Infrastructure & Latency)
  - Theater D (Information Asymmetry)
sub_agents: 20+
```

### TradingExecution
```yaml
name: Trade Executor
emoji: ⚡
role: Strategy Execution & Signal Routing
model: gpt-4o
provider: openai
session: trading
capabilities:
  - 50 arbitrage strategies
  - QuantumSignalAggregator
  - Paper trading engine
sub_agents: 49
```

### CryptoIntelligence
```yaml
name: Crypto Intel
emoji: 🔗
role: On-Chain Analysis & DeFi Intelligence
model: gpt-4o
provider: openai
session: crypto
capabilities:
  - Whale tracking
  - DeFi yield analysis
  - Cross-chain bridge monitoring
  - Mempool intelligence
sub_agents: 5+
```

### CentralAccounting
```yaml
name: Accountant
emoji: 📈
role: Financial Analysis & Portfolio Management
model: gpt-4o
provider: openai
session: accounting
capabilities:
  - P&L attribution
  - NAV tracking
  - Strategy performance
  - Risk metrics
sub_agents: 3+
```

### SharedInfrastructure
```yaml
name: Infra Monitor
emoji: 🏗️
role: System Health & Infrastructure
model: gpt-4o
provider: openai
session: infra
capabilities:
  - System health monitoring
  - Latency tracking
  - Uptime management
  - Self-healing operations
sub_agents: 5+
```

### NCC (Network Command Center)
```yaml
name: NCC Command
emoji: 📡
role: Network Operations & Command Center
model: gpt-4o
provider: openai
session: ncc
capabilities:
  - Cross-department communication
  - Alert routing
  - Operational dashboard
sub_agents: 3+
```

## Agent Routing Rules

All inbound messages are first received by **AZ SUPREME** on the `main` session.
AZ SUPREME classifies intent and routes to the appropriate department agent
via OpenClaw's `sessions_spawn` / `sessions_send` mechanism.

### Intent → Agent Mapping
| Intent | Agent | Session |
|---|---|---|
| Strategic Command | AZ SUPREME | main |
| Operational Query | AX HELIX | ax-helix |
| Market Data | BigBrain Research | bigbrain |
| Trading Signal | Trade Executor | trading |
| Risk Alert | AZ SUPREME (Risk Mode) | main |
| Portfolio Status | Accountant | accounting |
| Crypto Intel | Crypto Intel | crypto |
| Infrastructure | Infra Monitor | infra |
| Network Command | NCC Command | ncc |
| General Chat | AZ SUPREME | main |

## Multi-Agent Session Spawning

When AZ SUPREME needs a specialized agent, it spawns a new OpenClaw session:

```
AZ SUPREME (main session)
  ├── sessions_spawn → BigBrain Research (bigbrain session)
  ├── sessions_spawn → Trade Executor (trading session)
  ├── sessions_spawn → Crypto Intel (crypto session)
  └── sessions_send → AX HELIX (ax-helix session)
```

Each session has its own conversation context, memory, and tool access.
Results are routed back to the user via the originating channel.

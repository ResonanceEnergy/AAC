## AGENTS.md — BARREN WUFFET Multi-Agent Routing

Telegram Bot: @barrenwuffet069bot
Token: env(TELEGRAM_BOT_TOKEN)

### Routing Rules

**Primary Agent (default)**:
- All untagged messages → BARREN WUFFET (AZ SUPREME)
- AZ SUPREME handles executive decisions, strategic questions, and system status

**Department Routing**:
- @intel or /bw-intel → BigBrainIntelligence research agents
- @trade or /bw-signals → TradingExecution agents
- @crypto or /bw-crypto → CryptoIntelligence agents
- @accounting or /bw-accounting → CentralAccounting agents
- @risk or /bw-risk → Doctrine Engine & risk monitoring
- @jonny or /bw-jonny → Jonny Bravo Division agents
- @all → Broadcast to all departments

**Skill Routing** (35 skills auto-routed by command prefix):
- /bw-* → Matched to corresponding BARREN WUFFET skill
- /az* → AZ SUPREME executive commands
- /bw-crash → 2007 Crash Indicators (always-on monitoring)
- /bw-golden → Golden Ratio Finance (Dan Winter method)
- /bw-dd → SuperStonk DD research engine

### Agent Behavior Rules

Each agent:
1. Reads shared GOALS.md and doctrine state for context
2. Reads its own department memory and private notes
3. Processes the incoming message
4. Responds via Telegram with data-driven analysis
5. Updates shared state if response involves a decision or status change
6. Logs all interactions to CentralAccounting for audit trail

### Session Management

- **Main Session**: BARREN WUFFET primary loop (always active)
- **Research Sessions**: Spawned by BigBrain agents for deep analysis
- **Trading Sessions**: Per-strategy sessions for trade execution
- **Monitoring Sessions**: Always-on background sessions for:
  - Risk monitoring (BarrenWuffetState)
  - 2007 crash indicator scanning
  - Options flow tracking
  - Crypto whale alerts

### Priority Escalation

When an agent detects a critical event:
1. Log to doctrine memory
2. Escalate to AZ SUPREME via internal queue
3. AZ SUPREME decides response level (INFO/WARNING/CRITICAL)
4. CRITICAL events → immediate Telegram push to @barrenwuffet069bot
5. Update BarrenWuffetState if warranted

### Sub-Agent Spawning

For parallel analysis (e.g., scanning all 50 strategies simultaneously):
- Use OpenClaw sessions_spawn to create temporary sub-agents
- Each sub-agent inherits doctrine state and risk limits
- Results aggregated by parent agent
- Sub-agents auto-terminate after task completion

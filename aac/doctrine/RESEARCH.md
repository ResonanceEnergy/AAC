# RESEARCH.md — BARREN WUFFET OpenClaw Deep Dive Intelligence

> **Compiled**: 2026-02-25 | **Updated**: 2026-02-28 (Deep Dive Batch 2 + 40-Step Gap Fill)
> **Sources**: serif.ai, adversa.ai, theworldmag.com, GitHub VoltAgent/awesome-openclaw-skills, 30 local usecases, Reddit, ClawHub registry, Bitrue, Intellectia.ai, Aurpay, OpenClaws.io, Forbes, Yahoo Finance
> **Agent**: BARREN WUFFET (AZ SUPREME)
> **Classification**: DOCTRINE MEMORY — PERMANENT

---

## TABLE OF CONTENTS

1. [OpenClaw Platform Overview](#1-openclaw-platform-overview)
2. [Architecture Patterns for AAC](#2-architecture-patterns-for-aac)
3. [Finance & Trading Workflows](#3-finance--trading-workflows)
4. [Crypto & DeFi Intelligence](#4-crypto--defi-intelligence)
5. [Polymarket Autopilot Research](#5-polymarket-autopilot-research)
6. [Security Hardening (CRITICAL)](#6-security-hardening-critical)
7. [Community Skills Registry](#7-community-skills-registry)
8. [30 Usecase Patterns](#8-30-usecase-patterns)
9. [Skill Domain Research — All 35+ Domains](#9-skill-domain-research--all-35-domains)
10. [Implementation Patterns](#10-implementation-patterns)
11. [Regulatory Intelligence](#11-regulatory-intelligence)
12. [Deep Dive Batch 2 — External Source Intelligence](#12-deep-dive-batch-2--external-source-intelligence)
13. [FrankenClaw & Crypto Scam Intelligence](#13-frankenclaw--crypto-scam-intelligence)
14. [Backlog & Future Work](#14-backlog--future-work)

---

## 1. OPENCLAW PLATFORM OVERVIEW

### What Is OpenClaw
- **Originally**: Clawdbot (November 2025, Peter Steinberger, Austrian developer)
- **Renamed**: Clawdbot → Moltbot (Jan 27, 2026, Anthropic trademark) → OpenClaw (Jan 29, 2026)
- **Type**: Open-source, self-hosted AI personal assistant
- **Stars**: 145,000+ GitHub stars in weeks of going viral (January 2026)
- **Architecture**: Connects LLMs (Claude, GPT, Gemini) to messaging platforms (WhatsApp, Telegram, Signal, Discord, Slack, iMessage)
- **Capabilities**: File system access, shell commands, email, calendars, web browsers, cron jobs, persistent memory
- **Skills System**: SKILL.md files with YAML frontmatter, installed via `npx clawhub@latest install <slug>` or manually to `~/.openclaw/skills/`
- **Memory**: SOUL.md (identity), MEMORY.md (facts), HEARTBEAT.md (scheduled tasks)
- **Registry**: ClawHub — 5,705 community skills as of Feb 7, 2026

### Installation Paths
| Location | Priority |
|----------|----------|
| `~/.openclaw/skills/` (Global) | Lowest |
| `<project>/skills/` (Workspace) | Highest |
| Bundled | Default |

### Key Financial Skills on ClawHub (22 listed in awesome-openclaw-skills)
- `copilot-money` — Query Copilot Money personal finance data
- `financial-calculator` — Advanced financial calculator (future value, compound interest, amortization)
- `multi-factor-strategy` — Guide for multi-factor stock strategies
- `openinsider` — Fetch SEC Form 4 insider trading data (Directors, CEOs, Officers)
- `budget-variance-analyzer` — Analyze budget vs actual
- `bidclub` / `bidclub-ai` — Post investment ideas to AI-native investment community

---

## 2. ARCHITECTURE PATTERNS FOR AAC

### Pattern 1: Sub-Agent Spawning (Dynamic Dashboard)
Used in 6+ usecases. Spawn sub-agents for **parallel data collection**.
```
Main Agent → spawn(GitHub agent) + spawn(Twitter agent) + spawn(Polymarket agent)
           → aggregate results → render dashboard → alert on thresholds
```
**AAC Application**: BigBrainIntelligence theaters run in parallel sub-agents.

### Pattern 2: Cron-Driven Financial Monitoring
Used in 22 of 30 usecases. Schedule recurring tasks via HEARTBEAT.md.
```yaml
# Every 15 min: check risk
# Hourly: scan opportunities
# Daily 7AM: morning briefing
# Daily 6PM: end-of-day P&L
# Weekly Monday: regulatory scan
# Monthly 1st: performance report
```
**AAC Application**: Already implemented in HEARTBEAT.md with full daily/weekly/monthly schedule.

### Pattern 3: Telegram Topic-Based Routing
Separate Telegram topics for different domains:
```
📊 market-intel     — Real-time market signals
💹 trading-signals  — Active trade alerts
🔗 crypto-intel     — Crypto-specific feeds
📈 portfolio        — Dashboard & P&L
⚠️ risk-alerts      — Doctrine state changes
🧠 second-brain     — Knowledge capture
📰 morning-brief    — Daily briefings
💰 earnings         — Earnings season tracking
```
**AAC Application**: @barrenwuffet069bot routes to appropriate topic per intent.

### Pattern 4: Shared State Files (Multi-Agent Coordination)
Instead of central orchestrator, use shared files:
```
STATE.yaml    — Tasks, statuses, owners, blockers
GOALS.md      — Strategic objectives
DECISIONS.md  — Decision log with rationale
```
**AAC Application**: Doctrine engine serves as STATE machine, BarrenWuffetState as shared state.

### Pattern 5: Event-Driven State Management
Replace Kanban with event sourcing:
```
Event Types: progress, blocker, decision, pivot
Natural Language → auto state transitions
Git commits auto-linked to projects
```

### Pattern 6: n8n Proxy Pattern (Security-First)
```
OpenClaw → webhook call (NO credentials) → n8n Workflow (locked, with API keys) → External Service
```
Agent never touches credentials. Visual debugging. Lockable workflows.
**AAC Application**: Use for all external API integrations (exchanges, data providers).

### Pattern 7: Knowledge Base RAG
```
Input: URL/text → auto-ingest → semantic search over all memories
Query: "What do I know about XRP cross-border?" → relevant memories
```
**AAC Application**: Second Brain skill with Next.js dashboard.

### Pattern 8: Memory-Based Preference Learning
Agent learns user preferences over time. Daily digests get refined based on feedback.
**AAC Application**: BARREN WUFFET learns which strategies, tokens, and markets matter most.

---

## 3. FINANCE & TRADING WORKFLOWS

### Source: serif.ai/openclaw/finance (12 Finance Workflows)

1. **Real-Time Stock & Crypto Alerts**
   - Cron jobs checking prices via financial APIs
   - Configurable thresholds → alerts to Telegram/WhatsApp/Slack
   - Pattern: `cron(*/5) → check_price() → if threshold_breached → send_alert()`

2. **Earnings Calendar Monitoring**
   - Weekly Sunday preview: upcoming earnings for tracked companies
   - One-shot cron jobs scheduled per earnings date
   - Post-earnings: auto-search results, format beat/miss summary
   - Key metrics: revenue, EPS, guidance, AI highlights

3. **News & Sentiment Analysis**
   - Monitor SEC filings, social media, financial news
   - Sentiment scoring per asset/sector
   - Alert on significant sentiment shifts

4. **Regulatory & Compliance Monitoring**
   - Track SEC, FINRA, CFTC updates
   - Monitor for rule changes affecting strategies
   - **AAC Critical**: Calgary (Alberta Securities Commission) + Montevideo (BCU)

5. **Client Portfolio Updates**
   - Automated portfolio performance summaries
   - Rebalancing alerts on drift beyond tolerance bands

6. **Daily Market Briefings**
   - Overnight moves, pre-market analysis
   - P&L impact, analyst calls, economic data
   - **AAC**: Already in HEARTBEAT.md at 07:00 MT

7. **Company & Sector Research**
   - Earnings transcripts, SEC filings (10-K, 10-Q, 8-K)
   - Competitor analysis, sector rotation detection

8. **Screening & Opportunity Identification**
   - P/E ratios, dividend yields, moving average crossovers
   - Technical + fundamental screening hybrid

9. **Trade Logging & Documentation**
   - Automated trade journal with entry/exit, rationale, outcome
   - Strategy performance attribution

10. **Expense & Receipt Management**
    - Trading-related expense tracking (commissions, fees, subscriptions)

### Source: serif.ai/openclaw/financial-planning (10 CFP Workflows)

1. **Client Review Scheduling** (200+ clients, annual reviews)
2. **Plan Update Triggers** (market drawdowns, legislative changes, life events)
3. **Milestone Tracking** (college, mortgage payoff, Social Security, Medicare, RMDs)
4. **Portfolio Rebalancing Alerts** (drift beyond tolerance bands)
5. **Tax-Loss Harvesting** (wash-sale rule awareness, automated scanning)
6. **Estate Planning Coordination** (beneficiary reviews, document updates)
7. **Market Commentary Drafting** (firm voice, brand-consistent)
8. **Compliance Documentation** (ADV filings, suitability docs)
9. **Referral Network Maintenance** (CPA, attorney, insurance relationships)
10. **Continuing Education Tracking** (CFP CE credits)

---

## 4. CRYPTO & DEFI INTELLIGENCE

### Key Crypto Research Findings

**Bitcoin (BTC)**:
- On-chain analysis: UTXO age distribution, miner revenue, hash rate trends
- Halving cycle analysis (2024 halving impact assessment)
- Lightning Network capacity monitoring
- Institutional flow tracking (ETF inflows/outflows)

**Ethereum (ETH)**:
- DeFi protocol TVL monitoring (Aave, Uniswap, Compound, Lido)
- Gas price optimization windows
- L2 ecosystem tracking (Arbitrum, Optimism, Base, zkSync)
- MEV (Maximal Extractable Value) opportunity detection

**XRP/Ripple**:
- Cross-border payment corridor monitoring
- SEC litigation tracking and resolution analysis
- XRPL DEX order book analysis
- ODL (On-Demand Liquidity) corridor volumes

**Stablecoins**:
- Peg deviation monitoring (USDT, USDC, DAI, FRAX)
- Yield farming opportunities across protocols
- De-peg early warning system (historical: UST collapse patterns)
- Cross-chain stablecoin arbitrage

**Meme Coins**:
- Social velocity tracking (Reddit, X/Twitter, Telegram groups)
- New launch radar with rug-pull detection patterns
- Volume spike correlation with social mentions
- Whale wallet monitoring for meme token accumulation

**World Liberty Coin (WLFI)**:
- Political token tracking
- Governance token framework analysis
- Regulatory classification monitoring

**X Tokens**:
- Token ecosystem intelligence
- Cross-platform token utility analysis

---

## 5. POLYMARKET AUTOPILOT RESEARCH

### Source: theworldmag.com — OpenClaw Polymarket Bot ($116K in 24 Hours)

**Account**: Bidou28old
**Performance**: 52 trades, 83% success rate, $116,280.60 profit (Feb 12-13, 2026)

**Four-Stage Process**:
1. Continuously scan Polymarket for mispriced contracts & liquidity gaps
2. Calculate fair value using quantitative models (Black-Scholes adapted for binary options)
3. Compare theoretical price vs actual orderbook data from CLOB API
4. Execute trades only if fair value beats ask by ≥ 6-cent edge requirement

**Black-Scholes Connection**:
- Tracks BTC price deltas from Binance WebSocket feeds
- Calculates fair value for binary options
- 6-cent minimum edge = buffer for fees, slippage, execution risk

**Market Infrastructure**:
- Polymarket CLOB (Central Limit Order Book) architecture
- UMA Optimistic Oracle for resolution (2-hour undisputed resolution)
- CFTC regulatory approval (November 2025)

**AAC Paper Trading Strategies (from local usecases)**:
- **TAIL**: Follow trends (>60% probability + volume spike)
- **BONDING**: Contrarian on overreactions (drops >10%)
- **SPREAD**: Arbitrage when YES+NO > 1.05
- Starting capital: $10,000 paper
- SQL schema: `paper_trades` (market_id, strategy, direction, entry/exit, pnl)

**Risks**:
- Resolution disputes can lock capital
- Oracle failures
- Smart contract vulnerabilities
- Competition from other bots
- Regulatory uncertainty (Polymarket vs Massachusetts lawsuit, Feb 2026)

---

## 6. SECURITY HARDENING (CRITICAL)

### Source: adversa.ai — OpenClaw Security 101 (February 2026)

#### The Lethal Trifecta (Simon Willison)
1. **Access to private data**: emails, files, credentials, browser history, chat messages
2. **Exposure to untrusted content**: web browsing, incoming messages, third-party skills
3. **Ability to communicate externally**: sends emails, posts messages, API calls, data exfiltration
4. **+Persistent memory** (Palo Alto Networks extension): SOUL.md/MEMORY.md enable time-shifted prompt injection

#### Critical CVEs
| CVE | Type | CVSS | Description |
|-----|------|------|-------------|
| CVE-2026-25253 | One-Click RCE | 8.8 | Token exfiltration via malicious gatewayUrl |
| CVE-2026-24763 | Command Injection | HIGH | Unsanitized input fields in gateway |
| CVE-2026-25157 | Command Injection | HIGH | Additional injection vector |
| CVE-2026-22708 | Indirect Prompt Injection | HIGH | CSS-hidden instructions in web content |

#### ClawHavoc Supply Chain Attack
- 341 malicious skills found out of 2,857 audited (12%)
- 335 delivered Atomic Stealer (AMOS) macOS malware
- 6 contained reverse shell backdoors
- All shared single C2 IP: `91.92.242[.]30`
- Campaign window: January 27-29, 2026

#### AAC Security Measures (MANDATORY)
```
1. Set gateway.auth.password — NEVER run without authentication
2. Set API spending limits — hard caps on Anthropic/OpenAI keys
3. Run in Docker with --read-only and --cap-drop=ALL
4. Bind Control UI to 127.0.0.1 ONLY (never 0.0.0.0)
5. Use Tailscale or VPN for remote access
6. NEVER install unaudited ClawHub skills
7. Rotate all tokens regularly
8. Run: openclaw security audit --deep --fix
9. Enable TLS 1.3 for gateway communications
10. Use n8n proxy pattern — agent never touches API credentials directly
```

#### Credential Storage Warning
OpenClaw stores tokens in **plaintext Markdown and JSON files** under `~/.openclaw/`.
- RedLine, Lumma, Vidar malware families targeting this directory
- ALWAYS encrypt credentials at rest
- Use system keychains or vault services
- AAC uses `.env` (gitignored) — ensure this remains protected

#### Runaway API Cost Protection
- Benjamin De Kraker burned $20 overnight from time-checking heartbeat
- 120,000 tokens per context check × $0.75 each = $750/month for a simple reminder
- **AAC Mandate**: Hard spending limits on all API keys, cost monitoring alerts

---

## 7. COMMUNITY SKILLS REGISTRY

### Source: VoltAgent/awesome-openclaw-skills (GitHub)

**Total Skills**: 2,868 curated (from 5,705 on ClawHub)
**Filtered Out**: 1,180 spam, 672 crypto/finance, 492 duplicates, 396 malicious, 8 non-English

**Categories (28 total)**:
| Category | Count | Category | Count |
|----------|-------|----------|-------|
| AI & LLMs | 287 | Search & Research | 253 |
| DevOps & Cloud | 212 | Web & Frontend | 202 |
| Marketing & Sales | 143 | Browser & Automation | 139 |
| Productivity & Tasks | 135 | Coding Agents | 133 |
| Communication | 133 | CLI Utilities | 129 |
| Clawdbot Tools | 120 | Notes & PKM | 100 |
| Media & Streaming | 80 | Transportation | 76 |
| Git & GitHub | 66 | PDF & Documents | 67 |
| Speech & Transcription | 65 | Security & Passwords | 64 |
| Gaming | 61 | Image & Video Gen | 60 |
| Personal Development | 56 | Smart Home & IoT | 56 |
| Health & Fitness | 55 | Moltbook | 51 |
| Shopping & E-commerce | 51 | Calendar & Scheduling | 50 |
| Data & Analytics | 46 | **Finance** | **22** |
| Self-Hosted | 25 | Agent-to-Agent | 18 |

**Relevant Finance/Trading Skills on ClawHub**:
- `copilot-money` — Personal finance data queries
- `financial-calculator` — Advanced financial calculations
- `multi-factor-strategy` — Multi-factor stock strategy builder
- `openinsider` — SEC Form 4 insider trading data
- `budget-variance-analyzer` — Budget vs actual analysis
- `bidclub` / `bidclub-ai` — AI investment community

**Security Notice**: Skills curated but NOT audited. Always review source code.
Recommended scanners:
- [Snyk Skill Security Scanner](https://github.com/snyk/agent-scan)
- [Agent Trust Hub](https://ai.gendigital.com/agent-trust-hub)
- [Cisco Skill Scanner](https://blogs.cisco.com/ai/personal-ai-agents-like-openclaw-are-a-security-nightmare)

---

## 8. 30 USECASE PATTERNS

### Financial/Trading Usecases (Direct AAC Relevance)

| # | Usecase | AAC Skill Mapping |
|---|---------|-------------------|
| 8 | Earnings Tracker | bw-market-intelligence, bw-morning-briefing |
| 23 | Polymarket Autopilot | bw-polymarket-autopilot |
| 14 | Market Research & Product Factory | bw-market-intelligence |
| 7 | Dynamic Dashboard (Sub-Agents) | bw-portfolio-dashboard |
| 4 | Custom Morning Brief | bw-morning-briefing |
| 25 | Second Brain | bw-second-brain |
| 15 | Multi-Agent Team | Core AAC multi-agent architecture |
| 16 | Multi-Channel Assistant | BARREN WUFFET Telegram bot |

### Infrastructure Usecases (AAC Operations)

| # | Usecase | AAC Application |
|---|---------|----------------|
| 19 | n8n Workflow Orchestration | Exchange API integration security |
| 26 | Self-Healing Home Server | AAC infrastructure monitoring |
| 27 | Semantic Memory Search | Doctrine memory search |
| 2 | Autonomous Project Management | Shared STATE.yaml pattern |
| 24 | Project State Management | Event-driven state tracking |

### Key Cross-Cutting Patterns

| Pattern | Frequency |
|---------|-----------|
| Cron/Scheduled Jobs | 22/30 usecases |
| Telegram as primary interface | 15+ usecases |
| Sub-agent spawning | 6 usecases |
| SQLite/PostgreSQL databases | 5 usecases |
| Memory/preference learning | 4 usecases |
| RAG/knowledge base | 4 usecases |
| Security-first design | 3 usecases |

---

## 9. SKILL DOMAIN RESEARCH — ALL 35+ DOMAINS

### Domain 1: Market Intelligence
- **OpenClaw Pattern**: Cron-triggered scans across multiple data sources
- **APIs**: Alpha Vantage, Yahoo Finance, Polygon.io, Finnhub, TradingView
- **Sub-agents**: Narrative, Engagement, Content, Latency, Bridge, Gas, Liquidity
- **Output**: ResearchFinding objects with signal strength (0.0-1.0)

### Domain 2: Trading Signals
- **OpenClaw Pattern**: Pipeline architecture (Research → Signal → Aggregator → Consensus)
- **Signal Categories**: Statistical, Structural, Technology, Compliance, Cross-Chain
- **Risk Context**: Doctrine compliance, drawdown, Kelly criterion sizing

### Domain 3: Portfolio Management
- **OpenClaw Pattern**: Dashboard with parallel sub-agent data collection
- **Metrics**: NAV, P&L (daily/weekly/monthly), VaR, Sharpe, drawdown
- **Schema**: positions, strategies, attribution, contest leaderboard

### Domain 4: Digital Arbitrage
- **OpenClaw Pattern**: Real-time WebSocket feeds + CLOB API comparison
- **Sources**: Binance, Coinbase, Kraken, DEX aggregators
- **Edge**: ≥6-cent minimum edge (Polymarket model)

### Domain 5: Arbitrage (Traditional)
- **OpenClaw Pattern**: Multi-venue price comparison with latency optimization
- **50 Strategies**: Stat pairs, mean reversion, cointegration, microstructure, order book imbalance

### Domain 6: Banking Intelligence
- **OpenClaw Pattern**: Regulatory alert monitoring + compliance scanning
- **Scope**: Offshore structures, international wire optimization, multi-currency
- **Jurisdictions**: Calgary (Alberta), Montevideo (Uruguay), international corridors

### Domain 7: Hedging Strategies
- **OpenClaw Pattern**: Portfolio exposure analysis → hedge recommendation
- **Methods**: Options hedging (puts, collars), futures hedging, correlation-based

### Domain 8: Day Trading
- **OpenClaw Pattern**: Real-time momentum scanner with gap detection
- **Activation**: Market open (9:00 MT), morning scan for gaps
- **Methods**: Scalping, momentum plays, VWAP reversion, opening range breakout

### Domain 9: Crypto Intelligence
- **OpenClaw Pattern**: On-chain + off-chain data fusion
- **Capabilities**: Whale tracking, DeFi yields, MEV detection, mempool intelligence
- **Pipeline**: CryptoIntelligence → IntelligenceSignal → CryptoBigBrainBridge

### Domain 10: Stablecoins
- **OpenClaw Pattern**: Peg monitoring cron (every 15 min), yield comparison
- **Tracked**: USDT, USDC, DAI, FRAX, PYUSD
- **Alert**: De-peg deviation > 0.5%, historical patterns (UST collapse)

### Domain 11: Meme Coins
- **OpenClaw Pattern**: Social velocity + new launch radar
- **Data**: Reddit mentions, X/Twitter velocity, Telegram group activity
- **Rug-Pull Detection**: Liquidity lock check, contract audit, team wallet concentration

### Domain 12: 2007 Crash Indicators
- **OpenClaw Pattern**: Daily cron (06:45 MT) + weekly comparison
- **Indicators**: Credit spreads, yield curve inversion, VIX term structure, bank CDS, 
  housing starts, leverage ratios, LIBOR-OIS spread, commercial paper stress
- **Pattern Matching**: Current values vs 2007 timeline (6-month regression)

### Domain 13: Calls & Puts (Options Flow)
- **OpenClaw Pattern**: Pre-market scan (07:30 MT) + afternoon flow (14:00 MT)
- **Data**: Unusual options activity, sweep detection, block trades
- **Analysis**: Put/call ratio, max pain, gamma exposure, dark pool prints

### Domain 14: Options Trading
- **OpenClaw Pattern**: Strategy recommendation engine
- **Strategies**: Covered calls, protective puts, iron condors, butterflies, straddles
- **Greeks**: Real-time Delta, Gamma, Theta, Vega tracking
- **Chains**: Full options chain visualization with probability of profit

### Domain 15: SuperStonk DD
- **OpenClaw Pattern**: Weekly deep research cycle (Tuesday)
- **Methodology**: FTD cycle analysis (T+35), short interest data, institutional ownership
- **Sources**: SEC EDGAR, FINRA data, Reddit r/Superstonk, DD archive
- **Tracking**: Dark pool volume, share borrow rates, swap exposure estimates

### Domain 16: Jonny Bravo Trading Course
- **OpenClaw Pattern**: Weekly lesson + quiz (Thursday)
- **Curriculum**: Technical analysis, risk management, position sizing, 
  psychology, market structure, order flow
- **Division**: Jonny Bravo Division within AAC

### Domain 17: Dan Winter Golden Ratio Finance
- **OpenClaw Pattern**: Overnight harmonic convergence zones (22:00 MT)
- **Application**: Golden ratio (1.618) applied to:
  - Fibonacci retracements and extensions
  - Time cycle harmonics
  - Price-time confluence zones
  - Fractal market analysis

### Domain 18: Finance & Accounting
- **OpenClaw Pattern**: Daily reconciliation (16:00 MT)
- **Capabilities**: P&L reporting, fee analysis, tax lot tracking
- **Integration**: CentralAccounting department

### Domain 19: Money Planning & Mastery
- **OpenClaw Pattern**: Monthly review cycle (1st of month)
- **Framework**: Income tracking, expense categorization, savings rate,
  emergency fund status, debt payoff trajectory
- **Cash Flow**: Real-time cash position monitoring

### Domain 20: Wealth Building
- **OpenClaw Pattern**: Quarterly strategy review
- **Strategy**: Asset allocation, compound growth projections, 
  tax-advantaged account optimization
- **Long-term**: 10/20/30-year wealth projections with Monte Carlo simulation

### Domain 21: Generational Wealth
- **OpenClaw Pattern**: Weekly rebalance review (Monday)
- **Framework**: Estate planning triggers, trust structure monitoring,
  multi-generational transfer strategies, tax efficiency

### Domain 22: Offshore Accounts & International Banking
- **OpenClaw Pattern**: Regulatory alert + corridor optimization (Wednesday)
- **Corridors**: Calgary ↔ Montevideo primary
- **Monitoring**: International wire fees, FX spreads, compliance requirements
- **Structures**: Multi-currency accounts, international trusts

### Domain 23: Regulations
- **OpenClaw Pattern**: Monday regulatory scan
- **Local**: Alberta Securities Commission (ASC), Calgary bylaws
- **International**: BCU Uruguay, CFTC, SEC, FINRA, EU MiCA
- **Crypto**: Regulatory classification tracking per jurisdiction

### Domain 24: Cash & Currency Trading
- **OpenClaw Pattern**: Morning FX analysis (08:30 MT)
- **Focus**: CAD/USD, major pairs, emerging market currencies
- **Data**: Central bank rate decisions, intervention risk, carry trade analysis

### Domain 25: Bitcoin Intelligence
- **See Domain 9 + dedicated on-chain analysis**

### Domain 26: Ethereum & DeFi
- **See Domain 9 + dedicated DeFi protocol monitoring**

### Domain 27: X Tokens
- **OpenClaw Pattern**: Token ecosystem monitoring
- **Analysis**: Utility, governance, tokenomics review

### Domain 28: XRP/Ripple
- **OpenClaw Pattern**: Cross-border corridor intelligence
- **Focus**: ODL volumes, XRPL DEX, SEC case resolution impact

### Domain 29: World Liberty Coin
- **OpenClaw Pattern**: Political token tracking
- **Analysis**: Governance framework, regulatory classification

### Domain 30: Polymarket Autopilot
- **See Section 5 — Full Polymarket research**

### Domain 31: Second Brain
- **OpenClaw Pattern**: Zero-friction capture → semantic search
- **Architecture**: Text anything → agent remembers → search everything
- **Dashboard**: Next.js with Cmd+K global search

### Domain 32: Morning Briefing
- **OpenClaw Pattern**: 07:00 MT automated briefing via Telegram
- **Sections**: Market overview, P&L, signals, risk, crash score, actions
- **Inspired By**: Custom Morning Brief usecase

### Domain 33: Earnings Tracker
- **OpenClaw Pattern**: Weekly preview + per-date cron jobs
- **Workflow**: Sunday scan → user picks → schedule one-shots → post-report summary

### Domain 34: Dynamic Dashboard
- **OpenClaw Pattern**: Sub-agent spawning for parallel data
- **Sources**: GitHub, Twitter, Polymarket, system health
- **Storage**: PostgreSQL metrics + alerts tables

### Domain 35: Agent Roster
- **OpenClaw Pattern**: Live directory of 80+ agents
- **Display**: Name, department, status, specialization

---

## 10. IMPLEMENTATION PATTERNS

### SKILL.md File Format
```yaml
---
name: bw-market-intelligence
description: Scan markets across 3 operational theaters
metadata:
  openclaw:
    emoji: "📊"
    homepage: "https://github.com/ResonanceEnergy/AAC"
    requires:
      env:
        - AAC_API_KEY
    primaryEnv: AAC_API_KEY
    always: false  # true = always active
---

## Skill Instructions

Instructions for the agent go here in markdown format.
```

### Cron Job Pattern (HEARTBEAT.md)
```markdown
- Every N minutes: `action_name` — description
- Daily at HH:MM: `action_name` — description
- Weekly on DAY: `action_name` — description
```

### Sub-Agent Pattern
```python
# Spawn parallel sub-agents for data collection
sessions_spawn("market-data-agent", task="fetch BTC price from Binance")
sessions_spawn("news-agent", task="scan financial news last 1h")
sessions_spawn("social-agent", task="check Reddit/X sentiment")
# Wait for all → aggregate → render dashboard
```

### Database Pattern (PostgreSQL/SQLite)
```sql
CREATE TABLE paper_trades (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL,
    strategy TEXT NOT NULL,      -- TAIL, BONDING, SPREAD
    direction TEXT NOT NULL,     -- YES, NO
    entry_price DECIMAL(10,4),
    exit_price DECIMAL(10,4),
    pnl DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,        -- github, twitter, polymarket
    metric_name TEXT NOT NULL,
    metric_value DECIMAL(15,4),
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    condition TEXT NOT NULL,
    threshold DECIMAL(10,4),
    triggered BOOLEAN DEFAULT FALSE,
    last_triggered TIMESTAMP
);
```

### Telegram Bot Routing
```python
# Intent detection → skill routing
INTENT_MAP = {
    r"market|intel|scan|theater": "bw-market-intelligence",
    r"signal|trade|entry|exit": "bw-trading-signals",
    r"portfolio|dash|pnl|nav": "bw-portfolio-dashboard",
    r"risk|drawdown|doctrine|halt": "bw-risk-monitor",
    r"crypto|onchain|defi|whale": "bw-crypto-intel",
    r"bitcoin|btc|satoshi|halving": "bw-bitcoin-intel",
    r"ethereum|eth|gas|l2|defi": "bw-ethereum-defi",
    r"xrp|ripple|xrpl|odl": "bw-xrp-ripple",
    r"stable|usdt|usdc|dai|peg": "bw-stablecoins",
    r"meme|doge|shib|pepe|moon": "bw-meme-coins",
    r"options|calls|puts|greeks": "bw-options-trading",
    r"hedge|protect|collar|insurance": "bw-hedging-strategies",
    r"polymarket|prediction|binary": "bw-polymarket-autopilot",
    r"superstonk|gme|ftd|short": "bw-superstonk-dd",
    r"crash|2007|2008|indicator|vix": "bw-crash-indicators",
    r"golden.ratio|fibonacci|harmonic": "bw-golden-ratio-finance",
    r"jonny.bravo|lesson|quiz|course": "bw-jonny-bravo-course",
    r"brief|morning|overnight|recap": "bw-morning-briefing",
}
```

---

## 11. REGULATORY INTELLIGENCE

### Calgary, Alberta, Canada 🇨🇦
- **Regulator**: Alberta Securities Commission (ASC)
- **Key Regulations**: Alberta Securities Act, National Instruments (NI 31-103, NI 45-106)
- **Crypto**: CSA Staff Notice 21-327 (Guidance on Crypto Asset Trading Platforms)
- **Tax**: Canada Revenue Agency — crypto as commodity, capital gains 50% inclusion rate
- **Banking**: CDIC insured deposits, FINTRAC AML/KYC requirements
- **Timezone**: Mountain Time (America/Edmonton)

### Montevideo, Uruguay 🇺🇾
- **Regulator**: Banco Central del Uruguay (BCU)
- **Key Regulations**: Ley 18.627 (Securities Market), updated crypto framework
- **Crypto**: BCU Circular 2377 (virtual asset service providers)
- **Tax**: Territorial taxation, favorable for international income
- **Banking**: Multi-currency accounts, stable banking system
- **Timezone**: Uruguay Time (America/Montevideo, UTC-3)

### Cross-Jurisdictional Considerations
- Wire transfer corridors (CAD ↔ UYU ↔ USD)
- Tax treaty implications (Canada-Uruguay DTC)
- Crypto regulatory arbitrage opportunities (doctrine-compliant)
- International trust structures (estate planning)

### CFTC & Prediction Markets
- CFTC no-action letter to QCX LLC (September 2025)
- Polymarket authorized as intermediated trading platform
- Polymarket vs Massachusetts (filed Feb 9, 2026) — state regulation dispute

---

## 12. DEEP DIVE BATCH 2 — EXTERNAL SOURCE INTELLIGENCE

### Source: Bitrue.com — OpenClaw Trading Bot Review (Feb 21, 2026)
**URL**: bitrue.com/blog/openclaw-trading-bot-review-reliable-or-risky

**Key Findings**:
- OpenClaw has **150,000+ GitHub stars** (confirmed, up from 145K earlier)
- **Non-custodial**: Funds remain in user's exchange wallet at all times
- **Self-improving AI**: Adds circuit breakers after consecutive losses, skips trades during chop, applies ADX trend filters, uses Bollinger Bands for volatility
- **Tested win rates**: 58–75% in controlled environments (backtests ≠ live)
- **Lifetime license**: ~$249 post-demo (for managed configurations)

**Trading Modes Supported**:
| Mode | Description |
|------|-------------|
| DCA Ladders | Accumulate positions during dips |
| Grid Trading | Profit in range-bound markets |
| Contrarian | Bet against crowd panic |
| Arbitrage | Multi-exchange price discrepancies |
| Prediction Market | Polymarket execution |

**Reliability Assessment (Bitrue)**:
- ✅ Reliable as **framework** (structurally capable, open-source)
- ❌ NOT reliable as **passive income** (requires engineering, monitoring, risk control)
- ⚠️ LLM hallucinations affecting trade logic (~20% action failure rate)
- ⚠️ API misconfigurations, execution delays, overfitting risk
- ⚠️ Agent-based malware risks from unverified skills

**General Crypto Performance**:
- Modest returns (2–3%) in broader crypto benchmarks
- Low risk-adjusted metrics (weak Sortino ratios)
- Slippage and fee drag eroding profitability
- Over-optimization in backtests failing live

**Safe Usage Protocol (Bitrue)**:
1. Start with demo/paper trading
2. Backtest across bull, bear, sideways regimes
3. Limit risk to 1–2% of portfolio
4. Monitor VPS logs daily
5. NEVER set-and-forget

**AAC Application**: Validates our doctrine state machine approach (NORMAL → CAUTION → SAFE_MODE → HALT). The self-improving AI pattern maps directly to BarrenWuffetState transitions. DCA Ladders and Grid Trading should be added to the 50 arbitrage strategies CSV.

---

### Source: Intellectia.ai — OpenClaw Investment Journey Guide (Feb 13, 2026)
**URL**: intellectia.ai/blog/openclaw-ai-investing-impact-2026-guide

**Key Findings**:
- OpenClaw described as "democratization of institutional-grade investing"
- **Heartbeat Engine**: Proactively monitors markets 24/7, initiates actions autonomously
- **Natural Language Strategy**: Define complex strategies conversationally — no programming needed
  - Example: "Monitor AAPL and notify if it drops >5% below 50-day MA, provided market not in confirmed downtrend"
- **13+ messaging platform** support (WhatsApp, Telegram, Discord, email, etc.)
- **Graduated Permission System**: monitoring-only → paper trading → small positions → full automation
- **Complete audit trails**: Every action logged with timestamp and rationale

**Investor Type Applications**:
| Type | OpenClaw Use |
|------|-------------|
| Time-Constrained Professional | Auto rebalancing, stop-loss, earnings summaries |
| Active Trader | Multi-timeframe scanning, arbitrage, social sentiment, dynamic trailing stops |
| Long-Term Wealth Builder | DCA automation, tax-loss harvesting, Monte Carlo projections |

**Multi-Agent Ecosystem Future**:
- Specialized agents collaborating (technical analysis + fundamental research + risk management)
- OpenClaw's modular skill architecture positioned for multi-agent future
- **AAC Already Implements This**: 80+ agents, 7 departments, shared state coordination

**Security Advantage (Self-Hosted)**:
- API keys never leave your infrastructure
- Trading strategies remain private
- No counterparty risk from cloud providers
- Full control of security perimeter

**AAC Application**: The graduated permission system should be implemented in Telegram bot — `/setmode monitor|paper|small|full`. Heartbeat Engine pattern validates our HEARTBEAT.md cron approach. Natural language strategy definition maps to Telegram intent routing.

---

### Source: Aurpay.net — 10 Crypto Trading Use Cases (Feb 1, 2026)
**URL**: aurpay.net/aurspace/use-openclaw-moltbot-clawdbot-for-crypto-traders-enthusiasts/

**10 Detailed Crypto Automations**:

1. **Real-Time Market Monitoring**
   - Poll APIs for price, volume, on-chain data
   - Persistent memory tracks trends across sessions
   - Sample: "Monitor SOL and alert if 5% drop in 1h or whale dumps >1M tokens"
   - Sample: "Track ETH gas fees, notify when below 20 gwei"
   - Expert tip: Combine with cron jobs for 24/7 on VPS (e.g., Hetzner)

2. **Automated DEX/CEX Trading**
   - Connect to Uniswap, Binance via CCXT library
   - Sample: "Buy 0.5 ETH on Uniswap if BTC > $100K and slippage <2%"
   - Sample: "Sell half SOL if RSI >70 on 1H chart"
   - Strategies: DCA, grid trading, limit orders on DEXs
   - Expert tip: Backtest with Pandas before live trading

3. **Arbitrage Detection & Execution**
   - Scan price discrepancies across exchanges/protocols
   - Use Pumpmolt skill for Solana pump detection
   - Sample: "Scan BTC arb between Binance and Polymarket; execute if spread >1%"
   - Real example: -$500 → $106K with 95% win rate (isolated case)
   - Expert tip: Flash loans for capital-efficient arbs

4. **Social Sentiment & Alpha Scraping**
   - Analyze X, Reddit, Telegram for token sentiment
   - BeautifulSoup/API wrappers for scraping
   - Sample: "Analyze $ETH sentiment from 24h X posts; buy if bullish >70%"
   - Expert tip: NLP libraries (NLTK) for accurate sentiment scoring

5. **Portfolio Management & Rebalancing**
   - Track across wallets and chains via Etherscan/Solana RPC
   - Sample: "Rebalance to 40% BTC, 30% ETH, 30% stables if drift >5%"
   - Sample: "Generate weekly portfolio report with Sharpe ratio"
   - Expert tip: DePIN for decentralized compute on complex simulations

6. **Yield Farming & LP Automation**
   - Auto-deposit, claim, compound across DeFi protocols
   - Monitor via DeFiLlama API
   - Sample: "Farm highest APY USDC pool on Base; compound daily"
   - Expert tip: Add rug-pull detectors scanning contract code

7. **Prediction Market Bots**
   - Specialize on Polymarket with news-fed probability estimation
   - Sample: "Analyze US election odds; bet YES if probability >60%"
   - Real example: 95% win rates on short-term options via spread exploitation

8. **On-Chain Research & Token Analysis**
   - Pull transaction data, verify contracts via Dune Analytics/Etherscan
   - Sample: "Scan $MOLT contract for rugs; check dev wallet"
   - Real example: Agent verified token at $700K MCAP, spotted signals before 50x run

9. **Wallet Management & Multi-Chain**
   - Batch transactions, airdrop farming, cross-chain bridges
   - Sample: "Bridge 1 ETH from Mainnet to Base if fees low"
   - Expert tip: Hardware wallets for cold storage integration

10. **Risk Management & Strategy Optimization**
    - Implement stops, volatility filters, AI-optimized plays
    - Sample: "Pause all trades if BTC volatility >5%"
    - Expert tip: Persistent memory for evolving strategies from past performance

**Security Warnings (Aurpay)**:
- Use read-only API keys where possible
- Enable sandbox mode to block unwanted network calls
- Run in Docker for isolation
- Never store private keys in plain text — use environment variables
- **Beware fake tokens**: $CLAWD, $OPENCLAW tokens are rug-pulls, NOT official
- Founder Peter Steinberger confirms: **No official OpenClaw token exists**

**AAC Application**: All 10 patterns map to our existing 35 skills. Key additions: Flash loan arbitrage pattern, CCXT library for CEX integration, DeFiLlama API for yield farming, Pumpmolt skill for Solana. Persistent memory validates our doctrine MEMORY.md approach.

---

### Source: VoltAgent/awesome-openclaw-skills (GitHub, Feb 2026)
**URL**: github.com/VoltAgent/awesome-openclaw-skills

**Registry Deep Dive**:
- **3,255 lines** of curated skill listings (454 KB README)
- **2,868 skills** curated from 5,705 total on ClawHub
- **Filtering breakdown** (updated):
  | Reason | Count |
  |--------|-------|
  | Spam/bot/test/junk | 1,180 |
  | Crypto/Finance/Trade | 672 |
  | Duplicate/Similar | 492 |
  | Malicious (security audits) | 396 |
  | Non-English | 8 |
  | **Total filtered** | **2,748** |

**Security Tools (Updated)**:
- [Snyk Agent Scan](https://github.com/snyk/agent-scan) — Skill security scanner
- [Agent Trust Hub](https://ai.gendigital.com/agent-trust-hub) — Trust verification
- VirusTotal partnership — ClawHub skill scanning

**Notable Finance-Adjacent Skills Found**:
- `openinsider` — SEC Form 4 insider trading data (Directors, CEOs, Officers)
- `copilot-money` — Personal finance data queries
- `financial-calculator` — Future value, compound interest, amortization
- `multi-factor-strategy` — Multi-factor stock strategy builder
- `budget-variance-analyzer` — Budget vs actual analysis
- `bidclub` / `bidclub-ai` — AI investment community
- `trend-watcher` — GitHub Trending + tech community monitoring

**Publishing Rules**: Skills must be in `github.com/openclaw/skills` repo first; no personal repos, gists, or external sources accepted.

---

## 13. FRANKENCLAW & CRYPTO SCAM INTELLIGENCE

### Source: OpenClaws.io — FrankenClaw Scam Report (Feb 10, 2026)
**URL**: openclaws.io/blog/frankenclaw-crypto-scam/

**Incident Timeline**:
- **Date**: Early February 2026
- **Discovery**: Feb 8, 2026 (community flagged in Discord)
- **Response**: Within 24 hours, takedown requests filed

**Scam Anatomy**:
- Project called "FrankenClaw" — claimed to be "official tokenization of OpenClaw AI agent ecosystem"
- Promised **500% returns within 90 days** via "AI-powered trading agent"
- Used OpenClaw logo, color scheme, typography (modified)
- Fabricated developer testimonials, fake partnership announcements
- Professional website with technical jargon whitepaper
- Token: **FCLAW** issued on decentralized exchange
- Coordinated campaign: paid influencers, bot engagement, targeted ads
- Telegram group: **15,000+ members** (mostly bots), critics banned immediately
- Phrases used: "powered by OpenClaw," "the official OpenClaw token," "backed by the OpenClaw community"

**Financial Impact**:
- On-chain analysis: **~$2.3 million extracted** from investors
- Classic pump-and-dump: team held large FCLAW allocation, promoted aggressively, sold at peak
- Investors left with worthless tokens

**OpenClaws.io Response**:
1. Official statement published and pinned across all channels
2. Takedown requests to domain registrar, social media, DEX listing
3. FrankenClaw website taken offline
4. Social media accounts suspended
5. Reported to law enforcement and financial regulators (multiple jurisdictions)
6. IP counsel engaged for trademark protection

**How to Identify Scams (Official Guidance)**:
- ❌ **NO OpenClaw token, cryptocurrency, or investment vehicle exists**
- OpenClaw funded through grants, donations, corporate sponsorships
- Official channels ONLY: openclaws.io, official GitHub, official Discord, verified social accounts
- ANY "OpenClaw token" or "OpenClaw investment opportunity" = **FRAUD**
- Never trust guaranteed return promises
- Always verify via official community channels

**Other Known Scam Tokens (from Aurpay)**:
- $CLAWD — rug-pull
- $OPENCLAW — rug-pull
- $MOLT — mixed signals (one verified at $700K MCAP, others fake)

**AAC Security Doctrine Update**:
- Add FrankenClaw pattern to scam detection rules
- Monitor for AAC/BARREN WUFFET brand impersonation
- Implement token legitimacy checker in crypto intelligence pipeline
- Track on-chain evidence of pump-and-dump schemes

---

## 14. BACKLOG & FUTURE WORK

### Priority 1: Immediate (This Sprint)
- [x] ~~Deploy 35 SKILL.md files to `~/.openclaw/workspace/skills/`~~ → **DONE** (scripts/deploy_skills.py)
- [x] ~~Expand to 65 skills (30 Deep Dive Batch 3 skills)~~ → **DONE** (v2.5.0)
- [x] ~~Create 200 research insights document~~ → **DONE** (aac/doctrine/INSIGHTS_200.md)
- [ ] Test Telegram bot command routing for all 65 skills
- [x] ~~Configure API spending hard limits~~ → **DONE** (services/api_spending_limiter.py)
- [ ] Run `openclaw security audit --deep --fix`
- [ ] Set up n8n proxy for exchange API credentials
- [ ] Deploy new SKILL.md files for skills 36-65

### Priority 2: Near-Term (Next 2 Weeks)
- [x] ~~Implement Polymarket paper trading database~~ → **DONE** (data/paper_trading/polymarket_schema.py, SQLite)
- [x] ~~Build earnings tracker with one-shot cron scheduling~~ → **DONE** (trading/earnings_tracker.py)
- [x] ~~Deploy semantic memory search (memsearch) over doctrine files~~ → **DONE** (shared/doctrine_memsearch.py)
- [ ] Create Next.js Second Brain dashboard
- [x] ~~Implement sub-agent spawning for parallel data collection~~ → **DONE** (core/sub_agent_spawner.py)
- [ ] Implement graduated permission system (bw-graduated-mode: monitor → paper → small → full)
- [ ] Build NLP sentiment pipeline (VADER + FinBERT + LLM context)
- [ ] Integrate CCXT for unified CEX trading across 5+ exchanges

### Priority 3: Medium-Term (Next Month)
- [ ] Integrate Bird skill for X/Twitter analysis
- [ ] Build knowledge-base RAG pipeline
- [ ] Implement n8n workflow orchestration for all external APIs
- [ ] Create dynamic dashboard with sub-agent architecture
- [ ] Add multi-factor stock strategy builder
- [ ] Set up WebSocket feed manager (Binance, Coinbase, Kraken)
- [ ] Build real-time on-chain forensics pipeline (Dune + Etherscan)
- [ ] Implement Cisco/Snyk skill scanner for ClawHub skill vetting
- [ ] Create tax-loss harvesting engine with wash-sale awareness (CRA + IRS)
- [ ] Build estate planning document tracker (Canada + Uruguay structures)

### Priority 4: Long-Term (Quarter)
- [ ] Vector embeddings for semantic dedup (YouTube pipeline pattern)
- [ ] Event-driven state management (replace Kanban with event sourcing)
- [ ] Phone-based assistant (ClawdTalk integration)
- [ ] Self-healing infrastructure agent
- [ ] Multi-channel customer service for AAC stakeholders
- [ ] Full Black-Scholes pricing engine for prediction markets
- [ ] Monte Carlo backtester with 10,000+ simulation runs
- [ ] Cross-platform prediction market arbitrage (Polymarket/Kalshi/Metaculus)
- [ ] Kelly Criterion + VaR integration for portfolio-level risk management

### Research Gaps (Need More Data)
- [x] ~~Dan Winter golden ratio specific implementation details~~ → **FILLED** (strategies/golden_ratio_finance.py)
- [x] ~~Jonny Bravo trading course curriculum specifics~~ → **FILLED** (agent_jonny_bravo_division/jonny_bravo_agent.py)
- [x] ~~Bitrue exchange API integration patterns~~ → **FILLED** (Bitrue trading bot review, Section 12)
- [x] ~~Aurpay crypto payment gateway details~~ → **FILLED** (Aurpay 10 crypto use cases, Section 12)
- [x] ~~Intellectia AI financial analysis integration~~ → **FILLED** (Intellectia investment guide, Section 12)
- [x] ~~FrankenClaw scam details~~ → **FILLED** (OpenClaws.io full report, Section 13)
- [x] ~~adversa.ai security hardening full article~~ → **FILLED** (Deep Dive Batch 3 — CVEs, Lethal Trifecta, ClawHavoc, 10 controls)
- [x] ~~GitHub rayandcherry/OpenClaw-financial-intelligence repo analysis~~ → **FILLED** (Deep Dive Batch 3 — Trinity/Panic/2B, Kelly, VaR, ATR)
- [ ] Forbes article content (paywalled)
- [ ] Yahoo Finance article content (paywalled)
- [ ] Reddit deep dive threads (r/ThinkingDeeplyAI, r/ArtificialInteligence, r/hacking)

---

> **— BARREN WUFFET, AZ SUPREME**
> *"Every byte of information is an edge. Store it. Search it. Trade on it."*

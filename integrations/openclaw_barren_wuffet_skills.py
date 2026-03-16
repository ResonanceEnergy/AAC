"""
OpenClaw BARREN WUFFET Skills — Full Spectrum Financial Intelligence
=====================================================================

Comprehensive OpenClaw skills for BARREN WUFFET (Agent AZ) covering every
domain of financial intelligence, trading, wealth building, and market mastery.

Generates AgentSkills-compatible SKILL.md files for the OpenClaw ecosystem.
Each skill exposes AAC capabilities through the OpenClaw control plane,
accessible via Telegram (@barrenwuffet069bot), Discord, WhatsApp, or any
connected channel.

Skills (93 total):
    ─── CORE AAC (10) ───
     1. bw-market-intelligence       — 3-theater market scanning
     2. bw-trading-signals           — Quantum-aggregated signal delivery
     3. bw-portfolio-dashboard       — Real-time portfolio & P&L
     4. bw-risk-monitor              — Doctrine state & risk exposure
     5. bw-crypto-intel              — CryptoIntelligence engine
     6. bw-az-supreme-command        — Executive command interface
     7. bw-doctrine-status           — Doctrine engine state machine
     8. bw-morning-briefing          — Automated morning briefing
     9. bw-agent-roster              — 80+ agent directory
    10. bw-strategy-explorer         — 50 arbitrage strategies

    ─── TRADING & MARKETS (7) ───
    11. bw-digital-arbitrage         — Cross-exchange digital arbitrage
    12. bw-arbitrage-scanner         — Multi-venue arbitrage detection
    13. bw-day-trading               — Intraday momentum & scalping
    14. bw-options-trading           — Options strategies, Greeks, chains
    15. bw-calls-puts-flow           — Options flow: calls, puts, sweeps
    16. bw-hedging-strategies        — Portfolio hedging & risk management
    17. bw-currency-trading          — Forex & currency pair analysis

    ─── CRYPTO & DEFI (7) ───
    18. bw-bitcoin-intel             — Bitcoin on-chain & macro analysis
    19. bw-ethereum-defi             — ETH, DeFi protocols, gas, L2s
    20. bw-xrp-ripple                — XRP/Ripple cross-border payments
    21. bw-stablecoins               — Stablecoin yield, pegs, arbitrage
    22. bw-meme-coins                — Meme coin radar & social velocity
    23. bw-liberty-coin              — Liberty coin tracking & analysis
    24. bw-x-tokens                  — X token ecosystem intelligence

    ─── FINANCE & BANKING (3) ───
    25. bw-banking-intel             — Banking, offshore, international
    26. bw-accounting-engine         — Financial accounting & reporting
    27. bw-regulations               — Local & international regulations

    ─── WEALTH BUILDING (3) ───
    28. bw-money-mastery             — Money planning & cash flow mastery
    29. bw-wealth-building           — Generational wealth strategies
    30. bw-superstonk-dd             — SuperStonk due diligence research

    ─── ADVANCED ANALYSIS (3) ───
    31. bw-crash-indicators          — 2007/2008 crash pattern detection
    32. bw-golden-ratio-finance      — Dan Winter golden ratio harmonics
    33. bw-jonny-bravo-course        — Jonny Bravo trading methodology

    ─── OPENCLAW POWER-UPS (2) ───
    34. bw-polymarket-autopilot      — Prediction market paper trading
    35. bw-second-brain              — Zero-friction knowledge capture

    ─── DEEP DIVE BATCH 3: QUANTITATIVE & PRICING (5) ───
    36. bw-black-scholes             — Black-Scholes binary options pricing
    37. bw-security-hardening        — OpenClaw security & CVE monitoring
    38. bw-skill-scanner             — ClawHub skill security scanning
    39. bw-flash-loans               — DeFi flash loan arbitrage
    40. bw-dca-grid                  — DCA ladders & grid trading

    ─── DEEP DIVE BATCH 3: AI STRATEGIES (5) ───
    41. bw-trinity-scanner           — Trinity/Panic/2B Reversal strategies
    42. bw-backtester                — Multi-strategy backtesting engine
    43. bw-trade-journal             — Automated trade journaling
    44. bw-api-cost-guard            — API cost monitoring & spending limits
    45. bw-graduated-mode            — Graduated trading permissions

    ─── DEEP DIVE BATCH 3: DEFI & YIELD (5) ───
    46. bw-yield-optimizer           — DeFi yield farming optimization
    47. bw-onchain-forensics         — On-chain contract & wallet forensics
    48. bw-sentiment-engine          — NLP multi-source sentiment analysis
    49. bw-sec-monitor               — SEC filings & insider trading
    50. bw-earnings-engine           — Earnings calendar & tracking

    ─── DEEP DIVE BATCH 3: SECURITY & INFRASTRUCTURE (5) ───
    51. bw-scam-detector             — Crypto scam detection engine
    52. bw-websocket-feeds           — Real-time WebSocket price feeds
    53. bw-kelly-criterion           — Kelly Criterion position sizing
    54. bw-var-calculator            — Value at Risk calculator
    55. bw-tax-harvester             — Tax-loss harvesting automation

    ─── DEEP DIVE BATCH 3: FINANCIAL PLANNING (5) ───
    56. bw-rebalance-alerts          — Portfolio drift & rebalancing
    57. bw-market-commentary         — AI market commentary generation
    58. bw-compliance-engine         — Compliance documentation engine
    59. bw-wallet-manager            — Multi-chain wallet management
    60. bw-prediction-markets        — Prediction market intelligence

    ─── DEEP DIVE BATCH 3: ADVANCED OPS (5) ───
    61. bw-ccxt-exchange             — CCXT multi-exchange integration
    62. bw-milestone-tracker         — Financial milestone tracking
    63. bw-estate-planner            — Estate planning coordination
    64. bw-referral-network          — Professional referral network
    65. bw-insider-tracker           — SEC Form 4 insider tracking

    ─── DEEP DIVE BATCH 4: SECURITY & DEFI (15) ───
    66-80. (See v2.6.0 CHANGELOG for full list)

    ─── v2.7.0: OPTIONS DEEP DIVE (7) ───
    81. bw-gamma-exposure            — Dealer GEX, flip levels, vol surface
    82. bw-wheel-strategy            — Wheel strategy (CSP → CC cycle)
    83. bw-zero-dte                  — 0DTE gamma engine & session trading
    84. bw-vol-arb                   — Volatility arbitrage & regime trading
    85. bw-iv-crush                  — IV crush & earnings strategies
    86. bw-greeks-portfolio          — Portfolio-level Greeks & risk mgmt
    87. bw-options-strategy-engine   — 20+ strategy builder & scanner

    ─── v2.7.0: CRYPTO DEEP DIVE (6) ───
    88. bw-onchain-metrics           — MVRV, SOPR, NUPL, NVT, exchange flows
    89. bw-mev-protect               — MEV protection & Flashbots integration
    90. bw-defi-yield                — DeFi yield analysis & IL calculator
    91. bw-whale-tracker             — Whale wallet tracking & accumulation
    92. bw-funding-rates             — Funding rate & OI divergence analysis
    93. bw-liquidation-watch         — Liquidation cascades & dominance cycles

Reference: https://docs.openclaw.ai/tools/skills
Telegram: @barrenwuffet069bot
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(os.path.expanduser("~/.openclaw/workspace/skills"))

# ═══════════════════════════════════════════════════════════════════════════
# SKILL DEFINITIONS — 93 SKILLS (35 Original + 30 Batch 3 + 15 Batch 4 + 13 v2.7.0)
# ═══════════════════════════════════════════════════════════════════════════

BARREN_WUFFET_SKILLS: Dict[str, Dict] = {

# ─── 1. MARKET INTELLIGENCE ────────────────────────────────────────────────

"bw-market-intelligence": {
    "name": "bw-market-intelligence",
    "description": "Scan markets across Theater B (Attention/Narrative), Theater C (Infrastructure/Latency), and Theater D (Information Asymmetry) using AAC's BigBrainIntelligence research agents.",
    "metadata": {
        "openclaw": {
            "emoji": "📊",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
            "always": False,
        }
    },
    "instructions": """
## BARREN WUFFET Market Intelligence Scanner

Access AAC's BigBrainIntelligence research agents spanning 3 operational theaters.

### Theater B — Attention & Narrative (3 agents)
- **NarrativeAnalyzerAgent**: Tracks emerging narratives, social sentiment shifts, meme velocity
- **EngagementPredictorAgent**: Predicts engagement spikes and viral potential
- **ContentOptimizerAgent**: Identifies content-driven market opportunities

### Theater C — Infrastructure & Latency (4 agents)
- **LatencyMonitorAgent**: Monitors cross-exchange latency differentials
- **BridgeAnalyzerAgent**: Analyzes cross-chain bridge volumes and delays
- **GasOptimizerAgent**: Tracks gas price patterns and optimization windows
- **LiquidityTrackerAgent**: Maps liquidity depth across venues

### Theater D — Information Asymmetry (4 agents)
- **APIScannerAgent**: Scans for new API endpoints and data source changes
- **DataGapFinderAgent**: Identifies information gaps between markets
- **AccessArbitrageAgent**: Finds data access differentials for alpha
- **NetworkMapperAgent**: Maps participant networks and flow patterns

### Commands
```
/bw-intel theater=B focus=crypto_narratives
/bw-intel theater=all summary=true
/bw-intel theater=C focus=gas_optimization timeframe=1h
/bw-intel theater=D focus=data_gaps
```

### Output
Returns consolidated `ResearchFinding` reports with:
- Signal Strength (0.0-1.0 quantum-weighted confidence)
- Theater source, finding type, actionable flag
- Strategy IDs mapping to the 50 arbitrage strategies
""",
},

# ─── 2. TRADING SIGNALS ───────────────────────────────────────────────────

"bw-trading-signals": {
    "name": "bw-trading-signals",
    "description": "Quantum-aggregated trading signals from 49 agents executing 50 arbitrage strategies. Covers statistical, structural, technology, compliance, and cross-chain arbitrage.",
    "metadata": {
        "openclaw": {
            "emoji": "⚡",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Trading Signals

Access quantum-aggregated trading signals from 49 trading agents executing
across 50 arbitrage strategies.

### Signal Pipeline
```
Research Agents → ResearchFinding → QuantumSignal → QuantumSignalAggregator → Consensus → You
```

### Commands
```
/bw-signals active              — Active signals with confidence > 0.7
/bw-signals top=5               — Top 5 signals by quantum-weighted score
/bw-signals strategy=DEX        — DEX arbitrage signals only
/bw-signals strategy=STAT_PAIRS — Statistical pairs trading signals
/bw-signals history=24h         — Signal history over last 24 hours
/bw-signals performance         — Win rate, avg return, Sharpe by strategy
```

### Signal Categories
- **Statistical Arbitrage**: Mean reversion, pairs trading, cointegration
- **Structural Arbitrage**: Market microstructure, order book imbalances
- **Technology Arbitrage**: Latency, API, cross-venue timing advantages
- **Compliance Arbitrage**: Regulatory gap exploitation (doctrine-compliant)
- **Cross-Chain Arbitrage**: DeFi bridge, DEX spread, yield farming

### Risk Context
Every signal includes:
- Doctrine compliance (BarrenWuffetState: NORMAL/CAUTION/SAFE_MODE/HALT)
- Current drawdown and daily P&L impact
- Position sizing via Kelly criterion
- Maximum exposure recommendation
""",
},

# ─── 3. PORTFOLIO DASHBOARD ───────────────────────────────────────────────

"bw-portfolio-dashboard": {
    "name": "bw-portfolio-dashboard",
    "description": "Real-time BARREN WUFFET portfolio dashboard with P&L, positions, strategy attribution, risk metrics, and contest leaderboard.",
    "metadata": {
        "openclaw": {
            "emoji": "📈",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Portfolio Dashboard

Generate a rich portfolio dashboard from CentralAccounting.

### Commands
```
/bw-dash full          — Complete dashboard with all sections
/bw-dash summary       — Quick: NAV, daily P&L, top positions
/bw-dash strategies    — P&L attribution by strategy
/bw-dash risk          — VaR, drawdown, exposure by venue
/bw-dash contest       — $1K→$10K Trading Contest leaderboard
/bw-dash crypto        — Crypto-only portfolio view
/bw-dash options       — Options positions and Greeks
```

### Dashboard Layout
```
╔══════════════════════════════════════════════════╗
║     BARREN WUFFET — PORTFOLIO DASHBOARD         ║
╠══════════════════════════════════════════════════╣
║ NAV:              $XXX,XXX.XX                    ║
║ Daily P&L:        +$X,XXX.XX (+X.XX%)            ║
║ Drawdown:         -X.XX% (max: -X.XX%)           ║
║ Doctrine State:   NORMAL ✅                       ║
║ Active Strats:    XX/50                           ║
║ Open Positions:   XX                              ║
╠══════════════════════════════════════════════════╣
║ TOP STRATEGIES (24h)                              ║
║ 1. DEX Arbitrage        +$XXX  (XX trades)        ║
║ 2. Stat Pairs ETH       +$XXX  (XX trades)        ║
║ 3. Cross-Chain Bridge    +$XXX  (XX trades)        ║
╠══════════════════════════════════════════════════╣
║ RISK METRICS                                      ║
║ 1d VaR (95%):     $X,XXX                          ║
║ Sharpe (30d):     X.XX                             ║
║ Max Exposure:     $XXX,XXX                         ║
╚══════════════════════════════════════════════════╝
```
""",
},

# ─── 4. RISK MONITOR ──────────────────────────────────────────────────────

"bw-risk-monitor": {
    "name": "bw-risk-monitor",
    "description": "Real-time risk monitoring with BarrenWuffetState doctrine integration. Proactive alerts on drawdown, margin, and compliance breaches via Telegram.",
    "metadata": {
        "openclaw": {
            "emoji": "🛡️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
            "always": True,
        }
    },
    "instructions": """
## BARREN WUFFET Risk Monitor

Real-time risk monitoring with BarrenWuffetState doctrine integration.
Proactive alerts delivered via Telegram @barrenwuffet069bot.

### Doctrine State Machine
```
NORMAL ──(drawdown > 5%)──→ CAUTION ──(drawdown > 10%)──→ SAFE_MODE ──(daily loss > 2%)──→ HALT
   ↑                                                                                        │
   └────────────────────────── Manual Override / Recovery ──────────────────────────────────┘
```

### Commands
```
/bw-risk status              — Doctrine state + key risk metrics
/bw-risk doctrine            — All 8 doctrine pack statuses
/bw-risk exposure            — Position exposure by venue/asset/strategy
/bw-risk alerts              — Active risk alerts and breaches
/bw-risk circuit-breakers    — Circuit breaker status
/bw-risk var                 — Value at Risk analysis
/bw-risk stress-test         — Run stress test scenarios
```

### 8 Doctrine Packs
1. Capital Preservation  2. Position Sizing  3. Execution Quality
4. Market Risk  5. Counterparty Risk  6. Operational Risk
7. Compliance  8. Performance Attribution

### Proactive Telegram Alerts
This skill sends alerts via @barrenwuffet069bot when:
- BarrenWuffetState transitions (NORMAL → CAUTION)
- Drawdown exceeds thresholds (5%, 10%, 15%)
- Circuit breakers trigger
- Daily loss limits approach
- Counterparty exposure spikes
""",
},

# ─── 5. CRYPTO INTELLIGENCE ───────────────────────────────────────────────

"bw-crypto-intel": {
    "name": "bw-crypto-intel",
    "description": "AAC CryptoIntelligence engine for on-chain analysis, DeFi opportunities, whale tracking, mempool intelligence, and cross-chain arbitrage.",
    "metadata": {
        "openclaw": {
            "emoji": "🔗",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET CryptoIntelligence Engine

Deep crypto market intelligence from AAC's CryptoIntelligence department.

### Capabilities
- **On-Chain Analysis**: Transaction flow, whale movements, smart money tracking
- **DeFi Intelligence**: Yield farming, LP analysis, protocol TVL shifts
- **Cross-Chain Arbitrage**: Bridge volumes, cross-chain price differentials
- **Mempool Intelligence**: Pending transactions, MEV opportunity detection
- **Exchange Intelligence**: Order books, funding rates, open interest

### Commands
```
/bw-crypto overview              — Market-wide intelligence summary
/bw-crypto whale-watch           — Large transaction monitoring
/bw-crypto defi-ops              — Top DeFi yield & arb opportunities
/bw-crypto gas-analysis          — Gas price patterns & optimization
/bw-crypto bridge-flows          — Cross-chain bridge flows
/bw-crypto sentiment             — Crypto social sentiment analysis
/bw-crypto mev                   — MEV opportunity scan
/bw-crypto funding-rates         — Exchange funding rate comparison
```

### Pipeline
```
CryptoIntelligence → IntelligenceSignal → CryptoBigBrainBridge → QuantumSignalAggregator
```
""",
},

# ─── 6. AZ SUPREME COMMAND ────────────────────────────────────────────────

"bw-az-supreme-command": {
    "name": "bw-az-supreme-command",
    "description": "Direct executive command interface to AZ SUPREME — BARREN WUFFET's supreme executive AI governing the entire AAC ecosystem via Telegram.",
    "metadata": {
        "openclaw": {
            "emoji": "👑",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
            "always": True,
        }
    },
    "instructions": """
## AZ SUPREME — BARREN WUFFET Executive Command

Interface with AZ SUPREME, the supreme executive command agent of AAC.
All commands route through @barrenwuffet069bot on Telegram.

### Executive Capabilities
- Strategic oversight across all 6 departments
- Crisis management with real-time detection
- Cross-domain coordination bridging all agents
- Priority-weighted directive execution
- Performance monitoring of 80+ agents and 50 strategies

### Commands
```
/az status                  — Full system status report
/az directive HIGH <text>   — Issue executive directive
/az briefing                — Strategic briefing
/az crisis-mode on|off      — Toggle crisis management
/az departments             — Department status overview
/az question <text>         — Strategic question (100 question library)
/az agents list             — List all registered agents
/az override <state>        — Override doctrine state (CRITICAL auth)
```

### Directive Priorities
- **CRITICAL**: Immediate execution across all departments
- **HIGH**: Execute within current cycle
- **MEDIUM**: Queue for next operational cycle
- **LOW**: Backlog for strategic planning
""",
},

# ─── 7. DOCTRINE STATUS ───────────────────────────────────────────────────

"bw-doctrine-status": {
    "name": "bw-doctrine-status",
    "description": "Query and monitor the BarrenWuffetState Doctrine Engine — the safety-critical state machine governing all AAC trading operations.",
    "metadata": {
        "openclaw": {
            "emoji": "📜",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BarrenWuffetState Doctrine Engine

The Doctrine Engine enforces operational safety through 8 compliance packs
and a 4-state safety machine.

### State Machine
```
NORMAL ──(drawdown>5%)──→ CAUTION ──(drawdown>10%)──→ SAFE_MODE ──(loss>2%)──→ HALT
   ↑                                                                            │
   └──────────────────── Manual Override / Recovery ───────────────────────────┘
```

### 8 Doctrine Packs
1. **Capital Preservation**: Max drawdown, daily loss limits, position caps
2. **Position Sizing**: Kelly criterion, max position %, correlation limits
3. **Execution Quality**: Slippage tolerance, fill rate minimums, timing
4. **Market Risk**: VaR limits, volatility regimes, liquidity requirements
5. **Counterparty Risk**: Exchange exposure limits, settlement risk caps
6. **Operational Risk**: System health, latency SLAs, error rate caps
7. **Compliance**: Regulatory adherence, reporting, jurisdiction limits
8. **Performance Attribution**: P&L accuracy, benchmark tracking

### Commands
```
/bw-doctrine status         — Current state + all pack statuses
/bw-doctrine pack 1         — Detailed Capital Preservation status
/bw-doctrine history        — State transition history
/bw-doctrine override HALT  — Request state override (CRITICAL auth)
/bw-doctrine packs all      — All 8 packs detailed view
```
""",
},

# ─── 8. MORNING BRIEFING ──────────────────────────────────────────────────

"bw-morning-briefing": {
    "name": "bw-morning-briefing",
    "description": "Automated daily morning briefing via @barrenwuffet069bot covering overnight performance, markets, signals, risk, and strategic opportunities.",
    "metadata": {
        "openclaw": {
            "emoji": "☀️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY", "TELEGRAM_BOT_TOKEN"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Morning Briefing

AZ SUPREME delivers a daily briefing via @barrenwuffet069bot.

### Briefing Contents
1. **Overnight Performance**: P&L summary, strategy attribution, notable trades
2. **Market Conditions**: Key moves, volatility regime, macro events
3. **Crypto Markets**: BTC, ETH, XRP, meme coins, DeFi TVL changes
4. **Active Signals**: Top research findings from all 3 theaters
5. **Risk Status**: BarrenWuffetState, drawdown, exposure
6. **Options Activity**: Unusual options flow, key expirations
7. **2007 Crash Indicators**: Pattern matching against historical signals
8. **Golden Ratio Analysis**: Dan Winter harmonic confluence zones
9. **SuperStonk DD Feed**: Latest due diligence from Reddit research
10. **Strategic Opportunities**: AI-recommended actions for the day
11. **Agent Performance**: Top/bottom agents, anomalies
12. **Regulatory Updates**: Calgary & Montevideo jurisdiction alerts

### Auto-Schedule (OpenClaw Cron)
```
Schedule: 0 7 * * 1-5  (7 AM Mon-Fri, Mountain Time - Calgary)
Session: barren-wuffet-main
Channel: Telegram @barrenwuffet069bot
```

### Commands
```
/bw-briefing now              — Generate immediately
/bw-briefing schedule         — Show/update schedule
/bw-briefing sections         — Select which sections to include
/bw-briefing weekend          — Enable weekend briefings
```
""",
},

# ─── 9. AGENT ROSTER ──────────────────────────────────────────────────────

"bw-agent-roster": {
    "name": "bw-agent-roster",
    "description": "Browse and query the full roster of 80+ AAC agents across executive, research, trading, infrastructure, and division departments.",
    "metadata": {
        "openclaw": {
            "emoji": "👥",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Agent Roster

Browse all 80+ agents in the AAC ecosystem.

### Department Map
| Department | Agents | Lead |
|---|---|---|
| Executive Branch | 2 | AZ SUPREME |
| BigBrainIntelligence | 20+ | SuperBigBrainAgent |
| TradingExecution | 49 | SuperTradeExecutorAgent |
| CryptoIntelligence | 5+ | SuperCryptoIntelAgent |
| CentralAccounting | 3+ | SuperAccountingAgent |
| SharedInfrastructure | 5+ | SuperInfrastructureAgent |
| NCC (Network Command) | 3+ | SuperNCCCommandAgent |
| Jonny Bravo Division | 5+ | JonnyBravoDivisionLead |

### Commands
```
/bw-agents list                    — Full agent roster
/bw-agents department=BigBrain     — Filter by department
/bw-agents search=<query>          — Search by name/capability
/bw-agents status=<agent_id>       — Specific agent status
/bw-agents performance             — Performance leaderboard
/bw-agents super-agents            — List quantum-enhanced super agents
```
""",
},

# ─── 10. STRATEGY EXPLORER ────────────────────────────────────────────────

"bw-strategy-explorer": {
    "name": "bw-strategy-explorer",
    "description": "Explore, analyze, and compare AAC's 50 arbitrage strategies with performance metrics, parameters, and agent assignments.",
    "metadata": {
        "openclaw": {
            "emoji": "🎯",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Strategy Explorer

Explore the 50 arbitrage strategies from `50_arbitrage_strategies.csv`.

### Categories
- **Statistical Arbitrage**: Mean reversion, pairs, cointegration, factor models
- **Structural Arbitrage**: Order book, fragmentation, microstructure
- **Technology Arbitrage**: Latency, API, data pipeline advantages
- **Compliance Arbitrage**: Cross-jurisdiction regulatory gaps
- **Cross-Chain/DeFi**: DEX arb, bridge arb, yield farming, MEV

### Commands
```
/bw-strat list                 — All 50 strategies
/bw-strat category=STAT        — Statistical arbitrage strategies
/bw-strat detail=12            — Deep dive on strategy #12
/bw-strat performance          — Performance comparison table
/bw-strat correlations         — Strategy correlation matrix
/bw-strat backtest=12 days=90  — Backtest strategy over 90 days
```
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# TRADING & MARKETS (Skills 11-17)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 11. DIGITAL ARBITRAGE ─────────────────────────────────────────────────

"bw-digital-arbitrage": {
    "name": "bw-digital-arbitrage",
    "description": "Cross-exchange digital arbitrage engine. Detects price differentials across CEXs, DEXs, and cross-chain bridges in real-time.",
    "metadata": {
        "openclaw": {
            "emoji": "🔄",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Digital Arbitrage Engine

Real-time cross-exchange digital arbitrage detection across centralized
exchanges (CEX), decentralized exchanges (DEX), and cross-chain bridges.

### Arbitrage Types Monitored
- **CEX-CEX Spread**: Price differentials between Binance, Coinbase, Kraken, etc.
- **DEX-DEX Spread**: Uniswap vs SushiSwap vs Curve vs PancakeSwap
- **CEX-DEX Spread**: Centralized vs decentralized pricing gaps
- **Cross-Chain Bridge**: Arbitrum ↔ Optimism ↔ Base ↔ Polygon price differences
- **Triangular Arbitrage**: BTC→ETH→USDT→BTC circular routes
- **Funding Rate Arbitrage**: Long spot + short perpetual when funding is high
- **Stablecoin Depegs**: USDT/USDC/DAI temporary depeg opportunities

### Commands
```
/bw-arb scan                   — Full arbitrage scan across all venues
/bw-arb cex-cex pairs=BTC,ETH  — CEX-CEX spread for specific pairs
/bw-arb dex-dex chain=ethereum  — DEX arbitrage on Ethereum
/bw-arb cross-chain             — Cross-chain bridge opportunities
/bw-arb triangular              — Triangular arbitrage routes
/bw-arb funding                 — Funding rate arbitrage scanner
/bw-arb history=24h             — Recent profitable arbitrage events
```

### Execution Pipeline
```
Price Feed → Spread Detection → Profitability Filter → Gas/Fee Calculator → Signal
```

### Risk Management
- All arbitrage signals include estimated gas costs, slippage, and net profit
- Minimum spread thresholds configurable per venue pair
- Bridge timeout risk factored into cross-chain calculations
- Doctrine compliance checked before any execution signal
""",
},

# ─── 12. ARBITRAGE SCANNER ────────────────────────────────────────────────

"bw-arbitrage-scanner": {
    "name": "bw-arbitrage-scanner",
    "description": "Multi-venue arbitrage detection covering statistical, structural, regulatory, and technology arbitrage across traditional and crypto markets.",
    "metadata": {
        "openclaw": {
            "emoji": "🔍",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Arbitrage Scanner

Multi-venue, multi-asset arbitrage detection engine.

### Arbitrage Categories from 50_arbitrage_strategies.csv
1. **Statistical Arbitrage** (Strategies 1-12)
   - Pairs trading, mean reversion, cointegration
   - Z-score monitoring, Bollinger band signals
   - Cross-correlation matrix analysis

2. **Structural Arbitrage** (Strategies 13-22)
   - Market microstructure exploitation
   - Order book depth imbalances
   - Fragmentation across venues

3. **Technology Arbitrage** (Strategies 23-30)
   - Latency advantages between venues
   - API speed differentials
   - Data pipeline timing edges

4. **Compliance Arbitrage** (Strategies 31-38)
   - Cross-jurisdiction regulatory gaps
   - Reporting arbitrage windows
   - License differential advantages
   - Calgary/Montevideo regulatory asymmetries

5. **Cross-Chain/DeFi Arbitrage** (Strategies 39-50)
   - DEX liquidity pool arbitrage
   - Bridge timing advantages
   - Yield farming optimization
   - Flash loan routes

### Commands
```
/bw-scan all                   — Full multi-asset arbitrage scan
/bw-scan category=STAT         — Statistical arbitrage only
/bw-scan category=DEFI         — DeFi/cross-chain only
/bw-scan asset=BTC             — Bitcoin-specific arbitrage opportunities
/bw-scan jurisdiction=Calgary  — Calgary-relevant compliance arb
/bw-scan profitability=high    — Only high-profit opportunities
```
""",
},

# ─── 13. DAY TRADING ──────────────────────────────────────────────────────

"bw-day-trading": {
    "name": "bw-day-trading",
    "description": "Intraday momentum, scalping, and day trading intelligence. VWAP, tape reading, level 2 analysis, and intraday pattern recognition.",
    "metadata": {
        "openclaw": {
            "emoji": "📉",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Day Trading Intelligence

Intraday trading analysis, momentum detection, and scalping signals.

### Analysis Types
- **VWAP Analysis**: Volume-weighted average price crossovers & deviations
- **Tape Reading**: Real-time order flow analysis, large block detection
- **Level 2 Depth**: Order book imbalance and liquidity wall detection
- **Momentum Scanning**: RSI divergence, MACD crossovers, ADX trend strength
- **Pattern Recognition**: Bull/bear flags, head & shoulders, wedges, gaps
- **Scalping Signals**: Micro-structure patterns for quick in/out trades
- **Pre-Market/After-Hours**: Gap analysis and overnight flow assessment

### Intraday Timeframes
- 1-minute (scalping)
- 5-minute (momentum)
- 15-minute (swing intraday)
- 1-hour (intraday trend)

### Commands
```
/bw-day scan                    — Intraday opportunity scanner
/bw-day momentum top=10         — Top 10 momentum stocks/coins
/bw-day vwap asset=BTC          — VWAP analysis for Bitcoin
/bw-day tape asset=ETH          — Real-time tape reading ETH
/bw-day patterns timeframe=5m   — Pattern scan on 5-min charts
/bw-day gaps                    — Pre-market gap analysis
/bw-day scalp                   — Active scalping signals
```

### Risk Rules
- Max 3 simultaneous day trades without Pattern Day Trader flag
- Stop-loss required on all day trade signals (auto-calculated)
- Position sizing respects BarrenWuffetState doctrine limits
- Intraday P&L tracked in real-time via CentralAccounting
""",
},

# ─── 14. OPTIONS TRADING ──────────────────────────────────────────────────

"bw-options-trading": {
    "name": "bw-options-trading",
    "description": "Options strategies, Greeks analysis, chain scanning, IV rank/percentile, and multi-leg strategy construction including iron condors, butterflies, straddles, and spreads.",
    "metadata": {
        "openclaw": {
            "emoji": "🎲",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Options Trading Intelligence

Full-spectrum options analysis, strategy construction, and risk management.

### Options Strategies Library
**Bullish**:
- Long Call, Bull Call Spread, Bull Put Spread
- Diagonal Spread, LEAPS, Synthetic Long

**Bearish**:
- Long Put, Bear Put Spread, Bear Call Spread
- Synthetic Short

**Neutral/Income**:
- Covered Call, Cash-Secured Put, Iron Condor
- Iron Butterfly, Jade Lizard, Strangle (short)

**Volatility**:
- Long Straddle, Long Strangle, Calendar Spread
- Ratio Spread, Backspread

**Advanced**:
- Collar, Risk Reversal, Box Spread
- Broken Wing Butterfly, Christmas Tree Spread
- Wheel Strategy (systematic covered calls + CSPs)

### Greeks Analysis
- **Delta**: Directional exposure per $1 move
- **Gamma**: Rate of delta change (acceleration risk)
- **Theta**: Time decay rate (daily bleed)
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

### Commands
```
/bw-options chain SPY             — Full options chain for SPY
/bw-options greeks SPY 450C 0321  — Greeks for specific contract
/bw-options iv-rank SPY           — IV rank and IV percentile
/bw-options strategy iron-condor SPY — Build iron condor on SPY
/bw-options unusual-flow          — Unusual options activity scanner
/bw-options max-pain SPY          — Max pain calculation
/bw-options earnings NVDA         — Pre-earnings options strategy
/bw-options wheel SPY             — Wheel strategy setup
/bw-options spreads BTC           — Crypto options spreads
```

### Integration with IBKR
Connects to Interactive Brokers for real-time options data when
IBKR_API_KEY is configured. Paper trading supported.
""",
},

# ─── 15. CALLS & PUTS FLOW ────────────────────────────────────────────────

"bw-calls-puts-flow": {
    "name": "bw-calls-puts-flow",
    "description": "Real-time options flow tracking: unusual calls, puts, sweeps, block trades, and dark pool prints. Smart money detection via put/call ratio and flow analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "📞",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Options Flow Intelligence

Track smart money through options flow analysis.

### Flow Types Monitored
- **Sweeps**: Aggressive fills across multiple exchanges (bullish/bearish urgency)
- **Block Trades**: Large single-exchange prints (institutional positioning)
- **Unusual Activity**: Volume exceeding open interest (new money entering)
- **Dark Pool Prints**: Off-exchange large block notifications
- **Premium Analysis**: Total premium spent by calls vs puts by sector

### Put/Call Indicators
- **PCR (Put/Call Ratio)**: Market-wide sentiment gauge
  - PCR < 0.7 → Bullish sentiment (excessive call buying)
  - PCR 0.7-1.0 → Neutral
  - PCR > 1.0 → Bearish / Hedging activity
- **Equity PCR vs Index PCR**: Divergence signals institutional hedging
- **Sector Flow**: Where is the smart money rotating?

### Commands
```
/bw-flow live                     — Real-time options flow feed
/bw-flow sweeps                   — Sweep orders only (urgency trades)
/bw-flow blocks                   — Block trades > $100K premium
/bw-flow unusual                  — Volume > 3x open interest
/bw-flow pcr                      — Current put/call ratios
/bw-flow sector=tech              — Tech sector flow analysis
/bw-flow ticker=NVDA              — NVDA-specific flow
/bw-flow dark-pool                — Dark pool activity
/bw-flow whale-alerts             — $500K+ premium orders
```

### Smart Money Signals
When institutional flow diverges from price action, generate contrarian
signals. Example: Heavy put buying while stock is at ATH → potential
smart money hedging before news.
""",
},

# ─── 16. HEDGING STRATEGIES ───────────────────────────────────────────────

"bw-hedging-strategies": {
    "name": "bw-hedging-strategies",
    "description": "Portfolio hedging and risk management strategies including protective puts, collar strategies, tail risk hedging, and correlation-based hedging across crypto and traditional assets.",
    "metadata": {
        "openclaw": {
            "emoji": "🏰",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Hedging Strategies

Portfolio protection and risk management.

### Hedging Methods
- **Protective Puts**: Downside protection on long positions
- **Collar Strategy**: Capped upside for funded downside protection
- **Tail Risk Hedging**: Far OTM puts for black swan events
- **Delta Hedging**: Dynamic rebalancing to neutralize directional risk
- **Correlation Hedging**: Inverse-correlated asset pairs
- **VIX Hedging**: Volatility index instruments for portfolio insurance
- **Crypto-Specific Hedges**: Short perpetuals, put options, stablecoin rotation
- **Cross-Asset Hedging**: Gold/bonds vs equity exposure

### 2007-Style Crisis Hedging
Pre-positioned hedge structures based on 2007/2008 crisis indicators:
- Credit spread widening triggers
- Yield curve inversion response
- Liquidity contraction protocols
- Systematic de-risking waterfall

### Commands
```
/bw-hedge portfolio               — Recommended hedges for current portfolio
/bw-hedge protective-put BTC      — Protective put structure for BTC
/bw-hedge collar SPY              — Collar strategy for SPY position
/bw-hedge tail-risk               — Tail risk hedge recommendations
/bw-hedge cost                    — Total hedge cost vs coverage analysis
/bw-hedge crisis                  — Activate 2007-style crisis hedging
/bw-hedge vix                     — VIX-based portfolio insurance
```

### Integration with Doctrine
Hedging recommendations escalate based on BarrenWuffetState:
- NORMAL: Optional hedging, 10% portfolio insurance
- CAUTION: Recommended, 25% coverage
- SAFE_MODE: Mandatory, 50% coverage, active delta hedging
- HALT: Full hedge, all positions protected or closed
""",
},

# ─── 17. CURRENCY TRADING ─────────────────────────────────────────────────

"bw-currency-trading": {
    "name": "bw-currency-trading",
    "description": "Forex and currency pair analysis covering major, minor, and exotic pairs. CAD focus for Calgary operations. Cross-currency arbitrage and carry trades.",
    "metadata": {
        "openclaw": {
            "emoji": "💱",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Currency Trading

Forex analysis with focus on Calgary/CAD and Montevideo/UYU operations.

### Currency Pairs Monitored
**Majors**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
**Minors**: EUR/GBP, EUR/AUD, GBP/JPY, CAD/JPY, AUD/CAD
**Exotics**: USD/UYU (Montevideo), USD/BRL, EUR/TRY
**Crypto-Fiat**: BTC/CAD, ETH/CAD, XRP/USD, BTC/UYU

### Analysis Types
- **Macro Fundamentals**: Interest rate differentials, GDP, employment data
- **Technical Analysis**: Fibonacci levels, pivot points, moving averages
- **Carry Trade Scanner**: High-yield vs low-yield currency pairs
- **Cross-Currency Arbitrage**: Triangular forex arbitrage opportunities
- **News Impact Analysis**: Central bank statements, NFP, CPI impact
- **Seasonal Patterns**: Month-end flows, quarter-end rebalancing

### Commands
```
/bw-fx overview                  — Major pairs overview
/bw-fx pair=USD/CAD             — Detailed USD/CAD analysis
/bw-fx carry-trades             — Active carry trade opportunities
/bw-fx arbitrage                — Cross-currency triangular arb scan
/bw-fx calendar                 — Economic calendar & event impact
/bw-fx cad                      — CAD-centric analysis (Calgary ops)
/bw-fx uyu                      — UYU analysis (Montevideo ops)
```

### Calgary-Montevideo Corridor
Special monitoring for CAD/UYU conversion efficiency and regulatory
arbitrage opportunities between Canadian and Uruguayan financial systems.
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO & DEFI (Skills 18-24)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 18. BITCOIN INTELLIGENCE ─────────────────────────────────────────────

"bw-bitcoin-intel": {
    "name": "bw-bitcoin-intel",
    "description": "Bitcoin-specific on-chain analytics, halving cycle analysis, miner economics, hash rate monitoring, whale tracking, and macro correlation analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "₿",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Bitcoin Intelligence

Deep Bitcoin analysis: on-chain, macro, miner economics, and cycle positioning.

### On-Chain Metrics
- **HODL Waves**: Distribution of UTXOs by age (accumulation vs distribution)
- **MVRV Ratio**: Market Value to Realized Value (overbought/oversold)
- **SOPR**: Spent Output Profit Ratio (profit-taking signals)
- **NVT Signal**: Network Value to Transactions (valuation metric)
- **Hash Rate**: Network security and miner commitment
- **Miner Revenue**: Block rewards + fees, miner capitulation signals
- **Exchange Flows**: Net exchange inflows/outflows (selling vs accumulating)
- **Whale Wallets**: Tracking 1K+ BTC addresses

### Halving Cycle Analysis
- Current halving epoch position
- Historical post-halving price patterns
- Supply shock modeling
- Stock-to-Flow comparison

### Macro Correlation
- BTC vs Gold correlation rolling 30d
- BTC vs S&P 500 / NASDAQ correlation
- BTC vs DXY (Dollar Index) inverse correlation
- BTC as inflation hedge analysis

### Commands
```
/bw-btc overview                 — Full Bitcoin intelligence report
/bw-btc onchain                  — On-chain metrics dashboard
/bw-btc halving                  — Halving cycle position analysis
/bw-btc whales                   — Large wallet movement tracking
/bw-btc miners                   — Miner economics and hash rate
/bw-btc macro                    — Macro correlation analysis
/bw-btc sentiment                — Bitcoin-specific sentiment index
/bw-btc price-model              — Stock-to-Flow & power law models
```
""",
},

# ─── 19. ETHEREUM & DEFI ──────────────────────────────────────────────────

"bw-ethereum-defi": {
    "name": "bw-ethereum-defi",
    "description": "Ethereum ecosystem intelligence: DeFi protocol analysis, gas optimization, L2 comparisons, staking yields, MEV tracking, and smart contract monitoring.",
    "metadata": {
        "openclaw": {
            "emoji": "⟠",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Ethereum & DeFi Intelligence

Ethereum ecosystem analysis, DeFi protocol monitoring, and L2 tracking.

### DeFi Protocol Tracking
- **Lending**: Aave, Compound, MakerDAO — rates, utilization, liquidations
- **DEXs**: Uniswap, SushiSwap, Curve — volume, TVL, LP yields
- **Derivatives**: GMX, dYdX, Synthetix — OI, funding rates, volume
- **Liquid Staking**: Lido, Rocket Pool, cbETH — yields, market share
- **Yield Aggregators**: Yearn, Convex, Beefy — vault APYs
- **Bridges**: Across, Stargate, Hop — volume, fees, speed

### Gas Intelligence
- Real-time gas prices (slow/standard/fast/instant)
- Gas price prediction (next 1h, 4h, 24h)
- Optimal transaction timing recommendations
- EIP-1559 base fee trends and burn rate

### L2 Comparison
| L2 | TPS | Gas Cost | TVL | Bridge Time |
|---|---|---|---|---|
| Arbitrum | ... | ... | ... | ... |
| Optimism | ... | ... | ... | ... |
| Base | ... | ... | ... | ... |
| zkSync | ... | ... | ... | ... |

### Commands
```
/bw-eth overview                 — Ethereum ecosystem overview
/bw-eth defi-yields              — Top DeFi yields across protocols
/bw-eth gas                      — Gas price analysis and prediction
/bw-eth l2-compare               — L2 comparison dashboard
/bw-eth staking                  — Staking yields and opportunities
/bw-eth mev                      — MEV activity monitoring
/bw-eth liquidations             — DeFi liquidation tracking
/bw-eth smart-money              — Smart contract whale tracking
```
""",
},

# ─── 20. XRP / RIPPLE ─────────────────────────────────────────────────────

"bw-xrp-ripple": {
    "name": "bw-xrp-ripple",
    "description": "XRP and Ripple cross-border payment intelligence. XRPL monitoring, ODL (On-Demand Liquidity) flows, regulatory status, and institutional adoption tracking.",
    "metadata": {
        "openclaw": {
            "emoji": "💧",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET XRP & Ripple Intelligence

XRP/Ripple ecosystem monitoring focused on cross-border payments and ODL.

### Monitoring Areas
- **XRPL Network**: Transaction volume, fees, new accounts, escrow releases
- **ODL (On-Demand Liquidity)**: Cross-border payment corridor volumes
- **Institutional Adoption**: Bank partnerships, CBDC pilots, regulatory approvals
- **SEC Case Developments**: Ongoing regulatory clarity tracking
- **Price Analysis**: Technical analysis, support/resistance, whale accumulation
- **Competitive Landscape**: SWIFT gpi, Stellar (XLM), CBDC progress

### ODL Corridors Monitored
- USD → MXN (US-Mexico corridor)
- USD → PHP (US-Philippines corridor)
- EUR → USD (Europe-US corridor)
- AUD → USD (Australia-US corridor)
- JPY → USD (Japan-US corridor)
- Custom Calgary (CAD) corridors

### Commands
```
/bw-xrp overview                 — Full XRP intelligence report
/bw-xrp odl                      — ODL corridor volumes and growth
/bw-xrp escrow                   — Monthly escrow release tracking
/bw-xrp regulatory               — SEC/regulatory status updates
/bw-xrp whales                   — Top XRP wallet movements
/bw-xrp technical                — Price analysis and levels
/bw-xrp partnerships             — Institutional partnership tracker
```

### Calgary Connection
XRP/CAD pair monitoring for Calgary-based operations.
Ripple ODL corridor analysis for CAD-denominated cross-border payments.
""",
},

# ─── 21. STABLECOINS ──────────────────────────────────────────────────────

"bw-stablecoins": {
    "name": "bw-stablecoins",
    "description": "Stablecoin yield optimization, peg monitoring, arbitrage opportunities, and risk analysis across USDT, USDC, DAI, FRAX, and algorithmic stablecoins.",
    "metadata": {
        "openclaw": {
            "emoji": "💵",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Stablecoin Intelligence

Stablecoin yield, peg stability, and arbitrage monitoring.

### Stablecoins Tracked
| Stablecoin | Type | Backing | Market Cap |
|---|---|---|---|
| USDT (Tether) | Centralized | Reserves | $XXB |
| USDC (Circle) | Centralized | US Treasuries | $XXB |
| DAI (MakerDAO) | Decentralized | Over-collateralized | $XXB |
| FRAX | Hybrid | Partial algorithmic | $XXB |
| PYUSD (PayPal) | Centralized | USD deposits | $XXB |
| crvUSD (Curve) | Decentralized | LLAMMA | $XXB |

### Analysis Areas
- **Peg Monitoring**: Real-time deviation from $1.00 peg (alert on >0.5%)
- **Yield Optimization**: Best yields across lending, LP, and vault strategies
- **Depeg Arbitrage**: Profitable when stablecoins temporarily lose peg
- **Supply/Demand Dynamics**: Mint/burn analysis, market cap shifts
- **Regulatory Risk**: Jurisdiction-specific stablecoin regulations
- **Backing Transparency**: Reserve audit tracking, proof-of-reserves

### Commands
```
/bw-stable overview               — Stablecoin market overview
/bw-stable pegs                   — Real-time peg monitoring
/bw-stable yields                 — Best stablecoin yields ranked
/bw-stable arbitrage              — Depeg arbitrage opportunities
/bw-stable risk                   — Stablecoin risk assessment
/bw-stable flows                  — Mint/burn and transfer flows
/bw-stable regulatory             — Regulatory status by jurisdiction
```
""",
},

# ─── 22. MEME COINS ───────────────────────────────────────────────────────

"bw-meme-coins": {
    "name": "bw-meme-coins",
    "description": "Meme coin radar with social velocity tracking, rug pull detection, liquidity analysis, and early entry signals. Covers Solana, Base, and EVM meme launches.",
    "metadata": {
        "openclaw": {
            "emoji": "🐸",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Meme Coin Radar

Social velocity tracking, rug pull detection, and early entry analysis
for meme coins across Solana, Base, Ethereum, and BSC.

### Detection Pipeline
```
Social Scan → Token Discovery → Safety Check → Liquidity Analysis → Signal
```

### Safety Analysis (Rug Pull Detection)
- ✅ Liquidity locked? Duration?
- ✅ Contract renounced?
- ✅ Honeypot check (can you sell?)
- ✅ Top 10 wallet concentration
- ✅ Developer wallet activity
- ✅ Contract source code verification
- ✅ Similar token name/symbol scam check

### Social Velocity Metrics
- Reddit mention velocity (posts/hour, comment growth)
- Twitter/X engagement rate (likes, retweets, quote tweets)
- Telegram/Discord member growth rate
- Google Trends spike detection
- Influencer mention tracking

### Commands
```
/bw-meme trending                 — Top trending meme coins (24h)
/bw-meme new-launches             — New token launches (last 4h)
/bw-meme safety-check <token>     — Rug pull risk analysis
/bw-meme social <token>           — Social velocity metrics
/bw-meme solana                   — Solana meme coin radar
/bw-meme base                     — Base chain meme coin radar
/bw-meme whale-buys               — Large meme coin purchases
/bw-meme graduated                — Pump.fun graduated tokens
```

### Risk Warning
Meme coins are extremely high risk. All signals include:
- Risk score (1-10, where 10 = maximum risk)
- Liquidity depth and lock status
- Position sizing recommendation (micro-positions only)
- BarrenWuffetState doctrine compliance check
""",
},

# ─── 23. LIBERTY COIN ─────────────────────────────────────────────────────

"bw-liberty-coin": {
    "name": "bw-liberty-coin",
    "description": "Liberty Coin tracking, analysis, and integration. Price monitoring, community metrics, development activity, and philosophical alignment analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "🗽",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Liberty Coin Intelligence

Comprehensive Liberty Coin monitoring and analysis.

### Tracking Areas
- **Price Monitoring**: Real-time price across exchanges, volume, market cap
- **On-Chain Analytics**: Transaction count, active addresses, velocity
- **Community Metrics**: Social media growth, developer activity, governance
- **Philosophical Analysis**: Alignment with freedom/sovereignty principles
- **Comparative Analysis**: vs BTC, ETH, XRP, other sovereignty coins
- **Regulatory Landscape**: Legal status across jurisdictions
- **Calgary Integration**: CAD/Liberty trading pairs and local adoption

### Use Cases
- Portfolio allocation modeling
- Arbitrage between exchanges
- Community sentiment tracking
- Development milestone monitoring
- Governance participation alerts

### Commands
```
/bw-liberty price                 — Current price and volume
/bw-liberty analysis              — Technical and fundamental analysis
/bw-liberty community             — Community health metrics
/bw-liberty compare               — Comparative analysis vs major coins
/bw-liberty onchain               — On-chain activity metrics
/bw-liberty governance            — Active governance proposals
```
""",
},

# ─── 24. X TOKENS ─────────────────────────────────────────────────────────

"bw-x-tokens": {
    "name": "bw-x-tokens",
    "description": "X token ecosystem intelligence. Tracking all X-branded tokens, social platform tokens, and emerging crypto-social convergence assets.",
    "metadata": {
        "openclaw": {
            "emoji": "✖️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET X Token Intelligence

X token ecosystem monitoring and analysis.

### Tracked Assets
- **X Platform Tokens**: Any tokens associated with X (Twitter) ecosystem
- **Social-Fi Tokens**: Friend.tech, Lens Protocol, Farcaster ecosystem
- **Creator Economy Tokens**: Social platform monetization tokens
- **Cross-Platform Tokens**: Tokens bridging social media and DeFi

### Analysis Areas
- Platform adoption metrics and token utility
- Social engagement correlation with price
- Influencer holdings and activity tracking
- Regulatory risk for social platform tokens
- Integration opportunities with AAC agent roster
- Cross-platform arbitrage between social tokens

### Commands
```
/bw-xtokens overview              — X token ecosystem overview
/bw-xtokens social-fi             — Social-Fi token analysis
/bw-xtokens creators              — Creator economy token scan
/bw-xtokens arbitrage             — Cross-platform token arbitrage
/bw-xtokens influencers           — Top influencer token activity
/bw-xtokens adoption              — Platform adoption metrics
```
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# FINANCE & BANKING (Skills 25-27)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 25. BANKING INTELLIGENCE ─────────────────────────────────────────────

"bw-banking-intel": {
    "name": "bw-banking-intel",
    "description": "Banking intelligence covering offshore accounts, international banking, SWIFT/wire transfers, multi-currency accounts, and Calgary/Montevideo banking corridor optimization.",
    "metadata": {
        "openclaw": {
            "emoji": "🏦",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Banking Intelligence

International banking, offshore structures, and multi-jurisdiction optimization.

### Banking Areas
**Domestic (Calgary, Alberta)**:
- Canadian chartered bank analysis (Big Five optimization)
- TFSA/RRSP integration with trading strategies
- CDIC coverage optimization across accounts
- Canadian mortgage rate tracking and optimization
- CRA compliance and tax-efficient structuring

**International (Montevideo, Uruguay)**:
- Uruguayan banking system analysis
- Free Zone (Zona Franca) banking advantages
- USD-denominated accounts in Uruguay
- South American banking corridor optimization
- Mercosur financial integration opportunities

**Offshore Structures**:
- Multi-jurisdiction account optimization
- Correspondent banking relationship mapping
- SWIFT vs crypto cross-border payment comparison
- Currency diversification strategies (CAD/USD/UYU/EUR)
- Regulatory compliance across jurisdictions

### Cash Management
- Multi-currency cash flow optimization
- Sweep account strategies
- Treasury management for trading capital
- Emergency liquidity reserves (Doctrine Pack 1)

### Commands
```
/bw-bank overview                 — Banking structure overview
/bw-bank calgary                  — Calgary banking optimization
/bw-bank montevideo               — Montevideo banking analysis
/bw-bank offshore                 — Offshore structure analysis
/bw-bank wire-routes              — Optimal wire transfer routes
/bw-bank rates                    — Interest rate comparison
/bw-bank compliance               — Multi-jurisdiction compliance check
/bw-bank cash-management          — Cash position optimization
```
""",
},

# ─── 26. ACCOUNTING ENGINE ────────────────────────────────────────────────

"bw-accounting-engine": {
    "name": "bw-accounting-engine",
    "description": "Financial accounting, reporting, P&L tracking, tax optimization, and CentralAccounting integration. Real-time books across all AAC trading strategies.",
    "metadata": {
        "openclaw": {
            "emoji": "📑",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Accounting Engine

Financial accounting powered by AAC's CentralAccounting department.

### Accounting Capabilities
- **Real-Time P&L**: Per-trade, per-strategy, per-department tracking
- **Cost Basis Tracking**: FIFO, LIFO, HIFO, specific identification
- **Tax Lot Management**: Crypto and traditional asset tax lot optimization
- **Multi-Currency Books**: CAD, USD, UYU, crypto auto-conversion
- **Performance Attribution**: Where is the P&L coming from?
- **Fee Tracking**: Gas fees, exchange fees, slippage costs
- **Reconciliation**: Cross-exchange, cross-chain position reconciliation

### Tax Intelligence
- Capital gains optimization (short-term vs long-term)
- Tax-loss harvesting scanner
- CRA (Canada Revenue Agency) compliance
- DGI (Uruguay) tax reporting
- Crypto-specific tax event identification
- Wash sale rule monitoring

### Reports Generated
- Daily P&L Statement
- Weekly Strategy Attribution
- Monthly Portfolio Report
- Quarterly Tax Estimate
- Annual Tax Summary
- Ad-hoc custom reports

### Commands
```
/bw-accounting pnl today          — Today's P&L across all strategies
/bw-accounting pnl mtd            — Month-to-date P&L
/bw-accounting tax-harvest         — Tax-loss harvesting opportunities
/bw-accounting cost-basis BTC      — BTC cost basis analysis
/bw-accounting fees                — Fee analysis and optimization
/bw-accounting reconcile           — Position reconciliation check
/bw-accounting report monthly      — Generate monthly report
/bw-accounting tax-estimate        — Quarterly tax estimate
```
""",
},

# ─── 27. REGULATIONS ──────────────────────────────────────────────────────

"bw-regulations": {
    "name": "bw-regulations",
    "description": "Local and international financial regulations tracker. Calgary/Alberta/Canada and Montevideo/Uruguay focused, with global crypto regulation monitoring.",
    "metadata": {
        "openclaw": {
            "emoji": "⚖️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Regulatory Intelligence

Multi-jurisdiction regulatory tracking focused on Calgary and Montevideo operations.

### Calgary / Alberta / Canada
- **Securities**: Alberta Securities Commission (ASC) rules
- **Crypto**: CSA (Canadian Securities Administrators) crypto framework
- **Banking**: OSFI (Office of the Superintendent of Financial Institutions)
- **Tax**: CRA cryptocurrency taxation rules
- **AML/KYC**: FINTRAC registration and compliance
- **Trading**: IIROC universal market integrity rules
- **Alberta Advantage**: No provincial sales tax, favorable business climate

### Montevideo / Uruguay
- **Banking**: BCU (Banco Central del Uruguay) regulations
- **Free Zones**: Zona Franca tax incentives and reporting
- **Tax**: DGI (Dirección General Impositiva) rules
- **Crypto**: Uruguay digital asset framework
- **Cross-Border**: Mercosur financial regulations
- **Residency**: Fiscal residency requirements and benefits

### International
- **FATF**: Financial Action Task Force recommendations (AML/CFT)
- **Basel III**: Capital adequacy and liquidity requirements
- **MiCA**: EU Markets in Crypto-Assets regulation
- **SEC/CFTC**: US securities and commodities oversight
- **Singapore MAS**: Monetary Authority of Singapore framework
- **Dubai VARA**: Virtual Assets Regulatory Authority rules

### Compliance Arbitrage
Identifying regulatory differences between jurisdictions that create
legitimate opportunities. Fully doctrine-compliant, no grey-area exploitation.

### Commands
```
/bw-reg calgary                   — Calgary/Alberta regulatory summary
/bw-reg montevideo                — Montevideo/Uruguay regulatory summary
/bw-reg crypto-global             — Global crypto regulation map
/bw-reg changes                   — Recent regulatory changes
/bw-reg compliance-check          — Run compliance check on current ops
/bw-reg arbitrage                 — Regulatory arbitrage opportunities
/bw-reg fatf                      — FATF compliance status
/bw-reg alerts                    — Regulatory change alerts setup
```

### Regulatory Alert System
Auto-monitors regulatory body websites and news for changes affecting
AAC operations. Alerts via @barrenwuffet069bot when:
- New cryptocurrency regulations proposed or enacted
- Tax law changes in Canada or Uruguay
- AML/KYC requirement updates
- Securities classification changes
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# WEALTH BUILDING (Skills 28-30)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 28. MONEY MASTERY ────────────────────────────────────────────────────

"bw-money-mastery": {
    "name": "bw-money-mastery",
    "description": "Money planning, cash flow mastery, budgeting, emergency fund management, and financial freedom pathway. Combines traditional finance wisdom with crypto-era strategies.",
    "metadata": {
        "openclaw": {
            "emoji": "💰",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Money Mastery

Comprehensive money planning and cash flow management system.

### Money Mastery Framework
```
EARN → SAVE → INVEST → GROW → PROTECT → MULTIPLY
```

### Pillars
1. **Cash Flow Architecture**:
   - Income stream mapping (active, passive, portfolio)
   - Expense optimization and leak detection
   - Automated savings waterfall
   - Emergency fund management (3-12 months)

2. **Savings Optimization**:
   - High-yield savings account comparison
   - GIC laddering strategies (Canadian)
   - TFSA maximization (Tax-Free Savings Account)
   - RRSP vs TFSA optimization calculator

3. **Debt Management**:
   - Debt snowball vs avalanche analysis
   - Interest rate optimization
   - Mortgage vs investing calculator
   - Credit utilization optimization

4. **Financial Freedom Metrics**:
   - FI number calculation (25x expenses)
   - Savings rate tracking and optimization
   - Coast FI / Barista FI / Lean FI milestones
   - Passive income coverage ratio

5. **Cash Management for Traders**:
   - Trading bankroll management
   - Win/loss streak contingency
   - Lifestyle creep prevention
   - Tax reserve allocation

### Commands
```
/bw-money overview                — Financial health dashboard
/bw-money cashflow                — Cash flow analysis
/bw-money budget                  — Budget optimization
/bw-money savings-rate            — Savings rate tracker
/bw-money fi-number               — Financial independence calculator
/bw-money emergency-fund          — Emergency fund status & targets
/bw-money debt-plan               — Debt payoff optimization
/bw-money tfsa-rrsp               — TFSA vs RRSP optimization
```
""",
},

# ─── 29. WEALTH BUILDING ──────────────────────────────────────────────────

"bw-wealth-building": {
    "name": "bw-wealth-building",
    "description": "Generational wealth strategies, asset protection, estate planning, trust structures, and multi-generational financial architecture for Calgary and international operations.",
    "metadata": {
        "openclaw": {
            "emoji": "🏔️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Generational Wealth Builder

Long-term wealth creation, preservation, and transfer strategies.

### Wealth Building Phases
```
Phase 1: ACCUMULATE (0-$100K)    — Aggressive growth, skill building
Phase 2: ACCELERATE ($100K-$1M)  — Leverage, diversification, systems
Phase 3: PRESERVE ($1M-$10M)     — Asset protection, tax optimization
Phase 4: MULTIPLY ($10M+)        — Generational structures, legacy
```

### Asset Classes for Generational Wealth
- **Real Estate**: Canadian property, international diversification
- **Equities**: Index funds, dividend growth, quality companies
- **Crypto**: BTC, ETH as digital gold, DeFi yields
- **Business Equity**: Active business ownership and exit planning
- **Commodities**: Gold, silver, energy (Alberta advantage)
- **Private Equity / VC**: Accredited investor opportunities
- **Intellectual Property**: Royalties, patents, content

### Generational Wealth Structures
- **Family Trust (Canada)**: Inter vivos trusts, testamentary trusts
- **Holding Company**: Canadian CCPC (Canadian Controlled Private Corp)
- **Estate Freeze**: Locking current value, future growth to next gen
- **Insurance Structures**: Corporate-owned life insurance for tax-free transfer
- **Uruguay Structures**: Free Zone companies, SAU (Sociedad Anónima Uruguaya)
- **Multi-Jurisdiction**: Canada-Uruguay holding structure optimization

### Wealth Saving Strategies
- Pay yourself first (automated 20%+ savings)
- Dollar-cost averaging into appreciating assets
- Tax-advantaged account maximization (TFSA, RRSP, RESP)
- Dividend reinvestment programs (DRIPs)
- Geographic arbitrage (cost of living optimization)

### Commands
```
/bw-wealth overview               — Wealth building dashboard
/bw-wealth phase                  — Current wealth building phase
/bw-wealth allocation             — Asset allocation recommendation
/bw-wealth generational           — Generational wealth structures
/bw-wealth estate                 — Estate planning overview
/bw-wealth trust                  — Trust structure analysis
/bw-wealth tax-optimization       — Tax optimization strategies
/bw-wealth offshore               — International wealth structures
/bw-wealth insurance              — Insurance strategy analysis
/bw-wealth legacy                 — Legacy and succession planning
```
""",
},

# ─── 30. SUPERSTONK DD ────────────────────────────────────────────────────

"bw-superstonk-dd": {
    "name": "bw-superstonk-dd",
    "description": "SuperStonk-style due diligence research engine. Deep market research, short interest analysis, FTD tracking, dark pool activity, and retail vs institutional flow analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "🦍",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET SuperStonk DD Engine

Deep due diligence research inspired by SuperStonk methodology.

### DD Categories
1. **Short Interest Analysis**:
   - SI% of float tracking across exchanges
   - Short volume ratio monitoring
   - Cost-to-borrow rate tracking
   - Short squeeze probability scoring

2. **FTD (Failure to Deliver) Tracking**:
   - SEC FTD data analysis and T+35 cycle mapping
   - FTD spike pattern recognition
   - Correlation with price movements

3. **Dark Pool Activity**:
   - Off-exchange volume percentage tracking
   - Dark pool price vs lit exchange divergence
   - Unusual dark pool block sizes
   - PFOF (Payment for Order Flow) routing analysis

4. **Institutional Ownership**:
   - 13F filing analysis and changes
   - Insider trading (Form 4) monitoring
   - Institutional vs retail ownership ratios
   - ETF concentration and rebalancing effects

5. **DRS (Direct Registration)**:
   - Share count tracking where available
   - Transfer agent activity monitoring

6. **Market Microstructure**:
   - Order types and routing analysis
   - Odd-lot trading patterns
   - Tick-by-tick abnormality detection
   - Market maker positioning

### Research Methodology
```
Hypothesis → Data Collection → Statistical Analysis → Peer Review → Signal
```

### Commands
```
/bw-dd overview <ticker>          — Full DD report for any ticker
/bw-dd short-interest <ticker>    — Short interest deep dive
/bw-dd ftd <ticker>               — FTD cycle analysis
/bw-dd dark-pool <ticker>         — Dark pool activity report
/bw-dd institutional <ticker>     — 13F and insider analysis
/bw-dd reddit-feed                — Latest SuperStonk DD posts
/bw-dd squeeze-score <ticker>     — Short squeeze probability
/bw-dd market-structure           — Broad market microstructure analysis
```
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS (Skills 31-33)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 31. 2007 CRASH INDICATORS ────────────────────────────────────────────

"bw-crash-indicators": {
    "name": "bw-crash-indicators",
    "description": "2007/2008 financial crisis pattern detection. Monitors yield curves, credit spreads, housing data, bank CDS, leverage ratios, and systemic risk indicators for early warning.",
    "metadata": {
        "openclaw": {
            "emoji": "🚨",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
            "always": True,
        }
    },
    "instructions": """
## BARREN WUFFET 2007/2008 Crash Indicator Panel

Pattern matching against the 2007/2008 financial crisis for early warning signals.

### 2007-Style Indicators Monitored

**Credit Market Signals** (First to move in 2007):
- IG/HY Credit Spread: CDX.IG, CDX.HY spread widening
- Bank CDS Spreads: Major bank credit default swap costs
- LIBOR-OIS Spread: Interbank lending stress
- Commercial Paper Rates: Corporate short-term funding stress
- TED Spread: Treasury vs Eurodollar (banking system trust)

**Housing & Real Estate** (Root cause in 2007):
- Case-Shiller Home Price Index deceleration
- Mortgage delinquency rates rising
- Housing starts collapse
- ARM reset volume tracking
- CMBS (Commercial Mortgage-Backed Securities) distress

**Yield Curve** (Classic recession predictor):
- 2s10s spread (2-year vs 10-year Treasury)
- 3m10y spread (most reliable recession signal)
- Full yield curve shape analysis
- Duration of inversion vs historical patterns

**Leverage & Liquidity**:
- Bank leverage ratios vs 2007 levels
- Margin debt as % of GDP
- Repo market stress indicators
- Money market fund outflows
- VIX term structure (contango → backwardation)

**Systemic Risk**:
- Too-Big-To-Fail bank interconnectedness index
- Counterparty exposure concentration
- Shadow banking sector growth
- ETF liquidity mismatch risk
- Crypto-specific contagion channels (stablecoin runs, exchange failures)

### Crash Similarity Score
```
┌─────────────────────────────────────────┐
│  2007 CRASH SIMILARITY: XX/100          │
│  ████████░░░░░░░░░░░░ XX%              │
│                                         │
│  Credit Stress:     ███░░░░ 3/7        │
│  Housing Risk:      ██░░░░░ 2/7        │
│  Yield Curve:       █████░░ 5/7        │
│  Leverage:          ███░░░░ 3/7        │
│  Systemic Risk:     ██░░░░░ 2/7        │
│                                         │
│  Overall Assessment: ELEVATED CAUTION   │
└─────────────────────────────────────────┘
```

### Auto-Integration with Doctrine
When Crash Similarity Score > 60:
- BarrenWuffetState automatically moves to CAUTION
- Hedging recommendations activate
- Position sizing reduces per Doctrine Pack 2
- Crisis playbook engages

### Commands
```
/bw-crash dashboard               — Full crash indicator panel
/bw-crash score                   — Current crash similarity score
/bw-crash credit                  — Credit market stress indicators
/bw-crash housing                 — Housing market analysis
/bw-crash yield-curve             — Yield curve analysis
/bw-crash leverage                — Market leverage assessment
/bw-crash systemic                — Systemic risk evaluation
/bw-crash compare-2007            — Side-by-side comparison with 2007
/bw-crash history                 — Historical crash pattern library
/bw-crash alert-thresholds        — Configure early warning thresholds
```
""",
},

# ─── 32. GOLDEN RATIO FINANCE ─────────────────────────────────────────────

"bw-golden-ratio-finance": {
    "name": "bw-golden-ratio-finance",
    "description": "Dan Winter golden ratio harmonic analysis applied to financial markets. Fibonacci time/price confluences, phi-based support/resistance, and fractal wave analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "🌀",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Golden Ratio Finance (Dan Winter Method)

Apply Dan Winter's golden ratio / phi harmonic analysis to financial markets.

### Dan Winter Core Principles
- **Phase Conjugation**: Markets exhibit constructive interference at phi ratios
- **Golden Mean Wave Mechanics**: Price unfolds in golden ratio spirals
- **Fractal Compression**: Recursive self-similarity in market structures
- **Implosion Physics Applied to Markets**: Charge compression at golden mean nodes
  creates price attractors and repellers

### Phi (φ = 1.618) in Markets
**Fibonacci Price Levels**:
- 23.6% (1 - 0.618²), 38.2% (0.618²), 50%, 61.8% (φ⁻¹), 78.6% (√0.618)
- Extensions: 127.2%, 161.8% (φ), 261.8% (φ²), 423.6% (φ³)

**Fibonacci Time Analysis**:
- Time cycles between major turning points often relate by φ
- Day counts: 8, 13, 21, 34, 55, 89, 144, 233, 377 (Fibonacci sequence)
- Time/price confluence = highest probability reversal zones

**Golden Spiral on Charts**:
- Logarithmic spiral overlaid on price action
- Phi channels (parallel channels at φ spacing)
- Golden ratio fan lines from major pivots

### Dan Winter Specific Tools
- **Phase Conjugate Mirror Points**: Where multiple Fibonacci levels converge
  from different pivot points (creates "implosion" zones of high probability)
- **Fractal Field Mapping**: Multi-timeframe Fibonacci overlay showing
  recursive self-similarity in market structure
- **Golden Ratio Oscillator**: Custom oscillator measuring deviation from
  phi-based equilibrium levels
- **Harmonic Convergence Zones**: Where time and price Fibonacci levels
  meet simultaneously (highest confidence reversal signals)

### Commands
```
/bw-golden analysis BTC           — Golden ratio analysis for Bitcoin
/bw-golden fibonacci SPY          — Full Fibonacci level suite for SPY
/bw-golden time-cycles            — Fibonacci time cycle projections
/bw-golden spiral                 — Golden spiral overlay on current chart
/bw-golden convergence            — Phase conjugate convergence zones
/bw-golden harmonics              — Multi-timeframe harmonic analysis
/bw-golden oscillator BTC         — Golden ratio oscillator reading
/bw-golden dan-winter             — Full Dan Winter method report
```

### Integration with Trading Signals
When golden ratio convergence zones align with trading signals from
the QuantumSignalAggregator, signal confidence is boosted by 15-25%.
These high-confidence "harmonic convergence" signals are flagged
with 🌀 in all BARREN WUFFET outputs.
""",
},

# ─── 33. JONNY BRAVO TRADING COURSE ───────────────────────────────────────

"bw-jonny-bravo-course": {
    "name": "bw-jonny-bravo-course",
    "description": "Jonny Bravo trading methodology and course curriculum integrated into BARREN WUFFET. Advanced trading education with real-time application through AAC agents.",
    "metadata": {
        "openclaw": {
            "emoji": "💪",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET × Jonny Bravo Trading Division

The Jonny Bravo trading methodology integrated with AAC's agent infrastructure.
Access the full course curriculum and apply lessons in real-time through
the Jonny Bravo Division agents.

### Course Modules
1. **Module 1: Market Structure Fundamentals**
   - Support & resistance identification
   - Trend analysis (higher highs, lower lows)
   - Market phases: accumulation, markup, distribution, markdown
   - Volume confirmation principles

2. **Module 2: Price Action Mastery**
   - Candlestick patterns (engulfing, doji, hammer, shooting star)
   - Chart patterns (H&S, double top/bottom, flags, wedges)
   - Break-out vs fake-out identification
   - Multi-timeframe analysis

3. **Module 3: Risk Management (The Jonny Way)**
   - Never risk more than 1-2% per trade
   - Risk/reward ratio minimum 1:2
   - Position sizing calculator
   - Journal every trade (win or lose)
   - Psychology: handling losses, avoiding revenge trading

4. **Module 4: Technical Indicators**
   - Moving averages (EMA 9, 21, 50, 200)
   - RSI divergence trading
   - MACD histogram momentum
   - Bollinger Band squeeze plays
   - Volume profile and VPVR

5. **Module 5: Advanced Strategies**
   - Order block identification
   - Fair value gap (FVG) trading
   - Liquidity sweep setups
   - Smart money concepts (SMC)
   - ICT (Inner Circle Trader) methodology integration

6. **Module 6: Crypto-Specific Trading**
   - Crypto market structure differences
   - Funding rate trading
   - Altcoin rotation strategies
   - DeFi yield farming as active management
   - Meme coin entry/exit frameworks

### Jonny Bravo Division Agents
The Jonny Bravo Division in AAC has dedicated agents implementing
this methodology:
- **JB_TrendAnalyzer**: Market structure and trend identification
- **JB_RiskManager**: Position sizing and risk management
- **JB_PatternScanner**: Real-time pattern detection
- **JB_VolumeProfiler**: Volume analysis and VPR
- **JB_TradeJournaler**: Automated trade journaling

### Commands
```
/bw-jonny module 1                — Start Module 1 content
/bw-jonny quiz module=3           — Random quiz from Module 3
/bw-jonny setup trade=LONG BTC    — Get Jonny Bravo trade setup for long BTC
/bw-jonny review                  — Review recent trades (Jonny method)
/bw-jonny risk-calc               — Position sizing calculator
/bw-jonny journal                 — View/add to trade journal
/bw-jonny smc-setup               — Current SMC (Smart Money) setups
/bw-jonny level                   — Your current course level
/bw-jonny leaderboard             — Student leaderboard
```

### Real-Time Application
Every Jonny Bravo course concept is backed by live agents:
- Learn a concept → Agent applies it to live markets → See real results
- Paper trading mode for students (no real capital risk)
- Performance tracking against the methodology's expected win rate
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# OPENCLAW POWER-UPS (Skills 34-35)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 34. POLYMARKET AUTOPILOT ──────────────────────────────────────────────

"bw-polymarket-autopilot": {
    "name": "bw-polymarket-autopilot",
    "description": "Prediction market paper trading autopilot for Polymarket. TAIL, BONDING, and SPREAD strategies with daily P&L summaries via Telegram.",
    "metadata": {
        "openclaw": {
            "emoji": "🎰",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY", "TELEGRAM_BOT_TOKEN"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Polymarket Autopilot

Automated paper trading on Polymarket prediction markets.
Adapted from the OpenClaw Polymarket Autopilot usecase pattern.

### Strategies
- **TAIL**: Follow trends when volume spikes and momentum > 60%
- **BONDING**: Contrarian plays on overreactions (drops > 10% on news)
- **SPREAD**: Arbitrage when YES+NO > 1.05 (mispricing)
- **AAC-ENHANCED**: Uses BigBrainIntelligence research agents for edge

### Pipeline
```
Polymarket API → Market Data → Strategy Engine → Paper Trade → Portfolio Track → Telegram Report
```

### Commands
```
/bw-poly scan                     — Current market opportunities
/bw-poly portfolio                — Paper trading portfolio status
/bw-poly pnl                      — P&L summary
/bw-poly strategy=TAIL            — TAIL strategy signals
/bw-poly strategy=SPREAD          — SPREAD arbitrage opportunities
/bw-poly history                  — Trade history log
/bw-poly backtest days=30         — Backtest strategies
```

### Telegram Delivery
Daily summary to @barrenwuffet069bot at 8 AM:
- Yesterday's paper trades with entry/exit prices
- Current portfolio value and open positions
- Win rate by strategy
- Market insights and recommendations

### Starting Capital: $10,000 (paper)
""",
},

# ─── 35. SECOND BRAIN ─────────────────────────────────────────────────────

"bw-second-brain": {
    "name": "bw-second-brain",
    "description": "Zero-friction knowledge capture and memory system via Telegram. Text anything to BARREN WUFFET and it remembers permanently. Searchable knowledge base.",
    "metadata": {
        "openclaw": {
            "emoji": "🧠",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY", "TELEGRAM_BOT_TOKEN"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Second Brain

Zero-friction knowledge capture via @barrenwuffet069bot.
Text anything → it's stored permanently → searchable forever.

### How It Works
Just text your bot:
```
"Remember: NVDA earnings beat expectations, AI capex up 40%"
"Save this strategy: buy ETH when funding rates go negative"
"Note: Calgary property market showing 2007-style divergence"
"Read this later: https://example.com/golden-ratio-trading"
```

BARREN WUFFET stores everything in AAC's doctrine memory system.
Every note is tagged with timestamp, category (auto-detected),
and cross-referenced with existing knowledge.

### Auto-Categorization
Notes are automatically filed into:
- 📊 Market Intelligence (prices, signals, observations)
- 📝 Trading Ideas (strategy concepts, setups)
- 🏦 Banking & Finance (accounts, rates, structures)
- 📜 Regulations (rule changes, compliance notes)
- 🧠 Research (DD, analysis, reports)
- 💡 General Ideas (everything else)

### Search & Retrieval
```
/bw-brain search <query>          — Semantic search all memories
/bw-brain recent                  — Recent 10 entries
/bw-brain category=trading        — Filter by category
/bw-brain date=2026-02            — Filter by date
/bw-brain export                  — Export all notes as JSON/CSV
```

### Integration with AAC Doctrine
All captured notes feed into the doctrine memory layer, accessible
by all 80+ AAC agents. When an agent runs analysis, it can reference
your personal notes for context.

Based on the OpenClaw Second Brain usecase pattern.
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# DEEP DIVE BATCH 3 — NEW SKILLS (36-65)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 36. BLACK-SCHOLES PRICING ─────────────────────────────────────────────

"bw-black-scholes": {
    "name": "bw-black-scholes",
    "description": "Black-Scholes options pricing engine adapted for binary options, prediction markets (Polymarket), and traditional options. Fair value calculation with configurable edge thresholds.",
    "metadata": {
        "openclaw": {
            "emoji": "📐",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Black-Scholes Pricing Engine

Quantitative pricing engine adapted from the Polymarket bot that made $116K in 24 hours.
Uses Black-Scholes framework for both traditional options and binary prediction markets.

### Core Pricing Models
- **Traditional Black-Scholes**: European option pricing with continuous dividends
- **Binary Options Adaptation**: Fair value for YES/NO prediction market contracts
- **Implied Volatility Solver**: Newton-Raphson IV extraction from market prices
- **Greeks Calculation**: Full greek suite (Delta, Gamma, Theta, Vega, Rho)

### Polymarket Integration (Bidou28old Method)
```
Pipeline: Binance WebSocket → Price Delta → Black-Scholes Fair Value → Compare CLOB → Execute
```
- **6-cent minimum edge**: Only trade when fair value beats ask by ≥ 6 cents
- **WebSocket feeds**: Real-time BTC/ETH price from Binance for underlying
- **CLOB API**: Polymarket Central Limit Order Book for execution
- **UMA Oracle**: 2-hour undisputed resolution for settlement

### Strategy Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| min_edge | 0.06 | Minimum edge over market price |
| vol_window | 30d | Historical volatility lookback |
| risk_free_rate | 0.05 | Risk-free rate (US Treasury) |
| time_to_expiry | auto | Contract expiration in years |

### Commands
```
/bw-bs price underlying=BTC strike=100K expiry=30d   — Price a BTC option
/bw-bs binary market_id=<id>                          — Fair value for Polymarket contract
/bw-bs iv option_price=5.50 underlying=SPY            — Implied volatility extraction
/bw-bs greeks SPY 450C 0321                           — Full greeks for contract
/bw-bs edge-scan                                      — Scan for ≥6-cent edge opportunities
/bw-bs calibrate                                      — Calibrate model to market prices
```

### Source: theworldmag.com — OpenClaw Polymarket Bot ($116K/24h)
""",
},

# ─── 37. SECURITY HARDENING ───────────────────────────────────────────────

"bw-security-hardening": {
    "name": "bw-security-hardening",
    "description": "OpenClaw security hardening and CVE monitoring. Lethal Trifecta defense, ClawHub skill vetting, credential encryption, Docker isolation, and API cost protection.",
    "metadata": {
        "openclaw": {
            "emoji": "🔐",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
            "always": True,
        }
    },
    "instructions": """
## BARREN WUFFET Security Hardening

Comprehensive security for the AAC agent infrastructure based on adversa.ai research.

### The Lethal Trifecta + Memory (Simon Willison + Palo Alto)
1. ❌ Access to private data (emails, files, credentials, browser history)
2. ❌ Exposure to untrusted content (web browsing, incoming messages, skills)
3. ❌ Ability to communicate externally (sends emails, API calls, data exfil)
4. ❌ Persistent memory (SOUL.md/MEMORY.md enable time-shifted prompt injection)

### Critical CVE Monitoring
| CVE | Type | CVSS | Status |
|-----|------|------|--------|
| CVE-2026-25253 | 1-Click RCE | 8.8 | Patched v2026.1.29 |
| CVE-2026-24763 | Command Injection | HIGH | Patched |
| CVE-2026-25157 | Command Injection | HIGH | Patched |
| CVE-2026-22708 | Indirect Prompt Injection | HIGH | Mitigated |

### ClawHavoc Supply Chain Attack Defense
- **341/2,857 skills malicious** (12%) — NEVER install unaudited skills
- 335 delivered Atomic Stealer (AMOS) macOS malware
- 6 reverse shell backdoors — all sharing C2 IP: 91.92.242[.]30
- Use Cisco Skill Scanner, Snyk Agent Scan, Agent Trust Hub

### 10 Mandatory Security Controls
```
1. gateway.auth.password — NEVER run without auth
2. API spending hard limits — per-provider caps
3. Docker --read-only --cap-drop=ALL
4. Bind UI to 127.0.0.1 ONLY (never 0.0.0.0)
5. Tailscale/VPN for remote access
6. NEVER install unaudited ClawHub skills
7. Rotate all tokens regularly
8. Run: openclaw security audit --deep --fix
9. TLS 1.3 for all gateway comms
10. n8n proxy — agent never touches API creds
```

### Commands
```
/bw-security audit                — Run full security audit
/bw-security cve-check            — Check for known CVEs
/bw-security skill-scan <name>    — Scan a skill for threats
/bw-security credentials          — Credential rotation status
/bw-security api-costs            — API spending vs limits
/bw-security exposure             — Check for exposed instances
/bw-security docker               — Docker hardening status
/bw-security trifecta             — Lethal trifecta assessment
```

### Source: adversa.ai — OpenClaw Security 101 (Feb 2026)
""",
},

# ─── 38. SKILL SCANNER ────────────────────────────────────────────────────

"bw-skill-scanner": {
    "name": "bw-skill-scanner",
    "description": "ClawHub skill security scanner. Vets skills before installation using static analysis, behavioral analysis, LLM review, and VirusTotal integration.",
    "metadata": {
        "openclaw": {
            "emoji": "🛡️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Skill Security Scanner

Vet ClawHub skills before installation. 12% of skills contain malware.

### Scanning Pipeline
```
Skill URL → Download → Static Analysis → Behavioral Analysis → LLM Review → VirusTotal → Report
```

### Analysis Layers
1. **Static Analysis**: Code pattern matching for known malware signatures
   - Shell command injection patterns
   - Network exfiltration patterns (curl, wget to unknown IPs)
   - File system access outside expected paths
   - Credential harvesting patterns (env vars, keychains)

2. **Behavioral Analysis**: Sandbox execution and monitoring
   - Process spawning behavior
   - Network connection attempts
   - File system modifications
   - Memory resident patterns

3. **LLM Review**: AI-powered code understanding
   - Prompt injection detection in instructions
   - Hidden instruction identification (CSS display:none, etc.)
   - Tool poisoning detection (rug-pull descriptions vs actual behavior)

4. **VirusTotal Integration**: Hash-based malware lookup
   - Cross-reference with 70+ antivirus engines
   - Community reputation scoring

### Known Threat Indicators
- C2 IP: 91.92.242[.]30 (ClawHavoc campaign)
- AMOS (Atomic Stealer) signatures
- Reverse shell patterns (bash -i, nc, python -c)
- Fake prerequisite scripts (npm, pip install from malicious repos)

### Commands
```
/bw-scan-skill <url>              — Full security scan of a skill
/bw-scan-skill audit-all          — Scan all installed skills
/bw-scan-skill whitelist          — View/manage trusted skills
/bw-scan-skill threats            — Known threat database
/bw-scan-skill report <name>      — Detailed security report
```

### Source: adversa.ai, Cisco Skill Scanner, Snyk Agent Scan
""",
},

# ─── 39. FLASH LOANS ──────────────────────────────────────────────────────

"bw-flash-loans": {
    "name": "bw-flash-loans",
    "description": "DeFi flash loan arbitrage engine. Capital-efficient arbitrage with zero upfront capital using Aave, dYdX, and Uniswap flash loans.",
    "metadata": {
        "openclaw": {
            "emoji": "⚡",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Flash Loan Arbitrage Engine

Capital-efficient DeFi arbitrage using flash loans (zero upfront capital required).

### Flash Loan Providers
| Provider | Max Loan | Fee | Speed |
|----------|----------|-----|-------|
| Aave V3 | Unlimited (pool) | 0.09% | 1 block |
| dYdX | Unlimited (pool) | 0% (gas only) | 1 block |
| Uniswap V3 | Pool liquidity | 0.3% swap | 1 block |
| Balancer V2 | Pool liquidity | 0% | 1 block |

### Arbitrage Routes
- **DEX-DEX**: Borrow → Buy on DEX A → Sell on DEX B → Repay → Profit
- **Liquidation**: Borrow → Liquidate undercollateralized position → Sell collateral → Repay
- **Collateral Swap**: Borrow → Swap collateral on lending protocol → Repay
- **Price Oracle Exploit**: Borrow → Manipulate oracle (ethical: report only) → Repay

### Risk Factors
- Transaction reverts if not profitable (atomic: all-or-nothing)
- Gas costs can exceed arbitrage profit on mainnet
- MEV bots front-running flash loan transactions
- Smart contract risk in flash loan providers
- L2s reduce gas but limit flash loan availability

### Pipeline
```
Opportunity Detection → Route Optimization → Gas Estimation → Profitability Check → Execute/Skip
```

### Commands
```
/bw-flash scan                    — Scan for flash loan opportunities
/bw-flash routes                  — Available arbitrage routes
/bw-flash simulate <route>        — Simulate without executing
/bw-flash gas-estimate            — Current gas cost for flash loans
/bw-flash providers               — Flash loan provider comparison
/bw-flash history                 — Past flash loan execution log
```

### Source: Aurpay.net — 10 Crypto Trading Use Cases
""",
},

# ─── 40. DCA & GRID STRATEGIES ────────────────────────────────────────────

"bw-dca-grid": {
    "name": "bw-dca-grid",
    "description": "Dollar-cost averaging ladders and grid trading strategies. Automated accumulation during dips and range-bound profit extraction.",
    "metadata": {
        "openclaw": {
            "emoji": "📊",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET DCA & Grid Trading

Systematic accumulation and range-bound trading strategies.

### DCA Ladders
Accumulate positions during dips with configurable step sizes.
```
Step 1: Buy $100 at current price
Step 2: Buy $150 at -5% (bonus for dip)
Step 3: Buy $200 at -10% (bigger allocation on bigger dip)
Step 4: Buy $300 at -15% (maximum conviction dip buy)
Step 5: Buy $500 at -20% (generational buying opportunity)
```

### Grid Trading
Profit in range-bound markets with automated buy/sell grids.
```
Upper Bound: $110 ─── SELL ─── SELL ─── SELL
             $108 ─── SELL
             $106 ─── SELL
             $104 ─── SELL
Center:      $100 ─────────────────────────
             $96  ─── BUY
             $94  ─── BUY
             $92  ─── BUY
             $90  ─── BUY
Lower Bound: $88  ─── BUY ─── BUY ─── BUY
```

### Grid Parameters
| Parameter | Default | Range |
|-----------|---------|-------|
| grid_levels | 10 | 5-50 |
| grid_spacing | 2% | 0.5%-10% |
| position_size | equal | equal/geometric |
| upper_bound | +10% | configurable |
| lower_bound | -10% | configurable |

### Commands
```
/bw-dca setup BTC budget=1000 steps=5     — Set up DCA ladder for BTC
/bw-dca status                             — Current DCA positions
/bw-grid setup ETH range=90-110 levels=10  — Set up grid for ETH
/bw-grid status                            — Grid trading status
/bw-grid pnl                               — Grid P&L analysis
/bw-dca backtest BTC days=90               — Backtest DCA over 90 days
/bw-grid backtest ETH days=30              — Backtest grid strategy
```

### Source: Bitrue.com — OpenClaw Trading Bot Review
""",
},

# ─── 41. TRINITY SCANNER ──────────────────────────────────────────────────

"bw-trinity-scanner": {
    "name": "bw-trinity-scanner",
    "description": "Trinity, Panic, and 2B Reversal strategies from OpenClaw Financial Intelligence. AI-enhanced scanning with trend following, mean reversion, and swing failure patterns.",
    "metadata": {
        "openclaw": {
            "emoji": "🔺",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Trinity Strategy Scanner

Three complementary scanning strategies with AI context enhancement.

### Strategy 1: TRINITY (Trend Following)
- **Signal**: Price above EMA50 with momentum confirmation
- **Entry**: Pullback to EMA50 support in uptrend
- **Exit**: Price closes below EMA50 (trend broken)
- **AI Context**: Google Gemini analyzes news + DuckDuckGo for confirming/contradicting evidence
- **Best For**: Strong trending markets

### Strategy 2: PANIC (Mean Reversion)
- **Signal**: RSI < 30 AND price below lower Bollinger Band
- **Entry**: Extreme oversold with volume spike (capitulation signal)
- **Exit**: RSI crosses above 50 or price reaches middle Bollinger Band
- **AI Context**: Sentiment analysis confirms panic selling (not fundamental collapse)
- **Best For**: Volatile markets with overreaction patterns

### Strategy 3: 2B REVERSAL (Swing Failure)
- **Signal**: New high/low that fails to hold → reversal
- **Entry**: Price makes new high but closes below previous high (distribution)
- **Exit**: Previous swing low (or previous swing high for shorts)
- **AI Context**: Volume analysis confirms distribution / accumulation
- **Best For**: Range-bound markets with failed breakouts

### Risk Management (Per Strategy)
- Kelly Criterion position sizing based on historical win rate
- VaR (Value at Risk) calculation per position
- ATR Trailing Stops that adapt to volatility
- Ladder scaling: 50% profit at TP1, remainder on trailing stop

### Paper Trading Integration
- 3-year backtesting with parameter tuning
- Fractional shares support
- Spread simulation for realistic fills
- Tax estimation for short-term capital gains

### Commands
```
/bw-trinity scan                  — Run Trinity trend scan
/bw-panic scan                    — Run Panic mean reversion scan
/bw-2b scan                      — Run 2B Reversal scan
/bw-trinity all                   — All three strategies combined
/bw-trinity backtest days=90      — Backtest Trinity strategy
/bw-trinity performance           — Strategy performance comparison
```

### Source: github.com/rayandcherry/OpenClaw-financial-intelligence
""",
},

# ─── 42. BACKTESTER ───────────────────────────────────────────────────────

"bw-backtester": {
    "name": "bw-backtester",
    "description": "Multi-strategy backtesting engine. Walk-forward testing across bull/bear/sideways regimes with Monte Carlo simulation and strategy optimization.",
    "metadata": {
        "openclaw": {
            "emoji": "⏪",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Backtesting Engine

Comprehensive strategy backtesting with walk-forward validation.

### Backtesting Modes
1. **Simple Backtest**: Strategy on historical data with fixed parameters
2. **Walk-Forward**: Rolling in-sample optimization → out-of-sample validation
3. **Monte Carlo**: Random resampling of trades to estimate outcome distributions
4. **Multi-Regime**: Test across bull, bear, and sideways market conditions separately
5. **Stress Test**: Apply 2007/2008, 2020 COVID, 2022 crypto crash conditions

### Metrics Generated
| Metric | Description |
|--------|-------------|
| Total Return | Absolute P&L |
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Risk-adjusted return (target > 1.5) |
| Sortino Ratio | Downside deviation only |
| Max Drawdown | Worst peak-to-trough decline |
| Win Rate | % of profitable trades |
| Profit Factor | Gross profit / gross loss |
| Avg Win/Loss | Average winning vs losing trade |
| Kelly Fraction | Optimal position sizing |
| Calmar Ratio | CAGR / Max Drawdown |

### Data Sources
- Yahoo Finance (free, daily)
- Binance (crypto, 1m-1d candles)
- Polygon.io (US equities, 1m)
- Alpha Vantage (free tier, daily)

### Walk-Forward Validation
```
[=== In-Sample (Train) ===][== Out-of-Sample (Test) ==]
[=== Window 1 ===][=== Test 1 ===]
    [=== Window 2 ===][=== Test 2 ===]
        [=== Window 3 ===][=== Test 3 ===]
```

### Commands
```
/bw-backtest strategy=TRINITY asset=BTC days=365    — Backtest Trinity on BTC
/bw-backtest walk-forward strategy=PANIC days=720   — Walk-forward test
/bw-backtest monte-carlo n=10000                    — Monte Carlo simulation
/bw-backtest compare strategies=TRINITY,PANIC,2B    — Compare strategies
/bw-backtest regime-test strategy=DCA               — Multi-regime testing
/bw-backtest stress-test scenario=2008              — 2008 crisis simulation
/bw-backtest optimize strategy=GRID                  — Parameter optimization
/bw-backtest report                                  — Generate full backtest report
```

### Source: Combined research — Bitrue, rayandcherry, Intellectia
""",
},

# ─── 43. TRADE JOURNAL ────────────────────────────────────────────────────

"bw-trade-journal": {
    "name": "bw-trade-journal",
    "description": "Automated trade journaling with entry/exit logging, rationale capture, performance analytics, and psychology tracking. Jonny Bravo methodology integration.",
    "metadata": {
        "openclaw": {
            "emoji": "📓",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Trade Journal

Every trade documented with full context, from signal to outcome.

### Journal Entry Structure
```
┌─────────────────────────────────────────┐
│ TRADE #1234 — BTC/USD LONG             │
├─────────────────────────────────────────┤
│ Date: 2026-02-28 09:15 MT              │
│ Strategy: TRINITY (EMA50 bounce)       │
│ Signal Source: bw-trinity-scanner       │
│ Conviction: HIGH (8/10)                │
│                                         │
│ Entry: $97,250 | Size: 0.1 BTC         │
│ Stop Loss: $95,800 (-1.5%)             │
│ Target 1: $99,000 (+1.8%)              │
│ Target 2: $101,500 (+4.4%)             │
│                                         │
│ Rationale: EMA50 support held after    │
│ pullback, volume confirmed, AI context │
│ bullish from Gemini news analysis      │
│                                         │
│ Outcome: TP1 HIT +$175 (+1.8%)        │
│ Remaining: Trailing stop at $98,500    │
│                                         │
│ Psychology: Confident entry, managed   │
│ risk well, no FOMO on exit             │
│ Grade: A                                │
└─────────────────────────────────────────┘
```

### Analytics Dashboard
- Win rate by strategy, asset, time of day, day of week
- Average R-multiple (actual vs planned risk/reward)
- Streak analysis (win/loss streaks and response)
- Psychology heat map (emotional state vs trade quality)
- Strategy attribution (which strategies make money)
- Time-in-trade analysis (holding period optimization)

### Commands
```
/bw-journal log                    — Log a new trade entry
/bw-journal update <id>            — Update trade (exit, notes)
/bw-journal review today           — Today's trade review
/bw-journal analytics 30d          — 30-day performance analytics
/bw-journal psychology             — Psychology pattern analysis
/bw-journal strategies             — Strategy performance breakdown
/bw-journal export                 — Export journal as CSV/JSON
/bw-journal grade                  — Grade recent trades (A/B/C/D/F)
```

### Source: Jonny Bravo Division + rayandcherry research
""",
},

# ─── 44. API COST GUARD ───────────────────────────────────────────────────

"bw-api-cost-guard": {
    "name": "bw-api-cost-guard",
    "description": "API spending monitor and cost guard. Hard limits on LLM API costs with real-time tracking, burn rate alerts, and per-provider daily/monthly caps.",
    "metadata": {
        "openclaw": {
            "emoji": "💳",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
            "always": True,
        }
    },
    "instructions": """
## BARREN WUFFET API Cost Guard

Prevents runaway API costs. A simple heartbeat can burn $750/month.

### The Problem (Real Case)
- Benjamin De Kraker: $20 overnight from a time-checking heartbeat
- 120,000 tokens per context check × $0.75 each
- Projected: $750/month for a simple time reminder
- Runaway cron jobs can drain API budgets in hours

### Cost Tracking
| Provider | Cost/1K Input | Cost/1K Output | Daily Cap | Monthly Cap |
|----------|--------------|----------------|-----------|-------------|
| Anthropic (Claude) | $0.003 | $0.015 | $10 | $150 |
| OpenAI (GPT-4) | $0.01 | $0.03 | $10 | $150 |
| Google (Gemini) | $0.0005 | $0.0015 | $5 | $75 |
| xAI (Grok) | $0.005 | $0.015 | $5 | $75 |

### Alert Thresholds
- 50% of daily cap → INFO alert
- 75% of daily cap → WARNING alert
- 90% of daily cap → CRITICAL alert → reduce call frequency
- 100% of daily cap → HALT all non-essential API calls

### Cost Optimization Strategies
- Cache frequent queries (market prices, static data)
- Use smaller models for routine tasks (haiku/flash for monitoring)
- Batch multiple queries into single context windows
- Reduce context window size where possible
- Use local models for non-critical operations

### Commands
```
/bw-costs today                   — Today's API spending
/bw-costs monthly                 — Month-to-date spending
/bw-costs breakdown               — Spending by provider
/bw-costs limits                  — View/set spending limits
/bw-costs burn-rate               — Current burn rate projection
/bw-costs optimize                — Cost optimization suggestions
/bw-costs history                 — Spending history chart
```

### Source: adversa.ai — API cost warnings
""",
},

# ─── 45. GRADUATED MODE ───────────────────────────────────────────────────

"bw-graduated-mode": {
    "name": "bw-graduated-mode",
    "description": "Graduated trading permission system. Progress from monitor-only to full automation through 4 permission levels with safety checks at each stage.",
    "metadata": {
        "openclaw": {
            "emoji": "🎓",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Graduated Permission System

Progressive trading automation with safety at every level.

### Permission Levels
```
Level 1: MONITOR   → View-only: signals, analysis, alerts (NO trades)
Level 2: PAPER     → Paper trading: virtual trades, real signals, no capital risk
Level 3: SMALL     → Small positions: max 1-2% of portfolio per trade
Level 4: FULL      → Full automation: all strategies active, doctrine-limited
```

### Level Requirements
| Level | Requirement | Duration |
|-------|-------------|----------|
| MONITOR → PAPER | Understanding of risk | Immediate |
| PAPER → SMALL | 30 days paper trading, >50% win rate | 30 days min |
| SMALL → FULL | 90 days, positive Sharpe, no doctrine violations | 90 days min |
| Emergency Downgrade | Any doctrine HALT event | Immediate |

### What Each Level Allows
**MONITOR**:
- Real-time market intelligence & analysis
- Morning briefings and alerts
- Research and DD feeds
- Portfolio viewing (if connected)

**PAPER**:
- Everything in MONITOR +
- Virtual trades with paper capital ($10K starting)
- Strategy backtesting and optimization
- Performance tracking and journaling

**SMALL**:
- Everything in PAPER +
- Live trades with position limits (1-2% per trade)
- 5 simultaneous positions max
- Mandatory stop-losses on all positions
- Daily P&L limits enforced

**FULL**:
- Everything in SMALL +
- All 50 strategies available
- Position limits per Doctrine Pack 2
- Full Kelly Criterion sizing
- Automated hedging and rebalancing

### Commands
```
/bw-mode status                   — Current permission level
/bw-mode set monitor              — Set to monitor-only
/bw-mode set paper                — Set to paper trading
/bw-mode set small                — Request small positions (requires approval)
/bw-mode set full                 — Request full automation (requires approval)
/bw-mode history                  — Permission level history
/bw-mode requirements             — Requirements for next level
```

### Source: Intellectia.ai — Graduated Permission System
""",
},

# ─── 46. YIELD OPTIMIZER ──────────────────────────────────────────────────

"bw-yield-optimizer": {
    "name": "bw-yield-optimizer",
    "description": "DeFi yield farming optimization across protocols. Auto-compounding, APY comparison, rug-pull detection, impermanent loss calculation, and cross-chain yield harvesting.",
    "metadata": {
        "openclaw": {
            "emoji": "🌾",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET DeFi Yield Optimizer

Maximize DeFi yields with safety-first optimization.

### Data Sources
- **DeFiLlama API**: TVL, yields, protocol data across all chains
- **Aave**: Lending rates, utilization, liquidation data
- **Compound**: Supply/borrow rates, COMP rewards
- **Uniswap/Curve**: LP yields, volume-based fees
- **Yearn/Beefy**: Vault APYs, auto-compounding strategies
- **Lido/Rocket Pool**: Liquid staking yields

### Yield Types Tracked
| Type | Risk Level | Typical APY |
|------|-----------|-------------|
| Staking (ETH) | Low | 3-5% |
| Lending (USDC) | Low-Med | 2-8% |
| LP (Stable pairs) | Medium | 5-15% |
| LP (Volatile pairs) | High | 10-100%+ |
| Leveraged Farming | Very High | 20-200%+ |
| Points Farming | Speculative | Unknown |

### Safety Checks
- ✅ Protocol audit status (Certik, Trail of Bits, OpenZeppelin)
- ✅ TVL trend (growing vs declining)
- ✅ Impermanent loss calculator for LP positions
- ✅ Smart contract risk score
- ✅ Team doxx status and track record
- ✅ Token emission schedule (inflation risk)

### Auto-Compound Schedule
```yaml
Hourly: Check for profitable compound (gas vs rewards)
4-Hourly: Rebalance across highest-yield pools
Daily: Full portfolio yield optimization pass
Weekly: Risk reassessment and protocol audit check
```

### Commands
```
/bw-yield top                     — Top yields across all DeFi
/bw-yield compare USDC            — Compare USDC yields across protocols
/bw-yield il-calc ETH/USDC        — Impermanent loss calculator
/bw-yield safety <protocol>       — Safety score for a protocol
/bw-yield portfolio                — Current yield portfolio status
/bw-yield optimize                — Optimize yield allocation
/bw-yield compound                — Trigger manual compound
/bw-yield history                 — Yield farming history & returns
```

### Source: Aurpay.net — DeFi Yield Farming Automation
""",
},

# ─── 47. ON-CHAIN FORENSICS ───────────────────────────────────────────────

"bw-onchain-forensics": {
    "name": "bw-onchain-forensics",
    "description": "On-chain contract and wallet forensics. Smart contract analysis, whale wallet tracking, token verification, rug-pull detection, and transaction flow mapping.",
    "metadata": {
        "openclaw": {
            "emoji": "🔬",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET On-Chain Forensics

Deep on-chain analysis for contract verification and whale tracking.

### Tools & APIs
- **Etherscan/BSCScan/SolScan**: Contract verification, transaction history
- **Dune Analytics**: Custom SQL queries on blockchain data
- **Nansen/Arkham**: Wallet labeling and smart money tracking
- **DeBank**: Multi-chain portfolio tracking
- **Solana RPC**: Direct Solana program account queries

### Analysis Capabilities
1. **Contract Verification**
   - Source code audit (verified vs unverified)
   - Honeypot detection (can you sell?)
   - Owner privileges analysis (mint, pause, blacklist)
   - Proxy pattern detection (upgradeable = risky)

2. **Token Analysis**
   - Top holder concentration (>50% in few wallets = danger)
   - Liquidity lock verification (duration, LP burned?)
   - Transfer tax detection (buy/sell tax inspection)
   - Similar token scam detection (name/symbol clones)

3. **Wallet Forensics**
   - Transaction pattern analysis (bot vs human)
   - Related wallet clustering (same deployer)
   - Fund flow tracing (mixer usage, exchange deposits)
   - Serial scammer identification (wallet reuse patterns)

4. **Whale Tracking**
   - 1K+ BTC wallet movements
   - 10K+ ETH wallet activity
   - Smart money wallet labels (known funds, VCs)
   - Exchange inflow/outflow patterns

### Commands
```
/bw-forensics contract <address>  — Contract security analysis
/bw-forensics token <address>     — Token safety check
/bw-forensics wallet <address>    — Wallet forensics report
/bw-forensics whale-alert         — Real-time whale movements
/bw-forensics trace <tx_hash>     — Transaction flow tracing
/bw-forensics scam-check <addr>   — Known scam database lookup
/bw-forensics dune <query>        — Run Dune Analytics query
```

### Source: Aurpay.net + adversa.ai research
""",
},

# ─── 48. SENTIMENT ENGINE ─────────────────────────────────────────────────

"bw-sentiment-engine": {
    "name": "bw-sentiment-engine",
    "description": "NLP-powered multi-source sentiment analysis. Aggregate sentiment from X/Twitter, Reddit, Telegram, Discord, and news with real-time scoring and alert triggers.",
    "metadata": {
        "openclaw": {
            "emoji": "🎭",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Sentiment Engine

Multi-source NLP sentiment analysis for financial markets.

### Data Sources
| Source | Method | Update Frequency |
|--------|--------|-----------------|
| X/Twitter | API + scraping | Real-time |
| Reddit | API (r/wallstreetbets, r/cryptocurrency, r/Superstonk) | Every 5 min |
| Telegram | Channel monitoring | Real-time |
| Discord | Server monitoring | Real-time |
| Financial News | RSS + scraping | Every 15 min |
| SEC Filings | EDGAR API | On filing |
| Analyst Ratings | Aggregator APIs | Daily |

### NLP Pipeline
```
Raw Text → Preprocessing → NLTK/spaCy Tokenization → Sentiment Model → Score
```
- **VADER**: Fast rule-based sentiment (social media optimized)
- **FinBERT**: Finance-specific transformer model for financial text
- **Custom LLM**: Claude/Gemini for nuanced context understanding

### Sentiment Metrics
- **Sentiment Score**: -1.0 (extreme bearish) to +1.0 (extreme bullish)
- **Velocity**: Rate of change in sentiment (momentum)
- **Volume**: Number of mentions per unit time
- **Divergence**: Price direction vs sentiment direction (contrarian signal)
- **Influencer Weight**: Sentiment weighted by account influence/reach

### Alert Triggers
- Sentiment reversal (bull → bear or vice versa)
- Volume spike (>3x average mentions)
- Influencer cascade (3+ major accounts shifting sentiment)
- Divergence alert (price up + sentiment down = warning)

### Commands
```
/bw-sentiment BTC                 — Bitcoin sentiment dashboard
/bw-sentiment asset=ETH source=reddit — Reddit ETH sentiment
/bw-sentiment trending             — Trending sentiment shifts
/bw-sentiment whale-posts          — High-influence account activity
/bw-sentiment divergence           — Price-sentiment divergence scanner
/bw-sentiment velocity top=10     — Fastest-moving sentiment assets
/bw-sentiment alerts               — Active sentiment alerts
/bw-sentiment history BTC 7d       — 7-day sentiment history
```

### Source: Aurpay.net NLP patterns + serif.ai News & Sentiment workflows
""",
},

# ─── 49. SEC MONITOR ──────────────────────────────────────────────────────

"bw-sec-monitor": {
    "name": "bw-sec-monitor",
    "description": "SEC filing monitor and insider trading tracker. Real-time alerts on 10-K, 10-Q, 8-K filings, Form 4 insider transactions, 13F institutional holdings, and EDGAR searches.",
    "metadata": {
        "openclaw": {
            "emoji": "🏛️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET SEC Filing Monitor

Real-time monitoring of SEC EDGAR filings and insider transactions.

### Filing Types Monitored
| Filing | Purpose | Signal Value |
|--------|---------|-------------|
| 10-K | Annual report | Fundamental analysis |
| 10-Q | Quarterly report | Earnings quality |
| 8-K | Material events | Breaking news |
| Form 4 | Insider trading | Smart money signal |
| 13F | Institutional holdings | Big money positioning |
| S-1 | IPO registration | New listings |
| SC 13D | Activist holdings (>5%) | Activist campaigns |
| DEF 14A | Proxy statement | Governance changes |

### Insider Trading Intelligence (Form 4)
- Directors, CEOs, CFOs, Officers buying/selling
- Cluster buys (multiple insiders buying = bullish)
- Unusual transaction sizes vs historical pattern
- Open market purchases vs option exercises
- Source: `openinsider` ClawHub skill integration

### 13F Institutional Analysis
- Quarterly portfolio changes for hedge funds
- New positions, increased/decreased stakes, exits
- Sector rotation patterns across institutions
- Notable investors: Buffett, Ackman, Burry, Dalio tracking

### Commands
```
/bw-sec filings <ticker>          — Recent SEC filings for a company
/bw-sec insider <ticker>          — Insider trading activity
/bw-sec 13f <fund>                — Institutional 13F holdings
/bw-sec alerts setup              — Configure filing alerts
/bw-sec cluster-buys              — Insider cluster buy scanner
/bw-sec activist                  — Activist investor campaigns
/bw-sec ipo-pipeline              — Upcoming IPOs (S-1 tracker)
/bw-sec edgar <search>            — Full-text EDGAR search
```

### Source: serif.ai/openclaw/finance + openinsider skill
""",
},

# ─── 50. EARNINGS ENGINE ──────────────────────────────────────────────────

"bw-earnings-engine": {
    "name": "bw-earnings-engine",
    "description": "Earnings calendar monitoring with automated pre-earnings analysis, post-earnings summary, surprise tracking, and one-shot cron job scheduling per earnings date.",
    "metadata": {
        "openclaw": {
            "emoji": "📅",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Earnings Engine

Comprehensive earnings season management via @barrenwuffet069bot.

### Earnings Workflow
```
Sunday Preview → User Picks → Schedule One-Shots → Pre-Earnings Report → 
Post-Earnings Auto-Summary → Options Impact Analysis → Strategy Adjustment
```

### Weekly Sunday Preview
Every Sunday at 7 PM MT, scan upcoming week's earnings:
- Companies reporting (with market cap filter)
- Expected EPS vs consensus
- Recent analyst revisions
- Historical surprise rate
- Implied move from options pricing

### One-Shot Cron Jobs
Schedule automated tasks per earnings date:
```yaml
# Example: NVDA reporting Feb 26
- 4h before: Pre-earnings briefing (recent news, analyst sentiment, options setup)
- At release: Auto-fetch results (EPS, revenue, guidance)
- 1h after: Post-earnings analysis (beat/miss, stock reaction, guidance impact)
- Next morning: Follow-up analysis (analyst revisions, price target changes)
```

### Earnings Metrics
| Metric | Description |
|--------|-------------|
| EPS Surprise | Actual vs consensus EPS |
| Revenue Surprise | Actual vs expected revenue |
| Guidance | Forward outlook vs expectations |
| AI Highlights | Key mentions of AI, capex, margins |
| Options Implied Move | Expected move from straddle pricing |
| Actual Move | Post-earnings price change |
| Earnings Drift | Post-earnings drift pattern (PEAD) |

### Commands
```
/bw-earnings week                  — This week's earnings calendar
/bw-earnings track NVDA,AAPL       — Add to earnings watchlist
/bw-earnings pre NVDA              — Pre-earnings analysis
/bw-earnings post NVDA             — Post-earnings summary
/bw-earnings surprise history      — Historical surprise data
/bw-earnings options NVDA          — Earnings options strategy
/bw-earnings calendar monthly      — Full monthly calendar
```

### Source: serif.ai — Earnings Calendar Monitoring workflow
""",
},

# ─── 51. SCAM DETECTOR ────────────────────────────────────────────────────

"bw-scam-detector": {
    "name": "bw-scam-detector",
    "description": "Crypto scam detection engine. FrankenClaw-pattern detection, rug-pull analysis, pump-and-dump identification, phishing defense, and token legitimacy verification.",
    "metadata": {
        "openclaw": {
            "emoji": "🚫",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
            "always": True,
        }
    },
    "instructions": """
## BARREN WUFFET Scam Detection Engine

Always-on protection against crypto scams, fraud, and social engineering.

### Known Scam Tokens (PERMANENT BLACKLIST)
- ❌ $CLAWD — rug-pull
- ❌ $OPENCLAW — rug-pull
- ❌ $FCLAW (FrankenClaw) — pump-and-dump ($2.3M extracted)
- ❌ $MOLT — mixed signals (some fakes)
- ❌ $CLAWD (Solana) — reached $16M market cap before crash
- ✅ RULE: **NO official OpenClaw token exists** — funded by grants/donations only

### Detection Patterns

**Rug Pull Detection**:
- Liquidity not locked or lock expiring soon
- Contract not renounced (owner can mint/pause)
- Top 10 wallets hold >60% of supply
- Developer wallet actively selling

**Pump-and-Dump Detection**:
- Coordinated social media campaign (paid influencers)
- Telegram group banning critics
- Volume spike without organic growth
- Team holding large allocation pre-promotion

**Phishing Defense**:
- URL similarity matching (typosquatting detection)
- Fake website detection (modified branding)
- Impersonation account detection (fake verified badges)
- Email/DM scam pattern matching

**Red Flag Score (0-100)**:
- 0-20: LOW RISK (probably safe)
- 21-50: MODERATE RISK (proceed with caution)
- 51-80: HIGH RISK (avoid or micro-position only)
- 81-100: CONFIRMED SCAM (do not interact)

### Commands
```
/bw-scam check <token>            — Full scam analysis on a token
/bw-scam blacklist                — View permanent blacklist
/bw-scam report <token>           — Report a suspected scam
/bw-scam alerts                   — Active scam alerts in ecosystem
/bw-scam url-check <url>          — Check URL for phishing
/bw-scam brand-monitor            — AAC/BARREN WUFFET impersonation scan
/bw-scam red-flags                — Universal red flag checklist
```

### Source: OpenClaws.io FrankenClaw report + Aurpay scam alerts
""",
},

# ─── 52. WEBSOCKET FEEDS ──────────────────────────────────────────────────

"bw-websocket-feeds": {
    "name": "bw-websocket-feeds",
    "description": "Real-time WebSocket price feeds from Binance, Coinbase, Kraken, and DEX aggregators. Sub-second market data for all trading strategies.",
    "metadata": {
        "openclaw": {
            "emoji": "📡",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET WebSocket Price Feeds

Real-time market data infrastructure for all AAC trading strategies.

### Connected Exchanges
| Exchange | Protocol | Pairs | Latency |
|----------|----------|-------|---------|
| Binance | WebSocket | 500+ | <50ms |
| Coinbase | WebSocket | 300+ | <100ms |
| Kraken | WebSocket | 200+ | <100ms |
| Bybit | WebSocket | 400+ | <50ms |
| Polymarket | REST/WS | All | <200ms |

### Data Streams
- **Trade Stream**: Individual trades with price, volume, side
- **Order Book**: L2 depth updates (bids/asks top 20)
- **Kline/Candles**: 1s, 1m, 5m, 15m, 1h, 4h, 1d candles
- **Ticker**: 24h rolling stats (volume, high, low, change)
- **Funding Rates**: Perpetual swap funding rate updates

### Architecture
```
Exchange WebSocket → Message Queue → Price Aggregator → Strategy Engine → Signals
                                   → Arbitrage Detector
                                   → Risk Monitor
                                   → Dashboard
```

### Connection Management
- Auto-reconnect on disconnection with exponential backoff
- Heartbeat monitoring (ping/pong every 30s)
- Multi-exchange failover (if Binance drops, switch to Coinbase)
- Rate limit awareness and throttling

### Commands
```
/bw-feeds status                  — WebSocket connection status
/bw-feeds latency                 — Current latency per exchange
/bw-feeds subscribe BTC,ETH       — Subscribe to additional pairs
/bw-feeds unsubscribe DOGE        — Remove pair subscription
/bw-feeds orderbook BTC depth=20  — Live order book snapshot
/bw-feeds trades BTC last=50      — Recent trades stream
/bw-feeds funding                 — Funding rate comparison
```

### Source: theworldmag.com — Polymarket bot WebSocket architecture
""",
},

# ─── 53. KELLY CRITERION ──────────────────────────────────────────────────

"bw-kelly-criterion": {
    "name": "bw-kelly-criterion",
    "description": "Kelly Criterion position sizing engine. Optimal bet sizing that maximizes geometric growth rate based on win probability and payoff ratio.",
    "metadata": {
        "openclaw": {
            "emoji": "📏",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Kelly Criterion Position Sizing

Mathematically optimal position sizing for maximum geometric growth.

### The Kelly Formula
```
f* = (bp - q) / b

Where:
  f* = fraction of capital to bet
  b  = net odds (win amount / loss amount)
  p  = probability of winning
  q  = probability of losing (1 - p)
```

### Kelly Variants
| Variant | Formula | Use Case |
|---------|---------|----------|
| Full Kelly | f* | Maximum growth (high volatility) |
| Half Kelly | f*/2 | Reduced volatility (recommended) |
| Quarter Kelly | f*/4 | Conservative (beginners) |
| Fractional Kelly | f*/n | Adjustable risk appetite |

### Multi-Asset Kelly
For portfolios with multiple simultaneous positions:
- Covariance-adjusted Kelly for correlated assets
- Maximum total exposure capped at 100% (no leverage beyond 1x by default)
- Individual position caps (max 10% per position in NORMAL state)

### Integration with Doctrine
| Doctrine State | Kelly Fraction | Max Position |
|---------------|---------------|-------------|
| NORMAL | Half Kelly | 10% |
| CAUTION | Quarter Kelly | 5% |
| SAFE_MODE | Eighth Kelly | 2% |
| HALT | Zero | 0% |

### Commands
```
/bw-kelly calculate win=65% rr=2.0        — Calculate Kelly fraction
/bw-kelly portfolio                        — Portfolio-level Kelly analysis
/bw-kelly size strategy=TRINITY capital=50K — Position size for strategy
/bw-kelly optimize                         — Optimize across all strategies
/bw-kelly history                          — Historical sizing performance
/bw-kelly compare full vs half             — Compare Kelly variants
```

### Source: rayandcherry/OpenClaw-financial-intelligence
""",
},

# ─── 54. VAR CALCULATOR ───────────────────────────────────────────────────

"bw-var-calculator": {
    "name": "bw-var-calculator",
    "description": "Value at Risk (VaR) calculator with historical, parametric, and Monte Carlo methods. Portfolio-level and per-position risk quantification.",
    "metadata": {
        "openclaw": {
            "emoji": "📉",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Value at Risk Calculator

Quantify potential losses at specified confidence levels.

### VaR Methods
1. **Historical VaR**: Use actual historical return distribution
   - Non-parametric, captures fat tails
   - Requires sufficient historical data (>252 days)

2. **Parametric VaR**: Assume normal distribution
   - VaR = μ - σ × Z_α (where Z_95% = 1.645, Z_99% = 2.326)
   - Fast but may underestimate tail risk

3. **Monte Carlo VaR**: Simulate 10,000+ scenarios
   - Most flexible, handles complex portfolios
   - Can incorporate skewness and kurtosis

### Output
```
┌─────────────────────────────────────────┐
│ VALUE AT RISK REPORT                    │
├─────────────────────────────────────────┤
│ Portfolio Value:  $100,000              │
│                                         │
│ 1-Day VaR (95%): $2,150   (-2.15%)     │
│ 1-Day VaR (99%): $3,420   (-3.42%)     │
│ 10-Day VaR (95%): $6,800  (-6.80%)     │
│                                         │
│ Expected Shortfall (CVaR 95%): $3,100   │
│ Maximum Drawdown (252d): -15.3%         │
│                                         │
│ Risk Contribution by Asset:             │
│ BTC:  45% | ETH: 25% | SPY: 20% | XRP: 10% │
└─────────────────────────────────────────┘
```

### Commands
```
/bw-var portfolio                  — Full portfolio VaR analysis
/bw-var position BTC               — Per-position VaR
/bw-var stress 2008                — Stress test (2008 scenario)
/bw-var monte-carlo n=10000        — Monte Carlo VaR simulation
/bw-var historical days=252        — Historical VaR (1 year)
/bw-var cvar                       — Conditional VaR (Expected Shortfall)
/bw-var contribution               — Risk contribution by asset
```

### Source: rayandcherry research + standard risk management
""",
},

# ─── 55. TAX HARVESTER ────────────────────────────────────────────────────

"bw-tax-harvester": {
    "name": "bw-tax-harvester",
    "description": "Tax-loss harvesting automation with wash-sale rule awareness, replacement security selection, and cross-jurisdiction tax optimization (Canada/Uruguay).",
    "metadata": {
        "openclaw": {
            "emoji": "🌿",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Tax-Loss Harvesting Engine

Automated tax optimization through strategic loss realization.

### Tax-Loss Harvesting Process
```
Scan for Unrealized Losses → Check Wash Sale Window (30 days) →
Identify Replacement Securities → Calculate Tax Savings →
Execute Harvest → Track Cost Basis → Report
```

### Wash Sale Rules
**US (IRS)**:
- Cannot repurchase substantially identical security within 30 days (before OR after sale)
- Applies across all accounts (including spouse)
- Disallowed loss added to cost basis of replacement

**Canada (CRA)**:
- Superficial loss rule: 30-day window (before AND after)
- Applies to taxpayer, spouse, and affiliated persons
- Loss denied, added to cost basis of replacement
- Capital gains 50% inclusion rate

**Uruguay (DGI)**:
- More favorable tax treatment for international income
- Territorial taxation — only Uruguay-source income taxed
- Crypto-specific rules evolving

### Replacement Security Selection
When harvesting a loss, replace with correlated but "not identical" security:
- SPY → VOO or IVV (S&P 500 alternatives)
- BTC → WBTC or BTC ETF (maintain exposure, different wrapper)
- ETH → Ethereum L2 tokens (maintain sector exposure)
- Individual stocks → Sector ETF (maintain market exposure)

### Commands
```
/bw-tax scan                       — Scan for harvesting opportunities
/bw-tax harvest <position>         — Execute tax-loss harvest
/bw-tax wash-check <security>      — Check wash sale window
/bw-tax savings-estimate           — Projected tax savings
/bw-tax replacement <security>     — Suggested replacement securities
/bw-tax report annual              — Annual tax harvest report
/bw-tax jurisdiction Calgary       — Calgary-specific tax rules
/bw-tax jurisdiction Montevideo    — Montevideo-specific tax rules
```

### Source: serif.ai/financial-planning + Canadian/Uruguayan tax research
""",
},

# ─── 56. REBALANCE ALERTS ─────────────────────────────────────────────────

"bw-rebalance-alerts": {
    "name": "bw-rebalance-alerts",
    "description": "Portfolio rebalancing alerts with configurable drift tolerance bands, tax-aware rebalancing, and multi-asset class allocation monitoring.",
    "metadata": {
        "openclaw": {
            "emoji": "⚖️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Portfolio Rebalancing

Drift-based rebalancing with tax and transaction cost awareness.

### Target Allocation Example
```
BTC:     30% ± 5% band
ETH:     20% ± 5% band
SPY:     25% ± 5% band
Bonds:   15% ± 3% band
Cash:    10% ± 2% band
```

### Rebalancing Triggers
1. **Drift**: Any asset exceeds tolerance band
2. **Time**: Calendar-based (monthly, quarterly)
3. **Event**: Doctrine state change, new capital, withdrawal
4. **Threshold**: Combined portfolio drift > 10%

### Tax-Aware Rebalancing
- Prefer selling assets with smallest tax impact
- Use new contributions to rebalance (avoid selling)
- TFSA/RRSP rebalancing (tax-free, rebalance first)
- Tax-loss harvesting integrated into rebalancing

### Commands
```
/bw-rebalance status              — Current allocation vs targets
/bw-rebalance drift               — Drift analysis per asset
/bw-rebalance plan                — Proposed rebalancing trades
/bw-rebalance execute             — Execute rebalancing (graduated mode)
/bw-rebalance targets set         — Set/update target allocation
/bw-rebalance history             — Rebalancing history log
/bw-rebalance simulate            — Simulate rebalancing scenarios
```

### Source: serif.ai/openclaw/financial-planning
""",
},

# ─── 57. MARKET COMMENTARY ────────────────────────────────────────────────

"bw-market-commentary": {
    "name": "bw-market-commentary",
    "description": "AI-generated market commentary and analysis reports. Daily/weekly/monthly commentary in BARREN WUFFET voice with data-driven insights and actionable recommendations.",
    "metadata": {
        "openclaw": {
            "emoji": "📝",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Market Commentary

AI-generated financial commentary in AZ SUPREME's analytical voice.

### Commentary Types
1. **Daily Flash**: 2-3 paragraphs, key overnight moves, today's outlook
2. **Weekly Digest**: Full market review, strategy performance, upcoming events
3. **Monthly Deep Dive**: Sector rotation, macro trends, strategy adjustments
4. **Event-Driven**: Special commentary on major market events (rate decisions, crashes, rallies)

### Voice & Brand
- Data-first: Lead with numbers, then narrative
- Decisive: Clear recommendations, not hedging language
- Military precision: Subject, situation, action
- BARREN WUFFET signature: "— BARREN WUFFET, AZ SUPREME"

### Content Structure
```
MARKET INTELLIGENCE BRIEFING
Date: [timestamp]
Doctrine State: [state]
Crash Score: [XX/100]

KEY MOVES:
- BTC: [price] ([change]%) — [context]
- ETH: [price] ([change]%) — [context]
- SPY: [price] ([change]%) — [context]

SECTOR ROTATION: [analysis]
SENTIMENT: [bullish/bearish/neutral] with [score]
RISK LEVEL: [low/medium/high/extreme]

ACTION ITEMS:
1. [specific action]
2. [specific action]
3. [specific action]

— BARREN WUFFET, AZ SUPREME
```

### Commands
```
/bw-commentary daily              — Generate daily flash
/bw-commentary weekly             — Generate weekly digest
/bw-commentary monthly            — Generate monthly deep dive
/bw-commentary event <topic>      — Event-driven commentary
/bw-commentary history            — Past commentary archive
/bw-commentary tone formal        — Adjust tone (formal/casual/brief)
```

### Source: serif.ai/openclaw/financial-planning — Market Commentary Drafting
""",
},

# ─── 58. COMPLIANCE ENGINE ────────────────────────────────────────────────

"bw-compliance-engine": {
    "name": "bw-compliance-engine",
    "description": "Financial compliance documentation engine. Automated ADV filings, suitability documentation, meeting notes, audit trails, and multi-jurisdiction compliance reporting.",
    "metadata": {
        "openclaw": {
            "emoji": "📋",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Compliance Engine

Automated compliance documentation for multi-jurisdiction operations.

### Compliance Areas

**Trading Compliance**:
- Trade execution audit trails (every signal → decision → execution logged)
- Best execution documentation
- Position limit monitoring (per Doctrine Packs)
- Conflict of interest disclosure

**Regulatory Reporting**:
- ASC (Alberta) — quarterly compliance reports
- BCU (Uruguay) — periodic regulatory filings
- FINTRAC (Canada) — AML/KYC documentation
- Tax reporting (CRA, DGI) — trading activity summaries

**Operational Compliance**:
- System uptime and reliability logs
- Error rate tracking and incident reports
- Data protection and privacy compliance
- Agent behavior audit (80+ agent activity logs)

### Audit Trail Architecture
```
Action → Timestamp → Agent ID → Decision Rationale → Outcome → Hash Chain
```
Every action in AAC is recorded in a hash-chained audit log:
- Tamper-evident (each entry references previous hash)
- Compliant with `agent-audit-trail` ClawHub skill pattern
- Exportable for regulatory review

### Commands
```
/bw-compliance status              — Compliance dashboard
/bw-compliance audit-trail today   — Today's audit trail
/bw-compliance report quarterly    — Generate quarterly report
/bw-compliance gaps                — Identify compliance gaps
/bw-compliance aml-check           — AML/KYC status check
/bw-compliance regulatory-alerts   — Regulatory change alerts
/bw-compliance export <format>     — Export for regulators (PDF/CSV)
```

### Source: serif.ai/financial-planning + agent-audit-trail skill
""",
},

# ─── 59. WALLET MANAGER ───────────────────────────────────────────────────

"bw-wallet-manager": {
    "name": "bw-wallet-manager",
    "description": "Multi-chain wallet management. Track balances across EVM, Solana, Bitcoin, and exchanges. Batch transactions, bridge automation, and airdrop farming coordination.",
    "metadata": {
        "openclaw": {
            "emoji": "👛",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Wallet Manager

Multi-chain, multi-exchange wallet tracking and automation.

### Supported Chains
| Chain | RPC | Explorer |
|-------|-----|----------|
| Ethereum | Infura/Alchemy | Etherscan |
| Arbitrum | Alchemy | Arbiscan |
| Optimism | Alchemy | Optimistic Etherscan |
| Base | Alchemy | Basescan |
| Polygon | Alchemy | Polygonscan |
| Solana | Solana RPC | SolScan |
| Bitcoin | Mempool.space | Blockchain.com |

### Features
- **Balance Aggregation**: Total portfolio across all wallets/chains/exchanges
- **Transaction History**: Unified transaction log across all addresses
- **Gas Optimization**: Queue transactions for low-gas windows
- **Bridge Automation**: Cross-chain transfers when fees are optimal
- **Airdrop Coordination**: Track eligibility and claim windows
- **Security**: Read-only API keys, hardware wallet integration

### Hot/Cold Wallet Strategy
```
Hot Wallet (Trading): 20% of crypto portfolio
├── CEX accounts (Binance, Coinbase)
├── DeFi active positions
└── Bridge pending transfers

Cold Wallet (Storage): 80% of crypto portfolio
├── Hardware wallet (Ledger/Trezor)
├── Multi-sig vault
└── Paper backup
```

### Commands
```
/bw-wallet overview                — Total portfolio across all wallets
/bw-wallet chain=ethereum          — Ethereum-specific balances
/bw-wallet transactions 7d         — Recent transactions (7 days)
/bw-wallet gas-optimal             — Best time to transact (gas fee)
/bw-wallet bridge ETH→ARB 1.0     — Bridge 1 ETH to Arbitrum
/bw-wallet airdrops                — Airdrop eligibility checker
/bw-wallet security                — Wallet security audit
```

### Source: Aurpay.net — Wallet Management & Multi-Chain
""",
},

# ─── 60. PREDICTION MARKETS ───────────────────────────────────────────────

"bw-prediction-markets": {
    "name": "bw-prediction-markets",
    "description": "Generalized prediction market intelligence across Polymarket, Metaculus, Manifold, and Kalshi. Market analysis, probability calibration, and cross-platform arbitrage.",
    "metadata": {
        "openclaw": {
            "emoji": "🔮",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Prediction Market Intelligence

Multi-platform prediction market analysis and cross-platform arbitrage.

### Platforms Monitored
| Platform | Type | Regulation | Markets |
|----------|------|-----------|---------|
| Polymarket | Real money | CFTC approved | Crypto, Politics, Sports |
| Kalshi | Real money | CFTC regulated | Economics, Events |
| Metaculus | Points/Prestige | Unregulated | Science, Tech, Global |
| Manifold | Play money | Unregulated | Anything |

### Cross-Platform Arbitrage
When the same event is priced differently across platforms:
```
Example: "Will BTC exceed $100K by March 2026?"
Polymarket: YES at $0.72
Kalshi: YES at $0.68
Spread: 4 cents → Arbitrage opportunity (buy Kalshi, sell Polymarket)
```

### Probability Calibration
- Compare market probabilities to statistical models
- Track calibration over time (were 70% events right 70% of the time?)
- Identify systematically mispriced categories

### CFTC Regulatory Status (Nov 2025)
- Polymarket: CFTC no-action letter, authorized intermediated platform
- Kalshi: CFTC-regulated designated contract market (DCM)
- Legal battle: Polymarket vs Massachusetts (Feb 2026)

### Commands
```
/bw-predict trending               — Trending markets across platforms
/bw-predict arbitrage              — Cross-platform price discrepancies
/bw-predict category=crypto        — Crypto prediction markets
/bw-predict calibration            — Platform calibration analysis
/bw-predict model <event>          — AI probability estimation
/bw-predict compare <event>        — Compare pricing across platforms
/bw-predict portfolio              — Prediction market portfolio
```

### Source: theworldmag.com + Polymarket research
""",
},

# ─── 61. CCXT EXCHANGE ────────────────────────────────────────────────────

"bw-ccxt-exchange": {
    "name": "bw-ccxt-exchange",
    "description": "CCXT multi-exchange trading integration. Unified API for 100+ centralized exchanges with order management, balance tracking, and cross-exchange operations.",
    "metadata": {
        "openclaw": {
            "emoji": "🔗",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET CCXT Exchange Integration

Unified trading across 100+ exchanges via CCXT library.

### Connected Exchanges (Primary)
| Exchange | Features | Status |
|----------|----------|--------|
| Binance | Spot, Futures, Options | Active |
| Coinbase | Spot, Advanced Trade | Active |
| Kraken | Spot, Futures, Staking | Active |
| Bybit | Spot, Derivatives | Active |
| OKX | Spot, Futures, Options | Active |

### CCXT Capabilities
- **Unified API**: Same code for all exchanges
- **Market Data**: OHLCV, order books, tickers, trades
- **Trading**: Market/limit/stop orders, cancel/modify
- **Account**: Balances, positions, P&L, fees
- **Private Data**: Order history, fills, deposits/withdrawals

### Order Types Available
| Order Type | Description | Use Case |
|-----------|-------------|----------|
| Market | Execute at best price | Immediate fills |
| Limit | Execute at price or better | Patient entry |
| Stop-Loss | Trigger at price level | Risk management |
| Stop-Limit | Stop trigger + limit fill | Precise exits |
| Trailing Stop | Dynamic stop-loss | Profit protection |
| OCO | One cancels other | Bracket orders |

### Security (Non-Custodial)
- Funds remain in YOUR exchange wallet at all times
- Use read-only API keys where possible (monitoring)
- Trading API keys with IP whitelisting
- No withdrawal permissions on trading keys
- n8n proxy pattern: CCXT runs in n8n, not in agent

### Commands
```
/bw-ccxt balance                   — Balances across all exchanges
/bw-ccxt order buy BTC 0.1 market  — Place market buy
/bw-ccxt order sell ETH 1.0 limit=3500 — Place limit sell
/bw-ccxt orders open               — View open orders
/bw-ccxt history 7d                — Trade history (7 days)
/bw-ccxt fees                      — Fee comparison across exchanges
/bw-ccxt spread BTC                — BTC spread across exchanges
```

### Source: Aurpay.net — CCXT integration patterns
""",
},

# ─── 62. MILESTONE TRACKER ────────────────────────────────────────────────

"bw-milestone-tracker": {
    "name": "bw-milestone-tracker",
    "description": "Financial milestone tracking for life events. College savings, mortgage payoff, Social Security, Medicare eligibility, RMDs, retirement targets, and FI milestones.",
    "metadata": {
        "openclaw": {
            "emoji": "🏁",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Milestone Tracker

Track progress toward financial milestones and life events.

### Milestone Categories

**Savings Milestones**:
- Emergency fund (3/6/12 months)
- First $10K / $50K / $100K / $500K / $1M
- Coast FI target (never need to save again)
- Lean FI / Barista FI / Full FI

**Life Event Milestones**:
- Child education savings (RESP in Canada)
- Mortgage payoff date projection
- Car purchase/replacement fund
- Sabbatical/travel fund

**Retirement Milestones** (Canada-specific):
- CPP eligibility (age 60-70)
- OAS eligibility (age 65)
- RRSP → RRIF conversion (age 71)
- TFSA maximization target

**Trading Milestones**:
- $1K → $10K trading contest target
- First profitable month
- 100-trade profitability milestone
- Strategy-specific targets (e.g., 60% win rate sustained)

### Milestone Dashboard
```
╔══════════════════════════════════════╗
║ FINANCIAL MILESTONES                 ║
╠══════════════════════════════════════╣
║ ✅ Emergency Fund (6mo)    ACHIEVED  ║
║ 🔄 $100K Net Worth   72% [$72,000]  ║
║ 🔄 Mortgage Payoff   45% [2031 est] ║
║ ⬜ Coast FI          20% [$42,000]   ║
║ ⬜ Full FI            8% [$80,000]   ║
╚══════════════════════════════════════╝
```

### Commands
```
/bw-milestone status               — All milestones dashboard
/bw-milestone add <name> target=X  — Add new milestone
/bw-milestone update <id>          — Update progress
/bw-milestone projections          — Projected completion dates
/bw-milestone celebrate            — Recent milestone achievements
/bw-milestone trading              — Trading-specific milestones
```

### Source: serif.ai/financial-planning — Client Milestone Tracking
""",
},

# ─── 63. ESTATE PLANNER ───────────────────────────────────────────────────

"bw-estate-planner": {
    "name": "bw-estate-planner",
    "description": "Estate planning coordination for generational wealth transfer. Trust structures, beneficiary reviews, document tracking, and cross-jurisdiction estate optimization.",
    "metadata": {
        "openclaw": {
            "emoji": "🏛️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Estate Planner

Generational wealth transfer and estate planning coordination.

### Estate Structures (Canada)
- **Inter Vivos Trust**: Living trust for asset protection and tax planning
- **Testamentary Trust**: Created upon death, favorable tax treatment
- **Estate Freeze**: Lock current value, future growth to next generation
- **Alter Ego Trust**: For individuals 65+, probate avoidance
- **Joint Partner Trust**: For couples, probate avoidance
- **Holding Company (CCPC)**: Corporate structure for tax deferral

### Estate Structures (Uruguay)
- **SAU (Sociedad Anónima Uruguaya)**: Uruguayan corporation
- **Free Zone Company**: Tax-advantaged business vehicle
- **Fideicomiso**: Uruguayan trust equivalent

### Document Tracking
| Document | Status | Last Updated | Review Due |
|----------|--------|-------------|-----------|
| Will | Current | Date | Annual |
| Power of Attorney | Current | Date | Annual |
| Healthcare Directive | Current | Date | Annual |
| Trust Documents | Review needed | Date | Biannual |
| Beneficiary Forms | Current | Date | After life events |
| Insurance Policies | Current | Date | Annual |

### Beneficiary Review Triggers
- Marriage/divorce
- Birth/adoption of children
- Death of beneficiary
- Significant wealth change
- Tax law changes
- Cross-border relocation (Calgary ↔ Montevideo)

### Commands
```
/bw-estate overview                — Estate planning dashboard
/bw-estate documents               — Document status tracker
/bw-estate beneficiaries           — Beneficiary review status
/bw-estate trust-analysis          — Trust structure analysis
/bw-estate tax-impact              — Estate tax impact modeling
/bw-estate freeze                  — Estate freeze opportunity analysis
/bw-estate international           — Cross-border estate planning
```

### Source: serif.ai/financial-planning + Canadian/Uruguayan estate law
""",
},

# ─── 64. REFERRAL NETWORK ─────────────────────────────────────────────────

"bw-referral-network": {
    "name": "bw-referral-network",
    "description": "Professional referral network management. Track relationships with CPAs, attorneys, insurance agents, bankers, and other financial professionals across Calgary and Montevideo.",
    "metadata": {
        "openclaw": {
            "emoji": "🤝",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Referral Network

Professional network management for financial operations.

### Network Categories
| Category | Calgary | Montevideo |
|----------|---------|-----------|
| Tax/CPA | CRA specialists | DGI advisors |
| Legal | Securities lawyers | Fideicomiso attorneys |
| Insurance | Life/disability/P&C | International coverage |
| Banking | Big Five contacts | BCU-regulated banks |
| Real Estate | Alberta RE agents | Montevideo RE |
| Crypto Legal | CSA specialists | BCU crypto advisors |

### Relationship Tracking
- Contact frequency monitoring (quarterly touch minimum)
- Referral history (given and received)
- Specialization mapping (who helps with what)
- Engagement quality scoring
- Auto-reminder for relationship maintenance

### Commands
```
/bw-referral list                  — Full referral network
/bw-referral category=legal        — Filter by category
/bw-referral city=Calgary          — Filter by location
/bw-referral due                   — Contacts due for outreach
/bw-referral add <contact>         — Add new contact to network
/bw-referral history <contact>     — Interaction history
```

### Source: serif.ai/financial-planning — Referral Network Maintenance
""",
},

# ─── 65. INSIDER TRACKER ──────────────────────────────────────────────────

"bw-insider-tracker": {
    "name": "bw-insider-tracker",
    "description": "SEC Form 4 insider trading data aggregation. Track director, CEO, and officer buys/sells with cluster buy detection, conviction scoring, and historical pattern analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "🕵️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """
## BARREN WUFFET Insider Trading Tracker

SEC Form 4 insider transaction monitoring and signal generation.

### Insider Types
| Code | Description | Signal Weight |
|------|-------------|--------------|
| CEO | Chief Executive Officer | Highest |
| CFO | Chief Financial Officer | High |
| COO | Chief Operating Officer | High |
| Director | Board member | Medium |
| 10% Owner | Large shareholder | Medium |
| VP | Vice President | Lower |

### Signal Types
1. **Cluster Buys**: Multiple insiders buying within 30 days = strongest signal
2. **Large Open Market Purchases**: Insider buying with own money (not options)
3. **CEO Buys at 52-Week Low**: Highest conviction insider signal
4. **Insider Selling After Lock-Up**: Often routine, less meaningful
5. **Form 4 vs Form 144**: Intent to sell on schedule vs immediate sale

### Historical Performance
- Insider cluster buys outperform market by 7-10% on average (12 months)
- CEO buys at 52-week lows show 15%+ outperformance
- Routine option exercises have no predictive value

### Detection Pipeline
```
SEC EDGAR → Form 4 Parser → Transaction Classification → Signal Score → Alert
```

### Integration with openinsider ClawHub skill
Leverages the `openinsider` skill for direct SEC Form 4 data access.

### Commands
```
/bw-insider recent                 — Recent insider transactions
/bw-insider clusters               — Cluster buy detection
/bw-insider ticker=NVDA            — NVDA insider activity
/bw-insider ceo-buys               — CEO purchases (highest conviction)
/bw-insider top-buyers 30d         — Top insider buyers (30 days)
/bw-insider sector=tech            — Tech sector insider activity
/bw-insider signal-score <ticker>  — Insider signal strength
/bw-insider history <ticker> 1y    — 1-year insider activity
```

### Source: openinsider ClawHub skill + SEC EDGAR data
""",
},

# ─── 66. IRON CONDOR OPTIONS ─────────────────────────────────────────────
"bw-iron-condor": {
    "name": "BARREN WUFFET Iron Condor Options",
    "description": "Neutral options strategy: sell OTM put spread + OTM call spread for net credit in low-volatility range-bound markets.",
    "metadata": {
        "openclaw": {
            "emoji": "🦅",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Iron Condor Options Engine

Neutral options strategy for profiting from low-volatility, range-bound markets.
Combines a bull put spread + bear call spread for defined-risk income generation.

| Component | Detail |
|-----------|--------|
| Strategy | Sell OTM put spread + sell OTM call spread |
| Max Profit | Net credit received |
| Max Loss | Spread width - net credit |
| Breakeven Low | Short put strike - net credit |
| Breakeven High | Short call strike + net credit |
| Best Environment | Low volatility, range-bound, no earnings/FOMC |
| Risk-Reward | Typically 0.67 (risk $3 to make $2) |
| Greeks | Short vega, positive theta |

**Adjustments**: Bullish (move put spread closer), Bearish (move call spread closer).
Select short strikes at 15-25 delta. Execute all 4 legs simultaneously.

### Commands
```
/bw-iron-condor scan <ticker>       — Find iron condor setups
/bw-iron-condor setup <ticker> 30d  — Build 30-day iron condor
/bw-iron-condor breakevens          — Calculate breakeven points
/bw-iron-condor adjust bullish      — Bullish adjustment
/bw-iron-condor adjust bearish      — Bearish adjustment
/bw-iron-condor greeks              — Greeks analysis (theta, vega)
/bw-iron-condor roll                — Roll to next expiration
/bw-iron-condor pnl                 — Current P&L and probability
```

### Source: Investopedia Iron Condor options strategy deep dive
""",
},

# ─── 67. MONTE CARLO SIMULATOR ──────────────────────────────────────────
"bw-monte-carlo": {
    "name": "BARREN WUFFET Monte Carlo Simulator",
    "description": "Monte Carlo simulation engine for portfolio risk analysis, VaR calculation, and scenario stress testing.",
    "metadata": {
        "openclaw": {
            "emoji": "🎲",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Monte Carlo Simulation Engine

Multi-scenario probability modeling for portfolio risk assessment. Runs thousands
of randomised price paths using historical data to quantify potential outcomes.

| Parameter | Detail |
|-----------|--------|
| Method | Log-return based random walk with drift |
| Simulations | 10,000+ paths recommended |
| Distribution | Normal + Cornish-Fisher adjustment for crypto |
| Outputs | VaR (95%/99%), CVaR, probability of ruin |
| Formula | Next Price = Today × e^(Drift + σ×NORMSINV(RAND())) |
| Drift | Average Daily Return - Variance/2 |
| Applications | Options pricing, retirement planning, stress testing |

**Stress Scenarios**: 50% BTC drawdown, stablecoin de-peg, 300bp rate hike,
quantum computing breakthrough, miner capitulation.

### Commands
```
/bw-monte-carlo run <ticker> 1000    — 1000-path simulation
/bw-monte-carlo var 95               — 95% VaR calculation
/bw-monte-carlo var 99               — 99% VaR calculation
/bw-monte-carlo cvar                 — Conditional VaR (Expected Shortfall)
/bw-monte-carlo stress btc-crash     — BTC 50% drawdown scenario
/bw-monte-carlo portfolio            — Full portfolio stress test
/bw-monte-carlo retirement <years>   — Retirement probability analysis
/bw-monte-carlo cornish-fisher       — Non-normal distribution adjustment
```

### Source: Investopedia Monte Carlo Simulation + IBM 2024 AI integration research
""",
},

# ─── 68. GOLD vs BITCOIN MACRO ANALYZER ─────────────────────────────────
"bw-gold-btc-macro": {
    "name": "BARREN WUFFET Gold vs Bitcoin Macro Analyzer",
    "description": "Track the Gold-Bitcoin Great Divergence: correlation analysis, macro regime detection, and barbell portfolio strategy.",
    "metadata": {
        "openclaw": {
            "emoji": "⚖️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Gold vs Bitcoin Macro Analyzer

Track the Great Divergence (2024-2026). Gold acts as sovereignty insurance
while Bitcoin trades as leveraged tech. Correlation dropped to -0.17, providing
genuine portfolio diversification.

| Metric | Gold (Feb 2026) | Bitcoin (Feb 2026) |
|--------|----------------|-------------------|
| Price | ~$5,000/oz | $66K-$90K |
| YTD | +10-12% | Flat to -6.5% |
| Market Cap | ~$18.2T | ~$1.4-1.8T |
| Correlation to S&P | Low/Negative | High Positive (0.8) |
| Primary Driver | Geopolitical hedging | Liquidity cycles |
| Volatility | ~12-15% annual | ~60-70% annual |

**Barbell Strategy**: 10-15% Gold (ruin risk) + 2-5% BTC (FOMO risk).
**Central Bank Put**: 1,000+ tonnes/year sovereign buying = floor under gold.

### Commands
```
/bw-gold-btc correlation             — Current rolling correlation
/bw-gold-btc regime                  — Current macro regime phase
/bw-gold-btc divergence              — Divergence metrics and alerts
/bw-gold-btc barbell                 — Barbell portfolio allocation
/bw-gold-btc central-banks           — Sovereign buying activity
/bw-gold-btc etf-flows               — Gold vs BTC ETF flow comparison
/bw-gold-btc scenario 2030           — 2030 projection scenarios
/bw-gold-btc debasement              — Debasement trade metrics
```

### Source: Aurpay — The Great Divergence: Gold vs Bitcoin Correlation (2016-2026)
""",
},

# ─── 69. WEB3 SECURITY AUDITOR ──────────────────────────────────────────
"bw-web3-auditor": {
    "name": "BARREN WUFFET Web3 Security Auditor",
    "description": "Smart contract security analysis using BSA, semantic guards, invariant detection, and reentrancy analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "🛡️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Web3 Security Auditor

Pre-deployment smart contract security analysis. Immutable contracts cannot be
patched — this skill catches vulnerabilities before they become million-dollar exploits.

| Method | Purpose |
|--------|---------|
| Behavioral State Analysis (BSA) | Intent-based reasoning: what code SHOULD do vs what it DOES |
| Semantic Guard Analysis | Access control usage graph — catch missing onlyOwner |
| State Invariant Detection | Infer conservation rules (total supply = Σ balances) |
| Cross-Chain Reentrancy | Multi-contract call graphs, read-only reentrancy detection |
| Proxy & Upgrade Safety | EVM storage slot mapping across implementations |
| Constant-Time Analysis | Timing side-channels in cryptographic code (Trail of Bits) |

**Supported Proxies**: Transparent, UUPS, Beacon, Diamond (EIP-2535).
**Tools**: QuillAudits (1,500+ projects), Trail of Bits skills, Auditmos library.

### Commands
```
/bw-web3-audit contract <address>    — Full security audit
/bw-web3-audit bsa <file>            — Behavioral State Analysis
/bw-web3-audit guards                — Access control consistency check
/bw-web3-audit invariants            — State invariant detection
/bw-web3-audit reentrancy            — Cross-chain reentrancy scan
/bw-web3-audit proxy                 — Proxy upgrade safety validation
/bw-web3-audit fuzz                  — Generate Foundry fuzz tests
/bw-web3-audit timing                — Constant-time side-channel check
```

### Source: Aurpay — Claude Code Meets Web3: 20 Ways AI Is Changing Blockchain Development
""",
},

# ─── 70. DEFI PORTFOLIO RISK ENGINE ─────────────────────────────────────
"bw-defi-risk": {
    "name": "BARREN WUFFET DeFi Portfolio Risk Engine",
    "description": "DeFi-specific risk analysis with VaR, Expected Shortfall, Cornish-Fisher expansion for non-normal crypto returns.",
    "metadata": {
        "openclaw": {
            "emoji": "📊",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET DeFi Portfolio Risk Engine

Quantitative risk analysis for DeFi portfolios. Accounts for non-normal
distributions, fat tails, and correlated crashes unique to crypto markets.

| Method | Application |
|--------|-------------|
| VaR (Cornish-Fisher) | Adjusted for skewness/kurtosis in crypto returns |
| Expected Shortfall (CVaR) | Average loss beyond VaR threshold (tail risk) |
| Monte Carlo Stress Testing | Simultaneous multi-scenario simulation |
| Impermanent Loss Calculator | LP position P&L under price divergence |
| Protocol Risk Scoring | Smart contract audit + TVL + age + team score |
| Correlation Matrix | Cross-asset and cross-protocol correlation tracking |

**Stress Scenarios**: 50% BTC drawdown + stablecoin de-peg + bridge exploit.
**Rebalancing**: Cornish-Fisher adjusted risk matrices with ERC-4626 vault support.

### Commands
```
/bw-defi-risk portfolio              — Full portfolio risk report
/bw-defi-risk var <position>         — VaR with Cornish-Fisher
/bw-defi-risk cvar                   — Expected Shortfall (CVaR)
/bw-defi-risk impermanent <pool>     — Impermanent loss analysis
/bw-defi-risk protocol <name>        — Protocol risk score
/bw-defi-risk correlations           — Cross-protocol correlation matrix
/bw-defi-risk stress all             — Run all stress scenarios
/bw-defi-risk hedge                  — Recommended hedging positions
```

### Source: Aurpay Claude Code Web3 + Investopedia Monte Carlo + DeFi risk research
""",
},

# ─── 71. PREDICTION MARKET ARBITRAGE ────────────────────────────────────
"bw-prediction-arb": {
    "name": "BARREN WUFFET Prediction Market Arbitrage",
    "description": "Cross-platform prediction market arbitrage exploiting price lags between Polymarket, Binance, and other venues.",
    "metadata": {
        "openclaw": {
            "emoji": "🔮",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Prediction Market Arbitrage Engine

Exploit 30-second price lags between centralized exchanges and prediction markets.
Load YES/NO positions at low cents during spread exploitation windows.

| Metric | Detail |
|--------|--------|
| Lag Window | ~30 seconds (Binance → Polymarket) |
| Strategy | Buy YES/NO when spread > price delta |
| Win Rate | Up to 95% on short-term (15-min) options |
| Return Profile | 785% on individual windows |
| Proven P&L | -$500 → $106K documented |
| Oracle | UMA (2h undisputed resolution) |
| Execution | Polymarket CLOB API |

**Risk**: MEV competition, oracle delays, regulatory changes.
**Position sizing**: Max 1-2% portfolio per trade.

### Commands
```
/bw-prediction-arb scan               — Scan for arbitrage opportunities
/bw-prediction-arb lag                 — Current Binance-Polymarket lag
/bw-prediction-arb spread >1%         — Spreads above 1% threshold
/bw-prediction-arb execute <market>   — Execute arb trade
/bw-prediction-arb backtest           — Historical arb backtest
/bw-prediction-arb pnl                — Cumulative P&L report
/bw-prediction-arb oracle-status      — UMA oracle resolution status
/bw-prediction-arb risk               — Risk metrics and position exposure
```

### Source: Aurpay 10 OpenClaw Use Cases + theworldmag Polymarket bot analysis
""",
},

# ─── 72. SMART CONTRACT FORENSICS ───────────────────────────────────────
"bw-contract-forensics": {
    "name": "BARREN WUFFET Smart Contract Forensics",
    "description": "Transaction decoding, ABI analysis, Keccak-256 selector identification, and anomaly detection for on-chain investigations.",
    "metadata": {
        "openclaw": {
            "emoji": "🔬",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Smart Contract Forensics

On-chain transaction decoding and forensic analysis. Turn raw hexadecimal
contract data into human-readable intelligence.

| Capability | Detail |
|-----------|--------|
| Function Selector | Decode first 4 bytes (e.g., 0xa9059cbb = transfer) |
| Argument Parsing | Parse 32-byte chunks per ABI specification |
| Anomaly Detection | Flag obfuscated logic, unexpected interactions |
| Event Analysis | Decode emitted events for state change tracking |
| UCAI Integration | abi-to-mcp generate <address> for any contract |
| Multi-chain Support | Ethereum, Arbitrum, Base, BSC via QuickNode |
| Dune Analytics | Natural language → PostgreSQL for decoded blockchain data |

**Pattern**: Decode → Analyze → Flag anomalies → Generate report.

### Commands
```
/bw-forensics decode <txhash>        — Decode transaction
/bw-forensics abi <contract>         — ABI analysis and function map
/bw-forensics selector <hex>         — Identify function selector
/bw-forensics events <contract> 24h  — Event log analysis (24h)
/bw-forensics anomaly <contract>     — Anomaly detection scan
/bw-forensics wallet <address>       — Wallet activity forensics
/bw-forensics ucai <address>         — Generate MCP interface for contract
/bw-forensics dune <query>           — Natural language Dune query
```

### Source: Aurpay Claude Code Web3 — Transaction Decoding & UCAI framework
""",
},

# ─── 73. QUANTUM THREAT MONITOR ─────────────────────────────────────────
"bw-quantum-monitor": {
    "name": "BARREN WUFFET Quantum Threat Monitor",
    "description": "Monitor quantum computing threats to cryptographic assets — P2PK vulnerability tracking and post-quantum migration readiness.",
    "metadata": {
        "openclaw": {
            "emoji": "⚛️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Quantum Threat Monitor

Track quantum computing developments that threaten cryptographic security of
digital assets. Monitor migration to post-quantum cryptography standards.

| Threat | Status (2026) |
|--------|---------------|
| Shor's Algorithm | Could derive private keys from public keys |
| Vulnerable BTC | ~25% (5M+ BTC) in legacy P2PK addresses |
| Satoshi Coins | High-value P2PK target for quantum attack |
| PQC Testnets | ML-DSA (Dilithium) being tested on Bitcoin testnet |
| Price Impact | Quantum risk creates persistent discount on BTC |
| Timeline | Estimates range from 5-15 years for cryptographically relevant QC |

**Mitigation**: Move funds from P2PK to P2PKH/P2SH addresses. Monitor NIST PQC standards.

### Commands
```
/bw-quantum status                   — Current quantum threat level
/bw-quantum vulnerable <wallet>      — Check wallet for P2PK exposure
/bw-quantum btc-exposure             — Total vulnerable BTC estimate
/bw-quantum pqc-progress             — Post-quantum crypto migration status
/bw-quantum timeline                 — Estimated threat timeline
/bw-quantum migrate-plan             — Migration plan for at-risk holdings
/bw-quantum news                     — Latest quantum computing news
/bw-quantum impact                   — Price impact analysis from quantum FUD
```

### Source: Aurpay Gold vs Bitcoin — Quantum Countdown section + NIST PQC research
""",
},

# ─── 74. CENTRAL BANK GOLD TRACKER ──────────────────────────────────────
"bw-central-bank-gold": {
    "name": "BARREN WUFFET Central Bank Gold Tracker",
    "description": "Track sovereign gold accumulation, de-dollarization flows, and the central bank 'put' under gold prices.",
    "metadata": {
        "openclaw": {
            "emoji": "🏦",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Central Bank Gold Tracker

Monitor sovereign gold accumulation creating a structural "put" under gold prices.
Track the de-dollarization mechanism driving the Great Divergence.

| Central Bank | Holdings (2026) | Activity |
|-------------|-----------------|----------|
| PBoC (China) | Growing | 13 consecutive months buying |
| Poland | 543+ tonnes | 100+ tonnes added late 2025-2026 |
| India (RBI) | 876.2 tonnes | +40% over 5 years |
| Global Total | Record | 1,000+ tonnes annually |

**Key Insight**: Price-agnostic sovereign buying creates floor — whenever
gold dips, central banks accumulate. No G20 central bank accumulates BTC.
**Debasement Trade**: Gold rising WITH high rates (3.5%+) — historic anomaly.

### Commands
```
/bw-cb-gold reserves                 — Global central bank gold reserves
/bw-cb-gold buyers                   — Top sovereign buyers this quarter
/bw-cb-gold floor                    — Estimated sovereign put price floor
/bw-cb-gold dedollar                 — De-dollarization flow metrics
/bw-cb-gold anomaly                  — Gold vs real rates anomaly tracker
/bw-cb-gold etf-flows                — Gold ETF institutional flows
/bw-cb-gold forecast                 — Price forecast ($5,055 Q4 2026 JPM)
/bw-cb-gold mining                   — Mine production growth (1.5%/yr)
```

### Source: Aurpay Great Divergence — World Gold Council data, JPMorgan forecasts
""",
},

# ─── 75. YIELD FARM OPTIMIZER ────────────────────────────────────────────
"bw-yield-farm": {
    "name": "BARREN WUFFET Yield Farm Optimizer",
    "description": "Automated yield farming: auto-deposit, claim, compound, and migrate LP positions across DeFi protocols.",
    "metadata": {
        "openclaw": {
            "emoji": "🌾",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Yield Farm Optimizer

Maximize DeFi yields by automating deposit, claim, compound, and migration
workflows across protocols. Add rug-pull detection before entering any pool.

| Feature | Detail |
|---------|--------|
| Protocols | Aave, Compound, Uniswap, Curve, Convex |
| Data Source | DeFiLlama API for real-time APY data |
| Auto-Compound | Daily compounding for max returns |
| Migration | Auto-migrate LP when APR differential > 20% |
| Safety | Rug-pull detection, contract audit, liquidity lock check |
| Impermanent Loss | Real-time IL tracking with exit triggers |
| Multi-Chain | Ethereum, Arbitrum, Base, Solana, BSC |

**Risk**: Smart contract risk, impermanent loss, protocol exploits.
Always audit protocols first. Never invest more than you can afford to lose.

### Commands
```
/bw-yield-farm scan                  — Scan top yield opportunities
/bw-yield-farm deposit <pool>        — Deposit into yield pool
/bw-yield-farm compound              — Compound all pending rewards
/bw-yield-farm migrate <from> <to>   — Migrate LP to higher APY
/bw-yield-farm il-check              — Impermanent loss report
/bw-yield-farm rug-check <pool>      — Rug-pull risk assessment
/bw-yield-farm apy-compare           — Compare APY across protocols
/bw-yield-farm harvest               — Claim all pending rewards
```

### Source: Aurpay 10 OpenClaw Use Cases — Yield Farming & Liquidity Provision section
""",
},

# ─── 76. SENTIMENT ALPHA SCRAPER ────────────────────────────────────────
"bw-alpha-scraper": {
    "name": "BARREN WUFFET Sentiment Alpha Scraper",
    "description": "Social media sentiment analysis and alpha signal extraction from X, Reddit, Telegram using NLP scoring.",
    "metadata": {
        "openclaw": {
            "emoji": "📡",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Sentiment Alpha Scraper

Extract trading alpha from social media sentiment. Scrape, score, and act
on sentiment signals from X, Reddit, Telegram, and Discord.

| Source | Method | Signal |
|--------|--------|--------|
| X (Twitter) | NLTK sentiment scoring | Bullish/bearish ratio |
| Reddit | Superstonk DD methodology | Deep value conviction |
| Telegram | Alpha group monitoring | Early meme coin detection |
| Discord | Channel scraper | Community sentiment pulse |
| News | SEC filings + analyst ratings | Institutional sentiment |

**Auto-trade**: Buy if bullish sentiment > 70%. Requires human-in-loop above threshold.
**Fine-tuning**: NLP libraries (NLTK, spaCy) for accurate scoring.

### Commands
```
/bw-alpha sentiment <token>          — Multi-source sentiment score
/bw-alpha reddit <subreddit>         — Subreddit sentiment analysis
/bw-alpha twitter <topic> 24h        — X sentiment (24h window)
/bw-alpha telegram <group>           — Telegram alpha group monitor
/bw-alpha whale-social               — Whale wallet social mentions
/bw-alpha meme-detector              — Early meme coin detection
/bw-alpha news-sentiment <ticker>    — News + SEC filing sentiment
/bw-alpha auto-trade setup           — Configure sentiment-based trading
```

### Source: Aurpay OpenClaw Social Media Sentiment Analysis + Reddit Superstonk methodology
""",
},

# ─── 77. SOLIDITY GAS OPTIMIZER ──────────────────────────────────────────
"bw-gas-optimizer": {
    "name": "BARREN WUFFET Solidity Gas Optimizer",
    "description": "Smart contract gas optimization: storage packing, memory caching, unchecked blocks, and inline Yul assembly.",
    "metadata": {
        "openclaw": {
            "emoji": "⛽",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Solidity Gas Optimizer

Reduce smart contract gas costs on Ethereum mainnet where efficiency is
a financial imperative. Treats gas optimization as financial optimization.

| Technique | Savings |
|-----------|---------|
| Storage Packing | Pack variables into single 256-bit slots, reduce SSTORE/SLOAD |
| Memory Caching | Cache loop variables in memory vs re-reading storage |
| Unchecked Blocks | Skip overflow checks where logically impossible |
| Inline Yul | Low-level assembly for known-safe operations |
| Batch Operations | Combine multiple state changes into single transaction |
| Event Optimization | Use indexed params for efficient log filtering |

**Reality Check**: Claims of "40% savings" vary wildly by contract.
Audit and benchmark before/after. Never sacrifice safety for gas.

### Commands
```
/bw-gas analyze <contract>           — Full gas usage analysis
/bw-gas optimize storage             — Storage packing recommendations
/bw-gas optimize loops               — Memory caching for loops
/bw-gas unchecked                    — Safe unchecked block suggestions
/bw-gas yul                          — Inline Yul assembly opportunities
/bw-gas benchmark before-after       — Gas cost comparison
/bw-gas report                       — Full optimization report with estimates
/bw-gas deploy-estimate              — Deployment gas estimation
```

### Source: Aurpay Claude Code Web3 — Solidity Gas Optimization section
""",
},

# ─── 78. MULTISIG TREASURY MANAGER ──────────────────────────────────────
"bw-multisig-treasury": {
    "name": "BARREN WUFFET Multisig Treasury Manager",
    "description": "Safe multisig transaction proposals with zero-secret architecture — simulate, validate, and bundle treasury operations.",
    "metadata": {
        "openclaw": {
            "emoji": "🔐",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Multisig Treasury Manager

Gnosis Safe multisig transaction management with zero-secret architecture.
AI constructs payloads with algorithmic precision; human signers approve.

| Feature | Detail |
|---------|--------|
| Architecture | Zero-secret: simulates but NEVER touches private keys |
| Bundle Support | Multi-call transaction bundling (upgrade + transfer in one) |
| Format | EIP-712 typed data for human-readable signing |
| Validation | Nonce checking, threshold verification, parameter validation |
| Simulation | Full transaction simulation before proposal |
| Safety | Human-readable summary of all parameter changes |

**Critical**: Generated payload sent to Safe interface. Only human signers
with hardware wallets approve the final transaction.

### Commands
```
/bw-multisig propose <description>   — Draft multisig proposal
/bw-multisig simulate                — Simulate proposed transaction
/bw-multisig bundle <actions>        — Bundle multiple actions
/bw-multisig validate                — Validate nonces and thresholds
/bw-multisig status                  — Current proposal status
/bw-multisig signers                 — Required signers and threshold
/bw-multisig history                 — Past proposals and outcomes
/bw-multisig emergency-pause         — Emergency protocol pause proposal
```

### Source: Aurpay Claude Code Web3 — Safe Multisig Transaction Proposals section
""",
},

# ─── 79. SUBGRAPH INDEXER ───────────────────────────────────────────────
"bw-subgraph-indexer": {
    "name": "BARREN WUFFET Subgraph Indexer",
    "description": "The Graph subgraph schema design, deployment, and natural language querying of indexed on-chain data.",
    "metadata": {
        "openclaw": {
            "emoji": "📈",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Subgraph Indexer

Design, deploy, and query The Graph subgraphs for indexed on-chain data.
Natural language interface for complex blockchain data architectures.

| Feature | Detail |
|---------|--------|
| Schema Generation | ABI → optimized @entity schemas in schema.graphql |
| Best Practice | One-to-many on "one" side for query performance |
| IPFS Integration | Fetch existing deployment schemas via IPFS hashes |
| Multi-Protocol | Support for any EVM-compatible blockchain |
| Natural Language | Query aggregated indexed data conversationally |
| Neo4j Pairing | Visual relationship graphs for complex schemas |

**Workflow**: Analyze ABI → Generate entities → Configure mappings → Deploy → Query.

### Commands
```
/bw-subgraph design <contract>       — Generate subgraph schema from ABI
/bw-subgraph deploy <network>        — Deploy subgraph to hosted service
/bw-subgraph query <natural-lang>    — Natural language subgraph query
/bw-subgraph schema <ipfs-hash>      — Fetch existing subgraph schema
/bw-subgraph entities                — List all indexed entities
/bw-subgraph optimize                — Schema performance optimization
/bw-subgraph visualize               — Generate relationship graph (Neo4j)
/bw-subgraph sync-status             — Check indexing progress
```

### Source: Aurpay Claude Code Web3 — The Graph Subgraph MCP Integration section
""",
},

# ─── 80. CRYPTO SCAM SHIELD v2 ──────────────────────────────────────────
"bw-scam-shield-v2": {
    "name": "BARREN WUFFET Crypto Scam Shield v2",
    "description": "Enhanced scam detection: fake token identification, malicious repo scanning, supply chain attack prevention, and credential leak monitoring.",
    "metadata": {
        "openclaw": {
            "emoji": "🛑",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Crypto Scam Shield v2

Enhanced scam detection engine with intelligence from 2026 scam wave analysis.
Covers fake tokens, malicious repos, hijacked accounts, and supply chain attacks.

| Threat | Detection |
|--------|-----------|
| Fake Tokens | $CLAWD, $OPENCLAW, $FCLAW — pump-and-dump detection |
| Malicious Repos | Obfuscated crypto stealers in package.json/dependencies |
| Hijacked Accounts | Impersonation detection across X, Telegram, Discord |
| Supply Chain | Dependency analysis for known malicious packages |
| Credential Leaks | API key exposure scanning (1.5M keys leaked in one incident) |
| RCE via Web Pages | Malicious page targeting AI agent system access |
| WebSocket Hijacking | Cross-site WebSocket session takeover detection |

**Rules**: NO official OpenClaw/AAC token exists. Verify via official channels only.
**Response**: Auto-alert Telegram channel on threat detection.

### Commands
```
/bw-scam-shield scan <token>         — Token legitimacy analysis
/bw-scam-shield repo <url>           — Repository malware scan
/bw-scam-shield credentials          — API key exposure check
/bw-scam-shield supply-chain         — Dependency security audit
/bw-scam-shield impersonation        — Social media impersonation scan
/bw-scam-shield rug-detector <addr>  — On-chain rug-pull risk analysis
/bw-scam-shield alerts               — Current threat alerts
/bw-scam-shield report               — Generate security incident report
```

### Source: Aurpay OpenClaw Scam Report + Malwarebytes + ZDNet + Forbes security analysis
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# v2.7.0 — ADVANCED OPTIONS STRATEGIES (81-87)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 81. GAMMA EXPOSURE TRACKER ─────────────────────────────────────────
"bw-gamma-exposure": {
    "name": "BARREN WUFFET Gamma Exposure Tracker",
    "description": "Track dealer gamma exposure (GEX), flip levels, vol surface, and OPEX dynamics. Identify gamma squeeze setups and dealer hedging flows.",
    "metadata": {
        "openclaw": {
            "emoji": "📊",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Gamma Exposure Tracker

Real-time dealer gamma exposure analysis for market regime identification.

| Feature | Description |
|---------|-------------|
| GEX Calculation | Aggregate gamma at each strike × open interest × 100 |
| Flip Level | Price where dealer gamma crosses zero |
| Vol Surface | IV skew, term structure, smile analysis |
| Put/Call Analysis | Sentiment, contrarian signals, unusual flow |
| OPEX Dynamics | Monthly/quarterly gamma roll-off impact |
| Charm/Vanna Flow | Delta decay and IV-driven dealer rebalancing |

**Regime Classification**:
- Positive GEX → Market pinned, low vol, mean-reversion
- Near-zero GEX → Transition, directional risk
- Negative GEX → Amplified moves, gamma squeeze potential

### Commands
```
/bw-gamma-exposure gex <ticker>          — GEX at all strikes
/bw-gamma-exposure flip <ticker>         — Gamma flip level
/bw-gamma-exposure surface <ticker>      — Vol surface analysis
/bw-gamma-exposure putcall <ticker>      — Put/call sentiment
/bw-gamma-exposure opex                  — OPEX impact analysis
/bw-gamma-exposure regime                — Current gamma regime
/bw-gamma-exposure charm-flow            — End-of-day drift prediction
```

### Source: strategies/gamma_exposure_tracker.py + INSIGHTS_OPTIONS_250.md
""",
},

# ─── 82. WHEEL STRATEGY ENGINE ──────────────────────────────────────────
"bw-wheel-strategy": {
    "name": "BARREN WUFFET Wheel Strategy Engine",
    "description": "Automated wheel strategy: cash-secured puts → assignment → covered calls. Income generation with stock screener and yield tracking.",
    "metadata": {
        "openclaw": {
            "emoji": "🎡",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Wheel Strategy Engine

Systematic income generation through the wheel (CSP → CC → repeat).

| Phase | Action | Optimal Delta |
|-------|--------|---------------|
| Phase 1 | Sell cash-secured puts | 0.25-0.30 |
| Phase 2 | If assigned, sell covered calls | 0.30-0.35 |
| Phase 3 | If called away, restart | — |

**Screening Criteria**: IV Rank > 30%, strong fundamentals, no earnings in 30 days.
**Yield Target**: 1.5-3% monthly (18-36% annualized).

### Commands
```
/bw-wheel screen                        — Screen stocks for wheel candidates
/bw-wheel csp <ticker> <strike>         — Analyze CSP trade
/bw-wheel cc <ticker> <strike>          — Analyze covered call trade
/bw-wheel yield <ticker>                — Monthly/annual yield projection
/bw-wheel portfolio                     — Active wheel positions
/bw-wheel history                       — Wheel trade journal
```

### Source: strategies/options_income_systems.py + INSIGHTS_OPTIONS_250.md
""",
},

# ─── 83. ZERO DTE GAMMA ENGINE ──────────────────────────────────────────
"bw-zero-dte": {
    "name": "BARREN WUFFET Zero DTE Engine",
    "description": "0DTE options strategies: iron condors, gamma scalps, opening range breakouts, max pain analysis. Session-aware trading with phase detection.",
    "metadata": {
        "openclaw": {
            "emoji": "⚡",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Zero DTE Engine

Session-aware 0DTE strategies with real-time gamma tracking.

| Session Phase | Time (ET) | Strategy |
|---------------|-----------|----------|
| Pre-market | 07:00-09:30 | Opening range setup |
| Opening Drive | 09:30-10:00 | Momentum/breakout |
| Mid-morning | 10:00-11:30 | IC entry on fade |
| Lunch | 11:30-13:30 | Theta burn, reduce size |
| Power Hour | 15:00-16:00 | Gamma scalps, closing trades |

**Max Pain**: Price gravitates to strike with maximum aggregate option loss.
**Gamma Scalp**: Long straddle + delta-hedge for realized > implied vol profit.

### Commands
```
/bw-zero-dte session                    — Current session phase & strategy
/bw-zero-dte ic <ticker> <width>        — Generate 0DTE iron condor
/bw-zero-dte scalp <ticker>             — Gamma scalp setup
/bw-zero-dte range <ticker>             — Opening range levels
/bw-zero-dte maxpain <ticker>           — Max pain calculation
/bw-zero-dte pnl                        — Day's 0DTE P&L tracker
```

### Source: strategies/zero_dte_gamma_engine.py + INSIGHTS_OPTIONS_250.md
""",
},

# ─── 84. VOLATILITY ARBITRAGE ENGINE ────────────────────────────────────
"bw-vol-arb": {
    "name": "BARREN WUFFET Volatility Arbitrage Engine",
    "description": "Variance risk premium harvesting, term structure trades, skew analysis, and VIX regime detection. Systematic vol selling with tail hedging.",
    "metadata": {
        "openclaw": {
            "emoji": "🌀",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Volatility Arbitrage Engine

Systematic volatility trading across term structure, skew, and regimes.

| Strategy | Mechanism | Best When |
|----------|-----------|-----------|
| VRP Harvest | Sell IV > RV | VRP Z-score > 1.5 |
| Calendar Spread | Term structure contango | Front IV >> Back IV |
| Risk Reversal | Skew mean-reversion | 25Δ skew > 2σ from mean |
| VIX Regime | Regime-adapted strategy | Any environment |

**VIX Regimes**:
- 10-15: Iron condors, jade lizards
- 15-20: Credit spreads, calendars
- 20-30: Put spreads, ratio writes
- 30+: Long puts, VIX calls, cash

### Commands
```
/bw-vol-arb vrp <ticker>               — Variance risk premium analysis
/bw-vol-arb term-structure <ticker>     — Term structure (contango/backwardation)
/bw-vol-arb skew <ticker>              — Skew analysis & trade ideas
/bw-vol-arb regime                     — Current VIX/VVIX regime
/bw-vol-arb trade-ideas                — Best vol trades for current regime
/bw-vol-arb backtest <strategy>        — Historical vol strategy backtest
```

### Source: strategies/volatility_arbitrage_engine.py + INSIGHTS_OPTIONS_250.md
""",
},

# ─── 85. IV CRUSH & EARNINGS ENGINE ─────────────────────────────────────
"bw-iv-crush": {
    "name": "BARREN WUFFET IV Crush & Earnings Engine",
    "description": "Earnings IV crush strategies: expected move calculator, IC/calendar/butterfly setups, seasonal scanner. Pre & post-earnings analysis.",
    "metadata": {
        "openclaw": {
            "emoji": "💥",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET IV Crush & Earnings Engine

Capitalize on the systematic IV crush after earnings announcements.

| Metric | Detail |
|--------|--------|
| Expected Move | ATM straddle × 0.85 |
| IV Crush Size | Typically 30-70% IV drop post-earnings |
| Gap Fill Rate | 60-70% of earnings gaps fill within 10 days |
| Optimal Entry | 5-10 DTE before earnings |

**Strategies**: Iron condor inside expected move, calendar spread, butterfly at target.
**Position Sizing**: Half normal size (binary event risk).

### Commands
```
/bw-iv-crush expected-move <ticker>     — Expected move calculation
/bw-iv-crush scanner                    — Upcoming earnings with high IV rank
/bw-iv-crush setup <ticker>             — Best IV crush trade setup
/bw-iv-crush history <ticker>           — Historical expected vs actual moves
/bw-iv-crush calendar                   — Earnings calendar with IV data
/bw-iv-crush post-earnings <ticker>     — Post-earnings gap analysis
```

### Source: strategies/earnings_iv_crush_engine.py + INSIGHTS_OPTIONS_250.md
""",
},

# ─── 86. PORTFOLIO GREEKS RISK ENGINE ───────────────────────────────────
"bw-greeks-portfolio": {
    "name": "BARREN WUFFET Portfolio Greeks & Risk Engine",
    "description": "Portfolio-level Greeks aggregation, beta-weighted delta, hedging recommendations, and position sizing. Multi-underlying risk management.",
    "metadata": {
        "openclaw": {
            "emoji": "🧮",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Portfolio Greeks & Risk Engine

Comprehensive portfolio-level Greek analysis and risk management.

| Greek | Portfolio Use |
|-------|---------------|
| Beta-Weighted Delta | Normalize all positions to SPY-equivalent |
| Net Theta | Daily portfolio time decay income |
| Portfolio Vega | Sensitivity to broad IV changes |
| Portfolio Gamma | Convexity risk across all positions |

**Hedging**: Delta neutralization, gamma reduction, vega management, tail risk protection.
**Position Sizing**: Kelly criterion, max risk per trade, correlation-adjusted.

### Commands
```
/bw-greeks-portfolio summary            — All portfolio Greeks
/bw-greeks-portfolio beta-weighted      — SPY beta-weighted analysis
/bw-greeks-portfolio hedge              — Hedging recommendations
/bw-greeks-portfolio sizing <trade>     — Position size calculator
/bw-greeks-portfolio stress-test        — Portfolio stress scenarios
/bw-greeks-portfolio risk-class         — Risk classification report
```

### Source: strategies/greeks_portfolio_risk.py + INSIGHTS_OPTIONS_250.md
""",
},

# ─── 87. OPTIONS STRATEGY ENGINE ────────────────────────────────────────
"bw-options-strategy-engine": {
    "name": "BARREN WUFFET Options Strategy Engine",
    "description": "Master options strategy builder with 20+ strategies, BSM pricing, Greeks computation, and strategy scanner. Covers verticals, spreads, exotics.",
    "metadata": {
        "openclaw": {
            "emoji": "🏗️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Options Strategy Engine

Full-spectrum options strategy construction with live Greeks and risk analysis.

| Category | Strategies |
|----------|------------|
| Bullish | Long call, bull call spread, bull put spread, call ratio backspread |
| Bearish | Long put, bear put spread, bear call spread, put ratio backspread |
| Neutral | Iron condor, iron butterfly, short straddle, short strangle |
| Volatility | Long straddle, long strangle, calendar spread |
| Income | Covered call, cash-secured put, jade lizard |

**Pricing**: Black-Scholes with dividend adjustment.
**Risk Metrics**: Max profit/loss, breakevens, probability of profit, expected value.

### Commands
```
/bw-options-engine build <strategy>     — Build strategy with legs
/bw-options-engine price <legs>         — BSM pricing for strategy
/bw-options-engine greeks <legs>        — Full Greeks breakdown
/bw-options-engine scan <outlook>       — Best strategies for outlook
/bw-options-engine compare <s1> <s2>    — Compare two strategies
/bw-options-engine payoff <legs>        — Payoff diagram data
```

### Source: strategies/options_strategy_engine.py + INSIGHTS_OPTIONS_250.md
""",
},

# ═══════════════════════════════════════════════════════════════════════════
# v2.7.0 — CRYPTO INTELLIGENCE DEEP DIVE (88-93)
# ═══════════════════════════════════════════════════════════════════════════

# ─── 88. ON-CHAIN METRICS ENGINE ────────────────────────────────────────
"bw-onchain-metrics": {
    "name": "BARREN WUFFET On-Chain Metrics Engine",
    "description": "On-chain analysis: MVRV ratio, SOPR, NUPL, NVT, exchange flows, supply dynamics. Bitcoin & Ethereum cycle phase detection.",
    "metadata": {
        "openclaw": {
            "emoji": "⛓️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET On-Chain Metrics Engine

On-chain intelligence for cycle timing and accumulation/distribution detection.

| Metric | Bullish | Bearish |
|--------|---------|---------|
| MVRV | < 1.0 (undervalued) | > 3.5 (overvalued) |
| SOPR | > 1.0 (profit taking) | < 1.0 (capitulation) |
| NUPL | < 0 (capitulation) | > 0.75 (euphoria) |
| NVT | < 30 (undervalued) | > 150 (overvalued) |
| Exchange Flows | Net outflow (accumulation) | Net inflow (distribution) |

**Cycle Detection**: Composite of MVRV + NUPL + SOPR + exchange reserves.
**Supply Dynamics**: LTH/STH ratio, dormancy flow, supply shock ratio.

### Commands
```
/bw-onchain mvrv <asset>                — MVRV ratio & Z-score
/bw-onchain sopr <asset>                — SOPR (all variants)
/bw-onchain nupl <asset>                — Net Unrealized P&L phase
/bw-onchain nvt <asset>                 — NVT ratio & signal
/bw-onchain flows <asset>               — Exchange flow analysis
/bw-onchain supply <asset>              — Supply dynamics dashboard
/bw-onchain cycle <asset>               — Cycle phase composite
```

### Source: CryptoIntelligence/onchain_analysis_engine.py + INSIGHTS_CRYPTO_250.md
""",
},

# ─── 89. MEV PROTECTION SYSTEM ──────────────────────────────────────────
"bw-mev-protect": {
    "name": "BARREN WUFFET MEV Protection System",
    "description": "Detect and protect against MEV extraction: sandwich attacks, front-running, JIT liquidity. Flashbots integration and MEV-aware routing.",
    "metadata": {
        "openclaw": {
            "emoji": "🛡️",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET MEV Protection System

Protect DeFi transactions from MEV extraction (sandwiches, front-runs).

| Threat | Protection |
|--------|------------|
| Sandwich Attack | Private RPC (Flashbots Protect) |
| Front-running | MEV-aware transaction submission |
| JIT Liquidity | Detection alerts on LP positions |
| Slippage Exploit | Optimal slippage calculation per pool |

**Flashbots Protect RPC**: https://rpc.flashbots.net
**MEV-Share**: Up to 90% of MEV value returned to users.

### Commands
```
/bw-mev assess <tx_data>               — MEV risk assessment
/bw-mev protect <tx_data>              — Generate protection plan
/bw-mev flashbots-setup <chain>        — Flashbots RPC configuration
/bw-mev route <swap>                   — MEV-aware DEX routing
/bw-mev detect <block>                 — Detect sandwich attacks in block
/bw-mev stats                          — MEV extraction statistics
```

### Source: CryptoIntelligence/mev_protection_system.py + INSIGHTS_CRYPTO_250.md
""",
},

# ─── 90. DEFI YIELD ANALYZER ────────────────────────────────────────────
"bw-defi-yield": {
    "name": "BARREN WUFFET DeFi Yield Analyzer",
    "description": "DeFi yield analysis: impermanent loss calculator (standard + concentrated), yield sustainability scoring, protocol risk assessment, opportunity screening.",
    "metadata": {
        "openclaw": {
            "emoji": "🌾",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET DeFi Yield Analyzer

Comprehensive DeFi yield analysis with risk-adjusted opportunity screening.

| Analysis | Output |
|----------|--------|
| Impermanent Loss | IL % for any price ratio (standard + concentrated) |
| Yield Sustainability | Fee-based vs emission-based classification |
| Protocol Risk | 0-100 score (TVL, audits, admin keys, oracle deps) |
| Opportunity Screen | Risk-adjusted yield with allocation recommendation |

**Yield Tiers**: Conservative (2-5% APY), Moderate (5-15%), Aggressive (50-500%).
**Red Flags**: APY > 100% with no volume, anon team, no audit, fork of fork.

### Commands
```
/bw-defi-yield il <price_ratio>         — Impermanent loss calculator
/bw-defi-yield il-clmm <ratio> <range>  — Concentrated LP IL calculator
/bw-defi-yield sustain <protocol>       — Yield sustainability analysis
/bw-defi-yield risk <protocol>          — Protocol risk score
/bw-defi-yield screen                   — Best risk-adjusted opportunities
/bw-defi-yield compare <p1> <p2>        — Compare protocol yields
```

### Source: CryptoIntelligence/defi_yield_analyzer.py + INSIGHTS_CRYPTO_250.md
""",
},

# ─── 91. WHALE TRACKING SYSTEM ──────────────────────────────────────────
"bw-whale-tracker": {
    "name": "BARREN WUFFET Whale Tracking System",
    "description": "Whale wallet classification, flow direction detection, multi-whale accumulation signals, token unlock impact analysis. On-chain whale intelligence.",
    "metadata": {
        "openclaw": {
            "emoji": "🐋",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Whale Tracking System

Track whale wallet activity and detect accumulation/distribution patterns.

| Classification | Holdings |
|----------------|----------|
| Mega Whale | > 10,000 BTC |
| Whale | 1,000 - 10,000 BTC |
| Shark | 100 - 1,000 BTC |
| Dolphin | 10 - 100 BTC |

**Key Signals**:
- Exchange deposit = potential sell pressure
- Exchange withdrawal = accumulation
- 3+ whales buying same token = strong convergence signal
- Dormant wallet activation = long-term holder taking action

### Commands
```
/bw-whale track <address>               — Classify wallet & track activity
/bw-whale flows <token>                 — Whale flow direction analysis
/bw-whale accumulation <token>          — Multi-whale convergence detection
/bw-whale unlocks <token>               — Token unlock schedule & impact
/bw-whale alerts                        — Active whale movement alerts
/bw-whale top-wallets <token>           — Largest holders analysis
```

### Source: CryptoIntelligence/whale_tracking_system.py + INSIGHTS_CRYPTO_250.md
""",
},

# ─── 92. FUNDING RATES & OI ANALYZER ────────────────────────────────────
"bw-funding-rates": {
    "name": "BARREN WUFFET Funding Rates & Open Interest Analyzer",
    "description": "Perpetual futures funding rate analysis, open interest divergence detection, carry trade identification. Contrarian signal generator.",
    "metadata": {
        "openclaw": {
            "emoji": "📈",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Funding Rates & Open Interest Analyzer

Perp futures market microstructure analysis for directional edge.

| Signal | Read |
|--------|------|
| Extreme positive funding (> 0.1%) | Market overleveraged long — contrarian short |
| Extreme negative funding (< -0.05%) | Market overleveraged short — contrarian long |
| Rising OI + Rising price | New longs — trend confirmation |
| Rising OI + Falling price | New shorts — bearish pressure |
| Falling OI + Rising price | Short covering — weak bounce |

**Carry Trade**: Earn funding by taking opposite side of extreme rates.
**Divergence**: OI rising with price falling = new shorts = peak bearishness.

### Commands
```
/bw-funding rates <token>               — Current funding rates across exchanges
/bw-funding regime <token>              — Funding rate regime classification
/bw-funding carry <token>               — Carry trade opportunity analysis
/bw-funding oi-divergence <token>       — OI vs price divergence detection
/bw-funding contrarian                  — Contrarian signals scanner
/bw-funding history <token>             — Historical funding rate chart
```

### Source: CryptoIntelligence/crypto_technical_patterns.py + INSIGHTS_CRYPTO_250.md
""",
},

# ─── 93. LIQUIDATION CASCADE DETECTOR ───────────────────────────────────
"bw-liquidation-watch": {
    "name": "BARREN WUFFET Liquidation Cascade Detector",
    "description": "Detect and trade around liquidation cascades, dominance rotation cycles, CME gaps, and 200-week MA signals. Crypto market structure intelligence.",
    "metadata": {
        "openclaw": {
            "emoji": "🌊",
            "homepage": "https://github.com/ResonanceEnergy/AAC",
            "requires": {"env": ["AAC_API_KEY"]},
            "primaryEnv": "AAC_API_KEY",
        }
    },
    "instructions": """## BARREN WUFFET Liquidation Cascade Detector

Detect liquidation cascades, dominance cycles, and structural market signals.

| Feature | Description |
|---------|-------------|
| Cascade Detection | OI drop > 10% in < 1h with volume spike |
| Recovery Probability | 65% bounce within 4 hours post-cascade |
| BTC Dominance | Risk-on/off cycle, alt season index |
| CME Gap | 80%+ fill rate within 1-2 weeks |
| 200-Week MA | Generational bottom indicator |

**Rotation Order**: BTC → ETH → Large caps → Mid caps → Small caps → Memes.
**Alt Season Index**: 75+ = alt season, < 25 = BTC season.

### Commands
```
/bw-liquidation detect                  — Current cascade risk assessment
/bw-liquidation history                 — Recent cascade events & recovery
/bw-liquidation dominance              — BTC dominance phase analysis
/bw-liquidation rotation               — Alt rotation cycle position
/bw-liquidation cme-gap                — Open CME gaps
/bw-liquidation 200wma <asset>         — 200-week MA analysis
/bw-liquidation heatmap                — Liquidation level heatmap
```

### Source: CryptoIntelligence/crypto_technical_patterns.py + INSIGHTS_CRYPTO_250.md
""",
},

}  # End BARREN_WUFFET_SKILLS

# ═══════════════════════════════════════════════════════════════════════════
# SKILL FILE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_skill_md(skill_def: Dict) -> str:
    """Generate a SKILL.md content string from a skill definition."""
    metadata_str = json.dumps(skill_def["metadata"], indent=None)
    lines = [
        "---",
        f'name: {skill_def["name"]}',
        f'description: {skill_def["description"]}',
        f"metadata: {metadata_str}",
        "---",
        "",
        skill_def["instructions"].strip(),
        "",
    ]
    return "\n".join(lines)


def write_all_skills(base_dir: Optional[str] = None) -> List[str]:
    """Write all BARREN WUFFET skills to the OpenClaw workspace."""
    skills_base = Path(base_dir) if base_dir else SKILLS_DIR
    written = []
    for skill_name, skill_def in BARREN_WUFFET_SKILLS.items():
        skill_dir = skills_base / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md_path = skill_dir / "SKILL.md"
        content = generate_skill_md(skill_def)
        skill_md_path.write_text(content, encoding="utf-8")
        written.append(str(skill_dir))
        logger.info(f"  ✅ {skill_name} → {skill_md_path}")
    return written


def get_skill_names() -> List[str]:
    """Get list of all BARREN WUFFET skill names."""
    return list(BARREN_WUFFET_SKILLS.keys())


def get_skill_definition(name: str) -> Optional[Dict]:
    """Get skill definition by name."""
    return BARREN_WUFFET_SKILLS.get(name)


def get_skills_by_category(category: str) -> Dict[str, Dict]:
    """Get skills filtered by category prefix."""
    prefixes = {
        "core": ["bw-market-intelligence", "bw-trading-signals", "bw-portfolio",
                 "bw-risk", "bw-crypto-intel", "bw-az-supreme", "bw-doctrine",
                 "bw-morning", "bw-agent", "bw-strategy"],
        "trading": ["bw-digital", "bw-arbitrage", "bw-day-trading",
                    "bw-options", "bw-calls", "bw-hedging", "bw-currency",
                    "bw-gamma-exposure", "bw-wheel", "bw-zero-dte",
                    "bw-vol-arb", "bw-iv-crush", "bw-greeks-portfolio",
                    "bw-options-strategy-engine"],
        "crypto": ["bw-bitcoin", "bw-ethereum", "bw-xrp", "bw-stable",
                  "bw-meme", "bw-liberty", "bw-x-tokens",
                  "bw-onchain", "bw-mev", "bw-defi-yield",
                  "bw-whale", "bw-funding", "bw-liquidation"],
        "finance": ["bw-banking", "bw-accounting", "bw-regulations",
                   "bw-compliance", "bw-tax-"],
        "wealth": ["bw-money", "bw-wealth", "bw-superstonk",
                  "bw-milestone", "bw-estate", "bw-referral"],
        "advanced": ["bw-crash", "bw-golden", "bw-jonny",
                    "bw-black-scholes", "bw-trinity", "bw-backtester"],
        "powerups": ["bw-polymarket", "bw-second-brain",
                    "bw-prediction-markets", "bw-market-commentary"],
        "quantitative": ["bw-kelly", "bw-var-", "bw-black-scholes",
                        "bw-backtester", "bw-monte-carlo", "bw-iron-condor",
                        "bw-defi-risk"],
        "security": ["bw-security", "bw-skill-scanner", "bw-scam",
                    "bw-api-cost", "bw-web3-auditor", "bw-quantum-monitor",
                    "bw-gas-optimizer", "bw-scam-shield-v2"],
        "defi": ["bw-flash-loans", "bw-yield", "bw-onchain",
                "bw-dca-grid", "bw-wallet-manager", "bw-ccxt",
                "bw-yield-farm", "bw-subgraph-indexer", "bw-multisig-treasury"],
        "intelligence": ["bw-sentiment", "bw-sec-monitor", "bw-earnings",
                        "bw-insider", "bw-websocket", "bw-alpha-scraper",
                        "bw-gold-btc-macro", "bw-central-bank-gold",
                        "bw-prediction-arb", "bw-contract-forensics"],
        "operations": ["bw-trade-journal", "bw-graduated", "bw-rebalance"],
    }
    target_prefixes = prefixes.get(category.lower(), [])
    return {k: v for k, v in BARREN_WUFFET_SKILLS.items()
            if any(k.startswith(p) for p in target_prefixes)}


def get_skill_count() -> int:
    """Get total number of BARREN WUFFET skills."""
    return len(BARREN_WUFFET_SKILLS)


# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH-BACKED SKILL ENHANCEMENTS (Deep Dive Batch 2)
# ═══════════════════════════════════════════════════════════════════════════

RESEARCH_INTEL: Dict[str, Dict] = {
    # Source: Bitrue.com — OpenClaw Trading Bot Review (Feb 21, 2026)
    "trading_modes": {
        "dca_ladders": "Accumulate positions during dips with configurable step sizes",
        "grid_trading": "Profit in range-bound markets with automated buy/sell grids",
        "contrarian": "Bet against crowd panic with sentiment-inverted signals",
        "self_improving_ai": {
            "circuit_breakers": "Add after consecutive losses",
            "trend_filter": "Skip trades when ADX < 20 (choppy market)",
            "volatility_gate": "Bollinger Band width filter for entry timing",
            "tested_win_rates": "58-75% in controlled backtests",
            "llm_failure_rate": "~20% action failure rate from LLM hallucinations",
        },
    },
    # Source: Intellectia.ai — Investment Journey Guide (Feb 13, 2026)
    "investor_patterns": {
        "time_constrained": ["auto_rebalancing", "stop_loss", "earnings_summaries"],
        "active_trader": ["multi_timeframe_scan", "arbitrage", "social_sentiment", "dynamic_trailing_stops"],
        "long_term_builder": ["dca_automation", "tax_loss_harvesting", "monte_carlo_projections"],
        "graduated_permissions": ["monitor_only", "paper_trading", "small_positions", "full_automation"],
        "natural_language_strategy": "Define complex strategies conversationally without code",
    },
    # Source: Aurpay.net — 10 Crypto Use Cases (Feb 1, 2026)
    "crypto_patterns": {
        "dex_trading": {"library": "CCXT", "exchanges": ["Uniswap", "Raydium", "Jupiter"]},
        "sentiment_analysis": {"tools": ["BeautifulSoup", "NLTK"], "sources": ["X", "Reddit", "Telegram"]},
        "yield_farming": {"api": "DeFiLlama", "protocols": ["Aave", "Compound", "Uniswap"]},
        "flash_loans": "Capital-efficient arbitrage in DeFi",
        "on_chain_research": {"tools": ["Dune Analytics", "Etherscan"], "rpcs": ["Solana RPC"]},
        "wallet_mgmt": {"features": ["multi-chain", "airdrop_farming", "bridge_automation"]},
        "risk_mgmt": {"libs": ["TA-Lib", "Pandas"], "features": ["volatility_filter", "stop_loss", "backtesting"]},
        "solana_tools": {"skill": "Pumpmolt", "use": "Token launch detection on Solana"},
    },
    # Source: OpenClaws.io — FrankenClaw Scam Report (Feb 10, 2026)
    "scam_intelligence": {
        "known_scam_tokens": ["$CLAWD", "$OPENCLAW", "$FCLAW", "$MOLT (some fakes)"],
        "frankenclaw_extracted": "$2.3M from investors via pump-and-dump",
        "red_flags": [
            "Guaranteed returns (e.g., 500% in 90 days)",
            "Claims of 'official OpenClaw token'",
            "Paid influencer campaigns for token promotion",
            "Telegram groups banning critics",
            "Modified OpenClaw branding/logos",
        ],
        "official_rule": "NO OpenClaw token exists — funded by grants/donations only",
        "verification_channels": ["openclaws.io", "official GitHub", "official Discord"],
    },
    # Source: Bitrue — Reliability Assessment
    "reliability": {
        "framework_reliable": True,
        "passive_income_reliable": False,
        "general_crypto_returns": "2-3% in broader benchmarks",
        "sortino_ratio": "Weak in general crypto trading",
        "niche_performance": "Exceptional in prediction markets and specialized setups",
        "safe_usage": [
            "Start with demo/paper trading",
            "Backtest across bull/bear/sideways",
            "Limit risk to 1-2% of portfolio",
            "Monitor VPS logs daily",
            "NEVER set-and-forget",
        ],
        "non_custodial": True,
        "github_stars": "150,000+",
    },
    # ─── DEEP DIVE BATCH 3 RESEARCH INTEL ──────────────────────────────────
    # Source: rayandcherry/OpenClaw-financial-intelligence
    "financial_intelligence_strategies": {
        "trinity": {
            "type": "trend_following",
            "indicator": "EMA50",
            "entry": "Pullback to EMA50 support",
            "exit": "Close below EMA50",
            "ai_context": "Google Gemini + DuckDuckGo news analysis",
        },
        "panic": {
            "type": "mean_reversion",
            "entry": "RSI < 30 AND below lower Bollinger Band",
            "exit": "RSI > 50 or mid Bollinger Band",
            "best_for": "Capitulation events",
        },
        "reversal_2b": {
            "type": "swing_failure",
            "entry": "New high fails to hold, close below previous high",
            "exit": "Previous swing low",
            "best_for": "Failed breakouts",
        },
        "risk_management": {
            "position_sizing": "Kelly Criterion",
            "stop_method": "ATR Trailing Stops",
            "scaling": "Ladder scaling (50% at TP1, rest trailing)",
            "var_calculation": True,
        },
    },
    # Source: adversa.ai — OpenClaw Security 101 (Feb 2026)
    "security_hardening": {
        "cves": {
            "CVE-2026-25253": {"type": "1-Click RCE", "cvss": 8.8, "status": "Patched v2026.1.29"},
            "CVE-2026-24763": {"type": "Command Injection", "status": "Patched"},
            "CVE-2026-25157": {"type": "Command Injection", "status": "Patched"},
            "CVE-2026-22708": {"type": "Indirect Prompt Injection", "status": "Mitigated"},
        },
        "clawhavoc": {
            "malicious_skills": 341,
            "total_scanned": 2857,
            "malicious_rate": "12%",
            "primary_payload": "Atomic Stealer (AMOS) macOS malware",
            "reverse_shells": 6,
            "c2_ip": "91.92.242[.]30",
        },
        "lethal_trifecta": [
            "Access to private data",
            "Exposure to untrusted content",
            "Ability to communicate externally",
            "Persistent memory (bonus 4th)",
        ],
        "exposed_instances": 21639,
        "mandatory_controls": 10,
    },
    # Source: theworldmag.com — Polymarket Bot ($116K/24h)
    "polymarket_advanced": {
        "bidou28old_strategy": {
            "profit_24h": "$116,000",
            "win_rate": "83%",
            "pricing_model": "Black-Scholes binary adaptation",
            "min_edge": "6 cents",
            "data_source": "Binance WebSocket",
            "execution": "Polymarket CLOB API",
            "oracle": "UMA (2h undisputed resolution)",
        },
        "pipeline": ["Binance WebSocket", "Price Delta", "BS Fair Value", "Compare CLOB", "Execute"],
        "cftc_status": "No-action letter, authorized intermediated platform",
    },
    # Source: serif.ai/openclaw/financial-planning
    "financial_planning_workflows": {
        "annual_review_triggers": True,
        "milestone_tracking": True,
        "rebalancing_alerts": True,
        "tax_loss_harvesting": True,
        "estate_planning": True,
        "market_commentary": True,
        "compliance_documentation": True,
        "referral_network": True,
        "continuing_education": True,
        "earnings_monitoring": True,
    },
    # Source: VoltAgent/awesome-openclaw-skills
    "community_registry": {
        "total_skills": 13729,
        "curated_count": 5494,
        "malicious_filtered": 373,
        "finance_skills": 22,
        "dev_tools_skills": 350,
        "data_science_skills": 180,
        "key_finance_skills": [
            "openinsider", "sec-filings", "bloomberg-terminal",
            "crypto-portfolio", "stock-screener", "earnings-tracker",
        ],
    },
    # ─── DEEP DIVE BATCH 4 RESEARCH INTEL ──────────────────────────────────
    # Source: Aurpay — The Great Divergence: Gold vs Bitcoin (2016-2026)
    "macro_divergence": {
        "gold_price_2026": "$5,000/oz (Feb 2026)",
        "btc_range_2026": "$66K-$90K (volatile)",
        "correlation": -0.17,
        "gold_market_cap": "$18.2T",
        "btc_market_cap": "$1.4-1.8T",
        "central_bank_buying": "1,000+ tonnes/year (record pace since 2022)",
        "top_buyers": {
            "china_pboc": "13 consecutive months buying",
            "poland": "543+ tonnes, 100+ added late 2025-2026",
            "india_rbi": "876.2 tonnes (+40% over 5 years)",
        },
        "barbell_strategy": {
            "gold_allocation": "10-15% (ruin risk hedge)",
            "btc_allocation": "2-5% (FOMO risk hedge)",
            "rebalancing": "Quarterly or on 20% deviation",
        },
        "debasement_trade": "Gold rising WITH high rates (3.5%+) — historic anomaly",
        "gold_forecast_jpm": "$5,055 Q4 2026",
        "taco_trade": "Tariffs, Acquisitions, Central banks, Overweight gold",
        "quantum_btc_risk": "~25% BTC in vulnerable P2PK addresses",
        "greenland_crisis_impact": "+3.7% gold / -3.8% BTC in single session",
    },
    # Source: Investopedia — Iron Condor + Options Strategy Research
    "options_strategies": {
        "iron_condor": {
            "structure": "Sell OTM put spread + sell OTM call spread",
            "max_profit": "Net credit received",
            "max_loss": "Spread width - net credit",
            "breakeven_low": "Short put strike - net credit",
            "breakeven_high": "Short call strike + net credit",
            "best_environment": "Low IV, no earnings/FOMC, range-bound",
            "typical_risk_reward": "0.67 (risk $3 to make $2)",
            "greeks": {"theta": "positive (time decay profits)", "vega": "negative (IV crush profits)"},
            "short_strike_delta": "15-25 delta for probability of profit",
            "adjustments": {
                "bullish": "Move put spread closer to ATM",
                "bearish": "Move call spread closer to ATM",
                "neutral_defense": "Roll untested side closer",
            },
        },
        "execution_rules": [
            "Enter all 4 legs simultaneously (no legging in)",
            "30-45 DTE optimal for theta decay",
            "Close at 50% max profit or 21 DTE",
            "Avoid earnings, binary events, FOMC weeks",
        ],
    },
    # Source: Investopedia — Monte Carlo Simulation Methodology
    "monte_carlo_risk": {
        "methodology": {
            "step_1": "Determine historical returns and parameters",
            "step_2": "Generate random price paths using log-normal distribution",
            "step_3": "Calculate portfolio value across all paths",
            "step_4": "Derive VaR, CVaR, probability of ruin",
        },
        "formulas": {
            "drift": "Avg Daily Return - (Variance / 2)",
            "random_value": "σ × NORMSINV(RAND())",
            "next_price": "Today × e^(Drift + Random Value)",
        },
        "recommended_paths": 10000,
        "distribution_adjustment": "Cornish-Fisher expansion for crypto fat tails",
        "applications": [
            "Options pricing (Black-Scholes validation)",
            "Portfolio VaR at 95% and 99% confidence",
            "Retirement probability analysis",
            "Stress testing under extreme scenarios",
        ],
        "ai_integration": "IBM 2024: ML models improve MC input estimation accuracy",
    },
    # Source: Aurpay — Claude Code Web3: 20 Ways AI Is Changing Blockchain
    "web3_security_tooling": {
        "bsa": "Behavioral State Analysis — intent vs implementation reasoning",
        "semantic_guard": "Access control usage graph for missing auth checks",
        "invariant_detection": "Infer and verify conservation rules (totalSupply = sum(balances))",
        "reentrancy_analysis": "Cross-chain + read-only reentrancy call graph detection",
        "proxy_safety": "EVM storage slot mapping for Transparent/UUPS/Beacon/Diamond proxies",
        "constant_time": "Timing side-channel detection (Trail of Bits found real ECDSA bug)",
        "foundry_fuzz": "AI-generated invariant fuzz tests with Foundry",
        "gas_optimization": "Storage packing, memory caching, unchecked blocks, inline Yul",
        "ucai_framework": "abi-to-mcp: generate MCP interface for any smart contract",
        "quicknode_mcp": "Multi-chain RPC access (Ethereum, Base, Arbitrum, Solana)",
        "dune_analytics_mcp": "Natural language → PostgreSQL for decoded blockchain data",
        "safe_multisig": "Zero-secret transaction proposal with EIP-712 typed data",
        "the_graph": "ABI → optimized @entity subgraph schema generation",
        "cicd_security": "GitHub Actions pipeline for pre-commit contract auditing",
    },
    # Source: Aurpay — 10 OpenClaw Crypto Uses + Scam Report 2026
    "crypto_automation_patterns": {
        "arbitrage": {
            "documented_return": "$500 → $106K",
            "method": "Cross-exchange price discrepancy detection",
            "execution": "CCXT library for unified exchange API",
            "risk": "Slippage, withdrawal limits, exchange downtime",
        },
        "prediction_markets": {
            "win_rate": "Up to 95% on short-term 15-min options",
            "lag_window": "~30 seconds (Binance → Polymarket)",
            "return_profile": "785% on individual windows",
        },
        "yield_farming": {
            "api": "DeFiLlama",
            "auto_compound": "Daily reward reinvestment",
            "migration_trigger": "APR differential > 20%",
        },
        "scam_wave_2026": {
            "clawd_token": "$16M market cap pump-and-dump",
            "malicious_repos": "Obfuscated crypto stealers in forked repos",
            "api_key_leak": "1.5M+ API keys exposed in single incident",
            "rce_via_web": "Malicious pages targeting AI agent system access",
            "websocket_hijack": "Cross-site WebSocket session takeover",
            "steinberger_warning": "Official maintainer public warnings issued",
        },
        "sentiment_tools": ["NLTK", "spaCy", "BeautifulSoup"],
        "ta_tools": ["TA-Lib", "Pandas", "NumPy"],
        "wallet_management": ["Multi-chain", "Airdrop farming", "Bridge automation"],
    },
    # ─── v2.7.0 RESEARCH INTEL — OPTIONS DEEP DIVE ─────────────────────────
    # Source: INSIGHTS_OPTIONS_250.md (351-600) + strategy engine research
    "advanced_options_mechanics": {
        "black_scholes_inputs": ["S (spot)", "K (strike)", "T (time)", "r (rate)", "σ (vol)"],
        "greeks_hierarchy": {
            "first_order": ["delta", "theta", "vega", "rho"],
            "second_order": ["gamma", "vanna", "charm", "vomma"],
            "third_order": ["speed", "color", "ultima", "zomma"],
        },
        "put_call_parity": "C - P = S - K*e^(-rT)",
        "iv_vs_rv": "IV typically 2-4 vol points above RV (variance risk premium)",
        "vol_surface": {
            "skew": "OTM puts trade at higher IV than OTM calls (crash protection)",
            "term_structure": "Front-month IV > back-month = backwardation (fear)",
            "smile": "Deep OTM both sides trade at premium — fat tail pricing",
        },
        "pin_risk": "At expiration, delta flips 0/100 near strike — assignment lottery",
        "early_exercise": {
            "calls": "Only optimal for deep ITM before ex-dividend",
            "puts": "Deep ITM when time value < interest on intrinsic",
        },
        "max_pain_theory": "Price gravitates to strike with max aggregate option loss at expiry",
    },
    "options_income_systems": {
        "wheel_strategy": {
            "phase_1": "Sell cash-secured puts on stocks you want to own",
            "phase_2": "If assigned, sell covered calls on shares",
            "phase_3": "If called away, restart with CSPs",
            "optimal_delta": "0.25-0.30 for CSPs, 0.30-0.35 for CCs",
            "yield_target": "1.5-3% monthly (18-36% annualized)",
            "stock_criteria": "High IV rank, strong fundamentals, no earnings in 30 days",
        },
        "credit_spread_rules": {
            "width": "5-point spreads standard, 2.5 for small accounts",
            "credit_target": "Minimum 1/3 of spread width",
            "max_risk_per_trade": "2-5% of portfolio",
            "management": "Close at 50% profit or roll at 21 DTE",
        },
        "iron_condor_management": {
            "entry_iv_rank": "Above 30% (ideally 50%+)",
            "adjustment_trigger": "Short strike breached = roll or close tested side",
            "profit_target": "50% of max credit",
            "loss_limit": "2x credit received",
        },
        "covered_call_screening": {
            "iv_percentile_min": 30,
            "min_premium_yield": 1.5,
            "max_assignment_willingness": True,
            "ex_dividend_avoidance": True,
        },
    },
    "volatility_arbitrage": {
        "variance_risk_premium": {
            "definition": "Systematic overpricing of IV vs subsequent RV",
            "typical_premium": "2-4 vol points (equity index)",
            "harvesting": "Short vol strategies (straddles, strangles, variance swaps)",
            "risk": "Convex blowup during tail events — always hedge",
        },
        "term_structure_trades": {
            "contango_trade": "Sell front-month, buy back-month (calendar spread)",
            "backwardation_trade": "Buy front-month vol (event protection)",
            "roll_yield": "Contango = positive carry for short vol",
        },
        "skew_trading": {
            "risk_reversal": "Sell OTM put, buy OTM call (or vice versa)",
            "butterfly_skew": "(σ_25P + σ_25C) / 2 - σ_ATM",
            "skew_mean_reversion": "Trade when 25-delta skew exceeds 2σ from mean",
        },
        "vix_regime": {
            "low": {"range": "10-15", "strategy": "Iron condors, jade lizards"},
            "normal": {"range": "15-20", "strategy": "Credit spreads, calendars"},
            "elevated": {"range": "20-30", "strategy": "Put spreads, ratio writes"},
            "crisis": {"range": "30+", "strategy": "Long puts, VIX calls, cash"},
        },
    },
    "gamma_exposure_mechanics": {
        "dealer_gamma": {
            "positive_gamma": "Dealers buy dips, sell rips → dampens volatility",
            "negative_gamma": "Dealers sell dips, buy rips → amplifies volatility",
            "flip_level": "Price where aggregate dealer gamma crosses zero",
        },
        "gex_interpretation": {
            "high_positive": "Market pinned, low realized vol, mean-reversion",
            "near_zero": "Transition zone, increased directional risk",
            "negative": "Amplified moves, gamma squeeze potential, trending",
        },
        "opex_dynamics": {
            "monthly_opex": "3rd Friday, large gamma roll-off",
            "quarterly_opex": "Triple/Quad witching, massive gamma unwind",
            "0dte_impact": "Daily gamma creation/destruction cycle",
        },
        "charm_flow": "Delta decay forces dealer rebalancing — end-of-day drift",
        "vanna_flow": "IV changes force delta hedging — vol crush = buying pressure",
    },
    "earnings_volatility": {
        "expected_move": {
            "formula": "ATM straddle price × 0.85",
            "historical_accuracy": "Stock stays within expected move ~70% of time",
            "edge": "Sell premium when expected move > historical average move",
        },
        "iv_crush": {
            "mechanism": "IV drops 30-70% after earnings announcement",
            "strategies": {
                "iron_condor": "Sell inside expected move, collect premium, profit from crush",
                "calendar_spread": "Sell weekly, buy monthly — IV crush helps front leg",
                "butterfly": "Center at expected closing price, low cost high reward",
            },
        },
        "pre_earnings_setup": {
            "entry_timing": "5-10 days before earnings for optimal IV expansion",
            "avoid": "Entry < 3 days out (premium already elevated)",
            "position_sizing": "Half normal size — binary event risk",
        },
        "post_earnings_analysis": {
            "gap_fill_rate": "60-70% of earnings gaps fill within 10 trading days",
            "trend_continuation": "Earnings breakout with volume confirms trend 65% of time",
        },
    },
    # ─── v2.7.0 RESEARCH INTEL — CRYPTO DEEP DIVE ──────────────────────────
    # Source: INSIGHTS_CRYPTO_250.md (601-850) + on-chain research
    "crypto_onchain_metrics": {
        "mvrv_ratio": {
            "formula": "Market Cap / Realized Cap",
            "overbought": "> 3.5 (historically = cycle top)",
            "oversold": "< 1.0 (historically = cycle bottom)",
            "z_score_thresholds": {"extreme_overvalued": 7, "extreme_undervalued": -0.5},
        },
        "sopr": {
            "definition": "Spent Output Profit Ratio — profit/loss of moved coins",
            "above_1": "Coins moving at profit — bullish if sustained",
            "below_1": "Coins moving at loss — capitulation if sustained",
            "sth_sopr": "Short-term holder SOPR — more sensitive to local sentiment",
        },
        "nupl": {
            "phases": {
                "capitulation": "< 0 (best accumulation zone)",
                "hope": "0 - 0.25",
                "optimism": "0.25 - 0.5",
                "belief": "0.5 - 0.75",
                "euphoria": "> 0.75 (distribution zone)",
            },
        },
        "exchange_flows": {
            "net_inflow": "Bearish — coins moving to exchanges for selling",
            "net_outflow": "Bullish — coins moving to cold storage (accumulation)",
            "supply_squeeze": "Exchange reserves < 12% of supply = supply shock risk",
        },
        "nvt_ratio": {
            "formula": "Market Cap / Daily Transaction Volume",
            "overvalued": "> 150 (network underutilized relative to price)",
            "undervalued": "< 30 (network activity supports price)",
        },
    },
    "mev_dynamics": {
        "sandwich_attack": {
            "mechanism": "Front-run victim tx with buy, back-run with sell",
            "target": "Large DEX swaps with high slippage tolerance",
            "cost_to_victim": "0.5-5% of trade value",
            "protection": ["Private RPCs (Flashbots Protect)", "Low slippage", "Limit orders"],
        },
        "jit_liquidity": {
            "mechanism": "Add concentrated liquidity just before large swap, remove after",
            "effect": "Captures fees from single trade without IL exposure",
            "detection": "LP position exists for exactly 1 block",
        },
        "flashbots": {
            "protect_rpc": "https://rpc.flashbots.net",
            "mechanism": "Transactions sent directly to block builders, skip public mempool",
            "chains": ["Ethereum", "Polygon", "BSC (partial)"],
            "builder_tips": "Include priority fee for builder inclusion incentive",
        },
        "mev_share": {
            "mechanism": "Searchers share extracted value with users",
            "user_rebate": "Up to 90% of MEV value returned",
        },
        "mev_by_chain": {
            "ethereum": "Highest MEV — Flashbots dominant",
            "solana": "Jito tips — validator-level MEV",
            "arbitrum": "Sequencer ordering — limited MEV",
        },
    },
    "defi_yield_mechanics": {
        "impermanent_loss": {
            "formula": "IL = 2*sqrt(r)/(1+r) - 1 where r = price_ratio",
            "thresholds": {
                "10pct_move": "-0.11% IL",
                "50pct_move": "-2.02% IL",
                "100pct_move": "-5.72% IL",
                "500pct_move": "-25.46% IL",
            },
            "concentrated_lp": "IL amplified by concentration factor (Uniswap v3/v4)",
            "mitigation": ["Correlated pairs", "Narrow range + active management", "Fee income > IL"],
        },
        "yield_sustainability": {
            "fee_based": "Sustainable — driven by real trading volume/protocol revenue",
            "emission_based": "Unsustainable — inflationary token rewards (farm & dump)",
            "real_yield": "Protocol revenue distributed to stakers (not new token emissions)",
            "red_flags": ["APY > 100% with no volume", "Anonymous team", "No audit", "Fork of fork"],
        },
        "protocol_risk_factors": [
            "Smart contract risk (audit quality, bug bounty size)",
            "Admin key risk (multisig vs single EOA)",
            "Oracle risk (Chainlink vs custom)",
            "Bridge risk (cross-chain protocols)",
            "Regulatory risk (SEC enforcement actions)",
        ],
        "yield_strategies": {
            "conservative": "Blue-chip lending (Aave, Compound) — 2-5% APY",
            "moderate": "Stablecoin LPs (Curve, Uniswap) — 5-15% APY",
            "aggressive": "Farm new protocols, token emissions — 50-500% APY (high risk)",
        },
    },
    "whale_tracking": {
        "classification": {
            "mega_whale": "> 10,000 BTC (exchanges, funds, nation-states)",
            "whale": "1,000 - 10,000 BTC",
            "shark": "100 - 1,000 BTC",
            "dolphin": "10 - 100 BTC",
            "fish": "1 - 10 BTC",
        },
        "flow_signals": {
            "exchange_deposit": "Bearish — potential sell pressure",
            "exchange_withdrawal": "Bullish — moving to cold storage",
            "whale_to_whale": "Neutral — OTC or fund rebalancing",
            "dormant_wallet_activation": "Watch closely — long-term holder taking action",
        },
        "accumulation_patterns": {
            "multi_whale_convergence": "3+ whales accumulating same token = strong signal",
            "stealth_accumulation": "Many small buys from single entity across wallets",
            "post_crash_accumulation": "Whale buying during capitulation = bottom signal",
        },
        "token_unlock_impact": {
            "small": "< 1% supply — minimal impact",
            "medium": "1-5% supply — monitor for selling pressure",
            "large": "> 5% supply — high risk, often precedes dump",
            "cliff_vs_linear": "Cliff unlocks = sudden pressure, linear = gradual",
        },
    },
    "crypto_market_microstructure": {
        "funding_rates": {
            "positive": "Longs pay shorts — market overleveraged long",
            "negative": "Shorts pay longs — market overleveraged short",
            "extreme_positive": "> 0.1% per 8h = contrarian short signal",
            "extreme_negative": "< -0.05% per 8h = contrarian long signal",
            "carry_trade": "Earn funding by taking opposite side of extreme rates",
        },
        "open_interest": {
            "rising_oi_rising_price": "New longs entering — trend confirmation",
            "rising_oi_falling_price": "New shorts entering — bearish pressure",
            "falling_oi_rising_price": "Short covering rally — weak bounce",
            "falling_oi_falling_price": "Long liquidation — washout potentially near bottom",
        },
        "liquidation_cascades": {
            "mechanism": "Forced liquidations trigger price moves, triggering more liquidations",
            "detection": "OI drops > 10% in < 1 hour with volume spike",
            "recovery": "Post-cascade bounce probability: 65% within 4 hours",
            "trading": "Fade the cascade at support/resistance with tight stops",
        },
        "dominance_rotation": {
            "btc_dominance_rising": "Risk-off — capital flowing to BTC (alt season ending)",
            "btc_dominance_falling": "Risk-on — capital flowing to alts (alt season starting)",
            "alt_season_index": "75+ = alt season, < 25 = BTC season",
            "rotation_order": "BTC → ETH → Large caps → Mid caps → Small caps → Memes",
        },
        "cme_gap": {
            "definition": "Price gap between CME Friday close and Sunday open",
            "fill_rate": "80%+ of CME gaps fill within 1-2 weeks",
            "trading": "Note gap level, expect price to revisit",
        },
    },
}


def get_research_intel(domain: str) -> Optional[Dict]:
    """Get research intelligence for a specific domain."""
    return RESEARCH_INTEL.get(domain)


def get_scam_alerts() -> List[str]:
    """Get list of known scam tokens and red flags."""
    scam = RESEARCH_INTEL.get("scam_intelligence", {})
    alerts = []
    for token in scam.get("known_scam_tokens", []):
        alerts.append(f"⚠️ SCAM TOKEN: {token}")
    for flag in scam.get("red_flags", []):
        alerts.append(f"🚩 RED FLAG: {flag}")
    alerts.append(f"✅ RULE: {scam.get('official_rule', 'N/A')}")
    return alerts


def get_safe_trading_checklist() -> List[str]:
    """Get safe trading checklist from Bitrue research."""
    return RESEARCH_INTEL.get("reliability", {}).get("safe_usage", [])


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET × OpenClaw Skills Generator")
    print("=" * 55)
    print(f"Telegram: @barrenwuffet069bot")
    print(f"Skills: {get_skill_count()}")
    print(f"Research Intel Domains: {len(RESEARCH_INTEL)}")
    print(f"Target: {SKILLS_DIR}")
    print()
    print("Categories:")
    for cat in ["core", "trading", "crypto", "finance", "wealth", "advanced",
                "powerups", "quantitative", "security", "defi", "intelligence", "operations"]:
        skills = get_skills_by_category(cat)
        print(f"  {cat.upper():10s} → {len(skills)} skills")
    print()
    print("Scam Alerts:")
    for alert in get_scam_alerts():
        print(f"  {alert}")
    print()
    print("Safe Trading Checklist:")
    for i, step in enumerate(get_safe_trading_checklist(), 1):
        print(f"  {i}. {step}")
    print()
    written = write_all_skills()
    print()
    print(f"✅ Generated {len(written)} BARREN WUFFET OpenClaw skills")
    print("Run 'openclaw gateway --verbose' to load them.")

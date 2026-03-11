"""
OpenClaw Skills Generator for AAC (DEPRECATED)
================================================

⚠️ DEPRECATED: This module (10 skills, aac-* prefix) has been superseded by
   `openclaw_barren_wuffet_skills.py` (35 skills, bw-* prefix).
   
   Use the new module for all OpenClaw skill operations:
       from integrations.openclaw_barren_wuffet_skills import BARREN_WUFFET_SKILLS
   
   This file is retained for backward compatibility only. It will be
   removed in a future release.

Original Description:
Generates OpenClaw-compatible SKILL.md files that expose AAC capabilities
as skills within the OpenClaw ecosystem. Each skill follows the AgentSkills
spec (https://agentskills.io/) with YAML frontmatter and markdown instructions.

Skills Generated:
    1. aac-market-intelligence    — Real-time market scanning across 3 theaters
    2. aac-trading-signals        — Quantum-aggregated trading signal delivery
    3. aac-portfolio-dashboard    — Dynamic portfolio dashboard generation
    4. aac-risk-monitor           — Real-time risk exposure and doctrine state
    5. aac-crypto-intel           — CryptoIntelligence engine access
    6. aac-az-supreme-command     — AZ SUPREME executive command interface
    7. aac-doctrine-status        — BARREN WUFFET State machine status & overrides
    8. aac-morning-briefing       — Automated morning market briefing
    9. aac-agent-roster           — List and query all 80+ AAC agents
   10. aac-strategy-explorer      — Browse and analyze 50 arbitrage strategies

Reference: https://docs.openclaw.ai/tools/skills
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

SKILLS_DIR = Path(os.path.expanduser("~/.openclaw/workspace/skills"))


def _ensure_skill_dir(name: str) -> Path:
    """Create skill directory and return path"""
    skill_dir = SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    return skill_dir


# ─── Skill Definitions ─────────────────────────────────────────────────────

SKILL_DEFINITIONS: Dict[str, Dict] = {
    "aac-market-intelligence": {
        "name": "aac-market-intelligence",
        "description": "Scan markets across Theater B (Attention/Narrative), Theater C (Infrastructure/Latency), and Theater D (Information Asymmetry) using AAC's BigBrainIntelligence research agents.",
        "metadata": {
            "openclaw": {
                "emoji": "📊",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
                "always": False,
            }
        },
        "instructions": """
## AAC Market Intelligence Scanner

You have access to AAC's BigBrainIntelligence research agents spanning 3 theaters:

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

### How to Use
Run a scan by specifying the theater and optional focus:
```
/aac-market-intelligence theater=B focus=crypto_narratives
/aac-market-intelligence theater=all summary=true
/aac-market-intelligence theater=C focus=gas_optimization timeframe=1h
```

The scan runs all relevant agents and returns a consolidated `ResearchFinding` 
report with confidence scores and actionable signals.

### Output Format
Results include:
- **Signal Strength**: 0.0-1.0 quantum-weighted confidence
- **Theater**: Source theater (B/C/D)
- **Finding Type**: narrative_shift | latency_anomaly | data_gap | ...
- **Actionable**: Whether the finding maps to an executable strategy
- **Strategy IDs**: Which of the 50 strategies can exploit this finding
""",
    },

    "aac-trading-signals": {
        "name": "aac-trading-signals",
        "description": "Get quantum-aggregated trading signals from AAC's QuantumSignalAggregator. Covers 50 arbitrage strategies across crypto and traditional markets.",
        "metadata": {
            "openclaw": {
                "emoji": "⚡",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
            }
        },
        "instructions": """
## AAC Trading Signals

Access AAC's quantum-aggregated trading signals from 49 trading agents 
executing across 50 arbitrage strategies.

### Signal Pipeline
```
Research Agents → ResearchFinding → QuantumSignal → QuantumSignalAggregator → Consensus → You
```

### Available Commands
```
/aac-trading-signals active          — Show all active signals with confidence > 0.7
/aac-trading-signals top=5           — Top 5 signals by quantum-weighted score
/aac-trading-signals strategy=DEX    — Signals for DEX arbitrage strategies only
/aac-trading-signals history=24h     — Signal history over the last 24 hours
/aac-trading-signals performance     — Win rate, avg return, Sharpe by strategy
```

### Signal Categories
- **Statistical Arbitrage**: Mean reversion, pairs trading, cointegration
- **Structural Arbitrage**: Market microstructure, order book imbalances
- **Technology Arbitrage**: Latency, API, cross-venue timing
- **Compliance Arbitrage**: Regulatory gap exploitation (doctrine-compliant)
- **Cross-Chain Arbitrage**: DeFi bridge, DEX, yield farming

### Risk Context
Each signal includes:
- Doctrine compliance status (BARREN WUFFET State: NORMAL/CAUTION/SAFE_MODE/HALT)
- Current drawdown and daily P&L
- Position sizing recommendation based on Kelly criterion
""",
    },

    "aac-portfolio-dashboard": {
        "name": "aac-portfolio-dashboard",
        "description": "Generate a real-time AAC portfolio dashboard with P&L, positions, strategy attribution, and risk metrics.",
        "metadata": {
            "openclaw": {
                "emoji": "📈",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
            }
        },
        "instructions": """
## AAC Portfolio Dashboard

Generate a rich portfolio dashboard from CentralAccounting with real-time 
metrics from across all 6 departments.

### Dashboard Sections
```
/aac-portfolio-dashboard full        — Complete dashboard
/aac-portfolio-dashboard summary     — Quick summary (NAV, daily P&L, top positions)
/aac-portfolio-dashboard strategies  — P&L attribution by strategy
/aac-portfolio-dashboard risk        — Risk metrics (VaR, drawdown, exposure by venue)
/aac-portfolio-dashboard contest     — $1K→$10K Trading Contest leaderboard
```

### Dashboard Output
```
╔══════════════════════════════════════════╗
║     AAC PORTFOLIO DASHBOARD             ║
╠══════════════════════════════════════════╣
║ NAV:           $XXX,XXX.XX              ║
║ Daily P&L:     +$X,XXX.XX (+X.XX%)      ║
║ Drawdown:      -X.XX% (max: -X.XX%)     ║
║ BARREN WUFFET:      NORMAL ✅                ║
║ Active Strats: XX/50                     ║
║ Open Positions: XX                       ║
╠══════════════════════════════════════════╣
║ TOP STRATEGIES (24h)                     ║
║ 1. DEX Arb         +$XXX  (XX trades)   ║
║ 2. Stat Arb Pairs  +$XXX  (XX trades)   ║
║ 3. Bridge Arb      +$XXX  (XX trades)   ║
╠══════════════════════════════════════════╣
║ RISK METRICS                             ║
║ 1d VaR (95%):  $X,XXX                    ║
║ Sharpe (30d):  X.XX                      ║
║ Max Exposure:  $XXX,XXX                  ║
╚══════════════════════════════════════════╝
```
""",
    },

    "aac-risk-monitor": {
        "name": "aac-risk-monitor",
        "description": "Monitor AAC risk exposure and BARREN WUFFET Doctrine State in real-time. Alerts on drawdown, margin, and compliance breaches.",
        "metadata": {
            "openclaw": {
                "emoji": "🛡️",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
                "always": True,
            }
        },
        "instructions": """
## AAC Risk Monitor

Real-time risk monitoring with BARREN WUFFET Doctrine State integration.

### BARREN WUFFET State Machine
```
NORMAL → CAUTION → SAFE_MODE → HALT
```
- **NORMAL**: Full trading enabled, all strategies active
- **CAUTION**: Drawdown > 5%, risk throttling active, position sizes reduced
- **SAFE_MODE**: Drawdown > 10%, execution stopped, hedging only
- **HALT**: Daily loss > 2%, all operations stopped, manual override required

### Commands
```
/aac-risk-monitor status           — Current BARREN WUFFET State + key risk metrics
/aac-risk-monitor doctrine         — All 8 doctrine pack statuses
/aac-risk-monitor exposure         — Position exposure by venue, asset, strategy
/aac-risk-monitor alerts           — Active risk alerts and breaches
/aac-risk-monitor circuit-breakers — Circuit breaker status across all systems
```

### 8 Doctrine Packs Monitored
1. Capital Preservation  2. Position Sizing  3. Execution Quality
4. Market Risk  5. Counterparty Risk  6. Operational Risk
7. Compliance  8. Performance Attribution

### Proactive Alerts
This skill can send proactive alerts via OpenClaw when:
- BARREN WUFFET State transitions (e.g., NORMAL → CAUTION)
- Drawdown exceeds thresholds
- Circuit breakers trigger
- Daily loss limits approach
""",
    },

    "aac-crypto-intel": {
        "name": "aac-crypto-intel",
        "description": "Access AAC's CryptoIntelligence engine for on-chain analysis, DeFi opportunities, whale tracking, and cross-chain arbitrage intel.",
        "metadata": {
            "openclaw": {
                "emoji": "🔗",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
            }
        },
        "instructions": """
## AAC CryptoIntelligence Engine

Deep crypto market intelligence from AAC's dedicated CryptoIntelligence department.

### Capabilities
- **On-Chain Analysis**: Transaction flow, whale movements, smart money tracking
- **DeFi Intelligence**: Yield farming opportunities, liquidity pool analysis, protocol TVL shifts
- **Cross-Chain Arbitrage**: Bridge volume analysis, cross-chain price differentials
- **Mempool Intelligence**: Pending transaction analysis, MEV opportunity detection
- **Exchange Intelligence**: Order book depth, funding rates, open interest shifts

### Commands
```
/aac-crypto-intel overview              — Market-wide crypto intelligence summary
/aac-crypto-intel whale-watch           — Large transaction monitoring
/aac-crypto-intel defi-opportunities    — Top DeFi yield and arb opportunities
/aac-crypto-intel gas-analysis          — Gas price patterns and optimization
/aac-crypto-intel bridge-flows          — Cross-chain bridge volume and arb
/aac-crypto-intel sentiment             — Crypto-specific social sentiment analysis
```

### Integration with BigBrainIntelligence
CryptoIntelligence feeds directly into BigBrain via the CryptoBigBrainBridge:
```
CryptoIntelligence → IntelligenceSignal → CryptoBigBrainBridge → QuantumSignalAggregator
```
""",
    },

    "aac-az-supreme-command": {
        "name": "aac-az-supreme-command",
        "description": "Direct command interface to AZ SUPREME — the supreme executive AI governing the entire AAC ecosystem. Issue strategic directives, query system status, and manage crises.",
        "metadata": {
            "openclaw": {
                "emoji": "👑",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
                "always": True,
            }
        },
        "instructions": """
## AZ SUPREME — Executive Command Interface

You are interfacing with AZ SUPREME, the supreme executive command agent
of the Accelerated Arbitrage Corporation (AAC).

### Executive Capabilities
- **Strategic Oversight**: Enterprise-wide strategy coordination
- **Crisis Management**: Real-time crisis detection and response
- **Cross-Domain Coordination**: Bridge all 6 departments seamlessly
- **Executive Directives**: Issue priority-weighted commands to any agent
- **Performance Monitoring**: Track all 80+ agents and 50 strategies

### Command Structure
```
/az-supreme status                    — Full system status report
/az-supreme directive <priority> <text> — Issue an executive directive
/az-supreme briefing                  — Strategic briefing from AZ SUPREME
/az-supreme crisis-mode <on|off>      — Activate/deactivate crisis management
/az-supreme departments               — Department status overview
/az-supreme question <text>           — Ask AZ SUPREME strategic question (100 question library)
/az-supreme agents list               — List all registered agents
```

### Directive Priorities
- **CRITICAL**: Immediate execution across all departments
- **HIGH**: Execute within current cycle
- **MEDIUM**: Queue for next operational cycle
- **LOW**: Backlog for strategic planning

### Response Style
AZ SUPREME responds with authority and strategic insight. Responses include
voice-ready text compatible with text-to-speech via the AZ Response Library.
""",
    },

    "aac-doctrine-status": {
        "name": "aac-doctrine-status",
        "description": "Query and monitor the BARREN WUFFET Doctrine Engine — the safety-critical state machine governing all AAC trading operations.",
        "metadata": {
            "openclaw": {
                "emoji": "📜",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
            }
        },
        "instructions": """
## BARREN WUFFET Doctrine Engine

The Doctrine Engine enforces operational safety through 8 compliance packs
and a 4-state safety machine.

### State Machine
```
NORMAL ──(drawdown > 5%)──→ CAUTION ──(drawdown > 10%)──→ SAFE_MODE ──(daily loss > 2%)──→ HALT
   ↑                                                                                        │
   └────────────────────────── Manual Override / Recovery ──────────────────────────────────┘
```

### 8 Doctrine Packs
1. **Capital Preservation**: Max drawdown, daily loss limits
2. **Position Sizing**: Kelly criterion, max position %, correlation limits
3. **Execution Quality**: Slippage tolerance, fill rate minimums
4. **Market Risk**: VaR limits, volatility regimes, liquidity requirements
5. **Counterparty Risk**: Exchange exposure limits, settlement risk
6. **Operational Risk**: System health, latency SLAs, error rate caps
7. **Compliance**: Regulatory adherence, reporting obligations
8. **Performance Attribution**: P&L attribution accuracy, benchmark tracking

### Commands
```
/aac-doctrine status              — Current BARREN WUFFET State + all pack statuses
/aac-doctrine pack <number>       — Detailed status of a specific pack
/aac-doctrine history             — State transition history
/aac-doctrine override <state>    — Request state override (requires CRITICAL auth)
```
""",
    },

    "aac-morning-briefing": {
        "name": "aac-morning-briefing",
        "description": "Automated daily morning briefing from AAC covering overnight performance, market conditions, active signals, risk status, and strategic opportunities.",
        "metadata": {
            "openclaw": {
                "emoji": "☀️",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
            }
        },
        "instructions": """
## AAC Morning Briefing

Schedule a daily morning briefing that AZ SUPREME delivers via your preferred
OpenClaw channel (Telegram, Discord, WhatsApp, etc.).

### Briefing Contents
1. **Overnight Performance**: P&L summary, strategy attribution, notable trades
2. **Market Conditions**: Key market moves, volatility regime, macro events
3. **Active Signals**: Top research findings from all 3 theaters
4. **Risk Status**: BARREN WUFFET State, drawdown, exposure summary
5. **Strategic Opportunities**: AI-recommended actions for the day
6. **Agent Performance**: Top/bottom performing agents, anomalies
7. **Doctrine Compliance**: Any pack violations or warnings
8. **System Health**: Infrastructure status, latency, uptime

### Scheduling (via OpenClaw Cron)
This skill auto-registers a cron job when activated:
```
Schedule: 0 7 * * 1-5  (7 AM Mon-Fri)
Session: main
```

Customize the schedule by telling OpenClaw:
"Change my morning briefing to 6:30 AM including weekends"

### On-Demand
```
/aac-morning-briefing now         — Generate briefing immediately
/aac-morning-briefing schedule    — Show/update schedule
/aac-morning-briefing sections    — Select which sections to include
```
""",
    },

    "aac-agent-roster": {
        "name": "aac-agent-roster",
        "description": "Browse, search, and query the full roster of 80+ AAC agents across all departments — research, trading, executive, and infrastructure.",
        "metadata": {
            "openclaw": {
                "emoji": "👥",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
            }
        },
        "instructions": """
## AAC Agent Roster

Browse and query all 80+ agents in the AAC ecosystem.

### Agent Categories
- **Executive (2)**: AZ SUPREME, AX HELIX
- **Research (20)**: Theater B (3), Theater C (4), Theater D (4), Operational (9)
- **Trading (49)**: 1 agent per strategy, mapped to 50 arbitrage strategies
- **Executive Assistants (5)**: Alpha/Beta/Gamma/Delta/Epsilon Innovation
- **Department Super Agents (6)**: One super agent per department
- **Division Agents (12+)**: Ludwig Law, Insurance, Banking, Compliance, etc.

### Commands
```
/aac-agent-roster list                  — Full agent roster
/aac-agent-roster department=BigBrain   — Agents in BigBrainIntelligence
/aac-agent-roster search=<query>        — Search agents by name/capability
/aac-agent-roster status=<agent_id>     — Detailed status of a specific agent
/aac-agent-roster performance           — Agent performance leaderboard
/aac-agent-roster super-agents          — List quantum-enhanced super agents
```

### Department Mapping
| Department | Agent Count | Lead |
|---|---|---|
| Executive Branch | 2 | AZ SUPREME |
| BigBrainIntelligence | 20+ | SuperBigBrainAgent |
| TradingExecution | 49 | SuperTradeExecutorAgent |
| CryptoIntelligence | 5+ | SuperCryptoIntelAgent |
| CentralAccounting | 3+ | SuperAccountingAgent |
| SharedInfrastructure | 5+ | SuperInfrastructureAgent |
| NCC (Network Command) | 3+ | SuperNCCCommandAgent |
""",
    },

    "aac-strategy-explorer": {
        "name": "aac-strategy-explorer",
        "description": "Browse, analyze, and compare AAC's 50 arbitrage strategies with performance metrics, parameters, and agent assignments.",
        "metadata": {
            "openclaw": {
                "emoji": "🎯",
                "homepage": "https://github.com/accelerated-arbitrage-corp",
                "requires": {"env": ["AAC_API_KEY"]},
                "primaryEnv": "AAC_API_KEY",
            }
        },
        "instructions": """
## AAC Strategy Explorer

Explore, analyze, and compare the 50 arbitrage strategies in AAC's portfolio.

### Strategy Categories
- **Statistical Arbitrage**: Mean reversion, pairs, cointegration, factor models
- **Structural Arbitrage**: Market microstructure, order book, fragmentation
- **Technology Arbitrage**: Latency, API access, data pipeline advantages
- **Compliance Arbitrage**: Regulatory gap, cross-jurisdiction advantages
- **Cross-Chain / DeFi**: DEX arb, bridge arb, yield farming, MEV

### Commands
```
/aac-strategy-explorer list            — All 50 strategies with status
/aac-strategy-explorer category=<cat>  — Filter by category
/aac-strategy-explorer detail=<id>     — Deep dive on a specific strategy
/aac-strategy-explorer performance     — Performance comparison table
/aac-strategy-explorer parameters=<id> — View/tune strategy parameters
/aac-strategy-explorer correlations    — Strategy correlation matrix
```

### Performance Output
```
Strategy Performance (30 day):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# | Strategy         | P&L     | Win% | Sharpe | Trades
1 | DEX Arbitrage    | +$X,XXX | 72%  | 2.4    | 1,234
2 | Stat Pairs ETH   | +$X,XXX | 68%  | 1.9    | 856
3 | Bridge Arb L2    | +$X,XXX | 81%  | 3.1    | 423
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Strategy Source
All strategies are loaded from `50_arbitrage_strategies.csv` and 
managed by the `StrategyImplementationFactory`.
""",
    },
}


# ─── Skill File Generator ──────────────────────────────────────────────────

def generate_skill_md(skill_def: Dict) -> str:
    """Generate a SKILL.md content string from a skill definition"""
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
    """
    Write all AAC skill SKILL.md files to the OpenClaw workspace.
    
    Returns list of skill directory paths written.
    """
    if base_dir:
        skills_base = Path(base_dir)
    else:
        skills_base = SKILLS_DIR

    written = []
    for skill_name, skill_def in SKILL_DEFINITIONS.items():
        skill_dir = skills_base / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_md_path = skill_dir / "SKILL.md"
        content = generate_skill_md(skill_def)
        skill_md_path.write_text(content, encoding="utf-8")

        written.append(str(skill_dir))
        print(f"  ✅ {skill_name} → {skill_md_path}")

    return written


def get_skill_names() -> List[str]:
    """Get list of all AAC skill names"""
    return list(SKILL_DEFINITIONS.keys())


def get_skill_definition(name: str) -> Optional[Dict]:
    """Get skill definition by name"""
    return SKILL_DEFINITIONS.get(name)


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🦞 AAC × OpenClaw Skills Generator")
    print("=" * 50)
    print(f"Writing {len(SKILL_DEFINITIONS)} skills to {SKILLS_DIR}")
    print()
    written = write_all_skills()
    print()
    print(f"✅ Generated {len(written)} OpenClaw skills for AAC")
    print("Run 'openclaw gateway --verbose' to load them.")

# AAC ARCHITECTURE REWORK PLAN — v3.3
## "War Room Doctrine + Strategy Advisor Engine"

> Generated 2026-03-28 | Author: Copilot Agent
> Mandate corrections from owner applied. This is the definitive plan.

---

## 1. CORRECTED MANDATE

| Pillar | Role |
|--------|------|
| NCC | Supreme governance (Natrix Command & Control) |
| NCL (BRAIN) | ALL intelligence processed here — the brain |
| AAC (BANK) | Generates money, executes trades, manages capital |
| BIT RAGE SYSTEMS | AGENT WORK FORCE (not SuperAgency) |

**Key principles:**
- Doctrine follows **7 current strategies** + **manual input** (not just NCC NORMAL/CAUTION/HALT)
- ALL intelligence goes to **NCL BRAIN** for processing through relay
- 122 strategies serve as **ADVISOR ROLE** + **PAPER TRADING PROOFING** (needs engine)
- Doctrine packs available to **read on Matrix Monitor Display**
- Exchange connectors updated to **War Room** and **Lifeboat Moon Cycles**

---

## 2. THE 7 ACTIVE STRATEGIES

| # | Strategy | File | Status |
|---|----------|------|--------|
| 1 | **War Room Engine** | `strategies/war_room_engine.py` | EXISTS — CLI-based, integrator BROKEN (references nonexistent `WarRoomEngine` class) |
| 2 | **Storm Lifeboat** | `strategies/storm_lifeboat/` | EXISTS — package with core/capital modules, NOT wired to integrator |
| 3 | **Capital Engine** | `strategies/storm_lifeboat/capital_engine.py` | EXISTS — hourly gold/oil/silver rotation, wired to orchestrator `_capital_engine_loop()` but NOT in integrator |
| 4 | **Matrix Maximizer** | `strategies/matrix_maximizer/` | EXISTS — 16-module options architecture, NOT wired to integrator (despite being "TOP PRIORITY") |
| 5 | **Exploitation Matrix** | `strategies/blackswan_exploitation_matrix.py` | EXISTS — 8 investment verticals, NOT wired |
| 6 | **Polymarket Scanner** | `strategies/polymarket_blackswan_scanner.py` | EXISTS — `PolymarketBlackSwanScanner` class, NOT wired |
| 7 | **BlackSwan Authority** | `strategies/blackswan_authority_monitor.py` | EXISTS — 4 expert authority feeds, NOT wired |

---

## 3. WORK ITEMS

### 3A. Wire All 7 Strategies to Integrator (HIGH PRIORITY)

**Problem:** Only Capital Engine has an orchestrator loop. The other 6 are completely disconnected.

**Fix plan:**

1. **War Room Engine** — Create a thin `WarRoomEngine` wrapper class in `war_room_engine.py` that exposes `run()` / `get_mandate()`. Fix integrator reference.

2. **Storm Lifeboat** — Add integrator entry: `("Storm Lifeboat", "strategies.storm_lifeboat.capital_engine", "LifeboatCapitalEngine")`

3. **Matrix Maximizer** — Add integrator entry: `("Matrix Maximizer", "strategies.matrix_maximizer", "MatrixMaximizer")`

4. **Exploitation Matrix** — Create thin `ExploitationMatrixEngine` wrapper class. Add integrator entry.

5. **Polymarket Scanner** — Add integrator entry: `("Polymarket BlackSwan", "strategies.polymarket_blackswan_scanner", "PolymarketBlackSwanScanner")`

6. **BlackSwan Authority** — Create thin wrapper, add integrator entry.

7. **All 7** — Each strategy relays intelligence to NCL BRAIN via `NCCRelayClient.publish()`.

### 3B. Strategy Advisor Engine (GREENFIELD — HIGH PRIORITY)

**Problem:** 122 strategies exist but have no advisor role or paper-proofing mechanism.

**Plan:** Create `strategies/strategy_advisor_engine.py`:

```
class StrategyAdvisorEngine:
    """
    Evaluates ALL 122 strategies in ADVISOR mode:
    - Runs each strategy against live market data
    - Generates paper recommendations with confidence scores
    - Tracks paper P&L for proofing (forward-testing)
    - Requires manual approval before live execution
    - Ranks strategies by paper-proof performance
    """
    
    def __init__(self):
        self._strategies: List[StrategyAdapter] = []
        self._paper_ledger: Dict[str, PaperPosition] = {}
        self._performance: Dict[str, AdvisorPerformance] = {}
    
    async def evaluate_all(self) -> List[AdvisorRecommendation]:
        """Run all 122 strategies, return ranked recommendations."""
        
    async def paper_proof_cycle(self):
        """Update all paper positions with current prices, track P&L."""
        
    def get_leaderboard(self) -> List[AdvisorPerformance]:
        """Rank strategies by paper-proof performance (Sharpe, win rate, P&L)."""
        
    def approve_for_live(self, strategy_name: str) -> bool:
        """Promote a strategy from paper-proof to live execution."""
```

**Integration:**
- Orchestrator runs `strategy_advisor_engine.evaluate_all()` periodically (every 30 min)
- Results displayed on Matrix Monitor as "STRATEGY ADVISOR LEADERBOARD"
- All recommendations relayed to NCL BRAIN via relay
- Manual approval gate — no strategy goes live without explicit confirmation

### 3C. Doctrine Rework — Strategy-Aware Governance (HIGH PRIORITY)

**Problem:** Current doctrine is a simple multiplier (NORMAL=1.0, CAUTION=0.5, HALT=0.0). Owner wants doctrine to follow 7 strategies + manual input.

**Plan:** Extend `aac/doctrine/strategic_doctrine.py`:

```
class StrategyAwareDoctrine:
    """
    Doctrine engine that synthesizes:
    1. All 7 active strategy signals (War Room, Lifeboat, etc.)
    2. Manual input from owner (override commands)
    3. Market regime from MarketIntelligenceModel
    4. NCL BRAIN intelligence (relayed back)
    
    Output: Dynamic position sizing, sector allocation, risk posture
    per strategy — NOT just a single multiplier.
    """
    
    STRATEGY_DOCTRINE_MAP = {
        "war_room": {"regime": "crisis", "bias": "defensive"},
        "storm_lifeboat": {"regime": "lunar", "bias": "cyclical"},
        "capital_engine": {"regime": "commodity", "bias": "rotation"},
        "matrix_maximizer": {"regime": "options", "bias": "premium"},
        "exploitation_matrix": {"regime": "blackswan", "bias": "conviction"},
        "polymarket": {"regime": "event", "bias": "probability"},
        "blackswan_authority": {"regime": "expert", "bias": "consensus"},
    }
    
    def generate_composite_directive(self, strategy_signals, manual_override, ncl_intel):
        """Synthesize all inputs into per-strategy directives."""
```

**Integration:**
- `TradingExecution/risk_manager.py` currently calls `position_size_modifier()` — extend to be strategy-aware
- Manual input via `.env` override flags + Matrix Monitor command interface
- NCL intelligence feeds back via `_ncl_prime_loop()`

### 3D. Intelligence Relay to NCL BRAIN (MEDIUM PRIORITY)

**Current state:** `NCCRelayClient` publishes to NCC port 8787. `NCLBridge` writes `aac_intelligence.json` to NCL data dir.

**Problem:** Not ALL intelligence goes through. Only MarketIntelligenceModel pushes to NCL.

**Fix plan:**
1. Add relay publish calls to all 7 strategy outputs
2. Strategy Advisor Engine recommendations → relay
3. Doctrine state changes → relay
4. Paper-proof performance data → relay
5. All intelligence categories use envelope `ncl.sync.v1.bank.<category>`

**Envelope categories:**
```
ncl.sync.v1.bank.strategy.war_room
ncl.sync.v1.bank.strategy.storm_lifeboat
ncl.sync.v1.bank.strategy.capital_engine
ncl.sync.v1.bank.strategy.matrix_maximizer
ncl.sync.v1.bank.strategy.exploitation_matrix
ncl.sync.v1.bank.strategy.polymarket
ncl.sync.v1.bank.strategy.blackswan_authority
ncl.sync.v1.bank.advisor.leaderboard
ncl.sync.v1.bank.advisor.recommendation
ncl.sync.v1.bank.doctrine.state_change
ncl.sync.v1.bank.doctrine.manual_override
ncl.sync.v1.bank.paper_proof.performance
```

### 3E. Matrix Monitor Doctrine Display (MEDIUM PRIORITY)

**Current state:** Dashboard already shows doctrine compliance (8-pack). But:
- Says "8 packs" when 11 exist
- No strategy-specific doctrine view
- No doctrine pack reader/viewer

**Plan:**
1. Update dashboard docstring from "8" to "11" packs
2. Add new panel: "DOCTRINE PACK READER" — shows all 11 packs with rules and current state
3. Add new panel: "STRATEGY ADVISOR LEADERBOARD" — 122 strategies ranked by paper-proof performance
4. Add new panel: "ACTIVE STRATEGY DOCTRINE" — per-strategy doctrine directives for all 7
5. Surface exported doctrine packs from `config/doctrine_packs.yaml` in the dashboard

### 3F. Exchange Connectors — War Room + Lifeboat Moon Cycles (LOW PRIORITY)

**Current state:** IBKR (live port 7496), Moomoo (OpenD), NDAX (liquidated), crypto via CCXT.

**Problem:** Exchange connectors are generic — not aligned with War Room thesis or Lifeboat moon cycles.

**Plan:**
1. War Room connector: Options-focused IBKR integration for crisis puts, VIX calls
2. Storm Lifeboat connector: Commodity ETF integration (GLD, SLV, USO) with moon phase timing
3. Both connectors call `StrategicDoctrineEngine.assess_terrain()` before placing orders

---

## 4. EXECUTION ORDER

| Phase | Items | Priority |
|-------|-------|----------|
| **Phase 1** | Wire 7 strategies to integrator (3A) | **NOW** |
| **Phase 2** | Fix MI→orchestrator gap (DONE) | **DONE** |
| **Phase 3** | Build Strategy Advisor Engine (3B) | **NEXT** |
| **Phase 4** | Doctrine rework — strategy-aware (3C) | **NEXT** |
| **Phase 5** | Intelligence relay to NCL (3D) | **AFTER** |
| **Phase 6** | Matrix Monitor doctrine display (3E) | **AFTER** |
| **Phase 7** | Exchange connector alignment (3F) | **FUTURE** |

---

## 5. COMPLETED IN THIS SESSION

| Item | Status |
|------|--------|
| MarketIntelligenceModel→orchestrator gap | **FIXED** — `_ncl_prime_loop` now reads `get_recommendations()` every 60s and injects as QuantumSignals |
| NCL Governance Engine extracted | **DONE** — `aac/doctrine/ncl_governance.py` (8-criteria OMEGA/GAMMA/BETA/ALPHA scoring) |
| full_activation.py archived | **DONE** — moved to `_archive/`, unique features extracted |
| API key audit | **DONE** — 45 empty keys are ALL future/optional. 6 critical keys (IBKR, CoinGecko, FRED, Polygon, Finnhub, UW) are configured. |

---

## 6. API KEYS STATUS

### Configured & Active (critical for 7 strategies):
- IBKR (host, port, account) — War Room Engine, Matrix Maximizer
- COINGECKO_API_KEY — MarketIntelligenceModel, Capital Engine
- FRED_API_KEY — Macro indicators, War Room
- POLYGON_API_KEY — Options data, real-time
- FINNHUB_API_KEY — Insider/analyst/calendar, Authority Monitor
- UNUSUAL_WHALES_API_KEY — Options flow, Exploitation Matrix
- POLYMARKET_* (6 keys) — Polymarket Scanner

### Configured but secondary:
- ANTHROPIC/OPENAI/GOOGLE_AI/XAI — LLM providers
- ALPHAVANTAGE_API_KEY — backup data
- NEWS_API_KEY — authority monitor
- ETHERSCAN_API_KEY — crypto
- DISCORD_WEBHOOK/BOT_TOKEN — alerts

### Empty / Future (45 keys — no urgency):
- Exchange keys: Binance, Coinbase, Kraken (not currently trading)
- Social: Reddit full auth, Twitter full auth, Telegram
- Niche data: TwelveData, IEX, EODHD, Intrinio, Tradestie, Santiment, CMC
- MetalX/XPR/MetalPay — blockchain (future)
- Infrastructure: Redis password, Slack, Email alerts

---

## 7. DATA FLOW AFTER INTEGRATION

```
[7 STRATEGIES] ──signals──► [STRATEGY ADVISOR ENGINE] ──ranked recs──►
    │                              │                        │
    │                              ▼                        ▼
    │                     [PAPER PROOF LEDGER]     [MATRIX MONITOR]
    │                              │                        │
    ▼                              ▼                        ▼
[STRATEGY-AWARE DOCTRINE] ──directives──► [RISK MANAGER] ──sized orders──► [EXECUTION]
    │                                          │
    ▼                                          ▼
[NCL RELAY] ──all intelligence──► [NCL BRAIN] ──processed intel──► [ORCHESTRATOR]
    │                                                                    │
    ▼                                                                    ▼
[NCC port 8787] ──governance──► [NCC MASTER] ──commands──► [ALL PILLARS]
```

**Key insight:** NCL BRAIN is the intelligence hub. AAC generates raw signals/recommendations, relays everything to NCL, NCL processes and returns enriched intelligence, orchestrator consumes it. Doctrine is informed by BOTH strategy signals AND NCL intelligence.

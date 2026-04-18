# Community Research: Polymarket Bot Strategies & AI Trading Techniques

> **Date:** 2026-04-08
> **Sources:** GitHub (20+ repos), Substack (25+ articles), Polymarket official SDK
> **Purpose:** Catalog community-proven prediction market bot strategies for AAC integration

---

## 1. AAC's Current Polymarket Arsenal

AAC already has significant Polymarket infrastructure:

| Component | Location | Status |
|---|---|---|
| **PolymarketAgent** | `agents/polymarket_agent.py` | WORKING — 3 API layers (Gamma/Data/CLOB) |
| **BlackSwan Scanner** | `strategies/polymarket_blackswan_scanner.py` | WORKING — thesis-aligned tail-risk scanning, 3 tiers |
| **Paper Trading Division** | `divisions/trading/polymarket_paper/` | WORKING — 5 strategies (Grid/DCA/Momentum/MeanRev/Arbitrage) |
| **Paper Engine** | `strategies/paper_trading/` | WORKING — engine, optimizer, risk manager, regime detector, metrics |
| **Enterprise Integration** | `divisions/enterprise.py` | WIRED — polymarket_paper ↔ polymarket_council ↔ warroom |

**Gap:** Missing copy-trading, black swan hunting (AI-scored), ML-based market filtering, and institutional multi-agent debate strategies.

---

## 2. Polymarket Bot Strategy Taxonomy (Community-Sourced)

### Strategy 1: Black Swan Hunting (Favourite-Longshot Bias)
**Source:** `BekkiBay/polymarket_analyzer` (2★, Python)

**Core Concept:** Prediction markets systematically underprice tail risks. A 3% outcome may have true probability of 6-8%.

**EV Formula:**
```
EV = P_real × (1/P_market - 1) - (1 - P_real)
When P_real > P_market × 1.5 and EV > 0 → candidate
```

**Architecture — 4-Level Filter Pipeline:**
```
~20,000 raw markets
    ↓ Level 1 — Price filter (Sniper: $0.01-$0.10, Conveyor: $0.01-$0.15)
    ↓ Level 2 — Liquidity & time (volume ≥ $500, 1-90 days to expiry)
    ↓ Level 3 — Keyword filter (blacklist sports, whitelist geopolitics)
    ↓ Level 4 — AI classification (Gemini Flash → DuckDuckGo search → score)
    → ~5-15 candidates per cycle
```

**Dual Mode:**
- **Sniper** (50% budget): Rare asymmetric geopolitics/macro/crypto black swans, max $1.00/bet
- **Conveyor** (50% budget): High-volume sports/esports underdogs, max $0.50/bet

**Special Modes:**
- **Vulture Mode**: Detects 50%+ price jumps in 24h on cheap markets → immediate AI scoring
- **Panic Monitor**: Watches large markets (volume >$10k) for 15pp+/hr price changes or 3+ correlated moves

**Performance Expectations:**
- Win rate: 15-25%
- Average payout when win: 15-30x (buying at 3-7¢)
- One correct call at $0.20 bet → $6.60 profit → covers 33 losing bets

**AAC Integration:** Our `polymarket_blackswan_scanner.py` already does thesis-aligned scanning. Add the AI scoring pipeline (Gemini classify → DuckDuckGo enrich → Claude score) and Vulture/Panic modes.

---

### Strategy 2: AI-Powered Copy Trading
**Sources:** `codinglain/polymarket-ai-copytrade` (6★), `Drakkar-Software/OctoBot-Prediction-Market` (55★)

**Architecture:**
1. **Scan Top Traders** — Analyze ROI, win rate, trade size, consistency from Polymarket leaderboard
2. **AI Risk Filter** — Detect risky or low-confidence trades before copying
3. **Copy Execution** — Automatically replicate selected trades
4. **Adaptive Learning** — Improve strategy selection over time

**OctoBot Features:**
- Self-custody (keys never leave device)
- Paper trading mode for testing strategies
- Visual web UI + Telegram interface
- Copy trading + arbitrage strategies
- Docker one-click deploy

**Viral Case Study (Substack, 611 likes):**
> "Someone told an AI agent to find the best trader on Polymarket and copy their bets. 18 hours later, $900 became $7,200."

**AAC Integration:** Build a `CopyTradingStrategy` in paper trading that:
- Monitors Polymarket leaderboard via Data API
- Ranks traders by Sharpe-like metrics (not just ROI)
- AI-filters before mirroring positions
- Tracks copy performance vs source trader

---

### Strategy 3: ML-Scored Cheap Token Trading (20-min Hold)
**Source:** `alexpahhatajev/polymarket-bot` (1★, Python — most honest/detailed real-world results)

**Strategy:** Pure Hold 20min + ML
1. Scan ~1000 active markets every 5 seconds via Gamma API
2. Filter: cheap tokens (1-20¢), wide spreads (1¢+), high volume ($10k+/day)
3. ML Score: gradient boosting model on 31 features
4. Bid at `best_bid + 0.001`
5. Hold exactly 20 minutes
6. Sell at current bid
7. Emergency exit if price drops 50%

**31 ML Features (Gradient Boosting):**
- Numeric (12): entry price, spread, bid depth, ask depth, depth ratio, 24h volume, liquidity, 1h/1d price change, days to resolution, market age, hour of day
- Text category (19): topic classifiers (mrbeast, elon, trump, bitcoin, iran, crude oil, gold, fed, crypto, weather, elections, military, etc.)

**Critical Real-World Lessons:**
- **Preview ≠ Live**: Preview showed +15-25% returns; live showed -5% to -12%
- **Adverse selection is real**: When our bid gets filled, it's usually because someone with better info is dumping
- **Patient holding beats active selling**: Every stop-loss/trailing-stop mechanism tested REDUCED returns
- **Arbitrage doesn't exist for retail**: Professional bots close YES+NO gaps within milliseconds
- **Speed matters**: 5s polling is far too slow for competitive market making (WebSocket is minimum)

**AAC Integration:** Critical lessons for our ArbitrageStrategy — don't rely on YES+NO<$1 arb (bots close instantly). Focus on informational edges instead.

---

### Strategy 4: Cross-Market Prediction Arbitrage
**Sources:** `openclaw-cross-market-arbitrage-agent`, `AstroTick` (Kalshi)

**Concept:** Same event on different platforms (Polymarket vs Kalshi) may have different prices. Buy cheap side on one, sell expensive side on other.

**AstroTick (Kalshi 15-min BTC):**
```
Kalshi Orderbook
    ↓
strategy.py → momentum score + orderbook skew
    ↓
openclaw_client.py → enriches signal with AI context
    ↓
├─ AGENT_MODE=false: enriched composite_signal → bot.py decides
└─ AGENT_MODE=true: agent auto-executes if confidence > threshold
```

**Config:**
- `STOP_LOSS_CENTS`: 20 (max loss before exit)
- `TAKE_PROFIT_CENTS`: 30 (target profit)
- `MAX_DAILY_LOSS_CENTS`: 1000
- `MAX_DAILY_TRADES`: 20
- `OPENCLAW_CONFIDENCE_THRESHOLD`: 0.72

**Early Exit Logic:**
- Stop-loss on contract price drop
- Take-profit on price rise
- Signal reversal exits (holding YES but signal flips to NO)
- Daily risk limits block new entries

**AAC Integration:** Cross-platform arb requires Kalshi API (not configured). But the signal enrichment pattern (rule-based signal → AI enrichment → composite score) maps well to our OpenClaw bridge.

---

### Strategy 5: Official Polymarket Agents Framework
**Source:** `Polymarket/agents` (2.8k★, 638 forks — THE official framework)

**Architecture:**
- `Gamma.py` — Market/event metadata from Gamma API
- `Polymarket.py` — Order execution via CLOB DEX
- `Chroma.py` — ChromaDB for vectorizing news/data sources (RAG)
- `Objects.py` — Pydantic models (trades, markets, events)

**Key Features:**
- Local and remote RAG (Retrieval-Augmented Generation)
- Data sourcing from betting services, news, web search
- LLM prompt engineering tools
- CLI: `get-all-markets`, trade execution, news queries

**Related Repos:**
- `py-clob-client` — Python CLOB client
- `python-order-utils` — Order signing/building utilities

**AAC Integration:** We reference the official Polymarket CLOB in `agents/polymarket_agent.py`. The RAG approach (vectorize news → inform predictions) is the key technique AAC should adopt for thesis-aligned betting.

---

### Strategy 6: End-of-Window Micro-Profit (BTC-Correlated)
**Source:** `JittoJoseph/Strategic-Market-Engine` (1★, TypeScript)

**Concept:** Monitor BTC price movements and execute automated strategies on prediction markets in the final minutes of binary outcome windows.

**Logic:** BTC up/down 15-min markets have predictable momentum patterns near expiry. If BTC is trending strongly with 2-3 minutes left, the outcome is partially determined but the market hasn't fully priced it.

**AAC Integration:** Could add a `LastMinuteStrategy` to paper trading that watches BTC price momentum and enters prediction markets near expiry windows.

---

### Strategy 7: Multi-Agent Institutional Simulation
**Source:** `KylinMountain/TradingAgents-AShare` (141★, 48 forks)

**14-Agent Architecture:**
- **Analyst Team** (6 agents): Fundamentals, sentiment, news, technical, macro, main capital flow
- **Researcher Team** (3 agents): Bull researcher, bear researcher, research director (adjudicates via structured debate)
- **Decision & Risk** (5 agents): Trader, aggressive risk, conservative risk, neutral risk, portfolio manager

**Debate Workflow:**
```
6 Analysts → Structured Analysis
    ↓
Bull Researcher → Claim-driven arguments FOR
Bear Researcher → Claim-driven arguments AGAINST
    ↓
Research Director → Adjudicates (red-blue adversarial)
    ↓
Trader → Executable plan
    ↓
3 Risk Debaters → Aggressive/Conservative/Neutral debate
    ↓
Portfolio Manager → Final verdict
```

Published as `tradingagents-analysis` skill on ClawHub.

**AAC Integration:** Maps directly to AAC's 11 Enterprise divisions. The bull/bear debate pattern should inform how PolymarketCouncil makes decisions before sending to PolymarketPaperDivision.

---

## 3. AAC-Specific Implementation Priorities

### Priority 1: AI-Scored Black Swan Enhancement
**Effort:** Medium | **Impact:** High
- Enhance existing `polymarket_blackswan_scanner.py` with Gemini/Claude scoring pipeline
- Add Vulture Mode (price spike detection) and Panic Monitor (cluster crash detection)
- Use `shared/data_sources.py` for news enrichment before AI scoring

### Priority 2: Copy Trading Strategy
**Effort:** Medium | **Impact:** High
- New strategy class `CopyTradingStrategy` in `strategies/paper_trading/strategies.py`
- Uses Polymarket Data API leaderboard endpoint
- AI-filtered trader selection (not just top ROI)
- Paper trade first, then wire to live

### Priority 3: ML Feature Engineering
**Effort:** High | **Impact:** Medium
- 31-feature gradient boosting model for market filtering
- Key insight: topic/category features (19 text features) are as important as numeric features
- Train on paper trading history from `data/paper_trading/polymarket/`

### Priority 4: Signal Enrichment Pipeline
**Effort:** Medium | **Impact:** Medium
- Pattern: rule-based signal → websearch enrichment → LLM scoring → composite signal
- Integrates with existing OpenClaw skill bridge
- Configurable confidence threshold for auto-execution vs manual review

### Priority 5: Adverse Selection Awareness
**Effort:** Low | **Impact:** High (prevents losses)
- **Critical lesson from live testing:** Fills are adversely selected — when you get filled, price is likely going down
- Don't rely on YES+NO < $1 arb — bots close in milliseconds
- Patient holding (20 min) beats every form of active selling tested
- Build this into paper trading evaluation metrics

---

## 4. Key Community Lessons (Hard-Won)

| Lesson | Source | Implication for AAC |
|---|---|---|
| Preview ≠ Live performance | polymarket-bot | Always discount paper trading results by 15-25% |
| Adverse selection kills fills | polymarket-bot | Track fill quality, not just win rate |
| Arb gaps close in milliseconds | polymarket-bot | Our 5-min cycle ArbitrageStrategy won't catch arb |
| Favourite-longshot bias = edge | polymarket_analyzer | Focus on 1-10¢ outcomes with true probability > market |
| Copy trading works (viral $900→$7200) | Substack | Leaderboard analysis + AI filter = viable strategy |
| Speed < Information advantage | Multiple | AAC should compete on thesis quality, not speed |
| Vulture mode (price spikes) = alpha | polymarket_analyzer | Add spike detection to blackswan scanner |
| 31-feature ML model filters 30-40% bad trades | polymarket-bot | Topic/category features matter as much as price features |
| Self-custody > cloud custody | OctoBot | AAC already does this (local execution) |
| Human override on execution | polymarket_analyzer | Never fully auto-execute on prediction markets |

---

## 5. Repo Reference Table

| Repo | Stars | Language | Strategy Type | Key Technique |
|---|---|---|---|---|
| **Polymarket/agents** | 2,800 | Python | Framework | Official SDK, RAG, LLM tools |
| **TradingAgents-AShare** | 141 | Python | Multi-agent debate | 14 agents, bull/bear debate, risk debate |
| **OctoBot-Prediction-Market** | 55 | Python | Copy trading + Arb | Self-custody, visual UI, paper trading |
| **AstroTick** | 9 | Python | BTC prediction (Kalshi) | OpenClaw enrichment, momentum+orderbook |
| **polymarket-ai-copytrade** | 6 | Python | Copy trading | AI risk filter, adaptive learning |
| **polymarket_analyzer** | 2 | Python | Black swan hunting | 4-level filter, Gemini+Claude pipeline |
| **polymarket-bot** | 1 | Python | ML cheap token | 31-feature model, 20-min hold, adverse selection lessons |
| **Strategic-Market-Engine** | 1 | TypeScript | End-of-window micro-profit | BTC-correlated, last-minute momentum |
| **polymarket-trading-agent** | 0 | Rust | Market monitoring | Alerts, arbitrage support, local desktop |
| **cross-market-arbitrage** | 0 | TypeScript | Cross-platform arb | Polymarket vs Kalshi price gap detection |

---

## 6. Substack Intelligence (Trading + AI)

| Article | Author | Engagement | Key Insight |
|---|---|---|---|
| "900+ Hours of Claude Code for Trading" | Roman Blackwood | 163 likes | AI agent workflow for systematic trading |
| "I Built a Self-Updating Trading Bot" | LearnAIWithMe | — | Karpathy's method + NotebookLM for strategy evolution |
| "Skill That Makes All My Other Skills Better" | The AI Maker | 98 likes | Autoresearch meta-skill for continuous improvement |
| "Polymarket copy-trade $900→$7,200" | Opinion AI | 611 likes | AI copies best trader, 8x return in 18 hours |
| "Where AI fits in trading" | Nithin Kamath/Zerodha | 76 likes | Skeptical view — speed advantage is the only real edge |
| "Free OpenClaw compute ladder" | OpenClaw official | 588 likes | 80% free, 20% smart escalation for cost control |
| "Skill Graphs solve context window" | Linas Beliūnas | 47 likes | Architecture pattern for complex agent workflows |

---

## 7. Next Steps

1. [ ] Enhance `polymarket_blackswan_scanner.py` with AI scoring pipeline
2. [ ] Add `CopyTradingStrategy` to paper trading strategies  
3. [ ] Build ML feature engineering pipeline for market filtering
4. [ ] Add Vulture Mode (price spike detection) to scanner
5. [ ] Add Panic Monitor (cluster crash detection) to scanner
6. [ ] Update adverse selection awareness in paper trading metrics
7. [ ] Evaluate TradingAgents bull/bear debate pattern for council decisions

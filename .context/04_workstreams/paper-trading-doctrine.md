# Paper Trading Division — Goals, Mandates & Success Protocols

> **Created:** 2026-04-08
> **Divisions:** `polymarket_paper`, `crypto_paper`
> **Engine:** `strategies/paper_trading/` (engine, strategies, optimizer)
> **Gate:** PAPER (Gate 2 in BakeoffEngine)

---

## 1. MISSION

Find, validate, and rank the **best algorithmic trading strategies** for
Polymarket prediction markets and crypto spot markets — using simulated
capital only — before any real money is deployed.

**No real orders.  No real risk.  Pure signal.**

---

## 2. MANDATES (Non-Negotiable Rules)

| # | Mandate | Enforcement |
|---|---------|-------------|
| M1 | **Zero real capital at risk** | `persist=True` writes JSON only; no exchange connectors called |
| M2 | **Deterministic execution** | Slippage is fixed-bps, fees are fixed-rate — no `random.random()` |
| M3 | **Continuous scoring** | StrategyOptimizer runs every cycle; composite score decides promote/throttle/disable |
| M4 | **Data from councils only** | Paper traders consume INTEL_UPDATE signals — never scrape APIs directly |
| M5 | **Doctrine-gated** | `can_execute()` must return True; HALT/SAFE_MODE stops all paper activity |
| M6 | **Full audit trail** | Every fill, score, and allocation change persisted to `data/paper_trading/` |
| M7 | **Strategy isolation** | Each strategy's P&L tracked independently; no cross-contamination |

---

## 3. GOALS — Phase Targets

### Phase 1: BOOTSTRAP (Week 1-2) — Current
**Objective:** Get both paper divisions running, consuming council intel, executing simulated trades.

| Goal | Metric | Target |
|------|--------|--------|
| G1.1 | Both divisions online in Enterprise | 11 divisions, all HEALTHY |
| G1.2 | Consume intel signals end-to-end | ≥1 INTEL_UPDATE consumed per division per hour |
| G1.3 | Execute first paper trades | ≥10 paper fills per division |
| G1.4 | Persist state to JSON | `data/paper_trading/polymarket/`, `data/paper_trading/crypto/` files exist |
| G1.5 | All tests green | pytest suite passes with paper trading tests included |

### Phase 2: CALIBRATE (Week 3-4)
**Objective:** Tune strategy parameters. Establish baseline performance. Identify which strategies suit which market.

| Goal | Metric | Target |
|------|--------|--------|
| G2.1 | Each strategy ≥20 trades | Enough data for meaningful scoring |
| G2.2 | Optimizer composite scores populated | All 5 poly + 4 crypto strategies scored |
| G2.3 | Identify clear winner per division | Best strategy has ≥10pt composite lead |
| G2.4 | Max drawdown under control | No strategy >25% drawdown |
| G2.5 | Parameter sweep documented | At least 1 config variation tested per strategy |

### Phase 3: COMPETE (Week 5-8)
**Objective:** Strategies compete head-to-head. Capital allocation adapts in real-time. Losers disabled.

| Goal | Metric | Target |
|------|--------|--------|
| G3.1 | Capital allocation proportional to score | Top strategy gets >30% allocation |
| G3.2 | At least 1 strategy disabled by optimizer | Proves auto-disable works |
| G3.3 | At least 1 strategy promoted | Composite >50, status "active" |
| G3.4 | Portfolio-level positive P&L | Combined paper equity > starting balance |
| G3.5 | Sharpe ratio >0.5 for best strategy | Risk-adjusted returns are viable |

### Phase 4: VALIDATE (Week 9-12)
**Objective:** Confirm consistency. Prepare for Gate 3 (PILOT) promotion with real micro-capital.

| Goal | Metric | Target |
|------|--------|--------|
| G4.1 | 500+ total trades per division | Statistical significance |
| G4.2 | Best strategy win rate >55% | Sustained edge proven |
| G4.3 | Max drawdown <15% over 30 days | Controlled risk |
| G4.4 | Sharpe >1.0 for top strategy | Institutional-grade signal |
| G4.5 | Gate 3 (PILOT) checklist ready | BakeoffEngine validation passes |

### Phase 5: PILOT (Week 13+) — Future
**Objective:** Deploy winning strategy with **micro real capital** ($50-$100) on Polymarket or crypto exchange.

| Goal | Metric | Target |
|------|--------|--------|
| G5.1 | Live fills match paper fills within 20% | Execution quality verified |
| G5.2 | P&L positive after 50 real trades | Real edge confirmed |
| G5.3 | No safety incidents | Zero unexpected losses, zero API errors causing fills |

---

## 4. STRATEGY ROSTER & EXPECTATIONS

### Polymarket Paper (`polymarket_paper`) — $10,000 virtual

| Strategy | Bot Type | Market Fit | Expected Edge |
|----------|----------|------------|---------------|
| `poly_grid` | Grid | Range-bound markets (50-70¢ range) | Oscillation capture |
| `poly_dca` | DCA | High-conviction positions | Dip accumulation |
| `poly_momentum` | Trend-Following | Trending markets (election, macro events) | Trend-riding |
| `poly_mean_rev` | Mean-Reversion | Overreaction fades | Snap-back capture |
| `poly_arb` | Arbitrage | YES+NO < $1.00 mispricing | Risk-free edge |

**Expected winner:** `poly_arb` (structural mispricing) or `poly_momentum` (event-driven trends).

### Crypto Paper (`crypto_paper`) — $10,000 virtual

| Strategy | Bot Type | Market Fit | Expected Edge |
|----------|----------|------------|---------------|
| `crypto_grid` | Grid | BTC/ETH consolidation ranges | Range oscillation |
| `crypto_dca` | DCA | Systematic accumulation on dips | Dollar-cost averaging |
| `crypto_momentum` | Trend-Following | Altcoin breakouts | Sustained trend capture |
| `crypto_mean_rev` | Mean-Reversion | Volatile alt mean-reversion | Extreme move fading |

**Expected winner:** `crypto_dca` (systematic accumulation) or `crypto_momentum` (alt breakouts).

---

## 5. SUCCESS PROTOCOLS — How We Know It's Working

### 5.1 Health Check (Every Cycle)

```
✅ PASS if:
  - Division health = HEALTHY
  - Optimizer ran without exception
  - At least 1 strategy is "active" status
  - Account equity > 0
  - No strategy >25% drawdown

⚠️ DEGRADE if:
  - No intel received in last 3 cycles
  - All strategies "throttled"
  - Drawdown 15-25%

🛑 HALT if:
  - Account equity < 50% of starting balance
  - All strategies "disabled"
  - Doctrine state = HALT or SAFE_MODE
```

### 5.2 Weekly Review Protocol

Every Sunday, evaluate:

1. **Performance Dashboard**
   - Total P&L per division (target: >0 by week 4)
   - Best strategy name + composite score
   - Worst strategy name + composite score
   - Number of active/throttled/disabled strategies

2. **Strategy Scorecard**
   - Win rate per strategy (target: >50% by week 4)
   - Average PnL per trade (target: >0)
   - Sharpe ratio (target: >0.5 by week 6)
   - Max drawdown (target: <20%)

3. **Optimizer Health**
   - Are scores converging or oscillating?
   - Is capital allocation shifting toward winners?
   - Are disabled strategies truly underperforming?

4. **Action Items**
   - Tune parameters for underperforming strategies
   - Consider adding new strategy variants
   - Document findings in `.context/04_workstreams/`

### 5.3 Gate Promotion Criteria (PAPER → PILOT)

A strategy is eligible for real-capital PILOT when ALL of:

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Trade count | ≥100 | Total closed trades |
| Win rate | ≥55% | Winning trades / total trades |
| Total P&L | >0 | Net positive after fees |
| Sharpe ratio | ≥1.0 | Annualized risk-adjusted return |
| Max drawdown | <15% | Peak-to-trough equity |
| Composite score | ≥50 | StrategyOptimizer composite |
| Status | "active" for ≥7 consecutive days | Not throttled/disabled recently |
| Consecutive losing streak | <10 | No 10+ consecutive losing trades |

**Promotion process:**
1. Strategy meets all thresholds for 7+ consecutive days
2. Generate BakeoffEngine Gate 3 (PILOT) validation report
3. Human reviews and approves
4. Deploy with $50-$100 micro-capital, DRY_RUN=false
5. Monitor first 50 real trades for execution quality

### 5.4 Kill Switch Protocol

Emergency stop conditions (auto-enforced):

| Condition | Action |
|-----------|--------|
| Single trade loss >5% of account | Disable that strategy |
| Account equity drops >30% | Halt all paper trading |
| 20 consecutive losses (any strategy) | Disable that strategy |
| Doctrine state → HALT | Stop all division activity |
| No intel for 1 hour | Set division health → DEGRADED |

---

## 6. ROADMAP — 13-Week Plan

```
Week 1-2:  BOOTSTRAP ────────────────────────────
           ├─ Wire divisions into Enterprise      ✅
           ├─ First paper fills executing
           ├─ JSON persistence verified
           ├─ Tests green
           └─ Intel consumption confirmed

Week 3-4:  CALIBRATE ────────────────────────────
           ├─ All strategies have ≥20 trades
           ├─ Composite scores populated
           ├─ Parameter tuning round 1
           └─ Weekly review #1 + #2

Week 5-6:  COMPETE ──────────────────────────────
           ├─ Capital allocation adapting
           ├─ Weakest strategy auto-disabled
           ├─ Best strategy identified
           └─ Weekly review #3 + #4

Week 7-8:  REFINE ───────────────────────────────
           ├─ Parameter tuning round 2
           ├─ Add strategy variants if needed
           ├─ Sharpe >0.5 for top strategy
           └─ Weekly review #5 + #6

Week 9-10: VALIDATE ─────────────────────────────
           ├─ 500+ trades per division
           ├─ Win rate >55% for best
           ├─ Max drawdown <15%
           └─ Weekly review #7 + #8

Week 11-12: GATE REVIEW ─────────────────────────
           ├─ BakeoffEngine Gate 3 validation
           ├─ Generate promotion report
           ├─ Human approval gate
           └─ Prepare PILOT deployment

Week 13+:  PILOT ────────────────────────────────
           ├─ Deploy best strategy with $50-100
           ├─ Monitor execution quality
           ├─ Compare real vs paper fills
           └─ Scale decision (or kill)
```

---

## 7. TASK LIST — Immediate Actions

### Sprint 1 (This Week)

- [x] Build PaperTradingEngine (engine.py)
- [x] Build 5 strategy algorithms (strategies.py)
- [x] Build StrategyOptimizer (optimizer.py)
- [x] Build PolymarketPaperDivision
- [x] Build CryptoPaperDivision
- [x] Wire into Enterprise (11 divisions)
- [x] Define goals, mandates, success protocols (this file)
- [ ] Write comprehensive tests (test_paper_trading_divisions.py)
- [ ] Run full pytest suite — confirm green
- [ ] Smoke test: launch Enterprise, verify both divisions start

### Sprint 2 (Next Week)

- [ ] Run `launch.py monitor` — confirm paper divisions appear in dashboard
- [ ] Feed real council intel through signal pipeline
- [ ] Verify JSON persistence in `data/paper_trading/`
- [ ] First 10 paper fills per division
- [ ] First optimizer scoring cycle with real data
- [ ] Weekly review #1

### Sprint 3

- [ ] Parameter tuning: adjust grid range, DCA dip %, momentum lookback
- [ ] Add Polymarket-specific price extraction (condition_id → price mapping)
- [ ] Add CoinGecko price feed enhancement (more coins in intel)
- [ ] 50+ trades per division
- [ ] Weekly review #2

### Sprint 4

- [ ] Capital allocation adapting based on scores
- [ ] Identify best/worst strategy per division
- [ ] Document findings
- [ ] Consider adding Scalping or Market-Making strategy variants
- [ ] Weekly review #3

---

## 8. METRICS CANON — What We Measure

| Metric | Formula | Good | Warning | Critical |
|--------|---------|------|---------|----------|
| **Win Rate** | wins / total_trades × 100 | >55% | 45-55% | <45% |
| **Total P&L** | equity - starting_balance | >0 | -5% to 0 | <-5% |
| **P&L %** | total_pnl / starting_balance × 100 | >5% | 0-5% | <0% |
| **Max Drawdown** | (peak - current) / peak × 100 | <10% | 10-20% | >20% |
| **Sharpe Ratio** | mean_return / std_return | >1.0 | 0.5-1.0 | <0.5 |
| **Avg PnL/Trade** | total_pnl / total_trades | >0 | -$1 to 0 | <-$1 |
| **Composite Score** | weighted(PnL, WR, Sharpe, DD) | >50 | 10-50 | <10 |
| **Trade Frequency** | trades / cycles | 0.1-2.0 | <0.1 or >5 | 0 or >10 |

---

## 9. FILE MAP — Where Everything Lives

| File | Purpose |
|------|---------|
| `strategies/paper_trading/engine.py` | Core engine — orders, fills, positions, P&L |
| `strategies/paper_trading/strategies.py` | 5 strategy algorithms (Grid, DCA, Momentum, MeanRev, Arb) |
| `strategies/paper_trading/optimizer.py` | Scoring, ranking, capital allocation |
| `divisions/trading/polymarket_paper/division.py` | Polymarket paper trading division |
| `divisions/trading/crypto_paper/division.py` | Crypto paper trading division |
| `divisions/enterprise.py` | Wires all 11 divisions + subscriptions |
| `data/paper_trading/polymarket/` | Polymarket account state + scores (JSON) |
| `data/paper_trading/crypto/` | Crypto account state + scores (JSON) |
| `tests/test_paper_trading_divisions.py` | Test suite for paper trading |
| `.context/04_workstreams/paper-trading-doctrine.md` | This file — goals, mandates, roadmap |
| `aac/bakeoff/engine.py` | Gate progression engine (PAPER → PILOT) |

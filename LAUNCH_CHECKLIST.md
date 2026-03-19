# AAC Launch Checklist
**Generated: 2026-03-17**
**Version: v3.0-alpha → v3.1-launch-ready**

---

## Phase 0: Paper Trading Validation ✅ COMPLETE
- [x] Pipeline runner fetches live CoinGecko data
- [x] Fibonacci signal generator produces BUY/SELL/HOLD signals
- [x] ExecutionEngine fills paper trades with slippage model
- [x] CentralAccounting records transactions to SQLite
- [x] RiskManager enforces position limits, daily loss, concentration
- [x] 895 tests passing (0 failures)
- [x] 16/16 smoke import tests pass
- [x] DRY_RUN=true safety flag active

## Phase 1: API Keys & Data Sources ✅ COMPLETE
- [x] FRED — live validated (macro/economic data)
- [x] Polygon.io — live validated (stocks, options, crypto)
- [x] Finnhub — live validated (stocks, news, sentiment)
- [x] Alpha Vantage — live validated (fundamentals, forex)
- [x] NewsAPI — live validated (global news)
- [x] Etherscan — live validated (Ethereum on-chain)
- [x] CoinGecko Pro — configured (crypto prices)
- [x] Unusual Whales — validated (options flow, dark pool, congress)
- [x] OpenAI / Anthropic / Google AI / xAI — configured (AI/LLM)
- [ ] Tradier — needs signup (options chains, Greeks)
- [ ] Binance / Coinbase / Kraken exchange keys — need accounts

## Phase 2: Cross-Pillar Integration ✅ COMPLETE
- [x] `integrations/cross_pillar_hub.py` — NCC/NCL/BRS integration module
- [x] NCC governance gate in `pipeline_runner.py` — HALT/CAUTION/NORMAL modes
- [x] NCC governance gate in `execution_engine.py` — blocks trades on HALT
- [x] NCC risk multiplier applied to position sizing (0.5x in CAUTION)
- [x] NCL intelligence reader (file-based sync from NCL data path)
- [x] BRS pattern signal reader (file-based sync)
- [x] Pillar state persistence to `data/pillar_state/`
- [x] NCC_COORDINATOR_ENDPOINT, NCC_AUTH_TOKEN in .env

## Phase 3: Safety Controls ✅ COMPLETE
- [x] Kraken hard safety block (RuntimeError without KRAKEN_LIVE_CONFIRMED=yes)
- [x] DRY_RUN=true default in .env
- [x] PAPER_TRADING=true enforced in conftest.py
- [x] OneDrive paths eliminated (2 files fixed)
- [x] .env duplicate sections cleaned
- [x] market_data_aggregator.py — real CoinGecko wrapper (no longer returns 0)
- [x] order_generation_system.py — real validation (no longer approves everything)
- [x] RiskManager: 7 pre-trade checks active

## Phase 4: Limited Live Trading ⬜ NOT STARTED
Prerequisites before enabling:
- [ ] Exchange API keys configured (Binance or Coinbase)
- [ ] Set PAPER_TRADING=false (only after full paper validation)
- [ ] Set DRY_RUN=false
- [ ] Verify IBKR TWS connection (port 7497 paper → 7496 live)
- [ ] Test with minimum position size ($10-$50)
- [ ] Monitor for 48 hours with alerts
- [ ] Review all trade logs and P&L

## Phase 5: Production Hardening ⬜ NOT STARTED
- [ ] PostgreSQL migration (SQLite → production DB)
- [ ] Redis integration for caching/rate limiting
- [ ] Kafka for event streaming between pillars
- [ ] Docker deployment to cloud (compose already ready)
- [ ] Health endpoint monitoring (port 8080 ready)
- [ ] Incident response / kill switch
- [ ] Compliance / tax reporting
- [ ] Automated daily reports

---

## Known Risks
| Risk | Mitigation |
|------|-----------|
| Kraken has NO testnet | Hard RuntimeError blocks live orders without explicit confirmation |
| ~20/368 Python files actually functional | Non-critical files are stubs — core pipeline works |
| IBKR needs TWS running | Paper trading on port 7497; live requires manual switch |
| SQLite not production-grade | Sufficient for paper trading; PostgreSQL for Phase 5 |
| NCC/NCL servers not running | Hub gracefully degrades — falls back to file-based state |

## Quick Start
```bash
# Activate venv
.venv\Scripts\activate

# Run pipeline (paper trading)
python pipeline_runner.py

# Check API keys
python agents/api_key_agent.py --validate

# Run tests
python -m pytest tests/ --ignore=tests/security_integration_test.py -q --timeout=30

# Launch dashboard
python launch.py --mode dashboard
```

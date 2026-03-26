# AAC Integrations Registry

**Version:** 1.0  
**Jurisdiction:** Alberta, Canada  
**Priority Order:** Multi-Asset Arbitrage → Canadian Compliance → FX/CFD Execution  
**Last Updated:** 2026-03-07  

---

## Tiering Framework

| Tier | Purpose | Capital Policy | Recourse |
|------|---------|----------------|----------|
| **A — Core** | Treasury, arbitrage legs, primary execution | Full capital allowed | CIRO / CIPF protected |
| **B — Satellite** | Crypto rails, data feeds, secondary execution | Capped exposure | Regulated but limited |
| **C — Sandbox** | R&D, leverage experiments, CFD testing | Disposable capital only | Assume zero recourse |

---

## 1. Execution Venues (Brokers & Exchanges)

### Tier A — Core Capital & Arbitrage Backbone

#### Interactive Brokers (IBKR)
| Field | Value |
|-------|-------|
| **Tier** | A — Core |
| **Role** | Primary multi-asset execution, treasury, arbitrage backbone |
| **Regulation** | CIRO (Canada), SEC (US), FCA (UK), ASIC (AU) |
| **Assets** | Stocks, Options, Futures, Forex, Bonds, ETFs, Crypto |
| **API** | TWS API / IB Gateway (ib_insync) |
| **Connector** | `TradingExecution/exchange_connectors/ibkr_connector.py` |
| **Env Vars** | `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`, `IBKR_ACCOUNT`, `IBKR_PAPER` |
| **Paper Mode** | ✅ Port 7497 (paper) / 7496 (live) |
| **Status** | ✅ Production-grade (55 tests) |
| **Performance** | ⭐⭐⭐⭐ |
| **Efficiency** | ⭐⭐⭐⭐ |
| **Reliability** | ⭐⭐⭐⭐⭐ |

#### Moomoo (Futu) Canada
| Field | Value |
|-------|-------|
| **Tier** | A — Core |
| **Role** | Equity/options co-primary execution, Tier 1 alongside IBKR |
| **Regulation** | CIRO/CIPF via carrying broker (Canaccord) |
| **Assets** | US/HK/CN Stocks, Options |
| **API** | moomoo-api SDK via OpenD gateway (TCP, port 11111) |
| **Connector** | `TradingExecution/exchange_connectors/moomoo_connector.py` |
| **Env Vars** | `MOOMOO_API_KEY`, `MOOMOO_API_SECRET`, `MOOMOO_PAPER` |
| **Paper Mode** | ✅ TrdEnv.SIMULATE |
| **Status** | ✅ Implemented |
| **Performance** | ⭐⭐⭐ |
| **Efficiency** | ⭐⭐⭐ |
| **Reliability** | ⭐⭐⭐⭐ |
| **Note** | Requires OpenD gateway process running (extra ops overhead) |

---

### Tier B — Crypto Execution & CAD Rails

#### NDAX (National Digital Asset Exchange)
| Field | Value |
|-------|-------|
| **Tier** | B — Primary Crypto |
| **Role** | CAD ↔ crypto rails, Canadian crypto execution |
| **Regulation** | Canadian MSB, FINTRAC registered |
| **Assets** | BTC, ETH, XRP, LTC, EOS, DOGE, ADA, USDT (CAD pairs) |
| **API** | REST / WebSocket via CCXT |
| **Connector** | `TradingExecution/exchange_connectors/ndax_connector.py` |
| **Env Vars** | `NDAX_API_KEY`, `NDAX_API_SECRET`, `NDAX_USER_ID`, `NDAX_ACCOUNT_ID` |
| **Paper Mode** | ✅ Sandbox available |
| **Status** | ✅ Primary crypto connector |
| **Performance** | ⭐⭐⭐ |
| **Efficiency** | ⭐⭐⭐⭐ |
| **Reliability** | ⭐⭐⭐ |

#### Binance (REMOVED — Banned in Canada)
| Field | Value |
|-------|-------|
| **Tier** | ❌ Removed |
| **Role** | N/A — Binance exited Canada in June 2023 |
| **Regulation** | Banned by CSA/OSC for Canadian residents |
| **API** | CCXT (archived) |
| **Connector** | `_archive/binance_connector.py` (archived) |
| **Env Vars** | — |
| **Status** | ❌ Removed from all routing, defaults, and reconciliation |

#### Kraken (Deprioritized)
| Field | Value |
|-------|-------|
| **Tier** | B — Parked |
| **Role** | Crypto spot + margin (deprioritized) |
| **API** | CCXT |
| **Connector** | `TradingExecution/exchange_connectors/kraken_connector.py` |
| **Env Vars** | `KRAKEN_API_KEY`, `KRAKEN_API_SECRET` |
| **Status** | ✅ Implemented, deprioritized |

#### Coinbase Pro (Deprioritized)
| Field | Value |
|-------|-------|
| **Tier** | B — Parked |
| **Role** | Crypto spot (deprioritized) |
| **API** | CCXT |
| **Connector** | `TradingExecution/exchange_connectors/coinbase_connector.py` |
| **Env Vars** | `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `COINBASE_PASSPHRASE` |
| **Status** | ✅ Implemented, deprioritized |

---

### Tier C — Sandbox / High-Risk / Experimental

#### Noxi Rise (MT5)
| Field | Value |
|-------|-------|
| **Tier** | C — Sandbox Only |
| **Role** | Experimental MT5 strategies, leverage testing |
| **Regulation** | Anjouan (Comoros) — Tier 3 offshore |
| **Assets** | Forex, Indices, Commodities, Crypto CFDs |
| **API** | MetaTrader 5 Python API (Windows only) |
| **Connector** | `TradingExecution/exchange_connectors/noxi_rise_connector.py` |
| **Env Vars** | `MT5_PATH`, `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` |
| **Status** | ✅ Implemented |
| **Performance** | ⭐⭐⭐ |
| **Efficiency** | ⭐⭐ |
| **Reliability** | ⭐ |
| **WARNING** | No treasury funds. No arbitrage dependency. Assume zero recourse. |

#### OANDA (Optional FX Safety Valve)
| Field | Value |
|-------|-------|
| **Tier** | C — Optional |
| **Role** | Regulated FX anchor, compliance hedge |
| **Regulation** | CIRO (Canada), FCA (UK), CFTC/NFA (US) |
| **Assets** | FX pairs, CFDs |
| **API** | REST v20 API |
| **Connector** | `TradingExecution/exchange_connectors/oanda_connector.py` |
| **Env Vars** | `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_ENVIRONMENT` |
| **Status** | ⏳ Keys defined, connector not implemented |
| **Performance** | ⭐⭐⭐ |
| **Efficiency** | ⭐⭐⭐⭐ |
| **Reliability** | ⭐⭐⭐⭐⭐ |

#### IC Markets (Optional FX Satellite)
| Field | Value |
|-------|-------|
| **Tier** | C — Optional |
| **Role** | ECN/STP FX execution, algo/scalping |
| **Regulation** | ASIC (AU), CySEC (EU), FSA (Seychelles) |
| **Assets** | Forex, Indices, Commodities, Crypto CFDs |
| **API** | MetaTrader 5 Python API |
| **Connector** | `TradingExecution/exchange_connectors/icmarkets_connector.py` |
| **Env Vars** | `ICMARKETS_MT5_LOGIN`, `ICMARKETS_MT5_PASSWORD`, `ICMARKETS_MT5_SERVER` |
| **Status** | ⏳ Planned, not implemented |
| **Performance** | ⭐⭐⭐⭐⭐ |
| **Efficiency** | ⭐⭐⭐ |
| **Reliability** | ⭐⭐⭐ |

#### Pepperstone (Optional FX Satellite)
| Field | Value |
|-------|-------|
| **Tier** | C — Optional |
| **Role** | FX + indices, balance of leverage and regulation |
| **Regulation** | ASIC (AU), FCA (UK), CySEC (EU), BaFin (DE) |
| **Assets** | Forex, Indices, Commodities, Crypto CFDs |
| **API** | MetaTrader 5 Python API / cTrader |
| **Connector** | `TradingExecution/exchange_connectors/pepperstone_connector.py` |
| **Env Vars** | `PEPPERSTONE_MT5_LOGIN`, `PEPPERSTONE_MT5_PASSWORD`, `PEPPERSTONE_MT5_SERVER` |
| **Status** | ⏳ Planned, not implemented |
| **Performance** | ⭐⭐⭐⭐ |
| **Efficiency** | ⭐⭐⭐ |
| **Reliability** | ⭐⭐⭐ |

#### Metal X DEX
| Field | Value |
|-------|-------|
| **Tier** | B — DeFi |
| **Role** | Zero-gas on-chain CLOB (XPR Network) |
| **API** | REST + WebSocket |
| **Connector** | `TradingExecution/exchange_connectors/metalx_connector.py` |
| **Status** | ✅ Production (Vector 1) |

---

## 2. Market Data APIs

| API | Client File | Type | Key Env Var | Status |
|-----|-------------|------|-------------|--------|
| **CoinGecko** | `shared/data_sources.py` | Crypto prices/metadata | `COINGECKO_API_KEY` | ✅ Pro active |
| **Polygon.io** | `integrations/polygon_client.py` | US equities/options | `POLYGON_API_KEY` | ✅ Active |
| **Finnhub** | `integrations/finnhub_client.py` | Real-time quotes | `FINNHUB_API_KEY` | ✅ Active |
| **Tradier** | `integrations/tradier_client.py` | Options chains | `TRADIER_API_KEY` | ✅ Active |
| **FRED** | `integrations/fred_client.py` | Economic data | `FRED_API_KEY` | ✅ Active |
| **Fear & Greed** | `integrations/fear_greed_client.py` | Crypto sentiment index | — | ✅ Free |
| **Alpha Vantage** | — | Equities/FX | `ALPHA_VANTAGE_API_KEY` | ⏳ Key defined |
| **Twelve Data** | — | Equities/FX | `TWELVE_DATA_API_KEY` | ⏳ Key defined |
| **IEX Cloud** | — | US equities | `IEX_CLOUD_API_KEY` | ⏳ Key defined |
| **EODHD** | — | Historical data | `EODHD_API_KEY` | ⏳ Key defined |

---

## 3. Alternative Data & Sentiment

| API | Client File | Purpose | Key Env Var | Status |
|-----|-------------|---------|-------------|--------|
| **Unusual Whales** | `integrations/unusual_whales_client.py` | Options flow, dark pools | `UNUSUAL_WHALES_API_KEY` | ✅ Active |
| **Whale Alert** | `integrations/whale_alert_client.py` | On-chain whale txns | `WHALE_ALERT_API_KEY` | ✅ Active |
| **Santiment** | `integrations/santiment_client.py` | On-chain metrics | `SANTIMENT_API_KEY` | ✅ Active |
| **Google Trends** | `integrations/google_trends_client.py` | Search trends | — | ✅ No key |
| **NewsAPI** | `integrations/api_integration_hub.py` | News aggregation | `NEWS_API_KEY` | ✅ Active |
| **CoinMarketCap** | `integrations/api_integration_hub.py` | Crypto rankings | `COINMARKETCAP_API_KEY` | ✅ Active |
| **WallStreetOdds** | — | Market odds | `WALLSTREETODDS_API_KEY` | ⏳ Key defined |
| **TradeStie** | — | Reddit sentiment | `TRADESTIE_API_KEY` | ⏳ Key defined |

---

## 4. AI / LLM Providers

| Provider | Key Env Var | Primary Use | Status |
|----------|-------------|-------------|--------|
| **Anthropic Claude** | `ANTHROPIC_API_KEY` | Research agents, BigBrain | ✅ Active |
| **OpenAI GPT** | `OPENAI_API_KEY` | Fallback LLM | ✅ Active |
| **Google Gemini** | `GOOGLE_AI_API_KEY` | Research agent fallback | ✅ Active |
| **xAI Grok** | `XAI_API_KEY` | Real-time reasoning | ⏳ Configured |
| **OpenClaw/BARREN WUFFET** | Config-based | Skill-based agent orchestration | ✅ Active |

---

## 5. Social & Communication

| Platform | Key Env Var(s) | Purpose | Status |
|----------|--------------|---------|--------|
| **Reddit** | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | Social sentiment | ✅ Active |
| **Telegram** | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Alerts & notifications | ✅ Active |
| **Slack** | `SLACK_WEBHOOK_URL` | Team alerts | ✅ Active |

---

## 6. Web3 / DeFi (Metallicus Vectors)

| Vector | Component | Client File | Status |
|--------|-----------|-------------|--------|
| V1 | Metal X DEX | `integrations/metalx_client.py` | ✅ Production |
| V2 | XMD Cross-DEX Arb | `strategies/metalx_arb_strategy.py` | ✅ Active |
| V3 | XMD Treasury | `strategies/xmd_treasury_strategy.py` | ✅ Active |
| V4 | XPR Agents | `integrations/xpr_agents_bridge.py` | ✅ Configured |
| V5 | Metal Blockchain | `shared/data_sources.py` | ✅ Active |
| V6 | On-Chain Intel | `BigBrainIntelligence/xpr_intelligence_agent.py` | ✅ Active |
| V7 | Metal Pay | `integrations/metalpay_client.py` | ✅ Active |
| V8 | DeFi Yield | `strategies/metalx_yield_strategy.py` | ✅ Active |
| V10 | WebAuth | `integrations/webauth_client.py` | ✅ Active |
| V12 | DAO Governance | `agents/metal_dao_governance_agent.py` | ✅ Active |

---

## 7. Infrastructure & Security

| Component | File | Purpose |
|-----------|------|---------|
| **API Key Manager** | `shared/api_key_manager.py` | Encrypted key storage + rotation |
| **Secrets Manager** | `shared/secrets_manager.py` | Vault-style secret management |
| **Config Loader** | `shared/config_loader.py` | Unified env-to-config hydration |
| **Audit Logger** | `shared/audit_logger.py` | Tamper-evident event logging |
| **Audit Public Key** | `config/crypto/audit_public_key.pem` | Log signature verification |

---

## Summary Statistics

| Category | Active | Planned/Parked | Total |
|----------|--------|----------------|-------|
| Execution Venues | 6 | 5 | 11 |
| Market Data APIs | 6 | 4 | 10 |
| Alt Data / Sentiment | 6 | 2 | 8 |
| AI / LLM Providers | 4 | 1 | 5 |
| Social / Comms | 3 | 0 | 3 |
| Web3 / DeFi | 10 | 2 | 12 |
| **Total Integrations** | **35** | **14** | **49** |

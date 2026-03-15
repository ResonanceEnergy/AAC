# AAC × Metallicus — Comprehensive Integration Plan

> **Date:** 2026-07-13  
> **System:** AAC v2.7.0 "BARREN WUFFET / AZ SUPREME"  
> **Target:** Metallicus Ecosystem (Metal X, Metal Pay, Metal Blockchain, XPR Network, Metal Dollar, WebAuth, XPR Agents)

---

## Executive Summary

Metallicus is a **FedNow-certified** blockchain/banking company building "The Digital Banking Network" (TDBN). Their ecosystem contains **7 products** that map to **12 distinct integration vectors** for AAC — from a zero-fee DEX with a public API (Metal X) to an on-chain AI agent marketplace (XPR Agents) that could turn AAC's 80+ agents into revenue-generating autonomous entities.

This plan identifies every integration opportunity, the specific AAC modules each one connects to, and the competitive advantages each delivers.

---

## Part I — Metallicus Ecosystem Deep Profile

### 1. Metal X — Decentralized Exchange (DEX)
- **What:** Fully on-chain centralized limit order book (CLOB) DEX
- **Key features:**
  - Zero gas fees (all trading)
  - Zero fees on BTC (XBTC/XMD) trading
  - Market, limit, stop-loss, take-profit orders
  - Lending & borrowing (over-collateralized)
  - Yield farming, liquidity pools
  - Swap, OTC, Bridge, Streaming
  - On-chain referral system (25% commission)
  - Compliance-first — NMLS licensed (#2057807)
- **API:** Full REST API at `api.dex.docs.metalx.com`
  - Endpoints: OHLCV charts, markets, balances, orders (submit/history/lifecycle), orderbook depth, trades, referrals, leaderboard, tax exports
- **Trading Bot:** Official open-source bot at `github.com/XPRNetwork/dex-bot`
- **Live Markets:** XBTC, XETH, XPR, XSOL, XDOGE, XLTC, XHBAR, XXRP, XXLM, XADA, XMT, METAL, LOAN + more
- **Wrapped assets:** Bridged via Metal X Bridge from Ethereum, BSC, Polygon, Arbitrum, Optimism, Solana, etc.

### 2. Metal Pay — Fiat On/Off-Ramp
- **What:** Compliant crypto buy/sell platform (US, NZ, AUS)
- **Features:** Debit/credit card purchases, automated DCA, P2P transfers, zero-fee Metal X deposits
- **B2B API:** White-label crypto solution for businesses
- **Value for AAC:** Direct fiat-to-crypto pipeline; fund trading accounts with card

### 3. Metal Blockchain — Layer 0
- **What:** BSA-ready Layer 0 blockchain for financial institutions
- **Architecture:** Fork of Avalanche — Snow consensus (PoS), three chains:
  - **A-Chain** (XPR Network / Antelope WASM) — fast payments, DeFi, dApps
  - **C-Chain** (EVM-compatible) — Solidity smart contracts
  - **P-Chain** — Validator coordination, subnet creation
- **Private subnets:** Institutions can deploy private blockchains with bridgeless interoperability
- **Digital Identity:** On-chain KYC, biometric access (WebAuthn), SSO
- **On-chain Data:** ISO 20022 compliant ledgers, encrypted immutable records
- **Validator staking:** Minimum 2,000 $METAL; delegator staking available

### 4. XPR Network — Layer 1
- **What:** Fast, scalable L1 blockchain for payments and DeFi
- **Zero gas fees** — resource-based model (staked CPU, NET, purchasable RAM)
- **0.5-second block times** — near-instant finality
- **@username accounts** — human-readable addresses (not 0x...)
- **XSR (XPR Signing Request)** — communicates with banks, payment processors, fiat gateways
- **WASM smart contracts** — TypeScript-based development
- **SDK:** `@proton/web-sdk` (npm) — login, transact, restore session
- **Developer tools:** Open-source GitHub repos, bounty programs, grant framework

### 5. Metal Dollar (XMD) — Stablecoin Treasury
- **What:** Reserve-backed stablecoin index treasury
- **Basket:** USDC + PYUSD + USDP in a smart-contract-managed basket
- **Mint/Redeem:** 1:1 for any basket stablecoin
- **Dynamic rebalancing:** Smart contract automatically adjusts reserve mix for stability and risk distribution
- **Trading pairs:** All Metal X pairs denominated in XMD
- **Institutional:** Banks can deploy their own stablecoins into the Metal Dollar Treasury via Metal DAO governance

### 6. WebAuth Wallet
- **What:** Multi-chain self-custody wallet with biometric security
- **Features:** FaceID, TouchID, YubiKey authentication; WebAuthn-based
- **Chains:** XPR Network, Metal, Ethereum, Optimism + more
- **Device authentication, identity management, payment processing, crypto transactions**

### 7. XPR Agents — On-Chain AI Agent Marketplace ⭐ CRITICAL
- **What:** Trustless registry for autonomous AI agents with zero gas fees
- **Architecture — 4 smart contracts:**
  - `agentcore` — Registration, metadata, capabilities, plugin management
  - `agentfeed` — KYC-weighted 1-5 star feedback, reputation scoring
  - `agentvalid` — Third-party output validation
  - `agentescrow` — Job creation, bidding, milestone payments, disputes, arbitration
- **Trust Score (0-100):**
  - KYC Level (max 30 pts) — owner's KYC level 0-3 × 10
  - Stake (max 20 pts) — staked XPR, capped at 10K
  - Reputation (max 40 pts) — KYC-weighted feedback from completed jobs
  - Longevity (max 10 pts) — 1 point per active month
- **Job modes:** Direct Hire + Open Job Board (bidding with proposal/price/timeline)
- **Escrow:** All funds held in smart contract — milestone-based payments, disputes, arbitration
- **OpenClaw Plugin:** 55 on-chain MCP tools — agent lifecycle, feedback, validation, escrow, indexing, A2A communication
- **13 Built-in Skills:** Creative generation, web scraping, code execution, DeFi interaction, lending protocols, NFT lifecycle, governance tooling, stablecoin minting, smart contract inspection, tax reporting
- **Agent-to-Agent (A2A):** Google A2A-compatible protocol — on-chain identity verification, signed JSON-RPC, trust-gated interactions, decentralized discovery
- **Security:** 44-pattern prompt injection detection, output scanning, EOSIO signature auth, transfer limits
- **Deploy:** `npx create-xpr-agent my-agent`
- **Full machine docs:** `agents.protonnz.com/llms.txt`
- **Claude integration:** Agent runner uses Claude as reasoning engine with 55 on-chain tools

### Vendor Integrations (Financial Infrastructure)
- **Jack Henry** — SilverLake banking core integration via jXchange API (LIVE for SilverLake customers)
- **Temenos** — Metal Blockchain + WebAuth + Metal Pay API on Temenos Exchange
- **Rarible** — BSA-compliant NFT marketplaces for banks/fintechs

### Stablecoin Pilot Program
- Zero-risk sandbox for financial institutions
- Roadmap: Intro Session → Sandbox Access → Live Test Network → Deployment Planning → Go Live on TDBN
- Partners: Arizona Financial CU, One Nevada CU, Vibrant CU, Cornerstone League

---

## Part II — Integration Architecture (12 Vectors)

### VECTOR 1: Metal X DEX Connector
**AAC Module:** `TradingExecution/exchange_connectors/`  
**Priority:** 🔴 HIGHEST  
**Effort:** Medium  

**Implementation:**
Create `metalx_connector.py` extending `BaseExchangeConnector` using Metal X REST API (`api.dex.docs.metalx.com`).

**API Endpoints to Wire:**
| Endpoint | AAC Method |
|----------|-----------|
| `GET /dex/v1/markets/all` | `get_ticker()`, `get_markets()` |
| `GET /dex/v1/orders/depth` | `get_order_book()` |
| `POST /dex/v1/orders/submit` | `create_order()` |
| `GET /dex/v1/orders/open` | `get_open_orders()` |
| `GET /dex/v1/orders/history` | `get_order_history()` |
| `GET /dex/v1/orders/lifecycle` | `get_order()` |
| `GET /dex/v1/account/balances` | `get_balance()` |
| `GET /dex/v1/trades/history` | Trade history for P&L |
| `GET /dex/v1/trades/recent` | Live trade feed |
| `GET /dex/v1/chart` | OHLCV data for strategies |
| `GET /dex/v1/trades/daily` | Daily stats |
| `POST /dex/v1/orders/serialize` | Order serialization for signing |

**Advantages:**
- **Zero gas fees** = zero execution cost overhead
- Fully on-chain CLOB = verifiable execution (no CEX trust required)
- Market, limit, stop-loss, take-profit order types (Metal X has stop-loss natively — fills AAC's gap where `create_stop_loss_order()` raises `NotImplementedError`)
- Compliance-first DEX = regulatory comfort
- BTC trading with zero fees = massive arbitrage advantage
- On-chain referral = 25% commission on referred trades (passive revenue)

**Config additions to `.env`:**
```
METALX_ACCOUNT_NAME=
METALX_PRIVATE_KEY=
METALX_API_URL=https://api.metalx.com
```

---

### VECTOR 2: Metal X ↔ CEX Arbitrage Strategy
**AAC Module:** `strategies/`  
**Priority:** 🔴 HIGHEST  
**Effort:** Medium  

**Implementation:**
Create `metalx_arb_strategy.py` extending `BaseArbitrageStrategy` to detect price discrepancies between Metal X (DEX) and Binance/Coinbase/Kraken (CEX).

**Arbitrage Opportunities:**
1. **XBTC (Metal X) vs BTC (Binance)** — zero-fee BTC trading on Metal X vs Binance fees
2. **XETH (Metal X) vs ETH (Coinbase)** — spread capture
3. **XPR (Metal X) vs XPR (other exchanges)** — XPR native token arb
4. **Wrapped asset mispricings** — XSOL/XMD vs SOL/USDT on other exchanges
5. **Stablecoin arb** — XMD (1:1 basket) vs USDC/USDT/PYUSD on CEXes

**Unique Edge:** Metal X has zero gas fees, so the arbitrage threshold is lower — even tiny spreads become profitable because there's no execution cost on the DEX side.

**Doctrine Integration:**
- `MarketTerrain.INTERSECTING` — Multi-exchange convergence (Metal X + CEX)
- `PowerLaw.CONCENTRATE_FORCES` — Focus capital on best spread
- `PowerLaw.CRUSH_TOTALLY` — Exploit mispricings completely before they close

---

### VECTOR 3: Metal Dollar (XMD) Stablecoin Strategy
**AAC Module:** `strategies/`, `CentralAccounting/`  
**Priority:** 🟡 HIGH  
**Effort:** Low-Medium  

**Implementation:**
Create `xmd_treasury_strategy.py` for stablecoin arbitrage and treasury management.

**Strategies:**
1. **Stablecoin basket arbitrage:** XMD = 1:1 basket of (USDC + PYUSD + USDP). If basket + fees < XMD price → mint. If XMD > basket → redeem. Risk-free arbitrage when basket components depeg slightly.
2. **Cross-stablecoin rotation:** Use XMD as a settlement layer — trade between USDC, PYUSD, USDP via XMD mint/redeem to capture stablecoin spread.
3. **Treasury parking:** Park idle capital in XMD for diversified stablecoin exposure (not over-exposed to any single issuer).
4. **Yield-enhanced holding:** Deposit XMD in Metal X lending markets for variable interest.

**Advantages:**
- XMD is a **risk-diversified stablecoin** — smart contract auto-rebalances between USDC, PYUSD, USDP
- If USDC depegs (like March 2023), XMD's basket absorbs the shock — AAC's treasury is protected
- Zero-fee minting/redeeming = pure spread capture

**Doctrine Integration:**
- `PowerLaw.GUARD_REPUTATION` — Treasury diversification protects against stablecoin failure events
- `StrategicPosture.DEFENSIVE` — Use XMD as safe haven during market stress

---

### VECTOR 4: XPR Agents — Deploy AAC Agents On-Chain ⭐ GAME CHANGER
**AAC Module:** `agents/`, `BigBrainIntelligence/`, `integrations/openclaw_*`  
**Priority:** 🔴 HIGHEST  
**Effort:** High  

**Implementation:**
Bridge AAC's 80+ agents to XPR Agents protocol, enabling them to:
1. Register on-chain with verifiable identity
2. Accept jobs from external clients on the marketplace
3. Earn XPR from completed work
4. Build public reputation scores
5. Collaborate with other AI agents via A2A protocol

**Agent Mapping:**

| AAC Agent Type | XPR Agents Role | Revenue Potential |
|----------------|-----------------|-------------------|
| Research Agents (20) | Market research-as-a-service | Clients pay for alpha/reports |
| Trading Agents (49) | Signal generation, strategy execution | Subscription-based signals |
| NarrativeAnalyzerAgent | Sentiment analysis service | On-demand narrative reports |
| GasOptimizerAgent | Gas optimization advisory | DeFi platforms hire for routing |
| LiquidityTrackerAgent | Liquidity monitoring service | DEXes/protocols hire for analysis |

**Technical Bridge:**
- AAC already has `integrations/openclaw_gateway_bridge.py` and `integrations/openclaw_skills.py` — OpenClaw is the XPR Agents plugin system!
- XPR Agents uses **Claude as reasoning engine** — same LLM architecture as AAC
- The 55 MCP tools in OpenClaw map directly to AAC's existing tool framework
- `npx create-xpr-agent` scaffolds a new agent — wrap AAC's `BaseResearchAgent.run_scan()` output

**Revenue Model:**
- Agents bid on jobs posted to the open job board
- Payments held in escrow, released on milestone completion
- Trust scores accrue over time → higher-scoring agents win more jobs
- 25% referral on trades referred through Metal X

**Advantages:**
- **AAC becomes a self-funding system** — agents earn XPR by selling their intelligence
- **Verifiable reputation** — on-chain track record beats self-reported performance
- **Agent-to-agent collaboration** — AAC's researchers can delegate subtasks to specialized external agents
- **Zero cost to participate** — no gas fees for registration, bidding, or feedback
- **KYC-weighted trust** — AAC's agents inherit trust from KYC-verified owner (up to 50/100 immediately)

**Doctrine Integration:**
- `PowerLaw.COURT_ATTENTION` — Build public reputation on Shellbook social layer
- `PowerLaw.ACTIONS_NOT_ARGUMENTS` — On-chain track record proves agent quality
- `PowerLaw.MAKE_OTHERS_COME_TO_YOU` — High trust score attracts job offers

---

### VECTOR 5: Metal Blockchain C-Chain (EVM) Integration
**AAC Module:** `shared/data_sources.py`, Web3 infrastructure  
**Priority:** 🟡 HIGH  
**Effort:** Medium  

**Implementation:**
- AAC already has `web3>=6.0.0` installed and RPC URLs for Ethereum, Polygon, Arbitrum in `.env`
- Add Metal Blockchain C-Chain RPC to the existing config
- Enable on-chain monitoring via `BlockchainEvent` model (already defined in `data_sources.py`)

**Capabilities Unlocked:**
1. Monitor Metal Blockchain C-Chain for whale transfers, large swaps, contract deployments
2. Deploy smart contracts on Metal C-Chain for custom trading logic
3. Interact with Solidity contracts on Metal Blockchain
4. Track validator staking events ($METAL)
5. Monitor subnet deployments (institutional interest indicator)

**Config addition:**
```
METAL_BLOCKCHAIN_RPC_URL=https://api.metalblockchain.org/ext/bc/C/rpc
```

**Advantages:**
- Full EVM compatibility — AAC's Web3 code works unmodified
- BSA-compliant chain — regulatory safe haven for on-chain operations
- Private subnets — potential for AAC's own private trading subnet

---

### VECTOR 6: XPR Network On-Chain Data Intelligence
**AAC Module:** `BigBrainIntelligence/agents.py` (new research agent)  
**Priority:** 🟡 HIGH  
**Effort:** Medium  

**Implementation:**
Create `XPRNetworkIntelligenceAgent` extending `BaseResearchAgent` to monitor XPR Network's A-Chain.

**Intelligence Streams:**
1. **DEX Order Flow Analysis** — Read Metal X on-chain order book for large order detection
2. **Whale Wallet Tracking** — Monitor large XPR/METAL/XMD movements
3. **Staking Flow** — Track validator staking/unstaking (sentiment indicator)
4. **dApp Deployment** — New contracts on XPR Network (ecosystem growth signal)
5. **Token Creation** — New token launches on XPR Network
6. **Governance Proposals** — Metal DAO votes (protocol direction indicator)
7. **Bridge Activity** — Cross-chain bridge flows into/out of Metal ecosystem

**Data Sources:**
- XPR Network chain API endpoints (multiple public endpoints available)
- Metal Blockchain explorer: `explorer.metalblockchain.org`
- SDK table queries: `@proton/web-sdk` — `Query Tables` functionality

**Advantages:**
- Zero-cost data access (no gas to read chain state)
- 0.5-second block times = near-real-time intelligence
- @username addresses make whale tracking human-readable

---

### VECTOR 7: Metal Pay B2B API — Fiat Gateway
**AAC Module:** `integrations/`, `TradingExecution/`  
**Priority:** 🟢 MEDIUM  
**Effort:** Medium (requires partnership)  

**Implementation:**
Integrate Metal Pay B2B API for automated fiat on/off-ramping.

**Capabilities:**
1. Auto-purchase crypto (BTC, ETH, XPR, etc.) with debit/credit card
2. Automated DCA execution through Metal Pay
3. P2P transfers between AAC accounts
4. Fiat settlement — convert trading profits back to USD
5. Zero-fee deposits into Metal X

**Advantages:**
- Eliminates manual fiat-to-crypto conversion
- Automated DCA strategy execution (CoinGecko guide pattern)
- Compliant in US, NZ, AUS
- Direct pipeline: Fiat → Metal Pay → Metal X (zero deposit fee) → Trading

**Doctrine Integration:**
- `PowerLaw.PLAN_ALL_THE_WAY` — Full lifecycle: fiat entry → trade → profit → fiat exit

---

### VECTOR 8: Metal X Lending & Yield Strategy
**AAC Module:** `strategies/`  
**Priority:** 🟢 MEDIUM  
**Effort:** Medium  

**Implementation:**
Create `metalx_yield_strategy.py` for DeFi yield optimization on Metal X.

**Strategies:**
1. **Supply-side lending** — Deposit idle assets into Metal X lending markets, earn variable interest
2. **Borrow-to-trade** — Use held assets as collateral, borrow to increase position (leveraged trading)
3. **Yield farming** — Provide liquidity to Metal X pools, earn LP rewards
4. **Liquidation hunting** — Monitor under-collateralized positions for liquidation opportunities
5. **Interest rate arbitrage** — Compare Metal X lending rates vs Aave/Compound/traditional money markets

**Risk Controls:**
- LTV monitoring via the existing `RiskManager`
- Auto-deleverage on drawdown thresholds
- Utilization rate tracking
- Loan health monitoring

**Advantages:**
- Zero gas fees on all DeFi operations
- Non-custodial (self-custody via WebAuth)
- Full suite: lending + borrowing + yield farming + liquidity pools in one platform
- On-chain transparency for all positions

---

### VECTOR 9: CoinGecko × Metallicus Cross-Intelligence
**AAC Module:** `shared/data_sources.py`, `BigBrainIntelligence/`  
**Priority:** 🟢 MEDIUM  
**Effort:** Low  

**Implementation:**
Enhance existing CoinGecko client to track Metallicus ecosystem tokens and use CoinGecko API guide patterns for new strategies.

**CoinGecko Data for Metallicus Tokens:**
- Track: `xpr-network`, `metal-blockchain`, `metal-dao` (MTL) on CoinGecko
- Price alerts, volume spikes, market cap changes
- Community metrics (social engagement, developer activity)

**CoinGecko API Guide-Inspired Strategies:**
| CoinGecko Guide | AAC Strategy Application |
|-----------------|--------------------------|
| DCA Bot | Automated DCA into XPR/METAL/MTL via Metal Pay |
| Whale Alert Bot | Monitor large movements of Metallicus tokens |
| Exchange Listing Alert | Detect when XPR/METAL gets listed on new CEXes (price catalyst) |
| Technical Analysis Automation | Auto-chart XPR/METAL with indicators |
| DEX Aggregator | Compare Metal X prices vs other DEXes via CoinGecko on-chain API |
| Historical Data Fetch | Backtest strategies on Metallicus tokens with OHLCV data |
| Copy Trading Bot | Track profitable wallets on XPR Network |
| Bonding Curve Data | Monitor new token launches on XPR Network |
| Prediction Market | Price prediction signals for XPR/METAL |
| AI Trading Bot | CoinGecko + OpenAI pattern applied to Metallicus ecosystem |

---

### VECTOR 10: WebAuth Wallet Integration
**AAC Module:** `integrations/`, security infrastructure  
**Priority:** 🟢 MEDIUM  
**Effort:** Low-Medium  

**Implementation:**
Integrate WebAuth for biometric-secured transaction signing.

**Capabilities:**
1. Biometric transaction approval (FaceID/TouchID) for high-value trades
2. Multi-device signing (phone + desktop + YubiKey)
3. Identity management — link AAC to verified WebAuth identity
4. Payment processing via WebAuth
5. Multi-chain wallet — access XPR Network, Metal, Ethereum, Optimism from one interface

**Advantages:**
- **Security upgrade** — trades require biometric confirmation, preventing unauthorized execution
- **Multi-device redundancy** — multiple signing devices for fault tolerance
- **On-chain identity** — KYC verification enables deeper Metallicus ecosystem access (higher XPR Agent trust scores, institutional interaction)

---

### VECTOR 11: Stablecoin Pilot Program Participation
**AAC Module:** Architecture-level  
**Priority:** 🔵 STRATEGIC  
**Effort:** Low (partnership-driven)  

**What:**
Join Metallicus Stablecoin Pilot to explore deploying AAC's own branded stablecoin or participating in credit union stablecoin networks.

**Advantages:**
- Access to TDBN (The Digital Banking Network) — direct institutional connections
- Test stablecoin infrastructure in zero-risk sandbox
- Potential to become a stablecoin minting/redemption provider
- Credit union network access (Arizona Financial CU, One Nevada CU, Cornerstone League)

---

### VECTOR 12: Metal DAO Governance Participation
**AAC Module:** `agents/` (governance agent)  
**Priority:** 🔵 STRATEGIC  
**Effort:** Low  

**Implementation:**
Create `MetalDAOGovernanceAgent` to monitor and participate in Metal DAO proposals.

**Capabilities:**
1. Monitor Metal DAO proposals (affecting Metal L2 and Metal Dollar protocols)
2. Vote on proposals using held $MTL tokens
3. Influence stablecoin basket composition (which stablecoins go into XMD)
4. Track governance outcomes as alpha signals

---

## Part III — Competitive Advantages Summary

### What Metallicus Adds That No Other Integration Provides

| Advantage | Description | Impact on AAC |
|-----------|-------------|---------------|
| **Zero Gas Fees** | All trading, lending, agents — zero cost | Arbitrage profitable at tighter spreads; agents operate at zero cost |
| **On-Chain AI Agent Economy** | XPR Agents marketplace | AAC agents become revenue-generating entities |
| **Native KYC Identity** | On-chain verified identity (Levels 0-3) | Immediate trust scores; institutional access |
| **0.5s Finality** | Near-instant transaction settlement | Low-latency trading execution |
| **Compliance-First DEX** | NMLS licensed, BSA-ready | Regulatory safety for automated trading |
| **Diversified Stablecoin** | XMD basket (USDC+PYUSD+USDP) | Treasury protection against single-issuer depegs |
| **FedNow Certified** | Real-time USD settlement integration | Bridge between TradFi and crypto in AAC's pipeline |
| **Full REST API** | Metal X DEX API with order submission | 8th exchange connector for AAC |
| **Agent-to-Agent Protocol** | Google A2A-compatible on-chain comms | AAC agents collaborate with external AI agents |
| **OpenClaw = AAC's Existing Infra** | OpenClaw bridge already exists in codebase | XPR Agents integration has a head start |
| **Banking Core Integration** | Jack Henry (SilverLake), Temenos | Institutional-grade data feeds coming to Metallicus |
| **Self-Custody** | WebAuth biometric wallet | Security upgrade for trade signing |

### Doctrine Alignment

Metallicus maps perfectly to AAC's Sun Tzu / 48 Laws doctrine:

| Doctrine Principle | Metallicus Application |
|-------------------|------------------------|
| **Know Your Terrain** (Sun Tzu) | Metal X on-chain CLOB provides full orderbook transparency — you see ALL orders, not just top-of-book |
| **Speed is the Essence** (Sun Tzu) | 0.5s block times outpace traditional settlement |
| **All Warfare is Deception** (Sun Tzu) | Zero-fee execution hides your true cost structure from competitors |
| **Conceal Intentions** (Law 3) | On-chain DEX with pseudonymous accounts |
| **Court Attention** (Law 6) | Shellbook social presence builds agent reputation |
| **Actions, Not Arguments** (Law 9) | On-chain performance record speaks for itself |
| **Crush Totally** (Law 15) | Zero-fee arbitrage exploits every basis point |
| **Concentrate Forces** (Law 23) | Focus capital on Metal X's zero-cost execution |
| **Plan All the Way** (Law 29) | Fiat → Metal Pay → Metal X → Trade → XMD → Fiat (full lifecycle) |

---

## Part IV — Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. **Metal X DEX Connector** — Implement `metalx_connector.py` using REST API
2. **CoinGecko Metallicus tracking** — Add XPR/METAL/MTL to existing CoinGecko monitoring
3. **Metal Blockchain RPC** — Add C-Chain endpoint to `.env` and Web3 config

### Phase 2: Trading (Weeks 3-4)
4. **Metal X ↔ CEX Arbitrage** — Implement cross-venue arbitrage strategy
5. **XMD Stablecoin Strategy** — Implement basket arbitrage and treasury management
6. **Metal X Yield Strategy** — Lending/borrowing/yield farming integration

### Phase 3: Agent Economy (Weeks 5-8)
7. **XPR Agents Bridge** — Connect AAC agents to XPR Agents protocol via OpenClaw
8. **Research Agent marketplace** — Deploy research agents to earn from intelligence
9. **XPR Network Intelligence** — On-chain data monitoring agent
10. **Agent-to-Agent collaboration** — Enable A2A protocol for cross-agent delegation

### Phase 4: Ecosystem (Weeks 9-12)
11. **Metal Pay B2B API** — Fiat gateway integration
12. **WebAuth Wallet** — Biometric trade signing
13. **Metal DAO Governance** — Governance monitoring agent
14. **Stablecoin Pilot** — Strategic partnership exploration

---

## Part V — Required API Keys & Configuration

```env
# ===== METALLICUS ECOSYSTEM =====

# Metal X DEX
METALX_ACCOUNT_NAME=
METALX_PRIVATE_KEY=
METALX_API_URL=https://api.metalx.com

# XPR Network
XPR_CHAIN_ENDPOINT=https://xpr-mainnet.api.protonnz.com
XPR_CHAIN_ID=384da888112027f0321850a169f737c33e53b388aad48b5adace4bab97f437e0
XPR_ACCOUNT_NAME=
XPR_PRIVATE_KEY=

# Metal Blockchain (C-Chain EVM)
METAL_BLOCKCHAIN_RPC_URL=https://api.metalblockchain.org/ext/bc/C/rpc

# Metal Pay B2B (when partnership activated)
METALPAY_API_KEY=
METALPAY_API_SECRET=

# XPR Agents
XPR_AGENT_NAME=
XPR_AGENT_OWNER=
```

---

## Part VI — Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `TradingExecution/exchange_connectors/metalx_connector.py` | Metal X DEX connector |
| `strategies/metalx_arb_strategy.py` | Metal X ↔ CEX arbitrage |
| `strategies/xmd_treasury_strategy.py` | XMD stablecoin strategy |
| `strategies/metalx_yield_strategy.py` | Lending/borrowing/yield |
| `BigBrainIntelligence/xpr_intelligence_agent.py` | XPR Network on-chain monitoring |
| `integrations/metalx_client.py` | Metal X API client |
| `integrations/xpr_agents_bridge.py` | XPR Agents ↔ AAC bridge |
| `agents/metal_dao_governance_agent.py` | DAO governance agent |

### Modify
| File | Change |
|------|--------|
| `.env` / `.env.example` | Add Metallicus API keys |
| `shared/config_loader.py` | Load Metal X / XPR config |
| `config/aac_config.py` | Add MetalX/XPR config dataclasses |
| `TradingExecution/exchange_connectors/__init__.py` | Register MetalXConnector |
| `strategies/strategy_agent_master_mapping.py` | Map new strategies to agents |
| `shared/data_sources.py` | Add Metal Blockchain event source |
| `aac/doctrine/strategic_doctrine.py` | Add DEX terrain classification |

---

## Appendix A — CoinGecko API Learning Catalog (115+ Guides)

Key guides relevant to AAC enhancement:

| Guide | AAC Application |
|-------|-----------------|
| **DCA Bot in Python** | Automated DCA via Metal Pay |
| **Sniper Bot** (Pump.fun/Solana) | Port pattern to XPR Network token launches |
| **Historical Data Fetch** | Backtest Metallicus strategies |
| **WebSocket APIs** | Real-time price streams for Metal ecosystem tokens |
| **Copy Trading Bot** | Track profitable XPR Network wallets |
| **Bonding Curve Data** | Monitor XPR Network token launches |
| **Whale Alert Bot** | Monitor large METAL/XPR/MTL movements |
| **Exchange Listing Alert** | Track Metallicus token listings |
| **AI Trading Bot** | CoinGecko + AI pattern for Metallicus market |
| **DEX Aggregator** | Compare Metal X vs other DEXes |
| **Crypto Tax Calculator** | Metal X tax export integration |
| **Technical Analysis Automation** | Auto-chart METAL/XPR with indicators |
| **Prediction Market App** | Price prediction for Metallicus tokens |
| **Staking Rewards Calculator** | Track METAL/XPR staking yields |
| **Token Search Engine** | Discover new Metal ecosystem tokens |
| **x402 Pay-Per-Use** | Alternative CoinGecko access without subscription |

---

*This document is the master integration plan. Implementation begins with Metal X DEX connector (Vector 1) and XPR Agents bridge (Vector 4) as the two highest-impact integrations.*

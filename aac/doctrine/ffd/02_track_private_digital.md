# FFD Track 2: Private Digital Money
## Stablecoins, Tokenized Assets, Private Settlement Networks

**Last Updated:** 2026-03-15
**Evidence Level:** E0-E1 (Conceptual + Observed Patterns)

---

## THESIS

Privately-issued digital money (stablecoins) represents the fastest-growing
segment of the financial transition. At $316B market cap and $1.4T in annual
cross-border flows, stablecoins are already reshaping how money moves globally.
They are the bridge between legacy fiat and decentralized protocols — and the
battlefield where monetary sovereignty is being contested.

---

## 1. THE STABLECOIN LANDSCAPE

### Market Overview (as of October 2025)
- **$316 billion** total stablecoin market capitalization
- **$156 billion** daily trading volume
- **95%** are fiat-backed (vs. crypto-backed or algorithmic)
- **97%** of fiat-backed stablecoins are USD-denominated
- **$1.4 trillion** in cross-border flows during 2024 (IMF data)

### Dominant Players

#### USDT (Tether)
- **Status**: Largest stablecoin by market cap
- **Reserves**: Claims fiat-backing but NEVER completed a full audit (as of Feb 2026)
- **Risk**: Counterparty risk is the #1 concern in crypto
- **Usage**: Dominant in emerging markets, offshore trading, OTC settlements
- **FATF finding**: Increasingly used for illicit activity (ML, TF, sanctions evasion)
- **AAC Stance**: Monitor closely, limit direct exposure, trade the peg

#### USDC (Circle)
- **Status**: Second largest, fully regulated, transparent reserves
- **Reserves**: Regular attestations, primarily US Treasuries and cash
- **Risk**: De-pegged briefly during 2023 banking crisis (SVB exposure)
- **Usage**: DeFi integrations, institutional use, regulatory-preferred
- **AAC Stance**: Primary stablecoin for settled positions and DeFi deployment

#### RLUSD (Ripple)
- **Status**: Ripple's institutional-grade stablecoin
- **Integration**: Built for XRP Ledger and Ethereum
- **Target**: Cross-border payments, institutional settlement
- **AAC Stance**: Monitor adoption alongside XRP ODL corridors

#### EURC (Circle)
- **Status**: Euro-denominated stablecoin, $225M market cap
- **Significance**: Largest EUR stablecoin amid ECB sovereignty concerns
- **AAC Stance**: EUR/USD pair proxy, hedge for European exposure

### Emerging Stablecoins
- **Wyoming Frontier Stable Token**: First state-backed stablecoin (Jan 2026)
- **Abu Dhabi dirham stablecoin**: Sovereign wealth fund + First Abu Dhabi Bank (Apr 2025)
- **Japanese Yen stablecoin (JPYC)**: First yen-pegged, launched Oct 2025
- **European bank consortium stablecoin**: 9 banks, MiCAR-compliant (Sep 2025)

---

## 2. STABLECOINS AS GEOPOLITICAL WEAPONS

### Dollar Dominance Extension
- GENIUS Act (July 2025): Banks can issue stablecoins backed by US Treasuries
- Fed Governor Miran: stablecoins are "contributing to the dollar's dominance"
- BIS estimate: demand lowers 3-month T-bill yields by 2-2.5 basis points
- Effect comparable to small-scale quantitative easing
- **Strategic implication**: US is weaponizing stablecoins to finance national debt

### Sovereignty Under Threat
- **ECB Adviser Schaaf**: "Wide adoption of US dollar stablecoins provides US with
  strategic and economic advantages. Europeans would experience higher borrowing
  costs, reduced monetary policy autonomy, and geopolitical dependency."
- **South African Reserve Bank Governor Kganyago**: USD stablecoins are
  "undermining African currencies"
- **Bank of France**: Monetary sovereignty "threatened" by USD stablecoins
- **China**: Banned stablecoins entirely; pushing digital yuan (e-CNY)
- **Standard Chartered warning**: $1 trillion could flow from developing countries
  to stablecoins, depleting bank deposits

### AAC Intelligence Implications
This is not just a financial transition — it's a **monetary cold war**.
Countries will respond with:
1. Their own stablecoins (EU bank consortium, Abu Dhabi)
2. CBDCs (digital euro, e-CNY)
3. Outright bans (China's approach)
4. Regulated acceptance (Singapore, Hong Kong, Japan)

**Each response type creates different trading opportunities.**

---

## 3. RISK ANALYSIS

### De-Pegging Risk (CRITICAL)
| Event | Stablecoin | Loss | Duration | Cause |
|---|---|---|---|---|
| May 2022 | TerraUSD | $45B | 1 week | Algorithmic death spiral |
| June 2021 | IRON | ~$2B | Days | Partial collateral run |
| March 2023 | USDC | ~3% depeg | 3 days | SVB bank failure |

**AAC Response Protocol**:
- Real-time peg monitoring (deviation > 0.5% = alert, > 2% = auto-exit)
- Position sizing: no single stablecoin > 30% of stablecoin allocation
- Counterparty diversification across issuers AND custodians
- Kill switch: if Tether depegs > 5%, halt ALL stablecoin strategies

### Counterparty Risk
- Tether's refusal to complete audits is the single largest systemic risk in crypto
- Reserve concentration creates bank-run vulnerability
- **Mitigation**: Weight toward audited issuers (USDC, RLUSD), geographic diversification

### Quantum Computing Risk
- IMF analysis (Hélène Rey, September 2025): quantum computers may break
  public-key cryptography used by stablecoin networks
- Timeline: uncertain but advancing rapidly
- **Mitigation**: Monitor quantum computing milestones, prepare migration plan
  for quantum-resistant chains

### Regulatory Shock Risk
- Any major jurisdiction ban = market-moving event
- China banned stablecoins entirely (2025-2026)
- CSRC banned Renminbi-denominated stablecoins specifically (Feb 2026)
- **Mitigation**: Regulatory event detection system, geographic risk scoring

---

## 4. YIELD LANDSCAPE

### Centralized Yield (CASPs)
- Trading platforms offer yield on stablecoin deposits
- Mechanism: platform lends stablecoins to institutional borrowers
- **Regulatory status**: Prohibited in EU and Hong Kong; restricted in Singapore
- **Risk**: Platform insolvency (FTX precedent)

### Decentralized Yield (DeFi Farming)
- Provide stablecoin liquidity to DEX pools
- Earn trading fees + protocol incentives
- **Risk**: Impermanent loss, smart contract exploits, protocol insolvency
- **Lesson from Terra**: 19.5% yield on Anchor was unsustainable emission-based
- **AAC Rule**: Only pursue yield sources backed by REAL economic activity

### Yield Classification
| Source | Sustainability | Risk | AAC Tier |
|---|---|---|---|
| US Treasury backed (via stablecoin reserves) | High | Low | Tier 1 |
| Blue-chip DeFi lending (Aave, Compound) | Medium-High | Medium | Tier 1 |
| DEX LP fees (major pairs) | Medium | Medium | Tier 2 |
| Protocol emission rewards | Low | High | Tier 3 (avoid) |
| Leveraged yield farming | Very Low | Very High | PROHIBITED |

---

## 5. TOKENIZED REAL-WORLD ASSETS (RWAs)

### The Next Frontier
- US Treasuries on-chain: Ondo Finance, Backed, Franklin Templeton
- Real estate tokenization: Propy, RealT
- Commodities: PAX Gold, Tether Gold
- Carbon credits: previously NAB stablecoin (abandoned Jan 2024)

### Why RWAs Matter for FFD
- Bridges traditional finance directly to DeFi rails
- Enables 24/7 trading of traditionally 9-5 assets
- Creates new arbitrage between on-chain and off-chain prices
- Settlement speed advantage: on-chain RWA settles in seconds vs T+1/T+2

### AAC Monitoring Points
- RWA TVL growth across protocols
- Regulatory approvals for tokenized securities
- Institution-specific launches (BlackRock, Franklin Templeton, Goldman Sachs)
- Price premium/discount of tokenized vs. traditional assets

---

## 6. TRADING STRATEGIES

### Strategy 1: Stablecoin Peg Deviation Capture
- Monitor USDT/USDC/RLUSD peg to underlying
- Buy depegged stablecoins when deviation is panic-driven (not structural)
- Exit when peg restores
- **Risk**: Structural depeg (Terra) — requires causality analysis before entry

### Strategy 2: Stablecoin Basis Trade
- Exploit interest rate differential between fiat deposits and stablecoin yield
- Long stablecoin yield, short traditional money market when spread widens
- **Entry**: Spread > 200bps above risk-free rate
- **Exit**: Spread normalizes

### Strategy 3: Cross-Venue Stablecoin Arbitrage
- Same stablecoin priced differently across exchanges/DEXs
- High-frequency capture of spread differentials
- **Infrastructure**: Multi-venue order routing required (existing AAC capability)

### Strategy 4: Regulatory Catalyst Trading
- Position ahead of legislative events (GENIUS Act, MiCAR enforcement, etc.)
- Track legislative committee hearings via NLP analysis
- Historical pattern: stablecoin-related tokens rally on pro-regulatory news

### Strategy 5: Stablecoin Flow Signals
- Large stablecoin mints/burns signal incoming market moves
- Stablecoin flows to exchanges precede buying pressure
- Stablecoin outflows from exchanges signal de-risking
- **Integration**: Enhance existing CryptoIntelligence on-chain analysis

---

## RESEARCH LOG

| Date | Finding | Evidence | Action |
|---|---|---|---|
| 2026-03-15 | FFD Track 2 genesis | E0 | Doctrine created |

---

*Private digital money is the transition's highway — it connects the old world
to the new. Whoever controls the stablecoin rails controls the toll booth
of the emerging financial system.* — FFD Doctrine

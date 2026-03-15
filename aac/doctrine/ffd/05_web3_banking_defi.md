# FFD Track 2 Expansion: Web3 Banking & Decentralized Finance
## DeFi Protocols, DAOs, Web3 Infrastructure, Decentralized Economies

**Last Updated:** 2026-03-15
**Evidence Level:** E0-E1 (Conceptual + Observed Patterns)

---

## THESIS

Web3 and DeFi represent a fundamental restructuring of financial
intermediation. Smart contracts replace broker-dealers, AMMs replace
order books, DAOs replace corporate boards, and self-sovereign identity
replaces KYC databases. AAC must understand both the revolutionary
potential AND the centralization paradox at the heart of "decentralized"
finance.

---

## 1. WEB3 — The Decentralized Internet

### Definition & Origins
- **Coined by**: Gavin Wood (Polkadot/Ethereum co-founder), 2014
- **Core idea**: "Decentralized online ecosystem based on blockchain"
- Cambridge Bennett Institute definition: "putative next generation of web's
  technical, legal, and payments infrastructure"

### Key Pillars
| Pillar | Description | Status |
|--------|-------------|--------|
| Decentralization | Data and services distributed, not centralized | Partially achieved |
| Blockchain | Immutable ledger for state, identity, ownership | Production |
| Tokenomics | Economic incentives via tokens and governance | Active |
| Self-sovereign identity | Users own their identity and data | Early stage |
| Smart contracts | Programmable, autonomous agreements | Production |
| DAOs | Decentralized governance organizations | Growing |
| DeFi | Financial services without intermediaries | $55B+ TVL |
| NFTs | Non-fungible tokens for ownership proof | Boom-bust cycle |

### The Centralization Paradox
Critics correctly identify that "decentralized" Web3 often relies on
centralized infrastructure:
- **API providers**: Alchemy and Infura handle majority of Ethereum API calls
- **Exchanges**: Binance, Coinbase, MetaMask dominate user access
- **Stablecoins**: Tether controls >50% of stablecoin market
- **Wallet concentration**: Most tokens held by few addresses

**Moxie Marlinspike** (Signal founder) demonstrated in 2022 that most
"decentralized" apps rely on 2-3 centralized API providers.

### Investment Scale
- $27B+ invested in Web3 by end of 2021
- Andreessen Horowitz (a16z) crypto fund: $7.6B across four funds
- Investment cooled in 2022-2023, resurging 2024-2025

### Critic Voices
- **Elon Musk**: "marketing buzzword" (while building X payments)
- **Jack Dorsey**: "VC plaything" — not truly decentralized
- **Tim O'Reilly**: Web3 is "too focused on financial engineering"

### AAC Monitoring Points
- Web3 VC funding rounds and total deployed capital
- Decentralization metrics: Nakamoto coefficient for major chains
- API provider concentration (Alchemy, Infura market share)
- Developer activity: GitHub commits to major Web3 projects
- Self-sovereign identity adoption: ENS registrations, DID standards

---

## 2. DECENTRALIZED FINANCE (DeFi) — DEEP DIVE

### Market Overview
- **Total Value Locked (TVL)**: Peaked $178B (November 2021), declined to <$40B
  (2023), recovering to ~$55B+ (2025)
- **Weekly DEX volume**: ~$18.6B average (mid-2025)
- **Unique wallets**: 9.7M+ interacting with DeFi protocols (mid-2025)
- **Primary chain**: Ethereum (~60% of DeFi TVL)
- **Multi-chain expansion**: Solana, Arbitrum, Optimism, Base, BNB Chain

### Architecture: The DeFi Stack

```
┌─────────────────────────────────────────┐
│  Layer 5: AGGREGATOR                    │
│  DEX aggregators (1inch, Paraswap)      │
├─────────────────────────────────────────┤
│  Layer 4: APPLICATION                   │
│  Uniswap, Aave, MakerDAO, Compound     │
├─────────────────────────────────────────┤
│  Layer 3: PROTOCOL                      │
│  AMMs, lending pools, flash loans       │
├─────────────────────────────────────────┤
│  Layer 2: ASSET                         │
│  ERC-20, ERC-721, stablecoins           │
├─────────────────────────────────────────┤
│  Layer 1: SETTLEMENT                    │
│  Ethereum, Solana, Arbitrum, etc.       │
└─────────────────────────────────────────┘
```

### Major DeFi Protocols

#### Aave — The Lending Giant
- **TVL**: $25.4B (~45% of all DeFi TVL as of May 2025)
- **Function**: Lending/borrowing with variable and stable rates
- **Innovation**: Flash loans — uncollateralized loans repaid in a single transaction
  (invented by Marble Protocol 2018, popularized by Aave)
- **Governance**: AAVE token holders vote on protocol parameters
- **Multi-chain**: Ethereum, Polygon, Arbitrum, Optimism, Avalanche

#### Uniswap — The DEX King
- **Function**: Automated Market Maker (AMM) for token swaps
- **Growth**: $20M → $2.9B liquidity in 2020
- **Innovation**: Constant product formula (x * y = k)
- **Governance**: UNI token, but concentration concerns
- **Volume**: Routinely rivals centralized exchanges

#### MakerDAO/Sky — The Stablecoin Protocol
- **Rebranded**: September 2024 (MakerDAO → Sky, DAI → USDS)
- **Circulating supply**: ~$9B DAI + USDS combined (March 2025)
- **Function**: Over-collateralized stablecoin minting
- **Governance**: MKR/SKY token voting

#### Compound Finance
- **Function**: Algorithmic lending/borrowing
- **Innovation**: COMP token rewards (June 2020) — started "yield farming" craze
- **Impact**: Kicked off DeFi Summer 2020

#### Lido & EigenLayer — The Staking Layer
- **Lido**: Liquid staking for Ethereum — largest staking protocol
- **EigenLayer**: Restaking — use staked ETH to secure additional protocols
- **Risk**: Concentration of stake in few protocols = systemic risk

### DeFi Risks & Attack Vectors

| Risk | Description | Historical Impact |
|------|-------------|-------------------|
| Smart contract bugs | Code errors drain funds | The DAO hack ($50M, 2016) |
| Flash loan attacks | Manipulate prices within single tx | Multiple exploits 2020-2023 |
| Rug pulls | Developers drain liquidity pools | $129M in 2021 alone |
| Impermanent loss | LPs lose vs. holding base assets | Persistent for volatile pairs |
| Oracle manipulation | Feed incorrect price data | Multiple exploits |
| Front-running/MEV | Miners/validators extract value | Systemic on Ethereum |
| Governance attacks | Buy votes via token acquisition | Multiple incidents |

- **2021**: Half of all crypto crime was DeFi-related
- **Bancor hack** (July 2018): $13.5M stolen from "decentralized" exchange
- **The Economist** (2022): DeFi is part of a "three-way fight" for digital finance

### Flash Loans — AAC Opportunity
- Borrowed and repaid in a single atomic transaction
- Zero collateral required — only gas fees
- Use cases: arbitrage, liquidations, collateral swaps
- Risk: exploits use flash loans to manipulate prices
- **AAC potential**: Flash loan arbitrage across DEXes (E2+ evidence needed)

---

## 3. DECENTRALIZED AUTONOMOUS ORGANIZATIONS (DAOs)

### What They Are
- Organizations governed by smart contracts and token-holder voting
- No CEO, no board of directors — code is law (in theory)
- Treasury managed by multisig or governance votes

### DAO Landscape
| DAO | Purpose | Treasury | Notes |
|-----|---------|----------|-------|
| MakerDAO/Sky | Stablecoin governance | $9B+ | Largest by assets |
| Uniswap | DEX governance | $5B+ | UNI token holders |
| Aave | Lending governance | $25B+ TVL | AAVE token |
| Compound | Lending governance | Multi-billion | COMP token |
| Lido | Staking governance | $15B+ staked | LDO token |
| ConstitutionDAO | Failed bid for US Constitution | Dissolved | Pop culture moment |

### Governance Problems
- Token concentration: Majority of governance tokens held by VCs and insiders
- Voter apathy: Most tokens never used for voting
- Plutocratic: More tokens = more votes = wealth-driven governance
- Regulatory uncertainty: Who is liable when a DAO causes harm?

### AAC Monitoring Points
- DAO treasury sizes and growth rates
- Governance participation rates (votes cast / tokens outstanding)
- DAO-to-DAO partnerships and composability
- Regulatory rulings on DAO legal status

---

## 4. WEB3 BANKING — THE CONVERGENCE

### Traditional Finance Meets DeFi
Banks and TradFi institutions are integrating DeFi:
- **JPMorgan Quorum**: Permissioned Ethereum for institutional use
- **Visa**: Settling stablecoin transactions on Ethereum (2021)
- **MasterCard + UBS + JPMorgan**: $65M into ConsenSys (2021)
- **Barclays, UBS, Credit Suisse**: Ethereum experiments
- **Enterprise Ethereum Alliance**: 150+ corporate members
- **Innovate UK**: Cross-border payment prototypes

### DeFi Banking Services Comparison

| Service | TradFi | DeFi | Advantage |
|---------|--------|------|-----------|
| Savings | 0.5-5% APY | 2-20% APY (variable) | Higher yields (higher risk) |
| Lending | Bank approval, days | Instant, collateralized | Speed, permissionless |
| Trading | Exchange hours, T+2 | 24/7, instant settlement | Always-on, global |
| Insurance | Paper claims, weeks | Smart contract, automatic | Speed, transparency |
| Identity | KYC per institution | Self-sovereign (ENS, DID) | User owns data |
| Payments | Wire: 1-5 days, $25-50 | Stablecoin: seconds, < $1 | Speed, cost |

### The Unbanked Opportunity
- **1.4 billion** adults globally are unbanked (World Bank)
- DeFi requires only a smartphone and internet connection
- Stablecoins provide dollar-denominated savings in high-inflation countries
- Remittances: DeFi can reduce 6-10% fees to <1%
- **XRPL / Ripple**: Specifically targeting cross-border remittances

### WLFI & USD1 as Web3 Banking
- WLFI seeking US banking license — DeFi protocol becoming a bank
- USD1 stablecoin: $2B circulation, backed by treasuries
- Pakistan partnership: Cross-border payments via blockchain
- Regulatory advantage from political connections
- Could set precedent for DeFi-to-bank transition pathway

### AAC Monitoring Points
- Bank-DeFi partnership announcements
- Institutional DeFi TVL (separate from retail)
- Stablecoin usage for remittances (volume through corridors)
- Regulatory clarity on DeFi banking (licensing, compliance frameworks)
- Self-sovereign identity adoption rate

### Trading Strategies
- **DeFi yield harvesting**: Monitor and capture highest-yield lending pools
  (evidence level E2+ required before live deployment)
- **Flash loan arbitrage**: Cross-DEX price discrepancies
  (high technical complexity, needs dedicated infrastructure)
- **Institutional flow tracking**: Monitor on-chain flows from known institutional addresses
- **DeFi/TradFi convergence plays**: Long DeFi governance tokens ahead of TradFi announcements
- **Stablecoin corridor arbitrage**: Price differences across DeFi and CEX venues

---

## 5. DECENTRALIZED ECONOMY — MACRO THESIS

### Beyond DeFi: Decentralized Everything
The "Broken Money" thesis extends beyond finance:
- **Decentralized compute**: Render Network, Akash, Golem — GPU sharing
- **Decentralized storage**: Filecoin, Arweave, IPFS — censorship-resistant data
- **Decentralized social**: Farcaster, Lens Protocol, Nostr — user-owned social graphs
- **Decentralized identity**: ENS, Worldcoin, Polygon ID
- **Decentralized physical infrastructure (DePIN)**: Helium (wireless), Hivemapper (maps)

### Real World Assets (RWA) Tokenization
- Bringing traditional assets on-chain: real estate, bonds, commodities, art
- BlackRock's BUIDL fund: Tokenized money market fund on Ethereum
- US Treasury tokenization growing rapidly
- Market size potential: $16 trillion by 2030 (BCG estimate)
- Already tracked in FFD Track 2 (02_track_private_digital.md)

### Regulatory Landscape for DeFi
| Jurisdiction | Approach | Key Legislation |
|--------------|----------|-----------------|
| US | Evolving — GENIUS Act (stablecoins) + SAB121 repeal | Pro-crypto shift under Trump |
| EU | MiCAR — comprehensive framework | Enforced since June 2024 |
| UK | FCA sandbox approach | Crypto registration required |
| Singapore | MAS regulatory sandbox | Progressive but strict |
| Japan | Yen stablecoin framework | Institutional focus |
| China | Full ban (mainland) | CBDCs preferred |
| Hong Kong | Stablecoin Bill passed | Separate from mainland |
| FATF | Global guidance | DeFi in scope since October 2021 |

### AAC Monitoring Points
- RWA tokenization total value across protocols
- DePIN adoption metrics (Helium hotspots, Render GPU utilization)
- Cross-border stablecoin remittance volumes
- Regulatory actions affecting DeFi accessibility
- Web3 developer migration between chains

---

## AAC INTEGRATION

### New Metrics for FFD Engine
1. **defi_tvl_total** — Total DeFi TVL across all chains
2. **defi_tvl_ethereum_share** — Ethereum's share of total DeFi TVL
3. **dex_weekly_volume** — Aggregate DEX trading volume
4. **flash_loan_arb_opportunity** — Detected cross-DEX arbitrage frequency
5. **dao_governance_health** — Voter participation composite score
6. **web3_vc_funding_monthly** — Monthly VC investment into Web3 projects
7. **rwa_tokenized_value** — Total RWA value on-chain

### Risk Flags
- ⚠️ **Smart contract risk**: Code is law — bugs = permanent loss
- ⚠️ **Centralization paradox**: "Decentralized" apps on centralized infra
- ⚠️ **Regulatory uncertainty**: FATF guidance could restrict DeFi access
- ⚠️ **Yield sustainability**: DeFi yields often unsustainable long-term
- ⚠️ **Flash loan exploit risk**: Must monitor for protocol-level attacks
- ⚠️ **DAO governance capture**: Whale token holders can control protocol direction

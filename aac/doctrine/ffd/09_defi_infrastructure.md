# FFD-09: DeFi Infrastructure & Emerging Protocol Intelligence

> **Status**: Active Research Document
> **Created**: 2026-03-13
> **Scope**: DeFi protocols, L2 infrastructure, ZK technology, social protocols, RWA tokenization
> **Evidence Level**: E3-E4 (Published data + verified metrics)

---

## 1. Zero-Knowledge Proof Technology

### 1.1 Foundations
- **Conceived**: 1985 by Shafi Goldwasser, Silvio Micali, and Charles Rackoff (MIT)
- **Three Properties**: Completeness (true statements accepted), Soundness (false statements rejected), Zero-Knowledge (nothing else learned)
- **Significance**: Enables verification of computation WITHOUT revealing the underlying data

### 1.2 Protocol Families

| Protocol | Type | Trusted Setup | Proof Size | Verification Time | Key Use |
|---|---|---|---|---|---|
| **zk-SNARK** | Succinct Non-interactive ARgument of Knowledge | Yes (per-circuit) | ~200 bytes | ~10ms | Zcash, zkSync |
| **zk-STARK** | Scalable Transparent ARgument of Knowledge | No | ~50-200 KB | ~100ms | StarkNet, StarkEx |
| **PLONK** | Permutations over Lagrange-bases | Universal (one-time) | ~400 bytes | ~10ms | Polygon zkEVM, Scroll |
| **Bulletproofs** | Short proofs, no setup | No | ~700 bytes | ~1-10s | Monero, Mimblewimble |
| **Groth16** | Pairing-based SNARK | Yes (per-circuit) | ~130 bytes | ~5ms | Zcash Sapling |

### 1.3 zkVMs — The Next Frontier
- **RISC Zero**: zkVM based on RISC-V architecture — prove arbitrary computation in ZK
- **Succinct Labs (SP1)**: RISC-V based zkVM, partnered with multiple L2s
- **Significance**: General-purpose ZK proofs instead of application-specific circuits
- **Applications**: Verifiable off-chain computation, privacy-preserving DeFi, cross-chain verification

### 1.4 Security Considerations
- **96% of SNARK bugs** stem from under-constrained circuits (per academic research)
- **Trusted setup vulnerability**: If setup ceremony compromised, fake proofs possible (zk-SNARKs)
- **Quantum vulnerability**: SNARK/PLONK relying on elliptic curves will need post-quantum migration
- **STARKs advantage**: Hash-based, potentially quantum-resistant

**AAC Integration Signal**: HIGH. ZK technology is infrastructure-layer for privacy, L2 scaling, and cross-chain verification. AAC should monitor zkVM maturity for privacy-preserving trade execution.

---

## 2. Layer 2 Landscape

### 2.1 Arbitrum
- **Technology**: Optimistic rollup with fraud proofs
- **Developer**: Offchain Labs
- **Token**: ARB (governance)
- **Nitro Upgrade**: Replaced custom VM with WASM-based execution, 10x compression improvement
- **Position**: Largest L2 by TVL ($3B+) and developer count
- **Arbitrum Orbit**: Framework for creating L3 chains on top of Arbitrum (like Coinbase's L2 but one layer up)
- **Stylus**: Write smart contracts in Rust, C, C++ (in addition to Solidity)

### 2.2 Base (Coinbase L2)
- **Launched**: February 2023
- **Technology**: Optimism OP Stack (optimistic rollup)
- **No Token**: Coinbase stated "no plans" for a BASE token
- **Low Fees**: Sub-cent transactions after Dencun blobs
- **Strategy**: Onboard next 1B users to on-chain via Coinbase's 100M+ verified users
- **Smart Wallet**: Passkey-based wallet creation — no seed phrases
- **Stablecoin Payments**: Merchant payment service launched June 2025

### 2.3 OP Stack Ecosystem (Optimism Superchain)
- **Architecture**: Shared sequencer, shared bridge, shared governance
- **Members**: Optimism, Base, Zora, Mode, World Chain (Worldcoin), Uniswap Chain
- **Revenue Model**: Chains contribute sequencer revenue to the Optimism Collective
- **Significance**: Emerging as the "AWS of L2s" — standardized infrastructure for custom chains

### 2.4 L2 Comparison Matrix

| L2 | Type | TPS | Avg Fee | TVL | Key Differentiator |
|---|---|---|---|---|---|
| Arbitrum One | Optimistic | ~4,000 | $0.01-0.10 | $3B+ | Largest ecosystem, Stylus |
| Base | Optimistic | ~2,000 | $0.001-0.01 | $2B+ | Coinbase distribution |
| Optimism | Optimistic | ~2,000 | $0.01-0.05 | $1B+ | Superchain standard |
| zkSync Era | ZK Rollup | ~2,000 | $0.01-0.10 | $500M+ | Native account abstraction |
| StarkNet | ZK Rollup | ~1,000 | $0.01-0.05 | $300M+ | Cairo language, STARK proofs |
| Polygon zkEVM | ZK Rollup | ~2,000 | $0.01-0.05 | $200M+ | EVM equivalence via PLONK |

**AAC Integration Signal**: CRITICAL. L2 selection directly impacts execution cost and speed. Base = Coinbase integration, Arbitrum = deepest DeFi liquidity.

---

## 3. Liquid Staking & Restaking

### 3.1 Lido DAO
- **Protocol**: Liquid staking for Ethereum (and others)
- **Token**: stETH (staked ETH receipt token)
- **TVL**: $15B+ — single largest DeFi protocol by value locked
- **ETH Share**: ~30% of all staked ETH runs through Lido
- **Governance**: LDO token holders control protocol parameters
- **Concern**: Concentration risk — one protocol controlling 30% of consensus undermines decentralization

### 3.2 EigenLayer — Restaking Revolution
- **Concept**: "Restaking" — reuse staked ETH security for other protocols
- **TVL**: $15B+ (at peak, one of the largest DeFi protocols)
- **Founder**: Sreeram Kannan (University of Washington)
- **Launched**: April 2024 (mainnet)

**Actively Validated Services (AVSs)**:
- New protocols can "rent" Ethereum's security instead of bootstrapping their own validator set
- Examples: Oracle networks, data availability layers, bridges, sequencers
- Slashing: Restaked ETH can be slashed if the AVS misbehaves

**Risk Factors**:
- Cascading slashing: One AVS failure could trigger liquidations across multiple protocols
- Rehypothecation risk: Same ETH securing multiple services simultaneously
- Systemic exposure: If EigenLayer fails, it could impact Ethereum consensus itself

**AAC Integration Signal**: HIGH but HIGH RISK. Restaking yields are attractive but carry novel systemic risks. AAC should monitor EigenLayer health metrics but limit exposure.

---

## 4. DEX & AMM Protocols

### 4.1 Curve Finance
- **Specialization**: Stablecoin and pegged-asset swaps
- **Innovation**: StableSwap invariant — low slippage for correlated assets
- **TVL**: ~$2B
- **veTokenomics**: Vote-escrowed CRV (veCRV) — lock CRV for up to 4 years for voting power and yield boost
- **"Curve Wars"**: Protocols compete to accumulate veCRV to direct CRV emissions to their pools (Convex Finance dominant)
- **crvUSD**: Native stablecoin using LLAMMA (Lending-Liquidating AMM Algorithm) — soft liquidation instead of cliff liquidation

### 4.2 Jupiter (Solana)
- **Position**: Leading DEX aggregator on Solana
- **Token**: JUP (governance, launched January 2024)
- **Features**: Limit orders, DCA, perpetual futures
- **Significance**: Solana's equivalent of 1inch/Paraswap — routes trades across all Solana DEXs for best execution
- **JUP Airdrop**: One of the largest airdrops in crypto history ($700M+ distributed)

### 4.3 Hyperliquid
- **Type**: On-chain perpetual futures DEX
- **Technology**: HyperBFT consensus on own L1 blockchain (not built on Ethereum)
- **Innovation**: Order book model (not AMM) — closest to CEX experience on-chain
- **Performance**: Sub-second finality, 200,000 orders/second
- **Airdrop**: December 2024, one of the most valuable airdrops ever (~$1.5B total value)
- **HYPE Token**: No VC investors — 100% community/team distribution
- **TVL**: Rapidly growing, approaching top 5 DeFi protocols

**AAC Integration Signal**: HIGH. Jupiter and Hyperliquid represent next-gen DEX infrastructure. Hyperliquid's order book model is particularly relevant for algorithmic trading strategies.

---

## 5. Yield & RWA Protocols

### 5.1 Pendle Finance
- **Innovation**: Yield tokenization — split yield-bearing assets into Principal Tokens (PT) and Yield Tokens (YT)
- **TVL**: ~$5B (2025)
- **Mechanism**:
  - **PT (Principal Token)**: Claims the underlying asset at maturity (like a zero-coupon bond)
  - **YT (Yield Token)**: Claims all yield generated until maturity
  - **AMM**: Custom AMM optimized for time-decaying assets (YT approaches zero at maturity)
- **Use Cases**: Fixed-rate yield, yield speculation, yield hedging
- **Significance**: Brings interest rate swap mechanics to DeFi

### 5.2 Ethena (USDe)
- **Type**: Synthetic dollar — NOT a stablecoin (no fiat reserves)
- **Mechanism**: Delta-neutral strategy — hold spot ETH/BTC + short equivalent perpetual futures
- **TVL**: ~$3B
- **sUSDe Yield**: Generated from funding rate payments (perps shorts typically receive funding in bull markets)
- **Backed by**: Arthur Hayes (BitMEX co-founder)
- **Risk**: Negative funding rates could cause losses; custodian risk (centralized exchange exposure)
- **Significance**: First crypto-native "dollar" that generates yield from on-chain derivatives

### 5.3 Ondo Finance — Institutional RWA
- **Focus**: Tokenized US Treasuries and institutional-grade RWA
- **OUSG**: Tokenized short-term US Treasury fund
- **TVL**: ~$600M+
- **Partners**: BlackRock (BUIDL integration), Morgan Stanley
- **Significance**: Bridge between TradFi fixed income and DeFi composability
- **Regulatory**: Offers to qualified purchasers only (Reg D exemption)

### 5.4 Securitize
- **Role**: RWA tokenization platform and SEC-registered transfer agent
- **Notable**: BlackRock BUIDL fund is tokenized via Securitize
- **Partnerships**: BlackRock, Hamilton Lane, KKR
- **Significance**: The "plumbing" for institutional tokenization — handles compliance, cap tables, distributions

### 5.5 DTCC Blockchain Settlement
- **DTCC Overview**: Depository Trust & Clearing Corporation — settles $2.5 quadrillion annually
- **DTC**: $87.1T in securities custody, 3.5 million issues
- **NSCC**: Central counterparty, nets 98% of daily value, 4,000+ participants
- **T+1**: US moved to T+1 settlement May 2024 (was T+2)

**Landmark Event — December 2025**:
- **SEC No-Action Letter**: Authorized DTCC to process tokenized equities on blockchain for 3-year trial period
- **Significance**: The world's largest securities settlement organization officially integrating blockchain
- **Impact**: If successful, tokenized stocks could trade/settle 24/7 (vs current 9:30-4:00 ET)

**AAC Integration Signal**: CRITICAL. DTCC blockchain settlement + Securitize + BlackRock BUIDL = institutional tokenization infrastructure going live. AAC must prepare for tokenized equity trading.

---

## 6. DePIN — Decentralized Physical Infrastructure Networks

**Sector Overview**:
- **Definition**: Protocols that use token incentives to build and operate physical infrastructure
- **Market**: 321+ projects, ~$18.3B market cap (2025)
- **IEEE Recognition**: Published in IEEE Network journal (2024), granting academic legitimacy
- **Two Categories**:
  - **Physical Resource Networks (PRN)**: Geographically-dependent (wireless, energy, mobility)
  - **Digital Resource Networks (DRN)**: Location-independent (storage, compute, bandwidth)

**Major DePIN Protocols**:

| Protocol | Category | Function | Token | Market Cap |
|---|---|---|---|---|
| Filecoin | DRN | Decentralized storage | FIL | ~$3B |
| Helium | PRN | LoRaWAN + 5G wireless | HNT/MOBILE | ~$1B |
| Render | DRN | GPU rendering | RENDER | ~$4B |
| Hivemapper | PRN | Street-level mapping | HONEY | ~$100M |
| Akash | DRN | Cloud compute | AKT | ~$1B |
| Livepeer | DRN | Video transcoding | LPT | ~$500M |

**AAC Integration Signal**: LOW-MEDIUM. DePIN is infrastructure, not directly tradable alpha. Monitor Filecoin and Render as "picks and shovels" plays on AI/data demand.

---

## 7. Decentralized Social Protocols

### 7.1 Nostr
- **Full Name**: Notes and Other Stuff Transmitted by Relays
- **Created**: March 2020 by pseudonymous developer "fiatjaf"
- **Architecture**: Cryptographic public key identities + WebSocket relay servers
- **Protocol**: JSON-based events, NIPs (Nostr Implementation Possibilities)
- **Lightning Integration**: "Zaps" — send Bitcoin micropayments via Lightning Network
- **Notable Clients**: Damus (iOS, launched February 2023), Amethyst (Android), Primal
- **Backing**: Jack Dorsey donated ~$250K BTC (2023) + $10M cash (2025)
- **Users**: 18M+ registered keys (May 2023), active daily users much lower
- **Censorship Resistance**: No central server can ban users — relay operators choose what to host

### 7.2 Farcaster
- **Type**: Decentralized social protocol on Ethereum
- **Founder**: Dan Romero (former Coinbase VP)
- **Funding**: $150M+ raised (a]6]z venture, Paradigm)
- **Technology**: OP Stack (Optimism), "hubs" store social data
- **Client**: Warpcast (primary, Farcaster team-built)
- **Frames**: Interactive mini-apps embedded in social posts (buy NFTs, vote, play games in-feed)
- **User Base**: 500K+ registered users (2025)
- **Significance**: "Decentralized Twitter" with DeFi composability via Frames

### 7.3 Comparison

| Feature | Nostr | Farcaster |
|---|---|---|
| Architecture | Pure protocol (relays) | Protocol + hubs (Ethereum) |
| Identity | Cryptographic keys | Ethereum addresses + FID |
| Payments | Lightning (BTC) | ETH/USDC (Base) |
| Composability | Limited | High (Frames, DeFi) |
| Censorship Resistance | Very High | Medium-High |
| Backing | Jack Dorsey, grassroots | a16z, Paradigm |

**AAC Integration Signal**: LOW. Social protocols are sentiment data sources, not execution venues. Potential for market sentiment extraction from Nostr/Farcaster feeds.

---

## 8. Account Abstraction (ERC-4337)

### Core Concept
- **Problem**: Ethereum accounts (EOAs) require private key management, gas in ETH, no recovery
- **Solution**: Smart contract wallets with programmable transaction logic

### Architecture
- **EntryPoint Contract**: Singleton contract that validates and executes UserOperations
- **Bundlers**: Off-chain actors that aggregate UserOperations and submit them as transactions
- **Paymasters**: Sponsor gas fees on behalf of users (gasless transactions)
- **Smart Account**: Wallet is a contract — supports arbitrary verification logic

### Capabilities
- **Social Recovery**: Recover wallet via guardians (friends, family, institutions)
- **Session Keys**: Grant limited permissions to dApps without signing each transaction
- **Batched Transactions**: Multiple operations in a single transaction (approve + swap)
- **Gas Abstraction**: Pay gas in any token (ERC-20), or let dApps sponsor gas
- **Multi-Signature**: Built-in multisig without external contracts

### Current State
- **EIP-7702** (Pectra upgrade, May 2025): Lightweight account abstraction for existing EOAs
- **Safe (formerly Gnosis Safe)**: Largest smart account platform, $100B+ secured
- **Adoption**: Growing but still <5% of Ethereum wallets are smart accounts

**AAC Integration Signal**: MEDIUM. Account abstraction improves operational security (multisig, session keys) for AAC's wallet infrastructure. Monitor Safe and EIP-7702 adoption.

---

## 9. MEV — Maximal Extractable Value

### Definition
- **MEV**: Profit that block producers (or searchers) can extract by reordering, inserting, or censoring transactions within a block
- **Originally**: "Miner Extractable Value" (pre-merge); now "Maximal" (post-merge validators)

### MEV Types
| Type | Mechanism | Impact on Users |
|---|---|---|
| **Front-running** | Insert transaction before target | Worse execution price |
| **Sandwich Attack** | Buy before + sell after target swap | Slippage extraction |
| **Arbitrage** | Cross-DEX price equalization | Generally positive (price efficiency) |
| **Liquidation** | Race to liquidate under-collateralized positions | Neutral (necessary) |
| **JIT Liquidity** | Add/remove LP position around large swap | Neutral to positive |

### MEV Infrastructure
- **Flashbots**: Pioneered private transaction pools and MEV auctions
- **MEV-Boost**: Separates block building from block proposing (PBS — Proposer-Builder Separation)
- **Block Builders**: Specialized entities construct optimal blocks (Beaverbuild, Flashbots, Titan)
- **MEV Supply Chain**: Searchers → Builders → Proposers (validators)
- **Private Mempools**: ~80% of Ethereum blocks now go through MEV-Boost

### Scale
- **Extracted MEV**: $600M+ since merge (September 2022 – 2025)
- **Concern**: MEV extraction is effectively a "tax" on DeFi users
- **Solutions**: MEV-aware DEXs (CoW Swap), order flow auctions, encrypted mempools (Shutter)

**AAC Integration Signal**: HIGH. MEV protection is essential for any DeFi execution strategy. AAC must use private transaction submission (Flashbots Protect) and MEV-aware DEX routing.

---

## 10. Stablecoin & DeFi Yield Sources

### Yield Taxonomy

| Source | Mechanism | Typical APY | Risk Level |
|---|---|---|---|
| **Lending** | Supply assets to lending pools (Aave, Compound) | 2-8% | Low-Medium |
| **LP Fees** | Provide liquidity to AMMs (Uniswap, Curve) | 5-20% | Medium (IL risk) |
| **Staking** | Lock native tokens for consensus (ETH 3.5%) | 3-8% | Low |
| **Liquid Staking** | stETH/rETH (Lido/Rocket Pool) | 3-5% | Low-Medium |
| **Restaking** | Reuse staked ETH via EigenLayer | 5-15% | Medium-High |
| **Funding Rate Arb** | Delta-neutral (Ethena USDe model) | 10-30% | High |
| **Yield Tokenization** | PT/YT via Pendle | Variable | Medium |
| **RWA** | Tokenized Treasuries (BUIDL, OUSG) | 4-5% | Low |

### Risk Framework for AAC
1. **Sustainable**: RWA yield (Treasury-backed), ETH staking, blue-chip lending (Aave)
2. **Moderate**: Curve LP, liquid staking, Pendle PT
3. **Aggressive**: Funding rate arb, EigenLayer restaking, leveraged farming
4. **Avoid**: Unsustainable token emission yields ("real yield" test — is yield from fees or from printing tokens?)

---

## 11. First Digital USD (FDUSD) — Cautionary Tale

### Background
- **Issuer**: First Digital Trust (Hong Kong)
- **Platform**: Ethereum and BNB Chain
- **Backing**: Claimed 1:1 USD reserves

### April 2025 Depeg Event
- **Trigger**: Justin Sun (Tron founder) publicly alleged First Digital Trust was insolvent
- **Impact**: FDUSD briefly depegged, causing panic selling
- **Outcome**: First Digital denied allegations, threatened legal action
- **Lesson**: Even "major" stablecoins can face existential threats from social media attacks

**AAC Integration Signal**: CRITICAL LESSON. Any stablecoin without full regulatory backing (NYSE-listed issuer, bank charter, Fed access) carries existential depeg risk. Reinforces USDC-first policy.

---

## 12. Tokenized Treasuries Market

### Market Overview
- **Total Market**: $2B+ in tokenized government securities (2025)
- **Growth**: From <$100M (2023) to $2B+ — 20x growth in 2 years

### Leading Products

| Product | Issuer | Platform | AUM | Yield |
|---|---|---|---|---|
| **BUIDL** | BlackRock / Securitize | Ethereum | $500M+ | ~5% |
| **BENJI** | Franklin Templeton | Stellar, Polygon | $400M+ | ~5% |
| **OUSG** | Ondo Finance | Ethereum | $300M+ | ~5% |
| **USDY** | Ondo Finance | Multiple | $200M+ | ~5% |

### Why This Matters
- **DeFi Composability**: Tokenized Treasuries as collateral in DeFi protocols
- **24/7 Settlement**: Unlike traditional bond markets (business hours only)
- **Global Access**: Non-US investors can access US Treasuries without US brokerage accounts (where permitted)
- **DTCC Integration**: SEC no-action letter December 2025 paves way for tokenized equities on same infrastructure

**AAC Integration Signal**: CRITICAL. Tokenized Treasuries are the "risk-free rate" of DeFi. AAC should use BUIDL/OUSG as USDC yield alternative for cash reserves.

---

*FFD-09 maps the DeFi and infrastructure landscape with focus on protocols, risks, and AAC integration points. Data based on internet research + existing knowledge. Market data subject to rapid change.*

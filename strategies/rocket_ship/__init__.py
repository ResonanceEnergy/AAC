"""
Rocket Ship Module v1.0
========================
Post-Petrodollar Financial System Transition Engine

The companion module to Storm Lifeboat. While the Lifeboat protects capital
during the petrodollar unwind (2026 volatility), the Rocket Ship positions into
the emerging multipolar digital-hybrid financial system.

Architecture:
    Life Boat  →  Moons 1-12  (March 2026 – ~March 2027)
    IGNITION   →  Gulf Yuan Oil trigger OR Moon 13 new moon
    Rocket Ship→  Moons 13-39 (~April 2027 – June 2029)

Thesis foundation (Prof. Jiang Xueqin, March 21 2026):
    The 1974 petrodollar (oil priced in USD) is structurally unwinding:
    - 2022 sanctions weaponized USD → Russia/China yuan oil trade
    - 2024 Saudi joins mBridge; UAE executing government CBDC txns
    - BRICS Unit (Oct 2025 pilot): 40% gold + 60% BRICS basket
    - mBridge: $55.5B+ processed, 95% e-CNY, 4,000+ transactions
    - Central banks: 1,000+ tonnes gold/year since 2022
    - USD reserve share: ~56-58% (down from 72% in 2000)

New system rails (live or piloting):
    mBridge  →  Wholesale CBDC cross-border settlement (China/Saudi/UAE/HK/Thailand)
    BRICS Unit → 40% gold-backed digital settlement instrument
    XRP/ODL  →  Neutral instant FX bridge (ISO-20022 native, OCC charter)
    Flare    →  XRPFi — programmable DeFi layer on XRP (FXRP, Morpho, SparkDEX)
    Solana   →  High-TPS payments/Web3/RWA ($650B+/mo stablecoin volume)
    Ethereum →  Smart-contract RWA tokenization backbone
    Bitcoin  →  Digital gold (US Strategic Bitcoin Reserve, ETFs)

Base of Operations Roadmap:
    Panama (primary — #1 expat country, dollarized, Tocumen hub)
    Paraguay (secondary — 0% territorial, ultra-low cost, fast residency)
    UAE    (2027+ — mBridge hub, 0% tax, Golden Visa)
    Singapore / Malaysia / Switzerland (2028-2030 full diversification)

Ignition Trigger (Gulf Yuan Oil):
    Saudi Arabia or UAE confirms yuan-denominated oil contract settled via
    mBridge → timeline jumps forward immediately. Default: Moon 13 new moon.

Modules:
    core            — Enums, asset universe, system constants
    indicators      — 15 tracked indicators + Gulf Yuan ignition trigger
    lunar_cycles    — Moon cycles 13-39 timing engine (rebalance windows)
    trigger_engine  — Ignition logic, green-count, phase determination
    allocation      — Life Boat vs Rocket Ship allocation models
    geo_plan        — Panama/Paraguay base-of-operations task tracker
    runner          — CLI entry point
"""

__version__ = "1.0.0"
__phase__ = "LIFE_BOAT"   # Updated to ROCKET_SHIP on ignition

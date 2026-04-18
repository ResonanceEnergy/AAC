# Scenario Definitions

> Canonical source for MC scenario overrides. Do not duplicate.

## Core Scenarios (from storm_lifeboat)

43 scenarios are loaded dynamically from `strategies/storm_lifeboat/core.py`.
See the Python source for full ScenarioDefinition fields.

## Hand-Tuned MC Overrides

These override drift parameters for specific high-conviction scenarios:

### Hormuz Blockade
- Oil: $155, Gold: $5,800, VIX: 42, SPY: $580, BTC: $52K, HY: 700bp
- Drift overrides: oil +1.80, gold +1.20, spy -0.55, btc -0.80

### DeFi Cascade
- Oil: $92, Gold: $5,000, VIX: 35, SPY: $600, BTC: $38K, HY: 600bp
- Drift overrides: btc -1.20, eth -1.50, xrp -1.30

### Supercycle (Gold)
- Oil: $105, Gold: $7,000, VIX: 28, SPY: $620, BTC: $60K, HY: 500bp
- Drift overrides: gold +2.00, silver +1.80, gdx +2.50

## Extra Scenarios (not in storm_lifeboat)

| Scenario | Probability | Severity | Key Overrides |
|---|---|---|---|
| Iran De-Escalation | 25% | 0.40 | Oil $78, Gold $4.2K, SPY $700, BTC $95K |
| Private Credit Cascade | 12% | 0.75 | HY 800bp, XLF -0.65, XLRE -0.50, SPY -0.45 |
| Soft Landing (Base) | 40% | 0.20 | Oil $85, Gold $4.3K, SPY $680, BTC $85K |
| Black Swan (Multi-Front) | 1% | 0.99 | Oil $200, Gold $6.5K, VIX 60, SPY $420, BTC $28K |

## Running Scenarios

```
python strategies/war_room_engine.py --scenario hormuz
python strategies/war_room_engine.py --scenario defi_cascade
python strategies/war_room_engine.py --scenario supercycle
python strategies/war_room_engine.py --scenario soft_landing
python strategies/war_room_engine.py --scenario black_swan
```

# AAC Broker & API Governance Policy

**Version:** 1.0  
**Jurisdiction:** Alberta, Canada  
**Authority:** NCC Doctrine  
**Priority Stack:** Multi-Asset Arbitrage (2) → Canadian Compliance (3) → FX/CFD Execution (1)  
**Last Updated:** 2026-03-07  

---

## 1. Tiering Rules (Enforceable)

### Tier A — Core Capital (CIRO/CIPF Protected)

**Venues:** Interactive Brokers, Moomoo Canada

| Rule | Enforcement |
|------|-------------|
| Full capital deployment allowed | Managed by `strategy/risk_controls.py` |
| Daily reconciliation required | Automated via `CentralAccounting/` |
| All orders audited | `shared/audit_logger.py` |
| Paper mode default | `IBKR_PAPER=true`, `MOOMOO_PAPER=true` |
| Live trading gated | `LIVE_TRADING_ENABLED=true` required |

**Capital allocation:** Up to 80% of total AAC capital  
**Max drawdown trigger:** 10% account value → kill switch → manual review  

---

### Tier B — Satellite Execution (Capped Exposure)

**Venues:** NDAX, Metal X DEX, (Parked: Binance, Kraken, Coinbase)

| Rule | Enforcement |
|------|-------------|
| Exposure cap: 15% of total capital per venue | Risk controls |
| Daily balance reconciliation | Automated |
| Withdrawal monitoring | Weekly test withdrawal (small) |
| No single venue dependency for arb legs | Portfolio manager validation |

**Capital allocation:** Up to 30% of total AAC capital (combined)  
**Max per venue:** 15%  

---

### Tier C — Sandbox / High-Risk (Disposable Capital Only)

**Venues:** Noxi Rise, OANDA, IC Markets, Pepperstone

| Rule | Enforcement |
|------|-------------|
| R&D funds only — never treasury | Hard cap in config |
| Max $500 per venue | Enforced in risk_controls |
| No arbitrage legs depend on these | Strategy validation |
| Assume zero recourse | Documented in registry |
| Noxi Rise: assume funds unrecoverable | Governance flag |

**Capital allocation:** Max 5% of total AAC capital (combined across all Tier C)  
**Max per venue:** $500 or 2% of capital, whichever is less  

---

## 2. Capital Allocation Framework

```
Total AAC Capital = $X

Tier A (Core):     up to 80%
  ├── IBKR:        up to 60%  (primary)
  └── Moomoo:      up to 20%  (secondary)

Tier B (Satellite): up to 30%
  ├── NDAX:        up to 15%  (crypto/CAD rails)
  └── Metal X:     up to 10%  (DeFi)
  └── Parked:       0%        (Binance/Kraken/Coinbase)

Tier C (Sandbox):  max 5%
  ├── Noxi Rise:   max $500
  ├── OANDA:       max $500
  ├── IC Markets:  max $500
  └── Pepperstone: max $500

Note: Allocations may exceed 100% in sum because
Tier A and B serve different asset classes with
some overlap. Actual deployment must sum to 100%.
```

---

## 3. Operational Controls

### 3.1 Safety Switches (Global)

| Control | Default | Override |
|---------|---------|----------|
| `PAPER_TRADING` | `true` | Must be explicitly `false` |
| `LIVE_TRADING_ENABLED` | `false` | Must be explicitly `true` |
| `AAC_ENV` | `development` | `paper` or `production` |
| Kill switch (4 consecutive losses) | Active | 24h cooldown |
| Daily drawdown limit | 4% | Per-strategy config |
| Campaign drawdown limit | 20% | Risk controls |

### 3.2 Per-Venue Controls

| Venue | Max Leverage Used | Daily Loss Cap | Max Open Positions |
|-------|-------------------|----------------|-------------------|
| IBKR | Account default | 2% of venue capital | Per strategy |
| Moomoo | Account default | 2% of venue capital | Per strategy |
| NDAX | 1x (spot) | 2% of venue capital | 5 |
| Noxi Rise | 10x max (self-imposed) | $50 | 2 |
| OANDA | 1:50 max | $50 | 2 |
| IC Markets | 10x max | $50 | 2 |
| Pepperstone | 10x max | $50 | 2 |

---

## 4. Secrets & Key Management

### 4.1 Key Storage

- All API keys stored as environment variables (`.env`, never committed)
- Encrypted at rest via `shared/api_key_manager.py` (Fernet encryption)
- Master password: `ACC_MASTER_PASSWORD` (auto-generated)
- Public key for audit verification: `config/crypto/audit_public_key.pem`

### 4.2 Key Rotation Schedule

| Provider Type | Rotation Frequency | Alert Before Expiry |
|---------------|-------------------|---------------------|
| Tier A brokers | 90 days | 14 days |
| Tier B exchanges | 90 days | 14 days |
| Tier C sandboxes | 180 days | 30 days |
| Market data APIs | 365 days | 30 days |
| AI/LLM providers | 90 days | 14 days |

### 4.3 Key Hygiene Rules

1. **Never** log, print, or expose API keys in output
2. `.env` is permanently gitignored
3. Pre-commit hooks block secret commits (`detect-private-key`)
4. CI runs `pip-audit` + `trufflehog` on every push
5. All key access events are audit-logged

---

## 5. Monitoring & Ops Runbook

### 5.1 Daily Checks (Automated)

| Check | Frequency | Owner |
|-------|-----------|-------|
| Connector heartbeat (all active venues) | Every 5 min | `monitoring/` |
| Position reconciliation (internal vs venue) | Hourly | `CentralAccounting/` |
| P&L snapshot | End of day | `CentralAccounting/` |
| Error rate per connector | Continuous | `shared/audit_logger.py` |
| LLM token usage report | End of day | BigBrainIntelligence |

### 5.2 Weekly Checks (Manual + Automated)

| Check | Action |
|-------|--------|
| Test withdrawal (Tier B/C venues) | Send small amount, verify receipt |
| API scope review | Confirm no unnecessary permissions |
| Rate limit headroom | Check usage vs limits |
| Key expiry scan | `api_key_manager.cleanup_expired_keys()` |

### 5.3 Monthly Reviews

| Review | Action |
|--------|--------|
| Tier allocation audit | Verify capital matches governance policy |
| Strategy-to-venue mapping | Confirm no Tier C dependency in core strategies |
| Incident drill | Simulate venue outage, rate limit, bad fill |
| Cost review | LLM spend, data API costs, exchange fees |

---

## 6. Incident Playbooks

### 6.1 Venue Down

```
1. Kill switch triggers → halt all strategies using affected venue
2. Check venue status page / social
3. If > 15 min: reroute arb legs to backup venue (if available)
4. Log incident in audit trail
5. Post-mortem within 24h
```

### 6.2 Rate Limited

```
1. Exponential backoff (built into base_connector.py)
2. If persistent: reduce request frequency
3. If blocking: switch to polling mode
4. Review rate limit headroom in weekly check
```

### 6.3 Bad Fill / Slippage

```
1. Log fill details (expected vs actual)
2. Compare against venue's slippage model
3. If systematic: flag venue for review
4. If Tier C: consider removing venue
```

### 6.4 Key Compromise Suspected

```
1. Immediately rotate affected key
2. Revoke old key at provider
3. Audit all activity since last confirmed-safe state
4. Review access logs
5. Update rotation schedule
```

---

## 7. Connector Implementation Status

| Venue | Connector | Tests | Health Check | Reconciliation |
|-------|-----------|-------|--------------|----------------|
| IBKR | ✅ Production | 55 | ✅ | ✅ |
| Moomoo | ✅ Implemented | — | ⏳ | ⏳ |
| NDAX | ✅ Primary | — | ⏳ | ⏳ |
| Metal X | ✅ Production | — | ✅ | ⏳ |
| Binance | ✅ Parked | — | — | — |
| Kraken | ✅ Parked | — | — | — |
| Coinbase | ✅ Parked | — | — | — |
| Noxi Rise | ✅ Sandbox | — | ⏳ | — |
| OANDA | ⏳ Planned | — | — | — |
| IC Markets | ⏳ Planned | — | — | — |
| Pepperstone | ⏳ Planned | — | — | — |

---

## 8. Recommended Priority (Implementation Backlog)

### P0 — Must Have (Stability + Safety)
1. ✅ Integration Registry (this document + JSON)
2. Connector health checks for all active venues
3. Automated daily reconciliation (IBKR + NDAX)
4. Kill switch per venue (not just global)
5. Audit event taxonomy (standardize across all connectors)

### P1 — Strategy Enablement
6. OANDA connector implementation
7. Normalize market data schema across ccxt + IBKR + MT5
8. Fee/slippage model per venue
9. Latency metrics per route
10. Arb leg planner (venue-aware constraints)

### P2 — Scale & Resilience
11. IC Markets connector (if FX becomes core)
12. Pepperstone connector (if FX becomes core)
13. Redis/Kafka operational hardening
14. Rate limiters per API key (centralized)
15. Key rotation workflows + expiry alerts
16. Canary execution (tiny orders) + rollback

---

*This policy is enforced by NCC Doctrine. Violations require documented exception approval.*

# APR10 ROLL EXECUTION PLAN — WealthSimple TFSA

> **Execution Date:** April 10, 2026 (7 DTE before Apr 17 expiry)
> **Budget:** ~C$614 USD (~0.72 FX rate)
> **Account:** WealthSimple TFSA ($18,637.76 CAD as of Mar 29)

---

## Pre-Execution Checklist

- [ ] Market open 9:30 ET — wait 15 min for spreads to settle
- [ ] Check live bid/ask on all positions below
- [ ] Confirm WS buying power ≥ C$614
- [ ] VIX level — if >35, consider delaying rolls (elevated premiums favor waiting)
- [ ] No earnings on underlying tickers this week

---

## Roll Matrix

### 1. OBDC $10P x65 → Jul $7.5P x65

| Field | Details |
|-------|---------|
| Action | STC Apr $10P x65, BTO Jul $7.5P x65 |
| Current P&L | +17% profit |
| Est. Net Cost | ~$65 USD |
| Rationale | Core BDC thesis intact. Lower strike captures further downside. Jul gives 3 more months of runway. |
| Risk | OBDC NAV stable → puts decay to zero. Acceptable loss at $65. |

### 2. ARCC $16P x10 → Jun $15P x10

| Field | Details |
|-------|---------|
| Action | STC Apr $16P x10, BTO Jun $15P x10 |
| Current P&L | Break-even |
| Est. Net Cost | ~$280 USD |
| Rationale | Private credit stress thesis still building. Lower strike + Jun expiry. |
| Risk | ARCC is high-quality BDC — $15 strike is deep OTM. |

### 3. JNK $94P x5 → Roll 2, Expire 3

| Field | Details |
|-------|---------|
| Action | STC Apr $94P x2, BTO Jun $92P x2. Let remaining 3 expire. |
| Current P&L | +36% profit on batch |
| Est. Net Cost | ~$90 USD |
| Rationale | High-yield credit repricing thesis still valid. 3 contracts at $0 bid — let expire. Only roll contracts with remaining value. |
| Risk | Credit spreads tighten → puts worthless. Manageable at 2 contracts. |

### 4. KRE $60P x1 → CLOSE

| Field | Details |
|-------|---------|
| Action | STC Apr $60P x1 for ~$94 credit |
| Current P&L | +17% profit |
| Est. Net Cost | Nets positive (~$94 credit) |
| Rationale | Best risk/reward is to take profit. KRE already had a position expire Apr 4 on IBKR side. Reduce regional bank concentration. |

---

## Budget Summary

| Roll | Est. Cost |
|------|-----------|
| OBDC → Jul $7.5P | $65 |
| ARCC → Jun $15P | $280 |
| JNK → Jun $92P (x2) | $90 |
| KRE close | -$94 (credit) |
| **Net cost** | **~$341 USD** |
| **Budget remaining** | ~$273 of C$614 |

---

## Positions NOT Rolling (Hold Through)

| Ticker | Strike | Qty | Expiry | Action |
|--------|--------|-----|--------|--------|
| GLD | $515C | 1 | Mar 19, 2027 | LEAPS — HOLD (+25%) |
| XLE | $85C | 26 | Jan 15, 2027 | LEAPS — HOLD (+162%) |
| OWL | $8P | 5 | Jun 18 | Hold — 2.5 months left |

---

## Post-Execution

- [ ] Screenshot fills with timestamps
- [ ] Update `strategies/war_room_engine.py` CURRENT_POSITIONS
- [ ] Update `.context/STATUS.md` positions table
- [ ] Log in `data/war_room/intel_log.jsonl`: `{"date": "2026-04-10", "type": "roll", "details": "..."}`
- [ ] If any fill fails, note and retry next trading day

---

## Contingencies

- **If VIX > 40**: Delay rolls 1-2 days — elevated vol means better premiums for new puts
- **If bid/ask spread > 20%**: Use limit orders at mid-price, wait 30 min
- **If WS buying power insufficient**: Prioritize OBDC (largest position) → JNK → ARCC
- **If ARCC/OBDC announces dividend cut before Apr 10**: Accelerate roll to same day

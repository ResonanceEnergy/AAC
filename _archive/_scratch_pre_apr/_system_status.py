"""Full system status check — run once and delete."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from strategies.war_room_engine import (
    ACCOUNTS,
    CURRENT_POSITIONS,
    MILESTONES,
    SPOT_PRICES,
    IndicatorState,
    check_milestones,
    compute_composite_score,
    get_current_phase,
    get_portfolio_value_usd,
)

ind = IndicatorState()
comp = compute_composite_score(ind)

print("=" * 60)
print("  AAC WAR ROOM — FULL SYSTEM STATUS — March 29 2026")
print("=" * 60)

# 1. Composite
print("\n--- COMPOSITE CRISIS SCORE ---")
print(f"  Score:      {comp['composite_score']} / 100")
print(f"  Regime:     {comp['regime']}")
print(f"  Confidence: {comp['confidence']}")
print()
for k, v in comp["individual_scores"].items():
    bar = "#" * int(v / 5)
    print(f"  {k:20s}: {v:5.1f}  {bar}")

# 2. Portfolio
print("\n--- PORTFOLIO ---")
pv = get_portfolio_value_usd()
phase = get_current_phase()
print(f"  Total USD:  ${pv:,.2f}")
print(f"  Phase:      {phase}")
print(f"  Positions:  {len(CURRENT_POSITIONS)}")
for p in CURRENT_POSITIONS:
    print(f"    {p.symbol:6s} {p.position_type:4s} ${p.strike:.0f} exp {p.expiry}  "
          f"entry ${p.entry_price:.2f} mkt ${p.current_price:.2f}  "
          f"P&L ${p.pnl:+.2f} ({p.pnl_pct:+.1f}%)")

# 3. Accounts
print("\n--- ACCOUNTS ---")
for name, acct in ACCOUNTS.items():
    bal = acct.get("balance_usd", acct.get("balance_cad", 0))
    cur = "USD" if "balance_usd" in acct else "CAD"
    print(f"  {name:15s}: ${bal:>10,.2f} {cur}  ({acct['type']})")

# 4. Spots
print("\n--- SPOT PRICES (live Mar 29) ---")
for k, v in SPOT_PRICES.items():
    print(f"  {k:8s}: {v:>10,.2f}")

# 5. Milestones
print("\n--- MILESTONE CHECK ---")
state = {
    "oil_price": ind.oil_price, "gold_price": ind.gold_price,
    "vix": ind.vix, "spy_price": ind.spy_price, "btc_price": ind.btc_price,
    "hy_spread_bp": ind.hy_spread_bp, "bdc_nav_discount": ind.bdc_nav_discount,
    "bdc_nonaccrual_pct": ind.bdc_nonaccrual_pct, "eth_price": 1993.0,
    "xlf_price": 48.0, "defi_tvl_drop_pct": 35.0, "qqq_price": 563.0,
    "defi_tvl_change_pct": -35.0, "stablecoin_depeg_pct": 0.2,
    "fed_funds_rate": 4.5, "dxy": 103.0,
}
triggered = check_milestones(state)
if triggered:
    print(f"  NEWLY TRIGGERED ({len(triggered)}):")
    for m in triggered:
        print(f"    #{m.id}: {m.name}  (confidence {m.confidence:.0%})")

# Near-trigger
print("\n  NEAR-TRIGGER (within 15%):")
near_count = 0
for m in MILESTONES:
    if m.triggered:
        continue
    val = state.get(m.threshold_field)
    if val is None or val == 0:
        continue
    if m.threshold_op == ">" and m.threshold_value > 0:
        pct = val / m.threshold_value
        if 0.85 < pct < 1.0:
            print(f"    #{m.id}: {m.name} -- {val:.1f} / {m.threshold_value:.1f} ({pct:.0%})")
            near_count += 1
    elif m.threshold_op == "<" and val > 0:
        pct = val / m.threshold_value
        if 1.0 < pct < 1.18:
            print(f"    #{m.id}: {m.name} -- {val:.1f} / {m.threshold_value:.1f} ({1/pct:.0%})")
            near_count += 1
if near_count == 0:
    print("    (none)")

# Already triggered
already = [m for m in MILESTONES if m.triggered]
print(f"\n  ALREADY TRIGGERED ({len(already)}):")
for m in already:
    print(f"    #{m.id}: {m.name}  @ {m.triggered_date}")

# 6. Quick MC
print("\n--- MONTE CARLO (quick 20K paths) ---")
from strategies.war_room_engine import run_monte_carlo

mc = run_monte_carlo(n_paths=20_000)
print(f"  Portfolio mean:   ${mc.portfolio_mean:,.2f}")
print(f"  Portfolio median: ${mc.portfolio_median:,.2f}")
print(f"  P5 / P95:        ${mc.portfolio_p5:,.2f} / ${mc.portfolio_p95:,.2f}")
print(f"  VaR 95:           ${mc.var_95:,.2f}")
print(f"  CVaR 95:          ${mc.cvar_95:,.2f}")
print(f"  P(oil > $120):    {mc.prob_oil_above_120:.1%}")
print(f"  P(gold > $5500):  {mc.prob_gold_above_3500:.1%}")
print(f"  P(SPY < $560):    {mc.prob_spy_below_500:.1%}")
print(f"  P(BTC < $50K):    {mc.prob_btc_below_60k:.1%}")
print(f"  P(PF > $150K):    {mc.prob_portfolio_above_150k:.1%}")
print(f"  P(PF > $1M):      {mc.prob_portfolio_above_1m:.1%}")
print(f"  Runtime:          {mc.runtime_ms:.0f}ms")

# 7. Outstanding issues
print("\n" + "=" * 60)
print("  OUTSTANDING ISSUES")
print("=" * 60)

issues = []

# Stale positions (IBKR only — Moomoo was just scanned)
issues.append("IBKR POSITIONS: Option mkt values from Mar 20 (9 days stale) -- launch TWS + re-pull")

# Account balances
issues.append("NDAX: $4,492 CAD unverified -- ccxt ndax.login attr broken, exchange may have updated API")
issues.append("WealthSimple/EQ Bank: $0 -- no API, manual check needed")

# Expiry dates
for p in CURRENT_POSITIONS:
    if p.expiry and p.expiry <= "2026-04-17":
        issues.append(f"EXPIRY WARNING: {p.symbol} {p.position_type} ${p.strike:.0f} expires {p.expiry} (< 3 weeks)")

# VIX regime
if ind.vix > 30:
    issues.append(f"VIX at {ind.vix} -- CRISIS mode, IV elevated, new option entries expensive")

# Near milestones
for m in MILESTONES:
    if m.triggered:
        continue
    val = state.get(m.threshold_field)
    if val is None or val == 0:
        continue
    if m.threshold_op == ">" and m.threshold_value > 0:
        if val / m.threshold_value > 0.90:
            issues.append(f"MILESTONE #{m.id} '{m.name}' at {val/m.threshold_value:.0%} -- watch closely")
    elif m.threshold_op == "<" and val > 0:
        if m.threshold_value / val > 0.90:
            issues.append(f"MILESTONE #{m.id} '{m.name}' at {m.threshold_value/val:.0%} -- watch closely")

# Cloudflare tunnel
issues.append("CLOUDFLARE: Verify tunnel 'ncc' (afc84bab) still serving aac.bit-rage-labour.com")

# IBKR connection
issues.append("IBKR: Verify TWS/IB Gateway running on port 7496 LIVE before next trade")

# Storyboard
issues.append("STORYBOARD: Verify port 8502 still serving (war_room_storyboard.py)")
issues.append("DASHBOARD: Verify port 8501 still serving (streamlit_dashboard.py)")

for i, issue in enumerate(issues, 1):
    print(f"  {i:2d}. {issue}")

print(f"\n  Total issues: {len(issues)}")
print("=" * 60)

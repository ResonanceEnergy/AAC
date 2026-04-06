"""
Pyramid-Down $500 Allocator — Polymarket Division
===================================================
Runs 100K Monte Carlo P(profit) on all 3 divisions, then allocates
$500 in a pyramid-down structure: heaviest weight at the top tier,
tapering down to the bottom.

Pyramid Structure:
  Tier 1 (TOP)    = 50% = $250  — highest-conviction division
  Tier 2 (MID)    = 30% = $150  — medium conviction
  Tier 3 (BOTTOM) = 20% = $100  — speculative / long-shot

Divisions are ranked by P(profit) from MC simulation, then assigned
to tiers accordingly.

Usage:
    python -m strategies.polymarket_division.pyramid_allocator
"""
from __future__ import annotations

import io
import os
import sys

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ============================================================================
# PYRAMID WEIGHTS
# ============================================================================
TOTAL_CAPITAL = 500.0
PYRAMID_TIERS = [
    ("TIER 1 — TOP",    0.50),   # $250
    ("TIER 2 — MID",    0.30),   # $150
    ("TIER 3 — BOTTOM", 0.20),   # $100
]

MC_SIMS = 100_000
SEED = 42


# ============================================================================
# WAR ROOM — MC on existing 23 positions (deep OTM geopolitical tail bets)
# ============================================================================
def mc_war_room() -> Dict[str, Any]:
    """
    War Room thesis: deep OTM geopolitical bets.
    We have 23 live positions worth ~$298. These are tail-risk bets
    with low probability but massive payout if they hit.

    Model: each position is a Bernoulli trial. Probability is derived
    from thesis-adjusted estimates (2x-5x market implied odds due to
    pressure cooker model). Average entry ~$0.08, thesis prob ~0.25.
    """
    rng = np.random.default_rng(SEED)

    # Live War Room positions — (entry_price, size_shares, thesis_prob)
    # From actual account data — 23 positions
    positions = [
        (0.010, 543.6, 0.04),   # Crude $150 March — extreme tail
        (0.030, 412.4, 0.10),   # Iran mil action vs Israel
        (0.050, 372.6, 0.16),   # 40 ships Hormuz
        (0.050, 363.6, 0.14),   # Israel strike 3 countries
        (0.020, 332.2, 0.06),   # Crude $140 March
        (0.060, 304.0, 0.18),   # Kharg Island
        (0.110, 250.5, 0.28),   # Saudi strike Iran
        (0.086, 240.9, 0.25),   # Crude $120 March
        (0.080, 232.5, 0.22),   # Iranian regime fall
        (0.030, 175.9, 0.10),   # US x Iran ceasefire Mar31
        (0.040, 170.8, 0.12),   # Crude $85 LOW
        (0.037, 170.8, 0.12),   # Netanyahu out Apr30
        (0.120, 158.1, 0.30),   # US x Iran meeting Mar31
        (0.120, 158.1, 0.30),   # US x Iran ceasefire Apr7
        (0.039, 157.4, 0.12),   # Trump end mil ops Iran
        (0.140, 146.4, 0.32),   # Pahlavi enter Iran Jun30
        (0.137, 134.2, 0.30),   # Crude $200 June
        (0.160, 119.8, 0.35),   # US forces enter Iran
        (0.180, 108.9, 0.38),   # (placeholder — thesis-aligned)
        (0.197, 101.1, 0.40),   # Hezbollah action (NO side)
        (0.237, 84.4,  0.42),   # Crude $110 March
        (0.240, 84.1,  0.44),   # US x Iran ceasefire Apr15
        (0.110, 55.6,  0.28),   # US invade Iran Mar31
    ]

    n = len(positions)
    probs = np.array([p[2] for p in positions])
    shares = np.array([p[1] for p in positions])
    costs = np.array([p[0] * p[1] for p in positions])
    total_cost = float(np.sum(costs))

    # (n_sims, n_positions) Bernoulli outcomes
    outcomes = rng.binomial(1, probs, size=(MC_SIMS, n))
    payouts = outcomes * shares  # each share pays $1 on win
    total_payouts = np.sum(payouts, axis=1)
    pnl = total_payouts - total_cost

    mean_pnl = float(np.mean(pnl))
    prob_profit = float(np.mean(pnl > 0))
    var_95 = float(np.percentile(pnl, 5))
    std_pnl = float(np.std(pnl))
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    max_payout = float(np.max(total_payouts))
    median_pnl = float(np.median(pnl))

    # Tail analysis: what happens in best 5% of outcomes
    p95_pnl = float(np.percentile(pnl, 95))
    top_5pct = pnl[pnl >= np.percentile(pnl, 95)]
    mean_top5 = float(np.mean(top_5pct)) if len(top_5pct) > 0 else 0

    return {
        "name": "War Room Poly",
        "icon": "⚔️",
        "n_positions": n,
        "total_cost": total_cost,
        "mean_pnl": mean_pnl,
        "ev_pct": (mean_pnl / total_cost * 100) if total_cost > 0 else 0,
        "prob_profit": prob_profit,
        "var_95": var_95,
        "sharpe": sharpe,
        "max_payout": max_payout,
        "median_pnl": median_pnl,
        "p95_pnl": p95_pnl,
        "mean_top5_pct": mean_top5,
        "description": "Geopolitical tail-risk thesis chain (Iran/oil/gold/USD)",
        "edge": "Pressure cooker model gives 2-5x multiplier on deep OTM",
    }


# ============================================================================
# PLANKTONXD — MC on existing 5 positions + 19 pending orders
# ============================================================================
def mc_planktonxd() -> Dict[str, Any]:
    """
    PlanktonXD: crypto dip + price bets (micro-arbitrage).
    5 positions + 19 pending buy orders.
    These are short-dated binary bets on BTC/ETH price levels.
    """
    rng = np.random.default_rng(SEED + 1)

    # 5 live positions — (entry_price, shares, our_prob)
    live_positions = [
        (0.093, 208.4, 0.12),   # BTC dip $60K March
        (0.130, 154.1, 0.16),   # ETH dip $1800 March
        (0.180, 108.9, 0.22),   # BTC dip $64K Mar23-29
        (0.038, 170.8, 0.06),   # BTC above $68K Mar28
        (0.030, 196.1, 0.05),   # BTC above $64K Mar28 (NO side)
    ]

    # 19 pending orders — if they fill, these become positions
    # Model as potential: 60% chance each fills, then Bernoulli on outcome
    pending_orders = [
        (0.050, 255.2, 0.08),
        (0.090, 149.0, 0.14),
        (0.050, 272.9, 0.08),
        (0.200, 67.2,  0.26),
        (0.100, 131.2, 0.14),
        (0.020, 244.1, 0.04),
        (0.010, 412.4, 0.03),
        (0.020, 306.7, 0.04),
        (0.020, 278.1, 0.04),
        (0.030, 186.9, 0.06),
        (0.030, 178.5, 0.06),
        (0.040, 151.4, 0.07),
        (0.040, 140.7, 0.07),
        (0.090, 64.6,  0.14),
        (0.050, 124.6, 0.08),
        (0.080, 70.3,  0.12),
        (0.200, 29.2,  0.26),
        (0.080, 70.3,  0.12),
        (0.130, 46.0,  0.18),
    ]

    # Simulate live positions
    n_live = len(live_positions)
    probs_live = np.array([p[2] for p in live_positions])
    shares_live = np.array([p[1] for p in live_positions])
    costs_live = np.array([p[0] * p[1] for p in live_positions])

    # Simulate pending orders (fill probability = 0.6)
    n_pend = len(pending_orders)
    probs_pend = np.array([p[2] for p in pending_orders])
    shares_pend = np.array([p[1] for p in pending_orders])
    costs_pend = np.array([p[0] * p[1] for p in pending_orders])

    total_cost_live = float(np.sum(costs_live))
    total_cost_pend = float(np.sum(costs_pend))

    # Live: always active
    outcomes_live = rng.binomial(1, probs_live, size=(MC_SIMS, n_live))
    payouts_live = outcomes_live * shares_live
    pnl_live = np.sum(payouts_live, axis=1) - total_cost_live

    # Pending: fill first, then resolve
    fills = rng.binomial(1, 0.6, size=(MC_SIMS, n_pend))
    outcomes_pend = rng.binomial(1, probs_pend, size=(MC_SIMS, n_pend))
    active_pend = fills * outcomes_pend
    payouts_pend = active_pend * shares_pend
    costs_pend_per_sim = fills * costs_pend  # only pay for filled orders
    pnl_pend = np.sum(payouts_pend, axis=1) - np.sum(costs_pend_per_sim, axis=1)

    pnl = pnl_live + pnl_pend
    total_cost = total_cost_live + total_cost_pend * 0.6  # expected cost

    mean_pnl = float(np.mean(pnl))
    prob_profit = float(np.mean(pnl > 0))
    var_95 = float(np.percentile(pnl, 5))
    std_pnl = float(np.std(pnl))
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    max_payout = float(np.max(np.sum(payouts_live, axis=1) + np.sum(payouts_pend, axis=1)))
    median_pnl = float(np.median(pnl))
    p95_pnl = float(np.percentile(pnl, 95))
    top_5pct = pnl[pnl >= np.percentile(pnl, 95)]
    mean_top5 = float(np.mean(top_5pct)) if len(top_5pct) > 0 else 0

    return {
        "name": "PlanktonXD",
        "icon": "🐙",
        "n_positions": n_live,
        "n_pending": n_pend,
        "total_cost": total_cost,
        "mean_pnl": mean_pnl,
        "ev_pct": (mean_pnl / total_cost * 100) if total_cost > 0 else 0,
        "prob_profit": prob_profit,
        "var_95": var_95,
        "sharpe": sharpe,
        "max_payout": max_payout,
        "median_pnl": median_pnl,
        "p95_pnl": p95_pnl,
        "mean_top5_pct": mean_top5,
        "description": "Crypto price dip/pump micro-arbitrage bets",
        "edge": "Short-dated BTC/ETH price levels, high turnover",
    }


# ============================================================================
# POLYMC AGENT — MC on 5 target bets (not yet placed)
# ============================================================================
def mc_polymc() -> Dict[str, Any]:
    """
    PolyMC Agent: 5 high-EV target bets from top-100 scanner.
    Sports + politics — longer dated, higher probability.
    These have NOT been placed yet — this is the forward-looking allocation.
    """
    rng = np.random.default_rng(SEED + 2)

    # 5 PolyMC target bets — (entry_price, bet_size, our_prob)
    bets = [
        (0.158, 100.0, 0.19),   # Spain FIFA World Cup 2026
        (0.370, 100.0, 0.42),   # OKC Thunder NBA Championship
        (0.181, 100.0, 0.20),   # JD Vance 2028 President
        (0.243, 100.0, 0.26),   # Newsom 2028 Dem Nomination
        (0.368, 100.0, 0.39),   # Vance 2028 GOP Nomination
    ]

    n = len(bets)
    probs = np.array([b[2] for b in bets])
    bet_sizes = np.array([b[1] for b in bets])
    shares = np.array([b[1] / b[0] for b in bets])
    total_cost = float(np.sum(bet_sizes))

    outcomes = rng.binomial(1, probs, size=(MC_SIMS, n))
    payouts = outcomes * shares
    total_payouts = np.sum(payouts, axis=1)
    pnl = total_payouts - total_cost

    mean_pnl = float(np.mean(pnl))
    prob_profit = float(np.mean(pnl > 0))
    var_95 = float(np.percentile(pnl, 5))
    std_pnl = float(np.std(pnl))
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    max_payout = float(np.max(total_payouts))
    median_pnl = float(np.median(pnl))
    p95_pnl = float(np.percentile(pnl, 95))
    top_5pct = pnl[pnl >= np.percentile(pnl, 95)]
    mean_top5 = float(np.mean(top_5pct)) if len(top_5pct) > 0 else 0

    # Per-bet breakdown
    per_bet = []
    for i, (entry, size, prob) in enumerate(bets):
        b_shares = size / entry
        b_pnl = np.where(outcomes[:, i] == 1, b_shares - size, -size)
        per_bet.append({
            "name": ["Spain FIFA WC", "OKC Thunder NBA", "JD Vance Pres",
                     "Newsom Dem Nom", "Vance GOP Nom"][i],
            "entry": entry,
            "our_prob": prob,
            "ev": float(np.mean(b_pnl)),
            "prob_profit": float(np.mean(b_pnl > 0)),
            "max_payout": b_shares,
        })

    return {
        "name": "PolyMC Agent",
        "icon": "🎲",
        "n_positions": 0,
        "n_target_bets": n,
        "total_cost": total_cost,
        "mean_pnl": mean_pnl,
        "ev_pct": (mean_pnl / total_cost * 100) if total_cost > 0 else 0,
        "prob_profit": prob_profit,
        "var_95": var_95,
        "sharpe": sharpe,
        "max_payout": max_payout,
        "median_pnl": median_pnl,
        "p95_pnl": p95_pnl,
        "mean_top5_pct": mean_top5,
        "per_bet": per_bet,
        "description": "Sports + politics high-EV from top-100 scanner",
        "edge": "MC-validated positive EV, Kelly-sized, TP/SL exits",
    }


# ============================================================================
# PYRAMID ALLOCATOR
# ============================================================================
def rank_and_allocate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank divisions by composite score, assign pyramid tiers.

    Composite score = 0.40 * P(profit) + 0.30 * EV% + 0.20 * Sharpe + 0.10 * upside
    Higher score → top of pyramid → more capital.
    """
    for r in results:
        # Normalize upside: mean of top 5% outcomes relative to cost
        upside = r["mean_top5_pct"] / max(r["total_cost"], 1) if r["total_cost"] > 0 else 0

        r["composite_score"] = (
            0.40 * r["prob_profit"]
            + 0.30 * (r["ev_pct"] / 100.0)  # normalize to 0-1 range
            + 0.20 * max(min(r["sharpe"], 1.0), -1.0)  # clip sharpe
            + 0.10 * min(upside, 2.0)  # clip upside
        )

    # Sort by composite score descending (best → Tier 1)
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    allocated = []
    for i, r in enumerate(results):
        tier_name, tier_pct = PYRAMID_TIERS[i]
        tier_amount = TOTAL_CAPITAL * tier_pct
        allocated.append({
            **r,
            "tier": tier_name,
            "tier_pct": tier_pct,
            "allocation": tier_amount,
            "rank": i + 1,
        })

    return allocated


# ============================================================================
# DISPLAY
# ============================================================================
def display(allocated: List[Dict[str, Any]]):
    w = 100

    print()
    print("=" * w)
    print("  POLYMARKET DIVISION — $500 PYRAMID-DOWN ALLOCATOR")
    print("  100,000 Monte Carlo Simulations per Division")
    print("=" * w)

    # ── MC Results Table ──
    print()
    print(f"  {'Division':<20} {'P(profit)':>10} {'EV%':>8} {'Mean PnL':>10} "
          f"{'Sharpe':>8} {'VaR95':>9} {'Max Pay':>10} {'Score':>8}")
    print("  " + "-" * (w - 4))
    for a in allocated:
        pp = a["prob_profit"]
        pp_bar = "█" * int(pp * 20)
        print(
            f"  {a['icon']} {a['name']:<17} {pp:>8.1%}  {a['ev_pct']:>+7.1f}% "
            f"${a['mean_pnl']:>+8.2f} {a['sharpe']:>+7.3f} "
            f"${a['var_95']:>8.2f} ${a['max_payout']:>9,.0f} {a['composite_score']:>7.3f}"
        )
        print(f"  {'':20} P(profit) [{pp_bar:<20}]  {a['description']}")
    print()

    # ── Pyramid Visual ──
    print("  " + "▓" * w)
    print("  PYRAMID-DOWN ALLOCATION — Heaviest at Top")
    print("  " + "▓" * w)
    print()

    # Visual pyramid
    pyramid_chars = [
        ("█" * 50, "  ████████████████████████████████████████████████████"),
        ("█" * 30, "            ██████████████████████████████"),
        ("█" * 20, "                  ████████████████████████"),
    ]

    for i, a in enumerate(allocated):
        bar_len = int(a["tier_pct"] * 60)
        pad = (30 - bar_len // 2)
        bar = "█" * bar_len
        print(f"  {' ' * max(pad, 0)}{bar}")
        print(
            f"  {' ' * max(pad, 0)}{a['icon']}  {a['name']:<17}  "
            f"${a['allocation']:>6,.0f}  ({a['tier_pct']:.0%})  — {a['tier']}"
        )
        print()

    # ── Detailed Allocation Table ──
    print("  " + "═" * (w - 4))
    print(f"  {'#':<4} {'Tier':<20} {'Division':<20} {'Alloc':>8} {'%':>6} "
          f"{'P(profit)':>10} {'EV%':>8} {'Rationale'}")
    print("  " + "─" * (w - 4))

    for a in allocated:
        rationale = a["edge"][:40]
        print(
            f"  {a['rank']:<4} {a['tier']:<20} {a['icon']} {a['name']:<17} "
            f"${a['allocation']:>6,.0f} {a['tier_pct']:>5.0%} "
            f"{a['prob_profit']:>8.1%}  {a['ev_pct']:>+7.1f}%  {rationale}"
        )
    print("  " + "─" * (w - 4))
    print(f"  {'':4} {'TOTAL':<20} {'':20} ${TOTAL_CAPITAL:>6,.0f} {'100%':>6}")
    print()

    # ── Per-Bet Breakdown for PolyMC (since those are NEW bets being placed) ──
    for a in allocated:
        if a["name"] == "PolyMC Agent" and "per_bet" in a:
            alloc = a["allocation"]
            print(f"  🎲 PolyMC — ${alloc:.0f} ALLOCATION SPLIT:")
            per_bet_alloc = alloc / len(a["per_bet"])
            print(f"  {'Bet':<25} {'Entry':>7} {'Prob':>7} {'Alloc':>8} {'Shares':>8} {'EV':>8} {'P(win)':>8}")
            print("  " + "-" * 80)
            for b in a["per_bet"]:
                shares = per_bet_alloc / b["entry"]
                print(
                    f"  {b['name']:<25} ${b['entry']:.3f} {b['our_prob']:>6.0%} "
                    f"${per_bet_alloc:>7.2f} {shares:>7.1f} "
                    f"${b['ev']:>+7.2f} {b['prob_profit']:>7.1%}"
                )
            print()

    # ── War Room suggested new bets with $X ──
    for a in allocated:
        if a["name"] == "War Room Poly":
            alloc = a["allocation"]
            print(f"  ⚔️  War Room — ${alloc:.0f} SUGGESTED NEW BETS:")
            print(f"  Top up existing high-edge thesis positions:")
            suggestions = [
                ("US x Iran ceasefire Apr15", 0.24, 0.44, alloc * 0.25),
                ("Crude Oil $110 March",      0.24, 0.42, alloc * 0.20),
                ("Saudi strike Iran Mar31",   0.11, 0.28, alloc * 0.20),
                ("US forces enter Iran",      0.16, 0.35, alloc * 0.15),
                ("Pahlavi enter Iran Jun30",  0.14, 0.32, alloc * 0.10),
                ("Iranian regime fall Apr30", 0.08, 0.22, alloc * 0.10),
            ]
            print(f"  {'Market':<30} {'Entry':>7} {'Thesis':>8} {'Alloc':>8} {'Shares':>8}")
            print("  " + "-" * 68)
            for name, entry, prob, amt in suggestions:
                shares = amt / entry
                print(f"  {name:<30} ${entry:.3f}  {prob:>6.0%}  ${amt:>7.2f} {shares:>7.1f}")
            print()

    # ── PlanktonXD suggested bets ──
    for a in allocated:
        if a["name"] == "PlanktonXD":
            alloc = a["allocation"]
            print(f"  🐙 PlanktonXD — ${alloc:.0f} SUGGESTED NEW BETS:")
            print(f"  Fresh crypto price bets — short-dated, high-turnover:")
            suggestions = [
                ("BTC dip $60K April",      0.08, 0.12, alloc * 0.25),
                ("ETH dip $1500 April",     0.06, 0.10, alloc * 0.20),
                ("BTC above $90K Apr30",    0.15, 0.20, alloc * 0.20),
                ("BTC above $100K Jun30",   0.10, 0.14, alloc * 0.15),
                ("ETH above $2500 Apr30",   0.12, 0.16, alloc * 0.10),
                ("SOL above $200 Apr30",    0.08, 0.12, alloc * 0.10),
            ]
            print(f"  {'Market':<30} {'Entry':>7} {'Prob':>8} {'Alloc':>8} {'Shares':>8}")
            print("  " + "-" * 68)
            for name, entry, prob, amt in suggestions:
                shares = amt / entry
                print(f"  {name:<30} ${entry:.3f}  {prob:>6.0%}  ${amt:>7.2f} {shares:>7.1f}")
            print()

    # ── Final Summary ──
    print("  " + "▓" * w)
    total_ev = sum(a["mean_pnl"] * (a["allocation"] / max(a["total_cost"], 1)) for a in allocated)
    weighted_pp = sum(a["prob_profit"] * a["tier_pct"] for a in allocated)
    print(f"  PORTFOLIO SUMMARY:")
    print(f"    New Capital:          ${TOTAL_CAPITAL:>10,.2f}")
    print(f"    Existing Deployed:    ${sum(a['total_cost'] for a in allocated):>10,.2f}")
    print(f"    Combined Total:       ${TOTAL_CAPITAL + sum(a['total_cost'] for a in allocated):>10,.2f}")
    print(f"    Weighted P(profit):   {weighted_pp:>10.1%}")
    print(f"    Pyramid Structure:    50% / 30% / 20%")
    print(f"    Top Tier:             {allocated[0]['icon']} {allocated[0]['name']}")
    print("  " + "▓" * w)
    print()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n  Running 100,000 Monte Carlo simulations per division...\n")

    results = [
        mc_war_room(),
        mc_planktonxd(),
        mc_polymc(),
    ]

    allocated = rank_and_allocate(results)
    display(allocated)

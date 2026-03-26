"""
AAC Options Intelligence — Pre-Market Dry Run Simulation
=========================================================
Simulates a full 9:15 AM ET scan cycle without IBKR connection.
Uses realistic crisis data and market prices for March 24, 2026.
"""
from __future__ import annotations

import logging
import os
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from strategies.options_intelligence.pipeline import (
    OptionsIntelligencePipeline,
    PipelineResult,
)
from strategies.options_intelligence.ai_scorer import TradeSetup, TradeScore
from strategies.options_intelligence.discord_notifier import (
    GasketDiscordNotifier,
    TradeRecommendation,
)
from strategies.macro_crisis_put_strategy import (
    CrisisMonitor,
    CrisisAssessment,
    CrisisSignal,
    CrisisVector,
    MacroCrisisPutEngine,
    PUT_PLAYBOOK,
    PutOrderSpec,
)


def main() -> None:
    print("=" * 70)
    print("  AAC OPTIONS INTELLIGENCE — PRE-MARKET DRY RUN SIMULATION")
    print("  Simulated time: 9:14 AM ET, Monday March 24, 2026")
    print("=" * 70)

    # ── 1. Config ──
    ACCOUNT_BALANCE = 8800.0
    VIX = 28.5  # Elevated but not panic
    REGIME = "CRISIS"
    print(f"\n[CONFIG] Account: ${ACCOUNT_BALANCE:,.0f} | VIX: {VIX} | Regime: {REGIME}")
    print(f"[CONFIG] PUT_PLAYBOOK: {len(PUT_PLAYBOOK)} symbols")

    # ── 2. Build Crisis Assessment (realistic March 2026 thesis) ──
    assessment = CrisisAssessment(signals=[
        CrisisSignal(
            vector=CrisisVector.WAR_ESCALATION,
            severity=0.85,
            description="Ukraine/Russia stalemate, Hormuz tensions rising",
            data_source="geopolitical",
        ),
        CrisisSignal(
            vector=CrisisVector.OIL_SHOCK,
            severity=0.70,
            description="Brent $92, Hormuz insurance premiums 3x",
            data_source="commodities",
        ),
        CrisisSignal(
            vector=CrisisVector.PRIVATE_CREDIT_COLLAPSE,
            severity=0.80,
            description="Blue Owl redemptions 9%, OBDC NAV -4%",
            data_source="sec_filings",
        ),
        CrisisSignal(
            vector=CrisisVector.STAGFLATION,
            severity=0.65,
            description="Core PCE 2.9%, GDP 1.1%, unemployment 4.3%",
            data_source="fed",
        ),
        CrisisSignal(
            vector=CrisisVector.CREDIT_CONTAGION,
            severity=0.60,
            description="HY spreads 420bps, CCC tier widening",
            data_source="market",
        ),
    ])
    severity = assessment.composite_severity
    critical = assessment.critical_count
    deploy = assessment.should_deploy_puts
    print(f"\n[CRISIS] Composite severity: {severity:.2f}")
    print(f"[CRISIS] Critical signals: {critical}")
    print(f"[CRISIS] Should deploy puts: {deploy}")
    for sig in assessment.signals:
        marker = "!!" if sig.severity >= 0.7 else "  "
        print(f"  {marker} {sig.vector.value}: {sig.severity:.0%} — {sig.description}")

    # ── 3. Realistic prices (Friday March 21 close estimates) ──
    prices = {
        "SPY": 564.20,
        "XLF": 43.80,
        "HYG": 76.50,
        "OWL": 19.20,
        "OBDC": 14.60,
        "KRE": 54.30,
        "QQQ": 478.90,
        "IWM": 208.50,
        "BKLN": 21.10,
        "LQD": 107.80,
    }
    print(f"\n[PRICES] Loaded {len(prices)} symbols:")
    for sym, px in prices.items():
        print(f"  {sym:6s} ${px:>8.2f}")

    # ── 4. Run Pipeline ──
    print("\n" + "-" * 70)
    print("  RUNNING OPTIONS INTELLIGENCE PIPELINE")
    print("-" * 70)
    pipeline = OptionsIntelligencePipeline(
        account_balance=ACCOUNT_BALANCE,
        min_score=60,
        paper_trading=True,
    )
    result = pipeline.run_sync(
        assessment=assessment,
        underlying_prices=prices,
        flow_data=None,
        chain_data=None,
        regime=REGIME,
        vix=VIX,
    )
    print("\n" + result.summary())

    # ── 5. Detailed Results ──
    print("\n" + "=" * 70)
    print("  ACTIONABLE ORDERS (score >= 60)")
    print("=" * 70)
    total_premium = 0.0
    recommendations = []

    for i, entry in enumerate(result.actionable_orders, 1):
        o = entry["order"]
        s = entry["score"]
        cost = o.contracts * o.max_price * 100
        total_premium += cost
        print(f"\n  {i}. {o.symbol} ${o.strike:.0f}P exp {o.expiry}")
        print(f"     {o.contracts}x @ ${o.max_price:.2f} = ${cost:.0f}")
        print(f"     Score: {s.score}/100 (thesis={s.thesis_score} flow={s.flow_score} "
              f"greeks={s.greeks_score} timing={s.timing_score} risk={s.risk_score})")
        print(f"     Confidence: {s.confidence} | Model: {s.model_used}")
        if s.reasoning:
            print(f"     Reasoning: {s.reasoning[:140]}")

        # Build Discord recommendation
        vectors = []
        for sig in assessment.signals:
            if sig.severity >= 0.6:
                vectors.append(sig.vector.value)
        recommendations.append(TradeRecommendation(
            symbol=o.symbol,
            strike=o.strike,
            expiry=o.expiry,
            contracts=o.contracts,
            max_price=o.max_price,
            score=s.score,
            crisis_vectors=vectors[:3],
            description=f"Thesis={s.thesis_score} Flow={s.flow_score} "
                        f"Greeks={s.greeks_score} Timing={s.timing_score} Risk={s.risk_score}",
        ))

    alloc_pct = total_premium / ACCOUNT_BALANCE * 100
    print(f"\n  TOTAL PREMIUM: ${total_premium:,.0f} ({alloc_pct:.1f}% of account)")
    max_alloc = ACCOUNT_BALANCE * 0.15
    budget_ok = "WITHIN BUDGET" if total_premium <= max_alloc else "OVER BUDGET"
    print(f"  Max allocation: ${max_alloc:,.0f} (15%) — {budget_ok}")

    if result.strong_orders:
        print(f"\n  STRONG ORDERS (score >= 80): {len(result.strong_orders)}")
        for e in result.strong_orders:
            print(f"     {e['order'].symbol} — score {e['score'].score}")

    if result.rejected_orders:
        print(f"\n  REJECTED ({len(result.rejected_orders)}):")
        for e in result.rejected_orders:
            o = e["order"]
            s = e["score"]
            print(f"     {o.symbol} ${o.strike:.0f}P — score {s.score} ({s.confidence})")

    # ── 6. Engine Status ──
    print("\n" + "=" * 70)
    print("  ENGINE STATUS")
    print("=" * 70)
    has_flow = "YES" if result.flow_convictions else "NO (UW not connected)"
    has_chains = "YES" if result.optimal_strikes else "NO (IBKR offline — Sunday)"
    print(f"  Pipeline version:  {result.pipeline_version}")
    print(f"  Flow data:         {has_flow}")
    print(f"  Chain data:        {has_chains}")
    print(f"  AI scorer:         heuristic (4 LLM keys available as fallback)")
    print(f"  PUT_PLAYBOOK:      {len(PUT_PLAYBOOK)} symbols")
    print(f"  Dynamic expansion: {len(result.dynamic_candidates)} candidates")
    print(f"  Account balance:   ${ACCOUNT_BALANCE:,.0f}")
    print(f"  Max allocation:    15% = ${max_alloc:,.0f}")
    print(f"  Max per-position:  3% = ${ACCOUNT_BALANCE * 0.03:,.0f}")
    print(f"  DRY RUN:           YES (no orders sent)")

    # ── 7. Post to Discord ──
    notifier = GasketDiscordNotifier()
    if notifier.configured and recommendations:
        print("\n[DISCORD] Posting simulation results to Gasket...")
        summary = (
            f"DRY RUN SIMULATION — {len(recommendations)} trades | "
            f"${total_premium:,.0f} premium | "
            f"Severity: {severity:.0%} | VIX: {VIX}"
        )
        msg_id = notifier.send_trade_plan(
            recommendations=recommendations,
            summary=summary,
            timeout_minutes=0,  # No confirmation wait (simulation)
        )
        if msg_id:
            print(f"[DISCORD] Posted! Message ID: {msg_id}")
        else:
            # Fall back to plain text
            lines = ["**DRY RUN — Pre-Market Scan Simulation**"]
            lines.append(f"Crisis severity: {severity:.0%} | VIX: {VIX} | Regime: {REGIME}")
            lines.append(f"Trades: {len(recommendations)} | Premium: ${total_premium:,.0f}")
            lines.append("")
            for r in recommendations:
                cost = r.contracts * r.max_price * 100
                lines.append(f"**{r.symbol}** ${r.strike:.0f}P {r.expiry} "
                             f"| {r.contracts}x @ ${r.max_price:.2f} (${cost:.0f}) "
                             f"| Score: {r.score}")
            lines.append(f"\nTotal: ${total_premium:,.0f} / ${max_alloc:,.0f} budget")
            lines.append("_This is a simulation — no orders placed_")
            msg_id = notifier.send_status_update("\n".join(lines))
            if msg_id:
                print(f"[DISCORD] Sent plain text. ID: {msg_id}")
            else:
                print("[DISCORD] Failed to send")
    elif not notifier.configured:
        print("\n[DISCORD] Not configured — skipping")
    else:
        print("\n[DISCORD] No recommendations to post")

    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

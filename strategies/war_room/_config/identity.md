# War Room Identity & Voice

> Canonical source for thesis, mandate format, and reporting conventions.
> Referenced by: stages/05-report

## Thesis (April 2026)

**Stagflation + Geopolitical Crisis**

The world is entering a stagflationary regime driven by:
- Iran/Hormuz oil supply disruption (oil at $100+, ground troops talk)
- Private credit stress (BDC non-accruals rising, HY spreads at 600bp)
- Crypto winter (BTC at $66K, ETH crashed 48%, DeFi TVL down 35%)
- Gold at $4,524 (pulled back from $4,861 ATH, still in secular bull)
- VIX at 31, SPY down 5%, QQQ in correction (-8.3% YTD)

Probability-weighted: Stagflation 70%, Vol Shock 40/100.

## Mandate Format

The daily mandate is a JSON document with these fields:

```
{
  "date": "YYYY-MM-DD",
  "regime": "CALM|WATCH|ELEVATED|CRISIS",
  "composite_score": 0-100,
  "phase": "accumulation|growth|rotation|preservation",
  "portfolio_usd": 00000.00,
  "top_actions": ["action1", "action2", "action3"],
  "arm_status": { "arm_name": "% allocation" },
  "milestones_triggered": [],
  "milestones_approaching": [],
  "roll_alerts": ["position needing roll"],
  "risk_flags": ["any active risk warnings"]
}
```

## Reporting Conventions

- All dollar amounts in USD unless marked CAD
- Prices are closing prices from previous trading day
- Options values per-contract (multiply by 100 for total)
- P&L shown as both $ and % change
- Use ASCII only — no Unicode symbols (Windows cp1252 compatibility)
- Timestamps in UTC with ISO 8601 format

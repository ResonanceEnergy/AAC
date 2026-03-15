# Portfolio Entropy Budget + Valves (FF_AAC_RISK_000)

## Portfolio Budget
- Max daily loss: __%
- Max weekly loss: __%
- Max drawdown: __%
- Max leverage: __x
- Max correlation cluster: __ (define scale)
- Max allowed entropy regime: E__ (1–5)

## Valves (Automatic)
- If entropy regime rises by 1 level → reduce exposure by __%
- If correlation cluster breaches threshold → cap total gross exposure at __%
- If liquidity shock proxy triggers → pause tight-stop cells for __ hours/days
- If drawdown breaches __% → kill switch (portfolio) + cooldown

## Recovery Protocol
1) Pause / throttle
2) Diagnose (regime mismatch / execution / drift)
3) Re-validate
4) Resume with reduced size

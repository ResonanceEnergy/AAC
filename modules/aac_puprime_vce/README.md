# aac_puprime_vce — Volatility Compression → Expansion Strategy Module

Drop-in AAC module for hypothetical VCE (Volatility Compression → Expansion)
strategy backtesting, signal generation, and trade journaling.

**Instruments:** XAUUSD · EURUSD · BTCUSD  
**Data source (v1):** MT5/PU Prime CSV exports  
**Stack:** Python · VS Code · GitHub · Copilot  

> ⚠️ **Disclaimer**: This is a hypothetical strategy for education and testing.
> Not financial advice. CFDs carry high risk. Past performance ≠ future results.

## Quick Start

```bash
# 1. Drop MT5 CSV exports into data/raw/
cp ~/Downloads/XAUUSD_H1.csv modules/aac_puprime_vce/data/raw/

# 2. Ingest & normalize
python modules/aac_puprime_vce/scripts/run_ingest.py

# 3. Backtest all instruments
python modules/aac_puprime_vce/scripts/run_backtest.py --all

# 4. Generate current signals (paper mode)
python modules/aac_puprime_vce/scripts/run_signals.py --all

# 5. Build weekly report
python modules/aac_puprime_vce/scripts/run_report.py
```

## Strategy Overview (VCE)

Markets alternate between **compression** (low volatility) and **expansion**
(breakout / trend impulse). VCE detects compression regimes, applies a daily
trend filter, and triggers entries on range breakouts.

### Rules

| Step | Rule |
|------|------|
| Compression | ATR(14) in bottom 20% of last 100 bars AND BB-width in bottom 20% |
| Direction | Long only if D1 close > MA50; Short only if D1 close < MA50 |
| Entry | Close breaks above prior 20-bar high (long) or below 20-bar low (short) |
| Stop | 1.5 × ATR below/above entry |
| Take Profit | 2R (twice the risk distance) |
| Trail | After 1R, trailing stop at 1× ATR |
| Time Stop | Exit after 24 signal-TF bars if no TP/SL hit |

### Risk Controls

| Rule | Default |
|------|---------|
| Risk per trade | 1% of equity |
| Max daily drawdown | 4% |
| Max campaign drawdown | 20% |
| Max open positions | 2 |
| Kill-switch | 4 consecutive losses → 24h cooldown |

## Directory Structure

```
modules/aac_puprime_vce/
  config/           # YAML configuration (strategy, risk, costs, instruments)
  data/raw/         # Drop MT5 CSV exports here
  data/processed/   # Normalized candle data (auto-generated)
  src/              # Core Python package
    ingest/         # CSV parsing + normalization
    features/       # Indicators + regime detection
    strategy/       # VCE signals, sizing, risk controls
    backtest/       # Event-driven backtester + metrics
    journal/        # Trade journal schema + writer
    adapters/       # Broker interface (manual MT5 pack)
  scripts/          # CLI runner scripts
  tests/            # Unit + smoke tests
  reports/          # Generated reports + charts
```

## Extending

- Add instruments: edit `config/instruments.yaml`
- Change strategy params: edit `config/strategy.yaml`
- Adjust risk: edit `config/risk.yaml`
- Add broker API: implement `adapters/broker_base.py` interface

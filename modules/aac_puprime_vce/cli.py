"""CLI entrypoints for aac_puprime_vce module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the module root is on sys.path for direct script invocation
_MODULE_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _MODULE_ROOT.parent.parent
for p in (_MODULE_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.aac_puprime_vce.src.config import (
    load_instruments,
    load_strategy,
    load_risk,
    load_costs,
)
from modules.aac_puprime_vce.src.ingest.mt5_csv import read_mt5_csv
from modules.aac_puprime_vce.src.ingest.normalize import validate_ohlcv, normalize, save_processed
from modules.aac_puprime_vce.src.strategy.portfolio import compute_signals_for_instrument
from modules.aac_puprime_vce.src.backtest.engine import backtest_instrument
from modules.aac_puprime_vce.src.backtest.metrics import compute_metrics
from modules.aac_puprime_vce.src.backtest.reports import (
    generate_markdown_report,
    save_report,
    save_equity_curve_csv,
)
from modules.aac_puprime_vce.src.journal.writer import write_trades_to_journal
from modules.aac_puprime_vce.src.strategy.risk_controls import make_initial_state


_DATA_RAW = _MODULE_ROOT / "data" / "raw"
_DATA_PROC = _MODULE_ROOT / "data" / "processed"
_REPORTS_DIR = _MODULE_ROOT / "reports"
_JOURNAL_DIR = _MODULE_ROOT / "journal"

ALL_SYMBOLS = ["XAUUSD", "EURUSD", "BTCUSD"]


def _find_csv(symbol: str) -> Path | None:
    """Find a CSV file for a symbol in data/raw/."""
    for pattern in (f"{symbol}*", f"{symbol.lower()}*"):
        matches = list(_DATA_RAW.glob(pattern))
        if matches:
            return matches[0]
    return None


def _resample_to_daily(df):
    """Resample intraday OHLCV to daily for trend filter."""
    return df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])


def ingest(symbols: list[str] | None = None) -> None:
    """Ingest and normalize MT5 CSV exports."""
    symbols = symbols or ALL_SYMBOLS
    _DATA_RAW.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        csv_path = _find_csv(sym)
        if csv_path is None:
            print(f"[SKIP] No CSV found for {sym} in {_DATA_RAW}")
            continue

        print(f"[INGEST] {sym} <- {csv_path.name}")
        df = read_mt5_csv(csv_path)
        warnings = validate_ohlcv(df)
        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")
        df = normalize(df)
        out = save_processed(df, _DATA_PROC, sym)
        print(f"  -> {out} ({len(df)} bars)")


def backtest(symbols: list[str] | None = None) -> None:
    """Run VCE backtest on processed data."""
    symbols = symbols or ALL_SYMBOLS
    strategy_cfg = load_strategy()
    risk_cfg = load_risk()
    costs_cfg = load_costs()

    all_metrics = {}
    all_trades = {}
    risk_state = make_initial_state(risk_cfg["starting_equity"])

    for sym in symbols:
        proc_path = _DATA_PROC / f"{sym}_processed.csv"
        if not proc_path.exists():
            print(f"[SKIP] No processed data for {sym}. Run ingest first.")
            continue

        print(f"[BACKTEST] {sym}")
        df = _load_processed(proc_path)
        daily_df = _resample_to_daily(df)

        signal_df = compute_signals_for_instrument(df, daily_df, strategy_cfg)

        sym_costs = costs_cfg.get(sym, {"spread_points": 0, "slippage_points": 0, "commission_per_lot": 0})
        trades, risk_state, eq = backtest_instrument(
            signal_df, sym, sym_costs, risk_cfg, strategy_cfg, risk_state
        )

        metrics = compute_metrics(trades, eq)
        all_metrics[sym] = metrics
        all_trades[sym] = trades

        save_equity_curve_csv(eq, _REPORTS_DIR / "charts", sym)
        print(f"  Trades: {metrics['total_trades']}  WR: {metrics['win_rate']:.1%}  "
              f"PnL: ${metrics['total_pnl']:.2f}  MaxDD: {metrics['max_drawdown_pct']:.1%}")

    # Campaign summary
    campaign = {
        "total_trades": sum(m["total_trades"] for m in all_metrics.values()),
        "total_pnl": round(sum(m["total_pnl"] for m in all_metrics.values()), 2),
        "final_equity": round(risk_state.equity, 2),
    }

    report_md = generate_markdown_report(all_metrics, all_trades, campaign)
    report_path = save_report(report_md, {"campaign": campaign, "by_symbol": all_metrics}, _REPORTS_DIR)
    print(f"\n[REPORT] {report_path}")

    # Journal
    all_trade_list = []
    for sym_trades in all_trades.values():
        all_trade_list.extend(sym_trades)
    if all_trade_list:
        journal_path = write_trades_to_journal(all_trade_list, _JOURNAL_DIR, risk_cfg["starting_equity"])
        print(f"[JOURNAL] {journal_path}")


def signals(symbols: list[str] | None = None) -> None:
    """Generate current VCE signals (paper mode)."""
    symbols = symbols or ALL_SYMBOLS
    strategy_cfg = load_strategy()

    for sym in symbols:
        proc_path = _DATA_PROC / f"{sym}_processed.csv"
        if not proc_path.exists():
            print(f"[SKIP] No processed data for {sym}. Run ingest first.")
            continue

        df = _load_processed(proc_path)
        daily_df = _resample_to_daily(df)
        signal_df = compute_signals_for_instrument(df, daily_df, strategy_cfg)

        # Show latest signals
        last = signal_df.iloc[-1] if len(signal_df) > 0 else None
        if last is not None:
            compression = "YES" if last.get("compression", False) else "no"
            long_s = "LONG" if last.get("long_signal", False) else "-"
            short_s = "SHORT" if last.get("short_signal", False) else "-"
            print(f"[SIGNAL] {sym} | Compression: {compression} | "
                  f"Long: {long_s} | Short: {short_s} | "
                  f"ATR: {last.get('atr', 0):.4f} | Close: {last['close']:.4f}")
        else:
            print(f"[SIGNAL] {sym} | No data")


def report() -> None:
    """Generate weekly summary from journal data."""
    from modules.aac_puprime_vce.src.journal.review import load_journal, review_summary

    df = load_journal(_JOURNAL_DIR)
    if df.empty:
        print("[REPORT] No journal data found. Run a backtest first.")
        return

    summary = review_summary(df)
    print("\n=== VCE Campaign Review ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def _load_processed(path: Path):
    """Load processed CSV with datetime index."""
    import pandas as pd
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

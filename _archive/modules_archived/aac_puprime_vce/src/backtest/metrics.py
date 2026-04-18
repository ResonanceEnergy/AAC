"""Performance metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades: list, equity_curve: pd.Series) -> dict:
    """Compute standard trading metrics from trade records and equity curve.

    Parameters
    ----------
    trades : list of TradeRecord dataclass instances.
    equity_curve : Series of equity values indexed by datetime.

    Returns
    -------
    dict of metric_name → value.
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_r": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_abs": 0.0,
            "total_pnl": 0.0,
            "return_pct": 0.0,
            "avg_bars_held": 0.0,
        }

    pnls = [t.pnl for t in trades]
    r_multiples = [t.r_multiple for t in trades]
    bars_held = [t.bars_held for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total = len(pnls)
    win_count = len(wins)
    win_rate = win_count / total if total else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = sum(pnls)
    expectancy = total_pnl / total if total else 0.0
    avg_r = np.mean(r_multiples) if r_multiples else 0.0

    # Drawdown from equity curve
    peak = equity_curve.expanding().max()
    dd = equity_curve - peak
    max_dd_abs = float(dd.min())
    max_dd_pct = float((dd / peak).min()) if (peak > 0).all() else 0.0

    start_eq = equity_curve.iloc[0] if len(equity_curve) > 0 else 1.0
    return_pct = (equity_curve.iloc[-1] - start_eq) / start_eq if start_eq > 0 else 0.0

    return {
        "total_trades": total,
        "wins": win_count,
        "losses": total - win_count,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 2),
        "avg_r": round(avg_r, 4),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(return_pct, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "max_drawdown_abs": round(max_dd_abs, 2),
        "avg_bars_held": round(np.mean(bars_held), 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }

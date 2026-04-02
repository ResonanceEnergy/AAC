"""Event-driven backtest engine for VCE strategy.

Supports long and short trades, ATR-based stops, R-multiple targets,
trailing stops, time stops, and per-trade cost modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..strategy.risk_controls import (
    RiskState,
    can_open_position,
    make_initial_state,
)
from ..strategy.sizing import size_by_risk
from .fills import apply_costs


@dataclass
class TradeRecord:
    """TradeRecord class."""
    symbol: str
    side: str  # "long" or "short"
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    stop: float
    target: float
    size: float
    pnl: float
    r_multiple: float
    exit_reason: str  # "tp", "stop", "trail", "time_stop", "campaign_end"
    bars_held: int


def backtest_instrument(
    df: pd.DataFrame,
    symbol: str,
    costs: dict,
    risk_cfg: dict,
    strategy_cfg: dict,
    risk_state: RiskState | None = None,
) -> tuple[list[TradeRecord], RiskState, pd.Series]:
    """Run the VCE backtest on a single instrument's signal DataFrame.

    Parameters
    ----------
    df : DataFrame from compute_signals_for_instrument (has long_signal,
         short_signal, atr, stop_long, stop_short columns).
    symbol : instrument name (for trade records).
    costs : dict with spread_points, slippage_points, commission_per_lot.
    risk_cfg : risk config dict.
    strategy_cfg : strategy config dict.
    risk_state : optional shared RiskState (for multi-instrument campaigns).

    Returns
    -------
    (trades, risk_state, equity_curve)
    """
    if risk_state is None:
        risk_state = make_initial_state(risk_cfg["starting_equity"])

    exit_cfg = strategy_cfg["exits"]
    stop_mult = exit_cfg["stop_atr_mult"]
    tp_r = exit_cfg["takeprofit_r"]
    trail_after_r = exit_cfg["trail_after_r"]
    trail_atr_mult = exit_cfg["trail_atr_mult"]
    time_stop_bars = exit_cfg["time_stop_bars"]

    risk_pct = risk_cfg["risk_per_trade_pct"]
    max_open = risk_cfg["max_open_positions"]
    max_dd_daily = risk_cfg["max_daily_drawdown_pct"]
    max_dd_campaign = risk_cfg["max_campaign_drawdown_pct"]
    ks_consec = risk_cfg["kill_switch"]["consecutive_losses"]
    ks_cooldown = risk_cfg["kill_switch"]["cooldown_hours"]

    point_value = 1.0  # simplified; extend via instruments.yaml

    trades: list[TradeRecord] = []
    equity_curve: list[float] = []

    in_pos = False
    side = ""
    entry_price = stop_price = target_price = trailing_stop = 0.0
    pos_size = 0.0
    entry_time = None
    bars_in_trade = 0
    risk_per_unit = 0.0
    last_date = None

    for i in range(len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr_val = row["atr"] if pd.notna(row["atr"]) else 0.0

        # Day-boundary reset
        current_date = ts.date() if hasattr(ts, "date") else ts
        if last_date is not None and current_date != last_date:
            risk_state.new_day()
        last_date = current_date

        # --- Position management ---
        if in_pos:
            bars_in_trade += 1
            exit_price = None
            exit_reason = None

            if side == "long":
                # Stop hit?
                if low <= stop_price:
                    exit_price = apply_costs(
                        stop_price, "sell", costs["spread_points"], costs["slippage_points"]
                    )
                    exit_reason = "stop"
                # Take profit hit?
                elif high >= target_price:
                    exit_price = apply_costs(
                        target_price, "sell", costs["spread_points"], costs["slippage_points"]
                    )
                    exit_reason = "tp"
                # Time stop?
                elif bars_in_trade >= time_stop_bars:
                    exit_price = apply_costs(
                        close, "sell", costs["spread_points"], costs["slippage_points"]
                    )
                    exit_reason = "time_stop"
                else:
                    # Trailing stop update
                    unrealized_r = (close - entry_price) / risk_per_unit if risk_per_unit > 0 else 0
                    if unrealized_r >= trail_after_r and atr_val > 0:
                        new_trail = close - trail_atr_mult * atr_val
                        trailing_stop = max(trailing_stop, new_trail)
                        if low <= trailing_stop:
                            exit_price = apply_costs(
                                trailing_stop, "sell",
                                costs["spread_points"], costs["slippage_points"],
                            )
                            exit_reason = "trail"

            elif side == "short":
                if high >= stop_price:
                    exit_price = apply_costs(
                        stop_price, "buy", costs["spread_points"], costs["slippage_points"]
                    )
                    exit_reason = "stop"
                elif low <= target_price:
                    exit_price = apply_costs(
                        target_price, "buy", costs["spread_points"], costs["slippage_points"]
                    )
                    exit_reason = "tp"
                elif bars_in_trade >= time_stop_bars:
                    exit_price = apply_costs(
                        close, "buy", costs["spread_points"], costs["slippage_points"]
                    )
                    exit_reason = "time_stop"
                else:
                    unrealized_r = (entry_price - close) / risk_per_unit if risk_per_unit > 0 else 0
                    if unrealized_r >= trail_after_r and atr_val > 0:
                        new_trail = close + trail_atr_mult * atr_val
                        if trailing_stop == 0:
                            trailing_stop = new_trail
                        else:
                            trailing_stop = min(trailing_stop, new_trail)
                        if high >= trailing_stop:
                            exit_price = apply_costs(
                                trailing_stop, "buy",
                                costs["spread_points"], costs["slippage_points"],
                            )
                            exit_reason = "trail"

            if exit_price is not None and exit_reason is not None:
                if side == "long":
                    pnl = (exit_price - entry_price) * pos_size * point_value
                else:
                    pnl = (entry_price - exit_price) * pos_size * point_value

                r_mult = pnl / (risk_pct * risk_state.equity) if risk_state.equity > 0 else 0.0

                trades.append(TradeRecord(
                    symbol=symbol,
                    side=side,
                    entry_time=entry_time,
                    exit_time=ts,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop=stop_price,
                    target=target_price,
                    size=pos_size,
                    pnl=pnl,
                    r_multiple=r_mult,
                    exit_reason=exit_reason,
                    bars_held=bars_in_trade,
                ))
                risk_state.record_trade(pnl)
                risk_state.open_positions -= 1
                in_pos = False

        # --- Entry logic ---
        if not in_pos:
            now_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else None
            allowed, reason = can_open_position(
                risk_state, max_open, max_dd_daily, max_dd_campaign,
                ks_consec, ks_cooldown, now_dt,
            )

            if allowed and pd.notna(atr_val) and atr_val > 0:
                if row.get("long_signal", False):
                    entry_price = apply_costs(
                        close, "buy", costs["spread_points"], costs["slippage_points"]
                    )
                    stop_price = entry_price - stop_mult * atr_val
                    risk_per_unit = entry_price - stop_price
                    target_price = entry_price + tp_r * risk_per_unit
                    pos_size = size_by_risk(
                        risk_state.equity, risk_pct, entry_price, stop_price, point_value
                    )
                    side = "long"
                    in_pos = True
                    entry_time = ts
                    bars_in_trade = 0
                    trailing_stop = 0.0
                    risk_state.open_positions += 1

                elif row.get("short_signal", False):
                    entry_price = apply_costs(
                        close, "sell", costs["spread_points"], costs["slippage_points"]
                    )
                    stop_price = entry_price + stop_mult * atr_val
                    risk_per_unit = stop_price - entry_price
                    target_price = entry_price - tp_r * risk_per_unit
                    pos_size = size_by_risk(
                        risk_state.equity, risk_pct, entry_price, stop_price, point_value
                    )
                    side = "short"
                    in_pos = True
                    entry_time = ts
                    bars_in_trade = 0
                    trailing_stop = 0.0
                    risk_state.open_positions += 1

        equity_curve.append(risk_state.equity)

    # Close open position at end of campaign
    if in_pos:
        final_close = df["close"].iloc[-1]
        if side == "long":
            exit_price = apply_costs(
                final_close, "sell", costs["spread_points"], costs["slippage_points"]
            )
            pnl = (exit_price - entry_price) * pos_size * point_value
        else:
            exit_price = apply_costs(
                final_close, "buy", costs["spread_points"], costs["slippage_points"]
            )
            pnl = (entry_price - exit_price) * pos_size * point_value

        r_mult = pnl / (risk_pct * risk_state.equity) if risk_state.equity > 0 else 0.0
        trades.append(TradeRecord(
            symbol=symbol, side=side,
            entry_time=entry_time, exit_time=df.index[-1],
            entry_price=entry_price, exit_price=exit_price,
            stop=stop_price, target=target_price,
            size=pos_size, pnl=pnl, r_multiple=r_mult,
            exit_reason="campaign_end", bars_held=bars_in_trade,
        ))
        risk_state.record_trade(pnl)
        risk_state.open_positions -= 1

    eq_series = pd.Series(equity_curve, index=df.index[: len(equity_curve)], name="equity")
    return trades, risk_state, eq_series

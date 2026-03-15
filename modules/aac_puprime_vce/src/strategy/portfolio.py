"""Multi-instrument portfolio-level logic."""

from __future__ import annotations

import pandas as pd
from ..features import indicators as ind
from ..features.regimes import compression_flag
from .filters import daily_trend_bias
from .vce import vce_signals


def compute_signals_for_instrument(
    signal_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    strategy_cfg: dict,
) -> pd.DataFrame:
    """Run the full VCE signal pipeline for one instrument.

    Returns a copy of signal_df with extra columns:
      atr, bb_width, compression, allow_long, allow_short,
      long_signal, short_signal, stop_long, stop_short
    """
    cfg_ind = strategy_cfg["indicators"]
    cfg_comp = strategy_cfg["compression"]
    cfg_brk = strategy_cfg["breakout"]
    cfg_tf = strategy_cfg["trend_filter"]
    cfg_exit = strategy_cfg["exits"]

    out = signal_df.copy()

    # Indicators
    out["atr"] = ind.atr(out, n=cfg_ind["atr_period"])
    _, _, _, bbw = ind.bollinger(out, n=cfg_ind["bb_period"], k=cfg_ind["bb_k"])
    out["bb_width"] = bbw

    # Compression regime
    out["compression"] = compression_flag(
        out["atr"], out["bb_width"],
        lookback=cfg_comp["lookback"],
        threshold=cfg_comp["threshold_pct"],
    )

    # Daily trend filter
    allow_long, allow_short = daily_trend_bias(daily_df, ma_n=cfg_tf["ma_period"])

    # VCE signals
    long_sig, short_sig = vce_signals(
        out, out["compression"],
        allow_long, allow_short,
        range_lookback=cfg_brk["range_lookback"],
    )
    out["long_signal"] = long_sig
    out["short_signal"] = short_sig

    # Pre-compute stop levels
    out["stop_long"] = out["close"] - cfg_exit["stop_atr_mult"] * out["atr"]
    out["stop_short"] = out["close"] + cfg_exit["stop_atr_mult"] * out["atr"]

    return out

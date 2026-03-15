"""Regime detection — compression / expansion states."""

import pandas as pd


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling percentile rank (0–1) of each value within its window."""
    def _pct_rank(x: pd.Series) -> float:
        s = pd.Series(x)
        return float(s.rank(pct=True).iloc[-1])

    return series.rolling(window, min_periods=window).apply(_pct_rank, raw=False)


def compression_flag(
    atr_s: pd.Series,
    bbw_s: pd.Series,
    lookback: int = 100,
    threshold: float = 0.20,
) -> pd.Series:
    """Return True where both ATR and BB-width are in the bottom *threshold* percentile.

    This indicates a volatility compression regime.
    """
    atr_pct = rolling_percentile_rank(atr_s, lookback)
    bbw_pct = rolling_percentile_rank(bbw_s, lookback)
    return (atr_pct <= threshold) & (bbw_pct <= threshold)

"""Trend and direction filters."""

import pandas as pd

from ..features.indicators import moving_average


def daily_trend_bias(
    daily_df: pd.DataFrame, ma_n: int = 50
) -> tuple[pd.Series, pd.Series]:
    """Determine bullish / bearish bias from daily close vs MA.

    Returns (allow_long, allow_short) boolean Series indexed by daily date.
    """
    ma = moving_average(daily_df["close"], ma_n)
    allow_long = daily_df["close"] > ma
    allow_short = daily_df["close"] < ma
    return allow_long, allow_short

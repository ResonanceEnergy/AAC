"""VCE signal generation — breakout from compression with trend filter."""

import pandas as pd

from ..features.indicators import rolling_high, rolling_low


def vce_signals(
    df: pd.DataFrame,
    compression: pd.Series,
    allow_long: pd.Series,
    allow_short: pd.Series,
    range_lookback: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Generate long / short VCE signals.

    Parameters
    ----------
    df : DataFrame with OHLCV indexed by datetime (signal timeframe).
    compression : boolean Series — True when volatility is compressed.
    allow_long / allow_short : boolean Series from daily trend filter,
        reindexed (forward-filled) to match df's index.
    range_lookback : bars for the breakout range high/low.

    Returns
    -------
    (long_signal, short_signal) boolean Series.
    """
    hh = rolling_high(df["high"], range_lookback)
    ll = rolling_low(df["low"], range_lookback)

    # Forward-fill daily bias onto intraday index
    long_ok = allow_long.reindex(df.index, method="ffill").fillna(False)
    short_ok = allow_short.reindex(df.index, method="ffill").fillna(False)

    long_sig = compression & long_ok & (df["close"] > hh)
    short_sig = compression & short_ok & (df["close"] < ll)

    return long_sig.fillna(False), short_sig.fillna(False)

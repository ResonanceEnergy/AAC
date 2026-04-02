"""Technical indicators for VCE strategy."""

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def bollinger(
    df: pd.DataFrame, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.  Returns (ma, upper, lower, width)."""
    ma = df["close"].rolling(n, min_periods=n).mean()
    sd = df["close"].rolling(n, min_periods=n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / ma
    return ma, upper, lower, width


def moving_average(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(n, min_periods=n).mean()


def rolling_high(series: pd.Series, n: int) -> pd.Series:
    """Rolling maximum (shifted by 1 to avoid look-ahead)."""
    return series.rolling(n, min_periods=n).max().shift(1)


def rolling_low(series: pd.Series, n: int) -> pd.Series:
    """Rolling minimum (shifted by 1 to avoid look-ahead)."""
    return series.rolling(n, min_periods=n).min().shift(1)

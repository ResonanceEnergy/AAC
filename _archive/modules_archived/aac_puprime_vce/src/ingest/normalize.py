"""Normalize and validate ingested candle data."""

from pathlib import Path

import pandas as pd


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return a list of validation warnings (empty = clean)."""
    warnings: list[str] = []

    if df.empty:
        warnings.append("DataFrame is empty")
        return warnings

    # High must be >= Open, Close, Low
    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    if bad_high:
        warnings.append(f"{bad_high} bars where high < max(open, close, low)")

    # Low must be <= Open, Close, High
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    if bad_low:
        warnings.append(f"{bad_low} bars where low > min(open, close, high)")

    # Duplicated timestamps
    dupes = df.index.duplicated().sum()
    if dupes:
        warnings.append(f"{dupes} duplicate timestamps")

    # Non-monotonic
    if not df.index.is_monotonic_increasing:
        warnings.append("Index is not monotonically increasing")

    return warnings


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up a raw OHLCV DataFrame (deduplicate, sort, fill gaps)."""
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df


def save_processed(df: pd.DataFrame, out_dir: str | Path, symbol: str) -> Path:
    """Save normalized data to CSV in the processed directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_processed.csv"
    df.to_csv(out_path)
    return out_path

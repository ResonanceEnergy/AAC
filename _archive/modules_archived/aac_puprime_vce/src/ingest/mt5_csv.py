"""Parse MT5 / PU Prime exported CSV files into standardized DataFrames."""

from pathlib import Path

import pandas as pd

# MT5 typically exports: Date, Time, Open, High, Low, Close, Volume
# Some exports combine Date+Time into a single column.

_REQUIRED_COLS = {"open", "high", "low", "close"}


def read_mt5_csv(path: str | Path) -> pd.DataFrame:
    """Read an MT5-exported CSV and return a datetime-indexed OHLCV DataFrame.

    Handles common MT5 export layouts:
      - Separate Date / Time columns
      - Combined datetime column
      - Tab or comma delimited
    """
    path = Path(path)
    # Detect delimiter
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Build datetime index
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    else:
        # Try first column as datetime
        df["datetime"] = pd.to_datetime(df.iloc[:, 0])

    df = df.set_index("datetime").sort_index()

    # Normalize column names
    rename_map = {}
    for col in df.columns:
        for target in ("open", "high", "low", "close", "volume", "tick_volume"):
            if target in col:
                rename_map[col] = target
                break
    df = df.rename(columns=rename_map)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after parsing: {missing}")

    # Ensure numeric
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 0

    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])

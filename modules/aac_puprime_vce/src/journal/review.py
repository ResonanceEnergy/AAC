"""Journal review — post-campaign analysis from trade journal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_journal(journal_dir: str | Path) -> pd.DataFrame:
    """Load the trades.csv journal into a DataFrame."""
    path = Path(journal_dir) / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])


def review_summary(df: pd.DataFrame) -> dict:
    """Generate a quick review summary from journal data."""
    if df.empty:
        return {"status": "no_trades"}

    return {
        "total_trades": len(df),
        "by_symbol": df.groupby("symbol")["pnl"].agg(["count", "sum", "mean"]).to_dict(),
        "by_exit_reason": df["exit_reason"].value_counts().to_dict(),
        "by_side": df["side"].value_counts().to_dict(),
        "avg_r": round(df["r_multiple"].mean(), 4),
        "best_r": round(df["r_multiple"].max(), 4),
        "worst_r": round(df["r_multiple"].min(), 4),
        "avg_bars_held": round(df["bars_held"].mean(), 1),
    }

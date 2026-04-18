"""Write trade records to journal CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from .schema import JOURNAL_COLUMNS


def write_trades_to_journal(
    trades: list,
    journal_dir: str | Path,
    starting_equity: float,
) -> Path:
    """Append trade records to journal/trades.csv.

    Parameters
    ----------
    trades : list of TradeRecord dataclass instances.
    journal_dir : directory for journal files.
    starting_equity : campaign starting equity for equity tracking.
    """
    journal_dir = Path(journal_dir)
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_path = journal_dir / "trades.csv"

    file_exists = journal_path.exists()
    equity = starting_equity

    with open(journal_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLUMNS)
        if not file_exists:
            writer.writeheader()

        for i, t in enumerate(trades):
            eq_before = equity
            equity += t.pnl
            writer.writerow({
                "trade_id": i + 1,
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 6),
                "exit_price": round(t.exit_price, 6),
                "stop": round(t.stop, 6),
                "target": round(t.target, 6),
                "size": round(t.size, 6),
                "pnl": round(t.pnl, 2),
                "r_multiple": round(t.r_multiple, 4),
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "equity_before": round(eq_before, 2),
                "equity_after": round(equity, 2),
                "rule_violations": "",
                "notes": "",
            })

    return journal_path

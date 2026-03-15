"""Report generation — markdown summaries and equity curve charts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def generate_markdown_report(
    metrics_by_symbol: dict[str, dict],
    trades_by_symbol: dict[str, list],
    campaign_metrics: dict | None = None,
) -> str:
    """Build a Markdown report string from backtest results."""
    lines = [
        "# VCE Backtest Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]

    if campaign_metrics:
        lines.append("## Campaign Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in campaign_metrics.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    for symbol, m in metrics_by_symbol.items():
        lines.append(f"## {symbol}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in m.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # Trade list
        symbol_trades = trades_by_symbol.get(symbol, [])
        if symbol_trades:
            lines.append("### Trades")
            lines.append("")
            lines.append("| # | Side | Entry | Exit | PnL | R | Reason | Bars |")
            lines.append("|---|------|-------|------|-----|---|--------|------|")
            for i, t in enumerate(symbol_trades, 1):
                lines.append(
                    f"| {i} | {t.side} | {t.entry_price:.4f} | "
                    f"{t.exit_price:.4f} | {t.pnl:.2f} | {t.r_multiple:.2f} | "
                    f"{t.exit_reason} | {t.bars_held} |"
                )
            lines.append("")

    return "\n".join(lines)


def save_report(
    report_md: str,
    metrics: dict,
    out_dir: str | Path,
) -> Path:
    """Write report markdown and metrics JSON to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"backtest_report_{ts}.md"
    md_path.write_text(report_md, encoding="utf-8")

    json_path = out_dir / f"metrics_{ts}.json"
    json_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    return md_path


def save_equity_curve_csv(eq: pd.Series, out_dir: str | Path, symbol: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"equity_{symbol}.csv"
    eq.to_csv(path, header=True)
    return path

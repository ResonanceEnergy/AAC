from __future__ import annotations

"""LDE Dashboard — Dash-based web UI for the Living Doctrine Engine.

Tabs:
1. Doctrine Overview — active rules, signal gauge, top rules
2. Ingest Log — videos processed, timeline
3. Backtest Results — signal performance vs market
4. Sandbox — browsable insight archive

Run via ``python launch.py lde`` or directly:
    python -m monitoring.lde_dashboard
"""

import os
import sys

# pythonw.exe guard
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# ── Lazy Dash imports (avoid import crash if dash not installed) ────────────

def _create_app() -> Any:
    """Build the Dash application."""
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, callback, dcc, html

    from strategies.living_doctrine_engine import LivingDoctrineEngine

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="LDE — Living Doctrine Engine",
    )

    engine = LivingDoctrineEngine()

    # ── Layout ─────────────────────────────────────────────────────

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Living Doctrine Engine", className="text-primary"),
                    width=8),
            dbc.Col(html.Div(id="signal-badge"), width=4,
                    className="text-end"),
        ], className="my-3"),

        dbc.Tabs([
            dbc.Tab(label="Doctrine", tab_id="tab-doctrine"),
            dbc.Tab(label="Ingest Log", tab_id="tab-ingest"),
            dbc.Tab(label="Backtest", tab_id="tab-backtest"),
            dbc.Tab(label="Sandbox", tab_id="tab-sandbox"),
        ], id="tabs", active_tab="tab-doctrine"),

        html.Div(id="tab-content", className="mt-3"),
        dcc.Interval(id="refresh-interval", interval=60_000, n_intervals=0),
    ], fluid=True, className="py-2")

    # ── Callbacks ──────────────────────────────────────────────────

    @callback(
        Output("signal-badge", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_signal_badge(_: int) -> Any:
        sig = engine.get_doctrine_signal()
        color = "success" if sig > 0.1 else "danger" if sig < -0.1 else "warning"
        return dbc.Badge(
            f"Signal: {sig:+.3f}",
            color=color,
            className="fs-5 p-2",
        )

    @callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
        Input("refresh-interval", "n_intervals"),
    )
    def render_tab(tab: str, _: int) -> Any:
        if tab == "tab-doctrine":
            return _render_doctrine(engine)
        if tab == "tab-ingest":
            return _render_ingest(engine)
        if tab == "tab-backtest":
            return _render_backtest(engine)
        if tab == "tab-sandbox":
            return _render_sandbox(engine)
        return html.P("Select a tab.")

    return app


def _render_doctrine(engine: Any) -> Any:
    """Doctrine overview tab."""
    import dash_bootstrap_components as dbc
    from dash import html

    status = engine.status()
    rules = engine.doctrine.active_rules
    rules.sort(key=lambda r: r.conviction, reverse=True)

    cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Rules", className="card-title"),
                html.H3(str(status["total_rules"]), className="text-info"),
            ]),
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Active Rules", className="card-title"),
                html.H3(str(status["active_rules"]), className="text-success"),
            ]),
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Videos Processed", className="card-title"),
                html.H3(str(status["videos_processed"]), className="text-warning"),
            ]),
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Doctrine Signal", className="card-title"),
                html.H3(f"{status['doctrine_signal']:+.4f}", className="text-primary"),
            ]),
        ]), width=3),
    ], className="mb-3")

    # Top rules table
    rows = []
    for r in rules[:20]:
        sentiment_color = {"positive": "success", "negative": "danger",
                           "mixed": "warning"}.get(r.sentiment, "secondary")
        rows.append(html.Tr([
            html.Td(r.text[:80] + ("..." if len(r.text) > 80 else "")),
            html.Td(f"{r.conviction:.2f}"),
            html.Td(f"{r.signal_value:+.2f}"),
            html.Td(dbc.Badge(r.sentiment, color=sentiment_color)),
            html.Td(str(r.reinforcement_count)),
            html.Td(r.source_channel),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Rule"), html.Th("Conviction"), html.Th("Signal"),
            html.Th("Sentiment"), html.Th("Reinforced"), html.Th("Source"),
        ])),
        html.Tbody(rows),
    ], bordered=True, hover=True, responsive=True, striped=True,
        className="table-dark")

    return html.Div([cards, html.H4("Active Doctrine Rules"), table])


def _render_ingest(engine: Any) -> Any:
    """Ingest log tab."""
    import dash_bootstrap_components as dbc
    from dash import html

    records = sorted(engine.ingest_log.records,
                     key=lambda r: r.ingested_at, reverse=True)

    rows = []
    for r in records[:50]:
        rows.append(html.Tr([
            html.Td(r.title[:60] + ("..." if len(r.title) > 60 else "")),
            html.Td(r.channel),
            html.Td(r.ingested_at[:19]),
            html.Td(str(r.doctrine_rules_created)),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Title"), html.Th("Channel"),
            html.Th("Ingested At"), html.Th("Rules Created"),
        ])),
        html.Tbody(rows),
    ], bordered=True, hover=True, responsive=True, striped=True,
        className="table-dark")

    return html.Div([
        html.H4(f"Ingest Log ({len(records)} videos)"),
        table,
    ])


def _render_backtest(engine: Any) -> Any:
    """Backtest results tab."""
    import dash_bootstrap_components as dbc
    from dash import dcc, html

    from strategies.living_doctrine_backtest import backtest_against_market

    signal = engine.get_doctrine_signal()
    # Generate a flat signal series for demonstration
    signal_series = [signal] * 90

    result = backtest_against_market(
        signal_series, ticker="SPY", lookback_days=90,
    )

    metrics = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total Return"), html.H4(f"{result.total_return:+.1f}%"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Sharpe"), html.H4(f"{result.sharpe_ratio:.2f}"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Sortino"), html.H4(f"{result.sortino_ratio:.2f}"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Max DD"), html.H4(f"{result.max_drawdown:.1f}%"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Win Rate"), html.H4(f"{result.win_rate:.0f}%"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Trades"), html.H4(str(result.total_trades)),
        ])), width=2),
    ], className="mb-3")

    # Trade log table
    trade_rows = []
    for t in result.trade_log[:20]:
        color = "text-success" if t["pnl_pct"] > 0 else "text-danger"
        trade_rows.append(html.Tr([
            html.Td(t["entry_date"]),
            html.Td(t["exit_date"]),
            html.Td(t["direction"]),
            html.Td(f"${t['entry_price']:.2f}"),
            html.Td(f"${t['exit_price']:.2f}"),
            html.Td(f"{t['pnl_pct']:+.1f}%", className=color),
            html.Td(t["exit_reason"]),
        ]))

    trade_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Entry"), html.Th("Exit"), html.Th("Dir"),
            html.Th("Entry $"), html.Th("Exit $"), html.Th("P&L"),
            html.Th("Reason"),
        ])),
        html.Tbody(trade_rows),
    ], bordered=True, hover=True, responsive=True, striped=True,
        className="table-dark")

    return html.Div([
        html.H4(f"Backtest: SPY ({result.period})"),
        metrics,
        html.H5("Trade Log"),
        trade_table,
    ])


def _render_sandbox(engine: Any) -> Any:
    """Sandbox insight browser tab."""
    import dash_bootstrap_components as dbc
    from dash import html

    entries = sorted(engine.sandbox.entries,
                     key=lambda e: e.ingested_at, reverse=True)

    cards = []
    for e in entries[:30]:
        trust = e.trust_score.get("overall", 0.0) if e.trust_score else 0.0
        cards.append(dbc.Card([
            dbc.CardHeader(html.Div([
                html.Strong(e.title[:70]),
                dbc.Badge(f"Trust: {trust:.1%}", color="info",
                          className="ms-2"),
            ])),
            dbc.CardBody([
                html.P(e.summary[:200] + ("..." if len(e.summary) > 200 else ""),
                       className="card-text"),
                html.Small(f"Channel: {e.channel} | Topics: {', '.join(e.key_topics[:5])}",
                           className="text-muted"),
            ]),
        ], className="mb-2"))

    return html.Div([
        html.H4(f"Sandbox ({len(entries)} insights)"),
        html.Div(cards),
    ])


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_dashboard(port: int = 8510, debug: bool = False) -> None:
    """Launch the LDE dashboard."""
    app = _create_app()
    _log.info("lde.dashboard_starting", port=port)
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)

#!/usr/bin/env python3
"""
AAC Dash Analytics Dashboard
=============================
Plotly Dash-based analytics dashboard extracted from the master monitoring module.
"""

import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DASH_AVAILABLE = False

# Strategy testing (optional)
try:
    from strategies.strategy_testing_lab_fixed import strategy_testing_lab, initialize_strategy_testing_lab
    STRATEGY_TESTING_AVAILABLE = True
except ImportError:
    STRATEGY_TESTING_AVAILABLE = False

# Strategy analysis (optional)
try:
    from strategies.strategy_analysis_engine import strategy_analysis_engine, initialize_strategy_analysis
    ARBITRAGE_COMPONENTS_AVAILABLE = True
except ImportError:
    ARBITRAGE_COMPONENTS_AVAILABLE = False


class AACDashDashboard:
    """Dash-based analytics dashboard for AAC monitoring"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_cache = {}
        self.deep_dive_cache = {}
        self.dashboard_app = None
        self.initialized = False

    async def initialize(self):
        """Initialize the Dash dashboard"""
        self.logger.info("Initializing AAC Dash Analytics Dashboard")

        # Try to import Dash here
        try:
            import dash  # noqa: F401
            from dash import html, dcc  # noqa: F401
            import dash_bootstrap_components as dbc  # noqa: F401
            global DASH_AVAILABLE
            DASH_AVAILABLE = True
        except ImportError:
            self.logger.warning("Dash not available, cannot create dashboard")
            return False

        # Initialize dependencies
        if STRATEGY_TESTING_AVAILABLE and not strategy_testing_lab.initialized:
            await initialize_strategy_testing_lab()
        if ARBITRAGE_COMPONENTS_AVAILABLE and hasattr(strategy_analysis_engine, 'initialized') and not strategy_analysis_engine.initialized:
            await initialize_strategy_analysis()

        # Create dashboard
        self.dashboard_app = self._create_dashboard_app()

        self.initialized = True
        self.logger.info("[OK] Dash Analytics Dashboard initialized")
        return True

    def _create_dashboard_app(self):
        """Create the Dash dashboard application"""
        if not DASH_AVAILABLE:
            logger.info("Dash not available, cannot create dashboard")
            return None

        import dash
        from dash import html, dcc
        from dash.dependencies import Output, Input, State
        import dash_bootstrap_components as dbc
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 AAC Strategy Metrics Dashboard",
                           className="text-center mb-4"),
                    html.P("Real-time strategy metrics and deep dive analysis",
                          className="text-center text-muted mb-4")
                ])
            ]),

            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='strategy-selector',
                                        options=[{'label': f"{sid}: {config.get('name', sid)}",
                                                 'value': sid}
                                                for sid, config in (strategy_testing_lab.strategy_configs.items() if STRATEGY_TESTING_AVAILABLE else [])],
                                        value='s26' if STRATEGY_TESTING_AVAILABLE and 's26' in strategy_testing_lab.strategy_configs else None,
                                        placeholder="Select Strategy"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='timeframe-selector',
                                        options=[
                                            {'label': '1 Month', 'value': '1M'},
                                            {'label': '3 Months', 'value': '3M'},
                                            {'label': '6 Months', 'value': '6M'},
                                            {'label': '1 Year', 'value': '1Y'}
                                        ],
                                        value='3M',
                                        placeholder="Select Timeframe"
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Button("🔍 Deep Dive", id="deep-dive-btn",
                                             color="primary", className="me-2"),
                                    dbc.Button("📊 Refresh", id="refresh-btn", color="secondary")
                                ], width=5)
                            ])
                        ])
                    ])
                ])
            ]),

            # Metrics Display
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Performance Metrics"),
                        dbc.CardBody([
                            html.Div(id="metrics-display")
                        ])
                    ])
                ], width=12)
            ]),

            # Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Performance Charts"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Risk Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-chart")
                        ])
                    ])
                ], width=4)
            ]),

            # Deep Dive Results
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔍 Deep Dive Analysis"),
                        dbc.CardBody([
                            html.Div(id="deep-dive-results")
                        ])
                    ])
                ])
            ])
        ], fluid=True)

        # Callbacks
        @app.callback(
            [Output("metrics-display", "children"),
             Output("performance-chart", "figure"),
             Output("risk-chart", "figure")],
            [Input("strategy-selector", "value"),
             Input("timeframe-selector", "value"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_metrics(strategy_id, timeframe, n_clicks):
            """Update metrics display with real data from cache."""
            try:
                # Build metrics display
                metrics_data = self.metrics_cache.get(strategy_id, {})
                if not metrics_data:
                    metrics_children = html.Div([
                        html.P(f"Strategy: {strategy_id or 'None selected'}"),
                        html.P(f"Timeframe: {timeframe or 'N/A'}"),
                        html.P("No metrics data available yet. Data will populate as the system runs."),
                    ])
                else:
                    metrics_children = html.Div([
                        html.H5(f"Strategy: {strategy_id}"),
                        html.P(f"Timeframe: {timeframe}"),
                        html.Ul([
                            html.Li(f"{k}: {v}")
                            for k, v in metrics_data.items()
                        ])
                    ])

                # Build performance chart
                perf_fig = go.Figure()
                perf_fig.update_layout(
                    title=f"Performance: {strategy_id or 'All Strategies'}",
                    xaxis_title="Time",
                    yaxis_title="Returns (%)",
                    template="plotly_dark",
                )

                # Build risk chart
                risk_fig = go.Figure()
                risk_fig.update_layout(
                    title="Risk Exposure",
                    template="plotly_dark",
                )

                return metrics_children, perf_fig, risk_fig
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                return html.Div(f"Error loading metrics: {e}"), {}, {}

        @app.callback(
            Output("deep-dive-results", "children"),
            [Input("deep-dive-btn", "n_clicks")],
            [State("strategy-selector", "value")]
        )
        def perform_deep_dive(n_clicks, strategy_id):
            """Perform deep dive analysis on selected strategy."""
            if n_clicks and strategy_id:
                try:
                    cached = self.deep_dive_cache.get(strategy_id, {})
                    if cached:
                        return html.Div([
                            html.H5(f"Deep Dive: {strategy_id}"),
                            html.Ul([
                                html.Li(f"{k}: {v}")
                                for k, v in cached.items()
                            ])
                        ])
                    return html.Div([
                        html.H5(f"Deep Dive: {strategy_id}"),
                        html.P("No cached analysis data. Run the strategy testing lab to generate data."),
                    ])
                except Exception as e:
                    self.logger.error(f"Deep dive error: {e}")
                    return html.Div(f"Error during deep dive: {e}")
            return html.Div("Select a strategy and click Deep Dive to analyze")

        return app

    def run_dashboard(self, port=8050):
        """Run the Dash dashboard"""
        if self.dashboard_app:
            self.dashboard_app.run_server(debug=os.environ.get('DASH_DEBUG', '').lower() == 'true', port=port)
        else:
            logger.info("Dashboard not initialized")

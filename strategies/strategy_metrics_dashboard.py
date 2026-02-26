#!/usr/bin/env python3
"""
Strategy Metrics Dashboard & Deep Dive Analyzer
===============================================
Comprehensive metrics display and deep file analysis system for arbitrage strategies.
Provides real-time monitoring, deep dives, and actionable insights.
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy_testing_lab import strategy_testing_lab, initialize_strategy_testing_lab
from strategy_analysis_engine import strategy_analysis_engine, initialize_strategy_analysis


@dataclass
class DeepDiveResult:
    """Results from deep dive analysis"""
    strategy_id: str
    risk_level: str
    findings: Dict[str, Any]
    recommendations: List[str]


class MetricType(Enum):
    """Types of metrics to display"""
    PERFORMANCE = "performance"
    RISK = "risk"
    PREDICTIVE = "predictive"
    CORRELATION = "correlation"
    SENSITIVITY = "sensitivity"
    MARKET_REGIME = "market_regime"
    DEEP_DIVE = "deep_dive"


@dataclass
class MetricDisplay:
    """Metric display configuration"""
    name: str
    value: float
    unit: str
    format_string: str = ".2f"
    color: str = "primary"
    trend: Optional[str] = None  # "up", "down", "neutral"


@dataclass
class DeepDiveResult:
    """Deep dive analysis result"""
    file_path: str
    analysis_type: str
    findings: Dict[str, Any]
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high"
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyMetricsDashboard:
    """
    Comprehensive metrics dashboard with deep dive capabilities.
    Provides real-time monitoring and detailed analysis.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_cache: Dict[str, Any] = {}
        self.deep_dive_cache: Dict[str, DeepDiveResult] = {}
        self.dashboard_app = None
        self.initialized = False

    async def initialize(self):
        """Initialize the metrics dashboard"""
        self.logger.info("Initializing Strategy Metrics Dashboard")

        # Initialize dependencies
        if not strategy_testing_lab.initialized:
            await initialize_strategy_testing_lab()
        if not strategy_analysis_engine.initialized:
            await initialize_strategy_analysis()

        # Create dashboard
        self.dashboard_app = self._create_dashboard_app()

        self.initialized = True
        self.logger.info("[OK] Metrics Dashboard initialized")

    def _create_dashboard_app(self) -> dash.Dash:
        """Create the Dash dashboard application"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 Strategy Metrics Dashboard (ARB Currency)",
                           className="text-center mb-4"),
                    html.P("Real-time ARB simulation metrics and deep dive analysis",
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
                                        options=[{'label': f"{sid}: {config['name']}",
                                                 'value': sid}
                                                for sid, config in strategy_testing_lab.strategy_configs.items()],
                                        value='s26',
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
                                    dbc.Button("🔄 Refresh", id="refresh-btn",
                                             color="primary", className="me-2"),
                                    dbc.Button("🔍 Deep Dive", id="deep-dive-btn",
                                             color="info")
                                ], width=5)
                            ])
                        ])
                    ], className="mb-4")
                ])
            ]),

            # Key Metrics Row
            dbc.Row(id="key-metrics-row"),

            # Charts Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Chart"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-chart")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),

            # Deep Dive Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Deep Dive Analysis"),
                        dbc.CardBody([
                            html.Div(id="deep-dive-content")
                        ])
                    ], id="deep-dive-card", style={"display": "none"})
                ])
            ], className="mb-4"),

            # Recommendations Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("AI Recommendations"),
                        dbc.CardBody([
                            html.Div(id="recommendations-content")
                        ])
                    ])
                ])
            ])

        ], fluid=True)

        # Callbacks
        @app.callback(
            [Output("key-metrics-row", "children"),
             Output("performance-chart", "figure"),
             Output("risk-chart", "figure"),
             Output("recommendations-content", "children")],
            [Input("strategy-selector", "value"),
             Input("timeframe-selector", "value"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_dashboard(strategy_id, timeframe, n_clicks):
            if not strategy_id:
                return [], {}, {}, ""

            # Get metrics
            metrics = self._get_strategy_metrics(strategy_id, timeframe)

            # Create displays
            key_metrics = self._create_key_metrics_display(metrics)
            perf_chart = self._create_performance_chart(metrics)
            risk_chart = self._create_risk_chart(metrics)
            recommendations = self._create_recommendations_display(strategy_id, metrics)

            return key_metrics, perf_chart, risk_chart, recommendations

        @app.callback(
            [Output("deep-dive-card", "style"),
             Output("deep-dive-content", "children")],
            [Input("deep-dive-btn", "n_clicks")],
            [State("strategy-selector", "value")]
        )
        def show_deep_dive(n_clicks, strategy_id):
            if not n_clicks or not strategy_id:
                return {"display": "none"}, ""

            deep_dive = self._perform_deep_dive(strategy_id)
            content = self._create_deep_dive_display(deep_dive)

            return {"display": "block"}, content

        return app

    async def _get_strategy_metrics(self, strategy_id: str, timeframe: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a strategy"""
        # Run simulation if not cached
        cache_key = f"{strategy_id}_{timeframe}"
        if cache_key not in self.metrics_cache:
            sim_results = await strategy_testing_lab.run_strategy_simulation(
                strategy_id, timeframe, 1000
            )

            # Run analysis
            analysis_results = await strategy_analysis_engine.perform_comprehensive_analysis(
                [strategy_id], [
                    strategy_analysis_engine.AnalysisType.PERFORMANCE,
                    strategy_analysis_engine.AnalysisType.RISK,
                    strategy_analysis_engine.AnalysisType.PREDICTIVE
                ], timeframe
            )

            self.metrics_cache[cache_key] = {
                'simulation': sim_results,
                'analysis': analysis_results['strategy_analyses'][strategy_id]
            }

        return self.metrics_cache[cache_key]

    def _create_key_metrics_display(self, metrics: Dict) -> List:
        """Create key metrics display cards"""
        sim = metrics['simulation']
        analysis = metrics['analysis']

        perf = analysis.get('performance', {})
        risk = analysis.get('risk', {})
        pred = analysis.get('predictive', {})

        metrics_list = [
            MetricDisplay("Total Return", sim['total_return_pct'], "%", ".1f",
                         "success" if sim['total_return_pct'] > 10 else "warning"),
            MetricDisplay("Sharpe Ratio", sim['sharpe_ratio'], "", ".2f",
                         "success" if sim['sharpe_ratio'] > 1.5 else "warning"),
            MetricDisplay("Win Rate", perf.get('win_rate', 0) * 100, "%", ".1f",
                         "success" if perf.get('win_rate', 0) > 0.6 else "warning"),
            MetricDisplay("Max Drawdown", sim['max_drawdown'] * 100, "%", ".1f",
                         "danger" if sim['max_drawdown'] > 0.15 else "success"),
            MetricDisplay("Predicted Return", pred.get('predicted_return_pct', 0), "%", ".1f",
                         "info"),
            MetricDisplay("Risk Score", risk.get('risk_score', 0), "/100", ".1f",
                         "success" if risk.get('risk_score', 0) > 70 else "warning")
        ]

        cards = []
        for metric in metrics_list:
            color_class = f"border-{metric.color}"
            text_color = "text-white" if metric.color in ["success", "danger"] else "text-dark"

            card = dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{metric.value:{metric.format_string}}{metric.unit}",
                               className=f"card-title {text_color}"),
                        html.P(metric.name, className="card-text text-muted")
                    ])
                ], className=f"mb-3 {color_class}")
            ], width=2)

            cards.append(card)

        return cards

    def _create_performance_chart(self, metrics: Dict) -> go.Figure:
        """Create performance visualization chart"""
        sim = metrics['simulation']
        trade_history = sim.get('trade_history', [])

        if not trade_history:
            return go.Figure()

        # Create cumulative returns chart
        dates = [trade['date'] for trade in trade_history]
        balances = [trade['balance'] for trade in trade_history]
        returns = [trade['return'] for trade in trade_history]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Portfolio Balance (ARB)", "Daily Returns"))

        # Balance chart
        fig.add_trace(
            go.Scatter(x=dates, y=balances, mode='lines', name='Balance',
                      line=dict(color='green', width=2)),
            row=1, col=1
        )

        # Returns chart
        fig.add_trace(
            go.Scatter(x=dates, y=returns, mode='lines', name='Returns',
                      line=dict(color='blue')),
            row=2, col=1
        )

        fig.update_layout(height=400, showlegend=False)
        return fig

    def _create_risk_chart(self, metrics: Dict) -> go.Figure:
        """Create risk metrics visualization"""
        analysis = metrics['analysis']
        risk = analysis.get('risk', {})

        if not risk:
            return go.Figure()

        # Risk metrics gauge charts
        fig = go.Figure()

        # Sharpe Ratio gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk.get('sharpe_ratio', 0),
            title={'text': "Sharpe Ratio"},
            gauge={'axis': {'range': [0, 3]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 1], 'color': "red"},
                       {'range': [1, 2], 'color': "yellow"},
                       {'range': [2, 3], 'color': "green"}
                   ]}
        ))

        fig.update_layout(height=300)
        return fig

    def _create_recommendations_display(self, strategy_id: str, metrics: Dict) -> html.Div:
        """Create recommendations display"""
        analysis = metrics['analysis']
        recommendations = analysis.get('recommendations', [])

        if not recommendations:
            return html.Div("No recommendations available")

        rec_items = []
        for rec in recommendations[:5]:  # Show top 5
            rec_items.append(html.Li(rec, className="mb-2"))

        return html.Ul(rec_items)

    async def _perform_deep_dive(self, strategy_id: str) -> DeepDiveResult:
        """Perform deep dive analysis for a strategy"""
        cache_key = f"deep_dive_{strategy_id}"
        if cache_key in self.deep_dive_cache:
            return self.deep_dive_cache[cache_key]

        # Analyze strategy file
        config = strategy_testing_lab.strategy_configs.get(strategy_id, {})
        file_path = config.get('file_path')

        findings = {}
        recommendations = []
        risk_level = "low"

        if file_path and file_path.exists():
            # Analyze the strategy implementation
            findings = await self._analyze_strategy_file(file_path)
            recommendations = self._generate_file_recommendations(findings)
            risk_level = self._assess_file_risk(findings)

        # Analyze related system files
        system_findings = await self._analyze_related_files(strategy_id)
        findings.update(system_findings)

        result = DeepDiveResult(
            file_path=str(file_path) if file_path else "",
            analysis_type="comprehensive_strategy_analysis",
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )

        self.deep_dive_cache[cache_key] = result
        return result

    async def _analyze_strategy_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a strategy implementation file"""
        findings = {
            'code_quality': {},
            'parameters': {},
            'risk_management': {},
            'performance_indicators': {},
            'dependencies': []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Code quality analysis
            lines = content.split('\n')
            findings['code_quality'] = {
                'total_lines': len(lines),
                'classes': content.count('class '),
                'functions': content.count('def '),
                'async_functions': content.count('async def '),
                'error_handling': content.count('try:') > 0,
                'logging': content.count('logger.') > 0
            }

            # Parameter analysis
            import re
            param_matches = re.findall(r'self\.(\w+)\s*=\s*([\d.]+)', content)
            findings['parameters'] = {param: float(value) for param, value in param_matches}

            # Risk management analysis
            risk_indicators = {
                'stop_loss': 'stop_loss' in content.lower(),
                'position_sizing': 'position_size' in content.lower(),
                'risk_limits': 'max_' in content.lower(),
                'circuit_breakers': 'circuit' in content.lower()
            }
            findings['risk_management'] = risk_indicators

            # Performance indicators
            perf_indicators = {
                'sharpe_calculation': 'sharpe' in content.lower(),
                'returns_tracking': 'return' in content.lower(),
                'drawdown_monitoring': 'drawdown' in content.lower(),
                'benchmarking': 'benchmark' in content.lower()
            }
            findings['performance_indicators'] = perf_indicators

            # Dependencies
            import_lines = [line for line in lines if line.strip().startswith('from ') or line.strip().startswith('import ')]
            findings['dependencies'] = import_lines

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            findings['error'] = str(e)

        return findings

    async def _analyze_related_files(self, strategy_id: str) -> Dict[str, Any]:
        """Analyze files related to the strategy"""
        related_findings = {
            'test_files': [],
            'config_files': [],
            'data_files': [],
            'integration_status': {}
        }

        # Check for test files
        test_patterns = [
            f"test_{strategy_id}.py",
            f"test_{strategy_id.lower()}.py",
            f"{strategy_id}_test.py"
        ]

        for pattern in test_patterns:
            test_file = PROJECT_ROOT / "tests" / pattern
            if test_file.exists():
                related_findings['test_files'].append(str(test_file))

        # Check for config files
        config_file = PROJECT_ROOT / "config" / f"{strategy_id}.yaml"
        if config_file.exists():
            related_findings['config_files'].append(str(config_file))

        # Check for data files
        data_patterns = [
            f"{strategy_id}_data.csv",
            f"{strategy_id}_historical.csv",
            f"data/{strategy_id}/*"
        ]

        for pattern in data_patterns:
            data_path = PROJECT_ROOT / pattern
            if data_path.exists():
                if data_path.is_file():
                    related_findings['data_files'].append(str(data_path))
                elif data_path.is_dir():
                    related_findings['data_files'].extend([str(f) for f in data_path.glob("*")])

        return related_findings

    def _generate_file_recommendations(self, findings: Dict) -> List[str]:
        """Generate recommendations based on file analysis"""
        recommendations = []

        code_quality = findings.get('code_quality', {})

        if code_quality.get('total_lines', 0) > 500:
            recommendations.append("Consider breaking down large strategy file into smaller modules")

        if not code_quality.get('error_handling', False):
            recommendations.append("Add comprehensive error handling and exception management")

        if not code_quality.get('logging', False):
            recommendations.append("Implement structured logging for better debugging")

        risk_mgmt = findings.get('risk_management', {})
        if not risk_mgmt.get('stop_loss', False):
            recommendations.append("Implement stop-loss mechanisms for risk control")

        if not risk_mgmt.get('position_sizing', False):
            recommendations.append("Add dynamic position sizing based on volatility")

        perf_indicators = findings.get('performance_indicators', {})
        if not perf_indicators.get('sharpe_calculation', False):
            recommendations.append("Add Sharpe ratio calculation for risk-adjusted performance")

        if not findings.get('test_files'):
            recommendations.append("Create comprehensive unit tests for the strategy")

        return recommendations

    def _assess_file_risk(self, findings: Dict) -> str:
        """Assess risk level of the strategy implementation"""
        risk_score = 0

        # Code quality risks
        code_quality = findings.get('code_quality', {})
        if code_quality.get('total_lines', 0) > 1000:
            risk_score += 2
        if not code_quality.get('error_handling', False):
            risk_score += 1
        if not code_quality.get('logging', False):
            risk_score += 1

        # Risk management risks
        risk_mgmt = findings.get('risk_management', {})
        if not risk_mgmt.get('stop_loss', False):
            risk_score += 2
        if not risk_mgmt.get('position_sizing', False):
            risk_score += 1

        # Testing risks
        if not findings.get('test_files'):
            risk_score += 1

        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    def _create_deep_dive_display(self, deep_dive: DeepDiveResult) -> html.Div:
        """Create deep dive display"""
        findings = deep_dive.findings

        # Risk level badge
        risk_color = {"low": "success", "medium": "warning", "high": "danger"}[deep_dive.risk_level]
        risk_badge = dbc.Badge(f"Risk Level: {deep_dive.risk_level.upper()}",
                              color=risk_color, className="mb-3")

        # Code quality section
        code_quality = findings.get('code_quality', {})
        quality_items = [
            html.Li(f"Total Lines: {code_quality.get('total_lines', 0)}"),
            html.Li(f"Classes: {code_quality.get('classes', 0)}"),
            html.Li(f"Functions: {code_quality.get('functions', 0)}"),
            html.Li(f"Error Handling: {'Yes' if code_quality.get('error_handling') else 'No'}"),
            html.Li(f"Logging: {'Yes' if code_quality.get('logging') else 'No'}")
        ]

        # Risk management section
        risk_mgmt = findings.get('risk_management', {})
        risk_items = [
            html.Li(f"Stop Loss: {'Yes' if risk_mgmt.get('stop_loss') else 'No'}"),
            html.Li(f"Position Sizing: {'Yes' if risk_mgmt.get('position_sizing') else 'No'}"),
            html.Li(f"Risk Limits: {'Yes' if risk_mgmt.get('risk_limits') else 'No'}")
        ]

        # Recommendations
        rec_items = [html.Li(rec) for rec in deep_dive.recommendations]

        return html.Div([
            risk_badge,
            html.H5("Code Quality Analysis"),
            html.Ul(quality_items, className="mb-3"),
            html.H5("Risk Management"),
            html.Ul(risk_items, className="mb-3"),
            html.H5("Recommendations"),
            html.Ul(rec_items)
        ])

    async def run_dashboard(self, host: str = "127.0.0.1", port: int = 8050):
        """Run the dashboard server"""
        if not self.dashboard_app:
            await self.initialize()

        self.logger.info(f"Starting dashboard on http://{host}:{port}")
        self.dashboard_app.run(host=host, port=port, debug=False)

    async def generate_metrics_report(self, strategy_ids: List[str], output_dir: str = "reports/metrics") -> str:
        """Generate comprehensive metrics report"""
        self.logger.info(f"Generating metrics report for {len(strategy_ids)} strategies")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_path / f"metrics_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# Strategy Metrics Deep Dive Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("Comprehensive ARB currency simulation metrics and deep file analysis ")
            f.write("for arbitrage strategies.\n\n")

            for strategy_id in strategy_ids:
                f.write(f"## {strategy_id}: {strategy_testing_lab.strategy_configs[strategy_id]['name']}\n\n")

                # Get metrics
                try:
                    metrics = await self._get_strategy_metrics(strategy_id, "3M")
                    sim = metrics['simulation']
                    analysis = metrics['analysis']

                    f.write("### Performance Metrics\n")
                    f.write(f"- Initial Balance: ${sim['initial_balance']:,.2f} ARB\n")
                    f.write(f"- Final Balance: ${sim['final_balance']:,.2f} ARB\n")
                    f.write(f"- Total Return: {sim['total_return_pct']:.1f}%\n")
                    f.write(f"- Sharpe Ratio: {sim['sharpe_ratio']:.2f}\n")
                    f.write(f"- Max Drawdown: {sim['max_drawdown']:.1f}%\n\n")

                    # Deep dive
                    deep_dive = await self._perform_deep_dive(strategy_id)
                    f.write("### Deep Dive Analysis\n")
                    f.write(f"- Risk Level: {deep_dive['risk_level'].upper()}\n")
                    f.write("- Key Findings:\n")
                    for key, value in deep_dive['findings'].get('code_quality', {}).items():
                        f.write(f"  - {key}: {value}\n")
                    f.write("\n- Recommendations:\n")
                    for rec in deep_dive['recommendations']:
                        f.write(f"  - {rec}\n")
                    f.write("\n")

                except Exception as e:
                    f.write(f"Error analyzing {strategy_id}: {e}\n\n")

        self.logger.info(f"Metrics report generated: {report_path}")
        return str(report_path)

    async def _get_strategy_metrics(self, strategy_id: str, timeframe: str) -> Dict[str, Any]:
        """Get strategy metrics by running simulation"""
        # Run simulation
        sim_results = await strategy_testing_lab.run_strategy_simulation(strategy_id, timeframe)
        
        # Mock analysis for now
        analysis = {
            'performance_analysis': {
                'sharpe_ratio': sim_results['sharpe_ratio'],
                'total_return': sim_results['total_return_pct'],
                'volatility': sim_results['volatility']
            },
            'risk_analysis': {
                'max_drawdown': sim_results['max_drawdown'],
                'value_at_risk': sim_results['volatility'] * 1.645  # 95% VaR approximation
            },
            'predictive_analysis': {
                'forecast_accuracy': 0.75,
                'trend_prediction': 'bullish' if sim_results['total_return_pct'] > 0 else 'bearish'
            }
        }
        
        return {
            'simulation': sim_results,
            'analysis': analysis
        }

    async def _perform_deep_dive(self, strategy_id: str) -> Dict[str, Any]:
        """Perform deep dive analysis for a strategy"""
        from strategy_analysis_engine import StrategyAnalysisEngine, AnalysisType

        analysis_engine = StrategyAnalysisEngine()
        await analysis_engine.initialize()

        # Perform comprehensive analysis
        analysis = await analysis_engine.perform_comprehensive_analysis(
            [strategy_id], 
            [AnalysisType.PERFORMANCE, AnalysisType.RISK, AnalysisType.PREDICTIVE, AnalysisType.MARKET_REGIME]
        )

        # Extract strategy-specific analysis
        strategy_analysis = analysis.get('strategy_analyses', {}).get(strategy_id, {})

        # Create deep dive result
        deep_dive = {
            'strategy_id': strategy_id,
            'risk_level': strategy_analysis.get('market_regime', {}).get('risk_level', 'medium'),
            'findings': {
                'code_quality': strategy_analysis.get('performance', {}),
                'performance': strategy_analysis.get('performance', {}),
                'risk': strategy_analysis.get('risk', {})
            },
            'recommendations': analysis.get('recommendations', [])
        }

        return deep_dive


# Global dashboard instance
metrics_dashboard = StrategyMetricsDashboard()


async def initialize_metrics_dashboard():
    """Initialize the global metrics dashboard"""
    await metrics_dashboard.initialize()


async def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Metrics Dashboard & Deep Dive Analyzer")
    parser.add_argument('command', choices=['dashboard', 'report', 'analyze'],
                       help='Command to execute')
    parser.add_argument('--strategy-ids', nargs='+', help='Strategy IDs to analyze')
    parser.add_argument('--host', default='127.0.0.1', help='Dashboard host')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--output-dir', default='reports/metrics', help='Output directory')

    args = parser.parse_args()

    await initialize_metrics_dashboard()

    if args.command == 'dashboard':
        print("🚀 Starting Strategy Metrics Dashboard...")
        print(f"📊 Dashboard will be available at http://{args.host}:{args.port}")
        await metrics_dashboard.run_dashboard(args.host, args.port)

    elif args.command == 'report':
        strategy_ids = args.strategy_ids or list(strategy_testing_lab.strategy_configs.keys())[:5]
        print(f"📊 Generating metrics report for {len(strategy_ids)} strategies...")
        report_path = await metrics_dashboard.generate_metrics_report(strategy_ids, args.output_dir)
        print(f"📄 Report generated: {report_path}")

    elif args.command == 'analyze':
        strategy_ids = args.strategy_ids or ['s26']
        for strategy_id in strategy_ids:
            print(f"🔍 Deep dive analysis for {strategy_id}...")
            deep_dive = await metrics_dashboard._perform_deep_dive(strategy_id)
            print(f"Risk Level: {deep_dive['risk_level']}")
            print(f"Recommendations: {len(deep_dive['recommendations'])}")
            print()


if __name__ == "__main__":
    asyncio.run(main())

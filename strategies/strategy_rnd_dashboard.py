#!/usr/bin/env python3
"""
Strategy R&D Dashboard
=====================
Comprehensive research and development interface for strategy optimization.
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.strategy_parameter_tester import (
    strategy_parameter_tester,
    initialize_strategy_parameter_testing,
    OptimizationMethod
)


class StrategyRnDDashboard:
    """Comprehensive R&D dashboard for strategy development"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.experiments: Dict[str, Dict] = {}
        self.results_history: List[Dict] = []

    async def initialize(self):
        """Initialize the R&D dashboard"""
        self.logger.info("Initializing Strategy R&D Dashboard")
        await initialize_strategy_parameter_testing()
        self.logger.info("[OK] R&D Dashboard initialized")

    async def run_comprehensive_experiment(self,
                                         experiment_name: str,
                                         strategy_types: List[str],
                                         methods: List[OptimizationMethod],
                                         n_iterations: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive experiment across multiple strategy types and methods

        Args:
            experiment_name: Name for this experiment
            strategy_types: List of strategy types to test
            methods: List of optimization methods to use
            n_iterations: Number of iterations per strategy-method combination
        """

        self.logger.info(f"Starting comprehensive experiment: {experiment_name}")

        experiment_results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now(),
            "strategy_results": {},
            "comparative_analysis": {},
            "recommendations": []
        }

        # Run optimization for each strategy type and method
        for strategy_type in strategy_types:
            experiment_results["strategy_results"][strategy_type] = {}

            for method in methods:
                self.logger.info(f"Testing {strategy_type} with {method.value}")

                results = await strategy_parameter_tester.run_parameter_sweep(
                    strategy_type=strategy_type,
                    optimization_method=method,
                    n_iterations=n_iterations,
                    n_samples_per_param=5
                )

                experiment_results["strategy_results"][strategy_type][method.value] = [
                    {
                        "parameters": r.parameters,
                        "metrics": r.metrics,
                        "score": r.score,
                        "trade_count": r.trade_count
                    }
                    for r in results
                ]

        # Generate comparative analysis
        experiment_results["comparative_analysis"] = self._generate_comparative_analysis(experiment_results)

        # Generate recommendations
        experiment_results["recommendations"] = self._generate_recommendations(experiment_results)

        # Store experiment
        self.experiments[experiment_name] = experiment_results
        self.results_history.append(experiment_results)

        self.logger.info(f"Comprehensive experiment {experiment_name} completed")

        return experiment_results

    def _generate_comparative_analysis(self, experiment_results: Dict) -> Dict[str, Any]:
        """Generate comparative analysis across strategies and methods"""

        analysis = {
            "best_performers": {},
            "method_effectiveness": {},
            "strategy_comparison": {},
            "parameter_insights": {}
        }

        # Find best performers for each strategy
        for strategy_type, method_results in experiment_results["strategy_results"].items():
            best_score = -float('inf')
            best_method = None
            best_params = None

            for method, results in method_results.items():
                if results and results[0]["score"] > best_score:
                    best_score = results[0]["score"]
                    best_method = method
                    best_params = results[0]["parameters"]

            analysis["best_performers"][strategy_type] = {
                "best_method": best_method,
                "best_score": best_score,
                "best_parameters": best_params
            }

        # Compare method effectiveness
        method_scores = {}
        for strategy_type, method_results in experiment_results["strategy_results"].items():
            for method, results in method_results.items():
                if results:
                    if method not in method_scores:
                        method_scores[method] = []
                    method_scores[method].extend([r["score"] for r in results])

        analysis["method_effectiveness"] = {
            method: {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "max_score": max(scores),
                "n_experiments": len(scores)
            }
            for method, scores in method_scores.items()
        }

        return analysis

    def _generate_recommendations(self, experiment_results: Dict) -> List[str]:
        """Generate actionable recommendations based on experiment results"""

        recommendations = []

        comparative = experiment_results["comparative_analysis"]

        # Best overall strategy
        best_strategy = max(
            comparative["best_performers"].items(),
            key=lambda x: x[1]["best_score"]
        )
        recommendations.append(f"🎯 Focus on {best_strategy[0]} strategy (best score: {best_strategy[1]['best_score']:.4f})")

        # Best method
        best_method = max(
            comparative["method_effectiveness"].items(),
            key=lambda x: x[1]["mean_score"]
        )
        recommendations.append(f"🔬 Use {best_method[0]} optimization method (avg score: {best_method[1]['mean_score']:.4f})")

        # Parameter insights
        for strategy, data in comparative["best_performers"].items():
            recommendations.append(f"⚙️ For {strategy}, optimal parameters: {data['best_parameters']}")

        # Method comparison
        methods_sorted = sorted(
            comparative["method_effectiveness"].items(),
            key=lambda x: x[1]["mean_score"],
            reverse=True
        )
        recommendations.append(f"📊 Method ranking: {' > '.join([m[0] for m in methods_sorted])}")

        return recommendations

    def generate_experiment_report(self, experiment_name: str, output_dir: str = "reports/rnd") -> str:
        """Generate comprehensive experiment report"""

        if experiment_name not in self.experiments:
            return f"Experiment {experiment_name} not found"

        experiment = self.experiments[experiment_name]

        report = []
        report.append("# Strategy R&D Experiment Report")
        report.append(f"Experiment: {experiment_name}")
        report.append(f"Generated: {datetime.now()}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"This report presents the results of comprehensive parameter optimization across {len(experiment['strategy_results'])} strategy types.")
        report.append("")

        # Best Performers
        report.append("## Best Performing Strategies")
        for strategy, data in experiment["comparative_analysis"]["best_performers"].items():
            report.append(f"### {strategy}")
            report.append(f"- **Best Method**: {data['best_method']}")
            report.append(f"- **Best Score**: {data['best_score']:.4f}")
            report.append(f"- **Optimal Parameters**: {json.dumps(data['best_parameters'], indent=2)}")
            report.append("")

        # Method Effectiveness
        report.append("## Optimization Method Effectiveness")
        for method, stats in experiment["comparative_analysis"]["method_effectiveness"].items():
            report.append(f"### {method}")
            report.append(f"- **Mean Score**: {stats['mean_score']:.4f}")
            report.append(f"- **Std Deviation**: {stats['std_score']:.4f}")
            report.append(f"- **Max Score**: {stats['max_score']:.4f}")
            report.append(f"- **Experiments**: {stats['n_experiments']}")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        for rec in experiment["recommendations"]:
            report.append(f"- {rec}")
        report.append("")

        # Detailed Results
        report.append("## Detailed Results")
        for strategy, method_results in experiment["strategy_results"].items():
            report.append(f"### {strategy}")
            for method, results in method_results.items():
                if results:
                    best_result = results[0]
                    report.append(f"#### {method}")
                    report.append(f"- Top Score: {best_result['score']:.4f}")
                    report.append(f"- Parameters: {json.dumps(best_result['parameters'], indent=2)}")
                    report.append(f"- Metrics: {json.dumps(best_result['metrics'], indent=2)}")
                    report.append("")

        # Save report
        output_path = f"{output_dir}/{experiment_name}_report.md"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

        self.logger.info(f"Experiment report saved to {output_path}")

        return '\n'.join(report)

    def create_visualization_dashboard(self, experiment_name: str, output_dir: str = "reports/rnd"):
        """Create interactive visualization dashboard"""

        if experiment_name not in self.experiments:
            self.logger.error(f"Experiment {experiment_name} not found")
            return

        experiment = self.experiments[experiment_name]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Strategy Performance Comparison', 'Method Effectiveness',
                          'Score Distributions', 'Parameter Correlations'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}]]
        )

        # Strategy Performance Comparison
        strategies = list(experiment["comparative_analysis"]["best_performers"].keys())
        scores = [data["best_score"] for data in experiment["comparative_analysis"]["best_performers"].values()]

        fig.add_trace(
            go.Bar(x=strategies, y=scores, name='Best Scores'),
            row=1, col=1
        )

        # Method Effectiveness
        methods = list(experiment["comparative_analysis"]["method_effectiveness"].keys())
        mean_scores = [data["mean_score"] for data in experiment["comparative_analysis"]["method_effectiveness"].values()]

        fig.add_trace(
            go.Bar(x=methods, y=mean_scores, name='Mean Scores'),
            row=1, col=2
        )

        # Score Distributions (simplified - showing all scores)
        all_scores = []
        for strategy_results in experiment["strategy_results"].values():
            for method_results in strategy_results.values():
                all_scores.extend([r["score"] for r in method_results])

        fig.add_trace(
            go.Histogram(x=all_scores, name='Score Distribution'),
            row=2, col=1
        )

        # Parameter Correlations (simplified example)
        # This would need more sophisticated analysis
        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='markers', name='Parameter Correlation'),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(height=800, title_text=f"R&D Dashboard - {experiment_name}")

        # Save dashboard
        output_path = f"{output_dir}/{experiment_name}_dashboard.html"
        fig.write_html(output_path)

        self.logger.info(f"Visualization dashboard saved to {output_path}")

    async def run_ab_test(self,
                         strategy_type: str,
                         param_set_a: Dict[str, Any],
                         param_set_b: Dict[str, Any],
                         n_runs: int = 50) -> Dict[str, Any]:
        """
        Run A/B test between two parameter sets

        Args:
            strategy_type: Strategy type to test
            param_set_a: Parameters for variant A
            param_set_b: Parameters for variant B
            n_runs: Number of runs for each variant
        """

        self.logger.info(f"Running A/B test for {strategy_type}")

        results_a = []
        results_b = []

        # Run tests for variant A
        for i in range(n_runs):
            strategy = await strategy_parameter_tester._create_strategy_with_params(strategy_type, param_set_a)
            result = await strategy_parameter_tester._test_strategy_parameters(strategy, param_set_a)
            if result:
                results_a.append(result.score)

        # Run tests for variant B
        for i in range(n_runs):
            strategy = await strategy_parameter_tester._create_strategy_with_params(strategy_type, param_set_b)
            result = await strategy_parameter_tester._test_strategy_parameters(strategy, param_set_b)
            if result:
                results_b.append(result.score)

        # Statistical analysis
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(results_a, results_b)

        analysis = {
            "variant_a": {
                "parameters": param_set_a,
                "scores": results_a,
                "mean_score": np.mean(results_a),
                "std_score": np.std(results_a),
                "n_runs": len(results_a)
            },
            "variant_b": {
                "parameters": param_set_b,
                "scores": results_b,
                "mean_score": np.mean(results_b),
                "std_score": np.std(results_b),
                "n_runs": len(results_b)
            },
            "statistical_test": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "winner": "A" if np.mean(results_a) > np.mean(results_b) else "B"
            }
        }

        return analysis


# Global instance
rnd_dashboard = StrategyRnDDashboard()


async def run_comprehensive_rnd_demo():
    """Demo function for comprehensive R&D"""

    print("🔬 ACCELERATED ARBITRAGE CORP - Strategy R&D Dashboard")
    print("=" * 60)

    await rnd_dashboard.initialize()

    # Run comprehensive experiment
    experiment_name = "comprehensive_strategy_optimization_2026"
    strategy_types = ["statistical_arbitrage", "triangular_arbitrage"]
    methods = [OptimizationMethod.GRID_SEARCH, OptimizationMethod.RANDOM_SEARCH]

    print("\n🎯 Running comprehensive experiment...")
    results = await rnd_dashboard.run_comprehensive_experiment(
        experiment_name=experiment_name,
        strategy_types=strategy_types,
        methods=methods,
        n_iterations=20  # Small for demo
    )

    # Generate reports
    report = rnd_dashboard.generate_experiment_report(experiment_name)
    rnd_dashboard.create_visualization_dashboard(experiment_name)

    print("\n✅ Comprehensive R&D experiment completed!")
    print(f"Experiment: {experiment_name}")
    print(f"Strategies tested: {len(strategy_types)}")
    print(f"Methods tested: {len(methods)}")
    print(f"Report: reports/rnd/{experiment_name}_report.md")
    print(f"Dashboard: reports/rnd/{experiment_name}_dashboard.html")

    # Print top recommendations
    print("\n🎯 Top Recommendations:")
    for rec in results["recommendations"][:3]:
        print(f"  {rec}")

    return results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_rnd_demo())
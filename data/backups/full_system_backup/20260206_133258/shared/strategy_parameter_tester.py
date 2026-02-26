#!/usr/bin/env python3
"""
Strategy Parameter Testing & R&D Engine
======================================
Advanced parameter optimization, grid search, and strategy R&D capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import random
import itertools

# Import trading enums
from shared.paper_trading import OrderSide, OrderType
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger
from shared.paper_trading import paper_trading_engine, initialize_paper_trading


@dataclass
class ParameterRange:
    """Parameter range definition for optimization"""
    name: str
    min_val: float
    max_val: float
    step: float
    distribution: str = "uniform"  # uniform, log, normal
    log_base: float = 10.0

    def generate_values(self, n_samples: int) -> List[float]:
        """Generate parameter values based on distribution"""
        if self.distribution == "uniform":
            return np.linspace(self.min_val, self.max_val, n_samples).tolist()
        elif self.distribution == "log":
            log_min = np.log(self.min_val) / np.log(self.log_base)
            log_max = np.log(self.max_val) / np.log(self.log_base)
            log_values = np.linspace(log_min, log_max, n_samples)
            return [self.log_base ** x for x in log_values]
        elif self.distribution == "normal":
            mean = (self.min_val + self.max_val) / 2
            std = (self.max_val - self.min_val) / 6  # 99.7% within range
            return np.random.normal(mean, std, n_samples).tolist()
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class StrategyParameters:
    """Strategy parameter configuration"""
    strategy_type: str
    parameters: Dict[str, ParameterRange]
    fixed_parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Callable] = field(default_factory=list)

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter combination against constraints"""
        for constraint in self.constraints:
            if not constraint(params):
                return False
        return True


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    strategy_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    score: float
    timestamp: datetime
    test_duration: float
    trade_count: int


class OptimizationMethod(Enum):
    """Parameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    GRADIENT_DESCENT = "gradient_descent"


class StrategyParameterTester:
    """Advanced strategy parameter testing and optimization engine"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.audit_logger = get_audit_logger()
        self.paper_trading = None
        self.results_history: List[OptimizationResult] = []
        self.parameter_configs = self._load_parameter_configs()

    async def initialize(self):
        """Initialize the parameter testing system"""
        self.logger.info("Initializing Strategy Parameter Tester")

        # Initialize paper trading for testing
        if not self.paper_trading:
            await initialize_paper_trading()
            self.paper_trading = paper_trading_engine

        self.logger.info("[OK] Strategy Parameter Tester initialized")

    def _load_parameter_configs(self) -> Dict[str, StrategyParameters]:
        """Load parameter configurations for different strategy types"""
        configs = {}

        # Statistical Arbitrage Parameters
        configs["statistical_arbitrage"] = StrategyParameters(
            strategy_type="statistical_arbitrage",
            parameters={
                "entry_threshold": ParameterRange("entry_threshold", 0.5, 3.0, 0.25, "uniform"),
                "exit_threshold": ParameterRange("exit_threshold", 0.1, 1.0, 0.1, "uniform"),
                "lookback_period": ParameterRange("lookback_period", 20, 200, 20, "uniform"),
                "max_holding_period": ParameterRange("max_holding_period", 5, 50, 5, "uniform"),
                "position_size_pct": ParameterRange("position_size_pct", 0.01, 0.10, 0.01, "uniform"),
                "stop_loss_pct": ParameterRange("stop_loss_pct", 0.005, 0.05, 0.005, "uniform"),
            },
            fixed_parameters={
                "min_volume": 100000,
                "max_spread": 0.02,
            },
            constraints=[
                lambda p: p["entry_threshold"] > p["exit_threshold"],  # Entry > Exit
                lambda p: p["lookback_period"] > p["max_holding_period"],  # Lookback > Holding
            ]
        )

        # Triangular Arbitrage Parameters
        configs["triangular_arbitrage"] = StrategyParameters(
            strategy_type="triangular_arbitrage",
            parameters={
                "min_profit_threshold": ParameterRange("min_profit_threshold", 0.001, 0.01, 0.001, "uniform"),
                "max_slippage": ParameterRange("max_slippage", 0.0001, 0.001, 0.0001, "uniform"),
                "execution_delay_ms": ParameterRange("execution_delay_ms", 10, 500, 50, "uniform"),
                "volume_threshold": ParameterRange("volume_threshold", 10000, 1000000, 10000, "log"),
                "confidence_threshold": ParameterRange("confidence_threshold", 0.7, 0.95, 0.05, "uniform"),
            },
            fixed_parameters={
                "max_triangle_size": 3,
                "require_all_exchanges": True,
            }
        )

        # Cross-Exchange Arbitrage Parameters
        configs["cross_exchange_arbitrage"] = StrategyParameters(
            strategy_type="cross_exchange_arbitrage",
            parameters={
                "price_diff_threshold": ParameterRange("price_diff_threshold", 0.001, 0.05, 0.001, "uniform"),
                "min_volume_ratio": ParameterRange("min_volume_ratio", 0.1, 1.0, 0.1, "uniform"),
                "max_age_seconds": ParameterRange("max_age_seconds", 1, 300, 10, "uniform"),
                "transaction_fee_buffer": ParameterRange("transaction_fee_buffer", 0.0001, 0.01, 0.0001, "uniform"),
                "liquidity_threshold": ParameterRange("liquidity_threshold", 1000, 100000, 1000, "log"),
            }
        )

        # ML-based Strategy Parameters
        configs["ml_enhanced_arbitrage"] = StrategyParameters(
            strategy_type="ml_enhanced_arbitrage",
            parameters={
                "model_complexity": ParameterRange("model_complexity", 10, 1000, 50, "log"),
                "learning_rate": ParameterRange("learning_rate", 0.001, 0.1, 0.01, "log"),
                "feature_window": ParameterRange("feature_window", 5, 100, 5, "uniform"),
                "prediction_horizon": ParameterRange("prediction_horizon", 1, 60, 5, "uniform"),
                "confidence_threshold": ParameterRange("confidence_threshold", 0.5, 0.9, 0.05, "uniform"),
                "max_features": ParameterRange("max_features", 5, 50, 5, "uniform"),
            },
            fixed_parameters={
                "model_type": "gradient_boosting",
                "cv_folds": 5,
            }
        )

        return configs

    async def run_parameter_sweep(self,
                                strategy_type: str,
                                optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                                n_iterations: int = 50,
                                n_samples_per_param: int = 5) -> List[OptimizationResult]:
        """
        Run parameter optimization for a strategy type

        Args:
            strategy_type: Type of strategy to optimize
            optimization_method: Optimization method to use
            n_iterations: Number of parameter combinations to test
            n_samples_per_param: Number of samples per parameter (for grid search)
        """
        self.logger.info(f"Starting parameter sweep for {strategy_type} using {optimization_method.value}")

        if strategy_type not in self.parameter_configs:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        config = self.parameter_configs[strategy_type]
        parameter_combinations = self._generate_parameter_combinations(
            config, optimization_method, n_iterations, n_samples_per_param
        )

        results = []
        total_combinations = len(parameter_combinations)

        self.logger.info(f"Testing {total_combinations} parameter combinations")

        for i, params in enumerate(parameter_combinations):
            if not config.validate_parameters(params):
                continue

            self.logger.info(f"Testing combination {i+1}/{total_combinations}: {params}")

            # Create strategy with these parameters
            strategy = await self._create_strategy_with_params(strategy_type, params)

            # Test the strategy
            result = await self._test_strategy_parameters(strategy, params)

            if result:
                results.append(result)
                self.results_history.append(result)

            # Log progress
            if (i + 1) % 10 == 0:
                self.logger.info(f"Completed {i+1}/{total_combinations} combinations")

        # Sort results by score (higher is better)
        results.sort(key=lambda x: x.score, reverse=True)

        self.logger.info(f"Parameter sweep completed. Best score: {results[0].score if results else 'N/A'}")

        return results

    def _generate_parameter_combinations(self,
                                       config: StrategyParameters,
                                       method: OptimizationMethod,
                                       n_iterations: int,
                                       n_samples_per_param: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations based on optimization method"""

        if method == OptimizationMethod.GRID_SEARCH:
            # Create parameter grid
            param_grid = {}
            for param_name, param_range in config.parameters.items():
                param_grid[param_name] = param_range.generate_values(n_samples_per_param)

            grid = ParameterGrid(param_grid)
            return list(grid)

        elif method == OptimizationMethod.RANDOM_SEARCH:
            combinations = []
            for _ in range(n_iterations):
                params = {}
                for param_name, param_range in config.parameters.items():
                    values = param_range.generate_values(100)  # Generate many values
                    params[param_name] = random.choice(values)
                combinations.append(params)
            return combinations

        else:
            raise ValueError(f"Optimization method {method} not implemented yet")

    async def _create_strategy_with_params(self, strategy_type: str, params: Dict[str, Any]):
        """Create a strategy instance with given parameters"""
        # This would integrate with the AI strategy generator
        # For now, create a mock strategy object
        from shared.ai_strategy_generator import GeneratedStrategy

        strategy = GeneratedStrategy(
            strategy_id=f"param_test_{strategy_type}_{datetime.now().timestamp()}",
            name=f"Parameter Test {strategy_type}",
            description=f"Auto-generated {strategy_type} strategy with optimized parameters",
            strategy_type=strategy_type,
            risk_level="medium",
            symbols=["SPY", "QQQ"],  # Default symbols
            parameters=params,
            entry_logic="pass",  # Placeholder
            exit_logic="pass",   # Placeholder
            risk_management={"max_drawdown": 0.05},
            backtest_results={}
        )

        return strategy

    async def _test_strategy_parameters(self, strategy, params: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Test a strategy with given parameters"""
        try:
            start_time = datetime.now()

            # Reset paper account
            await self.paper_trading.reset_account()

            # Simulate strategy execution with parameters
            trades_executed = await self._simulate_parameterized_strategy(strategy, params)

            # Calculate metrics
            account_summary = self.paper_trading.get_account_summary()
            trade_history = self.paper_trading.get_trade_history(limit=100)

            if not trade_history:
                return None

            # Calculate comprehensive metrics
            metrics = self._calculate_strategy_metrics(trade_history, account_summary)

            # Calculate optimization score (weighted combination of metrics)
            score = self._calculate_optimization_score(metrics)

            test_duration = (datetime.now() - start_time).total_seconds()

            result = OptimizationResult(
                strategy_id=strategy.strategy_id,
                parameters=params,
                metrics=metrics,
                score=score,
                timestamp=datetime.now(),
                test_duration=test_duration,
                trade_count=len(trade_history)
            )

            return result

        except Exception as e:
            self.logger.error(f"Error testing parameters {params}: {e}")
            return None

    async def _simulate_parameterized_strategy(self, strategy, params: Dict[str, Any]) -> List[Dict]:
        """Simulate strategy execution with specific parameters"""
        trades_executed = []

        # Extract parameters with defaults
        entry_threshold = params.get("entry_threshold", 1.0)
        exit_threshold = params.get("exit_threshold", 0.5)
        max_holding_period = params.get("max_holding_period", 20)
        position_size_pct = min(params.get("position_size_pct", 0.05), 0.02)  # Cap at 2% for testing

        # Simulate trades based on strategy type and parameters
        for i in range(4):  # Simulate 4 trades
            # Generate realistic trade based on parameters
            if strategy.strategy_type == "statistical_arbitrage":
                # Simulate stat arb trade
                symbol = random.choice(["SPY", "QQQ"])
                quantity = max(1, int(1000000 * position_size_pct / 450))  # Much smaller position size
                price = 450 + random.uniform(-10, 10)

                # Simulate entry
                await self.paper_trading.submit_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )

                # Simulate exit after some time
                exit_price = price * (1 + random.uniform(-0.02, 0.03))
                await self.paper_trading.submit_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,  # Close full position
                    order_type=OrderType.MARKET
                )

                trades_executed.append({
                    "symbol": symbol,
                    "entry_price": price,
                    "exit_price": exit_price,
                    "quantity": quantity
                })

        return trades_executed

    def _calculate_strategy_metrics(self, trade_history: List[Dict], account_summary: Dict) -> Dict[str, float]:
        """Calculate comprehensive strategy performance metrics"""
        if not trade_history:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "avg_trade_pnl": 0.0
            }

        # Calculate returns
        pnls = [t['pnl'] for t in trade_history]
        total_pnl = sum(pnls)
        total_return = total_pnl / 1000000.0  # From $1M starting balance

        # Win rate
        winning_trades = len([p for p in pnls if p > 0])
        win_rate = winning_trades / len(pnls)

        # Maximum drawdown
        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        for pnl in pnls:
            cumulative_pnl += pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = peak_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max_drawdown / 1000000.0 if peak_pnl > 0 else 0

        # Sharpe ratio
        if len(pnls) > 1:
            returns = np.array(pnls) / 1000000.0
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Profit factor
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = min(total_wins / total_losses if total_losses > 0 else 10.0, 10.0)  # Cap at 10

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor,
            "total_trades": len(trade_history),
            "avg_trade_pnl": np.mean(pnls)
        }

    def _calculate_optimization_score(self, metrics: Dict[str, float]) -> float:
        """Calculate optimization score from metrics"""
        # Weighted combination of key metrics
        weights = {
            "total_return": 0.3,
            "win_rate": 0.25,
            "sharpe_ratio": 0.2,
            "profit_factor": 0.15,
            "max_drawdown": -0.1  # Negative weight for drawdown
        }

        score = 0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            if metric == "max_drawdown":
                # Invert drawdown (lower is better)
                value = 1 - min(value, 1)  # Cap at 1
            score += weight * value

        return score

    async def run_comparative_analysis(self,
                                     strategy_type: str,
                                     parameter_sets: List[Dict[str, Any]],
                                     n_runs_per_set: int = 3) -> Dict[str, List[OptimizationResult]]:
        """
        Run comparative analysis of different parameter sets

        Args:
            strategy_type: Strategy type to test
            parameter_sets: List of parameter dictionaries to compare
            n_runs_per_set: Number of runs per parameter set for statistical significance
        """
        self.logger.info(f"Running comparative analysis for {strategy_type} with {len(parameter_sets)} parameter sets")

        results = {}

        for i, params in enumerate(parameter_sets):
            self.logger.info(f"Testing parameter set {i+1}/{len(parameter_sets)}")

            set_results = []
            for run in range(n_runs_per_set):
                strategy = await self._create_strategy_with_params(strategy_type, params)
                result = await self._test_strategy_parameters(strategy, params)
                if result:
                    set_results.append(result)

            results[f"set_{i+1}"] = set_results

        return results

    def generate_optimization_report(self, results: List[OptimizationResult], output_path: Optional[str] = None) -> str:
        """Generate comprehensive optimization report"""
        if not results:
            return "No results to report"

        # Sort by score
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        report = []
        report.append("# Strategy Parameter Optimization Report")
        report.append(f"Generated: {datetime.now()}")
        report.append(f"Total Combinations Tested: {len(results)}")
        report.append("")

        # Best parameters
        best_result = sorted_results[0]
        report.append("## Best Parameter Combination")
        report.append(f"- Score: {best_result.score:.4f}")
        report.append(f"- Strategy ID: {best_result.strategy_id}")
        report.append(f"- Parameters: {json.dumps(best_result.parameters, indent=2)}")
        report.append(f"- Metrics: {json.dumps(best_result.metrics, indent=2)}")
        report.append("")

        # Top 10 results
        report.append("## Top 10 Parameter Combinations")
        for i, result in enumerate(sorted_results[:10]):
            report.append(f"### Rank {i+1} (Score: {result.score:.4f})")
            report.append(f"- Parameters: {json.dumps(result.parameters, indent=2)}")
            report.append(f"- Key Metrics: Return={result.metrics['total_return']:.1%}, Win Rate={result.metrics['win_rate']:.1%}, Sharpe={result.metrics['sharpe_ratio']:.2f}")
            report.append("")

        # Parameter sensitivity analysis
        report.append("## Parameter Sensitivity Analysis")
        param_sensitivity = self._analyze_parameter_sensitivity(results)
        for param, sensitivity in param_sensitivity.items():
            report.append(f"- {param}: {sensitivity}")
        report.append("")

        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                f.write('\n'.join(report))
            self.logger.info(f"Report saved to {output_path}")

        return '\n'.join(report)

    def _analyze_parameter_sensitivity(self, results: List[OptimizationResult]) -> Dict[str, str]:
        """Analyze which parameters have the most impact on performance"""
        if not results:
            return {}

        # Extract parameter names
        param_names = list(results[0].parameters.keys())

        sensitivity = {}
        for param in param_names:
            try:
                # Simple correlation analysis
                param_values = [r.parameters[param] for r in results]
                scores = [r.score for r in results]

                correlation = np.corrcoef(param_values, scores)[0, 1]
                if abs(correlation) > 0.3:
                    direction = "positive" if correlation > 0 else "negative"
                    strength = "strong" if abs(correlation) > 0.5 else "moderate"
                    sensitivity[param] = f"{strength} {direction} correlation ({correlation:.2f})"
                else:
                    sensitivity[param] = "weak correlation"
            except:
                sensitivity[param] = "analysis failed"

        return sensitivity

    def plot_optimization_results(self, results: List[OptimizationResult], output_path: Optional[str] = None):
        """Generate visualization plots for optimization results"""
        if not results:
            return

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Score distribution
        scores = [r.score for r in results]
        axes[0, 0].hist(scores, bins=20, alpha=0.7)
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].set_xlabel('Optimization Score')
        axes[0, 0].set_ylabel('Frequency')

        # Score vs key metrics
        returns = [r.metrics['total_return'] for r in results]
        axes[0, 1].scatter(scores, returns, alpha=0.6)
        axes[0, 1].set_title('Score vs Total Return')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Total Return')

        # Win rate distribution
        win_rates = [r.metrics['win_rate'] for r in results]
        axes[1, 0].hist(win_rates, bins=20, alpha=0.7)
        axes[1, 0].set_title('Win Rate Distribution')
        axes[1, 0].set_xlabel('Win Rate')
        axes[1, 0].set_ylabel('Frequency')

        # Sharpe ratio distribution
        sharpe_ratios = [r.metrics['sharpe_ratio'] for r in results]
        axes[1, 1].hist(sharpe_ratios, bins=20, alpha=0.7)
        axes[1, 1].set_title('Sharpe Ratio Distribution')
        axes[1, 1].set_xlabel('Sharpe Ratio')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {output_path}")
        else:
            plt.show()


# Global instance
strategy_parameter_tester = StrategyParameterTester()


async def initialize_strategy_parameter_testing():
    """Initialize the strategy parameter testing system"""
    await strategy_parameter_tester.initialize()


# Example usage and testing
async def run_parameter_optimization_demo():
    """Demo function for parameter optimization"""
    print("🔬 ACCELERATED ARBITRAGE CORP - Strategy Parameter R&D")
    print("=" * 60)

    await initialize_strategy_parameter_testing()

    # Run parameter sweep for statistical arbitrage
    print("\n🎯 Running Parameter Sweep for Statistical Arbitrage...")
    results = await strategy_parameter_tester.run_parameter_sweep(
        strategy_type="statistical_arbitrage",
        optimization_method=OptimizationMethod.GRID_SEARCH,
        n_samples_per_param=3  # Small for demo
    )

    print(f"\n✅ Parameter sweep completed! Tested {len(results)} combinations")

    # Generate report
    report = strategy_parameter_tester.generate_optimization_report(
        results,
        output_path="reports/parameter_optimization_report.md"
    )

    print("\n📊 Optimization Report Summary:")
    print(f"Best Score: {results[0].score:.4f}" if results else "No results")
    print(f"Report saved to: reports/parameter_optimization_report.md")

    return results


if __name__ == "__main__":
    asyncio.run(run_parameter_optimization_demo())
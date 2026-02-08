#!/usr/bin/env python3
"""
Strategy Parameter Optimization Runner
====================================
Command-line interface for running parameter optimization experiments.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.strategy_parameter_tester import (
    strategy_parameter_tester,
    initialize_strategy_parameter_testing,
    OptimizationMethod
)


async def run_optimization(strategy_type: str,
                          method: str,
                          n_iterations: int,
                          n_samples: int,
                          output_dir: str):
    """Run parameter optimization for a strategy type"""

    print(f"🔬 Starting parameter optimization for {strategy_type}")
    print(f"Method: {method}, Iterations: {n_iterations}, Samples: {n_samples}")

    # Initialize system
    await initialize_strategy_parameter_testing()

    # Convert method string to enum
    method_enum = OptimizationMethod[method.upper()]

    # Run optimization
    results = await strategy_parameter_tester.run_parameter_sweep(
        strategy_type=strategy_type,
        optimization_method=method_enum,
        n_iterations=n_iterations,
        n_samples_per_param=n_samples
    )

    if not results:
        print("❌ No results generated")
        return

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate report
    report_path = f"{output_dir}/parameter_optimization_{strategy_type}_{timestamp}.md"
    report = strategy_parameter_tester.generate_optimization_report(results, report_path)

    # Generate plots
    plot_path = f"{output_dir}/parameter_optimization_{strategy_type}_{timestamp}.png"
    strategy_parameter_tester.plot_optimization_results(results, plot_path)

    # Save raw results
    results_path = f"{output_dir}/parameter_optimization_{strategy_type}_{timestamp}.json"
    results_data = [
        {
            "strategy_id": r.strategy_id,
            "parameters": r.parameters,
            "metrics": r.metrics,
            "score": r.score,
            "timestamp": r.timestamp.isoformat(),
            "test_duration": r.test_duration,
            "trade_count": r.trade_count
        }
        for r in results
    ]

    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print("\n✅ Optimization completed!")
    print(f"Results: {len(results)} parameter combinations tested")
    print(f"Best Score: {results[0].score:.4f}")
    print(f"Report: {report_path}")
    print(f"Plots: {plot_path}")
    print(f"Raw Data: {results_path}")


async def run_comparative_analysis(strategy_type: str,
                                 parameter_sets_file: str,
                                 n_runs: int,
                                 output_dir: str):
    """Run comparative analysis of parameter sets"""

    print(f"🔬 Running comparative analysis for {strategy_type}")

    # Load parameter sets
    with open(parameter_sets_file, 'r') as f:
        parameter_sets = json.load(f)

    await initialize_strategy_parameter_testing()

    # Run comparative analysis
    results = await strategy_parameter_tester.run_comparative_analysis(
        strategy_type=strategy_type,
        parameter_sets=parameter_sets,
        n_runs_per_set=n_runs
    )

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results
    output_path = f"{output_dir}/comparative_analysis_{strategy_type}_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ Comparative analysis completed!")
    print(f"Results saved to: {output_path}")


async def run_parameter_sensitivity_analysis(strategy_type: str,
                                          n_samples: int,
                                          output_dir: str):
    """Run parameter sensitivity analysis"""

    print(f"🔬 Running parameter sensitivity analysis for {strategy_type}")

    await initialize_strategy_parameter_testing()

    # Run optimization to get data
    results = await strategy_parameter_tester.run_parameter_sweep(
        strategy_type=strategy_type,
        optimization_method=OptimizationMethod.RANDOM_SEARCH,
        n_iterations=n_samples
    )

    # Generate sensitivity report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/sensitivity_analysis_{strategy_type}_{timestamp}.md"

    sensitivity = strategy_parameter_tester._analyze_parameter_sensitivity(results)

    report = ["# Parameter Sensitivity Analysis", f"Strategy: {strategy_type}", f"Generated: {datetime.now()}", ""]
    report.append("## Sensitivity Results")
    for param, analysis in sensitivity.items():
        report.append(f"- **{param}**: {analysis}")
    report.append("")
    report.append("## Methodology")
    report.append("Sensitivity measured by correlation between parameter values and optimization scores.")

    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print("\n✅ Sensitivity analysis completed!")
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimization Runner")
    parser.add_argument("command", choices=["optimize", "compare", "sensitivity"],
                       help="Command to run")
    parser.add_argument("--strategy-type", required=True,
                       choices=["statistical_arbitrage", "triangular_arbitrage",
                               "cross_exchange_arbitrage", "ml_enhanced_arbitrage"],
                       help="Strategy type to optimize")
    parser.add_argument("--method", default="grid_search",
                       choices=["grid_search", "random_search"],
                       help="Optimization method")
    parser.add_argument("--n-iterations", type=int, default=50,
                       help="Number of iterations/samples")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of samples per parameter (grid search)")
    parser.add_argument("--n-runs", type=int, default=3,
                       help="Number of runs per parameter set (comparative)")
    parser.add_argument("--parameter-sets", help="JSON file with parameter sets (comparative)")
    parser.add_argument("--output-dir", default="reports/rnd",
                       help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.command == "optimize":
        asyncio.run(run_optimization(
            args.strategy_type, args.method, args.n_iterations,
            args.n_samples, args.output_dir
        ))
    elif args.command == "compare":
        if not args.parameter_sets:
            print("❌ --parameter-sets required for comparative analysis")
            sys.exit(1)
        asyncio.run(run_comparative_analysis(
            args.strategy_type, args.parameter_sets, args.n_runs, args.output_dir
        ))
    elif args.command == "sensitivity":
        asyncio.run(run_parameter_sensitivity_analysis(
            args.strategy_type, args.n_iterations, args.output_dir
        ))


if __name__ == "__main__":
    main()
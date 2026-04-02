#!/usr/bin/env python3
"""
Strategy R&D Runner
==================
Automated R&D experiment execution and result analysis.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy_rnd_dashboard import rnd_dashboard

from shared.strategy_parameter_tester import OptimizationMethod


async def run_quick_rnd(strategy_type: str = "statistical_arbitrage", output_dir: str = "reports/rnd"):
    """Run quick R&D test for development"""

    logger.info("🔬 Running Quick R&D Test")
    logger.info("=" * 40)

    await rnd_dashboard.initialize()

    experiment_name = f"quick_rnd_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = await rnd_dashboard.run_comprehensive_experiment(
        experiment_name=experiment_name,
        strategy_types=[strategy_type],
        methods=[OptimizationMethod.GRID_SEARCH],
        n_iterations=25
    )

    # Generate outputs
    rnd_dashboard.generate_experiment_report(experiment_name, output_dir)
    rnd_dashboard.create_visualization_dashboard(experiment_name, output_dir)

    logger.info("\n✅ Quick R&D test completed!")
    logger.info(f"Strategy: {strategy_type}")
    logger.info(f"Best Score: {results['comparative_analysis']['best_performers'][strategy_type]['best_score']:.4f}")

    return results


async def run_comprehensive_rnd(output_dir: str = "reports/rnd"):
    """Run comprehensive R&D across all strategies"""

    logger.info("🔬 Running Comprehensive R&D Suite")
    logger.info("=" * 50)

    await rnd_dashboard.initialize()

    experiment_name = f"comprehensive_rnd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    strategy_types = ["statistical_arbitrage", "triangular_arbitrage", "cross_exchange_arbitrage"]
    methods = [OptimizationMethod.GRID_SEARCH, OptimizationMethod.RANDOM_SEARCH]

    results = await rnd_dashboard.run_comprehensive_experiment(
        experiment_name=experiment_name,
        strategy_types=strategy_types,
        methods=methods,
        n_iterations=50
    )

    # Generate outputs
    rnd_dashboard.generate_experiment_report(experiment_name, output_dir)
    rnd_dashboard.create_visualization_dashboard(experiment_name, output_dir)

    logger.info("\n✅ Comprehensive R&D completed!")
    logger.info(f"Strategies tested: {len(strategy_types)}")
    logger.info(f"Methods tested: {len(methods)}")
    logger.info(f"Total experiments: {len(strategy_types) * len(methods)}")

    # Print top performers
    logger.info("\n🎯 Top Performing Strategies:")
    for strategy, data in results['comparative_analysis']['best_performers'].items():
        logger.info(f"  {strategy}: {data['best_score']:.4f} ({data['best_method']})")

    return results


async def run_ab_test(strategy_type: str, param_file: str, output_dir: str = "reports/rnd"):
    """Run A/B test between parameter sets"""

    logger.info("🔬 Running A/B Parameter Test")
    logger.info("=" * 35)

    # Load parameter sets
    with open(param_file, 'r') as f:
        ab_config = json.load(f)

    await rnd_dashboard.initialize()

    results = await rnd_dashboard.run_ab_test(
        strategy_type=strategy_type,
        param_set_a=ab_config['variant_a'],
        param_set_b=ab_config['variant_b'],
        n_runs=30
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/ab_test_{strategy_type}_{timestamp}.json"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n✅ A/B test completed!")
    logger.info(f"Strategy: {strategy_type}")
    logger.info(f"Variant A mean: {results['variant_a']['mean_score']:.4f}")
    logger.info(f"Variant B mean: {results['variant_b']['mean_score']:.4f}")
    logger.info(f"Winner: {results['statistical_test']['winner']}")
    logger.info(f"Statistically significant: {results['statistical_test']['significant']}")
    logger.info(f"Results saved to: {output_file}")

    return results


async def run_parameter_sensitivity(strategy_type: str, output_dir: str = "reports/rnd"):
    """Run parameter sensitivity analysis"""

    logger.info("🔬 Running Parameter Sensitivity Analysis")
    logger.info("=" * 45)

    from shared.strategy_parameter_tester import (
        initialize_strategy_parameter_testing,
        strategy_parameter_tester,
    )
    await initialize_strategy_parameter_testing()

    # Run optimization to gather data
    results = await strategy_parameter_tester.run_parameter_sweep(
        strategy_type=strategy_type,
        optimization_method=OptimizationMethod.RANDOM_SEARCH,
        n_iterations=100
    )

    # Generate sensitivity report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/sensitivity_{strategy_type}_{timestamp}.md"

    sensitivity = strategy_parameter_tester._analyze_parameter_sensitivity(results)

    report = ["# Parameter Sensitivity Analysis", f"Strategy: {strategy_type}", f"Generated: {datetime.now()}", ""]
    report.append("## Sensitivity Results")
    for param, analysis in sensitivity.items():
        report.append(f"- **{param}**: {analysis}")
    report.append("")
    report.append("## Methodology")
    report.append("- Analysis based on correlation between parameter values and optimization scores")
    report.append("- Higher absolute correlation indicates stronger parameter influence")
    report.append("- Positive correlation: higher parameter values improve performance")
    report.append("- Negative correlation: lower parameter values improve performance")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    logger.info("\n✅ Sensitivity analysis completed!")
    logger.info(f"Strategy: {strategy_type}")
    logger.info(f"Parameters analyzed: {len(sensitivity)}")
    logger.info(f"Report saved to: {report_path}")

    # Print top insights
    logger.info("\n🎯 Key Insights:")
    sorted_sensitivity = sorted(sensitivity.items(),
                               key=lambda x: abs(float(x[1].split()[-1].strip(')'))) if 'correlation' in x[1] else 0,
                               reverse=True)
    for param, analysis in sorted_sensitivity[:3]:
        logger.info(f"  {param}: {analysis}")

    return sensitivity


async def run_custom_experiment(config_file: str, output_dir: str = "reports/rnd"):
    """Run custom experiment from configuration file"""

    logger.info("🔬 Running Custom R&D Experiment")
    logger.info("=" * 40)

    # Load experiment configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    await rnd_dashboard.initialize()

    experiment_name = config.get('experiment_name', f"custom_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Convert method strings to enums
    methods = [OptimizationMethod[m.upper()] for m in config['methods']]

    results = await rnd_dashboard.run_comprehensive_experiment(
        experiment_name=experiment_name,
        strategy_types=config['strategy_types'],
        methods=methods,
        n_iterations=config.get('iterations', 50)
    )

    # Generate outputs
    rnd_dashboard.generate_experiment_report(experiment_name, output_dir)
    rnd_dashboard.create_visualization_dashboard(experiment_name, output_dir)

    logger.info("\n✅ Custom experiment completed!")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Configuration: {config_file}")

    return results


def main():
    """Main."""
    parser = argparse.ArgumentParser(description="Strategy R&D Experiment Runner")
    parser.add_argument("experiment_type", choices=["quick", "comprehensive", "ab_test", "sensitivity", "custom"],
                       help="Type of R&D experiment to run")
    parser.add_argument("--strategy-type", default="statistical_arbitrage",
                       choices=["statistical_arbitrage", "triangular_arbitrage",
                               "cross_exchange_arbitrage", "ml_enhanced_arbitrage"],
                       help="Strategy type for single-strategy experiments")
    parser.add_argument("--config-file", help="Configuration file for custom experiments")
    parser.add_argument("--param-file", help="Parameter file for A/B tests (JSON)")
    parser.add_argument("--output-dir", default="reports/rnd",
                       help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.experiment_type == "quick":
        asyncio.run(run_quick_rnd(args.strategy_type, args.output_dir))
    elif args.experiment_type == "comprehensive":
        asyncio.run(run_comprehensive_rnd(args.output_dir))
    elif args.experiment_type == "ab_test":
        if not args.param_file:
            logger.info("❌ --param-file required for A/B tests")
            sys.exit(1)
        asyncio.run(run_ab_test(args.strategy_type, args.param_file, args.output_dir))
    elif args.experiment_type == "sensitivity":
        asyncio.run(run_parameter_sensitivity(args.strategy_type, args.output_dir))
    elif args.experiment_type == "custom":
        if not args.config_file:
            logger.info("❌ --config-file required for custom experiments")
            sys.exit(1)
        asyncio.run(run_custom_experiment(args.config_file, args.output_dir))


if __name__ == "__main__":
    main()

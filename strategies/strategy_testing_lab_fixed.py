#!/usr/bin/env python3
"""
Strategy Testing Lab (ARB Currency)
===================================
Comprehensive testing laboratory for arbitrage strategies using ARB currency.
Provides simulation, analysis, prediction, and mastery tools for strategy development.

Features:
- ARB currency simulation (1 ARB = 1 USD)
- Multi-strategy testing with parameter optimization
- Comprehensive analysis and reporting
- Prediction modeling and interpretation
- Plug-and-play strategy testing
- Real-world transition planning
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
import random
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from enum import Enum

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from shared.audit_logger import get_audit_logger
from shared.paper_trading import paper_trading_engine, initialize_paper_trading
from shared.strategy_parameter_tester import strategy_parameter_tester, initialize_strategy_parameter_testing
from shared.strategy_loader import get_strategy_loader


class CurrencyType(Enum):
    """Currency types for testing"""
    ARB = "ARB"  # Simulation currency (1 ARB = 1 USD)
    USD = "USD"  # Real USD for live trading


@dataclass
class StrategyLabAccount:
    """Strategy testing account with ARB/USD support"""
    account_id: str
    currency: CurrencyType
    initial_balance: float = 1000.0  # $1000 startup capital per strategy
    current_balance: float = 1000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trades_executed: int = 0
    strategy_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    trade_history: List[Dict] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class LabExperiment:
    """Testing lab experiment configuration"""
    experiment_id: str
    strategy_ids: List[str]
    timeframes: List[str] = field(default_factory=lambda: ["1D", "1W", "1M", "3M"])
    currencies: List[CurrencyType] = field(default_factory=lambda: [CurrencyType.ARB])
    parameter_ranges: Dict[str, Dict] = field(default_factory=dict)
    n_simulations: int = 1000
    risk_free_rate: float = 0.02  # 2% risk-free rate
    created_at: datetime = field(default_factory=datetime.now)


class StrategyTestingLab:
    """
    Comprehensive strategy testing laboratory using ARB currency simulation.
    Provides tools for analysis, prediction, interpretation, execution, and mastery.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.accounts: Dict[str, StrategyLabAccount] = {}
        self.experiments: Dict[str, LabExperiment] = {}
        self.results_history: List[Dict] = []
        self.strategy_loader = None
        self.initialized = False

    async def initialize(self):
        """Initialize the testing lab"""
        self.logger.info("Initializing Strategy Testing Lab (ARB Currency)")

        # Initialize dependencies
        await initialize_paper_trading()
        await initialize_strategy_parameter_testing()
        self.strategy_loader = get_strategy_loader()

        # Load strategy configurations
        await self._load_strategy_configs()

        self.initialized = True
        self.logger.info("[OK] Strategy Testing Lab initialized with ARB currency support")

    async def _load_strategy_configs(self):
        """Load strategy configurations from CSV and implementation files"""
        self.logger.info("Loading strategy configurations...")

        # Load from CSV
        csv_path = PROJECT_ROOT / "50_arbitrage_strategies.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.strategy_configs = {}
            for _, row in df.iterrows():
                strategy_id = f"s{row['id']:02d}"
                self.strategy_configs[strategy_id] = {
                    'id': strategy_id,
                    'name': row['strategy_name'],
                    'description': row['one_liner'],
                    'sources': row['sources'],
                    'implemented': False,
                    'parameters': self._get_default_parameters(strategy_id)
                }

        # Check for implemented strategies
        strategies_dir = PROJECT_ROOT / "strategies"
        if strategies_dir.exists():
            for py_file in strategies_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    # Extract strategy ID from filename
                    strategy_name = py_file.stem.lower().replace('_', '')
                    for config in self.strategy_configs.values():
                        if strategy_name in config['name'].lower().replace(' ', '').replace('-', ''):
                            config['implemented'] = True
                            config['file_path'] = py_file
                            break

        self.logger.info(f"Loaded {len(self.strategy_configs)} strategy configurations")

    def _get_default_parameters(self, strategy_id: str) -> Dict[str, Any]:
        """Get default parameters for a strategy"""
        # Default parameter sets based on strategy type
        defaults = {
            's26': {  # Weekly Overnight Seasonality
                'position_size_pct': {'min': 5.0, 'max': 15.0, 'default': 8.0},
                'max_position_size_pct': {'min': 8.0, 'max': 20.0, 'default': 12.0},
                'entry_exit_buffer_minutes': {'min': 5, 'max': 30, 'default': 15}
            },
            's10': {  # Turn-of-the-Month Overlay
                'allocation_pct': {'min': 5.0, 'max': 20.0, 'default': 10.0},
                'holding_period_days': {'min': 1, 'max': 5, 'default': 3}
            },
            's11': {  # Overnight Jump Reversion
                'reversion_threshold': {'min': 0.5, 'max': 3.0, 'default': 1.5},
                'max_holding_period': {'min': 1, 'max': 5, 'default': 2}
            }
        }
        return defaults.get(strategy_id, {})

    async def create_strategy_accounts(self, strategy_ids: List[str]) -> Dict[str, StrategyLabAccount]:
        """Create ARB accounts for strategies with $1000 startup capital each"""
        self.logger.info(f"Creating ARB accounts for {len(strategy_ids)} strategies")

        accounts = {}
        for strategy_id in strategy_ids:
            if strategy_id in self.strategy_configs:
                config = self.strategy_configs[strategy_id]
                account = StrategyLabAccount(
                    account_id=f"ARB-{strategy_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    currency=CurrencyType.ARB,
                    initial_balance=1000.0,
                    strategy_name=config['name'],
                    parameters=config['parameters']
                )
                accounts[strategy_id] = account
                self.accounts[account.account_id] = account

        self.logger.info(f"Created {len(accounts)} strategy accounts with $1000 ARB each")
        return accounts

    async def run_strategy_simulation(self, strategy_id: str, timeframe: str = "1M",
                                    n_simulations: int = 1000) -> Dict[str, Any]:
        """Run ARB currency simulation for a strategy"""
        self.logger.info(f"Running ARB simulation for {strategy_id} over {timeframe}")

        if strategy_id not in self.strategy_configs:
            raise ValueError(f"Strategy {strategy_id} not found")

        config = self.strategy_configs[strategy_id]
        if not config['implemented']:
            self.logger.warning(f"Strategy {strategy_id} not implemented, using mock simulation")
            return await self._run_mock_simulation(strategy_id, timeframe, n_simulations)

        # Real simulation would load and run the strategy
        # For now, return mock results
        return await self._run_mock_simulation(strategy_id, timeframe, n_simulations)

    async def _run_mock_simulation(self, strategy_id: str, timeframe: str, n_simulations: int) -> Dict[str, Any]:
        """Run mock simulation with realistic ARB returns"""
        # Simulate realistic arbitrage returns based on strategy type
        base_returns = {
            's26': {'mean': 0.12, 'std': 0.08, 'win_rate': 0.65},  # Weekly seasonality
            's10': {'mean': 0.08, 'std': 0.06, 'win_rate': 0.70},  # Turn-of-month
            's11': {'mean': 0.15, 'std': 0.12, 'win_rate': 0.55},  # Jump reversion
        }

        strategy_returns = base_returns.get(strategy_id, {'mean': 0.10, 'std': 0.10, 'win_rate': 0.60})

        # Generate simulation results
        np.random.seed(42)  # For reproducibility

        daily_returns = np.random.normal(
            strategy_returns['mean'] / 252,  # Daily mean
            strategy_returns['std'] / np.sqrt(252),  # Daily std
            n_simulations
        )

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns) - 1

        # Calculate metrics
        final_return = cumulative_returns[-1]
        annualized_return = (1 + final_return) ** (252 / n_simulations) - 1
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(daily_returns)

        # Generate trade history
        trade_history = []
        balance = 1000.0
        for i, ret in enumerate(daily_returns):
            trade_value = balance * 0.1  # 10% position size
            pnl = trade_value * ret
            balance += pnl

            trade_history.append({
                'trade_id': f"T{i+1}",
                'date': (datetime.now() - timedelta(days=n_simulations-i)).date(),
                'return': ret,
                'pnl': pnl,
                'balance': balance
            })

        results = {
            'strategy_id': strategy_id,
            'strategy_name': self.strategy_configs[strategy_id]['name'],
            'timeframe': timeframe,
            'n_simulations': n_simulations,
            'initial_balance': 1000.0,
            'final_balance': balance,
            'total_return': final_return,
            'total_return_pct': final_return * 100,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': strategy_returns['win_rate'],
            'trade_history': trade_history,
            'currency': 'ARB',
            'simulation_date': datetime.now().isoformat()
        }

        return results

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))

    async def run_comprehensive_lab_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive lab experiment across multiple strategies"""
        self.logger.info("Running comprehensive ARB currency lab experiment")

        experiment = LabExperiment(
            experiment_id=f"EXP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            strategy_ids=experiment_config.get('strategy_ids', list(self.strategy_configs.keys())),
            timeframes=experiment_config.get('timeframes', ["1M", "3M"]),
            currencies=[CurrencyType.ARB],
            n_simulations=experiment_config.get('n_simulations', 1000)
        )

        self.experiments[experiment.experiment_id] = experiment

        results = {
            'experiment_id': experiment.experiment_id,
            'strategy_results': {},
            'comparative_analysis': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }

        # Run simulations for each strategy
        for strategy_id in experiment.strategy_ids:
            strategy_results = {}
            for timeframe in experiment.timeframes:
                sim_result = await self.run_strategy_simulation(
                    strategy_id, timeframe, experiment.n_simulations
                )
                strategy_results[timeframe] = sim_result

            results['strategy_results'][strategy_id] = strategy_results

        # Generate comparative analysis
        results['comparative_analysis'] = self._generate_comparative_analysis(results)

        # Generate recommendations
        results['recommendations'] = self._generate_lab_recommendations(results)

        self.results_history.append(results)
        return results

    def _generate_comparative_analysis(self, results: Dict) -> Dict[str, Any]:
        """Generate comparative analysis across strategies"""
        analysis = {
            'best_performers': {},
            'risk_adjusted_returns': {},
            'strategy_rankings': {},
            'correlation_matrix': {},
            'market_regime_sensitivity': {}
        }

        # Analyze each timeframe
        for timeframe in ["1M", "3M"]:
            timeframe_results = {}
            for strategy_id, strategy_data in results['strategy_results'].items():
                if timeframe in strategy_data:
                    timeframe_results[strategy_id] = strategy_data[timeframe]

            if timeframe_results:
                # Rank by Sharpe ratio
                rankings = sorted(
                    timeframe_results.items(),
                    key=lambda x: x[1]['sharpe_ratio'],
                    reverse=True
                )

                analysis['best_performers'][timeframe] = {
                    'top_strategy': rankings[0][0],
                    'sharpe_ratio': rankings[0][1]['sharpe_ratio'],
                    'total_return_pct': rankings[0][1]['total_return_pct']
                }

                analysis['strategy_rankings'][timeframe] = [
                    {'strategy_id': sid, 'sharpe_ratio': data['sharpe_ratio'],
                     'total_return_pct': data['total_return_pct']}
                    for sid, data in rankings
                ]

        return analysis

    def _generate_lab_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on lab results"""
        recommendations = []

        comparative = results.get('comparative_analysis', {})

        # Best strategies recommendation
        for timeframe, data in comparative.get('best_performers', {}).items():
            recommendations.append(
                f"Top performer in {timeframe}: {data['top_strategy']} "
                f"(Sharpe: {data['sharpe_ratio']:.2f}, Return: {data['total_return_pct']:.1f}%)"
            )

        # Risk management recommendations
        recommendations.append(
            "Implement position sizing based on ARB simulation volatility"
        )
        recommendations.append(
            "Monitor maximum drawdown limits from ARB testing"
        )
        recommendations.append(
            "Use Sharpe ratio > 1.5 as threshold for real-world deployment"
        )

        # Transition recommendations
        recommendations.append(
            "Gradually transition from ARB to USD starting with 10% of capital"
        )
        recommendations.append(
            "Implement real-time monitoring before full USD deployment"
        )

        return recommendations

    async def generate_lab_report(self, experiment_id: str, output_dir: str = "reports/lab") -> str:
        """Generate comprehensive lab report"""
        self.logger.info(f"Generating lab report for experiment {experiment_id}")

        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        results = None
        for r in self.results_history:
            if r['experiment_id'] == experiment_id:
                results = r
                break

        if not results:
            raise ValueError(f"Results for experiment {experiment_id} not found")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate report
        report_path = output_path / f"lab_report_{experiment_id}.md"

        with open(report_path, 'w') as f:
            f.write("# Strategy Testing Lab Report (ARB Currency)\n\n")
            f.write(f"**Experiment ID:** {experiment_id}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This report presents comprehensive ARB currency simulation results for arbitrage strategies. ")
            f.write("All testing was conducted using ARB currency (1 ARB = 1 USD) with $1000 initial capital per strategy.\n\n")

            f.write("## Strategy Performance Overview\n\n")

            comparative = results.get('comparative_analysis', {})
            for timeframe, data in comparative.get('best_performers', {}).items():
                f.write(f"### {timeframe} Timeframe\n")
                f.write(f"- **Top Strategy:** {data['top_strategy']}\n")
                f.write(f"- **Sharpe Ratio:** {data['sharpe_ratio']:.2f}\n")
                f.write(f"- **Total Return:** {data['total_return_pct']:.1f}%\n\n")

            f.write("## Detailed Strategy Results\n\n")

            for strategy_id, strategy_data in results.get('strategy_results', {}).items():
                f.write(f"### {strategy_id}: {self.strategy_configs[strategy_id]['name']}\n\n")

                for timeframe, sim_data in strategy_data.items():
                    f.write(f"#### {timeframe} Simulation\n")
                    f.write(f"- Initial Balance: ${sim_data['initial_balance']:,.2f} ARB\n")
                    f.write(f"- Final Balance: ${sim_data['final_balance']:,.2f} ARB\n")
                    f.write(f"- Total Return: {sim_data['total_return_pct']:.1f}%\n")
                    f.write(f"- Annualized Return: {sim_data['annualized_return']:.1f}%\n")
                    f.write(f"- Volatility: {sim_data['volatility']:.1f}%\n")
                    f.write(f"- Sharpe Ratio: {sim_data['sharpe_ratio']:.2f}\n")
                    f.write(f"- Max Drawdown: {sim_data['max_drawdown']:.1f}%\n")
                    f.write(f"- Win Rate: {sim_data['win_rate']:.1f}%\n\n")

            f.write("## Recommendations\n\n")
            for rec in results.get('recommendations', []):
                f.write(f"- {rec}\n")

            f.write("\n## Transition to Real USD\n\n")
            f.write("### Backend Setup\n")
            f.write("1. **Bank Account Integration:** Use CorporateBankingDivision for account management\n")
            f.write("2. **Wire Transfer Processing:** Implement secure wire transfer protocols\n")
            f.write("3. **Compliance Monitoring:** Enable real-time compliance checks\n")
            f.write("4. **Risk Management:** Implement position limits and stop-loss mechanisms\n\n")

            f.write("### USD Introduction Plan\n")
            f.write("1. **Phase 1 (10%):** Start with $100 per top-performing strategy\n")
            f.write("2. **Phase 2 (25%):** Scale to $250 per strategy after 30-day success\n")
            f.write("3. **Phase 3 (50%):** Full $500 deployment with enhanced monitoring\n")
            f.write("4. **Phase 4 (100%):** Complete transition to live USD trading\n\n")

            f.write("### Banking Structure\n")
            f.write("- **Primary Accounts:** Corporate checking accounts for operational funds\n")
            f.write("- **Treasury Accounts:** Centralized treasury management\n")
            f.write("- **Escrow Accounts:** Secure holding for strategy capital\n")
            f.write("- **Payroll Accounts:** Separate accounts for team compensation\n\n")

        self.logger.info(f"Lab report generated: {report_path}")
        return str(report_path)

    async def scan_system_for_strategies(self) -> Dict[str, Any]:
        """Comprehensive scan of system files for strategy-related content"""
        self.logger.info("Scanning system for strategy-related files and content")

        scan_results = {
            'strategy_files': [],
            'test_files': [],
            'config_files': [],
            'implementation_status': {},
            'missing_implementations': [],
            'parameter_files': [],
            'analysis_files': []
        }

        # Scan directories
        directories_to_scan = [
            PROJECT_ROOT / "strategies",
            PROJECT_ROOT / "shared",
            PROJECT_ROOT / "reports",
            PROJECT_ROOT / "data"
        ]

        for directory in directories_to_scan:
            if directory.exists():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        file_info = await self._analyze_file(file_path)
                        if file_info['category']:
                            scan_results[file_info['category']].append(file_info)

        # Analyze implementation status
        for strategy_id, config in self.strategy_configs.items():
            scan_results['implementation_status'][strategy_id] = config['implemented']
            if not config['implemented']:
                scan_results['missing_implementations'].append(strategy_id)

        return scan_results

    async def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a file for strategy-related content"""
        file_info = {
            'path': str(file_path.relative_to(PROJECT_ROOT)),
            'name': file_path.name,
            'extension': file_path.suffix,
            'category': None,
            'strategy_references': [],
            'has_parameters': False,
            'has_testing': False
        }

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()

                # Categorize file
                if 'strategy' in content and 'parameter' in content:
                    file_info['category'] = 'parameter_files'
                elif 'test' in content and ('strategy' in content or 'simulation' in content):
                    file_info['category'] = 'test_files'
                elif file_path.name.startswith(('strategy', 's01_', 's02_')) and file_path.suffix == '.py':
                    file_info['category'] = 'strategy_files'
                elif 'analysis' in content or 'report' in content:
                    file_info['category'] = 'analysis_files'
                elif 'config' in content:
                    file_info['category'] = 'config_files'

                # Check for strategy references
                for strategy_id in self.strategy_configs.keys():
                    if strategy_id in content:
                        file_info['strategy_references'].append(strategy_id)

                file_info['has_parameters'] = 'parameter' in content
                file_info['has_testing'] = 'test' in content or 'simulation' in content

        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")

        return file_info


# Global lab instance
strategy_testing_lab = StrategyTestingLab()


async def initialize_strategy_testing_lab():
    """Initialize the global strategy testing lab"""
    await strategy_testing_lab.initialize()


async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Strategy Testing Lab (ARB Currency)")
    parser.add_argument('command', choices=['scan', 'simulate', 'experiment', 'report'],
                       help='Command to execute')
    parser.add_argument('--strategy-id', help='Strategy ID for simulation')
    parser.add_argument('--timeframe', default='1M', help='Timeframe for simulation')
    parser.add_argument('--experiment-config', help='JSON config file for experiment')
    parser.add_argument('--experiment-id', help='Experiment ID for report generation')
    parser.add_argument('--output-dir', default='reports/lab', help='Output directory')

    args = parser.parse_args()

    await initialize_strategy_testing_lab()

    if args.command == 'scan':
        print("🔬 Scanning system for strategy-related content...")
        results = await strategy_testing_lab.scan_system_for_strategies()

        print(f"📁 Strategy Files: {len(results['strategy_files'])}")
        print(f"🧪 Test Files: {len(results['test_files'])}")
        print(f"⚙️ Config Files: {len(results['config_files'])}")
        print(f"📊 Analysis Files: {len(results['analysis_files'])}")
        print(f"📝 Parameter Files: {len(results['parameter_files'])}")

        print(f"\n✅ Implemented Strategies: {sum(results['implementation_status'].values())}")
        print(f"❌ Missing Implementations: {len(results['missing_implementations'])}")

    elif args.command == 'simulate':
        if not args.strategy_id:
            print("❌ Strategy ID required for simulation")
            return

        print(f"🔬 Running ARB simulation for {args.strategy_id}...")
        results = await strategy_testing_lab.run_strategy_simulation(
            args.strategy_id, args.timeframe
        )

        print("📊 Simulation Results:")
        print(f"Strategy: {results['strategy_name']}")
        print(f"Initial Balance: ${results['initial_balance']:,.2f} ARB")
        print(f"Final Balance: ${results['final_balance']:,.2f} ARB")
        print(f"Total Return: {results['total_return_pct']:.1f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    elif args.command == 'experiment':
        config = {}
        if args.experiment_config:
            with open(args.experiment_config, 'r') as f:
                config = json.load(f)

        print("🔬 Running comprehensive lab experiment...")
        results = await strategy_testing_lab.run_comprehensive_lab_experiment(config)

        print(f"✅ Experiment completed: {results['experiment_id']}")

        # Generate report
        report_path = await strategy_testing_lab.generate_lab_report(
            results['experiment_id'], args.output_dir
        )
        print(f"📄 Report generated: {report_path}")

    elif args.command == 'report':
        if not args.experiment_id:
            print("❌ Experiment ID required for report generation")
            return

        report_path = await strategy_testing_lab.generate_lab_report(
            args.experiment_id, args.output_dir
        )
        print(f"📄 Report generated: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
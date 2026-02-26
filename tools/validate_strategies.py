#!/usr/bin/env python3
"""
Strategy Validation Tool
========================
Command-line tool to validate and analyze the 50 arbitrage strategies.
Provides detailed reporting and automated checking capabilities.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.strategy_loader import get_strategy_loader, StrategyCategory, StrategyStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class StrategyValidator:
    """Command-line tool for strategy validation and analysis"""

    def __init__(self):
        self.loader = get_strategy_loader()

    async def run_validation(self, args):
        """Run the requested validation operation"""
        if args.command == 'check':
            await self.check_strategies()
        elif args.command == 'summary':
            await self.show_summary()
        elif args.command == 'category':
            await self.show_category(args.category)
        elif args.command == 'validate':
            await self.validate_strategies()
        elif args.command == 'export':
            await self.export_strategies(args.format, args.output)
        elif args.command == 'simulate':
            await self.simulate_strategies(args.duration, args.strategies, args.category)
        else:
            logger.error(f"Unknown command: {args.command}")

    async def check_strategies(self):
        """Basic strategy check - loads and validates all strategies"""
        print("🔍 Checking 50 Arbitrage Strategies")
        print("=" * 50)

        try:
            strategies = await self.loader.load_strategies()
            valid = [s for s in strategies if s.is_valid]
            invalid = [s for s in strategies if not s.is_valid]

            print(f"✅ Loaded {len(strategies)} strategies")
            print(f"✅ Valid strategies: {len(valid)}")
            print(f"[CROSS] Invalid strategies: {len(invalid)}")

            if invalid:
                print("\n[CROSS] Invalid Strategies:")
                for strategy in invalid:
                    print(f"  - {strategy.id}: {strategy.name}")
                    for error in strategy.validation_errors:
                        print(f"    • {error}")

            if len(valid) == 50:
                print("\n[CELEBRATION] All 50 strategies are valid and ready for implementation!")
            else:
                print(f"\n[WARN]️  {len(valid)}/50 strategies are valid. Review invalid strategies above.")

        except Exception as e:
            print(f"[CROSS] Error checking strategies: {e}")
            return False

        return True

    async def show_summary(self):
        """Show detailed summary of strategy distribution"""
        print("[MONITOR] Strategy Summary Report")
        print("=" * 50)

        try:
            summary = await self.loader.get_strategy_summary()

            print(f"Total Strategies: {summary['total_strategies']}")
            print(f"Valid Strategies: {summary['valid_strategies']}")
            print(f"Invalid Strategies: {summary['invalid_strategies']}")
            print(".1%")

            print("\n📂 Strategy Categories:")
            for category, count in summary['categories'].items():
                print(f"  {category.replace('_', ' ').title()}: {count}")

        except Exception as e:
            print(f"[CROSS] Error generating summary: {e}")

    async def show_category(self, category_name: str):
        """Show strategies in a specific category"""
        try:
            category = StrategyCategory(category_name.lower().replace(' ', '_'))
        except ValueError:
            print(f"[CROSS] Invalid category: {category_name}")
            print("Available categories:")
            for cat in StrategyCategory:
                print(f"  - {cat.value.replace('_', ' ').title()}")
            return

        print(f"📂 Strategies in {category.value.replace('_', ' ').title()} Category")
        print("=" * 50)

        try:
            strategies = await self.loader.get_strategies_by_category(category)

            if not strategies:
                print("No strategies found in this category.")
                return

            for strategy in strategies:
                status_icon = "✅" if strategy.is_valid else "[CROSS]"
                print(f"{status_icon} {strategy.id}: {strategy.name}")
                print(f"   {strategy.description}")
                if strategy.sources:
                    print(f"   Sources: {', '.join(strategy.sources)}")
                print()

        except Exception as e:
            print(f"[CROSS] Error showing category: {e}")

    async def validate_strategies(self):
        """Run comprehensive validation and show detailed results"""
        print("🔬 Comprehensive Strategy Validation")
        print("=" * 50)

        try:
            results = await self.loader.validate_all_strategies()

            print(f"✅ Passed: {len(results['passed'])} strategies")
            print(f"[CROSS] Failed: {len(results['failed'])} strategies")
            print(f"[WARN]️  Warnings: {len(results['warnings'])} strategies")

            if results['failed']:
                print("\n[CROSS] Failed Strategies:")
                for failure in results['failed']:
                    print(f"\n  Strategy {failure['id']}: {failure['name']}")
                    for error in failure['errors']:
                        print(f"    • {error}")

            if results['passed']:
                print("\n✅ Passed Strategies:")
                for strategy in results['passed'][:10]:  # Show first 10
                    print(f"  • {strategy['id']}: {strategy['name']} ({strategy['category']})")
                if len(results['passed']) > 10:
                    print(f"  ... and {len(results['passed']) - 10} more")

        except Exception as e:
            print(f"[CROSS] Error during validation: {e}")

    async def export_strategies(self, format_type: str, output_file: str):
        """Export strategies in specified format"""
        print(f"📤 Exporting strategies to {output_file} in {format_type} format")

        try:
            strategies = await self.loader.load_strategies()

            if format_type == 'json':
                await self._export_json(strategies, output_file)
            elif format_type == 'csv':
                await self._export_csv(strategies, output_file)
            elif format_type == 'markdown':
                await self._export_markdown(strategies, output_file)
            else:
                print(f"[CROSS] Unsupported format: {format_type}")
                return

            print("✅ Export completed successfully")

        except Exception as e:
            print(f"[CROSS] Error exporting strategies: {e}")

    async def _export_json(self, strategies, output_file: str):
        """Export strategies as JSON"""
        import json

        data = {
            'summary': await self.loader.get_strategy_summary(),
            'strategies': [
                {
                    'id': s.id,
                    'name': s.name,
                    'description': s.description,
                    'category': s.category.value,
                    'sources': s.sources,
                    'status': s.status.value,
                    'validation_errors': s.validation_errors
                }
                for s in strategies
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    async def simulate_strategies(self, duration: int, strategy_ids: Optional[List[str]], category: Optional[str]):
        """Run strategies in simulation mode"""
        print(f"[TARGET] Running Strategy Simulation ({duration}s)")
        print("=" * 50)

        try:
            # Load strategies
            strategies = await self.loader.load_strategies()
            
            # Filter strategies
            if strategy_ids:
                strategies = [s for s in strategies if str(s.id) in strategy_ids]
            elif category:
                from shared.strategy_loader import StrategyCategory
                try:
                    cat_enum = StrategyCategory(category.lower().replace(' ', '_'))
                    strategies = [s for s in strategies if s.category == cat_enum]
                except ValueError:
                    print(f"[CROSS] Invalid category: {category}")
                    return
            
            valid_strategies = [s for s in strategies if s.is_valid]
            
            if not valid_strategies:
                print("[CROSS] No valid strategies found matching criteria")
                return
            
            print(f"Simulating {len(valid_strategies)} strategies...")
            
            # Run simulation
            results = await self._run_simulation(valid_strategies, duration)
            
            # Display results
            self._display_simulation_results(results)
            
        except Exception as e:
            print(f"[CROSS] Error during simulation: {e}")

    async def _run_simulation(self, strategies: List[Any], duration: int) -> Dict[str, Any]:
        """Run the actual simulation"""
        import random
        import time
        
        results = {
            'strategies': {},
            'total_trades': 0,
            'total_pnl': 0.0,
            'duration': duration,
            'start_time': time.time()
        }
        
        # Simulate each strategy
        for strategy in strategies:
            strategy_result = {
                'trades': 0,
                'pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'category': strategy.category.value
            }
            
            # Simple random simulation (in real implementation, this would use actual market data)
            num_trades = random.randint(5, 20)
            wins = 0
            
            for _ in range(num_trades):
                # Simulate trade outcome
                pnl = random.uniform(-100, 200)  # Random P&L between -$100 and $200
                strategy_result['pnl'] += pnl
                strategy_result['trades'] += 1
                
                if pnl > 0:
                    wins += 1
                
                # Track max drawdown (simplified)
                if strategy_result['pnl'] < strategy_result['max_drawdown']:
                    strategy_result['max_drawdown'] = strategy_result['pnl']
            
            strategy_result['win_rate'] = (wins / num_trades) * 100 if num_trades > 0 else 0
            
            results['strategies'][strategy.name] = strategy_result
            results['total_trades'] += num_trades
            results['total_pnl'] += strategy_result['pnl']
        
        # Simulate time passing
        await asyncio.sleep(min(duration, 5))  # Cap at 5 seconds for demo
        
        results['end_time'] = time.time()
        return results

    def _display_simulation_results(self, results: Dict[str, Any]):
        """Display simulation results in a formatted way"""
        print("\n[MONITOR] Simulation Results")
        print(f"Duration: {results['duration']}s")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Total P&L: ${results['total_pnl']:.2f}")
        print()
        
        print("Strategy Performance:")
        print("-" * 80)
        print(f"{'Strategy':<35} {'Trades':<8} {'Win Rate':<10} {'P&L':<12}")
        print("-" * 80)
        
        for strategy_name, data in results['strategies'].items():
            print(f"{strategy_name[:34]:<35} "
                  f"{data['trades']:<8} "
                  f"{data['win_rate']:<10.1f} "
                  f"${data['pnl']:<12.1f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<35} "
              f"{results['total_trades']:<8} "
              f"{'N/A':<10} "
              f"${results['total_pnl']:<12.1f}")
        
        # Category breakdown
        print("\n📂 By Category:")
        categories = {}
        for strategy_name, data in results['strategies'].items():
            cat = data['category']
            if cat not in categories:
                categories[cat] = {'trades': 0, 'pnl': 0.0, 'count': 0}
            categories[cat]['trades'] += data['trades']
            categories[cat]['pnl'] += data['pnl']
            categories[cat]['count'] += 1
        
        for cat, data in categories.items():
            avg_pnl = data['pnl'] / data['count'] if data['count'] > 0 else 0
            print(f"{cat:<15} "
                  f"{data['trades']:<8} "
                  f"${avg_pnl:<10.1f} "
                  f"({data['count']} strategies)")

    async def _export_csv(self, strategies, output_file: str):
        """Export strategies as CSV"""
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name', 'Category', 'Status', 'Description', 'Sources'])

            for strategy in strategies:
                writer.writerow([
                    strategy.id,
                    strategy.name,
                    strategy.category.value,
                    strategy.status.value,
                    strategy.description,
                    ';'.join(strategy.sources)
                ])

    async def _export_json(self, strategies, output_file: str):
        """Export strategies as JSON"""
        import json

        data = {
            'summary': await self.loader.get_strategy_summary(),
            'strategies': [
                {
                    'id': s.id,
                    'name': s.name,
                    'description': s.description,
                    'category': s.category.value,
                    'sources': s.sources,
                    'status': s.status.value,
                    'validation_errors': s.validation_errors
                }
                for s in strategies
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    async def _export_markdown(self, strategies, output_file: str):
        """Export strategies as Markdown"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# ACC Arbitrage Strategies\n\n")
            f.write("## Summary\n\n")

            summary = await self.loader.get_strategy_summary()
            f.write(f"- Total Strategies: {summary['total_strategies']}\n")
            f.write(f"- Valid Strategies: {summary['valid_strategies']}\n")
            f.write(".1%\n\n")

            f.write("## Strategies by Category\n\n")

            for category in StrategyCategory:
                cat_strategies = [s for s in strategies if s.category == category]
                if cat_strategies:
                    f.write(f"### {category.value.replace('_', ' ').title()}\n\n")
                    for strategy in cat_strategies:
                        status = "[PASS]" if strategy.is_valid else "[FAIL]"
                        f.write(f"- {status} **{strategy.name}** (ID: {strategy.id})\n")
                        f.write(f"  - {strategy.description}\n")
                        if strategy.sources:
                            f.write(f"  - Sources: {', '.join(strategy.sources)}\n\n")


def main():
    parser = argparse.ArgumentParser(description="ACC Strategy Validation Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Check command
    subparsers.add_parser('check', help='Basic strategy validation check')

    # Summary command
    subparsers.add_parser('summary', help='Show strategy summary statistics')

    # Category command
    category_parser = subparsers.add_parser('category', help='Show strategies by category')
    category_parser.add_argument('category', help='Strategy category name')

    # Validate command
    subparsers.add_parser('validate', help='Run comprehensive validation')

    # Simulate command
    simulate_parser = subparsers.add_parser('simulate', help='Run strategies in simulation mode')
    simulate_parser.add_argument('--duration', type=int, default=60, help='Simulation duration in seconds')
    simulate_parser.add_argument('--strategies', nargs='*', help='Specific strategy IDs to simulate (default: all)')
    simulate_parser.add_argument('--category', help='Simulate only strategies in this category')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run the validation
    validator = StrategyValidator()
    asyncio.run(validator.run_validation(args))


if __name__ == '__main__':
    main()
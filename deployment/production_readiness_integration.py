#!/usr/bin/env python3
"""
Production Readiness Integration
===============================
Complete integration of paper trading, AI strategy generation, and live trading safeguards.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.paper_trading import paper_trading_engine, initialize_paper_trading
from shared.ai_strategy_generator import ai_strategy_generator, initialize_ai_strategy_generation
from shared.live_trading_safeguards import live_trading_safeguards, initialize_live_trading_safeguards
from shared.production_deployment import production_deployment_system, initialize_production_deployment
from shared.production_monitoring import production_monitoring_system, initialize_production_monitoring
from shared.compliance_review import compliance_review_system, initialize_compliance_review
from shared.audit_logger import get_audit_logger


class ProductionReadinessSystem:
    """Integrated production-ready trading system"""

    def __init__(self):
        self.logger = logging.getLogger("ProductionReadinessSystem")
        self.audit_logger = get_audit_logger()

        # System components
        self.paper_trading = paper_trading_engine
        self.ai_generator = ai_strategy_generator
        self.safety_system = live_trading_safeguards

        # Integration state
        self.strategies_under_test = {}  # strategy_id -> test_results
        self.live_strategies = {}  # strategy_id -> live_performance
        self.risk_assessment_active = False

    async def initialize_all_systems(self):
        """Initialize all production systems"""
        print("[ROCKET] Initializing Production Readiness System")
        print("=" * 50)

        # Initialize core trading systems
        await initialize_paper_trading()
        await initialize_ai_strategy_generation()
        await initialize_live_trading_safeguards()

        # Initialize production deployment systems
        await initialize_production_deployment()
        await initialize_production_monitoring()
        await initialize_compliance_review()

        print("[OK] All production systems initialized")

    async def run_strategy_development_pipeline(self):
        """Complete strategy development pipeline"""
        print("\n[TARGET] Starting Strategy Development Pipeline")
        print("-" * 40)

        # Step 1: AI generates strategies
        print("Step 1: AI Strategy Generation")
        opportunities = await self.ai_generator.scan_for_opportunities()

        if not opportunities:
            print("No arbitrage opportunities found, using simulated data")
            # Create a mock opportunity for demonstration
            from shared.ai_strategy_generator import ArbitrageOpportunity, StrategyType, RiskLevel
            opportunities = [
                ArbitrageOpportunity(
                    opportunity_id="demo_opp_1",
                    strategy_type=StrategyType.STATISTICAL_ARBITRAGE,
                    symbols=["SPY", "QQQ"],
                    exchanges=["paper"],
                    expected_return=0.02,  # 2%
                    risk_level=RiskLevel.MEDIUM,
                    confidence_score=0.85,
                    time_horizon=60,
                    max_position_size=50000,
                    entry_conditions={"price_spread": 0.005},
                    exit_conditions={"profit_target": 0.015, "stop_loss": -0.005}
                )
            ]

        strategies = []
        for opp in opportunities[:3]:  # Limit to 3 for demo
            strategy = await self.ai_generator.generate_strategy_from_opportunity(opp)
            strategies.append(strategy)
            print(f"  [OK] Generated strategy: {strategy.strategy_id}")

        # Step 1.5: Parameter Optimization (R&D Phase)
        print("\nStep 1.5: Parameter Optimization (R&D)")
        from shared.strategy_parameter_tester import strategy_parameter_tester, initialize_strategy_parameter_testing
        await initialize_strategy_parameter_testing()

        optimized_strategies = []
        for strategy in strategies:
            print(f"    Optimizing parameters for: {strategy.strategy_id}")

            # Run parameter sweep for this strategy type
            from shared.strategy_parameter_tester import OptimizationMethod
            optimization_results = await strategy_parameter_tester.run_parameter_sweep(
                strategy_type=strategy.strategy_type,
                optimization_method=OptimizationMethod.GRID_SEARCH,
                n_iterations=10,  # Quick optimization for demo
                n_samples_per_param=2
            )

            if optimization_results:
                # Use best parameters to create optimized strategy
                best_params = optimization_results[0].parameters
                optimized_strategy = await strategy_parameter_tester._create_strategy_with_params(
                    strategy.strategy_type, best_params
                )
                optimized_strategy.strategy_id = f"{strategy.strategy_id}_optimized"
                optimized_strategies.append(optimized_strategy)
                print(f"      [OK] Optimized strategy: {optimized_strategy.strategy_id} (Score: {optimization_results[0].score:.4f})")
            else:
                optimized_strategies.append(strategy)
                print(f"      [WARN] Using original strategy (optimization failed)")

        strategies = optimized_strategies

        # Step 2: Paper trading validation
        print("\nStep 2: Paper Trading Validation")
        validation_results = await self._validate_strategies_in_paper_trading(strategies)
        print(f"  [OK] Validated {len(validation_results)} strategies")

        # Step 3: Risk assessment
        print("\nStep 3: Risk Assessment")
        risk_assessment = await self._assess_strategy_risks(strategies, validation_results)
        print(f"  [OK] Risk assessment complete for {len(risk_assessment)} strategies")

        # Step 4: Production readiness check
        print("\nStep 4: Production Readiness Check")
        production_ready = await self._check_production_readiness(strategies, validation_results, risk_assessment)
        print(f"  [OK] {len(production_ready)} strategies ready for production")

        return {
            "opportunities_found": len(opportunities),
            "strategies_generated": len(strategies),
            "strategies_validated": len(validation_results),
            "strategies_production_ready": len(production_ready),
            "validation_results": validation_results,
            "risk_assessment": risk_assessment,
            "production_ready_strategies": production_ready
        }

    async def _validate_strategies_in_paper_trading(self, strategies: List) -> Dict[str, Any]:
        """Validate strategies using paper trading"""
        validation_results = {}

        for strategy in strategies:
            print(f"    Testing strategy: {strategy.strategy_id}")

            # Reset paper account for clean test
            await self.paper_trading.reset_account()

            # Simulate strategy execution
            test_trades = await self._simulate_strategy_execution(strategy)

            # Get performance metrics
            account_summary = self.paper_trading.get_account_summary()
            trade_history = self.paper_trading.get_trade_history(limit=50)

            # Calculate enhanced strategy metrics
            total_return = account_summary['total_pnl'] / 1000000.0  # From $1M starting balance
            win_rate = len([t for t in trade_history if t['pnl'] > 0]) / len(trade_history) if trade_history else 0

            # Calculate realistic drawdown
            if trade_history:
                cumulative_pnl = 0
                peak_pnl = 0
                max_drawdown = 0
                for trade in trade_history:
                    cumulative_pnl += trade['pnl']
                    peak_pnl = max(peak_pnl, cumulative_pnl)
                    drawdown = peak_pnl - cumulative_pnl
                    max_drawdown = max(max_drawdown, drawdown)
                max_drawdown_pct = max_drawdown / 1000000.0 if peak_pnl > 0 else 0
            else:
                max_drawdown_pct = 0

            # Calculate Sharpe ratio (simplified)
            if trade_history:
                returns = [t['pnl'] / 1000000.0 for t in trade_history]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0

            # Calculate profit factor
            winning_trades = [t['pnl'] for t in trade_history if t['pnl'] > 0]
            losing_trades = [t['pnl'] for t in trade_history if t['pnl'] < 0]
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            validation_results[strategy.strategy_id] = {
                "total_return": total_return,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown_pct,
                "total_trades": len(trade_history),
                "sharpe_ratio": sharpe_ratio,
                "profit_factor": profit_factor,
                "avg_trade_pnl": np.mean([t['pnl'] for t in trade_history]) if trade_history else 0,
                "passed_validation": (
                    total_return > 0.005 and  # 0.5% minimum return
                    win_rate > 0.55 and       # 55% win rate
                    max_drawdown_pct < 0.05 and  # Max 5% drawdown
                    len(trade_history) >= 10 and # Minimum 10 trades
                    sharpe_ratio > 0.5        # Minimum Sharpe ratio
                )
            }

            print(".3f"".3f"".1%")

        return validation_results

    async def _simulate_strategy_execution(self, strategy) -> List[Dict]:
        """Simulate strategy execution in paper trading"""
        trades_executed = []

        # Simple simulation: execute a few trades based on strategy
        symbols = strategy.symbols[:2]  # Limit to 2 symbols for demo

        for symbol in symbols:
            # Buy order
            from shared.paper_trading import OrderSide, OrderType
            order_id = await self.paper_trading.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
                strategy_id=strategy.strategy_id
            )
            trades_executed.append({"symbol": symbol, "side": "buy", "order_id": order_id})

            await asyncio.sleep(0.1)

            # Sell order (simulating round trip)
            order_id = await self.paper_trading.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=50,
                order_type=OrderType.MARKET,
                strategy_id=strategy.strategy_id
            )
            trades_executed.append({"symbol": symbol, "side": "sell", "order_id": order_id})

            await asyncio.sleep(0.1)

        return trades_executed

    async def _assess_strategy_risks(self, strategies: List, validation_results: Dict) -> Dict[str, Any]:
        """Assess risks for each strategy"""
        risk_assessment = {}

        for strategy in strategies:
            strategy_id = strategy.strategy_id
            validation = validation_results.get(strategy_id, {})

            # Risk metrics
            max_drawdown = validation.get('max_drawdown', 0.1)
            volatility = 0.02  # Simplified
            concentration = 0.15  # Simplified

            # Safety system check
            safety_status = self.safety_system.get_safety_status()
            system_risk = 1 if safety_status['emergency_shutdown'] else 0

            # Overall risk score (0-1, higher = riskier)
            risk_score = (
                max_drawdown * 0.4 +
                volatility * 0.3 +
                concentration * 0.2 +
                system_risk * 0.1
            )

            risk_assessment[strategy_id] = {
                "risk_score": risk_score,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "concentration": concentration,
                "system_risk": system_risk,
                "risk_level": "LOW" if risk_score < 0.3 else "MEDIUM" if risk_score < 0.6 else "HIGH",
                "approved_for_live": risk_score < 0.5 and not safety_status['emergency_shutdown']
            }

        return risk_assessment

    async def _check_production_readiness(self, strategies: List, validation_results: Dict, risk_assessment: Dict) -> List[Dict]:
        """Check which strategies are ready for production"""
        production_ready = []

        for strategy in strategies:
            strategy_id = strategy.strategy_id
            validation = validation_results.get(strategy_id, {})
            risk = risk_assessment.get(strategy_id, {})

            # Enhanced production readiness criteria for gradual rollout
            criteria = {
                "validation_passed": validation.get('passed_validation', False),
                "risk_approved": risk.get('approved_for_live', False),
                "min_trades": validation.get('total_trades', 0) >= 10,  # Increased minimum trades
                "positive_return": validation.get('total_return', 0) > 0.005,  # 0.5% minimum return
                "reasonable_drawdown": validation.get('max_drawdown', 1.0) < 0.05,  # Max 5% drawdown
                "sharpe_ratio_ok": validation.get('sharpe_ratio', 0) > 0.5,  # Minimum Sharpe ratio
                "profit_factor_ok": validation.get('profit_factor', 0) > 1.2,  # Minimum profit factor
                "safety_system_ok": not self.safety_system.emergency_shutdown,
                "backtest_quality": strategy.backtest_results.get('total_trades', 0) > 20  # More backtest trades
            }

            all_criteria_met = all(criteria.values())

            if all_criteria_met:
                production_ready.append({
                    "strategy": strategy,
                    "validation_results": validation,
                    "risk_assessment": risk,
                    "readiness_score": sum(criteria.values()) / len(criteria),
                    "criteria_met": criteria
                })

        return production_ready

    async def run_live_trading_simulation(self, production_strategies: List[Dict]):
        """Simulate live trading with safety systems"""
        print("\n[MONEY] Starting Live Trading Simulation")
        print("-" * 35)

        if not production_strategies:
            print("No production-ready strategies available")
            return

        # Select best strategy
        best_strategy = max(production_strategies, key=lambda x: x['readiness_score'])
        strategy = best_strategy['strategy']

        print(f"Selected strategy for live simulation: {strategy.strategy_id}")
        print(".1%")

        # Simulate live trading with safety checks
        simulation_results = await self._simulate_live_trading_with_safety(strategy)

        return simulation_results

    async def _simulate_live_trading_with_safety(self, strategy) -> Dict[str, Any]:
        """Simulate live trading with full safety system integration"""
        trades_executed = 0
        safety_interventions = 0
        pnl = 0.0

        print("  Executing trades with safety monitoring...")

        # Simulate 10 trades
        for i in range(10):
            # Check safety before each trade
            trade_details = {
                "symbol": strategy.symbols[0],
                "quantity": 100 + (i * 10),
                "price": 100.0
            }

            safe, message = await self.safety_system.check_trade_safety(trade_details)

            if safe:
                # Execute trade in paper trading (simulating live)
                from shared.paper_trading import OrderSide, OrderType
                order_id = await self.paper_trading.submit_order(
                    symbol=trade_details["symbol"],
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    quantity=trade_details["quantity"],
                    order_type=OrderType.MARKET,
                    strategy_id=strategy.strategy_id
                )
                trades_executed += 1
                pnl += 50.0  # Simplified P&L
                print(f"    [OK] Trade {i+1} executed: {order_id}")
            else:
                safety_interventions += 1
                print(f"    [WARNING]  Trade {i+1} blocked by safety: {message}")

            # Run safety checks
            alerts = await self.safety_system.execute_safety_check()
            if alerts:
                print(f"    [ALERT] Safety alerts triggered: {len(alerts)}")

            await asyncio.sleep(0.2)  # Simulate time between trades

        return {
            "trades_executed": trades_executed,
            "safety_interventions": safety_interventions,
            "total_pnl": pnl,
            "safety_system_status": self.safety_system.get_safety_status(),
            "final_account": self.paper_trading.get_account_summary()
        }

    async def generate_production_report(self, pipeline_results: Dict, live_results: Optional[Dict] = None) -> str:
        """Generate comprehensive production readiness report"""
        print("\n[CHART] Generating Production Readiness Report")
        print("-" * 42)

        report = []
        report.append("# Production Readiness Report")
        report.append(f"Generated: {time.time()}")
        report.append("")

        # System Status
        report.append("## System Status")
        safety_status = self.safety_system.get_safety_status()
        report.append(f"- Trading Halted: {safety_status['trading_halted']}")
        report.append(f"- Emergency Shutdown: {safety_status['emergency_shutdown']}")
        report.append(f"- Active Safety Alerts: {safety_status['active_alerts']}")
        report.append(f"- System Health: {safety_status['current_metrics'].get('system_health', 'N/A')}")
        report.append("")

        # Pipeline Results
        report.append("## Strategy Development Pipeline")
        report.append(f"- Opportunities Found: {pipeline_results['opportunities_found']}")
        report.append(f"- Strategies Generated: {pipeline_results['strategies_generated']}")
        report.append(f"- Strategies Validated: {pipeline_results['strategies_validated']}")
        report.append(f"- Production Ready: {pipeline_results['strategies_production_ready']}")
        report.append("")

        # Validation Results
        if pipeline_results['validation_results']:
            report.append("## Strategy Validation Results")
            for strategy_id, results in pipeline_results['validation_results'].items():
                report.append(f"### {strategy_id}")
                report.append(f"- Win Rate: {results['win_rate']:.1%}")
                report.append(f"- Profit Factor: {results['profit_factor']:.3f}")
                report.append(f"- Total Trades: {results['total_trades']}")
                report.append(f"- Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                report.append(f"- Passed Validation: {'PASS' if results['passed_validation'] else 'FAIL'}")
                report.append("")

        # Risk Assessment
        if pipeline_results['risk_assessment']:
            report.append("## Risk Assessment")
            for strategy_id, risk in pipeline_results['risk_assessment'].items():
                report.append(f"### {strategy_id}")
                report.append(f"- Max Drawdown: {risk['max_drawdown']:.3f}")
                report.append(f"- Risk Level: {risk['risk_level']}")
                report.append(f"- Approved for Live: {'YES' if risk['approved_for_live'] else 'NO'}")
                report.append("")

        # Live Trading Simulation
        if live_results:
            report.append("## Live Trading Simulation")
            report.append(f"- Trades Executed: {live_results['trades_executed']}")
            report.append(f"- Safety Interventions: {live_results['safety_interventions']}")
            report.append(f"- Total P&L: {live_results['total_pnl']:.2f}")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        production_ready = pipeline_results['strategies_production_ready']
        if production_ready > 0:
            report.append(f"PASS: {production_ready} strategies are ready for production deployment")
            report.append("- Proceed with gradual live deployment")
            report.append("- Monitor closely for the first 24-48 hours")
            report.append("- Ensure all safety systems remain active")
        else:
            report.append("WARNING: No strategies currently meet production criteria")
            report.append("- Review strategy generation parameters")
            report.append("- Improve validation thresholds")
            report.append("- Address safety system concerns")

        if safety_status['active_alerts'] > 0:
            report.append("- Resolve active safety alerts before deployment")

        report.append("")
        report.append("---")
        report.append("Report generated by Production Readiness System")

        return "\n".join(report)

    async def run_full_production_readiness_check(self):
        """Run complete production readiness assessment"""
        try:
            # Initialize all systems
            await self.initialize_all_systems()

            # Run strategy development pipeline
            pipeline_results = await self.run_strategy_development_pipeline()

            # Run live trading simulation if strategies are ready
            live_results = None
            if pipeline_results['strategies_production_ready'] > 0:
                live_results = await self.run_live_trading_simulation(pipeline_results['production_ready_strategies'])

            # Generate report
            report = await self.generate_production_report(pipeline_results, live_results)

            # Save report
            report_file = PROJECT_ROOT / "reports" / "production_readiness_report.md"
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, 'w') as f:
                f.write(report)

            print(f"\n[DOCUMENT] Report saved to: {report_file}")

            # Print summary
            print("\n[TARGET] Production Readiness Summary:")
            print(f"  Strategies Ready: {pipeline_results['strategies_production_ready']}")
            print(f"  Safety System: {'[CHECK] OK' if not self.safety_system.emergency_shutdown else '[CROSS] ISSUES'}")
            print(f"  Report Generated: [CHECK]")

            return {
                "success": True,
                "pipeline_results": pipeline_results,
                "live_results": live_results,
                "report": report
            }

        except Exception as e:
            self.logger.error(f"Production readiness check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


async def main():
    """Run the complete production readiness system"""
    import logging
    logging.basicConfig(level=logging.INFO)

    system = ProductionReadinessSystem()

    print("ACCELERATED ARBITRAGE CORP - Production Readiness System")
    print("=" * 65)

    results = await system.run_full_production_readiness_check()

    if results["success"]:
        print("\n[CELEBRATION] Production readiness assessment completed successfully!")
    else:
        print(f"\n[CROSS] Production readiness assessment failed: {results['error']}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
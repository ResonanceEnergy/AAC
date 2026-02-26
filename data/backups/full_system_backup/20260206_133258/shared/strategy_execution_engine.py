"""
Strategy Execution Engine
=========================

Loads and executes arbitrage strategies from CSV definitions.
Converts strategy configurations into real-time trading algorithms.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
import yaml
import re

from shared.strategy_framework import BaseArbitrageStrategy, StrategyFactory, StrategyConfig, TradingSignal
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class StrategyExecutionEngine:
    """
    Engine for loading, managing, and executing arbitrage strategies.

    Responsibilities:
    - Load strategy definitions from CSV
    - Load strategy configurations from YAML
    - Instantiate and manage strategy objects
    - Route market data to strategies
    - Collect and forward trading signals
    - Monitor strategy performance
    """

    def __init__(self, communication: CommunicationFramework, audit_logger: AuditLogger):
        self.communication = communication
        self.audit_logger = audit_logger

        # Strategy storage
        self.strategies: Dict[str, BaseArbitrageStrategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}

        # Market data routing
        self.market_data_subscriptions = set()

        # Performance tracking
        self.strategy_performance = {}

        # Configuration paths
        self.csv_path = Path(__file__).parent.parent / "50_arbitrage_strategies.csv"
        self.config_path = Path(__file__).parent.parent / "config" / "strategy_department_matrix.yaml"

    async def initialize(self) -> bool:
        """Initialize the strategy execution engine."""
        try:
            logger.info("Initializing Strategy Execution Engine")

            # Load strategy definitions from CSV
            await self._load_strategy_definitions()

            # Load strategy configurations from YAML
            await self._load_strategy_configs()

            # Instantiate strategies
            await self._instantiate_strategies()

            # Set up market data routing
            await self._setup_market_data_routing()

            await self.audit_logger.log_event(
                'strategy_engine_initialization',
                f'Strategy Execution Engine initialized with {len(self.strategies)} strategies'
            )

            logger.info(f"Strategy Execution Engine initialized with {len(self.strategies)} strategies")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Strategy Execution Engine: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the strategy execution engine."""
        try:
            logger.info("Shutting down Strategy Execution Engine")

            # Shutdown all strategies
            for strategy in self.strategies.values():
                await strategy.shutdown()

            self.strategies.clear()

            await self.audit_logger.log_event(
                'strategy_engine_shutdown',
                'Strategy Execution Engine shut down successfully'
            )

            return True

        except Exception as e:
            logger.error(f"Error shutting down Strategy Execution Engine: {e}")
            return False

    async def process_market_data(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Process market data and generate trading signals from all strategies."""
        all_signals = []

        try:
            # Route data to relevant strategies
            relevant_strategies = self._get_relevant_strategies(data)
            logger.info(f"Routing {data.get('type')} data to {len(relevant_strategies)} strategies: {relevant_strategies}")

            for strategy in relevant_strategies:
                if strategy in self.strategies:
                    logger.info(f"Processing data for strategy {strategy}")
                    signals = await self.strategies[strategy].process_market_data(data)
                    all_signals.extend(signals)
                else:
                    logger.warning(f"Strategy {strategy} not found in active strategies")

            # Log signal generation
            if all_signals:
                await self.audit_logger.log_event(
                    'strategy_signals_generated',
                    f'Generated {len(all_signals)} signals from {len(relevant_strategies)} strategies',
                    {'total_signals': len(all_signals)}
                )

        except Exception as e:
            logger.error(f"Error processing market data: {e}")

        return all_signals

    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategies."""
        status = {}

        for strategy_id, strategy in self.strategies.items():
            status[strategy_id] = {
                'name': strategy.config.name,
                'status': strategy.status.value,
                'position_size': strategy.position_size,
                'unrealized_pnl': strategy.unrealized_pnl,
                'last_signal_time': strategy.last_signal_time.isoformat() if strategy.last_signal_time else None
            }

        return status

    async def activate_strategy(self, strategy_id: str) -> bool:
        """Activate a specific strategy."""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            return await strategy.initialize()
        return False

    async def deactivate_strategy(self, strategy_id: str) -> bool:
        """Deactivate a specific strategy."""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            return await strategy.shutdown()
        return False

    async def _load_strategy_definitions(self):
        """Load strategy definitions from CSV file."""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    strategy_id = f"s{row['id'].zfill(2)}_{re.sub(r'[^a-zA-Z0-9]', '_', row['strategy_name'].lower())}"

                    # Create basic config from CSV
                    config = StrategyConfig(
                        strategy_id=strategy_id,
                        name=row['strategy_name'],
                        strategy_type="stat_arb",  # Default, will be overridden by YAML
                        edge_source="pricing_inefficiency",  # Default
                        time_horizon="intraday",  # Default
                        complexity="medium",  # Default
                        data_requirements=[],
                        execution_requirements=[],
                        risk_envelope={},
                        cross_department_dependencies={}
                    )

                    self.strategy_configs[strategy_id] = config

            logger.info(f"Loaded {len(self.strategy_configs)} strategy definitions from CSV")

        except Exception as e:
            logger.error(f"Error loading strategy definitions: {e}")
            raise

    async def _load_strategy_configs(self):
        """Load detailed strategy configurations from YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            strategies_config = config_data.get('strategies', {})

            # Update configs with YAML data
            for strategy_key, strategy_data in strategies_config.items():
                if strategy_key in self.strategy_configs:
                    config = self.strategy_configs[strategy_key]

                    # Update with YAML data
                    config.strategy_type = strategy_data.get('strategy_type', config.strategy_type)
                    config.edge_source = strategy_data.get('edge_source', config.edge_source)
                    config.time_horizon = strategy_data.get('time_horizon', config.time_horizon)
                    config.complexity = strategy_data.get('complexity', config.complexity)
                    config.data_requirements = strategy_data.get('data_requirements', [])
                    config.execution_requirements = strategy_data.get('execution_requirements', [])
                    config.risk_envelope = strategy_data.get('risk_envelope', {})
                    config.cross_department_dependencies = strategy_data.get('cross_department_dependencies', {})

            logger.info(f"Updated strategy configurations from YAML")

        except Exception as e:
            logger.error(f"Error loading strategy configurations: {e}")
            # Don't raise - YAML is optional, CSV is required

    async def _instantiate_strategies(self):
        """Instantiate and initialize strategy objects."""
        instantiated_count = 0

        for strategy_id, config in self.strategy_configs.items():
            try:
                # Create strategy instance
                strategy = await StrategyFactory.create_strategy(
                    strategy_id, config, self.communication, self.audit_logger
                )

                if strategy:
                    # Initialize the strategy
                    if await strategy.initialize():
                        self.strategies[strategy_id] = strategy
                        instantiated_count += 1
                        logger.info(f"Instantiated and initialized strategy: {strategy_id}")
                    else:
                        logger.warning(f"Failed to initialize strategy: {strategy_id}")
                else:
                    logger.warning(f"No implementation found for strategy: {strategy_id}")

            except Exception as e:
                logger.error(f"Error instantiating strategy {strategy_id}: {e}")

        logger.info(f"Instantiated {instantiated_count}/{len(self.strategy_configs)} strategies")

    async def _setup_market_data_routing(self):
        """Set up market data subscriptions and routing."""
        try:
            # Subscribe to all required market data feeds
            all_data_requirements = set()
            for config in self.strategy_configs.values():
                all_data_requirements.update(config.data_requirements)

            for requirement in all_data_requirements:
                if requirement == "etf_prices":
                    await self.communication.subscribe_to_messages("strategy_execution_engine", ["market_data.etf.*"])
                elif requirement == "nav_calculations":
                    await self.communication.subscribe_to_messages("strategy_execution_engine", ["bigbrain.nav.*"])
                elif requirement == "index_futures":
                    await self.communication.subscribe_to_messages("strategy_execution_engine", ["market_data.futures.*"])
                elif requirement == "crypto_prices":
                    await self.communication.subscribe_to_messages("strategy_execution_engine", ["market_data.crypto.*"])
                elif requirement == "options_data":
                    await self.communication.subscribe_to_messages("strategy_execution_engine", ["market_data.options.*"])

            logger.info(f"Set up market data routing for {len(all_data_requirements)} data types")

        except Exception as e:
            logger.error(f"Error setting up market data routing: {e}")

    def _get_relevant_strategies(self, data: Dict[str, Any]) -> List[str]:
        """Get strategies that should receive this market data."""
        relevant_strategies = []
        data_type = data.get('type', 'unknown')

        for strategy_id, config in self.strategy_configs.items():
            if self._strategy_needs_data(config, data_type):
                relevant_strategies.append(strategy_id)

        return relevant_strategies

    def _strategy_needs_data(self, config: StrategyConfig, data_type: str) -> bool:
        """Check if a strategy needs a particular type of market data."""
        data_requirements = config.data_requirements

        if data_type == 'etf_price' and 'etf_prices' in data_requirements:
            return True
        elif data_type == 'nav_calculation' and 'nav_calculations' in data_requirements:
            return True
        elif data_type == 'futures_price' and 'index_futures' in data_requirements:
            return True
        elif data_type == 'crypto_price' and 'crypto_prices' in data_requirements:
            return True
        elif data_type == 'options_data' and 'options_data' in data_requirements:
            return True

        return False


    async def get_strategy_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all strategies."""
        status = {}

        for strategy_id, strategy in self.strategies.items():
            status[strategy_id] = {
                'name': self.strategy_configs[strategy_id].name if strategy_id in self.strategy_configs else 'Unknown',
                'status': strategy.status.value if hasattr(strategy, 'status') else 'inactive',
                'last_signal_time': strategy.last_signal_time if hasattr(strategy, 'last_signal_time') else None,
                'position_size': strategy.position_size if hasattr(strategy, 'position_size') else 0.0,
                'unrealized_pnl': strategy.unrealized_pnl if hasattr(strategy, 'unrealized_pnl') else 0.0
            }

        # Add strategies that are defined but not implemented
        for strategy_id, config in self.strategy_configs.items():
            if strategy_id not in status:
                status[strategy_id] = {
                    'name': config.name,
                    'status': 'not_implemented',
                    'last_signal_time': None,
                    'position_size': 0.0,
                    'unrealized_pnl': 0.0
                }

        return status


async def get_strategy_execution_engine(communication: CommunicationFramework,
                                      audit_logger: AuditLogger) -> StrategyExecutionEngine:
    """Factory function to create and initialize the strategy execution engine."""
    engine = StrategyExecutionEngine(communication, audit_logger)

    if await engine.initialize():
        return engine
    else:
        raise RuntimeError("Failed to initialize Strategy Execution Engine")
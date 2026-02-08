#!/usr/bin/env python3
"""
AI Strategy Generation Engine
============================
Machine learning-powered arbitrage strategy discovery and generation.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger
from shared.market_data_integration import market_data_integration


class StrategyType(Enum):
    """Types of arbitrage strategies"""
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    TRIANGULAR_ARBITRAGE = "triangular_arbitrage"
    CROSS_EXCHANGE_ARBITRAGE = "cross_exchange_arbitrage"
    OPTIONS_ARBITRAGE = "options_arbitrage"
    FUTURES_ARBITRAGE = "futures_arbitrage"
    CRYPTO_ARBITRAGE = "crypto_arbitrage"


class RiskLevel(Enum):
    """Strategy risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    opportunity_id: str
    strategy_type: StrategyType
    symbols: List[str]
    exchanges: List[str]
    expected_return: float
    risk_level: RiskLevel
    confidence_score: float
    time_horizon: int  # minutes
    max_position_size: float
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedStrategy:
    """AI-generated trading strategy"""
    strategy_id: str
    name: str
    description: str
    strategy_type: StrategyType
    risk_level: RiskLevel
    symbols: List[str]
    parameters: Dict[str, Any]
    entry_logic: str  # Python code snippet
    exit_logic: str   # Python code snippet
    risk_management: Dict[str, Any]
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class MarketFeatures:
    """Market data features for ML models"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    volatility: float
    spread: float
    order_book_depth: float
    momentum: float
    trend_strength: float
    market_regime: str
    arbitrage_score: float = 0.0


class AIStrategyGenerator:
    """AI-powered strategy generation engine"""

    def __init__(self):
        self.logger = logging.getLogger("AIStrategyGenerator")
        self.audit_logger = get_audit_logger()

        # ML Models
        self.opportunity_detector = None
        self.return_predictor = None
        self.risk_assessor = None
        self.feature_scaler = StandardScaler()

        # Strategy storage
        self.generated_strategies: Dict[str, GeneratedStrategy] = {}
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}

        # Configuration
        self.min_confidence_threshold = 0.7
        self.max_risk_level = RiskLevel.HIGH
        self.strategy_generation_interval = 300  # 5 minutes
        self.opportunity_scan_interval = 60     # 1 minute

        # Load existing models and strategies
        # asyncio.create_task(self._initialize_models())  # Moved to initialize()

    async def initialize(self):
        """Initialize the AI strategy generator with async tasks"""
        await self._initialize_models()

    async def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Try to load existing models
            model_dir = PROJECT_ROOT / "data" / "ai_models"
            model_dir.mkdir(parents=True, exist_ok=True)

            if (model_dir / "opportunity_detector.pkl").exists():
                self.opportunity_detector = joblib.load(model_dir / "opportunity_detector.pkl")
                self.return_predictor = joblib.load(model_dir / "return_predictor.pkl")
                self.risk_assessor = joblib.load(model_dir / "risk_assessor.pkl")
                self.logger.info("Loaded existing AI models")
            else:
                # Train initial models
                await self._train_initial_models()

            # Load strategies
            await self._load_strategies()

        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}")

    async def _train_initial_models(self):
        """Train initial ML models with synthetic data"""
        self.logger.info("Training initial AI models with synthetic data...")

        # Generate synthetic training data
        training_data = await self._generate_synthetic_training_data(10000)

        if not training_data:
            self.logger.warning("No training data available, skipping model training")
            return

        # Prepare features and labels
        features = []
        opportunity_labels = []
        return_labels = []
        risk_labels = []

        for data_point in training_data:
            features.append([
                data_point.price,
                data_point.volume,
                data_point.volatility,
                data_point.spread,
                data_point.order_book_depth,
                data_point.momentum,
                data_point.trend_strength,
            ])

            # Simulate labels
            opportunity_labels.append(1 if data_point.arbitrage_score > 0.5 else 0)
            return_labels.append(data_point.arbitrage_score * 100)  # Expected return %
            risk_labels.append(self._calculate_risk_score(data_point))

        X = np.array(features)
        X_scaled = self.feature_scaler.fit_transform(X)

        # Train opportunity detector
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, opportunity_labels, test_size=0.2, random_state=42
        )

        self.opportunity_detector = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.opportunity_detector.fit(X_train, y_train)

        # Evaluate
        y_pred = self.opportunity_detector.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Opportunity detector accuracy: {accuracy:.3f}")

        # Train return predictor
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, return_labels, test_size=0.2, random_state=42
        )

        self.return_predictor = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        self.return_predictor.fit(X_train, y_train)

        # Evaluate
        y_pred = self.return_predictor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.logger.info(f"Return predictor MSE: {mse:.3f}")

        # Train risk assessor (simplified)
        self.risk_assessor = RandomForestClassifier(n_estimators=50, random_state=42)
        risk_X_train, risk_X_test, risk_y_train, risk_y_test = train_test_split(
            X_scaled, risk_labels, test_size=0.2, random_state=42
        )
        self.risk_assessor.fit(risk_X_train, risk_y_train)

        # Save models
        model_dir = PROJECT_ROOT / "data" / "ai_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.opportunity_detector, model_dir / "opportunity_detector.pkl")
        joblib.dump(self.return_predictor, model_dir / "return_predictor.pkl")
        joblib.dump(self.risk_assessor, model_dir / "risk_assessor.pkl")

        self.logger.info("AI models trained and saved")

    def _calculate_risk_score(self, features: MarketFeatures) -> int:
        """Calculate risk score (0-3 for LOW-MEDIUM-HIGH-EXTREME)"""
        risk_score = 0

        if features.volatility > 0.05:  # High volatility
            risk_score += 1
        if features.spread > 0.001:  # Wide spread
            risk_score += 1
        if features.order_book_depth < 100:  # Low liquidity
            risk_score += 1

        return min(risk_score, 3)  # Cap at EXTREME

    async def _generate_synthetic_training_data(self, n_samples: int) -> List[MarketFeatures]:
        """Generate synthetic training data"""
        data = []

        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "BTC/USDT", "ETH/USDT"]

        for _ in range(n_samples):
            symbol = random.choice(symbols)
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))

            # Generate realistic market features
            price = random.uniform(50, 50000)  # Wide price range
            volume = random.uniform(1000, 1000000)
            volatility = random.uniform(0.001, 0.1)
            spread = random.uniform(0.0001, 0.01)
            order_book_depth = random.uniform(10, 1000)
            momentum = random.uniform(-0.1, 0.1)
            trend_strength = random.uniform(0, 1)

            # Determine market regime
            regime = "bull" if momentum > 0.02 else "bear" if momentum < -0.02 else "sideways"

            # Calculate arbitrage score (simplified)
            arbitrage_score = random.uniform(0, 1)

            features = MarketFeatures(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                volume=volume,
                volatility=volatility,
                spread=spread,
                order_book_depth=order_book_depth,
                momentum=momentum,
                trend_strength=trend_strength,
                market_regime=regime,
                arbitrage_score=arbitrage_score
            )

            data.append(features)

        return data

    async def _load_strategies(self):
        """Load existing generated strategies"""
        try:
            strategy_file = PROJECT_ROOT / "data" / "ai_strategies.json"
            if strategy_file.exists():
                with open(strategy_file, 'r') as f:
                    strategies_data = json.load(f)

                for strategy_dict in strategies_data:
                    strategy = GeneratedStrategy(**strategy_dict)
                    self.generated_strategies[strategy.strategy_id] = strategy

                self.logger.info(f"Loaded {len(self.generated_strategies)} AI strategies")

        except Exception as e:
            self.logger.error(f"Failed to load strategies: {e}")

    async def _save_strategies(self):
        """Save generated strategies"""
        try:
            strategy_file = PROJECT_ROOT / "data" / "ai_strategies.json"
            strategy_file.parent.mkdir(parents=True, exist_ok=True)

            strategies_data = [strategy.__dict__ for strategy in self.generated_strategies.values()]

            with open(strategy_file, 'w') as f:
                json.dump(strategies_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save strategies: {e}")

    async def scan_for_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan market data for arbitrage opportunities"""
        opportunities = []

        try:
            # Get market data for analysis
            symbols = ["SPY", "QQQ", "AAPL", "MSFT", "BTC/USDT", "ETH/USDT"]

            for symbol in symbols:
                context = await market_data_integration.get_market_context(symbol)
                if not context or not context.current_price:
                    continue

                # Extract features
                features = await self._extract_features(symbol, context)

                if not features:
                    continue

                # Predict opportunity
                if self.opportunity_detector:
                    feature_vector = np.array([[
                        features.price,
                        features.volume,
                        features.volatility,
                        features.spread,
                        features.order_book_depth,
                        features.momentum,
                        features.trend_strength,
                    ]])

                    feature_vector_scaled = self.feature_scaler.transform(feature_vector)

                    # Check for opportunity
                    opportunity_prob = self.opportunity_detector.predict_proba(feature_vector_scaled)[0][1]

                    if opportunity_prob >= self.min_confidence_threshold:
                        # Predict return and risk
                        expected_return = self.return_predictor.predict(feature_vector_scaled)[0] if self.return_predictor else 0.5
                        risk_score = self.risk_assessor.predict(feature_vector_scaled)[0] if self.risk_assessor else 1

                        risk_level = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME][risk_score]

                        if risk_level.value <= self.max_risk_level.value:
                            opportunity = ArbitrageOpportunity(
                                opportunity_id=f"opp_{symbol}_{int(datetime.now().timestamp())}",
                                strategy_type=StrategyType.STATISTICAL_ARBITRAGE,
                                symbols=[symbol],
                                exchanges=["simulated"],
                                expected_return=expected_return,
                                risk_level=risk_level,
                                confidence_score=opportunity_prob,
                                time_horizon=30,  # 30 minutes
                                max_position_size=min(10000, context.current_price.price * 100),
                                entry_conditions={
                                    "price_threshold": features.price * 0.995,
                                    "volume_threshold": features.volume * 0.8,
                                },
                                exit_conditions={
                                    "profit_target": expected_return * 0.8,
                                    "stop_loss": -expected_return * 0.2,
                                },
                                expires_at=datetime.now() + timedelta(minutes=30)
                            )

                            opportunities.append(opportunity)
                            self.active_opportunities[opportunity.opportunity_id] = opportunity

            self.logger.info(f"Found {len(opportunities)} arbitrage opportunities")

        except Exception as e:
            self.logger.error(f"Error scanning for opportunities: {e}")

        return opportunities

    async def _extract_features(self, symbol: str, context) -> Optional[MarketFeatures]:
        """Extract ML features from market context"""
        try:
            current_price = context.current_price.price
            volume = context.current_price.volume or 1000

            # Calculate volatility (simplified)
            volatility = 0.02  # Default 2%

            # Calculate spread (simplified)
            spread = current_price * 0.001  # 0.1%

            # Order book depth (simplified)
            order_book_depth = 500

            # Momentum (simplified)
            momentum = random.uniform(-0.05, 0.05)

            # Trend strength (simplified)
            trend_strength = random.uniform(0, 1)

            return MarketFeatures(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=volume,
                volatility=volatility,
                spread=spread,
                order_book_depth=order_book_depth,
                momentum=momentum,
                trend_strength=trend_strength,
                market_regime="neutral"
            )

        except Exception as e:
            self.logger.debug(f"Failed to extract features for {symbol}: {e}")
            return None

    async def generate_strategy_from_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> GeneratedStrategy:
        """Generate a trading strategy from an arbitrage opportunity"""

        strategy_id = f"strategy_{opportunity.opportunity_id}"

        # Generate strategy logic
        entry_logic = self._generate_entry_logic(opportunity)
        exit_logic = self._generate_exit_logic(opportunity)

        strategy = GeneratedStrategy(
            strategy_id=strategy_id,
            name=f"AI {opportunity.strategy_type.value.title()} Strategy",
            description=f"AI-generated {opportunity.strategy_type.value} strategy for {', '.join(opportunity.symbols)}",
            strategy_type=opportunity.strategy_type,
            risk_level=opportunity.risk_level,
            symbols=opportunity.symbols,
            parameters={
                "expected_return": opportunity.expected_return,
                "confidence_score": opportunity.confidence_score,
                "time_horizon": opportunity.time_horizon,
                "max_position_size": opportunity.max_position_size,
            },
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            risk_management={
                "max_position_size": opportunity.max_position_size,
                "stop_loss_pct": 0.05,
                "take_profit_pct": opportunity.expected_return * 0.8,
                "max_hold_time": opportunity.time_horizon * 60,  # seconds
            }
        )

        # Run backtest
        backtest_results = await self._backtest_strategy(strategy)
        strategy.backtest_results = backtest_results

        self.generated_strategies[strategy_id] = strategy
        await self._save_strategies()

        await self.audit_logger.log_event(
            "ai_strategy_generated",
            {"strategy_id": strategy_id, "opportunity_id": opportunity.opportunity_id},
            "info"
        )

        self.logger.info(f"Generated AI strategy: {strategy_id}")
        return strategy

    def _generate_entry_logic(self, opportunity: ArbitrageOpportunity) -> str:
        """Generate Python code for strategy entry logic"""
        conditions = opportunity.entry_conditions

        code = f"""
def should_enter(symbol, context):
    \"\"\"Check if strategy should enter position\"\"\"
    current_price = context.current_price.price if context.current_price else 0

    # Entry conditions
    price_ok = current_price <= {conditions.get('price_threshold', 0)}
    volume_ok = context.current_price.volume >= {conditions.get('volume_threshold', 0)} if context.current_price else True

    return price_ok and volume_ok
"""

        return code

    def _generate_exit_logic(self, opportunity: ArbitrageOpportunity) -> str:
        """Generate Python code for strategy exit logic"""
        conditions = opportunity.exit_conditions

        code = f"""
def should_exit(position, current_price, entry_price, hold_time):
    \"\"\"Check if strategy should exit position\"\"\"
    profit_pct = (current_price - entry_price) / entry_price

    # Exit conditions
    profit_target_hit = profit_pct >= {conditions.get('profit_target', 0.01)}
    stop_loss_hit = profit_pct <= {conditions.get('stop_loss', -0.02)}
    time_expired = hold_time >= {opportunity.time_horizon * 60}

    return profit_target_hit or stop_loss_hit or time_expired
"""

        return code

    async def _backtest_strategy(self, strategy: GeneratedStrategy) -> Dict[str, Any]:
        """Run backtest for generated strategy"""
        # Simplified backtest with synthetic data
        try:
            # Generate synthetic price data
            n_periods = 1000
            initial_price = 100.0
            prices = []
            price = initial_price

            for _ in range(n_periods):
                # Random walk with drift
                change = np.random.normal(0.0001, 0.01)  # Small drift, 1% volatility
                price *= (1 + change)
                prices.append(price)

            # Simulate strategy performance
            capital = 10000.0
            position = 0
            entry_price = 0
            trades = []
            max_drawdown = 0
            peak_capital = capital

            for i, price in enumerate(prices):
                # Simple entry/exit logic simulation
                if position == 0 and i % 50 == 0:  # Enter every 50 periods
                    position = min(capital * 0.1 / price, strategy.parameters['max_position_size'] / price)
                    entry_price = price
                    capital -= position * price

                elif position > 0:
                    profit_pct = (price - entry_price) / entry_price
                    hold_time = i * 60  # Assume 1 minute per period

                    # Check exit conditions
                    if (profit_pct >= strategy.risk_management['take_profit_pct'] or
                        profit_pct <= -strategy.risk_management['stop_loss_pct'] or
                        hold_time >= strategy.risk_management['max_hold_time']):

                        # Exit position
                        capital += position * price
                        pnl = position * (price - entry_price)
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': price,
                            'pnl': pnl,
                            'hold_time': hold_time
                        })
                        position = 0
                        entry_price = 0

                # Track drawdown
                if capital > peak_capital:
                    peak_capital = capital
                current_drawdown = (peak_capital - capital) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)

            # Calculate metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades

            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0

            profit_factor = abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) /
                              sum(t['pnl'] for t in trades if t['pnl'] < 0)) if losing_trades > 0 else float('inf')

            return {
                "total_return": total_pnl / 10000.0,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "sharpe_ratio": total_pnl / (np.std([t['pnl'] for t in trades]) * np.sqrt(252)) if trades else 0,
                "backtest_period_days": n_periods / (24 * 60),  # Assuming 1 reading per minute
            }

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {"error": str(e)}

    async def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get currently active arbitrage opportunities"""
        # Clean expired opportunities
        current_time = datetime.now()
        expired = [oid for oid, opp in self.active_opportunities.items()
                  if opp.expires_at and opp.expires_at < current_time]

        for oid in expired:
            del self.active_opportunities[oid]

        return list(self.active_opportunities.values())

    def get_generated_strategies(self) -> List[GeneratedStrategy]:
        """Get all generated strategies"""
        return list(self.generated_strategies.values())

    async def start_strategy_generation_loop(self):
        """Start continuous strategy generation"""
        while True:
            try:
                # Scan for opportunities
                opportunities = await self.scan_for_opportunities()

                # Generate strategies from high-confidence opportunities
                for opportunity in opportunities:
                    if opportunity.confidence_score >= 0.8:  # High confidence threshold
                        strategy = await self.generate_strategy_from_opportunity(opportunity)
                        self.logger.info(f"Generated high-confidence strategy: {strategy.strategy_id}")

                # Wait before next scan
                await asyncio.sleep(self.strategy_generation_interval)

            except Exception as e:
                self.logger.error(f"Strategy generation loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def retrain_models(self):
        """Retrain ML models with new data"""
        self.logger.info("Retraining AI models...")
        await self._train_initial_models()
        self.logger.info("AI models retrained")


# Global AI strategy generator instance
ai_strategy_generator = AIStrategyGenerator()


async def initialize_ai_strategy_generation():
    """Initialize the AI strategy generation system"""
    print("[AI] Initializing AI Strategy Generation Engine...")

    # Initialize the AI strategy generator instance
    await ai_strategy_generator.initialize()

    # Start background strategy generation
    asyncio.create_task(ai_strategy_generator.start_strategy_generation_loop())

    print("[OK] AI strategy generation initialized")
    print(f"  Active opportunities: {len(await ai_strategy_generator.get_active_opportunities())}")
    print(f"  Generated strategies: {len(ai_strategy_generator.get_generated_strategies())}")


if __name__ == "__main__":
    # Example usage
    asyncio.run(initialize_ai_strategy_generation())
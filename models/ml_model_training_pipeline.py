#!/usr/bin/env python3
"""
ML Model Training Pipeline
==========================
Automated machine learning pipeline for training predictive risk models and feature engineering.
Provides continuous model training, validation, and deployment for arbitrage strategies.

Features:
- Real market data training pipeline
- Automated feature engineering
- Model validation and backtesting
- Continuous learning and retraining
- Risk prediction models
- Performance monitoring and alerting
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
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger
from shared.market_data_feeds import get_market_data_feed


class ModelType(Enum):
    """Types of ML models"""
    OPPORTUNITY_DETECTOR = "opportunity_detector"
    RETURN_PREDICTOR = "return_predictor"
    RISK_ASSESSOR = "risk_assessor"
    VOLATILITY_PREDICTOR = "volatility_predictor"
    CORRELATION_PREDICTOR = "correlation_predictor"
    SENTIMENT_ANALYZER = "sentiment_analyzer"


class TrainingDataType(Enum):
    """Types of training data"""
    HISTORICAL_PRICES = "historical_prices"
    ORDER_BOOK_DATA = "order_book_data"
    TRADE_DATA = "trade_data"
    SENTIMENT_DATA = "sentiment_data"
    FUNDAMENTAL_DATA = "fundamental_data"
    TECHNICAL_INDICATORS = "technical_indicators"


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_type: ModelType
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    training_time: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    cross_val_scores: List[float] = field(default_factory=list)
    last_trained: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingDataset:
    """Training dataset with features and labels"""
    data_type: TrainingDataType
    features: pd.DataFrame
    labels: pd.Series
    feature_names: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 0.0


class MLModelTrainingPipeline:
    """
    Automated ML model training pipeline for arbitrage strategies.
    Provides continuous learning, feature engineering, and model deployment.
    """

    def __init__(self):
        self.logger = logging.getLogger("MLModelTrainingPipeline")
        self.audit_logger = get_audit_logger()
        self.market_data = None

        # Model storage
        self.models: Dict[ModelType, Any] = {}
        self.model_performance: Dict[ModelType, ModelPerformance] = {}
        self.feature_scalers: Dict[ModelType, Any] = {}

        # Training configuration
        self.training_intervals = {
            ModelType.OPPORTUNITY_DETECTOR: 3600,  # 1 hour
            ModelType.RETURN_PREDICTOR: 7200,      # 2 hours
            ModelType.RISK_ASSESSOR: 1800,         # 30 minutes
            ModelType.VOLATILITY_PREDICTOR: 3600,  # 1 hour
            ModelType.CORRELATION_PREDICTOR: 7200, # 2 hours
            ModelType.SENTIMENT_ANALYZER: 1800,    # 30 minutes
        }

        self.min_training_samples = 1000
        self.validation_split = 0.2
        self.cv_folds = 5

        # Feature engineering
        self.feature_engineering_pipeline = None

        # Model artifacts directory
        self.models_dir = PROJECT_ROOT / "data" / "ml_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("ML Model Training Pipeline initialized")

    async def initialize(self):
        """Initialize the ML training pipeline"""
        try:
            self.logger.info("Initializing ML Model Training Pipeline...")

            # Initialize market data feed
            self.market_data = await get_market_data_feed()

            # Load existing models
            await self._load_existing_models()

            # Initialize feature engineering pipeline
            await self._initialize_feature_engineering()

            # Start continuous training tasks
            asyncio.create_task(self._continuous_training_loop())

            self.logger.info("ML Model Training Pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ML training pipeline: {e}")
            raise

    async def _load_existing_models(self):
        """Load existing trained models"""
        try:
            for model_type in ModelType:
                model_path = self.models_dir / f"{model_type.value}.pkl"
                perf_path = self.models_dir / f"{model_type.value}_performance.json"
                scaler_path = self.models_dir / f"{model_type.value}_scaler.pkl"

                if model_path.exists():
                    self.models[model_type] = joblib.load(model_path)
                    self.logger.info(f"Loaded existing {model_type.value} model")

                if perf_path.exists():
                    with open(perf_path, 'r') as f:
                        perf_data = json.load(f)
                        # Parse datetime strings back to datetime objects
                        if 'last_trained' in perf_data and isinstance(perf_data['last_trained'], str):
                            perf_data['last_trained'] = datetime.fromisoformat(perf_data['last_trained'])
                        self.model_performance[model_type] = ModelPerformance(
                            model_type=model_type,
                            **perf_data
                        )

                if scaler_path.exists():
                    self.feature_scalers[model_type] = joblib.load(scaler_path)

        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")

    async def _initialize_feature_engineering(self):
        """Initialize automated feature engineering pipeline"""
        try:
            self.feature_engineering_pipeline = {
                'technical_indicators': self._calculate_technical_indicators,
                'market_microstructure': self._calculate_market_microstructure_features,
                'volatility_features': self._calculate_volatility_features,
                'correlation_features': self._calculate_correlation_features,
                'sentiment_features': self._calculate_sentiment_features,
            }
            self.logger.info("Feature engineering pipeline initialized")

        except Exception as e:
            self.logger.error(f"Error initializing feature engineering: {e}")

    async def _continuous_training_loop(self):
        """Continuous model training and retraining loop"""
        while True:
            try:
                # Check if any models need retraining
                for model_type in ModelType:
                    last_trained = self.model_performance.get(model_type)
                    if not last_trained or \
                       (datetime.now() - last_trained.last_trained).seconds > self.training_intervals[model_type]:
                        await self._train_model(model_type)

                # Wait before next training cycle
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in continuous training loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _train_model(self, model_type: ModelType):
        """Train a specific model type"""
        try:
            self.logger.info(f"Training {model_type.value} model...")

            # Collect training data
            training_data = await self._collect_training_data(model_type)
            if not training_data or len(training_data.features) < self.min_training_samples:
                self.logger.warning(f"Insufficient training data for {model_type.value}")
                return

            # Feature engineering
            engineered_features = await self._apply_feature_engineering(training_data)

            # Train model
            model, performance = await self._train_model_instance(model_type, engineered_features)

            # Validate model
            is_valid = await self._validate_model(model_type, model, performance)
            if not is_valid:
                self.logger.warning(f"Model validation failed for {model_type.value}")
                return

            # Deploy model
            await self._deploy_model(model_type, model, performance)

            self.logger.info(f"Successfully trained and deployed {model_type.value} model")

        except Exception as e:
            self.logger.error(f"Error training {model_type.value} model: {e}")

    async def _collect_training_data(self, model_type: ModelType) -> Optional[TrainingDataset]:
        """Collect training data for a specific model type"""
        try:
            if model_type == ModelType.OPPORTUNITY_DETECTOR:
                return await self._collect_opportunity_training_data()
            elif model_type == ModelType.RETURN_PREDICTOR:
                return await self._collect_return_training_data()
            elif model_type == ModelType.RISK_ASSESSOR:
                return await self._collect_risk_training_data()
            elif model_type == ModelType.VOLATILITY_PREDICTOR:
                return await self._collect_volatility_training_data()
            elif model_type == ModelType.CORRELATION_PREDICTOR:
                return await self._collect_correlation_training_data()
            elif model_type == ModelType.SENTIMENT_ANALYZER:
                return await self._collect_sentiment_training_data()
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error collecting training data for {model_type.value}: {e}")
            return None

    async def _collect_opportunity_training_data(self) -> TrainingDataset:
        """Collect training data for opportunity detection"""
        try:
            # Get historical price data
            symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL']
            features = []
            labels = []

            data_collected = False

            for symbol in symbols:
                try:
                    # Get historical data
                    historical_data = await self.market_data.get_historical_prices(symbol, days=30)

                    if historical_data.empty or len(historical_data) < 50:
                        continue

                    data_collected = True

                    # Calculate arbitrage opportunities
                    for i in range(len(historical_data) - 1):
                        current_price = historical_data.iloc[i]['close']
                        next_price = historical_data.iloc[i + 1]['close']
                        volume = historical_data.iloc[i]['volume']
                        volatility = historical_data.iloc[i]['high'] - historical_data.iloc[i]['low']

                        # Simple opportunity detection (price movement > 1%)
                        opportunity = abs(next_price - current_price) / current_price > 0.01

                        features.append([current_price, volume, volatility])
                        labels.append(1 if opportunity else 0)

                except Exception as e:
                    self.logger.warning(f"Error collecting data for {symbol}: {e}")
                    continue

            # If no real data available, generate synthetic data
            if not data_collected or len(features) < self.min_training_samples:
                self.logger.info("Using synthetic data for opportunity detection training")
                features, labels = self._generate_synthetic_opportunity_data()

            return TrainingDataset(
                data_type=TrainingDataType.HISTORICAL_PRICES,
                features=pd.DataFrame(features, columns=['price', 'volume', 'volatility']),
                labels=pd.Series(labels),
                feature_names=['price', 'volume', 'volatility']
            )

        except Exception as e:
            self.logger.error(f"Error collecting opportunity training data: {e}")
            # Fallback to synthetic data
            features, labels = self._generate_synthetic_opportunity_data()
            return TrainingDataset(
                data_type=TrainingDataType.HISTORICAL_PRICES,
                features=pd.DataFrame(features, columns=['price', 'volume', 'volatility']),
                labels=pd.Series(labels),
                feature_names=['price', 'volume', 'volatility']
            )

    def _generate_synthetic_opportunity_data(self) -> Tuple[List[List[float]], List[int]]:
        """Generate synthetic training data for opportunity detection"""
        np.random.seed(42)
        n_samples = max(self.min_training_samples, 2000)

        features = []
        labels = []

        for _ in range(n_samples):
            # Generate realistic price data (around $100-500 range)
            price = np.random.normal(300, 50)
            price = max(50, min(1000, price))  # Clamp to reasonable range

            # Generate volume (1000-100000)
            volume = np.random.exponential(10000)
            volume = max(1000, min(100000, volume))

            # Generate volatility (0.5-5.0)
            volatility = np.random.exponential(1.0)
            volatility = max(0.1, min(10.0, volatility))

            features.append([price, volume, volatility])

            # Generate opportunity label based on features
            # Higher volatility and volume increase chance of opportunity
            opportunity_prob = min(0.3, (volatility * 0.1) + (volume / 100000) * 0.2)
            opportunity = np.random.random() < opportunity_prob

            labels.append(1 if opportunity else 0)

        return features, labels

    def _generate_synthetic_return_data(self) -> Tuple[List[List[float]], List[float]]:
        """Generate synthetic training data for return prediction"""
        np.random.seed(123)
        n_samples = max(self.min_training_samples, 2000)

        features = []
        labels = []

        for _ in range(n_samples):
            # Generate realistic price data
            price = np.random.normal(300, 50)
            price = max(50, min(1000, price))

            # Generate volume
            volume = np.random.exponential(10000)
            volume = max(1000, min(100000, volume))

            features.append([price, volume])

            # Generate realistic returns (mean ~0, std ~2%)
            returns = np.random.normal(0, 0.02)
            # Add some autocorrelation and mean reversion
            if len(labels) > 0:
                returns = 0.7 * returns + 0.3 * labels[-1] + np.random.normal(0, 0.005)

            labels.append(returns)

        return features, labels

    def _generate_synthetic_risk_data(self) -> Tuple[List[List[float]], List[float]]:
        """Generate synthetic training data for risk assessment"""
        np.random.seed(456)
        n_samples = max(self.min_training_samples, 2000)

        features = []
        labels = []

        for _ in range(n_samples):
            # Generate realistic volatility (0.005-0.05)
            volatility = np.random.exponential(0.015)
            volatility = max(0.005, min(0.1, volatility))

            # Generate drawdown (-0.3 to 0)
            drawdown = -np.random.exponential(0.05)
            drawdown = max(-0.5, min(0, drawdown))

            features.append([volatility, drawdown])

            # Risk score based on volatility and drawdown
            risk_score = (volatility * 0.6) + (abs(drawdown) * 0.4)
            labels.append(min(risk_score, 1.0))

        return features, labels

    async def _collect_return_training_data(self) -> TrainingDataset:
        """Collect training data for return prediction"""
        try:
            # Similar to opportunity data but for return prediction
            symbols = ['SPY', 'QQQ', 'IWM']
            features = []
            labels = []

            data_collected = False

            for symbol in symbols:
                try:
                    historical_data = await self.market_data.get_historical_prices(symbol, days=30)

                    if historical_data.empty or len(historical_data) < 50:
                        continue

                    data_collected = True

                    for i in range(len(historical_data) - 1):
                        current_price = historical_data.iloc[i]['close']
                        next_price = historical_data.iloc[i + 1]['close']
                        returns = (next_price - current_price) / current_price

                        features.append([current_price, historical_data.iloc[i]['volume']])
                        labels.append(returns)

                except Exception as e:
                    self.logger.warning(f"Error collecting return data for {symbol}: {e}")
                    continue

            # If no real data available, generate synthetic data
            if not data_collected or len(features) < self.min_training_samples:
                self.logger.info("Using synthetic data for return prediction training")
                features, labels = self._generate_synthetic_return_data()

            return TrainingDataset(
                data_type=TrainingDataType.HISTORICAL_PRICES,
                features=pd.DataFrame(features, columns=['price', 'volume']),
                labels=pd.Series(labels),
                feature_names=['price', 'volume']
            )

        except Exception as e:
            self.logger.error(f"Error collecting return training data: {e}")
            # Fallback to synthetic data
            features, labels = self._generate_synthetic_return_data()
            return TrainingDataset(
                data_type=TrainingDataType.HISTORICAL_PRICES,
                features=pd.DataFrame(features, columns=['price', 'volume']),
                labels=pd.Series(labels),
                feature_names=['price', 'volume']
            )

    async def _collect_risk_training_data(self) -> TrainingDataset:
        """Collect training data for risk assessment"""
        try:
            # Use volatility and drawdown data for risk assessment
            symbols = ['SPY', 'QQQ']
            features = []
            labels = []

            data_collected = False

            for symbol in symbols:
                try:
                    historical_data = await self.market_data.get_historical_prices(symbol, days=60)

                    if historical_data.empty or len(historical_data) < 50:
                        continue

                    data_collected = True

                    # Calculate rolling volatility and drawdowns
                    historical_data['returns'] = historical_data['close'].pct_change()
                    historical_data['volatility'] = historical_data['returns'].rolling(20).std()
                    historical_data['drawdown'] = (historical_data['close'] - historical_data['close'].rolling(20).max()) / historical_data['close'].rolling(20).max()

                    for i in range(20, len(historical_data)):
                        vol = historical_data.iloc[i]['volatility']
                        drawdown = historical_data.iloc[i]['drawdown']

                        # Risk score based on volatility and drawdown
                        risk_score = (vol * 0.6) + (abs(drawdown) * 0.4)

                        features.append([vol, drawdown])
                        labels.append(min(risk_score, 1.0))  # Normalize to 0-1

                except Exception as e:
                    self.logger.warning(f"Error collecting risk data for {symbol}: {e}")
                    continue

            # If no real data available, generate synthetic data
            if not data_collected or len(features) < self.min_training_samples:
                self.logger.info("Using synthetic data for risk assessment training")
                features, labels = self._generate_synthetic_risk_data()

            return TrainingDataset(
                data_type=TrainingDataType.TECHNICAL_INDICATORS,
                features=pd.DataFrame(features, columns=['volatility', 'drawdown']),
                labels=pd.Series(labels),
                feature_names=['volatility', 'drawdown']
            )

        except Exception as e:
            self.logger.error(f"Error collecting risk training data: {e}")
            # Fallback to synthetic data
            features, labels = self._generate_synthetic_risk_data()
            return TrainingDataset(
                data_type=TrainingDataType.TECHNICAL_INDICATORS,
                features=pd.DataFrame(features, columns=['volatility', 'drawdown']),
                labels=pd.Series(labels),
                feature_names=['volatility', 'drawdown']
            )

    async def _collect_volatility_training_data(self) -> TrainingDataset:
        """Collect training data for volatility prediction"""
        symbols = ['VXX', 'SPY']
        features = []
        labels = []

        for symbol in symbols:
            historical_data = await self.market_data.get_historical_prices(symbol, days=30)

            if historical_data.empty:
                continue

            # Calculate realized volatility
            historical_data['returns'] = historical_data['close'].pct_change()
            historical_data['realized_vol'] = historical_data['returns'].rolling(20).std()

            for i in range(20, len(historical_data) - 1):
                # Features: past volatility, volume, price changes
                past_vol = historical_data.iloc[i]['realized_vol']
                volume = historical_data.iloc[i]['volume']
                price_change = historical_data.iloc[i]['returns']

                # Label: next period volatility
                next_vol = historical_data.iloc[i + 1]['realized_vol']

                features.append([past_vol, volume, price_change])
                labels.append(next_vol)

        return TrainingDataset(
            data_type=TrainingDataType.TECHNICAL_INDICATORS,
            features=pd.DataFrame(features, columns=['past_volatility', 'volume', 'price_change']),
            labels=pd.Series(labels),
            feature_names=['past_volatility', 'volume', 'price_change']
        )

    async def _collect_correlation_training_data(self) -> TrainingDataset:
        """Collect training data for correlation prediction"""
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
        features = []
        labels = []

        # Get data for all symbols
        symbol_data = {}
        for symbol in symbols:
            data = await self.market_data.get_historical_prices(symbol, days=30)
            if not data.empty:
                symbol_data[symbol] = data['close'].pct_change()

        if len(symbol_data) < 2:
            return None

        # Calculate rolling correlations
        combined_data = pd.DataFrame(symbol_data)
        correlation_matrix = combined_data.corr()

        # Use SPY vs QQQ correlation as example
        if 'SPY' in combined_data.columns and 'QQQ' in combined_data.columns:
            rolling_corr = combined_data['SPY'].rolling(20).corr(combined_data['QQQ'])

            for i in range(20, len(rolling_corr) - 1):
                # Features: current correlation, individual volatilities
                current_corr = rolling_corr.iloc[i]
                spy_vol = combined_data['SPY'].rolling(20).std().iloc[i]
                qqq_vol = combined_data['QQQ'].rolling(20).std().iloc[i]

                # Label: next correlation
                next_corr = rolling_corr.iloc[i + 1]

                features.append([current_corr, spy_vol, qqq_vol])
                labels.append(next_corr)

        return TrainingDataset(
            data_type=TrainingDataType.TECHNICAL_INDICATORS,
            features=pd.DataFrame(features, columns=['correlation', 'spy_volatility', 'qqq_volatility']),
            labels=pd.Series(labels),
            feature_names=['correlation', 'spy_volatility', 'qqq_volatility']
        )

    async def _collect_sentiment_training_data(self) -> TrainingDataset:
        """Collect training data for sentiment analysis"""
        # This would integrate with sentiment data sources
        # For now, return synthetic data structure
        features = pd.DataFrame({
            'sentiment_score': np.random.normal(0, 1, 1000),
            'volume': np.random.exponential(1000, 1000),
            'price_change': np.random.normal(0, 0.02, 1000)
        })
        labels = pd.Series(np.random.choice([0, 1], 1000))

        return TrainingDataset(
            data_type=TrainingDataType.SENTIMENT_DATA,
            features=features,
            labels=labels,
            feature_names=['sentiment_score', 'volume', 'price_change']
        )

    async def _apply_feature_engineering(self, training_data: TrainingDataset) -> TrainingDataset:
        """Apply automated feature engineering"""
        try:
            engineered_features = training_data.features.copy()

            # Apply feature engineering functions
            for feature_type, engineering_func in self.feature_engineering_pipeline.items():
                try:
                    additional_features = await engineering_func(training_data.features)
                    if additional_features is not None:
                        engineered_features = pd.concat([engineered_features, additional_features], axis=1)
                except Exception as e:
                    self.logger.warning(f"Feature engineering failed for {feature_type}: {e}")

            return TrainingDataset(
                data_type=training_data.data_type,
                features=engineered_features,
                labels=training_data.labels,
                feature_names=list(engineered_features.columns),
                timestamp=training_data.timestamp,
                data_quality_score=training_data.data_quality_score
            )

        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            return training_data

    async def _calculate_technical_indicators(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            indicators = pd.DataFrame(index=features.index)

            if 'price' in features.columns:
                # Simple moving averages
                indicators['sma_5'] = features['price'].rolling(5).mean()
                indicators['sma_20'] = features['price'].rolling(20).mean()

                # RSI
                delta = features['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs))

                # MACD
                ema_12 = features['price'].ewm(span=12).mean()
                ema_26 = features['price'].ewm(span=26).mean()
                indicators['macd'] = ema_12 - ema_26

            return indicators.fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return None

    async def _calculate_market_microstructure_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        try:
            microstructure = pd.DataFrame(index=features.index)

            if 'volume' in features.columns:
                # Volume-based features
                microstructure['volume_sma'] = features['volume'].rolling(20).mean()
                microstructure['volume_ratio'] = features['volume'] / features['volume'].rolling(20).mean()

            return microstructure.fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating microstructure features: {e}")
            return None

    async def _calculate_volatility_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features"""
        try:
            volatility_features = pd.DataFrame(index=features.index)

            if 'price' in features.columns:
                returns = features['price'].pct_change()
                volatility_features['realized_vol'] = returns.rolling(20).std()
                volatility_features['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * ((np.log(features.get('high', features['price'])/features.get('low', features['price'])))**2).rolling(20).mean())

            return volatility_features.fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {e}")
            return None

    async def _calculate_correlation_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation-based features"""
        # This would require multiple asset data
        return None

    async def _calculate_sentiment_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment-based features"""
        # This would integrate with sentiment analysis
        return None

    async def _train_model_instance(self, model_type: ModelType, training_data: TrainingDataset) -> Tuple[Any, ModelPerformance]:
        """Train a specific model instance"""
        try:
            start_time = datetime.now()

            X = training_data.features.values
            y = training_data.labels.values

            # Feature scaling
            if model_type not in self.feature_scalers:
                self.feature_scalers[model_type] = RobustScaler()

            X_scaled = self.feature_scalers[model_type].fit_transform(X)

            # Split data with time series awareness
            train_size = int(len(X_scaled) * (1 - self.validation_split))
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Select model based on type
            if model_type in [ModelType.OPPORTUNITY_DETECTOR, ModelType.SENTIMENT_ANALYZER]:
                # Classification models
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                performance = ModelPerformance(
                    model_type=model_type,
                    accuracy=accuracy,
                    precision=report['weighted avg']['precision'],
                    recall=report['weighted avg']['recall'],
                    f1_score=report['weighted avg']['f1-score'],
                    training_time=(datetime.now() - start_time).total_seconds(),
                    last_trained=datetime.now()
                )

            else:
                # Regression models
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))

                # R² score
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                performance = ModelPerformance(
                    model_type=model_type,
                    mse=mse,
                    rmse=rmse,
                    mae=mae,
                    r2_score=r2,
                    training_time=(datetime.now() - start_time).total_seconds(),
                    last_trained=datetime.now()
                )

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                performance.feature_importance = dict(zip(training_data.feature_names, model.feature_importances_))

            # Cross-validation scores
            tscv = TimeSeriesSplit(n_splits=min(self.cv_folds, len(X_train) // 100))
            if model_type in [ModelType.OPPORTUNITY_DETECTOR, ModelType.SENTIMENT_ANALYZER]:
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                cv_scores = -cv_scores  # Convert to positive MSE

            performance.cross_val_scores = cv_scores.tolist()

            return model, performance

        except Exception as e:
            self.logger.error(f"Error training {model_type.value} model: {e}")
            raise

    async def _validate_model(self, model_type: ModelType, model: Any, performance: ModelPerformance) -> bool:
        """Validate trained model performance"""
        try:
            # Basic validation thresholds
            if model_type in [ModelType.OPPORTUNITY_DETECTOR, ModelType.RISK_ASSESSOR]:
                min_accuracy = 0.55  # 55% minimum accuracy
                return performance.accuracy >= min_accuracy
            else:
                max_rmse = 0.1  # 10% maximum RMSE for regression
                return performance.rmse <= max_rmse

        except Exception as e:
            self.logger.error(f"Error validating {model_type.value} model: {e}")
            return False

    async def _deploy_model(self, model_type: ModelType, model: Any, performance: ModelPerformance):
        """Deploy trained model to production"""
        try:
            # Save model
            model_path = self.models_dir / f"{model_type.value}.pkl"
            joblib.dump(model, model_path)

            # Save performance metrics
            perf_path = self.models_dir / f"{model_type.value}_performance.json"
            perf_data = {
                'accuracy': float(performance.accuracy),
                'precision': float(performance.precision),
                'recall': float(performance.recall),
                'f1_score': float(performance.f1_score),
                'mse': float(performance.mse),
                'rmse': float(performance.rmse),
                'mae': float(performance.mae),
                'r2_score': float(performance.r2_score),
                'training_time': float(performance.training_time),
                'feature_importance': {k: float(v) for k, v in performance.feature_importance.items()},
                'cross_val_scores': [float(x) for x in performance.cross_val_scores],
                'last_trained': performance.last_trained.isoformat()
            }

            with open(perf_path, 'w') as f:
                json.dump(perf_data, f, indent=2)

            # Save scaler
            scaler_path = self.models_dir / f"{model_type.value}_scaler.pkl"
            if model_type in self.feature_scalers:
                joblib.dump(self.feature_scalers[model_type], scaler_path)

            # Update in-memory models
            self.models[model_type] = model
            self.model_performance[model_type] = performance

            # Audit log
            await self.audit_logger.log_event(
                "ml_model_deployed",
                {
                    "model_type": model_type.value,
                    "performance": perf_data,
                    "deployment_time": datetime.now().isoformat()
                }
            )

            self.logger.info(f"Successfully deployed {model_type.value} model")

        except Exception as e:
            self.logger.error(f"Error deploying {model_type.value} model: {e}")
            raise

    async def predict_opportunity(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict arbitrage opportunity"""
        return await self._predict(ModelType.OPPORTUNITY_DETECTOR, features)

    async def predict_return(self, features: Dict[str, float]) -> float:
        """Predict expected return"""
        _, prediction = await self._predict(ModelType.RETURN_PREDICTOR, features)
        return prediction

    async def assess_risk(self, features: Dict[str, float]) -> float:
        """Assess risk score"""
        _, risk_score = await self._predict(ModelType.RISK_ASSESSOR, features)
        return risk_score

    async def predict_volatility(self, features: Dict[str, float]) -> float:
        """Predict volatility"""
        _, volatility = await self._predict(ModelType.VOLATILITY_PREDICTOR, features)
        return volatility

    async def _predict(self, model_type: ModelType, features: Dict[str, float]) -> Tuple[bool, float]:
        """Generic prediction method"""
        try:
            if model_type not in self.models:
                return False, 0.0

            model = self.models[model_type]
            scaler = self.feature_scalers.get(model_type)

            # Prepare features
            feature_vector = np.array([list(features.values())])

            if scaler:
                feature_vector = scaler.transform(feature_vector)

            # Make prediction
            if model_type in [ModelType.OPPORTUNITY_DETECTOR, ModelType.RISK_ASSESSOR]:
                prediction_proba = model.predict_proba(feature_vector)[0]
                prediction = model.predict(feature_vector)[0]

                if model_type == ModelType.OPPORTUNITY_DETECTOR:
                    confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                    return bool(prediction), confidence
                else:
                    return True, prediction_proba[0]  # Risk score

            else:
                prediction = model.predict(feature_vector)[0]
                return True, prediction

        except Exception as e:
            self.logger.error(f"Error making {model_type.value} prediction: {e}")
            return False, 0.0

    async def get_model_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive model performance report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "models": {}
            }

            for model_type, performance in self.model_performance.items():
                report["models"][model_type.value] = {
                    "accuracy": performance.accuracy,
                    "precision": performance.precision,
                    "recall": performance.recall,
                    "f1_score": performance.f1_score,
                    "mse": performance.mse,
                    "rmse": performance.rmse,
                    "mae": performance.mae,
                    "r2_score": performance.r2_score,
                    "training_time": performance.training_time,
                    "last_trained": performance.last_trained.isoformat(),
                    "cross_val_mean": np.mean(performance.cross_val_scores) if performance.cross_val_scores else 0,
                    "cross_val_std": np.std(performance.cross_val_scores) if performance.cross_val_scores else 0,
                    "top_features": sorted(performance.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                }

            return report

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {}

    async def retrain_all_models(self):
        """Force retraining of all models"""
        try:
            self.logger.info("Starting forced retraining of all models...")

            for model_type in ModelType:
                await self._train_model(model_type)

            self.logger.info("Completed forced retraining of all models")

        except Exception as e:
            self.logger.error(f"Error in forced retraining: {e}")
            raise


# Global ML training pipeline instance
_ml_training_pipeline = None

async def get_ml_training_pipeline() -> MLModelTrainingPipeline:
    """Get or create ML training pipeline instance"""
    global _ml_training_pipeline
    if _ml_training_pipeline is None:
        _ml_training_pipeline = MLModelTrainingPipeline()
        await _ml_training_pipeline.initialize()
    return _ml_training_pipeline

async def initialize_ml_training_pipeline():
    """Initialize the ML model training pipeline"""
    pipeline = await get_ml_training_pipeline()
    return pipeline


if __name__ == "__main__":
    # Example usage
    async def main():
        logging.basicConfig(level=logging.INFO)

        # Initialize pipeline
        pipeline = await initialize_ml_training_pipeline()

        # Force retrain all models
        await pipeline.retrain_all_models()

        # Get performance report
        report = await pipeline.get_model_performance_report()
        print(json.dumps(report, indent=2))

    asyncio.run(main())
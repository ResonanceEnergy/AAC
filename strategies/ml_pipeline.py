"""ML Pipeline — XGBoost/LightGBM alpha prediction with walk-forward training.

Provides a production ML system that generates live alpha predictions
using gradient-boosted trees, with proper walk-forward validation to
prevent look-ahead bias.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class PredictionTarget(Enum):
    DIRECTION = "direction"         # up/down binary
    RETURN_5D = "return_5d"         # 5-day forward return
    RETURN_10D = "return_10d"       # 10-day forward return
    VOLATILITY = "volatility"       # forward realized vol


@dataclass
class FeatureImportance:
    feature: str
    importance: float
    rank: int


@dataclass
class MLPrediction:
    """Single prediction output."""

    ticker: str
    timestamp: str
    signal: float           # continuous signal [-1, +1]
    probability: float      # probability of predicted class (binary)
    predicted_return: float  # predicted forward return
    confidence: float       # model confidence (0-1)
    model_type: str = ""
    features_used: int = 0


@dataclass
class MLModelState:
    """Serializable model state for persistence."""

    model_type: str
    target: str
    train_start: str
    train_end: str
    n_train_samples: int
    n_features: int
    feature_names: list[str] = field(default_factory=list)
    top_features: list[FeatureImportance] = field(default_factory=list)
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    val_sharpe: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "target": self.target,
            "n_train": self.n_train_samples,
            "n_features": self.n_features,
            "train_acc": round(self.train_accuracy, 4),
            "val_acc": round(self.val_accuracy, 4),
            "val_sharpe": round(self.val_sharpe, 4),
            "top_features": [
                {"name": f.feature, "importance": round(f.importance, 4)}
                for f in self.top_features[:10]
            ],
        }


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Standard technical + macro feature set for alpha prediction."""

    @staticmethod
    def build_features(prices: pd.DataFrame, extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build feature matrix from price data.

        Parameters
        ----------
        prices : pd.DataFrame
            Must include at least a 'close' column. May include
            'open', 'high', 'low', 'volume'.
        extra : pd.DataFrame, optional
            Additional features (VIX, spreads, sentiment, etc.)
            to merge on index.
        """
        df = prices.copy()
        close = df["close"] if "close" in df.columns else df.iloc[:, 0]

        # Returns
        for period in [1, 2, 3, 5, 10, 21]:
            df[f"ret_{period}d"] = close.pct_change(period)

        # Momentum
        for period in [5, 10, 21, 63]:
            df[f"mom_{period}d"] = close / close.shift(period) - 1

        # Volatility
        daily_ret = close.pct_change()
        for window in [5, 10, 21, 63]:
            df[f"vol_{window}d"] = daily_ret.rolling(window).std() * np.sqrt(252)

        # RSI
        for period in [14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
            rs = gain / loss.replace(0, 1e-10)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Moving average ratios
        for window in [10, 21, 50, 200]:
            ma = close.rolling(window).mean()
            df[f"ma_ratio_{window}"] = close / ma.replace(0, 1e-10) - 1

        # Bollinger Band width
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_width"] = (2 * std20 / ma20.replace(0, 1e-10))
        df["bb_pctb"] = (close - (ma20 - 2 * std20)) / (4 * std20).replace(0, 1e-10)

        # Volume features
        if "volume" in df.columns:
            vol = df["volume"]
            df["vol_ratio_10"] = vol / vol.rolling(10).mean().replace(0, 1e-10)
            df["vol_ratio_21"] = vol / vol.rolling(21).mean().replace(0, 1e-10)

        # Range features
        if "high" in df.columns and "low" in df.columns:
            df["range_pct"] = (df["high"] - df["low"]) / close.replace(0, 1e-10)
            df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean() / close.replace(0, 1e-10)

        # Merge extra features
        if extra is not None:
            df = df.join(extra, how="left")

        return df

    @staticmethod
    def build_target(
        prices: pd.DataFrame,
        target: PredictionTarget = PredictionTarget.DIRECTION,
        forward_days: int = 5,
    ) -> pd.Series:
        """Build target variable (forward-looking)."""
        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        fwd_ret = close.shift(-forward_days) / close - 1

        if target == PredictionTarget.DIRECTION:
            return (fwd_ret > 0).astype(int)
        if target in (PredictionTarget.RETURN_5D, PredictionTarget.RETURN_10D):
            return fwd_ret
        if target == PredictionTarget.VOLATILITY:
            daily_ret = close.pct_change()
            return daily_ret.rolling(forward_days).std().shift(-forward_days) * np.sqrt(252)
        return fwd_ret


# ---------------------------------------------------------------------------
# ML Pipeline
# ---------------------------------------------------------------------------

class MLPipeline:
    """Walk-forward ML alpha prediction pipeline.

    Parameters
    ----------
    model_type : ModelType
        Which gradient-boosted framework to use.
    target : PredictionTarget
        What to predict (direction, returns, vol).
    train_window : int
        Training window in trading days.
    retrain_every : int
        Retrain frequency in trading days.
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.XGBOOST,
        target: PredictionTarget = PredictionTarget.DIRECTION,
        train_window: int = 252,
        retrain_every: int = 21,
        forward_days: int = 5,
    ) -> None:
        self.model_type = model_type
        self.target = target
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.forward_days = forward_days
        self._model: Any = None
        self._feature_names: list[str] = []
        self._state: Optional[MLModelState] = None
        self._feature_builder = FeatureBuilder()

    # ── Model creation ────────────────────────────────────────────────────

    def _create_model(self, is_classifier: bool = True) -> Any:
        if self.model_type == ModelType.XGBOOST:
            import xgboost as xgb
            if is_classifier:
                return xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                )
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbosity=0,
                random_state=42,
            )

        # LightGBM
        import lightgbm as lgb
        if is_classifier:
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbose=-1,
                random_state=42,
            )
        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            random_state=42,
        )

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        prices: pd.DataFrame,
        extra_features: Optional[pd.DataFrame] = None,
    ) -> MLModelState:
        """Train the model on historical data.

        Parameters
        ----------
        prices : pd.DataFrame
            Must have at least 'close' column, sorted by date.
        extra_features : pd.DataFrame, optional
            Additional features to merge (VIX, credit spreads, etc.).
        """
        features = self._feature_builder.build_features(prices, extra_features)
        target = self._feature_builder.build_target(
            prices, self.target, self.forward_days,
        )

        # Align and drop NaN
        combined = features.join(target.rename("_target"))
        combined = combined.dropna()

        if len(combined) < 50:
            raise ValueError(f"Not enough clean data: {len(combined)} rows (need ≥50)")

        X = combined.drop(columns=["_target"])
        # Remove price columns, keep only features
        exclude = {"open", "high", "low", "close", "volume", "adj_close"}
        feature_cols = [c for c in X.columns if c.lower() not in exclude]
        X = X[feature_cols]
        y = combined["_target"]

        # Train/val split (last 20% for validation)
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        is_classifier = self.target == PredictionTarget.DIRECTION
        self._model = self._create_model(is_classifier)
        self._feature_names = feature_cols

        self._model.fit(X_train, y_train)

        # Evaluate
        train_score = float(self._model.score(X_train, y_train))
        val_score = float(self._model.score(X_val, y_val))

        # Feature importance
        importances = self._model.feature_importances_
        fi = [
            FeatureImportance(feature=f, importance=float(imp), rank=0)
            for f, imp in zip(feature_cols, importances)
        ]
        fi.sort(key=lambda x: x.importance, reverse=True)
        for rank, f in enumerate(fi):
            f.rank = rank + 1

        self._state = MLModelState(
            model_type=self.model_type.value,
            target=self.target.value,
            train_start=str(X_train.index[0]),
            train_end=str(X_train.index[-1]),
            n_train_samples=len(X_train),
            n_features=len(feature_cols),
            feature_names=feature_cols,
            top_features=fi[:20],
            train_accuracy=train_score,
            val_accuracy=val_score,
        )

        _log.info(
            "ml_model_trained model=%s target=%s n_train=%d n_val=%d train_score=%.4f val_score=%.4f",
            self.model_type.value,
            self.target.value,
            len(X_train),
            len(X_val),
            train_score,
            val_score,
        )
        return self._state

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        prices: pd.DataFrame,
        ticker: str = "SPY",
        extra_features: Optional[pd.DataFrame] = None,
    ) -> MLPrediction:
        """Generate a prediction for the latest data point."""
        if self._model is None:
            raise RuntimeError("Call train() before predict()")

        features = self._feature_builder.build_features(prices, extra_features)
        features = features[self._feature_names].dropna()

        if len(features) == 0:
            raise ValueError("No valid feature rows after dropping NaN")

        X_latest = features.iloc[[-1]]
        is_classifier = self.target == PredictionTarget.DIRECTION

        if is_classifier:
            pred_class = int(self._model.predict(X_latest)[0])
            proba = self._model.predict_proba(X_latest)[0]
            signal = 2.0 * proba[1] - 1.0  # map [0,1] → [-1,+1]
            probability = float(max(proba))
            predicted_return = signal * 0.01  # rough approximation
        else:
            pred_value = float(self._model.predict(X_latest)[0])
            signal = np.clip(pred_value * 100, -1.0, 1.0)
            probability = min(abs(signal), 1.0)
            predicted_return = pred_value

        confidence = probability

        return MLPrediction(
            ticker=ticker,
            timestamp=str(features.index[-1]),
            signal=round(float(signal), 4),
            probability=round(probability, 4),
            predicted_return=round(predicted_return, 6),
            confidence=round(confidence, 4),
            model_type=self.model_type.value,
            features_used=len(self._feature_names),
        )

    def predict_batch(
        self,
        prices: pd.DataFrame,
        ticker: str = "SPY",
        extra_features: Optional[pd.DataFrame] = None,
    ) -> list[MLPrediction]:
        """Generate predictions for all rows (for backtesting)."""
        if self._model is None:
            raise RuntimeError("Call train() before predict_batch()")

        features = self._feature_builder.build_features(prices, extra_features)
        features = features[self._feature_names].dropna()

        if len(features) == 0:
            return []

        is_classifier = self.target == PredictionTarget.DIRECTION
        predictions: list[MLPrediction] = []

        if is_classifier:
            pred_classes = self._model.predict(features)
            probas = self._model.predict_proba(features)
            for i in range(len(features)):
                signal = 2.0 * probas[i][1] - 1.0
                predictions.append(MLPrediction(
                    ticker=ticker,
                    timestamp=str(features.index[i]),
                    signal=round(float(signal), 4),
                    probability=round(float(max(probas[i])), 4),
                    predicted_return=round(float(signal * 0.01), 6),
                    confidence=round(float(max(probas[i])), 4),
                    model_type=self.model_type.value,
                    features_used=len(self._feature_names),
                ))
        else:
            pred_values = self._model.predict(features)
            for i in range(len(features)):
                val = float(pred_values[i])
                signal = float(np.clip(val * 100, -1.0, 1.0))
                predictions.append(MLPrediction(
                    ticker=ticker,
                    timestamp=str(features.index[i]),
                    signal=round(signal, 4),
                    probability=round(min(abs(signal), 1.0), 4),
                    predicted_return=round(val, 6),
                    confidence=round(min(abs(signal), 1.0), 4),
                    model_type=self.model_type.value,
                    features_used=len(self._feature_names),
                ))

        return predictions

    # ── Walk-forward training ─────────────────────────────────────────────

    def walk_forward_train(
        self,
        prices: pd.DataFrame,
        extra_features: Optional[pd.DataFrame] = None,
    ) -> list[MLModelState]:
        """Retrain in walk-forward fashion and report per-fold stats."""
        n = len(prices)
        min_required = self.train_window + self.forward_days + 10
        if n < min_required:
            raise ValueError(f"Need ≥{min_required} rows, got {n}")

        states: list[MLModelState] = []
        pos = 0

        while pos + self.train_window + self.forward_days < n:
            chunk = prices.iloc[pos : pos + self.train_window + self.forward_days]
            extra_chunk = None
            if extra_features is not None:
                extra_chunk = extra_features.iloc[pos : pos + self.train_window + self.forward_days]

            try:
                state = self.train(chunk, extra_chunk)
                states.append(state)
            except ValueError as exc:
                _log.warning("wf_fold_skip pos=%d error=%s", pos, str(exc))

            pos += self.retrain_every

        _log.info("walk_forward_complete n_folds=%d", len(states))
        return states

    @property
    def state(self) -> Optional[MLModelState]:
        return self._state

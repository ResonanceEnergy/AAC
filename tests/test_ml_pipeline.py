"""Tests for strategies/ml_pipeline.py — ML Alpha Prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_df():
    """OHLCV DataFrame with 500 rows — required for FeatureBuilder."""
    np.random.seed(77)
    n = 500
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.002, n)),
        "high": close * (1 + abs(np.random.normal(0, 0.005, n))),
        "low": close * (1 - abs(np.random.normal(0, 0.005, n))),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)


class TestFeatureBuilder:
    def test_build_features(self, ohlcv_df):
        from strategies.ml_pipeline import FeatureBuilder
        fb = FeatureBuilder()
        features = fb.build_features(ohlcv_df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert features.shape[1] >= 10

    def test_no_nan_after_dropna(self, ohlcv_df):
        from strategies.ml_pipeline import FeatureBuilder
        fb = FeatureBuilder()
        features = fb.build_features(ohlcv_df)
        # Raw features can have NaN from rolling windows
        cleaned = features.dropna()
        assert cleaned.isna().sum().sum() == 0
        assert len(cleaned) > 0

    def test_feature_names(self, ohlcv_df):
        from strategies.ml_pipeline import FeatureBuilder
        fb = FeatureBuilder()
        features = fb.build_features(ohlcv_df)
        cols = features.columns.tolist()
        assert any("ret" in c.lower() for c in cols)
        assert any("vol" in c.lower() for c in cols)
        assert any("rsi" in c.lower() for c in cols)

    def test_build_target(self, ohlcv_df):
        from strategies.ml_pipeline import FeatureBuilder, PredictionTarget
        target = FeatureBuilder.build_target(ohlcv_df, PredictionTarget.DIRECTION)
        assert isinstance(target, pd.Series)
        assert set(target.dropna().unique()).issubset({0, 1})


class TestMLPipeline:
    def test_train_xgboost(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline, ModelType
        pipe = MLPipeline(model_type=ModelType.XGBOOST)
        state = pipe.train(ohlcv_df)
        assert state is not None
        assert state.n_features > 0
        assert len(state.top_features) > 0

    def test_train_lightgbm(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline, ModelType
        pipe = MLPipeline(model_type=ModelType.LIGHTGBM)
        state = pipe.train(ohlcv_df)
        assert state is not None
        assert state.n_features > 0

    def test_predict_after_train(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline
        pipe = MLPipeline()
        pipe.train(ohlcv_df)
        pred = pipe.predict(ohlcv_df)
        assert pred is not None
        assert -1.0 <= pred.signal <= 1.0
        assert 0.0 <= pred.probability <= 1.0

    def test_predict_before_train(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline
        pipe = MLPipeline()
        with pytest.raises(RuntimeError, match="Call train"):
            pipe.predict(ohlcv_df)

    def test_predict_batch(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline
        pipe = MLPipeline()
        pipe.train(ohlcv_df)
        preds = pipe.predict_batch(ohlcv_df)
        assert isinstance(preds, list)
        assert len(preds) > 0
        for p in preds:
            assert -1.0 <= p.signal <= 1.0

    def test_walk_forward_train(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline
        # Need large dataset: train_window chunks must have ≥50 clean rows after feature engineering
        np.random.seed(88)
        n = 2000
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        big_df = pd.DataFrame({
            "open": close * (1 + np.random.normal(0, 0.002, n)),
            "high": close * (1 + abs(np.random.normal(0, 0.005, n))),
            "low": close * (1 - abs(np.random.normal(0, 0.005, n))),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        }, index=dates)
        pipe = MLPipeline(train_window=500, retrain_every=300)
        states = pipe.walk_forward_train(big_df)
        assert isinstance(states, list)
        assert len(states) >= 1

    def test_prediction_target_direction(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline, PredictionTarget
        pipe = MLPipeline(target=PredictionTarget.DIRECTION)
        pipe.train(ohlcv_df)
        pred = pipe.predict(ohlcv_df)
        assert pred is not None
        assert -1.0 <= pred.signal <= 1.0

    def test_short_series(self):
        from strategies.ml_pipeline import MLPipeline
        short = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        pipe = MLPipeline()
        with pytest.raises(ValueError):
            pipe.train(short)

    def test_state_to_dict(self, ohlcv_df):
        from strategies.ml_pipeline import MLPipeline
        pipe = MLPipeline()
        state = pipe.train(ohlcv_df)
        d = state.to_dict()
        assert "model_type" in d
        assert "n_features" in d
        assert "top_features" in d

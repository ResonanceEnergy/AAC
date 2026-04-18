from __future__ import annotations

"""Market regime detection for adaptive strategy behaviour.

Classifies the current market environment into one of several regimes
so that strategies can adjust their parameters or be paused/activated:

- **TRENDING_UP** — sustained upward momentum
- **TRENDING_DOWN** — sustained downward momentum
- **RANGING** — sideways/consolidating, good for grid/mean-reversion
- **VOLATILE** — high volatility, tighten stops
- **CRISIS** — extreme drawdown/vol, reduce exposure or halt
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

_log = structlog.get_logger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


# Strategy-level recommendations per regime
REGIME_GUIDANCE: dict[MarketRegime, dict[str, Any]] = {
    MarketRegime.TRENDING_UP: {
        "preferred": ["momentum"],
        "reduce": ["mean_reversion"],
        "pause": [],
        "stop_loss_mult": 1.2,   # widen stops in trends
        "take_profit_mult": 1.5,
        "position_size_mult": 1.0,
    },
    MarketRegime.TRENDING_DOWN: {
        "preferred": ["momentum"],
        "reduce": ["dca", "grid"],
        "pause": [],
        "stop_loss_mult": 0.8,   # tighten stops
        "take_profit_mult": 1.0,
        "position_size_mult": 0.7,
    },
    MarketRegime.RANGING: {
        "preferred": ["grid", "mean_reversion", "arbitrage"],
        "reduce": ["momentum"],
        "pause": [],
        "stop_loss_mult": 1.0,
        "take_profit_mult": 1.0,
        "position_size_mult": 1.0,
    },
    MarketRegime.VOLATILE: {
        "preferred": ["grid"],
        "reduce": ["dca", "momentum"],
        "pause": ["mean_reversion"],
        "stop_loss_mult": 0.7,   # tighter stops
        "take_profit_mult": 1.3,
        "position_size_mult": 0.5,
    },
    MarketRegime.CRISIS: {
        "preferred": [],
        "reduce": [],
        "pause": ["grid", "dca", "momentum", "mean_reversion", "arbitrage"],
        "stop_loss_mult": 0.5,
        "take_profit_mult": 1.0,
        "position_size_mult": 0.2,
    },
    MarketRegime.UNKNOWN: {
        "preferred": [],
        "reduce": [],
        "pause": [],
        "stop_loss_mult": 1.0,
        "take_profit_mult": 1.0,
        "position_size_mult": 1.0,
    },
}


@dataclass
class RegimeConfig:
    """Thresholds for regime classification."""

    trend_threshold: float = 0.03       # 3% return over lookback → trending
    volatility_threshold: float = 0.04  # daily vol > 4% std → volatile
    crisis_threshold: float = 0.08      # 8% drop from recent peak → crisis
    lookback_periods: int = 20          # number of price observations
    min_observations: int = 5           # need at least this many to classify


class RegimeDetector:
    """Detect market regime from price history.

    Maintains a rolling window of price observations and classifies
    the current regime based on returns, volatility, and drawdown.

    Usage::

        detector = RegimeDetector()
        detector.update_prices({"BTC": 60000, "ETH": 3000})
        regime = detector.current_regime
        guidance = detector.get_guidance()
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        self._price_history: dict[str, list[float]] = {}  # symbol → [prices]
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._regime_confidence: float = 0.0
        self._metrics: dict[str, float] = {}

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def regime_confidence(self) -> float:
        return self._regime_confidence

    def update_prices(self, prices: dict[str, float]) -> MarketRegime:
        """Ingest new price snapshot and reclassify regime."""
        for symbol, price in prices.items():
            if price <= 0:
                continue
            hist = self._price_history.setdefault(symbol, [])
            hist.append(price)
            # Keep only lookback window
            max_len = self.config.lookback_periods + 5
            if len(hist) > max_len:
                self._price_history[symbol] = hist[-max_len:]

        self._classify()
        return self._current_regime

    def get_guidance(self) -> dict[str, Any]:
        """Return strategy guidance for current regime."""
        guidance = REGIME_GUIDANCE.get(self._current_regime, REGIME_GUIDANCE[MarketRegime.UNKNOWN])
        return {
            "regime": self._current_regime.value,
            "confidence": round(self._regime_confidence, 2),
            **guidance,
            "metrics": self._metrics,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "regime": self._current_regime.value,
            "confidence": round(self._regime_confidence, 2),
            "symbols_tracked": len(self._price_history),
            "metrics": self._metrics,
        }

    def _classify(self) -> None:
        """Classify regime from aggregated price data."""
        returns: list[float] = []
        daily_returns: list[float] = []
        drawdowns: list[float] = []

        for hist in self._price_history.values():
            if len(hist) < self.config.min_observations:
                continue

            window = hist[-self.config.lookback_periods:]
            if len(window) < 2:
                continue

            # Period return
            period_ret = (window[-1] - window[0]) / window[0]
            returns.append(period_ret)

            # Daily returns for volatility
            for i in range(1, len(window)):
                if window[i - 1] > 0:
                    daily_returns.append((window[i] - window[i - 1]) / window[i - 1])

            # Drawdown from peak
            peak = max(window)
            if peak > 0:
                dd = (peak - window[-1]) / peak
                drawdowns.append(dd)

        if not returns:
            self._current_regime = MarketRegime.UNKNOWN
            self._regime_confidence = 0.0
            self._metrics = {}
            return

        avg_return = sum(returns) / len(returns)
        avg_dd = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
        vol = self._std(daily_returns) if daily_returns else 0.0

        self._metrics = {
            "avg_return": round(avg_return, 4),
            "avg_drawdown": round(avg_dd, 4),
            "volatility": round(vol, 4),
            "num_symbols": len(returns),
        }

        # Classification logic (priority: crisis > volatile > trending > ranging)
        if avg_dd >= self.config.crisis_threshold:
            self._current_regime = MarketRegime.CRISIS
            self._regime_confidence = min(avg_dd / self.config.crisis_threshold, 2.0) * 0.5
        elif vol >= self.config.volatility_threshold:
            self._current_regime = MarketRegime.VOLATILE
            self._regime_confidence = min(vol / self.config.volatility_threshold, 2.0) * 0.5
        elif avg_return >= self.config.trend_threshold:
            self._current_regime = MarketRegime.TRENDING_UP
            self._regime_confidence = min(avg_return / self.config.trend_threshold, 2.0) * 0.5
        elif avg_return <= -self.config.trend_threshold:
            self._current_regime = MarketRegime.TRENDING_DOWN
            self._regime_confidence = min(abs(avg_return) / self.config.trend_threshold, 2.0) * 0.5
        else:
            self._current_regime = MarketRegime.RANGING
            self._regime_confidence = 0.6

        _log.debug(
            "regime_detector.classified",
            regime=self._current_regime.value,
            confidence=round(self._regime_confidence, 2),
            avg_return=round(avg_return, 4),
            vol=round(vol, 4),
        )

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

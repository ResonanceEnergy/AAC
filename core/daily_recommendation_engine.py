#!/usr/bin/env python3
"""
AAC Daily Recommendation Engine
=================================

Generates daily trade recommendations by combining:
  1. Technical signal generators (RSI, MA Crossover, MACD, Volume)
  2. Fibonacci / Golden Ratio signals (existing FibSignalGenerator)
  3. Unusual Whales intelligence (flow alerts, dark pool, congress trades)
  4. Signal aggregation + consensus ranking

Output: A ranked list of recommendations formatted as a daily brief.

Usage:
    engine = DailyRecommendationEngine()
    brief = await engine.generate_daily_brief()
    print(brief)

Hooks into:
    - core/autonomous_engine.py (registered as scheduled task)
    - integrations/barren_wuffet_telegram_bot.py (delivery)
"""

import asyncio
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config

logger = logging.getLogger("DailyRecommendation")


# ════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ════════════════════════════════════════════════════════════════════════

class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """A single signal from one generator."""
    generator: str       # e.g. "RSI", "MACD", "UW_Flow"
    symbol: str
    direction: SignalDirection
    confidence: float    # 0.0 - 1.0
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Recommendation:
    """Aggregated recommendation for a single symbol."""
    symbol: str
    direction: SignalDirection
    consensus_score: float    # -1.0 (strong sell) to +1.0 (strong buy)
    confidence: float         # 0.0 - 1.0
    signals: List[Signal] = field(default_factory=list)
    rank: int = 0
    summary: str = ""


# ════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS — Pure math, no external deps
# ════════════════════════════════════════════════════════════════════════

def compute_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index. Returns 0-100 or None if not enough data."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = deltas[-(period):]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_sma(prices: List[float], period: int) -> Optional[float]:
    """Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def compute_ema(prices: List[float], period: int) -> Optional[float]:
    """Exponential Moving Average."""
    if len(prices) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def compute_macd(
    prices: List[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[Tuple[float, float, float]]:
    """MACD line, signal line, histogram. Returns None if not enough data."""
    if len(prices) < slow + signal_period:
        return None
    fast_ema = compute_ema(prices, fast)
    slow_ema = compute_ema(prices, slow)
    if fast_ema is None or slow_ema is None:
        return None
    macd_line = fast_ema - slow_ema

    # Build MACD history for signal line
    macd_vals = []
    for i in range(slow, len(prices)):
        subset = prices[:i + 1]
        f = compute_ema(subset, fast)
        s = compute_ema(subset, slow)
        if f is not None and s is not None:
            macd_vals.append(f - s)

    if len(macd_vals) < signal_period:
        return None
    signal_line = sum(macd_vals[-signal_period:]) / signal_period
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_volume_ratio(volumes: List[float], period: int = 20) -> Optional[float]:
    """Current volume vs average volume ratio. >1.5 = elevated."""
    if len(volumes) < period + 1:
        return None
    avg = sum(volumes[-period - 1:-1]) / period
    if avg == 0:
        return None
    return volumes[-1] / avg


def compute_bollinger_position(
    prices: List[float], period: int = 20, num_std: float = 2.0,
) -> Optional[float]:
    """Where price sits within Bollinger Bands. Returns 0-1 (0=lower, 1=upper)."""
    if len(prices) < period:
        return None
    window = prices[-period:]
    sma = sum(window) / period
    variance = sum((p - sma) ** 2 for p in window) / period
    std = math.sqrt(variance) if variance > 0 else 0.001
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower
    if band_width == 0:
        return 0.5
    return (prices[-1] - lower) / band_width


# ════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS
# ════════════════════════════════════════════════════════════════════════

class RSISignalGenerator:
    """Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)."""

    OVERSOLD = 30.0
    OVERBOUGHT = 70.0
    EXTREME_OVERSOLD = 20.0
    EXTREME_OVERBOUGHT = 80.0

    def generate(self, symbol: str, prices: List[float]) -> Signal:
        rsi = compute_rsi(prices)
        if rsi is None:
            return Signal("RSI", symbol, SignalDirection.HOLD, 0.0, "Insufficient data for RSI")

        if rsi <= self.EXTREME_OVERSOLD:
            return Signal("RSI", symbol, SignalDirection.BUY, 0.85,
                          f"RSI={rsi:.1f} — extreme oversold, strong bounce expected",
                          metadata={"rsi": rsi})
        elif rsi <= self.OVERSOLD:
            return Signal("RSI", symbol, SignalDirection.BUY, 0.65,
                          f"RSI={rsi:.1f} — oversold territory",
                          metadata={"rsi": rsi})
        elif rsi >= self.EXTREME_OVERBOUGHT:
            return Signal("RSI", symbol, SignalDirection.SELL, 0.85,
                          f"RSI={rsi:.1f} — extreme overbought, pullback likely",
                          metadata={"rsi": rsi})
        elif rsi >= self.OVERBOUGHT:
            return Signal("RSI", symbol, SignalDirection.SELL, 0.65,
                          f"RSI={rsi:.1f} — overbought territory",
                          metadata={"rsi": rsi})
        else:
            return Signal("RSI", symbol, SignalDirection.HOLD, 0.3,
                          f"RSI={rsi:.1f} — neutral range",
                          metadata={"rsi": rsi})


class MACrossoverGenerator:
    """Buy on golden cross (short MA > long MA), sell on death cross."""

    def generate(self, symbol: str, prices: List[float]) -> Signal:
        sma_short = compute_sma(prices, 10)
        sma_long = compute_sma(prices, 30)
        if sma_short is None or sma_long is None:
            return Signal("MA_Cross", symbol, SignalDirection.HOLD, 0.0,
                          "Insufficient data for MA crossover")

        spread_pct = (sma_short - sma_long) / sma_long * 100

        if spread_pct > 2.0:
            return Signal("MA_Cross", symbol, SignalDirection.BUY, 0.7,
                          f"SMA10 > SMA30 by {spread_pct:.1f}% — bullish trend",
                          metadata={"sma10": sma_short, "sma30": sma_long, "spread_pct": spread_pct})
        elif spread_pct < -2.0:
            return Signal("MA_Cross", symbol, SignalDirection.SELL, 0.7,
                          f"SMA10 < SMA30 by {abs(spread_pct):.1f}% — bearish trend",
                          metadata={"sma10": sma_short, "sma30": sma_long, "spread_pct": spread_pct})
        else:
            return Signal("MA_Cross", symbol, SignalDirection.HOLD, 0.3,
                          f"MA spread {spread_pct:+.1f}% — no clear trend",
                          metadata={"sma10": sma_short, "sma30": sma_long, "spread_pct": spread_pct})


class MACDSignalGenerator:
    """Buy on bullish MACD crossover (histogram turns positive), sell on bearish."""

    def generate(self, symbol: str, prices: List[float]) -> Signal:
        result = compute_macd(prices)
        if result is None:
            return Signal("MACD", symbol, SignalDirection.HOLD, 0.0,
                          "Insufficient data for MACD")

        macd_line, signal_line, histogram = result

        if histogram > 0 and macd_line > 0:
            return Signal("MACD", symbol, SignalDirection.BUY, 0.7,
                          f"MACD bullish — histogram={histogram:.4f}, line above zero",
                          metadata={"macd": macd_line, "signal": signal_line, "histogram": histogram})
        elif histogram < 0 and macd_line < 0:
            return Signal("MACD", symbol, SignalDirection.SELL, 0.7,
                          f"MACD bearish — histogram={histogram:.4f}, line below zero",
                          metadata={"macd": macd_line, "signal": signal_line, "histogram": histogram})
        else:
            return Signal("MACD", symbol, SignalDirection.HOLD, 0.3,
                          f"MACD mixed — histogram={histogram:.4f}",
                          metadata={"macd": macd_line, "signal": signal_line, "histogram": histogram})


class VolumeSignalGenerator:
    """Confirm trends with volume. High volume + price move = conviction."""

    def generate(self, symbol: str, prices: List[float], volumes: List[float]) -> Signal:
        vol_ratio = compute_volume_ratio(volumes)
        if vol_ratio is None:
            return Signal("Volume", symbol, SignalDirection.HOLD, 0.0,
                          "Insufficient volume data")

        # Price direction over last 3 periods
        if len(prices) < 4:
            return Signal("Volume", symbol, SignalDirection.HOLD, 0.0,
                          "Insufficient price data for volume analysis")

        recent_change = (prices[-1] - prices[-4]) / prices[-4] * 100

        if vol_ratio > 2.0 and recent_change > 1.0:
            return Signal("Volume", symbol, SignalDirection.BUY, 0.75,
                          f"Volume {vol_ratio:.1f}x avg + price up {recent_change:.1f}% — strong buying",
                          metadata={"vol_ratio": vol_ratio, "price_change_pct": recent_change})
        elif vol_ratio > 2.0 and recent_change < -1.0:
            return Signal("Volume", symbol, SignalDirection.SELL, 0.75,
                          f"Volume {vol_ratio:.1f}x avg + price down {recent_change:.1f}% — heavy selling",
                          metadata={"vol_ratio": vol_ratio, "price_change_pct": recent_change})
        elif vol_ratio > 1.5:
            return Signal("Volume", symbol, SignalDirection.HOLD, 0.4,
                          f"Elevated volume {vol_ratio:.1f}x avg but direction unclear",
                          metadata={"vol_ratio": vol_ratio, "price_change_pct": recent_change})
        else:
            return Signal("Volume", symbol, SignalDirection.HOLD, 0.2,
                          f"Normal volume {vol_ratio:.1f}x avg",
                          metadata={"vol_ratio": vol_ratio, "price_change_pct": recent_change})


class BollingerSignalGenerator:
    """Buy near lower band, sell near upper band."""

    def generate(self, symbol: str, prices: List[float]) -> Signal:
        position = compute_bollinger_position(prices)
        if position is None:
            return Signal("Bollinger", symbol, SignalDirection.HOLD, 0.0,
                          "Insufficient data for Bollinger Bands")

        if position < 0.05:
            return Signal("Bollinger", symbol, SignalDirection.BUY, 0.8,
                          f"Price at lower Bollinger Band ({position:.0%}) — oversold bounce likely",
                          metadata={"bb_position": position})
        elif position < 0.2:
            return Signal("Bollinger", symbol, SignalDirection.BUY, 0.6,
                          f"Price near lower Bollinger Band ({position:.0%})",
                          metadata={"bb_position": position})
        elif position > 0.95:
            return Signal("Bollinger", symbol, SignalDirection.SELL, 0.8,
                          f"Price at upper Bollinger Band ({position:.0%}) — overbought",
                          metadata={"bb_position": position})
        elif position > 0.8:
            return Signal("Bollinger", symbol, SignalDirection.SELL, 0.6,
                          f"Price near upper Bollinger Band ({position:.0%})",
                          metadata={"bb_position": position})
        else:
            return Signal("Bollinger", symbol, SignalDirection.HOLD, 0.3,
                          f"Price mid-range Bollinger ({position:.0%})",
                          metadata={"bb_position": position})


# ════════════════════════════════════════════════════════════════════════
# UNUSUAL WHALES INTELLIGENCE LAYER
# ════════════════════════════════════════════════════════════════════════

class UnusualWhalesIntelligence:
    """Pulls options flow, dark pool, congress data from Unusual Whales API."""

    def __init__(self):
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                from integrations.unusual_whales_client import UnusualWhalesClient
                self._client = UnusualWhalesClient()
            except Exception as e:
                logger.warning(f"Unusual Whales client init failed: {e}")
        return self._client

    async def get_flow_signals(self, limit: int = 20) -> List[Signal]:
        """Convert options flow alerts into signals."""
        client = await self._get_client()
        if not client:
            return []
        signals = []
        try:
            alerts = await client.get_flow_alerts(limit=limit)
            # Aggregate by ticker — count bullish vs bearish flow
            ticker_flow: Dict[str, Dict[str, int]] = defaultdict(lambda: {"bullish": 0, "bearish": 0})
            for alert in alerts:
                ticker = alert.get("ticker", alert.get("symbol", ""))
                sentiment = alert.get("sentiment", "").lower()
                if not ticker:
                    continue
                if sentiment in ("bullish", "very_bullish"):
                    ticker_flow[ticker]["bullish"] += 1
                elif sentiment in ("bearish", "very_bearish"):
                    ticker_flow[ticker]["bearish"] += 1

            for ticker, counts in ticker_flow.items():
                total = counts["bullish"] + counts["bearish"]
                if total < 2:
                    continue
                ratio = counts["bullish"] / total
                if ratio > 0.7:
                    signals.append(Signal(
                        "UW_Flow", ticker, SignalDirection.BUY,
                        min(0.8, 0.5 + ratio * 0.3),
                        f"Unusual flow: {counts['bullish']}B/{counts['bearish']}S — bullish skew",
                        metadata={"bullish": counts["bullish"], "bearish": counts["bearish"]},
                    ))
                elif ratio < 0.3:
                    signals.append(Signal(
                        "UW_Flow", ticker, SignalDirection.SELL,
                        min(0.8, 0.5 + (1 - ratio) * 0.3),
                        f"Unusual flow: {counts['bullish']}B/{counts['bearish']}S — bearish skew",
                        metadata={"bullish": counts["bullish"], "bearish": counts["bearish"]},
                    ))
        except Exception as e:
            logger.warning(f"UW flow signals error: {e}")
        return signals

    async def get_darkpool_signals(self, limit: int = 20) -> List[Signal]:
        """Detect large dark pool prints as institutional activity signals."""
        client = await self._get_client()
        if not client:
            return []
        signals = []
        try:
            trades = await client.get_dark_pool(limit=limit)
            # Look for large prints (> $1M notional)
            big_prints: Dict[str, List[Dict]] = defaultdict(list)
            for trade in trades:
                if isinstance(trade, dict):
                    ticker = trade.get("ticker", trade.get("symbol", ""))
                    notional = float(trade.get("premium", trade.get("notional_value", 0)) or 0)
                else:
                    # DarkPoolTrade dataclass from client
                    ticker = trade.ticker
                    notional = trade.notional
                if ticker and notional > 1_000_000:
                    big_prints[ticker].append({"ticker": ticker, "notional": notional})

            for ticker, prints in big_prints.items():
                total_notional = sum(p["notional"] for p in prints)
                signals.append(Signal(
                    "UW_DarkPool", ticker, SignalDirection.HOLD,
                    0.5,
                    f"Dark pool: {len(prints)} large prints totaling ${total_notional/1e6:.1f}M — institutional interest",
                    metadata={"print_count": len(prints), "total_notional": total_notional},
                ))
        except Exception as e:
            logger.warning(f"UW darkpool signals error: {e}")
        return signals

    async def get_congress_signals(self, limit: int = 20) -> List[Signal]:
        """Congress trades as a signal (they have a suspiciously good track record)."""
        client = await self._get_client()
        if not client:
            return []
        signals = []
        try:
            trades = await client.get_congress_trades(limit=limit)
            ticker_activity: Dict[str, Dict[str, int]] = defaultdict(lambda: {"buy": 0, "sell": 0})
            for trade in trades:
                ticker = trade.get("ticker", trade.get("asset", ""))
                tx_type = trade.get("txn_type", trade.get("transaction_type", trade.get("type", ""))).lower()
                if not ticker:
                    continue
                if "purchase" in tx_type or "buy" in tx_type:
                    ticker_activity[ticker]["buy"] += 1
                elif "sale" in tx_type or "sell" in tx_type:
                    ticker_activity[ticker]["sell"] += 1

            for ticker, counts in ticker_activity.items():
                total = counts["buy"] + counts["sell"]
                if total < 2:
                    continue
                if counts["buy"] > counts["sell"]:
                    signals.append(Signal(
                        "UW_Congress", ticker, SignalDirection.BUY, 0.55,
                        f"Congress: {counts['buy']} buys vs {counts['sell']} sells",
                        metadata=counts,
                    ))
                elif counts["sell"] > counts["buy"]:
                    signals.append(Signal(
                        "UW_Congress", ticker, SignalDirection.SELL, 0.55,
                        f"Congress: {counts['sell']} sells vs {counts['buy']} buys",
                        metadata=counts,
                    ))
        except Exception as e:
            logger.warning(f"UW congress signals error: {e}")
        return signals

    async def get_sector_summary(self) -> Dict[str, Any]:
        """Get sector ETF performance for the brief header."""
        client = await self._get_client()
        if not client:
            return {}
        try:
            sectors = await client.get_sector_etfs()
            return {"sectors": sectors}
        except Exception as e:
            logger.warning(f"UW sector summary error: {e}")
            return {}

    async def get_insider_summary(self, limit: int = 10) -> List[Dict]:
        """Get recent insider transactions."""
        client = await self._get_client()
        if not client:
            return []
        try:
            return await client.get_insider_transactions(limit=limit)
        except Exception as e:
            logger.warning(f"UW insider summary error: {e}")
            return []


# ════════════════════════════════════════════════════════════════════════
# SIGNAL AGGREGATOR — Combines all generators into ranked recommendations
# ════════════════════════════════════════════════════════════════════════

class SignalAggregator:
    """
    Aggregates signals across multiple generators for the same symbol.

    Consensus score: weighted average of signal directions.
      BUY = +1, SELL = -1, HOLD = 0, weighted by confidence.
    """

    # Weight multipliers per generator (higher = more trusted)
    WEIGHTS = {
        "Fibonacci": 1.2,
        "RSI": 1.0,
        "MACD": 1.0,
        "MA_Cross": 0.9,
        "Volume": 0.8,
        "Bollinger": 0.9,
        "UW_Flow": 1.1,
        "UW_DarkPool": 0.7,
        "UW_Congress": 0.6,
    }

    def aggregate(self, signals: List[Signal]) -> List[Recommendation]:
        """Group signals by symbol and compute consensus."""
        by_symbol: Dict[str, List[Signal]] = defaultdict(list)
        for sig in signals:
            by_symbol[sig.symbol].append(sig)

        recommendations = []
        for symbol, sigs in by_symbol.items():
            # Compute weighted consensus
            weighted_sum = 0.0
            weight_total = 0.0
            for sig in sigs:
                direction_val = {
                    SignalDirection.BUY: 1.0,
                    SignalDirection.SELL: -1.0,
                    SignalDirection.HOLD: 0.0,
                }[sig.direction]
                gen_weight = self.WEIGHTS.get(sig.generator, 1.0)
                w = sig.confidence * gen_weight
                weighted_sum += direction_val * w
                weight_total += w

            consensus = weighted_sum / weight_total if weight_total > 0 else 0.0

            # Determine final direction
            if consensus > 0.2:
                direction = SignalDirection.BUY
            elif consensus < -0.2:
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.HOLD

            avg_confidence = sum(s.confidence for s in sigs) / len(sigs)

            # Build summary from strongest signals
            strong = sorted(sigs, key=lambda s: s.confidence, reverse=True)[:3]
            summary_parts = [s.reason for s in strong]

            recommendations.append(Recommendation(
                symbol=symbol,
                direction=direction,
                consensus_score=round(consensus, 3),
                confidence=round(avg_confidence, 3),
                signals=sigs,
                summary=" | ".join(summary_parts),
            ))

        # Rank by absolute consensus strength
        recommendations.sort(key=lambda r: abs(r.consensus_score), reverse=True)
        for i, rec in enumerate(recommendations, 1):
            rec.rank = i

        return recommendations


# ════════════════════════════════════════════════════════════════════════
# DAILY BRIEF FORMATTER
# ════════════════════════════════════════════════════════════════════════

class DailyBriefFormatter:
    """Formats recommendations into a readable daily brief."""

    @staticmethod
    def format(
        recommendations: List[Recommendation],
        sector_data: Dict[str, Any],
        insider_data: List[Dict],
        timestamp: Optional[datetime] = None,
    ) -> str:
        ts = timestamp or datetime.now(timezone.utc)
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append(f"  AAC DAILY TRADE BRIEF — {ts.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("  BARREN WUFFET / AZ SUPREME — v2.7.0")
        lines.append("=" * 60)
        lines.append("")

        # Sector overview
        sectors = sector_data.get("sectors", [])
        if sectors:
            lines.append("── SECTOR PERFORMANCE ──────────────────────────────")
            for s in sectors[:12]:
                name = s.get("name", s.get("ticker", "?"))
                change = s.get("change_percent", s.get("performance", "?"))
                if isinstance(change, (int, float)):
                    arrow = "▲" if change > 0 else "▼" if change < 0 else "─"
                    lines.append(f"  {name:20s} {arrow} {change:+.2f}%")
                else:
                    lines.append(f"  {name:20s}   {change}")
            lines.append("")

        # Top recommendations
        actionable = [r for r in recommendations if r.direction != SignalDirection.HOLD]
        holds = [r for r in recommendations if r.direction == SignalDirection.HOLD]

        if actionable:
            lines.append("── TOP RECOMMENDATIONS ─────────────────────────────")
            for rec in actionable[:10]:
                emoji = "🟢 BUY " if rec.direction == SignalDirection.BUY else "🔴 SELL"
                lines.append(
                    f"  #{rec.rank} {emoji} {rec.symbol:12s} "
                    f"consensus={rec.consensus_score:+.2f}  "
                    f"conf={rec.confidence:.0%}  "
                    f"({len(rec.signals)} signals)"
                )
                lines.append(f"     {rec.summary}")
                lines.append("")
        else:
            lines.append("── NO ACTIONABLE SIGNALS ───────────────────────────")
            lines.append("  All indicators neutral — sit tight.")
            lines.append("")

        # Hold/Watch list
        if holds:
            lines.append("── WATCH LIST (HOLD) ───────────────────────────────")
            for rec in holds[:5]:
                lines.append(
                    f"  {rec.symbol:12s} consensus={rec.consensus_score:+.2f}  "
                    f"({len(rec.signals)} signals)"
                )
            lines.append("")

        # Insider activity
        if insider_data:
            lines.append("── INSIDER ACTIVITY ────────────────────────────────")
            for ins in insider_data[:5]:
                name = ins.get("insider_name", ins.get("name", "Unknown"))
                ticker = ins.get("ticker", ins.get("symbol", "?"))
                tx = ins.get("transaction_type", ins.get("type", "?"))
                lines.append(f"  {name}: {tx} {ticker}")
            lines.append("")

        # Footer
        lines.append("── RISK NOTICE ─────────────────────────────────────")
        lines.append("  PAPER TRADING ONLY. Not financial advice.")
        lines.append("  All signals are algorithmic — verify before acting.")
        lines.append("=" * 60)

        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# THE ENGINE — Orchestrates everything
# ════════════════════════════════════════════════════════════════════════

class DailyRecommendationEngine:
    """
    Main entry point. Call generate_daily_brief() for the full pipeline:
      1. Fetch market data from CoinGecko
      2. Run all technical signal generators
      3. Fetch Unusual Whales intelligence
      4. Aggregate and rank
      5. Format as daily brief
    """

    # Coins to track (CoinGecko IDs → display symbols)
    TRACKED_COINS = {
        "bitcoin": "BTC/USD",
        "ethereum": "ETH/USD",
        "solana": "SOL/USD",
        "cardano": "ADA/USD",
        "ripple": "XRP/USD",
    }

    def __init__(self):
        self.rsi_gen = RSISignalGenerator()
        self.ma_gen = MACrossoverGenerator()
        self.macd_gen = MACDSignalGenerator()
        self.volume_gen = VolumeSignalGenerator()
        self.bollinger_gen = BollingerSignalGenerator()
        self.uw_intel = UnusualWhalesIntelligence()
        self.aggregator = SignalAggregator()
        self.formatter = DailyBriefFormatter()

        self._coingecko = None
        self._fib_generator = None

    async def _get_coingecko(self):
        if self._coingecko is None:
            from shared.data_sources import CoinGeckoClient
            self._coingecko = CoinGeckoClient()
            await self._coingecko.connect()
        return self._coingecko

    async def _get_fib_generator(self):
        if self._fib_generator is None:
            from pipeline_runner import FibSignalGenerator
            self._fib_generator = FibSignalGenerator()
        return self._fib_generator

    async def _fetch_coin_data(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """Fetch price + 30d history + volumes for a coin."""
        cg = await self._get_coingecko()
        try:
            chart = await cg.get_coin_market_chart(coin_id, days=30)
            if not chart:
                return None
            prices_raw = chart.get("prices", [])
            volumes_raw = chart.get("total_volumes", [])
            if len(prices_raw) < 15:
                return None

            prices = [p[1] if isinstance(p, list) else p for p in prices_raw]
            volumes = [v[1] if isinstance(v, list) else v for v in volumes_raw]

            tick = await cg.get_price(coin_id)
            current_price = tick.price if tick else prices[-1]
            change_24h = tick.change_24h if tick else 0.0

            return {
                "prices": prices,
                "volumes": volumes,
                "current_price": current_price,
                "high_30d": max(prices),
                "low_30d": min(prices),
                "change_24h": change_24h,
            }
        except Exception as e:
            logger.warning(f"Failed to fetch data for {coin_id}: {e}")
            return None

    async def generate_signals(self) -> List[Signal]:
        """Run all signal generators across all tracked coins."""
        all_signals: List[Signal] = []

        # 1. Technical signals from CoinGecko data
        for coin_id, symbol in self.TRACKED_COINS.items():
            data = await self._fetch_coin_data(coin_id)
            if not data:
                logger.warning(f"No data for {coin_id}, skipping")
                continue

            prices = data["prices"]
            volumes = data["volumes"]

            # RSI
            all_signals.append(self.rsi_gen.generate(symbol, prices))
            # MA Crossover
            all_signals.append(self.ma_gen.generate(symbol, prices))
            # MACD
            all_signals.append(self.macd_gen.generate(symbol, prices))
            # Volume
            if volumes:
                all_signals.append(self.volume_gen.generate(symbol, prices, volumes))
            # Bollinger
            all_signals.append(self.bollinger_gen.generate(symbol, prices))

            # Fibonacci (existing working generator)
            try:
                fib = await self._get_fib_generator()
                fib_result = fib.analyze(
                    symbol=symbol,
                    current_price=data["current_price"],
                    high_30d=data["high_30d"],
                    low_30d=data["low_30d"],
                    prices_30d=prices,
                    change_24h=data["change_24h"],
                )
                fib_signal_str = fib_result.get("signal", "HOLD")
                fib_dir = {
                    "BUY": SignalDirection.BUY,
                    "SELL": SignalDirection.SELL,
                }.get(fib_signal_str, SignalDirection.HOLD)
                all_signals.append(Signal(
                    "Fibonacci", symbol, fib_dir,
                    fib_result.get("confidence", 0.5),
                    fib_result.get("reason", "Fibonacci analysis"),
                    metadata=fib_result,
                ))
            except Exception as e:
                logger.warning(f"Fibonacci analysis failed for {symbol}: {e}")

            # Rate limit between coins
            await asyncio.sleep(0.5)

        # 2. Unusual Whales signals (equities/options — different universe)
        uw_signals = await self.uw_intel.get_flow_signals(limit=50)
        all_signals.extend(uw_signals)

        dp_signals = await self.uw_intel.get_darkpool_signals(limit=50)
        all_signals.extend(dp_signals)

        congress_signals = await self.uw_intel.get_congress_signals(limit=50)
        all_signals.extend(congress_signals)

        return all_signals

    async def generate_recommendations(self) -> List[Recommendation]:
        """Generate + aggregate signals into ranked recommendations."""
        signals = await self.generate_signals()
        return self.aggregator.aggregate(signals)

    async def generate_daily_brief(self) -> str:
        """Full pipeline: data → signals → aggregate → format."""
        logger.info("Generating daily recommendation brief...")

        recommendations = await self.generate_recommendations()

        # Supplementary data for the brief
        sector_data = await self.uw_intel.get_sector_summary()
        insider_data = await self.uw_intel.get_insider_summary(limit=10)

        brief = self.formatter.format(
            recommendations=recommendations,
            sector_data=sector_data,
            insider_data=insider_data,
        )

        # Save to file
        brief_path = Path(PROJECT_ROOT) / "reports" / "daily_brief.txt"
        brief_path.parent.mkdir(exist_ok=True)
        brief_path.write_text(brief, encoding="utf-8")
        logger.info(f"Brief saved to {brief_path}")

        return brief

    async def cleanup(self):
        """Clean up connections."""
        if self._coingecko and hasattr(self._coingecko, "disconnect"):
            try:
                await self._coingecko.disconnect()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ════════════════════════════════════════════════════════════════════════

async def main():
    """Run the daily recommendation engine standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    engine = DailyRecommendationEngine()
    try:
        brief = await engine.generate_daily_brief()
        print(brief)
    finally:
        await engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

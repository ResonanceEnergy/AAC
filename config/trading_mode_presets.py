"""
Trading Mode Presets — BARREN WUFFET Configuration
===================================================

Predefined trading mode configurations for different strategies:
- PAPER: Safe mode, no real orders
- DCA: Dollar-cost averaging automation
- GRID: Grid trading with range parameters
- CONTRARIAN: Contrarian signals (buy fear, sell greed)
- MOMENTUM: Trend-following with breakout detection
- ARBITRAGE: Cross-venue price differential capture

Each preset overrides default parameters in the orchestrator.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TradingMode(Enum):
    """Available trading mode presets."""
    PAPER = "paper"
    DCA = "dca"
    GRID = "grid"
    CONTRARIAN = "contrarian"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    CUSTOM = "custom"


@dataclass
class ModePreset:
    """Configuration preset for a trading mode."""
    mode: TradingMode
    name: str
    description: str
    live_enabled: bool = False
    max_position_pct: float = 5.0       # Max % of portfolio per position
    max_daily_trades: int = 50
    max_drawdown_pct: float = 10.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    cooldown_minutes: int = 5
    allowed_venues: List[str] = field(default_factory=lambda: ["ndax", "kraken"])
    allowed_pairs: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "mode": self.mode.value,
            "name": self.name,
            "description": self.description,
            "live_enabled": self.live_enabled,
            "max_position_pct": self.max_position_pct,
            "max_daily_trades": self.max_daily_trades,
            "max_drawdown_pct": self.max_drawdown_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "cooldown_minutes": self.cooldown_minutes,
            "allowed_venues": self.allowed_venues,
            "allowed_pairs": self.allowed_pairs,
            **self.custom_params,
        }


# ── Built-in Presets ───────────────────────────────────────────────────

PRESETS: Dict[TradingMode, ModePreset] = {
    TradingMode.PAPER: ModePreset(
        mode=TradingMode.PAPER,
        name="Paper Trading",
        description="Simulated trading with no real orders. Safe for testing.",
        live_enabled=False,
        max_position_pct=10.0,
        max_daily_trades=200,
        max_drawdown_pct=50.0,
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        cooldown_minutes=0,
    ),
    TradingMode.DCA: ModePreset(
        mode=TradingMode.DCA,
        name="Dollar-Cost Averaging",
        description="Time-based recurring buys at fixed intervals.",
        live_enabled=True,
        max_position_pct=2.0,
        max_daily_trades=5,
        max_drawdown_pct=15.0,
        stop_loss_pct=0.0,  # DCA doesn't use stop loss
        take_profit_pct=0.0,
        cooldown_minutes=60,
        custom_params={
            "interval_hours": 24,
            "amount_per_buy_usd": 50.0,
            "buy_on_dip_threshold_pct": 5.0,
        },
    ),
    TradingMode.GRID: ModePreset(
        mode=TradingMode.GRID,
        name="Grid Trading",
        description="Place buy/sell orders at regular price intervals within a range.",
        live_enabled=True,
        max_position_pct=3.0,
        max_daily_trades=100,
        max_drawdown_pct=20.0,
        stop_loss_pct=15.0,
        take_profit_pct=0.0,  # Grid exits at grid levels
        cooldown_minutes=1,
        custom_params={
            "grid_levels": 10,
            "grid_range_pct": 10.0,
            "order_size_usd": 25.0,
            "geometric": False,
        },
    ),
    TradingMode.CONTRARIAN: ModePreset(
        mode=TradingMode.CONTRARIAN,
        name="Contrarian",
        description="Buy extreme fear, sell extreme greed. Uses Fear & Greed Index.",
        live_enabled=True,
        max_position_pct=5.0,
        max_daily_trades=10,
        max_drawdown_pct=15.0,
        stop_loss_pct=8.0,
        take_profit_pct=15.0,
        cooldown_minutes=120,
        custom_params={
            "fear_buy_threshold": 20,     # Buy when F&G < 20
            "greed_sell_threshold": 80,    # Sell when F&G > 80
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        },
    ),
    TradingMode.MOMENTUM: ModePreset(
        mode=TradingMode.MOMENTUM,
        name="Momentum / Trend Following",
        description="Follow trends with breakout detection and trailing stops.",
        live_enabled=True,
        max_position_pct=5.0,
        max_daily_trades=20,
        max_drawdown_pct=12.0,
        stop_loss_pct=3.0,
        take_profit_pct=0.0,  # Uses trailing stop
        cooldown_minutes=15,
        custom_params={
            "ema_fast": 9,
            "ema_slow": 21,
            "breakout_lookback": 20,
            "trailing_stop_pct": 2.5,
            "volume_confirmation": True,
        },
    ),
    TradingMode.ARBITRAGE: ModePreset(
        mode=TradingMode.ARBITRAGE,
        name="Arbitrage",
        description="Cross-venue price differential capture. Requires multi-exchange accounts.",
        live_enabled=True,
        max_position_pct=8.0,
        max_daily_trades=200,
        max_drawdown_pct=5.0,
        stop_loss_pct=0.5,
        take_profit_pct=0.3,
        cooldown_minutes=0,
        allowed_venues=["ndax", "kraken", "coinbase", "bybit"],
        custom_params={
            "min_spread_pct": 0.15,
            "max_execution_time_ms": 2000,
            "include_dex": True,
            "slippage_tolerance_pct": 0.1,
        },
    ),
}


def get_preset(mode: TradingMode) -> ModePreset:
    """Get a trading mode preset."""
    return PRESETS.get(mode, PRESETS[TradingMode.PAPER])


def list_presets() -> List[Dict[str, str]]:
    """List all available presets with summary."""
    return [
        {"mode": p.mode.value, "name": p.name, "description": p.description, "live": str(p.live_enabled)}
        for p in PRESETS.values()
    ]

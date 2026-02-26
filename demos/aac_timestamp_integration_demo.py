"""
AAC Timestamp Integration Example

This example demonstrates how to integrate the AAC timestamp converter
into arbitrage operations for proper timing and signal management.
"""

from aac_timestamp_converter import (
    AACTimestampConverter,
    AACArbitrageTiming,
    epoch_to_human,
    human_to_epoch,
    current_epoch
)
from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class ArbitrageSignal:
    """AAC arbitrage signal with timestamp information."""
    symbol: str
    expected_return: float
    confidence_score: float
    timestamp_epoch: int
    source: str
    market_hours: bool = False

    def __post_init__(self):
        """Automatically determine if signal was generated during market hours."""
        signal_time = AACTimestampConverter.epoch_to_datetime(self.timestamp_epoch)
        self.market_hours = AACArbitrageTiming.is_market_hours(signal_time)

    @property
    def timestamp_human(self) -> str:
        """Get human-readable timestamp."""
        return epoch_to_human(self.timestamp_epoch)

    @property
    def timestamp_arbitrage_format(self) -> str:
        """Get timestamp formatted for AAC arbitrage display."""
        return AACArbitrageTiming.format_arbitrage_timestamp(self.timestamp_epoch)

    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if signal is older than specified age."""
        current = current_epoch()
        return (current - self.timestamp_epoch) > max_age_seconds


class AACArbitrageEngine:
    """Example AAC arbitrage engine with timestamp integration."""

    def __init__(self):
        self.signals: List[ArbitrageSignal] = []
        self.last_update = current_epoch()

    def generate_signal(self, symbol: str, expected_return: float,
                       confidence: float, source: str) -> ArbitrageSignal:
        """Generate a new arbitrage signal with current timestamp."""
        signal = ArbitrageSignal(
            symbol=symbol,
            expected_return=expected_return,
            confidence_score=confidence,
            timestamp_epoch=current_epoch(),
            source=source
        )
        self.signals.append(signal)
        return signal

    def get_recent_signals(self, max_age_seconds: int = 3600) -> List[ArbitrageSignal]:
        """Get signals that are not stale."""
        return [s for s in self.signals if not s.is_stale(max_age_seconds)]

    def get_market_hours_signals(self) -> List[ArbitrageSignal]:
        """Get signals generated during market hours."""
        return [s for s in self.signals if s.market_hours]

    def get_signal_summary(self) -> Dict:
        """Get summary of current signals."""
        recent_signals = self.get_recent_signals()
        market_signals = self.get_market_hours_signals()

        return {
            "total_signals": len(self.signals),
            "recent_signals": len(recent_signals),
            "market_hours_signals": len(market_signals),
            "last_update": epoch_to_human(self.last_update),
            "current_time": epoch_to_human(current_epoch()),
            "is_market_open": AACArbitrageTiming.is_market_hours()
        }


def demonstrate_timestamp_integration():
    """Demonstrate AAC timestamp converter integration."""
    print("AAC Timestamp Integration Demo")
    print("=" * 50)

    # Initialize arbitrage engine
    engine = AACArbitrageEngine()

    # Generate some sample signals
    signals_data = [
        ("AAPL", 2.5, 0.85, "multi_source_arbitrage"),
        ("TSLA", 1.8, 0.92, "binance_arbitrage"),
        ("GOOGL", 3.2, 0.78, "polygon_arbitrage"),
        ("MSFT", 1.5, 0.88, "world_bank_integration")
    ]

    print("\nGenerating Arbitrage Signals:")
    print("-" * 30)

    for symbol, ret, conf, source in signals_data:
        signal = engine.generate_signal(symbol, ret, conf, source)
        print(f"📊 {signal.symbol}: {signal.expected_return:.1f}% return")
        print(f"   Confidence: {signal.confidence_score:.1%}")
        print(f"   Time: {signal.timestamp_arbitrage_format}")
        print(f"   Source: {signal.source}")
        print()

        # Simulate time passing
        time.sleep(0.1)

    # Show summary
    print("Signal Summary:")
    print("-" * 20)
    summary = engine.get_signal_summary()
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print(f"\nNext Market Open: {epoch_to_human(AACTimestampConverter.datetime_to_epoch(AACArbitrageTiming.get_next_market_open()))}")

    # Demonstrate timestamp conversions
    print("\nTimestamp Conversion Examples:")
    print("-" * 35)

    test_timestamps = [
        current_epoch(),  # Current time
        current_epoch(AACTimestampConverter.MILLISECONDS),  # Milliseconds
        human_to_epoch("2026-02-06 09:30:00"),  # Next market open
        human_to_epoch("2026-12-31 23:59:59")  # Year end
    ]

    for ts in test_timestamps:
        precision = AACTimestampConverter.detect_precision(ts)
        human_readable = epoch_to_human(ts)
        print(f"Epoch: {ts} ({precision}x precision)")
        print(f"Human: {human_readable}")
        print()


if __name__ == "__main__":
    demonstrate_timestamp_integration()
# AAC Timestamp Converter Documentation

## Overview

The AAC Timestamp Converter provides comprehensive Unix epoch timestamp conversion utilities specifically designed for the Accelerated Arbitrage Corp (AAC) arbitrage system. This module handles timestamp conversions for market data, arbitrage signals, and time-sensitive financial operations.

## Key Features

- **Multi-Precision Support**: Handles timestamps in seconds, milliseconds, microseconds, and nanoseconds
- **Automatic Precision Detection**: Intelligently detects timestamp precision from magnitude
- **Timezone-Aware**: Full UTC and timezone support for global arbitrage operations
- **AAC-Specific Integration**: Specialized utilities for market hours detection and arbitrage timing
- **Human-Readable Formats**: Converts timestamps to various human-readable formats

## Quick Start

```python
from aac_timestamp_converter import (
    epoch_to_human,
    human_to_epoch,
    current_epoch,
    AACArbitrageTiming
)

# Get current timestamp
now = current_epoch()
print(f"Current time: {epoch_to_human(now)}")

# Convert human date to epoch
epoch_time = human_to_epoch("2026-02-06 09:30:00")
print(f"Market open: {epoch_to_human(epoch_time)}")

# Check market hours
is_open = AACArbitrageTiming.is_market_hours()
print(f"Market open: {is_open}")
```

## Core Classes

### AACTimestampConverter

Main conversion utility class with static methods for all timestamp operations.

**Key Methods:**
- `epoch_to_datetime()` - Convert epoch to datetime object
- `datetime_to_epoch()` - Convert datetime to epoch timestamp
- `epoch_to_string()` - Convert epoch to formatted string
- `string_to_epoch()` - Convert formatted string to epoch
- `detect_precision()` - Auto-detect timestamp precision
- `get_current_epoch()` - Get current timestamp
- `format_duration()` - Format seconds to human-readable duration

### AACArbitrageTiming

Specialized timing utilities for financial markets and arbitrage operations.

**Key Methods:**
- `is_market_hours()` - Check if current time is during market hours (9:30 AM - 4:00 PM ET)
- `get_next_market_open()` - Get next market open time
- `format_arbitrage_timestamp()` - Format timestamp with market status indicator

## Usage Examples

### Basic Conversions

```python
# Epoch to human-readable
timestamp = 1770334235
human_time = epoch_to_human(timestamp)
# Output: "2026-02-05 23:30:35 UTC"

# Human-readable to epoch
date_str = "2026-02-06 09:30:00"
epoch_time = human_to_epoch(date_str)
# Output: 1770370200

# Different precisions
ms_timestamp = 1770334235628  # milliseconds
human_time = epoch_to_human(ms_timestamp)
# Auto-detects millisecond precision
```

### Arbitrage Signal Timing

```python
from aac_timestamp_integration_demo import ArbitrageSignal

# Create signal with automatic timestamp
signal = ArbitrageSignal(
    symbol="AAPL",
    expected_return=2.5,
    confidence_score=0.85,
    timestamp_epoch=current_epoch(),
    source="multi_source_arbitrage"
)

print(signal.timestamp_arbitrage_format)
# Output: "2026-02-05 23:30:35 UTC | ⏰ AFTER HOURS"

# Check if signal is stale
if signal.is_stale(max_age_seconds=300):  # 5 minutes
    print("Signal is too old")
```

### Market Hours Detection

```python
# Check current market status
is_open = AACArbitrageTiming.is_market_hours()
print(f"Market is {'open' if is_open else 'closed'}")

# Get next market open time
next_open = AACArbitrageTiming.get_next_market_open()
print(f"Next open: {epoch_to_human(next_open)}")
```

## Integration with AAC System

The timestamp converter integrates seamlessly with existing AAC components:

- **World Bank Integration**: Timestamp economic indicators and arbitrage signals
- **Reddit Sentiment Analysis**: Timestamp sentiment data collection
- **Multi-Source Arbitrage**: Synchronize timing across different data sources
- **Monitoring Dashboard**: Display timestamps in arbitrage-friendly formats

## Dependencies

None required - uses only Python standard library modules:
- `datetime`
- `time`
- `calendar`
- `typing`

## Error Handling

The converter includes robust error handling for:
- Invalid timestamp formats
- Unsupported precision values
- Timezone conversion issues
- Date parsing errors

## Performance Notes

- All operations are lightweight and fast
- No external API calls or network dependencies
- Thread-safe for concurrent arbitrage operations
- Memory efficient with minimal object creation

## Testing

Run the integration demo to verify functionality:

```bash
python aac_timestamp_integration_demo.py
```

This will demonstrate all major features and provide example output for validation.
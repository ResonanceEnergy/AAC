"""
AAC Timestamp Utilities - Epoch & Unix Timestamp Conversion Tools

This module provides comprehensive timestamp conversion utilities for the AAC arbitrage system,
supporting epoch timestamps in seconds, milliseconds, microseconds, and nanoseconds.

Key Features:
- Convert epoch timestamps to human-readable dates
- Convert human-readable dates to epoch timestamps
- Handle multiple timestamp precisions
- Timezone-aware conversions
- Integration with AAC arbitrage signal timing
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Union, Optional, Tuple
import calendar


class AACTimestampConverter:
    """
    Comprehensive timestamp conversion utilities for AAC arbitrage system.

    Supports Unix epoch timestamps (seconds since 1970-01-01 00:00:00 UTC)
    in multiple precisions: seconds, milliseconds, microseconds, nanoseconds.
    """

    # Epoch reference point
    EPOCH_START = datetime(1970, 1, 1, tzinfo=timezone.utc)

    # Precision constants
    SECONDS = 1
    MILLISECONDS = 1000
    MICROSECONDS = 1000000
    NANOSECONDS = 1000000000

    @staticmethod
    def detect_precision(timestamp: Union[int, float, str]) -> int:
        """
        Detect the precision of an epoch timestamp based on its magnitude.

        Args:
            timestamp: The timestamp value to analyze

        Returns:
            Precision constant (SECONDS, MILLISECONDS, MICROSECONDS, or NANOSECONDS)
        """
        if isinstance(timestamp, str):
            timestamp = float(timestamp)

        # Convert to string to check digit count
        ts_str = str(int(timestamp))

        if len(ts_str) <= 10:  # Up to 10 digits = seconds
            return AACTimestampConverter.SECONDS
        elif len(ts_str) <= 13:  # 11-13 digits = milliseconds
            return AACTimestampConverter.MILLISECONDS
        elif len(ts_str) <= 16:  # 14-16 digits = microseconds
            return AACTimestampConverter.MICROSECONDS
        else:  # 17+ digits = nanoseconds
            return AACTimestampConverter.NANOSECONDS

    @classmethod
    def epoch_to_datetime(cls, timestamp: Union[int, float, str],
                         precision: Optional[int] = None,
                         tz: Optional[timezone] = None) -> datetime:
        """
        Convert epoch timestamp to datetime object.

        Args:
            timestamp: Epoch timestamp (seconds, milliseconds, microseconds, or nanoseconds)
            precision: Timestamp precision (auto-detected if None)
            tz: Target timezone (UTC if None)

        Returns:
            datetime object
        """
        if isinstance(timestamp, str):
            timestamp = float(timestamp)

        if precision is None:
            precision = cls.detect_precision(timestamp)

        # Convert to seconds
        seconds = timestamp / precision

        # Create datetime from epoch
        dt = cls.EPOCH_START + timedelta(seconds=seconds)

        # Convert to target timezone if specified
        if tz is not None:
            dt = dt.astimezone(tz)

        return dt

    @classmethod
    def datetime_to_epoch(cls, dt: datetime,
                         precision: int = SECONDS) -> Union[int, float]:
        """
        Convert datetime object to epoch timestamp.

        Args:
            dt: datetime object to convert
            precision: Desired timestamp precision

        Returns:
            Epoch timestamp in specified precision
        """
        # Ensure datetime is UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        # Calculate seconds since epoch
        seconds = (dt - cls.EPOCH_START).total_seconds()

        # Convert to desired precision
        return int(seconds * precision)

    @classmethod
    def epoch_to_string(cls, timestamp: Union[int, float, str],
                       precision: Optional[int] = None,
                       format_str: str = "%Y-%m-%d %H:%M:%S UTC",
                       tz: Optional[timezone] = None) -> str:
        """
        Convert epoch timestamp to formatted string.

        Args:
            timestamp: Epoch timestamp
            precision: Timestamp precision (auto-detected if None)
            format_str: strftime format string
            tz: Target timezone (UTC if None)

        Returns:
            Formatted date/time string
        """
        dt = cls.epoch_to_datetime(timestamp, precision, tz)
        return dt.strftime(format_str)

    @classmethod
    def string_to_epoch(cls, date_string: str,
                       precision: int = SECONDS,
                       format_str: str = "%Y-%m-%d %H:%M:%S") -> int:
        """
        Convert formatted date string to epoch timestamp.

        Args:
            date_string: Date string to parse
            precision: Desired timestamp precision
            format_str: strptime format string

        Returns:
            Epoch timestamp
        """
        # Parse the string to datetime
        if format_str == "auto":
            # Try common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%d-%m-%Y %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y"
            ]
            dt = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_string, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                raise ValueError(f"Could not parse date string: {date_string}")
        else:
            dt = datetime.strptime(date_string, format_str)

        # Assume UTC if no timezone specified
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return cls.datetime_to_epoch(dt, precision)

    @classmethod
    def get_current_epoch(cls, precision: int = SECONDS) -> Union[int, float]:
        """
        Get current epoch timestamp.

        Args:
            precision: Desired timestamp precision

        Returns:
            Current epoch timestamp
        """
        return cls.datetime_to_epoch(datetime.now(timezone.utc), precision)

    @classmethod
    def format_duration(cls, seconds: Union[int, float]) -> str:
        """
        Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        seconds = int(seconds)

        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes} minutes {remaining_seconds} seconds"
        elif seconds < 86400:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{hours} hours {remaining_minutes} minutes"
        else:
            days = seconds // 86400
            remaining_hours = (seconds % 86400) // 3600
            return f"{days} days {remaining_hours} hours"

    @classmethod
    def get_epoch_ranges(cls, year: Optional[int] = None,
                        month: Optional[int] = None,
                        day: Optional[int] = None) -> Tuple[int, int]:
        """
        Get epoch timestamps for start and end of specified time period.

        Args:
            year: Year (current year if None)
            month: Month (current month if None)
            day: Day (current day if None)

        Returns:
            Tuple of (start_epoch, end_epoch)
        """
        now = datetime.now(timezone.utc)

        if year is None:
            year = now.year
        if month is None:
            month = now.month
        if day is None:
            day = now.day

        if day is not None:
            # Specific day
            start = datetime(year, month, day, tzinfo=timezone.utc)
            end = start + timedelta(days=1)
        elif month is not None:
            # Specific month
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            if month == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        else:
            # Specific year
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

        return (cls.datetime_to_epoch(start), cls.datetime_to_epoch(end))


# AAC-specific timestamp utilities
class AACArbitrageTiming:
    """
    Specialized timing utilities for AAC arbitrage operations.
    """

    @staticmethod
    def is_market_hours(dt: Optional[datetime] = None,
                       timezone_str: str = "US/Eastern") -> bool:
        """
        Check if given datetime is during market hours (9:30 AM - 4:00 PM ET).

        Args:
            dt: datetime to check (current time if None)
            timezone_str: Timezone string

        Returns:
            True if during market hours
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Convert to Eastern Time
        eastern = timezone(timedelta(hours=-5))  # EST (adjust for DST as needed)
        dt_et = dt.astimezone(eastern)

        # Check if weekday and within hours
        if dt_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= dt_et <= market_close

    @staticmethod
    def get_next_market_open(dt: Optional[datetime] = None) -> datetime:
        """
        Get the next market open time.

        Args:
            dt: Starting datetime (current time if None)

        Returns:
            Next market open datetime
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        eastern = timezone(timedelta(hours=-5))
        dt_et = dt.astimezone(eastern)

        # If it's a weekend, move to Monday
        days_ahead = 0
        if dt_et.weekday() >= 5:  # Weekend
            days_ahead = 7 - dt_et.weekday()

        next_open = dt_et + timedelta(days=days_ahead)
        next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)

        # If we're already past today's open, move to next day
        if dt_et >= next_open and days_ahead == 0:
            next_open = next_open + timedelta(days=1)
            if next_open.weekday() >= 5:  # Would be weekend
                next_open = next_open + timedelta(days=7 - next_open.weekday())

        return next_open.astimezone(timezone.utc)

    @staticmethod
    def format_arbitrage_timestamp(timestamp: Union[int, float, str]) -> str:
        """
        Format timestamp for arbitrage signal display.

        Args:
            timestamp: Epoch timestamp

        Returns:
            Formatted string for AAC display
        """
        dt = AACTimestampConverter.epoch_to_datetime(timestamp)
        market_hours = AACArbitrageTiming.is_market_hours(dt)

        format_str = "%Y-%m-%d %H:%M:%S UTC"
        time_str = dt.strftime(format_str)

        status = "📈 MARKET OPEN" if market_hours else "⏰ AFTER HOURS"
        return f"{time_str} | {status}"


# Convenience functions for direct use
def epoch_to_human(timestamp: Union[int, float, str],
                  precision: Optional[int] = None) -> str:
    """Convert epoch timestamp to human-readable string."""
    return AACTimestampConverter.epoch_to_string(timestamp, precision)


def human_to_epoch(date_string: str,
                  precision: int = AACTimestampConverter.SECONDS) -> int:
    """Convert human-readable date string to epoch timestamp."""
    return AACTimestampConverter.string_to_epoch(date_string, precision)


def current_epoch(precision: int = AACTimestampConverter.SECONDS) -> Union[int, float]:
    """Get current epoch timestamp."""
    return AACTimestampConverter.get_current_epoch(precision)


if __name__ == "__main__":
    # Demo and testing
    print("AAC Timestamp Converter Demo")
    print("=" * 40)

    # Current time
    now = current_epoch()
    print(f"Current epoch: {now}")
    print(f"Current human: {epoch_to_human(now)}")
    print(f"Arbitrage format: {AACArbitrageTiming.format_arbitrage_timestamp(now)}")

    # Test conversions
    test_date = "2026-02-05 23:30:00"
    epoch_test = human_to_epoch(test_date)
    print(f"\nTest conversion:")
    print(f"Human: {test_date}")
    print(f"Epoch: {epoch_test}")
    print(f"Back to human: {epoch_to_human(epoch_test)}")

    # Market hours check
    print(f"\nMarket hours: {AACArbitrageTiming.is_market_hours()}")
    next_open = AACArbitrageTiming.get_next_market_open()
    print(f"Next market open: {epoch_to_human(AACTimestampConverter.datetime_to_epoch(next_open))}")
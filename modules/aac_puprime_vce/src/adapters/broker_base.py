"""Base broker adapter interface for future API integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BrokerAdapter(ABC):
    """Abstract base class for broker execution adapters.

    For now, only the manual adapter is implemented.
    Future: wire MT5 API, PU Prime API, etc.
    """

    @abstractmethod
    def get_account_info(self) -> dict[str, Any]:
        """Return account balance, equity, margin info."""

    @abstractmethod
    def place_order(
        self, symbol: str, side: str, size: float,
        stop: float, take_profit: float,
    ) -> dict[str, Any]:
        """Place an order. Returns order confirmation dict."""

    @abstractmethod
    def close_position(self, symbol: str) -> dict[str, Any]:
        """Close an open position."""

    @abstractmethod
    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return list of open positions."""

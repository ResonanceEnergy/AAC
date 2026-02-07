#!/usr/bin/env python3
"""
Base Exchange Connector
=======================
Abstract base class defining the unified exchange interface.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

# Import audit logger
try:
    from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


class ExchangeError(Exception):
    """Base exception for exchange-related errors"""
    pass


class ConnectionError(ExchangeError):
    """Failed to connect to exchange"""
    pass


class AuthenticationError(ExchangeError):
    """Authentication failed"""
    pass


class InsufficientFundsError(ExchangeError):
    """Insufficient funds for order"""
    pass


class OrderError(ExchangeError):
    """Order placement/modification failed"""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded"""
    pass


@dataclass
class Ticker:
    """Market ticker data"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        return (self.spread / self.bid) * 100 if self.bid > 0 else 0


@dataclass
class OrderBook:
    """Order book snapshot"""
    symbol: str
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def best_bid(self) -> Optional[tuple]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[tuple]:
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid[0] + self.best_ask[0]) / 2
        return 0.0


@dataclass
class Balance:
    """Account balance for a single asset"""
    asset: str
    free: float
    locked: float
    
    @property
    def total(self) -> float:
        return self.free + self.locked


@dataclass
class ExchangeOrder:
    """Exchange order representation"""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', etc.
    quantity: float
    price: Optional[float]
    status: str  # 'open', 'filled', 'cancelled', etc.
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fee: float = 0.0
    fee_currency: str = ''
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    raw: Dict = field(default_factory=dict)


class BaseExchangeConnector(ABC):
    """
    Abstract base class for exchange connectors.
    
    All exchange-specific implementations must inherit from this
    and implement the abstract methods.
    """

    def __init__(
        self,
        api_key: str = '',
        api_secret: str = '',
        passphrase: str = '',  # For Coinbase
        testnet: bool = True,
        rate_limit: int = 1200,  # requests per minute
        enable_audit: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        self.rate_limit = rate_limit
        self.enable_audit = enable_audit and AUDIT_AVAILABLE
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = None
        self._connected = False
        self._last_request_time = 0
        
        # Get audit logger if enabled
        self._audit = get_audit_logger() if self.enable_audit else None
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name"""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.
        Returns True if successful.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the exchange."""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Ticker object with current market data
        """
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair
            limit: Number of levels to fetch
            
        Returns:
            OrderBook object
        """
        pass

    @abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        """
        Get account balances.
        
        Returns:
            Dict mapping asset name to Balance object
        """
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            quantity: Amount to trade
            price: Price for limit orders
            client_order_id: Optional custom order ID
            
        Returns:
            ExchangeOrder object
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair
            
        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """
        Get order details.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair
            
        Returns:
            ExchangeOrder object
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional filter by symbol
            
        Returns:
            List of ExchangeOrder objects
        """
        pass

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """
        Get trading fees for a symbol.
        
        Returns:
            Dict with 'maker' and 'taker' fee rates
        """
        # Default implementation - override in subclasses
        return {'maker': 0.001, 'taker': 0.001}

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """
        Create a stop-loss order on the exchange.
        
        This places an actual stop-loss or stop-limit order with the exchange,
        not a client-side simulation.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'sell' for long positions, 'buy' for short positions
            quantity: Amount to trade
            stop_price: Trigger price for the stop order
            limit_price: Optional limit price (for stop-limit orders).
                        If None, uses a market stop order.
            client_order_id: Optional custom order ID
            
        Returns:
            ExchangeOrder object representing the stop-loss order
            
        Raises:
            NotImplementedError: If exchange doesn't support stop orders
            OrderError: If order creation fails
        """
        raise NotImplementedError(
            f"{self.name} connector has not implemented stop-loss orders. "
            "Override create_stop_loss_order() in the exchange-specific connector."
        )

    async def create_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        stop_limit_price: float,
        take_profit_price: float,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, ExchangeOrder]:
        """
        Create an OCO (One-Cancels-Other) order with stop-loss and take-profit.
        
        This places both a stop-loss and a take-profit order, where filling
        one automatically cancels the other.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'sell' for long positions, 'buy' for short positions
            quantity: Amount to trade
            stop_price: Stop-loss trigger price
            stop_limit_price: Stop-loss limit price
            take_profit_price: Take-profit limit price
            client_order_id: Optional custom order ID prefix
            
        Returns:
            Dict with 'stop_loss' and 'take_profit' ExchangeOrder objects
            
        Raises:
            NotImplementedError: If exchange doesn't support OCO orders
            OrderError: If order creation fails
        """
        raise NotImplementedError(
            f"{self.name} connector has not implemented OCO orders. "
            "Override create_oco_order() in the exchange-specific connector."
        )

    def _check_credentials(self) -> bool:
        """Check if API credentials are configured"""
        return bool(self.api_key and self.api_secret)

    async def _rate_limit_wait(self):
        """Wait if necessary to respect rate limits"""
        min_interval = 60.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    async def _audit_api_call(
        self,
        endpoint: str,
        method: str = "GET",
        status: str = "success",
        duration_ms: Optional[float] = None,
        request_data: Optional[Dict] = None,
        response_summary: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Log API call to audit log if enabled"""
        if self._audit:
            await self._audit.log_api_call(
                exchange=self.name,
                endpoint=endpoint,
                method=method,
                status=status,
                duration_ms=duration_ms,
                request_data=request_data,
                response_summary=response_summary,
                error_message=error_message,
            )

    async def _audit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: str = "created",
        error_message: Optional[str] = None,
    ):
        """Log order event to audit log if enabled"""
        if self._audit:
            await self._audit.log_order(
                exchange=self.name,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                order_id=order_id,
                status=status,
                error_message=error_message,
            )

    async def _audit_auth(self, status: str, error_message: Optional[str] = None):
        """Log authentication event to audit log if enabled"""
        if self._audit:
            await self._audit.log_authentication(
                exchange=self.name,
                status=status,
                error_message=error_message,
            )

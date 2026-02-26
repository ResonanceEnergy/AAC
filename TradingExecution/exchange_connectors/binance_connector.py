#!/usr/bin/env python3
"""
Binance Exchange Connector
==========================
Implementation of the exchange connector for Binance.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from shared.utils import with_circuit_breaker, CircuitOpenError

from .base_connector import (
    BaseExchangeConnector,
    Ticker,
    OrderBook,
    Balance,
    ExchangeOrder,
    ExchangeError,
    ConnectionError,
    AuthenticationError,
    InsufficientFundsError,
    OrderError,
)

# Try to import ccxt
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class BinanceConnector(BaseExchangeConnector):
    """
    Binance exchange connector using ccxt library.
    
    Supports both mainnet and testnet.
    """

    @property
    def name(self) -> str:
        return "binance"

    def __init__(
        self,
        api_key: str = '',
        api_secret: str = '',
        testnet: bool = True,
        rate_limit: int = 1200,
    ):
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=rate_limit,
        )
        
        # Load from config if not provided
        if not api_key:
            config = get_config()
            self.api_key = config.binance.api_key
            self.api_secret = config.binance.api_secret
            self.testnet = config.binance.testnet

    async def connect(self) -> bool:
        """Connect to Binance"""
        import time
        start_time = time.time()
        
        if not CCXT_AVAILABLE:
            self.logger.error("ccxt library not installed. Run: pip install ccxt")
            await self._audit_auth("failure", "ccxt library not installed")
            return False
        
        try:
            options = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            }
            
            if self.testnet:
                # Proper testnet configuration for Binance
                # sandboxMode doesn't work - need to set URLs directly
                options['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    },
                    'www': 'https://testnet.binance.vision',
                }
                # Also set sandbox mode for compatibility
                options['options']['sandboxMode'] = True
                options['options']['adjustForTimeDifference'] = True
                self.logger.info("Connecting to Binance TESTNET (testnet.binance.vision)")
            else:
                self.logger.info("Connecting to Binance MAINNET")
            
            self._client = ccxt_async.binance(options)
            
            # For testnet, manually set sandbox mode after creation
            if self.testnet:
                self._client.set_sandbox_mode(True)
            
            # Test connection
            await self._client.load_markets()
            
            self._connected = True
            self.logger.info(f"Connected to Binance ({'testnet' if self.testnet else 'mainnet'})")
            
            # Audit successful connection
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("load_markets", "GET", "success", duration_ms)
            await self._audit_auth("success")
            
            return True
            
        except ccxt.AuthenticationError as e:
            self.logger.error(f"Binance authentication failed: {e}")
            await self._audit_auth("failure", str(e))
            raise AuthenticationError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "GET", "failure", duration_ms, error_message=str(e))
            raise ConnectionError(str(e))

    async def disconnect(self) -> None:
        """Disconnect from Binance"""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
            self.logger.info("Disconnected from Binance")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker for symbol"""
        await self._rate_limit_wait()
        
        return await self._get_ticker_with_breaker(symbol)

    @with_circuit_breaker("binance_ticker", failure_threshold=5, timeout=30.0)
    async def _get_ticker_with_breaker(self, symbol: str) -> Ticker:
        """Get ticker with circuit breaker protection"""
        try:
            ticker = await self._client.fetch_ticker(symbol)
            return Ticker(
                symbol=symbol,
                bid=ticker.get('bid', 0) or 0,
                ask=ticker.get('ask', 0) or 0,
                last=ticker.get('last', 0) or 0,
                volume_24h=ticker.get('quoteVolume', 0) or 0,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.now(),
            )
        except CircuitOpenError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book for symbol"""
        await self._rate_limit_wait()
        
        return await self._get_orderbook_with_breaker(symbol, limit)

    @with_circuit_breaker("binance_orderbook", failure_threshold=5, timeout=30.0)
    async def _get_orderbook_with_breaker(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book with circuit breaker protection"""
        try:
            book = await self._client.fetch_order_book(symbol, limit)
            return OrderBook(
                symbol=symbol,
                bids=[(b[0], b[1]) for b in book.get('bids', [])],
                asks=[(a[0], a[1]) for a in book.get('asks', [])],
                timestamp=datetime.fromtimestamp(book['timestamp'] / 1000) if book.get('timestamp') else datetime.now(),
            )
        except CircuitOpenError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances"""
        if not self._check_credentials():
            self.logger.warning("No API credentials - returning empty balances")
            return {}
        
        await self._rate_limit_wait()
        
        try:
            balance = await self._client.fetch_balance()
            result = {}
            
            for asset, data in balance.get('total', {}).items():
                if data > 0:
                    result[asset] = Balance(
                        asset=asset,
                        free=balance.get('free', {}).get(asset, 0),
                        locked=balance.get('used', {}).get(asset, 0),
                    )
            
            return result
        except ccxt.AuthenticationError as e:
            raise AuthenticationError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to fetch balances: {e}")
            raise ExchangeError(str(e))

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create a new order on Binance"""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required for trading")
        
        await self._rate_limit_wait()
        
        return await self._create_order_with_breaker(
            symbol, side, order_type, quantity, price, client_order_id
        )

    @with_circuit_breaker("binance_orders", failure_threshold=3, timeout=60.0)
    async def _create_order_with_breaker(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create order with circuit breaker protection"""
        import time
        start_time = time.time()
        
        try:
            params = {}
            if client_order_id:
                params['newClientOrderId'] = client_order_id
            
            order = await self._client.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=quantity,
                price=price,
                params=params,
            )
            
            result = self._parse_order(order)
            
            # Audit successful order
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "success", duration_ms)
            await self._audit_order(symbol, side, order_type, quantity, price, result.order_id, "created")
            
            return result
            
        except CircuitOpenError:
            raise
        except ccxt.InsufficientFunds as e:
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise InsufficientFundsError(str(e))
        except ccxt.InvalidOrder as e:
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise OrderError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "failure", duration_ms, error_message=str(e))
            raise ExchangeError(str(e))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required")
        
        await self._rate_limit_wait()
        
        try:
            await self._client.cancel_order(order_id, symbol)
            return True
        except ccxt.OrderNotFound:
            self.logger.warning(f"Order {order_id} not found")
            return False
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get order details"""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required")
        
        await self._rate_limit_wait()
        
        try:
            order = await self._client.fetch_order(order_id, symbol)
            return self._parse_order(order)
        except ccxt.OrderNotFound as e:
            raise OrderError(f"Order {order_id} not found")
        except Exception as e:
            self.logger.error(f"Failed to fetch order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders"""
        if not self._check_credentials():
            return []
        
        await self._rate_limit_wait()
        
        try:
            orders = await self._client.fetch_open_orders(symbol)
            return [self._parse_order(o) for o in orders]
        except Exception as e:
            self.logger.error(f"Failed to fetch open orders: {e}")
            raise ExchangeError(str(e))

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Get actual trading fees from exchange"""
        if not self._check_credentials():
            # Return defaults if no credentials
            return {'maker': 0.001, 'taker': 0.001}
        
        await self._rate_limit_wait()
        
        try:
            # Fetch actual trading fees from Binance
            fees = await self._client.fetch_trading_fee(symbol)
            return {
                'maker': fees.get('maker', 0.001),
                'taker': fees.get('taker', 0.001),
            }
        except Exception as e:
            self.logger.warning(f"Failed to fetch trade fees, using defaults: {e}")
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
        Create a stop-loss order on Binance.
        
        Uses STOP_LOSS_LIMIT for limit orders or STOP_LOSS for market stops.
        """
        if not self._check_credentials():
            raise AuthenticationError("API credentials required for stop-loss orders")
        
        await self._rate_limit_wait()
        
        import time
        start_time = time.time()
        
        try:
            params = {
                'stopPrice': stop_price,
            }
            if client_order_id:
                params['newClientOrderId'] = client_order_id
            
            if limit_price:
                # Stop-limit order
                order_type = 'STOP_LOSS_LIMIT'
                params['timeInForce'] = 'GTC'
                
                order = await self._client.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=limit_price,
                    params=params,
                )
            else:
                # Stop-market order (uses stop_price as trigger)
                order_type = 'STOP_LOSS'
                
                order = await self._client.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    params=params,
                )
            
            result = self._parse_order(order)
            
            # Audit successful order
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_stop_loss", "POST", "success", duration_ms)
            await self._audit_order(
                symbol, side, order_type, quantity, limit_price, 
                result.order_id, "stop_loss_created"
            )
            
            self.logger.info(f"Stop-loss order created: {result.order_id} @ {stop_price}")
            return result
            
        except ccxt.InsufficientFunds as e:
            await self._audit_order(symbol, side, "stop_loss", quantity, limit_price, None, "failed", str(e))
            raise InsufficientFundsError(str(e))
        except ccxt.InvalidOrder as e:
            await self._audit_order(symbol, side, "stop_loss", quantity, limit_price, None, "failed", str(e))
            raise OrderError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to create stop-loss order: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_stop_loss", "POST", "failure", duration_ms, error_message=str(e))
            raise ExchangeError(str(e))

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
        Create an OCO (One-Cancels-Other) order on Binance.
        
        This creates both a stop-loss and take-profit order, where filling
        one automatically cancels the other.
        """
        if not self._check_credentials():
            raise AuthenticationError("API credentials required for OCO orders")
        
        await self._rate_limit_wait()
        
        import time
        start_time = time.time()
        
        try:
            params = {
                'stopPrice': stop_price,
                'stopLimitPrice': stop_limit_price,
                'stopLimitTimeInForce': 'GTC',
            }
            if client_order_id:
                params['listClientOrderId'] = client_order_id
            
            # Binance OCO: create_order with type='oco'
            # The 'price' is the take-profit limit price
            order_result = await self._client.create_order(
                symbol=symbol,
                type='oco',
                side=side,
                amount=quantity,
                price=take_profit_price,
                params=params,
            )
            
            # Parse OCO response - contains multiple orders
            orders = order_result.get('orders', [])
            
            result = {}
            for order in orders:
                parsed = self._parse_order(order)
                if order.get('type') == 'STOP_LOSS_LIMIT':
                    result['stop_loss'] = parsed
                elif order.get('type') == 'LIMIT_MAKER':
                    result['take_profit'] = parsed
            
            # Audit successful OCO
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_oco", "POST", "success", duration_ms)
            await self._audit_order(
                symbol, side, "oco", quantity, take_profit_price,
                order_result.get('orderListId', ''), "oco_created"
            )
            
            self.logger.info(
                f"OCO order created: stop_loss @ {stop_price}, take_profit @ {take_profit_price}"
            )
            return result
            
        except ccxt.InsufficientFunds as e:
            await self._audit_order(symbol, side, "oco", quantity, take_profit_price, None, "failed", str(e))
            raise InsufficientFundsError(str(e))
        except ccxt.InvalidOrder as e:
            await self._audit_order(symbol, side, "oco", quantity, take_profit_price, None, "failed", str(e))
            raise OrderError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to create OCO order: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_oco", "POST", "failure", duration_ms, error_message=str(e))

    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get trade history from exchange.
        
        Args:
            symbol: Optional trading pair filter
            since: Optional start time
            limit: Max number of trades to return
        
        Returns:
            List of trade dictionaries
        """
        if not self._check_credentials():
            return []
        
        await self._rate_limit_wait()
        
        try:
            since_ts = int(since.timestamp() * 1000) if since else None
            trades = await self._client.fetch_my_trades(
                symbol=symbol,
                since=since_ts,
                limit=limit,
            )
            
            return [
                {
                    'trade_id': t['id'],
                    'order_id': t.get('order'),
                    'symbol': t['symbol'],
                    'side': t['side'],
                    'price': t['price'],
                    'quantity': t['amount'],
                    'cost': t['cost'],
                    'fee': t.get('fee', {}).get('cost', 0),
                    'fee_currency': t.get('fee', {}).get('currency', ''),
                    'timestamp': datetime.fromtimestamp(t['timestamp'] / 1000),
                }
                for t in trades
            ]
        except Exception as e:
            self.logger.error(f"Failed to fetch trade history: {e}")
            raise ExchangeError(str(e))

    def _parse_order(self, order: Dict) -> ExchangeOrder:
        """Parse ccxt order into ExchangeOrder"""
        return ExchangeOrder(
            order_id=order['id'],
            client_order_id=order.get('clientOrderId'),
            symbol=order['symbol'],
            side=order['side'],
            order_type=order['type'],
            quantity=order['amount'],
            price=order.get('price'),
            status=order['status'],
            filled_quantity=order.get('filled', 0),
            average_price=order.get('average', 0) or 0,
            fee=order.get('fee', {}).get('cost', 0) if order.get('fee') else 0,
            fee_currency=order.get('fee', {}).get('currency', '') if order.get('fee') else '',
            created_at=datetime.fromtimestamp(order['timestamp'] / 1000) if order.get('timestamp') else datetime.now(),
            raw=order,
        )


# Test the connector
if __name__ == '__main__':
    async def test():
        connector = BinanceConnector(testnet=True)
        
        try:
            # Connect (will work without credentials for public data)
            connected = await connector.connect()
            print(f"Connected: {connected}")
            
            if connected:
                # Get ticker
                ticker = await connector.get_ticker('BTC/USDT')
                print(f"BTC/USDT Ticker: bid={ticker.bid}, ask={ticker.ask}, spread={ticker.spread_pct:.4f}%")
                
                # Get orderbook
                book = await connector.get_orderbook('BTC/USDT', 5)
                print(f"Orderbook: {len(book.bids)} bids, {len(book.asks)} asks, mid={book.mid_price}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await connector.disconnect()
    
    asyncio.run(test())

#!/usr/bin/env python3
"""
Kraken Exchange Connector
=========================
Implementation of the exchange connector for Kraken.
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


class KrakenConnector(BaseExchangeConnector):
    """
    Kraken exchange connector using ccxt library.
    
    WARNING: Kraken does not offer a public testnet. All API calls with
    valid credentials will execute on LIVE/PRODUCTION systems.
    Use paper_trading mode in execution engine for testing.
    """

    @property
    def name(self) -> str:
        return "kraken"

    def __init__(
        self,
        api_key: str = '',
        api_secret: str = '',
        testnet: bool = False,  # Kraken doesn't have a public testnet
        rate_limit: int = 300,
        paper_trading: bool = True,  # Safety flag - default to paper trading
    ):
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=rate_limit,
        )
        
        self.paper_trading = paper_trading
        
        # Load from config if not provided
        if not api_key:
            config = get_config()
            self.api_key = config.kraken.api_key
            self.api_secret = config.kraken.api_secret
        
        # Safety warning
        if not paper_trading and api_key:
            self.logger.warning(
                "[WARN]️  KRAKEN LIVE TRADING ENABLED - No testnet available! "
                "All orders will execute on PRODUCTION. Set paper_trading=True for testing."
            )

    async def connect(self) -> bool:
        """Connect to Kraken"""
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
            }
            
            if self.paper_trading:
                self.logger.info("Connecting to Kraken (PAPER TRADING MODE)")
            else:
                self.logger.warning("[WARN]️  Connecting to Kraken LIVE PRODUCTION")
            
            self._client = ccxt_async.kraken(options)
            
            # Test connection
            await self._client.load_markets()
            
            self._connected = True
            mode = "paper" if self.paper_trading else "LIVE"
            self.logger.info(f"Connected to Kraken ({mode})")
            
            # Audit successful connection
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("load_markets", "GET", "success", duration_ms)
            await self._audit_auth("success")
            
            return True
            
        except ccxt.AuthenticationError as e:
            self.logger.error(f"Kraken authentication failed: {e}")
            await self._audit_auth("failure", str(e))
            raise AuthenticationError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to connect to Kraken: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "GET", "failure", duration_ms, error_message=str(e))
            raise ConnectionError(str(e))

    async def disconnect(self) -> None:
        """Disconnect from Kraken"""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
            self.logger.info("Disconnected from Kraken")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker for symbol"""
        await self._rate_limit_wait()
        return await self._get_ticker_with_breaker(symbol)

    @with_circuit_breaker("kraken_ticker", failure_threshold=5, timeout=30.0)
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

    @with_circuit_breaker("kraken_orderbook", failure_threshold=5, timeout=30.0)
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
        """Create a new order on Kraken"""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required for trading")
        
        await self._rate_limit_wait()
        
        try:
            params = {}
            if client_order_id:
                params['userref'] = client_order_id
            
            order = await self._client.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=quantity,
                price=price,
                params=params,
            )
            
            return self._parse_order(order)
            
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(str(e))
        except ccxt.InvalidOrder as e:
            raise OrderError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
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
            return {'maker': 0.0016, 'taker': 0.0026}
        
        await self._rate_limit_wait()
        
        try:
            fees = await self._client.fetch_trading_fee(symbol)
            return {
                'maker': fees.get('maker', 0.0016),
                'taker': fees.get('taker', 0.0026),
            }
        except Exception as e:
            self.logger.warning(f"Failed to fetch trade fees, using defaults: {e}")
            return {'maker': 0.0016, 'taker': 0.0026}

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
        Create a stop-loss order on Kraken.
        
        Kraken supports stop-loss and stop-loss-limit order types.
        """
        if self.paper_trading:
            self.logger.info(f"[PAPER] Would create stop-loss: {side} {quantity} {symbol} @ {stop_price}")
            return ExchangeOrder(
                order_id=f"PAPER_SL_{datetime.now().timestamp()}",
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type="stop_loss",
                quantity=quantity,
                price=limit_price,
                status="open",
            )
        
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
                params['clientOrderId'] = client_order_id
            
            if limit_price:
                # Stop-loss-limit order
                order_type = 'stop-loss-limit'
                
                order = await self._client.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=limit_price,
                    params=params,
                )
            else:
                # Stop-loss order (market when triggered)
                order_type = 'stop-loss'
                
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

    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get trade history from exchange"""
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
        connector = KrakenConnector()
        
        try:
            connected = await connector.connect()
            print(f"Connected: {connected}")
            
            if connected:
                ticker = await connector.get_ticker('BTC/USD')
                print(f"BTC/USD Ticker: bid={ticker.bid}, ask={ticker.ask}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await connector.disconnect()
    
    asyncio.run(test())

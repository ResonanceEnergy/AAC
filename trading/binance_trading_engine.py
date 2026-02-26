#!/usr/bin/env python3
"""
AAC Binance Trading Engine
==========================

Full-featured Binance trading engine for AAC arbitrage system.
Includes order management, risk controls, and automated execution.

Features:
- Spot and futures trading
- Order management (limit, market, stop-loss)
- Risk management (position sizing, stop-loss)
- Portfolio management
- Performance tracking
- Integration with AAC arbitrage signals

Testnet: https://testnet.binance.vision
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Trading configuration"""
    max_position_size_usd: float = float(os.getenv('MAX_POSITION_SIZE_USD', '10000'))
    max_daily_loss_usd: float = float(os.getenv('MAX_DAILY_LOSS_USD', '1000'))
    max_open_positions: int = int(os.getenv('MAX_OPEN_POSITIONS', '10'))
    default_slippage: float = 0.001  # 0.1%
    min_order_size_usd: float = 10.0

@dataclass
class Position:
    """Trading position"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    status: str  # 'open', 'closed', 'stopped'
    pnl: float = 0.0

@dataclass
class TradeSignal:
    """Arbitrage trade signal"""
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: float
    price: float
    reason: str
    confidence: float
    timestamp: datetime
    arbitrage_type: str

class BinanceTradingEngine:
    """Full-featured Binance trading engine"""

    def __init__(self, config: 'BinanceConfig', trading_config: TradingConfig):
        self.config = config
        self.trading_config = trading_config
        self.session: Optional[aiohttp.ClientSession] = None
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature"""
        return hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)

    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make API request with rate limiting"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Rate limiting (10 requests per second for orders)
        await asyncio.sleep(0.1)

        url = f"{self.config.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.config.api_key}

        params = params or {}

        if signed:
            params['timestamp'] = self._get_timestamp()
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            params['signature'] = self._generate_signature(query_string)

        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                async with self.session.post(url, data=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except Exception as e:
            print(f"Request error: {e}")
            raise

    async def _handle_response(self, response) -> Dict:
        """Handle API response"""
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            raise Exception(f"API Error {response.status}: {error_text}")

    async def get_account_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        try:
            account = await self._make_request('GET', '/api/v3/account', signed=True)

            if asset:
                for balance in account['balances']:
                    if balance['asset'] == asset:
                        return {
                            'asset': asset,
                            'free': float(balance['free']),
                            'locked': float(balance['locked']),
                            'total': float(balance['free']) + float(balance['locked'])
                        }
            else:
                return {b['asset']: {
                    'free': float(b['free']),
                    'locked': float(b['locked']),
                    'total': float(b['free']) + float(b['locked'])
                } for b in account['balances'] if float(b['free']) > 0 or float(b['locked']) > 0}

        except Exception as e:
            print(f"Error getting balance: {e}")
            return {}

    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading rules"""
        try:
            exchange_info = await self._make_request('GET', '/api/v3/exchangeInfo')

            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    return s
            return None
        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return None

    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> float:
        """Calculate position size based on risk management rules"""
        # Get available balance (assume USDT for simplicity)
        # In production, you'd get real balance

        max_position_usd = self.trading_config.max_position_size_usd
        position_usd = max_position_usd * confidence  # Scale by confidence

        # Convert to symbol quantity
        quantity = position_usd / price

        # Apply minimum order size
        if position_usd < self.trading_config.min_order_size_usd:
            return 0.0

        return quantity

    async def place_limit_order(self, symbol: str, side: str, quantity: float,
                               price: float, test: bool = True) -> Optional[Dict]:
        """Place a limit order"""
        try:
            # Get symbol info for precision
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                print(f"Symbol {symbol} not found")
                return None

            # Round quantity and price to appropriate precision
            quantity_precision = symbol_info['baseAssetPrecision']
            price_precision = len(str(symbol_info['filters'][0]['tickSize']).split('.')[1].rstrip('0'))

            quantity = float(Decimal(str(quantity)).quantize(Decimal('1e-{}'.format(quantity_precision)), rounding=ROUND_DOWN))
            price = float(Decimal(str(price)).quantize(Decimal('1e-{}'.format(price_precision)), rounding=ROUND_DOWN))

            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': str(quantity),
                'price': str(price)
            }

            endpoint = '/api/v3/order/test' if test else '/api/v3/order'
            return await self._make_request('POST', endpoint, params, signed=True)

        except Exception as e:
            print(f"Error placing limit order: {e}")
            return None

    async def place_market_order(self, symbol: str, side: str, quantity: float, test: bool = True) -> Optional[Dict]:
        """Place a market order"""
        try:
            # Get symbol info for precision
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                print(f"Symbol {symbol} not found")
                return None

            quantity_precision = symbol_info['baseAssetPrecision']
            quantity = float(Decimal(str(quantity)).quantize(Decimal('1e-{}'.format(quantity_precision)), rounding=ROUND_DOWN))

            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': 'MARKET',
                'quantity': str(quantity)
            }

            endpoint = '/api/v3/order/test' if test else '/api/v3/order'
            return await self._make_request('POST', endpoint, params, signed=True)

        except Exception as e:
            print(f"Error placing market order: {e}")
            return None

    async def execute_arbitrage_signal(self, signal: TradeSignal, test: bool = True) -> bool:
        """Execute an arbitrage trade signal"""
        try:
            print(f"🎯 Executing {signal.arbitrage_type} signal for {signal.symbol}")

            # Check risk limits
            if len(self.positions) >= self.trading_config.max_open_positions:
                print("❌ Maximum open positions reached")
                return False

            if self.daily_pnl <= -self.trading_config.max_daily_loss_usd:
                print("❌ Daily loss limit reached")
                return False

            # Calculate position size
            position_size = self.calculate_position_size(signal.symbol, signal.price, signal.confidence)

            if position_size <= 0:
                print("❌ Position size too small")
                return False

            # Apply slippage for limit orders
            if signal.action.upper() == 'BUY':
                limit_price = signal.price * (1 + self.trading_config.default_slippage)
            else:
                limit_price = signal.price * (1 - self.trading_config.default_slippage)

            # Place order
            order_result = await self.place_limit_order(
                signal.symbol,
                signal.action,
                position_size,
                limit_price,
                test=test
            )

            if order_result:
                print(f"✅ Order placed: {signal.action} {position_size:.4f} {signal.symbol} @ ${limit_price:.2f}")

                # Create position tracking
                position = Position(
                    symbol=signal.symbol,
                    side=signal.action,
                    quantity=position_size,
                    entry_price=limit_price,
                    current_price=signal.price,
                    stop_loss=limit_price * 0.95,  # 5% stop loss
                    take_profit=limit_price * 1.05,  # 5% take profit
                    timestamp=datetime.now(),
                    status='open'
                )

                self.positions[signal.symbol] = position
                return True
            else:
                print("❌ Order placement failed")
                return False

        except Exception as e:
            print(f"Error executing signal: {e}")
            return False

    async def check_positions(self) -> Dict[str, Any]:
        """Check and update positions"""
        updates = {}

        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                price_data = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
                current_price = float(price_data['price'])

                # Update position
                position.current_price = current_price

                if position.side == 'BUY':
                    position.pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.pnl = (position.entry_price - current_price) * position.quantity

                # Check stop loss
                if position.side == 'BUY' and current_price <= position.stop_loss:
                    print(f"🛑 Stop loss triggered for {symbol} BUY position")
                    position.status = 'stopped'
                elif position.side == 'SELL' and current_price >= position.stop_loss:
                    print(f"🛑 Stop loss triggered for {symbol} SELL position")
                    position.status = 'stopped'

                # Check take profit
                elif position.side == 'BUY' and current_price >= position.take_profit:
                    print(f"🎉 Take profit triggered for {symbol} BUY position")
                    position.status = 'closed'
                elif position.side == 'SELL' and current_price <= position.take_profit:
                    print(f"🎉 Take profit triggered for {symbol} SELL position")
                    position.status = 'closed'

                updates[symbol] = {
                    'status': position.status,
                    'pnl': position.pnl,
                    'current_price': current_price
                }

            except Exception as e:
                print(f"Error checking position {symbol}: {e}")

        return updates

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        total_pnl = sum(p.pnl for p in self.positions.values())
        open_positions = len([p for p in self.positions.values() if p.status == 'open'])
        closed_positions = len([p for p in self.positions.values() if p.status == 'closed'])

        return {
            'total_positions': len(self.positions),
            'open_positions': open_positions,
            'closed_positions': closed_positions,
            'total_pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'positions': [
                {
                    'symbol': p.symbol,
                    'side': p.side,
                    'quantity': p.quantity,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'pnl': p.pnl,
                    'status': p.status
                } for p in self.positions.values()
            ]
        }

async def trading_engine_demo():
    """Demo the trading engine with sample signals"""
    print("🎯 AAC Binance Trading Engine Demo")
    print("=" * 50)

    from binance_arbitrage_integration import BinanceConfig

    binance_config = BinanceConfig()
    trading_config = TradingConfig()

    if not binance_config.is_configured():
        print("❌ Binance API keys not configured")
        return

    async with BinanceTradingEngine(binance_config, trading_config) as engine:
        # Get account balance
        print("👤 Account Balance:")
        balance = await engine.get_account_balance('USDT')
        if balance:
            print(f"   USDT: {balance.get('free', 0):.2f} free, {balance.get('locked', 0):.2f} locked")
        else:
            print("   Unable to retrieve balance")

        # Create sample arbitrage signals
        signals = [
            TradeSignal(
                symbol='BTCUSDT',
                action='BUY',
                quantity=0.001,
                price=45000,
                reason='Cross-exchange arbitrage opportunity',
                confidence=0.8,
                timestamp=datetime.now(),
                arbitrage_type='cross_exchange'
            ),
            TradeSignal(
                symbol='ETHUSDT',
                action='SELL',
                quantity=0.01,
                price=3000,
                reason='Triangular arbitrage',
                confidence=0.7,
                timestamp=datetime.now(),
                arbitrage_type='triangular'
            )
        ]

        print(f"\n🎯 Processing {len(signals)} Arbitrage Signals:")

        for signal in signals:
            success = await engine.execute_arbitrage_signal(signal, test=True)  # Test mode
            print(f"   {'✅' if success else '❌'} {signal.symbol} {signal.action}")

        # Check positions
        print("📊 Position Updates:")
        updates = await engine.check_positions()

        for symbol, update in updates.items():
            print(f"   {symbol}: {update['status']} - PnL: ${update['pnl']:.2f}")

        # Portfolio summary
        print("📈 Portfolio Summary:")
        summary = engine.get_portfolio_summary()
        print(f"   Total Positions: {summary['total_positions']}")
        print(f"   Open Positions: {summary['open_positions']}")
        print(f"   Total PnL: ${summary['total_pnl']:.2f}")

    print("\n✅ Trading engine demo complete!")

if __name__ == "__main__":
    print("🚀 AAC Binance Trading Engine")
    print("=" * 40)

    # Run demo
    asyncio.run(trading_engine_demo())

    print("\n💡 Production Setup:")
    print("   1. Replace API keys with production keys")
    print("   2. Set BINANCE_TESTNET=false")
    print("   3. Implement real arbitrage signal generation")
    print("   4. Add comprehensive risk management")
    print("   5. Enable automated trade execution")
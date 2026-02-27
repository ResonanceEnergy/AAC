"""Async Coinbase Exchange REST API client for authenticated trading operations."""

import os
import time
import hmac
import hashlib
import base64
import aiohttp
from typing import Dict, Any

class CoinbaseAPI:
    """Async Coinbase Exchange API Client (REST)"""
    def __init__(self, api_key=None, api_secret=None, passphrase=None, sandbox=False):
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        self.passphrase = passphrase or os.getenv('COINBASE_PASSPHRASE')
        self.base_url = (
            'https://api-public.sandbox.exchange.coinbase.com'
            if sandbox else 'https://api.exchange.coinbase.com'
        )
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _sign(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        timestamp = str(time.time())
        message = timestamp + method.upper() + request_path + body
        hmac_key = base64.b64decode(self.api_secret)
        signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }

    async def request(self, method: str, path: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None) -> Any:
        url = self.base_url + path
        body = '' if not data else json.dumps(data)
        headers = self._sign(method, path, body)
        async with self.session.request(method, url, headers=headers, params=params, data=body) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_accounts(self):
        return await self.request('GET', '/accounts')

    async def get_fills(self, product_id: str):
        return await self.request('GET', f'/fills', params={'product_id': product_id})

    async def place_order(self, product_id: str, side: str, size: str, order_type: str = 'market', price: str = None):
        data = {
            'product_id': product_id,
            'side': side,
            'size': size,
            'type': order_type
        }
        if price:
            data['price'] = price
        return await self.request('POST', '/orders', data=data)

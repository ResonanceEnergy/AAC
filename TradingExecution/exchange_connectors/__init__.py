"""
Exchange Connectors Package
===========================
Unified interface for multiple cryptocurrency and traditional exchanges.
"""

from .base_connector import BaseExchangeConnector, ExchangeError
try:
    from .binance_connector import BinanceConnector
except (ImportError, ModuleNotFoundError):
    BinanceConnector = None
from .coinbase_connector import CoinbaseConnector
from .kraken_connector import KrakenConnector
try:
    from .ibkr_connector import IBKRConnector
except (ImportError, RuntimeError):
    IBKRConnector = None
from .ndax_connector import NDAXConnector
from .moomoo_connector import MoomooConnector
from .noxi_rise_connector import NoxiRiseConnector
from .metalx_connector import MetalXConnector

__all__ = [
    'BaseExchangeConnector',
    'ExchangeError',
    'BinanceConnector',
    'CoinbaseConnector',
    'KrakenConnector',
    'IBKRConnector',
    'MoomooConnector',
    'NoxiRiseConnector',
    'MetalXConnector',
]

"""
Exchange Connectors Package
===========================
Unified interface for multiple cryptocurrency and traditional exchanges.
"""

from .base_connector import BaseExchangeConnector, ExchangeError
from .coinbase_connector import CoinbaseConnector
from .kraken_connector import KrakenConnector

try:
    from .ibkr_connector import IBKRConnector
except (ImportError, RuntimeError):
    IBKRConnector = None
from .metalx_connector import MetalXConnector
from .moomoo_connector import MoomooConnector
from .ndax_connector import NDAXConnector
from .noxi_rise_connector import NoxiRiseConnector

try:
    from .snaptrade_connector import SnapTradeConnector
except (ImportError, ModuleNotFoundError):
    SnapTradeConnector = None

__all__ = [
    'BaseExchangeConnector',
    'ExchangeError',
    'CoinbaseConnector',
    'KrakenConnector',
    'IBKRConnector',
    'MoomooConnector',
    'NoxiRiseConnector',
    'MetalXConnector',
    'SnapTradeConnector',
]

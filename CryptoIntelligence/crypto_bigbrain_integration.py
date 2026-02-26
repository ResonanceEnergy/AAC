#!/usr/bin/env python3
"""
Crypto-BigBrain Intelligence Integration
========================================
Bridge module connecting CryptoIntelligence analysis with BigBrain research agents.
Enables cross-system intelligence sharing and coordinated opportunity detection.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path


@dataclass
class IntelligenceSignal:
    """Standardized signal format for cross-system communication"""
    signal_id: str
    source: str  # 'crypto_intel' or 'bigbrain'
    signal_type: str  # 'arbitrage', 'whale', 'narrative', 'technical', etc.
    symbol: Optional[str]
    confidence: float  # 0.0 to 1.0
    urgency: str  # 'low', 'medium', 'high', 'critical'
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if signal is still valid (not expired)"""
        if self.expires_at is None:
            return True
        return datetime.now() < self.expires_at
    
    def to_dict(self) -> Dict:
        return {
            'signal_id': self.signal_id,
            'source': self.source,
            'signal_type': self.signal_type,
            'symbol': self.symbol,
            'confidence': self.confidence,
            'urgency': self.urgency,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity"""
    opportunity_id: str
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    estimated_profit_usd: float
    confidence: float
    detected_at: datetime = field(default_factory=datetime.now)
    valid_until: datetime = field(default_factory=lambda: datetime.now() + timedelta(seconds=30))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_profitable(self) -> bool:
        return self.spread_pct > 0.1  # > 0.1% spread
    
    @property
    def is_valid(self) -> bool:
        return datetime.now() < self.valid_until


class CryptoBigBrainIntegration:
    """
    Integration layer between CryptoIntelligence and BigBrain systems.
    
    Responsibilities:
    - Receive signals from both systems
    - Correlate and enrich data
    - Detect cross-system opportunities
    - Route actionable intelligence to TradingExecution
    """

    def __init__(self):
        self.config = get_config()
        self.logger = self._setup_logging()
        
        # Signal buffers
        self.crypto_signals: List[IntelligenceSignal] = []
        self.bigbrain_signals: List[IntelligenceSignal] = []
        self.opportunities: List[ArbitrageOpportunity] = []
        
        # State
        self.is_running = False
        self._signal_counter = 0
        
        self.logger.info("CryptoBigBrainIntegration initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('CryptoBigBrainIntegration')
        logger.setLevel(logging.INFO)

        log_dir = get_project_path('CryptoIntelligence', 'logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_dir / 'integration.log')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        self._signal_counter += 1
        return f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._signal_counter:04d}"

    def _generate_opportunity_id(self) -> str:
        """Generate unique opportunity ID"""
        return f"ARB_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    # ==========================================
    # Signal Ingestion
    # ==========================================

    def receive_crypto_signal(
        self,
        signal_type: str,
        symbol: Optional[str],
        confidence: float,
        urgency: str = 'medium',
        data: Optional[Dict] = None,
        ttl_seconds: int = 300,
    ) -> IntelligenceSignal:
        """
        Receive a signal from CryptoIntelligence system.
        
        Args:
            signal_type: Type of signal (arbitrage, whale, liquidation, etc.)
            symbol: Trading symbol if applicable
            confidence: Confidence level 0.0-1.0
            urgency: low/medium/high/critical
            data: Additional signal data
            ttl_seconds: Time-to-live in seconds
        """
        signal = IntelligenceSignal(
            signal_id=self._generate_signal_id(),
            source='crypto_intel',
            signal_type=signal_type,
            symbol=symbol,
            confidence=confidence,
            urgency=urgency,
            data=data or {},
            expires_at=datetime.now() + timedelta(seconds=ttl_seconds),
        )
        
        self.crypto_signals.append(signal)
        self._cleanup_expired_signals()
        
        self.logger.info(f"Received crypto signal: {signal.signal_type} for {signal.symbol} (conf={signal.confidence:.2f})")
        
        # Check for correlations
        self._check_signal_correlations(signal)
        
        return signal

    def receive_bigbrain_signal(
        self,
        signal_type: str,
        confidence: float,
        urgency: str = 'medium',
        data: Optional[Dict] = None,
        ttl_seconds: int = 600,
    ) -> IntelligenceSignal:
        """
        Receive a signal from BigBrain Intelligence system.
        
        Args:
            signal_type: Type of signal (narrative, sentiment, prediction, etc.)
            confidence: Confidence level 0.0-1.0
            urgency: low/medium/high/critical
            data: Additional signal data
            ttl_seconds: Time-to-live in seconds
        """
        signal = IntelligenceSignal(
            signal_id=self._generate_signal_id(),
            source='bigbrain',
            signal_type=signal_type,
            symbol=data.get('symbol') if data else None,
            confidence=confidence,
            urgency=urgency,
            data=data or {},
            expires_at=datetime.now() + timedelta(seconds=ttl_seconds),
        )
        
        self.bigbrain_signals.append(signal)
        self._cleanup_expired_signals()
        
        self.logger.info(f"Received BigBrain signal: {signal.signal_type} (conf={signal.confidence:.2f})")
        
        # Check for correlations
        self._check_signal_correlations(signal)
        
        return signal

    # ==========================================
    # Signal Processing
    # ==========================================

    def _cleanup_expired_signals(self):
        """Remove expired signals from buffers"""
        self.crypto_signals = [s for s in self.crypto_signals if s.is_valid()]
        self.bigbrain_signals = [s for s in self.bigbrain_signals if s.is_valid()]
        self.opportunities = [o for o in self.opportunities if o.is_valid]

    def _check_signal_correlations(self, new_signal: IntelligenceSignal):
        """
        Check if new signal correlates with existing signals
        to generate higher-confidence opportunities.
        """
        if new_signal.source == 'crypto_intel':
            # Check against BigBrain signals
            for bb_signal in self.bigbrain_signals:
                correlation = self._calculate_correlation(new_signal, bb_signal)
                if correlation > 0.7:
                    self.logger.info(f"High correlation detected: {new_signal.signal_id} <-> {bb_signal.signal_id}")
                    self._generate_correlated_opportunity(new_signal, bb_signal, correlation)
        else:
            # Check against Crypto signals
            for crypto_signal in self.crypto_signals:
                correlation = self._calculate_correlation(crypto_signal, new_signal)
                if correlation > 0.7:
                    self.logger.info(f"High correlation detected: {crypto_signal.signal_id} <-> {new_signal.signal_id}")
                    self._generate_correlated_opportunity(crypto_signal, new_signal, correlation)

    def _calculate_correlation(
        self,
        crypto_signal: IntelligenceSignal,
        bb_signal: IntelligenceSignal
    ) -> float:
        """
        Calculate correlation score between crypto and BigBrain signals.
        
        Returns value between 0.0 (no correlation) and 1.0 (perfect correlation).
        """
        score = 0.0
        
        # Symbol match
        if crypto_signal.symbol and bb_signal.symbol:
            if crypto_signal.symbol == bb_signal.symbol:
                score += 0.3
        
        # Time proximity (signals within 5 minutes)
        time_diff = abs((crypto_signal.timestamp - bb_signal.timestamp).total_seconds())
        if time_diff < 300:
            score += 0.2 * (1 - time_diff / 300)
        
        # Urgency alignment
        urgency_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        urgency_diff = abs(urgency_map.get(crypto_signal.urgency, 2) - urgency_map.get(bb_signal.urgency, 2))
        if urgency_diff <= 1:
            score += 0.2
        
        # Combined confidence
        avg_confidence = (crypto_signal.confidence + bb_signal.confidence) / 2
        score += 0.3 * avg_confidence
        
        return min(score, 1.0)

    def _generate_correlated_opportunity(
        self,
        crypto_signal: IntelligenceSignal,
        bb_signal: IntelligenceSignal,
        correlation: float
    ):
        """Generate an opportunity from correlated signals"""
        # Boost confidence based on correlation
        boosted_confidence = min(
            (crypto_signal.confidence + bb_signal.confidence) / 2 * (1 + correlation * 0.2),
            0.99
        )
        
        self.logger.info(
            f"Correlated opportunity: {crypto_signal.signal_type}+{bb_signal.signal_type} "
            f"conf={boosted_confidence:.2f}"
        )
        
        # Would trigger trading execution here
        # For now, log to opportunities list

    # ==========================================
    # Arbitrage Detection
    # ==========================================

    def detect_cross_exchange_arbitrage(
        self,
        prices: Dict[str, Dict[str, float]]
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across exchanges.
        
        Args:
            prices: Dict of {symbol: {exchange: price}}
                   e.g., {'BTC/USDT': {'binance': 45000, 'coinbase': 45100}}
        
        Returns:
            List of detected arbitrage opportunities
        """
        opportunities = []
        
        for symbol, exchange_prices in prices.items():
            exchanges = list(exchange_prices.keys())
            
            for i, buy_exchange in enumerate(exchanges):
                for sell_exchange in exchanges[i+1:]:
                    buy_price = exchange_prices[buy_exchange]
                    sell_price = exchange_prices[sell_exchange]
                    
                    # Check both directions
                    for bp, sp, bx, sx in [
                        (buy_price, sell_price, buy_exchange, sell_exchange),
                        (sell_price, buy_price, sell_exchange, buy_exchange)
                    ]:
                        if sp > bp:
                            spread_pct = ((sp - bp) / bp) * 100
                            
                            # Account for fees (estimated 0.2% round trip)
                            net_spread = spread_pct - 0.2
                            
                            if net_spread > 0.1:  # Minimum 0.1% profit after fees
                                opp = ArbitrageOpportunity(
                                    opportunity_id=self._generate_opportunity_id(),
                                    symbol=symbol,
                                    buy_exchange=bx,
                                    sell_exchange=sx,
                                    buy_price=bp,
                                    sell_price=sp,
                                    spread_pct=spread_pct,
                                    estimated_profit_usd=net_spread * 100,  # Per $10k
                                    confidence=min(0.5 + net_spread * 0.1, 0.95),
                                )
                                opportunities.append(opp)
                                self.opportunities.append(opp)
                                
                                self.logger.info(
                                    f"Arbitrage detected: {symbol} "
                                    f"BUY@{bx}({bp:.2f}) SELL@{sx}({sp:.2f}) "
                                    f"spread={spread_pct:.3f}%"
                                )
        
        return opportunities

    # ==========================================
    # Data Aggregation
    # ==========================================

    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Aggregate market sentiment from all sources for a symbol.
        
        Returns combined sentiment analysis.
        """
        sentiment = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'crypto_signals': [],
            'bigbrain_signals': [],
            'overall_sentiment': 'neutral',
            'confidence': 0.0,
        }
        
        # Collect relevant crypto signals
        for signal in self.crypto_signals:
            if signal.symbol == symbol and signal.is_valid():
                sentiment['crypto_signals'].append(signal.to_dict())
        
        # Collect relevant BigBrain signals
        for signal in self.bigbrain_signals:
            if signal.symbol == symbol and signal.is_valid():
                sentiment['bigbrain_signals'].append(signal.to_dict())
        
        # Calculate overall sentiment
        all_signals = sentiment['crypto_signals'] + sentiment['bigbrain_signals']
        if all_signals:
            avg_confidence = sum(s['confidence'] for s in all_signals) / len(all_signals)
            sentiment['confidence'] = avg_confidence
            
            # Simple sentiment determination (would be more sophisticated in production)
            bullish_count = sum(1 for s in all_signals if s.get('data', {}).get('direction') == 'bullish')
            bearish_count = sum(1 for s in all_signals if s.get('data', {}).get('direction') == 'bearish')
            
            if bullish_count > bearish_count:
                sentiment['overall_sentiment'] = 'bullish'
            elif bearish_count > bullish_count:
                sentiment['overall_sentiment'] = 'bearish'
        
        return sentiment

    def get_active_opportunities(self) -> List[Dict]:
        """Get all active arbitrage opportunities"""
        self._cleanup_expired_signals()
        return [
            {
                'opportunity_id': o.opportunity_id,
                'symbol': o.symbol,
                'buy_exchange': o.buy_exchange,
                'sell_exchange': o.sell_exchange,
                'spread_pct': o.spread_pct,
                'estimated_profit_usd': o.estimated_profit_usd,
                'confidence': o.confidence,
                'detected_at': o.detected_at.isoformat(),
                'valid_until': o.valid_until.isoformat(),
            }
            for o in self.opportunities
            if o.is_valid
        ]

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        self._cleanup_expired_signals()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'active_crypto_signals': len(self.crypto_signals),
            'active_bigbrain_signals': len(self.bigbrain_signals),
            'active_opportunities': len([o for o in self.opportunities if o.is_valid]),
            'total_signals_processed': self._signal_counter,
        }

    # ==========================================
    # Lifecycle
    # ==========================================

    async def start(self):
        """Start the integration service"""
        self.logger.info("Starting CryptoBigBrainIntegration...")
        self.is_running = True
        self.logger.info("CryptoBigBrainIntegration started")

    async def stop(self):
        """Stop the integration service"""
        self.logger.info("Stopping CryptoBigBrainIntegration...")
        self.is_running = False
        self.logger.info("CryptoBigBrainIntegration stopped")


# Singleton instance
_integration: Optional[CryptoBigBrainIntegration] = None


def get_integration() -> CryptoBigBrainIntegration:
    """Get the singleton integration instance"""
    global _integration
    if _integration is None:
        _integration = CryptoBigBrainIntegration()
    return _integration


# CLI test
if __name__ == '__main__':
    async def test():
        integration = get_integration()
        await integration.start()
        
        # Simulate receiving signals
        integration.receive_crypto_signal(
            signal_type='whale_movement',
            symbol='BTC/USDT',
            confidence=0.85,
            urgency='high',
            data={'amount': 500, 'direction': 'bullish'}
        )
        
        integration.receive_bigbrain_signal(
            signal_type='narrative_shift',
            confidence=0.78,
            urgency='high',
            data={'symbol': 'BTC/USDT', 'narrative': 'institutional_adoption', 'direction': 'bullish'}
        )
        
        # Test arbitrage detection
        prices = {
            'BTC/USDT': {'binance': 45000, 'coinbase': 45150, 'kraken': 44980},
            'ETH/USDT': {'binance': 2800, 'coinbase': 2805, 'kraken': 2798},
        }
        opportunities = integration.detect_cross_exchange_arbitrage(prices)
        
        print("\n=== Integration Status ===")
        status = integration.get_integration_status()
        print(json.dumps(status, indent=2))
        
        print("\n=== Active Opportunities ===")
        opps = integration.get_active_opportunities()
        for opp in opps:
            print(f"  {opp['symbol']}: {opp['buy_exchange']} -> {opp['sell_exchange']} ({opp['spread_pct']:.3f}%)")
        
        print("\n=== BTC/USDT Sentiment ===")
        sentiment = integration.get_market_sentiment('BTC/USDT')
        print(f"  Overall: {sentiment['overall_sentiment']} (conf={sentiment['confidence']:.2f})")
        
        await integration.stop()
    
    asyncio.run(test())

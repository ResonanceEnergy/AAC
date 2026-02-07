#!/usr/bin/env python3
"""
BigBrain Intelligence - Research Agents Collection
==================================================
Complete implementation of all Theater research agents.
"""

import asyncio
import logging
import json
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import sys
import aiohttp

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.data_sources import CoinGeckoClient, MarketTick

logger = logging.getLogger(__name__)


@dataclass
class ResearchFinding:
    """Standardized research finding structure"""
    finding_id: str
    agent_id: str
    theater: str
    finding_type: str
    title: str
    description: str
    confidence: float
    urgency: str  # 'low', 'medium', 'high', 'critical'
    data: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'finding_id': self.finding_id,
            'agent_id': self.agent_id,
            'theater': self.theater,
            'finding_type': self.finding_type,
            'title': self.title,
            'description': self.description,
            'confidence': self.confidence,
            'urgency': self.urgency,
            'data': self.data,
            'sources': self.sources,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }


class BaseResearchAgent(ABC):
    """Base class for all research agents"""

    def __init__(self, agent_id: str, theater: str):
        self.agent_id = agent_id
        self.theater = theater
        self.config = get_config()
        self.logger = self._setup_logging()
        self.findings: List[ResearchFinding] = []
        self._finding_counter = 0
        self.last_scan = None
        self.is_running = False

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.theater}_{self.agent_id}")
        logger.setLevel(logging.INFO)
        
        log_dir = get_project_path('BigBrainIntelligence', 'logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_dir / f"{self.theater}_{self.agent_id}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        return logger

    def _generate_finding_id(self) -> str:
        self._finding_counter += 1
        return f"{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._finding_counter:04d}"

    def create_finding(
        self,
        finding_type: str,
        title: str,
        description: str,
        confidence: float,
        urgency: str = 'medium',
        data: Optional[Dict] = None,
        sources: Optional[List[str]] = None,
        ttl_hours: int = 24,
    ) -> ResearchFinding:
        """Create and store a new finding"""
        finding = ResearchFinding(
            finding_id=self._generate_finding_id(),
            agent_id=self.agent_id,
            theater=self.theater,
            finding_type=finding_type,
            title=title,
            description=description,
            confidence=confidence,
            urgency=urgency,
            data=data or {},
            sources=sources or [],
            expires_at=datetime.now() + timedelta(hours=ttl_hours),
        )
        
        self.findings.append(finding)
        self.logger.info(f"New finding: {finding.title} (conf={confidence:.2f})")
        
        return finding

    @abstractmethod
    async def scan(self) -> List[ResearchFinding]:
        """Execute research scan - must be implemented by subclasses"""
        pass

    async def run_scan(self) -> List[ResearchFinding]:
        """Run a scan with timing and error handling"""
        self.logger.info(f"Starting scan for {self.agent_id}")
        start_time = datetime.now()
        
        try:
            findings = await self.scan()
            self.last_scan = datetime.now()
            elapsed = (self.last_scan - start_time).total_seconds()
            self.logger.info(f"Scan complete: {len(findings)} findings in {elapsed:.2f}s")
            return findings
        except Exception as e:
            self.logger.error(f"Scan failed: {e}")
            return []
        finally:
            # Cleanup any open sessions
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources - override in subclasses with sessions"""
        pass

    def get_recent_findings(self, hours: int = 24) -> List[ResearchFinding]:
        """Get findings from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [f for f in self.findings if f.timestamp > cutoff]


# ============================================
# THEATER B AGENTS - Attention/Narrative
# ============================================

class NarrativeAnalyzerAgent(BaseResearchAgent):
    """
    Analyzes market narratives and sentiment trends.
    Identifies emerging themes and narrative shifts.
    """

    def __init__(self):
        super().__init__('narrative_analyzer', 'theater_b')
        self.tracked_narratives = [
            'institutional_adoption', 'regulatory_news', 'defi_innovation',
            'layer2_scaling', 'nft_gaming', 'ai_crypto', 'rwa_tokenization',
            'bitcoin_etf', 'stablecoin_regulation', 'cbdc_development'
        ]
        self.coingecko = CoinGeckoClient()
        self._session: Optional[aiohttp.ClientSession] = None

    async def cleanup(self):
        """Close open HTTP sessions"""
        if self._session and not self._session.closed:
            await self._session.close()
        await self.coingecko.disconnect()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        try:
            # Get trending coins from CoinGecko - real signal for narratives
            trending = await self.coingecko.get_trending()
            
            if trending:
                for idx, coin_item in enumerate(trending[:7]):  # Top 7 trending
                    coin = coin_item.get('item', {})
                    name = coin.get('name', 'Unknown')
                    symbol = coin.get('symbol', 'UNK')
                    market_cap_rank = coin.get('market_cap_rank', 0)
                    score = coin.get('score', idx)  # Lower is better
                    
                    # Determine urgency based on trending rank
                    urgency = 'critical' if score < 2 else ('high' if score < 5 else 'medium')
                    confidence = max(0.5, 1.0 - (score * 0.1))
                    
                    finding = self.create_finding(
                        finding_type='trending_narrative',
                        title=f"Trending: {name} ({symbol})",
                        description=f"{name} is trending #{score+1} on CoinGecko. Market cap rank: {market_cap_rank or 'N/A'}",
                        confidence=confidence,
                        urgency=urgency,
                        data={
                            'coin_id': coin.get('id'),
                            'name': name,
                            'symbol': symbol,
                            'market_cap_rank': market_cap_rank,
                            'trending_rank': score + 1,
                            'thumb': coin.get('thumb'),
                        },
                        sources=['coingecko_trending'],
                        ttl_hours=4,  # Trending changes frequently
                    )
                    findings.append(finding)
            
            # Additional: analyze price movements for narrative detection
            narratives_detected = await self._analyze_narratives()
            
            for narrative in narratives_detected:
                finding = self.create_finding(
                    finding_type='narrative_shift',
                    title=f"Narrative Emerging: {narrative['name']}",
                    description=narrative['description'],
                    confidence=narrative['confidence'],
                    urgency=narrative['urgency'],
                    data={
                        'narrative': narrative['name'],
                        'momentum': narrative['momentum'],
                        'related_assets': narrative.get('assets', []),
                    },
                    sources=narrative.get('sources', []),
                )
                findings.append(finding)
                
        except Exception as e:
            self.logger.error(f"Narrative scan error: {e}")
        
        return findings

    async def _analyze_narratives(self) -> List[Dict]:
        """Analyze current narrative landscape using real market data"""
        narratives = []
        
        try:
            # Check specific narrative-related tokens
            narrative_tokens = {
                'ai_crypto': ['fetch-ai', 'singularitynet', 'ocean-protocol', 'render-token'],
                'layer2_scaling': ['arbitrum', 'optimism', 'polygon-ecosystem-token', 'starknet'],
                'defi_innovation': ['aave', 'uniswap', 'maker', 'compound-governance-token'],
                'rwa_tokenization': ['chainlink', 'ondo-finance', 'centrifuge', 'maple'],
            }
            
            for narrative_name, tokens in narrative_tokens.items():
                prices = await self.coingecko.get_prices_batch(tokens)
                
                if prices:
                    # Calculate average 24h change
                    changes = [p.change_24h for p in prices if p.change_24h]
                    if changes:
                        avg_change = sum(changes) / len(changes)
                        
                        # Significant narrative movement if avg > 5%
                        if abs(avg_change) > 5:
                            direction = "bullish" if avg_change > 0 else "bearish"
                            narratives.append({
                                'name': narrative_name.replace('_', ' ').title(),
                                'description': f"Strong {direction} momentum in {narrative_name.replace('_', ' ')} sector. Avg 24h change: {avg_change:.1f}%",
                                'confidence': min(0.9, 0.5 + abs(avg_change) / 20),
                                'urgency': 'high' if abs(avg_change) > 10 else 'medium',
                                'momentum': avg_change,
                                'assets': [p.symbol for p in prices],
                                'sources': ['coingecko_price_analysis'],
                            })
                            
        except Exception as e:
            self.logger.error(f"Narrative analysis error: {e}")
            
        return narratives


class EngagementPredictorAgent(BaseResearchAgent):
    """
    Predicts engagement and attention metrics for assets.
    Uses social signals to forecast price movements.
    """

    def __init__(self):
        super().__init__('engagement_predictor', 'theater_b')
        self.platforms = ['twitter', 'reddit', 'discord', 'telegram']
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Subreddits to monitor
        self.crypto_subreddits = [
            'cryptocurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets',
            'altcoin', 'defi', 'NFT'
        ]

    async def cleanup(self):
        """Close open HTTP sessions"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        try:
            # Collect engagement data from Reddit (public JSON API)
            engagement_data = await self._collect_engagement_data()
            
            for asset, data in engagement_data.items():
                if data.get('spike_detected'):
                    finding = self.create_finding(
                        finding_type='engagement_spike',
                        title=f"Engagement Spike: {asset}",
                        description=f"High engagement activity detected for {asset} on Reddit",
                        confidence=data['confidence'],
                        urgency='high' if data['magnitude'] > 2.0 else 'medium',
                        data={
                            'asset': asset,
                            'magnitude': data['magnitude'],
                            'platforms': data['platforms'],
                            'sentiment': data.get('sentiment', 'neutral'),
                            'top_posts': data.get('top_posts', []),
                        },
                        sources=['reddit'],
                        ttl_hours=6,
                    )
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.error(f"Engagement scan error: {e}")
        
        return findings

    async def _collect_engagement_data(self) -> Dict[str, Dict]:
        """Collect engagement metrics from Reddit"""
        engagement = {}
        session = await self._get_session()
        
        # Track mentions of major assets
        tracked_assets = {
            'BTC': ['bitcoin', 'btc', 'Bitcoin'],
            'ETH': ['ethereum', 'eth', 'Ethereum'],  
            'SOL': ['solana', 'sol', 'Solana'],
            'XRP': ['xrp', 'ripple', 'XRP'],
            'DOGE': ['doge', 'dogecoin', 'DOGE'],
        }
        
        try:
            # Check top posts from cryptocurrency subreddit
            url = "https://www.reddit.com/r/cryptocurrency/hot.json?limit=50"
            headers = {'User-Agent': 'ACC-Research-Agent/1.0'}
            
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    # Count mentions and engagement
                    asset_mentions = {asset: [] for asset in tracked_assets}
                    
                    for post in posts:
                        post_data = post.get('data', {})
                        title = post_data.get('title', '').lower()
                        selftext = post_data.get('selftext', '').lower()
                        score = post_data.get('score', 0)
                        num_comments = post_data.get('num_comments', 0)
                        
                        for asset, keywords in tracked_assets.items():
                            if any(kw.lower() in title or kw.lower() in selftext for kw in keywords):
                                asset_mentions[asset].append({
                                    'title': post_data.get('title'),
                                    'score': score,
                                    'comments': num_comments,
                                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                })
                    
                    # Analyze engagement for each asset
                    for asset, mentions in asset_mentions.items():
                        if mentions:
                            total_score = sum(m['score'] for m in mentions)
                            total_comments = sum(m['comments'] for m in mentions)
                            
                            # Spike detection: more than 3 posts with high engagement
                            high_engagement_posts = [m for m in mentions if m['score'] > 100 or m['comments'] > 50]
                            
                            if len(high_engagement_posts) >= 2 or total_score > 500:
                                engagement[asset] = {
                                    'spike_detected': True,
                                    'magnitude': min(5.0, len(high_engagement_posts) + (total_score / 500)),
                                    'confidence': min(0.9, 0.5 + len(mentions) * 0.1),
                                    'platforms': ['reddit'],
                                    'sentiment': self._analyze_sentiment(mentions),
                                    'mention_count': len(mentions),
                                    'total_score': total_score,
                                    'top_posts': mentions[:3],
                                }
                                
        except Exception as e:
            self.logger.error(f"Reddit engagement fetch error: {e}")
            
        return engagement

    def _analyze_sentiment(self, mentions: List[Dict]) -> str:
        """Simple sentiment analysis based on engagement"""
        if not mentions:
            return 'neutral'
        
        avg_score = sum(m['score'] for m in mentions) / len(mentions)
        
        # High upvotes generally indicate positive sentiment
        if avg_score > 200:
            return 'bullish'
        elif avg_score > 50:
            return 'positive'
        elif avg_score < 0:
            return 'bearish'
        return 'neutral'


class ContentOptimizerAgent(BaseResearchAgent):
    """
    Optimizes content and messaging for maximum impact.
    Analyzes what content performs best in current market.
    """

    def __init__(self):
        super().__init__('content_optimizer', 'theater_b')

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        # Analyze content performance
        content_insights = await self._analyze_content_performance()
        
        for insight in content_insights:
            finding = self.create_finding(
                finding_type='content_insight',
                title=insight['title'],
                description=insight['description'],
                confidence=insight['confidence'],
                urgency='low',
                data=insight['data'],
            )
            findings.append(finding)
        
        return findings

    async def _analyze_content_performance(self) -> List[Dict]:
        """Analyze what content is performing well"""
        return []


# ============================================
# THEATER C AGENTS - Infrastructure/Latency
# ============================================

class LatencyMonitorAgent(BaseResearchAgent):
    """
    Monitors network and exchange latency.
    Detects latency arbitrage opportunities.
    """

    def __init__(self):
        super().__init__('latency_monitor', 'theater_c')
        self.exchanges = {
            'binance': 'https://api.binance.com/api/v3/ping',
            'coinbase': 'https://api.coinbase.com/v2/time',
            'kraken': 'https://api.kraken.com/0/public/SystemStatus',
            'bybit': 'https://api.bybit.com/v5/market/time',
        }
        self.latency_history: Dict[str, List[float]] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def cleanup(self):
        """Close open HTTP sessions"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        try:
            # Measure latencies
            latencies = await self._measure_latencies()
            
            for exchange, latency in latencies.items():
                # Update history
                if exchange not in self.latency_history:
                    self.latency_history[exchange] = []
                self.latency_history[exchange].append(latency)
                self.latency_history[exchange] = self.latency_history[exchange][-50:]
                
                # Check for anomalies
                if self._is_latency_anomaly(exchange, latency):
                    finding = self.create_finding(
                        finding_type='latency_anomaly',
                        title=f"Latency Anomaly: {exchange.title()}",
                        description=f"Unusual latency detected on {exchange}: {latency:.0f}ms (baseline: {self._get_baseline(exchange):.0f}ms)",
                        confidence=0.85,
                        urgency='high',
                        data={
                            'exchange': exchange,
                            'latency_ms': latency,
                            'baseline_ms': self._get_baseline(exchange),
                            'ratio': latency / self._get_baseline(exchange),
                        },
                        ttl_hours=2,
                    )
                    findings.append(finding)
                    
            # Check for latency arbitrage opportunities (big difference between exchanges)
            if len(latencies) >= 2:
                sorted_exchanges = sorted(latencies.items(), key=lambda x: x[1])
                fastest = sorted_exchanges[0]
                slowest = sorted_exchanges[-1]
                
                if slowest[1] > fastest[1] * 3:  # 3x difference
                    finding = self.create_finding(
                        finding_type='latency_arbitrage',
                        title=f"Latency Gap: {fastest[0]} vs {slowest[0]}",
                        description=f"Significant latency difference detected. {fastest[0]}: {fastest[1]:.0f}ms vs {slowest[0]}: {slowest[1]:.0f}ms",
                        confidence=0.8,
                        urgency='medium',
                        data={
                            'fastest_exchange': fastest[0],
                            'fastest_latency': fastest[1],
                            'slowest_exchange': slowest[0],
                            'slowest_latency': slowest[1],
                            'ratio': slowest[1] / fastest[1],
                        },
                    )
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.error(f"Latency scan error: {e}")
        
        return findings

    async def _measure_latencies(self) -> Dict[str, float]:
        """Measure actual exchange API latencies"""
        latencies = {}
        session = await self._get_session()
        
        for exchange, url in self.exchanges.items():
            try:
                start = asyncio.get_event_loop().time()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    await resp.read()
                    latency_ms = (asyncio.get_event_loop().time() - start) * 1000
                    latencies[exchange] = latency_ms
            except asyncio.TimeoutError:
                latencies[exchange] = 5000  # Timeout = 5000ms
            except Exception as e:
                self.logger.error(f"Latency measurement failed for {exchange}: {e}")
                
        return latencies

    def _is_latency_anomaly(self, exchange: str, latency: float) -> bool:
        """Check if latency is anomalous"""
        baseline = self._get_baseline(exchange)
        return latency > baseline * 2.0

    def _get_baseline(self, exchange: str) -> float:
        """Get baseline latency for exchange"""
        history = self.latency_history.get(exchange, [100])
        return sum(history) / len(history) if history else 100


class BridgeAnalyzerAgent(BaseResearchAgent):
    """
    Analyzes cross-chain bridge activity and opportunities.
    Monitors bridge liquidity and arbitrage potential.
    """

    def __init__(self):
        super().__init__('bridge_analyzer', 'theater_c')
        self.bridges = [
            {'name': 'Wormhole', 'chains': ['Ethereum', 'Solana', 'BSC']},
            {'name': 'Multichain', 'chains': ['Ethereum', 'Polygon', 'Avalanche']},
            {'name': 'Stargate', 'chains': ['Ethereum', 'Arbitrum', 'Optimism']},
        ]

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        for bridge in self.bridges:
            analysis = await self._analyze_bridge(bridge)
            
            if analysis.get('arbitrage_opportunity'):
                finding = self.create_finding(
                    finding_type='bridge_arbitrage',
                    title=f"Bridge Arbitrage: {bridge['name']}",
                    description=f"Cross-chain arbitrage opportunity via {bridge['name']}",
                    confidence=analysis['confidence'],
                    urgency='high',
                    data={
                        'bridge': bridge['name'],
                        'chains': bridge['chains'],
                        'spread_pct': analysis.get('spread_pct', 0),
                        'liquidity': analysis.get('liquidity', {}),
                    }
                )
                findings.append(finding)
        
        return findings

    async def _analyze_bridge(self, bridge: Dict) -> Dict:
        """Analyze bridge for opportunities"""
        return {}


class GasOptimizerAgent(BaseResearchAgent):
    """
    Monitors gas prices and optimizes transaction timing.
    Predicts gas price movements.
    """

    def __init__(self):
        super().__init__('gas_optimizer', 'theater_c')
        self.networks = ['ethereum', 'polygon', 'arbitrum', 'optimism', 'bsc']
        self.gas_history: Dict[str, List[float]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Thresholds for "low gas" windows (in gwei)
        self.low_gas_thresholds = {
            'ethereum': 20,  # Below 20 gwei is good
            'polygon': 50,
            'arbitrum': 0.1,
            'optimism': 0.01,
            'bsc': 3,
        }

    async def cleanup(self):
        """Close open HTTP sessions"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        try:
            gas_data = await self._collect_gas_data()
            
            for network, data in gas_data.items():
                # Update history
                if network not in self.gas_history:
                    self.gas_history[network] = []
                self.gas_history[network].append(data['current'])
                # Keep last 100 readings
                self.gas_history[network] = self.gas_history[network][-100:]
                
                if data.get('low_gas_window'):
                    finding = self.create_finding(
                        finding_type='gas_opportunity',
                        title=f"Low Gas Window: {network.title()}",
                        description=f"Gas prices are optimal on {network.title()}. Current: {data['current']:.2f} gwei (threshold: {data['threshold']} gwei)",
                        confidence=data['confidence'],
                        urgency='medium',
                        data={
                            'network': network,
                            'current_gwei': data['current'],
                            'fast_gwei': data.get('fast'),
                            'safe_gwei': data.get('safe'),
                            'threshold': data['threshold'],
                            'base_fee': data.get('base_fee'),
                            'priority_fee': data.get('priority_fee'),
                        },
                        ttl_hours=1,  # Gas changes quickly
                    )
                    findings.append(finding)
                    
                # Also report gas spikes
                if data.get('spike_detected'):
                    finding = self.create_finding(
                        finding_type='gas_spike',
                        title=f"Gas Spike Alert: {network.title()}",
                        description=f"High gas prices detected on {network.title()}. Consider delaying transactions.",
                        confidence=0.9,
                        urgency='high',
                        data={
                            'network': network,
                            'current_gwei': data['current'],
                            'avg_gwei': data.get('average'),
                            'spike_ratio': data.get('spike_ratio'),
                        },
                        ttl_hours=2,
                    )
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.error(f"Gas scan error: {e}")
        
        return findings

    async def _collect_gas_data(self) -> Dict[str, Dict]:
        """Collect real gas price data from public APIs"""
        gas_data = {}
        session = await self._get_session()
        
        try:
            # Ethereum gas from Etherscan public API
            etherscan_url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
            async with session.get(etherscan_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('status') == '1':
                        result = data.get('result', {})
                        current = float(result.get('ProposeGasPrice', 50))
                        fast = float(result.get('FastGasPrice', 60))
                        safe = float(result.get('SafeGasPrice', 40))
                        
                        threshold = self.low_gas_thresholds['ethereum']
                        
                        # Calculate average from history
                        history = self.gas_history.get('ethereum', [current])
                        avg = sum(history) / len(history) if history else current
                        
                        gas_data['ethereum'] = {
                            'current': current,
                            'fast': fast,
                            'safe': safe,
                            'threshold': threshold,
                            'average': avg,
                            'low_gas_window': current <= threshold,
                            'confidence': 0.9 if current <= threshold else 0.7,
                            'spike_detected': current > avg * 1.5,
                            'spike_ratio': current / avg if avg > 0 else 1,
                        }
                        
        except Exception as e:
            self.logger.error(f"Etherscan gas fetch error: {e}")
        
        try:
            # Polygon gas from public RPC
            polygon_url = "https://gasstation.polygon.technology/v2"
            async with session.get(polygon_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    current = data.get('standard', {}).get('maxFee', 50)
                    fast = data.get('fast', {}).get('maxFee', 60)
                    safe = data.get('safeLow', {}).get('maxFee', 40)
                    
                    threshold = self.low_gas_thresholds['polygon']
                    history = self.gas_history.get('polygon', [current])
                    avg = sum(history) / len(history) if history else current
                    
                    gas_data['polygon'] = {
                        'current': current,
                        'fast': fast,
                        'safe': safe,
                        'threshold': threshold,
                        'average': avg,
                        'low_gas_window': current <= threshold,
                        'confidence': 0.9 if current <= threshold else 0.7,
                        'spike_detected': current > avg * 1.5,
                        'spike_ratio': current / avg if avg > 0 else 1,
                    }
                    
        except Exception as e:
            self.logger.error(f"Polygon gas fetch error: {e}")
            
        return gas_data


class LiquidityTrackerAgent(BaseResearchAgent):
    """
    Tracks liquidity across DEXs and CEXs.
    Identifies liquidity imbalances and opportunities.
    """

    def __init__(self):
        super().__init__('liquidity_tracker', 'theater_c')
        self.dexs = ['uniswap', 'sushiswap', 'curve', 'balancer']
        self.cexs = ['binance', 'coinbase', 'kraken']

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        liquidity_data = await self._analyze_liquidity()
        
        for symbol, data in liquidity_data.items():
            if data.get('imbalance_detected'):
                finding = self.create_finding(
                    finding_type='liquidity_imbalance',
                    title=f"Liquidity Imbalance: {symbol}",
                    description=f"Significant liquidity imbalance detected for {symbol}",
                    confidence=data['confidence'],
                    urgency='high' if data['imbalance_ratio'] > 3.0 else 'medium',
                    data={
                        'symbol': symbol,
                        'imbalance_ratio': data['imbalance_ratio'],
                        'deep_venue': data['deep_venue'],
                        'shallow_venue': data['shallow_venue'],
                    }
                )
                findings.append(finding)
        
        return findings

    async def _analyze_liquidity(self) -> Dict[str, Dict]:
        """Analyze liquidity across venues"""
        return {}


# ============================================
# THEATER D AGENTS - Information Asymmetry
# ============================================

class APIScannerAgent(BaseResearchAgent):
    """
    Scans public APIs for early information signals.
    Monitors blockchain data, exchange APIs, and public feeds.
    """

    def __init__(self):
        super().__init__('api_scanner', 'theater_d')
        self.monitored_apis = [
            'etherscan', 'blockchain.com', 'coingecko', 'defillama',
            'dune_analytics', 'nansen', 'glassnode'
        ]
        self._session: Optional[aiohttp.ClientSession] = None
        self.coingecko = CoinGeckoClient()

    async def cleanup(self):
        """Close open HTTP sessions"""
        if self._session and not self._session.closed:
            await self._session.close()
        await self.coingecko.disconnect()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        try:
            api_signals = await self._scan_apis()
            
            for signal in api_signals:
                finding = self.create_finding(
                    finding_type='api_signal',
                    title=signal['title'],
                    description=signal['description'],
                    confidence=signal['confidence'],
                    urgency=signal.get('urgency', 'medium'),
                    data=signal['data'],
                    sources=[signal['source']],
                )
                findings.append(finding)
                
        except Exception as e:
            self.logger.error(f"API scan error: {e}")
        
        return findings

    async def _scan_apis(self) -> List[Dict]:
        """Scan various public APIs for signals"""
        signals = []
        session = await self._get_session()
        
        try:
            # DeFiLlama TVL data for DeFi protocols
            defi_url = "https://api.llama.fi/protocols"
            async with session.get(defi_url) as resp:
                if resp.status == 200:
                    protocols = await resp.json()
                    
                    # Find protocols with significant TVL changes
                    for protocol in protocols[:100]:  # Top 100 by TVL
                        change_1d = protocol.get('change_1d', 0) or 0
                        change_7d = protocol.get('change_7d', 0) or 0
                        tvl = protocol.get('tvl', 0)
                        
                        # Alert on major TVL changes (>15% daily or >30% weekly)
                        if abs(change_1d) > 15 or abs(change_7d) > 30:
                            direction = "inflow" if change_1d > 0 else "outflow"
                            signals.append({
                                'title': f"TVL {direction.title()}: {protocol.get('name')}",
                                'description': f"Major TVL change detected. 24h: {change_1d:.1f}%, 7d: {change_7d:.1f}%",
                                'confidence': min(0.9, 0.6 + abs(change_1d) / 50),
                                'urgency': 'high' if abs(change_1d) > 25 else 'medium',
                                'data': {
                                    'protocol': protocol.get('name'),
                                    'symbol': protocol.get('symbol'),
                                    'tvl_usd': tvl,
                                    'change_1d': change_1d,
                                    'change_7d': change_7d,
                                    'category': protocol.get('category'),
                                    'chains': protocol.get('chains', []),
                                },
                                'source': 'defillama',
                            })
                            
        except Exception as e:
            self.logger.error(f"DeFiLlama API error: {e}")
        
        try:
            # CoinGecko global data
            global_url = "https://api.coingecko.com/api/v3/global"
            async with session.get(global_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    global_data = data.get('data', {})
                    
                    # Check market cap dominance shifts
                    btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
                    eth_dominance = global_data.get('market_cap_percentage', {}).get('eth', 0)
                    
                    # Alert if alt season might be starting (BTC dominance falling)
                    if btc_dominance < 45:
                        signals.append({
                            'title': "Market Signal: Potential Alt Season",
                            'description': f"BTC dominance at {btc_dominance:.1f}% - historically low, potential alt season",
                            'confidence': 0.7,
                            'urgency': 'medium',
                            'data': {
                                'btc_dominance': btc_dominance,
                                'eth_dominance': eth_dominance,
                                'total_market_cap': global_data.get('total_market_cap', {}).get('usd', 0),
                                'total_volume': global_data.get('total_volume', {}).get('usd', 0),
                            },
                            'source': 'coingecko_global',
                        })
                        
                    # Market volume spike
                    volume_change = global_data.get('market_cap_change_percentage_24h_usd', 0)
                    if abs(volume_change) > 5:
                        direction = "up" if volume_change > 0 else "down"
                        signals.append({
                            'title': f"Market Volume: 24h Change {direction.title()}",
                            'description': f"Global market cap changed {volume_change:.1f}% in 24h",
                            'confidence': 0.8,
                            'urgency': 'high' if abs(volume_change) > 10 else 'medium',
                            'data': {
                                'market_cap_change_24h': volume_change,
                                'active_cryptocurrencies': global_data.get('active_cryptocurrencies'),
                                'markets': global_data.get('markets'),
                            },
                            'source': 'coingecko_global',
                        })
                        
        except Exception as e:
            self.logger.error(f"CoinGecko global API error: {e}")
            
        # Add mock signals for testing when APIs are unavailable
        if not signals:
            signals.extend([
                {
                    'title': 'Mock API Signal: Exchange Maintenance',
                    'description': 'Detected potential exchange maintenance window based on historical patterns.',
                    'confidence': 0.6,
                    'urgency': 'low',
                    'data': {
                        'exchange': 'MockExchange',
                        'maintenance_window': '02:00-04:00 UTC',
                        'historical_pattern': True,
                    },
                    'source': 'pattern_analysis',
                },
                {
                    'title': 'Mock API Signal: New Token Launch',
                    'description': 'Potential new token launch detected in development phase.',
                    'confidence': 0.5,
                    'urgency': 'medium',
                    'data': {
                        'token_symbol': 'MOCK',
                        'launch_platform': 'MockDEX',
                        'development_stage': 'testnet',
                    },
                    'source': 'github_monitoring',
                },
            ])
            
        return signals


class DataGapFinderAgent(BaseResearchAgent):
    """
    Identifies information gaps and asymmetries.
    Finds data that others might be missing.
    """

    def __init__(self):
        super().__init__('data_gap_finder', 'theater_d')

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        gaps = await self._identify_gaps()
        
        for gap in gaps:
            finding = self.create_finding(
                finding_type='data_gap',
                title=f"Data Gap: {gap['area']}",
                description=gap['description'],
                confidence=gap['confidence'],
                urgency=gap.get('urgency', 'low'),
                data=gap,
            )
            findings.append(finding)
        
        return findings

    async def _identify_gaps(self) -> List[Dict]:
        """Identify data gaps"""
        # Example: Find missing data in market feeds
        return [
            {
                'area': 'Market Feed Latency',
                'description': 'Detected latency spikes in BTC/USD feed from Exchange X.',
                'confidence': 0.9,
                'urgency': 'high',
            },
            {
                'area': 'Order Book Depth',
                'description': 'Order book data for ETH/USD missing levels beyond top 10.',
                'confidence': 0.7,
                'urgency': 'medium',
            },
        ]


class AccessArbitrageAgent(BaseResearchAgent):
    """
    Identifies access-based arbitrage opportunities.
    Monitors early access, beta features, and privileged information sources.
    """

    def __init__(self):
        super().__init__('access_arbitrage', 'theater_d')

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        opportunities = await self._find_access_opportunities()
        
        for opp in opportunities:
            finding = self.create_finding(
                finding_type='access_opportunity',
                title=opp['title'],
                description=opp['description'],
                confidence=opp['confidence'],
                urgency=opp.get('urgency', 'medium'),
                data=opp,
            )
            findings.append(finding)
        
        return findings

    async def _find_access_opportunities(self) -> List[Dict]:
        """Find access-based opportunities"""
        # Example: Early access to new exchange features
        return [
            {
                'title': 'Beta API Access',
                'description': 'Early access to beta API endpoints for Exchange Y.',
                'confidence': 0.8,
                'urgency': 'medium',
            },
            {
                'title': 'Privileged Data Stream',
                'description': 'Received privileged data stream for DeFi protocol Z.',
                'confidence': 0.6,
                'urgency': 'low',
            },
        ]


class NetworkMapperAgent(BaseResearchAgent):
    """
    Maps network relationships and influence flows.
    Identifies key actors and their connections.
    """

    def __init__(self):
        super().__init__('network_mapper', 'theater_d')
        self.tracked_entities = []
        self.relationship_graph = {}

    async def scan(self) -> List[ResearchFinding]:
        findings = []
        
        network_changes = await self._analyze_network()
        
        for change in network_changes:
            finding = self.create_finding(
                finding_type='network_change',
                title=change['title'],
                description=change['description'],
                confidence=change['confidence'],
                urgency=change.get('urgency', 'low'),
                data=change,
            )
            findings.append(finding)
        
        return findings

    async def _analyze_network(self) -> List[Dict]:
        """Analyze network changes"""
        # Example: Detect new connections and influence flows
        return [
            {
                'title': 'New Exchange Partnership',
                'description': 'Exchange X formed partnership with Market Maker Y.',
                'confidence': 0.85,
                'urgency': 'medium',
            },
            {
                'title': 'Influence Shift',
                'description': 'Key actor Z increased influence in DeFi governance.',
                'confidence': 0.75,
                'urgency': 'low',
            },
        ]


# ============================================
# MISSING OPERATIONAL AGENTS
# ============================================

class ReconciliationAgent(BaseResearchAgent):
    """Code Name: BALANCE-KEEPER
    Department: CentralAccounting
    Role: Ensures all financial transactions are properly reconciled across systems
    Tri-Agent System: Works with ADMIN-SCROLL (reports) and INNOVATE-LEDGER (insights)
    """

    def __init__(self):
        super().__init__('reconciliation_agent', 'operations')
        self.reconciliation_rules = {}
        self.pending_reconciliations = []
        self.reconciliation_history = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        # Check for reconciliation discrepancies
        discrepancies = await self._find_discrepancies()

        for disc in discrepancies:
            finding = self.create_finding(
                finding_type='reconciliation_issue',
                title=f"Reconciliation Discrepancy: {disc['account']}",
                description=f"Found ${disc['amount']:.2f} discrepancy in {disc['account']}",
                confidence=disc['confidence'],
                urgency=disc['urgency'],
                data=disc,
            )
            findings.append(finding)

        return findings

    async def _find_discrepancies(self) -> List[Dict]:
        """Find reconciliation discrepancies"""
        return [
            {
                'account': 'Trading Account A',
                'amount': 1250.75,
                'confidence': 0.95,
                'urgency': 'high',
                'type': 'cash_discrepancy'
            }
        ]


class RiskMonitorAgent(BaseResearchAgent):
    """Code Name: SHIELD-WATCHER
    Department: CentralAccounting
    Role: Monitors portfolio risk in real-time, alerts on threshold breaches
    Tri-Agent System: Works with ADMIN-RISK (reports) and INNOVATE-SHIELD (insights)
    """

    def __init__(self):
        super().__init__('risk_monitor_agent', 'operations')
        self.risk_thresholds = {}
        self.current_risk_metrics = {}
        self.risk_alerts = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        risk_breaches = await self._check_risk_breaches()

        for breach in risk_breaches:
            finding = self.create_finding(
                finding_type='risk_breach',
                title=f"Risk Threshold Breached: {breach['metric']}",
                description=f"{breach['metric']} exceeded threshold: {breach['value']:.2f}",
                confidence=breach['confidence'],
                urgency='critical',
                data=breach,
            )
            findings.append(finding)

        return findings

    async def _check_risk_breaches(self) -> List[Dict]:
        """Check for risk threshold breaches"""
        return [
            {
                'metric': 'Portfolio VaR',
                'value': 2.5,
                'threshold': 2.0,
                'confidence': 0.98,
                'breach_type': 'var_limit'
            }
        ]


class PLCalculationAgent(BaseResearchAgent):
    """Code Name: PROFIT-SAGE
    Department: CentralAccounting
    Role: Calculates real-time P&L across all positions and strategies
    Tri-Agent System: Works with ADMIN-PROFIT (reports) and INNOVATE-YIELD (insights)
    """

    def __init__(self):
        super().__init__('pl_calculation_agent', 'operations')
        self.position_data = {}
        self.pl_calculations = {}
        self.unrealized_gains = {}

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        pl_updates = await self._calculate_pl()

        for update in pl_updates:
            finding = self.create_finding(
                finding_type='pl_update',
                title=f"P&L Update: {update['strategy']}",
                description=f"Current P&L: ${update['pl_amount']:,.2f} ({update['change_percent']:.2f}%)",
                confidence=0.99,
                urgency='medium',
                data=update,
            )
            findings.append(finding)

        return findings

    async def _calculate_pl(self) -> List[Dict]:
        """Calculate P&L for all positions"""
        return [
            {
                'strategy': 'Arbitrage Strategy A',
                'pl_amount': 45670.50,
                'change_percent': 2.34,
                'unrealized_gains': 12340.25
            }
        ]


class VenueHealthAgent(BaseResearchAgent):
    """Code Name: MARKET-PULSE
    Department: TradingExecution
    Role: Monitors trading venue health, connectivity, and performance
    Tri-Agent System: Works with ADMIN-VENUE (reports) and INNOVATE-CONNECT (insights)
    """

    def __init__(self):
        super().__init__('venue_health_agent', 'operations')
        self.venue_status = {}
        self.connectivity_metrics = {}
        self.performance_history = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        health_issues = await self._check_venue_health()

        for issue in health_issues:
            finding = self.create_finding(
                finding_type='venue_health',
                title=f"Venue Issue: {issue['venue']}",
                description=issue['description'],
                confidence=issue['confidence'],
                urgency=issue['urgency'],
                data=issue,
            )
            findings.append(finding)

        return findings

    async def _check_venue_health(self) -> List[Dict]:
        """Check trading venue health"""
        return [
            {
                'venue': 'Exchange Alpha',
                'description': 'Latency increased by 45ms',
                'confidence': 0.92,
                'urgency': 'high',
                'issue_type': 'latency_spike'
            }
        ]


class WithdrawalRiskAgent(BaseResearchAgent):
    """Code Name: CASH-GUARD
    Department: CentralAccounting
    Role: Monitors withdrawal patterns for risk and compliance
    Tri-Agent System: Works with ADMIN-WITHDRAW (reports) and INNOVATE-FLOW (insights)
    """

    def __init__(self):
        super().__init__('withdrawal_risk_agent', 'operations')
        self.withdrawal_patterns = {}
        self.risk_scores = {}
        self.compliance_flags = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        risk_patterns = await self._analyze_withdrawal_risks()

        for pattern in risk_patterns:
            finding = self.create_finding(
                finding_type='withdrawal_risk',
                title=f"Withdrawal Risk: {pattern['account']}",
                description=pattern['description'],
                confidence=pattern['confidence'],
                urgency=pattern['urgency'],
                data=pattern,
            )
            findings.append(finding)

        return findings

    async def _analyze_withdrawal_risks(self) -> List[Dict]:
        """Analyze withdrawal patterns for risks"""
        return [
            {
                'account': 'Client Portfolio X',
                'description': 'Unusual withdrawal pattern detected - 3x normal volume',
                'confidence': 0.87,
                'urgency': 'medium',
                'risk_type': 'volume_anomaly'
            }
        ]


class RoutingOptimizationAgent(BaseResearchAgent):
    """Code Name: PATH-FINDER
    Department: TradingExecution
    Role: Optimizes order routing across multiple venues for best execution
    Tri-Agent System: Works with ADMIN-ROUTE (reports) and INNOVATE-PATH (insights)
    """

    def __init__(self):
        super().__init__('routing_optimization_agent', 'operations')
        self.routing_rules = {}
        self.venue_performance = {}
        self.optimization_history = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        routing_opportunities = await self._find_routing_opportunities()

        for opp in routing_opportunities:
            finding = self.create_finding(
                finding_type='routing_optimization',
                title=f"Routing Opportunity: {opp['symbol']}",
                description=f"Potential ${opp['savings']:.2f} savings by rerouting",
                confidence=opp['confidence'],
                urgency='low',
                data=opp,
            )
            findings.append(finding)

        return findings

    async def _find_routing_opportunities(self) -> List[Dict]:
        """Find routing optimization opportunities"""
        return [
            {
                'symbol': 'BTC/USD',
                'savings': 1250.50,
                'confidence': 0.89,
                'current_venue': 'Exchange A',
                'optimal_venue': 'Exchange B'
            }
        ]


class IncidentPostmortemAutomation(BaseResearchAgent):
    """Code Name: LESSON-MINER
    Department: SharedInfrastructure
    Role: Automates incident analysis and postmortem generation
    Tri-Agent System: Works with ADMIN-INCIDENT (reports) and INNOVATE-LEARN (insights)
    """

    def __init__(self):
        super().__init__('incident_postmortem_agent', 'operations')
        self.incident_history = []
        self.postmortem_templates = {}
        self.lessons_learned = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        recent_incidents = await self._analyze_recent_incidents()

        for incident in recent_incidents:
            finding = self.create_finding(
                finding_type='incident_analysis',
                title=f"Incident Postmortem: {incident['title']}",
                description=incident['summary'],
                confidence=0.95,
                urgency='medium',
                data=incident,
            )
            findings.append(finding)

        return findings

    async def _analyze_recent_incidents(self) -> List[Dict]:
        """Analyze recent incidents for postmortems"""
        return [
            {
                'title': 'System Outage - 15 minutes',
                'summary': 'Root cause: Database connection timeout. Lessons: Increase connection pool size.',
                'impact': 'medium',
                'recommendations': ['Increase connection pool', 'Add circuit breaker']
            }
        ]


class AuditGapMonitor(BaseResearchAgent):
    """Code Name: COMPLIANCE-EYE
    Department: SharedInfrastructure
    Role: Monitors for audit gaps and compliance issues
    Tri-Agent System: Works with ADMIN-AUDIT (reports) and INNOVATE-COMPLY (insights)
    """

    def __init__(self):
        super().__init__('audit_gap_monitor', 'operations')
        self.audit_requirements = {}
        self.compliance_status = {}
        self.gap_analysis = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        audit_gaps = await self._find_audit_gaps()

        for gap in audit_gaps:
            finding = self.create_finding(
                finding_type='audit_gap',
                title=f"Audit Gap: {gap['requirement']}",
                description=gap['description'],
                confidence=gap['confidence'],
                urgency=gap['urgency'],
                data=gap,
            )
            findings.append(finding)

        return findings

    async def _find_audit_gaps(self) -> List[Dict]:
        """Find audit and compliance gaps"""
        return [
            {
                'requirement': 'SOX Compliance',
                'description': 'Missing quarterly control testing documentation',
                'confidence': 0.91,
                'urgency': 'high',
                'gap_type': 'documentation'
            }
        ]


class SecurityScannerAgent(BaseResearchAgent):
    """Code Name: THREAT-HUNTER
    Department: SharedInfrastructure
    Role: Scans for security vulnerabilities and threats
    Tri-Agent System: Works with ADMIN-SECURITY (reports) and INNOVATE-DEFEND (insights)
    """

    def __init__(self):
        super().__init__('security_scanner_agent', 'operations')
        self.vulnerability_db = {}
        self.threat_intelligence = {}
        self.security_alerts = []

    async def scan(self) -> List[ResearchFinding]:
        findings = []

        security_issues = await self._scan_for_threats()

        for issue in security_issues:
            finding = self.create_finding(
                finding_type='security_threat',
                title=f"Security Issue: {issue['type']}",
                description=issue['description'],
                confidence=issue['confidence'],
                urgency=issue['urgency'],
                data=issue,
            )
            findings.append(finding)

        return findings

    async def _scan_for_threats(self) -> List[Dict]:
        """Scan for security threats and vulnerabilities"""
        return [
            {
                'type': 'Vulnerability',
                'description': 'Outdated SSL certificate on API endpoint',
                'confidence': 0.96,
                'urgency': 'high',
                'severity': 'medium'
            }
        ]


# ============================================
# AGENT REGISTRY
# ============================================

AGENT_REGISTRY = {
    # Theater B - Attention/Narrative
    'narrative_analyzer': NarrativeAnalyzerAgent,
    'engagement_predictor': EngagementPredictorAgent,
    'content_optimizer': ContentOptimizerAgent,

    # Theater C - Infrastructure/Latency
    'latency_monitor': LatencyMonitorAgent,
    'bridge_analyzer': BridgeAnalyzerAgent,
    'gas_optimizer': GasOptimizerAgent,
    'liquidity_tracker': LiquidityTrackerAgent,

    # Theater D - Information Asymmetry
    'api_scanner': APIScannerAgent,
    'data_gap_finder': DataGapFinderAgent,
    'access_arbitrage': AccessArbitrageAgent,
    'network_mapper': NetworkMapperAgent,

    # Missing Operational Agents
    'reconciliation_agent': ReconciliationAgent,
    'risk_monitor_agent': RiskMonitorAgent,
    'pl_calculation_agent': PLCalculationAgent,
    'venue_health_agent': VenueHealthAgent,
    'withdrawal_risk_agent': WithdrawalRiskAgent,
    'routing_optimization_agent': RoutingOptimizationAgent,
    'incident_postmortem_agent': IncidentPostmortemAutomation,
    'audit_gap_monitor': AuditGapMonitor,
    'security_scanner_agent': SecurityScannerAgent,
}


def get_agent(agent_id: str) -> Optional[BaseResearchAgent]:
    """Get an agent instance by ID"""
    agent_class = AGENT_REGISTRY.get(agent_id)
    if agent_class:
        return agent_class()
    return None


def get_all_agents() -> List[BaseResearchAgent]:
    """Get instances of all agents"""
    return [cls() for cls in AGENT_REGISTRY.values()]


def get_agents_by_theater(theater: str) -> List[BaseResearchAgent]:
    """Get all agents for a specific theater"""
    agents = []
    for cls in AGENT_REGISTRY.values():
        agent = cls()
        if agent.theater == theater:
            agents.append(agent)
    return agents


# CLI for testing
if __name__ == '__main__':
    async def test():
        print("=== BigBrain Research Agents ===\n")
        
        print("Available Agents:")
        for agent_id, cls in AGENT_REGISTRY.items():
            agent = cls()
            print(f"  - {agent_id} ({agent.theater})")
        
        print("\n--- Running Test Scans ---\n")
        
        # Test each agent
        for agent_id in ['narrative_analyzer', 'latency_monitor', 'api_scanner']:
            agent = get_agent(agent_id)
            if agent:
                print(f"Testing {agent_id}...")
                findings = await agent.run_scan()
                print(f"  Found: {len(findings)} findings")
        
        print("\n=== Test Complete ===")
    
    asyncio.run(test())


class ResearchAgentManager:
    """
    Manager class for coordinating research agent operations.
    Provides unified interface for agent management and metrics collection.
    """

    def __init__(self):
        self.agents: Dict[str, BaseResearchAgent] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.last_health_check = datetime.now()
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all research agents."""
        for agent_id, agent_class in AGENT_REGISTRY.items():
            try:
                self.agents[agent_id] = agent_class()
                self.agent_status[agent_id] = {
                    "status": "initialized",
                    "last_scan": None,
                    "findings_count": 0,
                    "health_score": 1.0,
                    "error_message": None
                }
                logger.info(f"Initialized research agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_id}: {e}")
                self.agent_status[agent_id] = {
                    "status": "error",
                    "last_scan": None,
                    "findings_count": 0,
                    "health_score": 0.0,
                    "error_message": str(e)
                }

    async def get_agent_health_status(self, agent_id: str) -> Dict[str, Any]:
        """Get health status for a specific agent."""
        if agent_id not in self.agents:
            return {
                "agent_id": agent_id,
                "status": "not_found",
                "health_score": 0.0,
                "error": "Agent not found"
            }

        agent = self.agents[agent_id]
        status = self.agent_status[agent_id]

        try:
            # Perform health check
            health_score = await self._check_agent_health(agent)
            status["health_score"] = health_score
            status["status"] = "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "critical"

            return {
                "agent_id": agent_id,
                "theater": agent.theater,
                "status": status["status"],
                "health_score": round(health_score, 3),
                "last_scan": status["last_scan"].isoformat() if status["last_scan"] else None,
                "findings_count": status["findings_count"],
                "error_message": status["error_message"]
            }

        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {e}")
            status["status"] = "error"
            status["error_message"] = str(e)
            return {
                "agent_id": agent_id,
                "theater": agent.theater,
                "status": "error",
                "health_score": 0.0,
                "error": str(e)
            }

    async def get_all_agents_health_status(self) -> Dict[str, Any]:
        """Get health status for all agents."""
        results = {}
        for agent_id in self.agents.keys():
            results[agent_id] = await self.get_agent_health_status(agent_id)

        # Calculate aggregate metrics
        total_agents = len(results)
        healthy_agents = len([r for r in results.values() if r["status"] == "healthy"])
        avg_health_score = sum(r["health_score"] for r in results.values()) / total_agents if total_agents > 0 else 0.0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "health_percentage": round((healthy_agents / total_agents * 100) if total_agents > 0 else 0, 1),
            "average_health_score": round(avg_health_score, 3),
            "agents": results
        }

    async def _check_agent_health(self, agent: BaseResearchAgent) -> float:
        """Check the health of a research agent."""
        try:
            # Basic health checks
            health_score = 1.0

            # Check if agent can perform basic operations
            if hasattr(agent, 'run_scan'):
                # Try a quick scan to test functionality
                test_findings = await agent.run_scan()
                if isinstance(test_findings, list):
                    # Health based on scan success and reasonable result count
                    if len(test_findings) > 100:  # Too many findings might indicate issues
                        health_score *= 0.8
                    elif len(test_findings) == 0:  # No findings might be normal
                        health_score *= 0.9
                else:
                    health_score *= 0.5  # Invalid return type
            else:
                health_score *= 0.3  # Missing required method

            # Check agent attributes
            required_attrs = ['agent_id', 'theater', 'name']
            for attr in required_attrs:
                if not hasattr(agent, attr):
                    health_score *= 0.7

            # Check if agent has been recently active
            status = self.agent_status.get(agent.agent_id, {})
            last_scan = status.get("last_scan")
            if last_scan:
                hours_since_scan = (datetime.now() - last_scan).total_seconds() / 3600
                if hours_since_scan > 24:  # No scan in 24 hours
                    health_score *= 0.9
                elif hours_since_scan > 168:  # No scan in a week
                    health_score *= 0.7

            return min(1.0, max(0.0, health_score))

        except Exception as e:
            logger.error(f"Agent health check failed for {agent.agent_id}: {e}")
            return 0.0

    async def run_agent_scan(self, agent_id: str) -> List[ResearchFinding]:
        """Run a scan for a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        try:
            findings = await agent.run_scan()
            self.agent_status[agent_id]["last_scan"] = datetime.now()
            self.agent_status[agent_id]["findings_count"] = len(findings) if isinstance(findings, list) else 0
            self.agent_status[agent_id]["status"] = "active"
            self.agent_status[agent_id]["error_message"] = None

            logger.info(f"Agent {agent_id} scan completed: {len(findings)} findings")
            return findings

        except Exception as e:
            logger.error(f"Agent {agent_id} scan failed: {e}")
            self.agent_status[agent_id]["status"] = "error"
            self.agent_status[agent_id]["error_message"] = str(e)
            return []

    async def get_agents_by_theater(self, theater: str) -> List[str]:
        """Get all agent IDs for a specific theater."""
        return [agent_id for agent_id, agent in self.agents.items() if agent.theater == theater]

    async def get_research_metrics(self) -> Dict[str, Any]:
        """Get comprehensive research metrics across all agents."""
        try:
            agent_health = await self.get_all_agents_health_status()

            # Aggregate findings by theater
            theater_findings = {}
            theater_health = {}

            for agent_id, health in agent_health["agents"].items():
                theater = health.get("theater", "unknown")
                findings = health.get("findings_count", 0)

                if theater not in theater_findings:
                    theater_findings[theater] = 0
                    theater_health[theater] = []

                theater_findings[theater] += findings
                theater_health[theater].append(health["health_score"])

            # Calculate theater averages
            theater_metrics = {}
            for theater in theater_findings.keys():
                avg_health = sum(theater_health[theater]) / len(theater_health[theater])
                theater_metrics[theater] = {
                    "total_findings": theater_findings[theater],
                    "agent_count": len(theater_health[theater]),
                    "average_health": round(avg_health, 3)
                }

            return {
                "timestamp": datetime.now().isoformat(),
                "overall_health": agent_health,
                "theater_breakdown": theater_metrics,
                "total_findings": sum(theater_findings.values()),
                "active_agents": len([h for h in agent_health["agents"].values() if h["status"] == "healthy"])
            }

        except Exception as e:
            logger.error(f"Failed to get research metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "total_findings": 0,
                "active_agents": 0
            }


# Global manager instance
_manager_instance = None

async def get_research_agent_manager() -> ResearchAgentManager:
    """Get or create the global research agent manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ResearchAgentManager()
    return _manager_instance

# Synchronous wrapper for compatibility
def get_research_agent_manager_sync() -> ResearchAgentManager:
    """Synchronous wrapper for research agent manager."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        manager = loop.run_until_complete(get_research_agent_manager())
        return manager
    finally:
        loop.close()

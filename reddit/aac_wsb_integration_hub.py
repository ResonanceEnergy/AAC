#!/usr/bin/env python3
"""
AAC WallStreetBets Integration Hub
==================================

Integrates the advanced WallStreetBets sentiment analysis with the full AAC arbitrage system.
Combines GME-style analysis with real-time market data and arbitrage signal generation.

Features:
- Real-time WSB sentiment analysis
- GME-style squeeze detection
- Multi-source arbitrage signals
- Integration with WallStreetOdds data
- Temporal sentiment tracking
- Word association analysis

This creates the complete sentiment-enhanced arbitrage detection system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Import AAC components
from aac_wsb_sentiment_analyzer import AACWSBSentimentAnalyzer, SentimentResult
from aac_reddit_web_scraper import AACRedditWebScraper
from aac_wallstreetodds_integration import AACWallStreetOddsIntegration
from aac_arbitrage_execution_system import AACArbitrageEngine

@dataclass
class WSBAnalysisResult:
    """Comprehensive WSB analysis result"""
    sentiment_summary: Dict[str, Any]
    word_associations: Dict[str, Any]
    squeeze_signals: Dict[str, Any]
    arbitrage_signals: List[Dict[str, Any]]
    temporal_analysis: pd.DataFrame
    confidence_score: float
    timestamp: datetime

@dataclass
class EnhancedArbitrageSignal:
    """Arbitrage signal enhanced with WSB sentiment"""
    ticker: str
    signal_type: str
    strength: float
    confidence: float
    sources: List[str]
    sentiment_score: float
    technical_score: float
    fundamental_score: float
    combined_score: float
    risk_level: str
    timestamp: datetime
    metadata: Dict[str, Any]

class AACWSBIntegrationHub:
    """
    Central hub integrating WallStreetBets sentiment analysis with AAC arbitrage system.

    Combines multiple data sources for enhanced arbitrage signal generation.
    """

    def __init__(self):
        """Initialize the WSB integration hub"""
        self.sentiment_analyzer = AACWSBSentimentAnalyzer()
        self.reddit_scraper = AACRedditWebScraper()
        self.wallstreetodds = AACWallStreetOddsIntegration()
        self.arbitrage_engine = AACArbitrageEngine()

        # Key terms to analyze (from GME case study)
        self.key_terms = ['short', 'squeeze', 'citron', 'retail', 'tendies',
                         'diamond', 'hands', 'moon', 'rocket', 'crash']

        print("✅ AAC WallStreetBets Integration Hub initialized")

    async def analyze_current_wsb_sentiment(self, limit_posts: int = 100) -> WSBAnalysisResult:
        """Analyze current WallStreetBets sentiment for arbitrage signals"""
        print("🔍 Analyzing current WallStreetBets sentiment...")

        try:
            # Get recent Reddit data
            reddit_data = await self._get_recent_reddit_data(limit_posts)

            if not reddit_data:
                print("⚠️  No Reddit data available")
                return self._create_empty_result()

            # Analyze sentiment
            sentiment_results = []
            texts = []
            timestamps = []

            for post in reddit_data:
                result = self.sentiment_analyzer.analyze_sentiment(post['text'])
                if result:
                    sentiment_results.append(result)
                    texts.append(post['text'])
                    timestamps.append(post['timestamp'])

            # Word association analysis
            word_associations = {}
            for term in self.key_terms:
                assoc = self.sentiment_analyzer.analyze_word_associations(texts, term)
                word_associations[term] = {
                    'cooccurrence_count': assoc.cooccurrence_count,
                    'sentiment_correlation': assoc.sentiment_correlation,
                    'top_associations': dict(sorted(assoc.associated_words.items(),
                                                  key=lambda x: x[1], reverse=True)[:5])
                }

            # Temporal analysis
            temporal_df = self.sentiment_analyzer.analyze_temporal_sentiment(texts, timestamps)

            # Squeeze detection
            squeeze_signals = self.sentiment_analyzer.detect_gme_style_squeeze(temporal_df)

            # Generate arbitrage signals
            arbitrage_signals = self.sentiment_analyzer.generate_arbitrage_signals(sentiment_results)

            # Calculate overall confidence
            confidence_score = np.mean([r.confidence for r in sentiment_results]) if sentiment_results else 0

            # Sentiment summary
            sentiment_summary = self._create_sentiment_summary(sentiment_results)

            return WSBAnalysisResult(
                sentiment_summary=sentiment_summary,
                word_associations=word_associations,
                squeeze_signals=squeeze_signals,
                arbitrage_signals=arbitrage_signals,
                temporal_analysis=temporal_df,
                confidence_score=confidence_score,
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"❌ WSB analysis failed: {e}")
            return self._create_empty_result()

    async def _get_recent_reddit_data(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent Reddit data (placeholder - would integrate with actual scraper)"""
        # This would integrate with the actual Reddit scraper
        # For demo purposes, return sample data
        return [
            {
                'text': 'GME to the moon! Short squeeze incoming!',
                'timestamp': datetime.now() - timedelta(hours=1),
                'score': 150,
                'comments': 45
            },
            {
                'text': 'Just bought more TSLA calls. This is the rocket we need.',
                'timestamp': datetime.now() - timedelta(hours=2),
                'score': 89,
                'comments': 23
            },
            {
                'text': 'Citron research is bullshit. Retail traders unite!',
                'timestamp': datetime.now() - timedelta(hours=3),
                'score': 234,
                'comments': 67
            }
        ][:limit]

    def _create_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Create sentiment summary statistics"""
        if not results:
            return {'total_posts': 0, 'avg_sentiment': 0, 'sentiment_distribution': {}}

        sentiments = [r.overall_sentiment for r in results]
        sentiment_counts = pd.Series(sentiments).value_counts().to_dict()

        return {
            'total_posts': len(results),
            'avg_vader_compound': np.mean([r.vader_compound for r in results]),
            'avg_textblob_polarity': np.mean([r.textblob_polarity for r in results]),
            'avg_financial_score': np.mean([r.financial_score for r in results]),
            'avg_wsb_score': np.mean([r.wsb_score for r in results]),
            'sentiment_distribution': sentiment_counts,
            'confidence_avg': np.mean([r.confidence for r in results])
        }

    def _create_empty_result(self) -> WSBAnalysisResult:
        """Create empty result for error cases"""
        return WSBAnalysisResult(
            sentiment_summary={},
            word_associations={},
            squeeze_signals={},
            arbitrage_signals=[],
            temporal_analysis=pd.DataFrame(),
            confidence_score=0.0,
            timestamp=datetime.now()
        )

    def enhance_arbitrage_signals(self, wsb_analysis: WSBAnalysisResult,
                                wallstreetodds_signals: List[Dict[str, Any]] = None) -> List[EnhancedArbitrageSignal]:
        """Enhance arbitrage signals with WSB sentiment data"""
        enhanced_signals = []

        # Get base signals from WSB analysis
        base_signals = wsb_analysis.arbitrage_signals

        # Enhance with WallStreetOdds data if available
        if wallstreetodds_signals:
            base_signals.extend(wallstreetodds_signals)

        # Group signals by ticker
        ticker_signals = {}
        for signal in base_signals:
            ticker = signal.get('ticker', 'UNKNOWN')
            if ticker not in ticker_signals:
                ticker_signals[ticker] = []
            ticker_signals[ticker].append(signal)

        # Create enhanced signals
        for ticker, signals in ticker_signals.items():
            # Aggregate different signal types
            sentiment_signals = [s for s in signals if 'sentiment' in s.get('signal_type', '')]
            technical_signals = [s for s in signals if 'technical' in s.get('signal_type', '')]
            fundamental_signals = [s for s in signals if 'fundamental' in s.get('signal_type', '')]

            # Calculate component scores
            sentiment_score = np.mean([s.get('strength', 0) for s in sentiment_signals]) if sentiment_signals else 0
            technical_score = np.mean([s.get('strength', 0) for s in technical_signals]) if technical_signals else 0
            fundamental_score = np.mean([s.get('strength', 0) for s in fundamental_signals]) if fundamental_signals else 0

            # Combined score with weights
            combined_score = (sentiment_score * 0.4 + technical_score * 0.4 + fundamental_score * 0.2)

            # Determine overall signal type
            if combined_score > 0.6:
                signal_type = 'strong_bullish'
            elif combined_score > 0.3:
                signal_type = 'moderate_bullish'
            elif combined_score < -0.6:
                signal_type = 'strong_bearish'
            elif combined_score < -0.3:
                signal_type = 'moderate_bearish'
            else:
                signal_type = 'neutral'

            # Risk assessment
            confidence_scores = [s.get('confidence', 0) for s in signals]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

            if avg_confidence > 0.8 and len(signals) > 2:
                risk_level = 'low'
            elif avg_confidence > 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'

            # Sources tracking
            sources = list(set([s.get('source', 'unknown') for s in signals]))

            enhanced_signals.append(EnhancedArbitrageSignal(
                ticker=ticker,
                signal_type=signal_type,
                strength=abs(combined_score),
                confidence=avg_confidence,
                sources=sources,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                combined_score=combined_score,
                risk_level=risk_level,
                timestamp=datetime.now(),
                metadata={
                    'signal_count': len(signals),
                    'sentiment_signals': len(sentiment_signals),
                    'technical_signals': len(technical_signals),
                    'fundamental_signals': len(fundamental_signals),
                    'squeeze_probability': wsb_analysis.squeeze_signals.get('squeeze_probability', 0)
                }
            ))

        return enhanced_signals

    def detect_market_manipulation_signals(self, wsb_analysis: WSBAnalysisResult) -> List[Dict[str, Any]]:
        """Detect potential market manipulation signals from WSB data"""
        manipulation_signals = []

        # Check for GME-style squeeze patterns
        squeeze_prob = wsb_analysis.squeeze_signals.get('squeeze_probability', 0)
        if squeeze_prob > 0.7:
            manipulation_signals.append({
                'type': 'short_squeeze_setup',
                'probability': squeeze_prob,
                'description': 'High probability of coordinated short squeeze',
                'indicators': ['accelerating_positive_sentiment', 'high_short_interest_discussion']
            })

        # Check for word association patterns
        associations = wsb_analysis.word_associations
        if 'short' in associations:
            short_assoc = associations['short']
            if short_assoc['sentiment_correlation'] < -0.5:
                manipulation_signals.append({
                    'type': 'negative_short_sentiment',
                    'correlation': short_assoc['sentiment_correlation'],
                    'description': 'Strong negative sentiment associated with short positions'
                })

        # Check for retail vs institutional narrative
        sentiment_summary = wsb_analysis.sentiment_summary
        if sentiment_summary.get('avg_wsb_score', 0) > 1.0:
            manipulation_signals.append({
                'type': 'retail_vs_institutional_narrative',
                'score': sentiment_summary['avg_wsb_score'],
                'description': 'Strong retail trader vs institutional narrative detected'
            })

        return manipulation_signals

    async def run_comprehensive_arbitrage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive arbitrage analysis combining all AAC data sources"""
        print("🚀 Running comprehensive AAC arbitrage analysis...")

        # Analyze WSB sentiment
        wsb_analysis = await self.analyze_current_wsb_sentiment()

        # Get WallStreetOdds signals (if configured)
        wallstreetodds_signals = []
        if self.wallstreetodds.api_key:
            try:
                # This would get real signals from WallStreetOdds
                wallstreetodds_signals = []  # Placeholder
            except Exception as e:
                print(f"⚠️  WallStreetOdds integration failed: {e}")

        # Enhance signals
        enhanced_signals = self.enhance_arbitrage_signals(wsb_analysis, wallstreetodds_signals)

        # Detect manipulation signals
        manipulation_signals = self.detect_market_manipulation_signals(wsb_analysis)

        # Generate final report
        report = {
            'timestamp': datetime.now(),
            'wsb_sentiment_summary': wsb_analysis.sentiment_summary,
            'enhanced_arbitrage_signals': [self._signal_to_dict(s) for s in enhanced_signals],
            'manipulation_signals': manipulation_signals,
            'word_associations': wsb_analysis.word_associations,
            'squeeze_analysis': wsb_analysis.squeeze_signals,
            'confidence_score': wsb_analysis.confidence_score,
            'data_sources': ['wallstreetbets_sentiment', 'wallstreetodds'] if self.wallstreetodds.api_key else ['wallstreetbets_sentiment']
        }

        return report

    def _signal_to_dict(self, signal: EnhancedArbitrageSignal) -> Dict[str, Any]:
        """Convert EnhancedArbitrageSignal to dictionary"""
        return {
            'ticker': signal.ticker,
            'signal_type': signal.signal_type,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'sources': signal.sources,
            'sentiment_score': signal.sentiment_score,
            'technical_score': signal.technical_score,
            'fundamental_score': signal.fundamental_score,
            'combined_score': signal.combined_score,
            'risk_level': signal.risk_level,
            'timestamp': signal.timestamp.isoformat(),
            'metadata': signal.metadata
        }


async def demo_comprehensive_wsb_integration():
    """Demo the comprehensive WSB integration with AAC arbitrage system"""
    print("🎯 AAC WallStreetBets Integration Demo")
    print("=" * 50)

    hub = AACWSBIntegrationHub()

    # Run comprehensive analysis
    analysis_report = await hub.run_comprehensive_arbitrage_analysis()

    # Display results
    print("
📊 Analysis Summary:"    print(f"   Timestamp: {analysis_report['timestamp']}")
    print(".2f"    print(f"   Data Sources: {', '.join(analysis_report['data_sources'])}")

    sentiment = analysis_report['wsb_sentiment_summary']
    if sentiment:
        print("
💬 Sentiment Summary:"        print(f"   Total Posts Analyzed: {sentiment.get('total_posts', 0)}")
        print(".3f"        print(".3f"        print(f"   Sentiment Distribution: {sentiment.get('sentiment_distribution', {})}")

    signals = analysis_report['enhanced_arbitrage_signals']
    if signals:
        print("
📈 Enhanced Arbitrage Signals:"        for signal in signals[:5]:  # Show top 5
            print(f"   🎯 {signal['ticker']}: {signal['signal_type']} "
                  ".2f"
                  f" (Risk: {signal['risk_level']})")

    manipulation = analysis_report['manipulation_signals']
    if manipulation:
        print("
🚨 Market Manipulation Signals:"        for signal in manipulation:
            print(f"   ⚠️  {signal['type']}: {signal['description']}")

    squeeze = analysis_report['squeeze_analysis']
    if squeeze and squeeze.get('squeeze_probability', 0) > 0:
        print("
🧹 Short Squeeze Analysis:"        print(".2f"        print(".3f"        print(".3f"
    print("
✅ Comprehensive WSB integration analysis complete!"    print("This demonstrates how GME-style sentiment analysis enhances AAC arbitrage detection.")


if __name__ == "__main__":
    asyncio.run(demo_comprehensive_wsb_integration())
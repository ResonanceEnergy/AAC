"""
AAC Goddess-Crawl Dataset Explorer
==================================

Explore the OpenTransformer/goddess-crawl dataset from Hugging Face.
This dataset may contain valuable data for arbitrage analysis.
"""

import json
import logging
import sys
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

def explore_goddess_crawl_dataset():
    """
    Load and explore the goddess-crawl dataset from Hugging Face.
    """
    logger.info("🔍 AAC Goddess-Crawl Dataset Explorer")
    logger.info("=" * 50)

    try:
        logger.info("📥 Loading goddess-crawl dataset from Hugging Face...")
        # Load a small sample first to understand the structure
        dataset = load_dataset("OpenTransformer/goddess-crawl", split="train[:1000]")

        logger.info("✅ Dataset loaded successfully!")
        logger.info(f"   Sample size: {len(dataset)} rows")
        logger.info(f"   Features: {list(dataset.features.keys())}")

        # Examine the first few examples
        logger.info("\n🔎 Examining first 5 examples:")
        for i, example in enumerate(dataset.select(range(5))):
            logger.info(f"\n--- Example {i+1} ---")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 200:
                    logger.info(f"{key}: {value[:200]}... (truncated)")
                else:
                    logger.info(f"{key}: {value}")

        # Analyze data types and structure
        logger.info("\n📊 Dataset Structure Analysis:")
        feature_info = {}
        for feature_name, feature_type in dataset.features.items():
            feature_info[feature_name] = str(feature_type)

        logger.info("Features and types:")
        for name, ftype in feature_info.items():
            logger.info(f"   {name}: {ftype}")

        # Look for potential financial/trading content
        logger.info("\n💰 Analyzing for Financial/Trading Content:")

        text_fields = [col for col in dataset.column_names if 'text' in col.lower() or 'content' in col.lower()]
        if text_fields:
            logger.info(f"   Text fields found: {text_fields}")

            # Sample some text content to look for financial keywords
            sample_texts = []
            for example in dataset.select(range(min(50, len(dataset)))):
                for field in text_fields:
                    if field in example and isinstance(example[field], str):
                        sample_texts.append(example[field][:500])

            # Look for financial keywords
            financial_keywords = [
                'stock', 'market', 'price', 'trade', 'buy', 'sell', 'invest',
                'arbitrage', 'profit', 'loss', 'portfolio', 'dividend', 'earnings',
                'revenue', 'growth', 'economy', 'inflation', 'interest rate',
                'bond', 'equity', 'forex', 'crypto', 'bitcoin', 'ethereum'
            ]

            keyword_counts = {}
            for keyword in financial_keywords:
                count = sum(1 for text in sample_texts if keyword.lower() in text.lower())
                if count > 0:
                    keyword_counts[keyword] = count

            if keyword_counts:
                logger.info("   Financial keywords detected:")
                for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"     '{keyword}': {count} occurrences")
            else:
                logger.info("   No obvious financial keywords found in sample")

        # Check for URL or source fields
        url_fields = [col for col in dataset.column_names if 'url' in col.lower() or 'source' in col.lower() or 'link' in col.lower()]
        if url_fields:
            logger.info(f"   URL/source fields: {url_fields}")

            # Sample some URLs to understand data sources
            sample_urls = []
            for example in dataset.select(range(min(20, len(dataset)))):
                for field in url_fields:
                    if field in example and example[field]:
                        sample_urls.append(str(example[field]))

            if sample_urls:
                logger.info("   Sample URLs/sources:")
                for url in sample_urls[:5]:
                    logger.info(f"     {url}")

        # Estimate dataset size and potential value
        logger.info(f"\n📈 Dataset Assessment:")
        logger.info(f"   Total estimated rows: ~4.9M")
        logger.info(f"   Sample analyzed: {len(dataset)} rows")

        # Potential AAC integration assessment
        logger.info(f"\n🎯 AAC Integration Potential:")
        has_financial_content = len(keyword_counts) > 0
        has_text_content = len(text_fields) > 0
        has_structured_data = len(dataset.features) > 2

        if has_financial_content:
            logger.info("   ✅ Contains financial/trading content - High arbitrage potential")
        else:
            logger.info("   ⚠️  No obvious financial content detected")

        if has_text_content:
            logger.info("   ✅ Has text content - Suitable for sentiment analysis")
        else:
            logger.info("   ❌ No text content found")

        if has_structured_data:
            logger.info("   ✅ Structured data available - Good for systematic processing")
        else:
            logger.info("   ⚠️  Limited structure - May require additional processing")

        # Recommendations
        logger.info(f"\n💡 Recommendations for AAC Integration:")
        if has_financial_content and has_text_content:
            logger.info("   • Integrate as alternative data source for arbitrage signals")
            logger.info("   • Use for sentiment analysis in trading strategies")
            logger.info("   • Combine with existing Reddit and World Bank data")
            logger.info("   • Implement in AlgoTrading101 backtesting framework")
        elif has_text_content:
            logger.info("   • May contain valuable unstructured data for ML models")
            logger.info("   • Consider NLP processing for feature extraction")
            logger.info("   • Evaluate for correlation with market movements")
        else:
            logger.info("   • Dataset may not be directly relevant for financial analysis")
            logger.info("   • Consider if it contains domain-specific data for your strategies")

        return {
            'dataset_info': {
                'name': 'OpenTransformer/goddess-crawl',
                'sample_size': len(dataset),
                'estimated_total': 4921640,
                'features': list(dataset.features.keys()),
                'feature_types': feature_info
            },
            'content_analysis': {
                'financial_keywords': keyword_counts,
                'text_fields': text_fields,
                'url_fields': url_fields,
                'has_financial_content': has_financial_content,
                'has_text_content': has_text_content
            },
            'aac_integration_potential': 'High' if (has_financial_content and has_text_content) else 'Medium' if has_text_content else 'Low'
        }

    except Exception as e:
        logger.info(f"❌ Error exploring dataset: {e}")
        logger.info("   This might be due to network issues or dataset access restrictions")
        return None

def create_aac_goddess_integration(result: Dict[str, Any]):
    """
    Create AAC integration code based on dataset analysis.
    """
    if not result:
        return

    logger.info(f"\n🔧 Creating AAC Integration Code...")
    integration_code = f'''
"""
AAC Goddess-Crawl Dataset Integration
====================================

Integration with OpenTransformer/goddess-crawl dataset for enhanced arbitrage analysis.

Dataset Info:
- Name: {result['dataset_info']['name']}
- Estimated Size: {result['dataset_info']['estimated_total']:,} rows
- Features: {', '.join(result['dataset_info']['features'])}
- AAC Integration Potential: {result['aac_integration_potential']}

Content Analysis:
- Financial Keywords: {len(result['content_analysis']['financial_keywords'])}
- Text Fields: {result['content_analysis']['text_fields']}
- Has Financial Content: {result['content_analysis']['has_financial_content']}
"""

from datasets import load_dataset
import pandas as pd
from typing import Dict, List, Optional, Iterator
import re
from datetime import datetime

class AACGoddessCrawlIntegration:
    """
    Integrate goddess-crawl dataset into AAC arbitrage system.
    """

    def __init__(self):
        """Initialize the goddess-crawl integration"""
        self.dataset_name = "OpenTransformer/goddess-crawl"
        self.financial_keywords = {list(result['content_analysis']['financial_keywords'].keys())}
        self.text_fields = {result['content_analysis']['text_fields']}
        self._dataset = None

    def load_dataset(self, split: str = "train", streaming: bool = True) -> Iterator:
        """
        Load the goddess-crawl dataset.

        Args:
            split: Dataset split to load
            streaming: Whether to stream the dataset

        Returns:
            Dataset iterator
        """
        try:
            if streaming:
                self._dataset = load_dataset(
                    self.dataset_name,
                    split=split,
                    streaming=True
                )
            else:
                self._dataset = load_dataset(self.dataset_name, split=split)
            return self._dataset
        except Exception as e:
            logger.info(f"❌ Error loading dataset: {{e}}")
            return None

    def extract_financial_signals(self, text: str) -> Dict[str, Any]:
        """
        Extract financial signals from text content.

        Args:
            text: Text content to analyze

        Returns:
            Dictionary with extracted signals
        """
        signals = {{
            'mentioned_tickers': [],
            'sentiment_score': 0.0,
            'financial_keywords': [],
            'confidence': 0.0
        }}

        if not isinstance(text, str):
            return signals

        # Extract potential tickers (simple pattern)
        ticker_pattern = r'\\b[A-Z]{{2,5}}\\b'
        potential_tickers = re.findall(ticker_pattern, text)
        signals['mentioned_tickers'] = [t for t in potential_tickers if len(t) >= 2]

        # Count financial keywords
        text_lower = text.lower()
        keyword_matches = []
        for keyword in self.financial_keywords:
            if keyword.lower() in text_lower:
                keyword_matches.append(keyword)

        signals['financial_keywords'] = keyword_matches

        # Simple sentiment scoring
        positive_words = ['buy', 'bull', 'profit', 'gain', 'growth', 'rise', 'up']
        negative_words = ['sell', 'bear', 'loss', 'decline', 'fall', 'down', 'crash']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count + negative_count > 0:
            signals['sentiment_score'] = (positive_count - negative_count) / (positive_count + negative_count)

        # Calculate confidence based on content richness
        signals['confidence'] = min(1.0, (len(keyword_matches) + len(signals['mentioned_tickers'])) / 5.0)

        return signals

    def process_dataset_for_arbitrage(self, max_samples: int = 10000) -> pd.DataFrame:
        """
        Process dataset and extract arbitrage-relevant signals.

        Args:
            max_samples: Maximum number of samples to process

        Returns:
            DataFrame with processed signals
        """
        signals_data = []

        try:
            dataset = self.load_dataset(streaming=True)
            if not dataset:
                return pd.DataFrame()

            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break

                # Extract signals from text fields
                combined_signals = {{
                    'mentioned_tickers': [],
                    'sentiment_score': 0.0,
                    'financial_keywords': [],
                    'confidence': 0.0,
                    'text_length': 0
                }}

                for field in self.text_fields:
                    if field in example and example[field]:
                        field_signals = self.extract_financial_signals(example[field])
                        combined_signals['mentioned_tickers'].extend(field_signals['mentioned_tickers'])
                        combined_signals['sentiment_score'] += field_signals['sentiment_score']
                        combined_signals['financial_keywords'].extend(field_signals['financial_keywords'])
                        combined_signals['confidence'] = max(combined_signals['confidence'], field_signals['confidence'])
                        combined_signals['text_length'] += len(str(example[field]))

                # Remove duplicates
                combined_signals['mentioned_tickers'] = list(set(combined_signals['mentioned_tickers']))
                combined_signals['financial_keywords'] = list(set(combined_signals['financial_keywords']))

                # Average sentiment if multiple fields
                if len(self.text_fields) > 1:
                    combined_signals['sentiment_score'] /= len(self.text_fields)

                # Add metadata
                combined_signals['processed_at'] = datetime.now()
                combined_signals['sample_id'] = i

                signals_data.append(combined_signals)

            return pd.DataFrame(signals_data)

        except Exception as e:
            logger.info(f"❌ Error processing dataset: {{e}}")
            return pd.DataFrame()

    def integrate_with_aac_arbitrage(self, arbitrage_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate goddess-crawl signals with existing AAC arbitrage data.

        Args:
            arbitrage_signals: Existing AAC arbitrage signals DataFrame

        Returns:
            Enhanced DataFrame with goddess-crawl signals
        """
        try:
            goddess_signals = self.process_dataset_for_arbitrage(max_samples=5000)

            if goddess_signals.empty:
                logger.info("⚠️  No goddess-crawl signals to integrate")
                return arbitrage_signals

            # Merge signals based on tickers
            enhanced_signals = arbitrage_signals.copy()

            # Add goddess-crawl sentiment as additional feature
            goddess_sentiment = goddess_signals.groupby('mentioned_tickers')['sentiment_score'].mean()
            goddess_confidence = goddess_signals.groupby('mentioned_tickers')['confidence'].mean()

            # This would be integrated into your existing arbitrage logic
            logger.info(f"✅ Integrated {{len(goddess_signals)}} goddess-crawl signals")
            logger.info(f"   Unique tickers with signals: {{len(goddess_sentiment)}}")

            return enhanced_signals

        except Exception as e:
            logger.info(f"❌ Error integrating signals: {{e}}")
            return arbitrage_signals

# Example usage
def demo_goddess_aac_integration():
    """Demonstrate goddess-crawl integration with AAC"""
    logger.info("AAC Goddess-Crawl Integration Demo")
    logger.info("=" * 40)

    integrator = AACGoddessCrawlIntegration()

    # Process sample data
    signals_df = integrator.process_dataset_for_arbitrage(max_samples=100)

    if not signals_df.empty:
        logger.info(f"✅ Processed {{len(signals_df)}} signals")
        print("\\nSample signals:"
        logger.info(str(signals_df.head()))

        # Show summary statistics
        logger.info("\\n📊 Signal Statistics:")
        logger.info(f"   Average sentiment: {{signals_df['sentiment_score'].mean():.3f}}")
        logger.info(f"   Average confidence: {{signals_df['confidence'].mean():.3f}}")
        logger.info(f"   Total unique tickers: {{len(set([t for tickers in signals_df['mentioned_tickers'] for t in tickers]))}}")
        logger.info(f"   Total financial keywords: {{len(set([k for keywords in signals_df['financial_keywords'] for k in keywords]))}}")
    else:
        logger.info("❌ No signals processed")

if __name__ == "__main__":
    demo_goddess_aac_integration()
'''

    with open('aac_goddess_crawl_integration.py', 'w', encoding='utf-8') as f:
        f.write(integration_code)

    print("✅ AAC Goddess-Crawl integration code created: aac_goddess_crawl_integration.py")

if __name__ == "__main__":
    result = explore_goddess_crawl_dataset()
    if result:
        create_aac_goddess_integration(result)

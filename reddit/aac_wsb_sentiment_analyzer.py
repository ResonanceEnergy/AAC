#!/usr/bin/env python3
"""
AAC WallStreetBets Sentiment Analysis Integration
===============================================

Advanced sentiment analysis for WallStreetBets data inspired by the GameStop case study.
Integrates multiple sentiment lexicons and temporal analysis for arbitrage signal generation.

Based on: https://github.com/Mafer104/WallStreet-bets-
Original R analysis by Maria Fernanda Pernillo

Features:
- Multi-lexicon sentiment analysis (VADER, TextBlob, custom financial sentiment)
- Word association analysis (like "citron", "short", "retail")
- Temporal sentiment tracking
- GME-style squeeze detection
- Integration with AAC arbitrage signals

Lexicons Used:
- VADER: Social media sentiment
- TextBlob: General purpose sentiment
- Financial lexicon: Market-specific terms
- Custom WSB lexicon: Reddit-specific slang
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    text: str
    vader_compound: float
    vader_pos: float
    vader_neg: float
    vader_neu: float
    textblob_polarity: float
    textblob_subjectivity: float
    financial_score: float
    wsb_score: float
    overall_sentiment: str
    confidence: float
    timestamp: datetime

@dataclass
class WordAssociation:
    """Word association analysis result"""
    target_word: str
    associated_words: Dict[str, float]
    cooccurrence_count: int
    sentiment_correlation: float

class AACWSBSentimentAnalyzer:
    """
    Advanced WallStreetBets sentiment analyzer inspired by GME case study.

    Implements multi-lexicon sentiment analysis and word association mining.
    """

    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.vader = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()

        # Financial sentiment lexicon (positive/negative market terms)
        self.financial_lexicon = {
            # Positive terms
            'bullish': 2.0, 'moon': 3.0, 'tendies': 2.5, 'diamond': 2.0, 'hands': 1.5,
            'hodl': 1.0, 'buy': 1.5, 'long': 1.5, 'calls': 1.5, 'yolo': 2.0,
            'squeeze': 2.5, 'shortsqueeze': 3.0, 'rocket': 2.5, 'to the moon': 3.0,
            'green': 1.0, 'up': 1.0, 'higher': 1.0, 'gains': 2.0, 'profit': 2.0,

            # Negative terms
            'bearish': -2.0, 'short': -1.5, 'puts': -1.5, 'sell': -1.5, 'crash': -2.5,
            'dump': -2.0, 'red': -1.0, 'down': -1.0, 'lower': -1.0, 'losses': -2.0,
            'bagholder': -2.0, 'rekt': -2.5, 'paper hands': -1.5, 'citron': -2.0,
            'melvin': -2.0, 'hedge': -1.0, 'gamma': -0.5, 'theta': -0.5
        }

        # WallStreetBets specific lexicon
        self.wsb_lexicon = {
            'stonks': 1.5, 'stonk': 1.5, 'tendies': 2.5, 'diamond hands': 2.0,
            'paper hands': -2.0, 'autist': 0.5, 'autists': 0.5, 'retard': -1.0,
            'fucking': -0.5, 'shit': -1.0, 'bullshit': -1.5, 'retail': 0.5,
            'institutional': -0.5, 'whale': 1.0, 'bagholder': -2.0, 'clown': -1.0
        }

        # Stop words
        self.stop_words = set(stopwords.words('english'))
        self.custom_stops = {'just', 'like', 'shit', 'fucking', 'fuck', 'make', 'big',
                           'put', 'gamestop', 'gme', 'thing', 'made', 'wsb', 'would',
                           'get', 'one', 'also', 'even', 'go', 'us', 'will', 'see'}

        print("✅ AAC WallStreetBets Sentiment Analyzer initialized")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words and token not in self.custom_stops
                 and len(token) > 2]

        return ' '.join(tokens)

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Perform comprehensive sentiment analysis"""
        if not isinstance(text, str) or not text.strip():
            return None

        # Preprocess text
        clean_text = self.preprocess_text(text)

        # VADER analysis
        vader_scores = self.vader.polarity_scores(clean_text)

        # TextBlob analysis
        blob = TextBlob(clean_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        # Financial lexicon scoring
        financial_score = self._calculate_lexicon_score(clean_text, self.financial_lexicon)

        # WSB lexicon scoring
        wsb_score = self._calculate_lexicon_score(clean_text, self.wsb_lexicon)

        # Overall sentiment determination
        scores = [vader_scores['compound'], textblob_polarity, financial_score, wsb_score]
        avg_score = np.mean(scores)

        if avg_score > 0.1:
            overall_sentiment = 'positive'
        elif avg_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'

        # Confidence based on agreement between lexicons
        confidence = 1 - np.std(scores) / 2  # Normalize to 0-1

        return SentimentResult(
            text=clean_text,
            vader_compound=vader_scores['compound'],
            vader_pos=vader_scores['pos'],
            vader_neg=vader_scores['neg'],
            vader_neu=vader_scores['neu'],
            textblob_polarity=textblob_polarity,
            textblob_subjectivity=textblob_subjectivity,
            financial_score=financial_score,
            wsb_score=wsb_score,
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            timestamp=datetime.now()
        )

    def _calculate_lexicon_score(self, text: str, lexicon: Dict[str, float]) -> float:
        """Calculate sentiment score using a custom lexicon"""
        words = text.split()
        score = 0
        matches = 0

        for word in words:
            if word in lexicon:
                score += lexicon[word]
                matches += 1

        return score / max(matches, 1)  # Avoid division by zero

    def analyze_word_associations(self, texts: List[str], target_word: str,
                                min_cooccurrence: int = 5) -> WordAssociation:
        """Analyze word associations similar to the GME case study"""
        target_word = target_word.lower()
        cooccurrences = defaultdict(int)
        target_mentions = 0

        for text in texts:
            if not isinstance(text, str):
                continue

            words = self.preprocess_text(text).split()

            if target_word in words:
                target_mentions += 1
                # Count co-occurring words
                for word in words:
                    if word != target_word and len(word) > 2:
                        cooccurrences[word] += 1

        # Filter by minimum cooccurrence
        filtered_assoc = {word: count for word, count in cooccurrences.items()
                         if count >= min_cooccurrence}

        # Calculate sentiment correlation for associated words
        sentiment_correlations = {}
        for word, count in filtered_assoc.items():
            if word in self.financial_lexicon:
                sentiment_correlations[word] = self.financial_lexicon[word]
            else:
                # Use VADER for unknown words
                vader_score = self.vader.polarity_scores(word)['compound']
                sentiment_correlations[word] = vader_score

        # Average sentiment correlation
        avg_sentiment_corr = np.mean(list(sentiment_correlations.values())) if sentiment_correlations else 0

        return WordAssociation(
            target_word=target_word,
            associated_words=filtered_assoc,
            cooccurrence_count=target_mentions,
            sentiment_correlation=avg_sentiment_corr
        )

    def analyze_temporal_sentiment(self, texts: List[str], dates: List[datetime],
                                 window_days: int = 7) -> pd.DataFrame:
        """Analyze sentiment over time (like the GME temporal analysis)"""
        if len(texts) != len(dates):
            raise ValueError("Texts and dates must have the same length")

        # Analyze sentiment for each text
        sentiments = []
        for text in texts:
            result = self.analyze_sentiment(text)
            if result:
                sentiments.append(result)

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates[:len(sentiments)],
            'vader_compound': [s.vader_compound for s in sentiments],
            'textblob_polarity': [s.textblob_polarity for s in sentiments],
            'financial_score': [s.financial_score for s in sentiments],
            'wsb_score': [s.wsb_score for s in sentiments],
            'overall_sentiment': [s.overall_sentiment for s in sentiments],
            'confidence': [s.confidence for s in sentiments]
        })

        # Resample to daily averages
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Rolling averages
        df['sentiment_ma'] = df['vader_compound'].rolling(window=window_days).mean()
        df['sentiment_trend'] = df['vader_compound'].diff()

        return df

    def detect_gme_style_squeeze(self, sentiment_data: pd.DataFrame,
                               price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Detect GME-style short squeeze patterns in sentiment data"""
        signals = {
            'squeeze_probability': 0.0,
            'sentiment_momentum': 0.0,
            'retail_vs_institutional': 0.0,
            'short_interest_indicators': [],
            'temporal_acceleration': 0.0
        }

        if sentiment_data.empty:
            return signals

        # Calculate sentiment momentum (rate of change)
        sentiment_momentum = sentiment_data['vader_compound'].diff().mean()
        signals['sentiment_momentum'] = sentiment_momentum

        # Look for accelerating positive sentiment (squeeze indicator)
        recent_sentiment = sentiment_data['vader_compound'].tail(10)
        if len(recent_sentiment) >= 5:
            acceleration = np.polyfit(range(len(recent_sentiment)), recent_sentiment, 1)[0]
            signals['temporal_acceleration'] = acceleration

            # High acceleration + positive momentum = squeeze signal
            if acceleration > 0.01 and sentiment_momentum > 0.05:
                signals['squeeze_probability'] = min(0.9, acceleration * sentiment_momentum * 100)

        # Analyze word patterns that indicate squeeze
        # This would be enhanced with actual text analysis

        return signals

    def generate_arbitrage_signals(self, sentiment_results: List[SentimentResult],
                                 price_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict[str, Any]]:
        """Generate arbitrage signals based on sentiment analysis"""
        signals = []

        if not sentiment_results:
            return signals

        # Aggregate sentiment by ticker mentions
        ticker_sentiment = defaultdict(list)

        for result in sentiment_results:
            # Extract ticker mentions from text
            tickers = self._extract_tickers(result.text)
            for ticker in tickers:
                ticker_sentiment[ticker].append(result)

        # Generate signals for each ticker
        for ticker, sentiments in ticker_sentiment.items():
            if len(sentiments) < 3:  # Need minimum data points
                continue

            # Calculate aggregate metrics
            avg_compound = np.mean([s.vader_compound for s in sentiments])
            avg_confidence = np.mean([s.confidence for s in sentiments])
            sentiment_trend = np.mean([s.vader_compound for s in sentiments[-5:]]) - \
                            np.mean([s.vader_compound for s in sentiments[:5]])

            # Determine signal strength
            if avg_compound > 0.2 and sentiment_trend > 0.1:
                signal_type = 'bullish_sentiment'
                strength = min(1.0, (avg_compound + sentiment_trend) / 2)
            elif avg_compound < -0.2 and sentiment_trend < -0.1:
                signal_type = 'bearish_sentiment'
                strength = min(1.0, abs(avg_compound + sentiment_trend) / 2)
            else:
                signal_type = 'neutral_sentiment'
                strength = 0.5

            signals.append({
                'ticker': ticker,
                'signal_type': signal_type,
                'strength': strength,
                'confidence': avg_confidence,
                'sentiment_score': avg_compound,
                'sentiment_trend': sentiment_trend,
                'data_points': len(sentiments),
                'source': 'wallstreetbets_sentiment',
                'timestamp': datetime.now()
            })

        return signals

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Simple regex for ticker extraction (can be enhanced)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)

        # Filter common tickers (this would be enhanced with a proper ticker list)
        common_tickers = {'AAPL', 'TSLA', 'GME', 'AMC', 'NOK', 'BB', 'PLTR', 'NVDA',
                         'AMD', 'SPY', 'QQQ', 'IWM', 'VXX', 'UVXY'}

        return [ticker for ticker in potential_tickers if ticker in common_tickers]

    def visualize_sentiment_analysis(self, sentiment_data: pd.DataFrame,
                                   save_path: Optional[str] = None):
        """Create visualizations similar to the GME case study"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('WallStreetBets Sentiment Analysis - GME Style', fontsize=16)

        # Sentiment over time
        axes[0, 0].plot(sentiment_data.index, sentiment_data['vader_compound'],
                       label='VADER Compound', color='blue')
        axes[0, 0].plot(sentiment_data.index, sentiment_data['sentiment_ma'],
                       label='7-day MA', color='red', linewidth=2)
        axes[0, 0].set_title('Sentiment Timeline')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Sentiment distribution
        sentiment_data['vader_compound'].hist(bins=30, ax=axes[0, 1], color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Sentiment Distribution')
        axes[0, 1].set_xlabel('Sentiment Score')
        axes[0, 1].set_ylabel('Frequency')

        # Confidence over time
        axes[1, 0].plot(sentiment_data.index, sentiment_data['confidence'],
                       color='green', label='Confidence')
        axes[1, 0].set_title('Analysis Confidence')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].grid(True, alpha=0.3)

        # Sentiment vs Subjectivity
        axes[1, 1].scatter(sentiment_data['vader_compound'], sentiment_data['confidence'],
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('Sentiment vs Confidence')
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Confidence Score')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Sentiment analysis visualization saved to {save_path}")

        plt.show()


def demo_gme_sentiment_analysis():
    """Demo the sentiment analysis with sample GME-related texts"""
    analyzer = AACWSBSentimentAnalyzer()

    # Sample WallStreetBets posts inspired by GME saga
    sample_posts = [
        "GME to the moon! Diamond hands forever! Short squeeze incoming!",
        "Citron is a criminal. Melvin capital getting rekt. Retail traders unite!",
        "Just bought more GME calls. This is the squeeze we've been waiting for.",
        "Paper hands selling? More shares for me! TENDIES INCOMING",
        "Short interest is 100%. This squeeze will be legendary.",
        "GME mentions exploding on WSB. The wall is coming down.",
        "Bought GME at 5, holding through the pain. Diamond hands!",
        "Citron research is bullshit. GME fundamentals are strong.",
        "Retail traders vs Wall Street hedge funds. We will win!",
        "GME short squeeze could be the biggest in history."
    ]

    print("🔍 Analyzing GME-related WallStreetBets sentiment...")

    # Analyze each post
    results = []
    for post in sample_posts:
        result = analyzer.analyze_sentiment(post)
        if result:
            results.append(result)
            print(".2f"
                  ".2f")

    # Word association analysis (like the original study)
    print("\n🔗 Analyzing word associations with 'short'...")
    assoc_short = analyzer.analyze_word_associations(sample_posts, 'short')
    print(f"Target word: {assoc_short.target_word}")
    print(f"Co-occurrence count: {assoc_short.cooccurrence_count}")
    print(f"Sentiment correlation: {assoc_short.sentiment_correlation:.3f}")
    print("Top associated words:")
    for word, count in sorted(assoc_short.associated_words.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {word}: {count}")

    # Generate arbitrage signals
    print("\n📈 Generating arbitrage signals...")
    signals = analyzer.generate_arbitrage_signals(results)
    for signal in signals:
        print(f"🎯 {signal['ticker']}: {signal['signal_type']} "
              ".2f"
              ".2f")

    print("\n✅ GME-style sentiment analysis demo complete!")
    print("This analysis methodology can be applied to current market data for arbitrage signals.")


if __name__ == "__main__":
    demo_gme_sentiment_analysis()
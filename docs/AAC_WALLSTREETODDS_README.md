# AAC WallStreetOdds Integration

## Overview

The AAC WallStreetOdds Integration brings comprehensive financial market data and advanced analytics to the Accelerated Arbitrage Corp system. WallStreetOdds provides institutional-grade market data, technical indicators, sentiment analysis, and historical probability analysis for enhanced arbitrage signal generation.

## Key Features

### 🔴 Real-Time Market Data
- **Live Stock Prices**: Real-time quotes for NASDAQ, NYSE, and BATS exchanges
- **Live Crypto Prices**: 24/7 cryptocurrency pricing data
- **Extended Hours**: Pre-market and after-hours trading data
- **Volume Analysis**: Relative volume and trading activity metrics

### 📊 Technical Analysis
- **Moving Averages**: SMA 20, 50, 200 with deviation analysis
- **Price Levels**: Week/Month/Year/All-time highs and lows
- **Performance Metrics**: Percentage changes across multiple timeframes
- **Technical Indicators**: RSI, MACD, and custom calculations

### 🎲 Historical Odds Analysis
- **Probability Modeling**: Historical success rates for price movements
- **Time Horizons**: Same-day, one-day, and one-week odds analysis
- **Statistical Returns**: Average, maximum, and minimum historical returns
- **Confidence Intervals**: Statistical significance of price predictions

### 💬 Social Sentiment
- **StockTwits Integration**: Real-time social sentiment from traders
- **Watchers Count**: Social following and engagement metrics
- **Sentiment Momentum**: Changes in social sentiment over time
- **Community Analysis**: Trader sentiment and market psychology

### 📈 Analyst Research
- **Ratings Feed**: Latest analyst recommendations and price targets
- **Rating Changes**: Upgrades, downgrades, and initiations tracking
- **Price Target Analysis**: Average, median, and range of targets
- **Firm Coverage**: Analyst quality scores and coverage breadth

### 📰 News & Sentiment
- **Real-Time News**: Latest financial news with sentiment scoring
- **Source Analysis**: Publisher credibility and news impact
- **Sentiment Classification**: Positive, negative, and neutral scoring
- **Market Impact**: Historical price reactions to news events

## Installation & Setup

### Prerequisites
```bash
pip install requests pandas numpy python-dotenv
```

### API Key Configuration
1. **Sign up** at [WallStreetOdds](https://wallstreetodds.com/register-api/)
2. **Get your API key** from the dashboard
3. **Add to environment**:
```env
WALLSTREETODDS_API_KEY=your_api_key_here
```

### Free vs Premium Plans
- **Free Plan**: 1,000 API credits/month, basic endpoints
- **Premium Plans**: Higher limits, advanced features
- **Enterprise**: Custom pricing for high-volume users

## Usage Examples

### Basic Integration
```python
from aac_wallstreetodds_integration import AACWallStreetOddsIntegration

# Initialize
wso = AACWallStreetOddsIntegration()

# Get real-time prices
prices = wso.get_real_time_stock_prices(['AAPL', 'TSLA', 'NVDA'])
print(prices.head())
```

### Technical Analysis
```python
# Get technical indicators
tech_data = wso.get_technical_stock_data(['AAPL', 'MSFT'])
print(tech_data[['symbol', 'sma20', 'sma50', 'perOffSma20']])
```

### Arbitrage Signal Generation
```python
# Generate arbitrage signals
signals = wso.generate_arbitrage_signals(['AAPL', 'TSLA', 'NVDA'])
for signal in signals:
    print(f"{signal.symbol}: {signal.signal_type} (strength: {signal.strength:.2f})")
```

### Analyst Ratings Analysis
```python
# Get latest analyst ratings
ratings = wso.get_analyst_ratings(['AAPL', 'AMZN'], limit=20)
bullish_ratings = ratings[ratings['ratingStandardized'] == 'buy']
print(f"Bullish ratings: {len(bullish_ratings)}")
```

### News Sentiment Analysis
```python
# Get news with sentiment
news = wso.get_news_sentiment('AAPL', limit=10)
positive_news = news[news['sentiment'] == 'ps']
print(f"Positive news items: {len(positive_news)}")
```

## Arbitrage Signal Types

### Momentum Signals
- **Bullish Momentum**: Strong upward price deviation from moving averages
- **Bearish Momentum**: Strong downward price deviation from moving averages
- **Signal Strength**: Based on magnitude of SMA deviations

### Probability Signals
- **High Odds Bullish**: >70% historical probability of upside
- **High Odds Bearish**: <30% historical probability of upside
- **Confidence**: Based on statistical significance

### Sentiment Signals
- **Social Momentum**: Strong positive sentiment momentum on StockTwits
- **Community Size**: Weighted by number of watchers/followers
- **Trend Analysis**: Changes in sentiment over time

## Integration with AAC System

### Enhanced Arbitrage Detection
```
AAC Traditional Signals + WallStreetOdds Data = Enhanced Arbitrage Detection
    ↓                           ↓                           ↓
Economic Data +         Technical + Social +         Multi-Source
Reddit Sentiment        Indicators  Sentiment       Arbitrage Signals
```

### Signal Combination Logic
1. **Traditional Analysis**: World Bank economics + Reddit sentiment
2. **Technical Confirmation**: WallStreetOdds technical indicators
3. **Probability Validation**: Historical odds analysis
4. **Sentiment Confirmation**: StockTwits social sentiment
5. **Risk Assessment**: Analyst ratings and news sentiment

### Backtesting Integration
```python
# Integrate with AlgoTrading101 backtesting
from aac_algotrading101_hub import AACAlgoTrading101Hub

hub = AACAlgoTrading101Hub()
wso_signals = wso.generate_arbitrage_signals(symbols)

# Backtest combined signals
results = hub.analyze_arbitrage_strategy(
    "AAC + WallStreetOdds Enhanced",
    price_data,
    lambda prices, pos: wso_signals  # Use WSO signals
)
```

## API Endpoints & Data Formats

### Response Formats
- **JSON**: Full-featured with all metadata
- **JSON Lean**: Reduced size for performance
- **JSON Array**: Minimal size for high-frequency use
- **CSV**: For data analysis and export

### Rate Limiting
- **Free Plan**: 1,000 credits/month
- **Premium Plans**: 10,000+ credits/month
- **Credit Costs**: Vary by endpoint complexity
- **Caching**: Implement local caching to reduce API calls

## Performance Optimization

### Efficient Data Retrieval
```python
# Batch requests for multiple symbols
symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN']
batch_data = wso.get_real_time_stock_prices(symbols)

# Use minimal fields for faster responses
minimal_fields = ['symbol', 'price', 'percentChange']
fast_data = wso.get_real_time_stock_prices(symbols, fields=minimal_fields)
```

### Caching Strategies
```python
# Cache frequently accessed data
import time
cache = {}

def get_cached_data(symbol, max_age=300):  # 5 minutes
    if symbol not in cache or time.time() - cache[symbol]['timestamp'] > max_age:
        cache[symbol] = {
            'data': wso.get_real_time_stock_prices([symbol]),
            'timestamp': time.time()
        }
    return cache[symbol]['data']
```

## Risk Management

### Signal Validation
- **Multi-Source Confirmation**: Require signals from multiple data sources
- **Statistical Significance**: Only act on high-confidence signals
- **Market Hours Filtering**: Respect exchange trading hours
- **Position Sizing**: Risk-based position sizing based on signal strength

### Error Handling
```python
try:
    signals = wso.generate_arbitrage_signals(symbols)
except Exception as e:
    print(f"Signal generation failed: {e}")
    # Fallback to traditional signals
    signals = generate_traditional_signals()
```

## Demo & Testing

Run the integration demo:
```bash
python aac_wallstreetodds_integration.py
```

This will test all available endpoints and show configuration status.

## Troubleshooting

### Common Issues

**API Key Not Configured**
```
Error: WallStreetOdds API key not configured
```
- Add `WALLSTREETODDS_API_KEY` to your `.env` file
- Verify the key is active in your WallStreetOdds account

**Rate Limit Exceeded**
```
Error: API credit limit reached
```
- Upgrade to a premium plan
- Implement request throttling
- Cache frequently used data

**Invalid Symbol**
```
Error: Symbol not found
```
- Verify symbol format (AAPL, TSLA, etc.)
- Check if symbol is actively traded
- Use the search endpoint to validate symbols

**Network Timeout**
```
Error: Request timeout
```
- Increase timeout parameter
- Check internet connection
- Use JSON Array format for faster responses

## Support & Resources

- **WallStreetOdds API Docs**: https://wallstreetodds.com/api-documentation/
- **AAC Integration**: Contact AAC development team
- **Community Support**: WallStreetOdds Discord/Forum
- **Premium Support**: Available for enterprise customers

## License & Terms

This integration uses the WallStreetOdds API. Users must comply with WallStreetOdds Terms of Service and API usage policies. The AAC system enhances arbitrage capabilities through multiple data sources while maintaining compliance with financial regulations.

---

**Accelerated Arbitrage Corp** - Advanced Algorithmic Trading Solutions
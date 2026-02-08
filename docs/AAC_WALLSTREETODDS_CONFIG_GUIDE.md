# AAC WallStreetOdds API Configuration Guide

## Overview

This guide helps you configure the AAC WallStreetOdds Integration with the necessary API credentials to unlock comprehensive financial market data and advanced analytics for arbitrage signal generation.

## WallStreetOdds API Overview

WallStreetOdds provides institutional-grade financial data including:
- Real-time stock and crypto prices
- Advanced technical indicators
- Historical probability analysis ("Live Odds")
- StockTwits sentiment data
- Analyst ratings and price targets
- News with sentiment analysis
- Seasonality data and company profiles

## API Plans & Pricing

### Free Plan
- **API Credits**: 1,000 credits/month
- **Cost**: $0 (perfect for testing)
- **Rate Limits**: Standard throttling
- **Support**: Community forums

### Premium Plans
- **Basic Plan**: $49/month - 10,000 credits
- **Pro Plan**: $99/month - 25,000 credits
- **Enterprise**: Custom pricing - 100,000+ credits

### Credit Usage Examples
- Real-time stock price: 1 credit
- Technical indicators: 2 credits
- Historical data (100 days): 5 credits
- Analyst ratings feed: 3 credits per request

## Setup Instructions

### Step 1: Create WallStreetOdds Account
1. Visit https://wallstreetodds.com/register-api/
2. Fill out the registration form
3. Verify your email address
4. Complete account setup

### Step 2: Get API Key
1. Log into your WallStreetOdds account
2. Navigate to "API" section in the dashboard
3. Click "Generate API Key"
4. Copy the API key (keep it secure!)

### Step 3: Configure Environment
Add the API key to your `.env` file:

```env
# WallStreetOdds API
WALLSTREETODDS_API_KEY=your_wallstreetodds_api_key_here
```

### Step 4: Test Configuration
Run the demo to verify everything works:

```bash
python aac_wallstreetodds_integration.py
```

You should see "✅ Configured" messages instead of warnings.

## API Endpoints & Usage

### Authentication
All requests require:
- `apikey`: Your API key
- `format`: Response format (json, jsonlean, jsonarray, csv)
- `fields`: Comma-separated list of fields to return

### Response Formats
- **json**: Full response with metadata
- **jsonlean**: Reduced whitespace for efficiency
- **jsonarray**: Minimal size, fastest
- **csv**: For data export and analysis

### Rate Limiting
- **Free Plan**: 1,000 credits/month
- **Premium**: 10,000+ credits/month
- **Reset**: Monthly on your billing cycle
- **Monitoring**: Check usage in dashboard

## Integration Examples

### Basic Usage
```python
from aac_wallstreetodds_integration import AACWallStreetOddsIntegration

wso = AACWallStreetOddsIntegration()

# Get real-time prices
prices = wso.get_real_time_stock_prices(['AAPL', 'TSLA'])
print(prices.head())
```

### Advanced Arbitrage Signals
```python
# Generate comprehensive arbitrage signals
signals = wso.generate_arbitrage_signals(['AAPL', 'TSLA', 'NVDA'])

for signal in signals:
    print(f"{signal.symbol}: {signal.signal_type}")
    print(f"  Strength: {signal.strength:.2f}")
    print(f"  Confidence: {signal.confidence:.2f}")
```

### Multi-Source Data Integration
```python
# Combine with existing AAC data
from aac_arbitrage_execution_system import AACArbitrageEngine

# Get WallStreetOdds signals
wso_signals = wso.generate_arbitrage_signals(symbols)

# Integrate with AAC engine
engine = AACArbitrageEngine()
enhanced_signals = wso.integrate_with_aac_arbitrage(existing_signals)
```

## Data Sources Integration

### With AlgoTrading101 Backtesting
```python
from aac_algotrading101_hub import AACAlgoTrading101Hub

hub = AACAlgoTrading101Hub()

# Use WallStreetOdds signals in backtesting
results = hub.analyze_arbitrage_strategy(
    "AAC + WallStreetOdds Enhanced",
    price_data,
    lambda prices, pos: wso_signals
)
```

### With Reddit Sentiment Analysis
```python
from aac_reddit_web_scraper import AACRedditWebScraper

# Get Reddit sentiment
reddit_scraper = AACRedditWebScraper()
reddit_sentiment = reddit_scraper.get_market_sentiment()

# Combine with WallStreetOdds data
combined_signals = wso.combine_sentiment_sources(
    reddit_sentiment,
    wso.get_stocktwits_sentiment(symbols)
)
```

## Best Practices

### API Efficiency
```python
# Batch requests for multiple symbols
symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT']
batch_data = wso.get_real_time_stock_prices(symbols)

# Use minimal fields for speed
fields = ['symbol', 'price', 'percentChange']
fast_data = wso.get_real_time_stock_prices(symbols, fields=fields)

# Use jsonarray for maximum speed
wso._make_request('endpoint', {'format': 'jsonarray'})
```

### Caching Strategy
```python
import time

class CachedWallStreetOdds:
    def __init__(self, wso_integration, cache_ttl=300):
        self.wso = wso_integration
        self.cache = {}
        self.cache_ttl = cache_ttl

    def get_cached_prices(self, symbols):
        cache_key = f"prices_{','.join(symbols)}"
        now = time.time()

        if (cache_key not in self.cache or
            now - self.cache[cache_key]['timestamp'] > self.cache_ttl):
            self.cache[cache_key] = {
                'data': self.wso.get_real_time_stock_prices(symbols),
                'timestamp': now
            }

        return self.cache[cache_key]['data']
```

### Error Handling
```python
try:
    signals = wso.generate_arbitrage_signals(symbols)
except Exception as e:
    print(f"WallStreetOdds error: {e}")
    # Fallback to alternative data sources
    signals = generate_backup_signals()
```

## Signal Types & Interpretation

### Momentum Signals
- **Bullish**: Strong upward deviation from moving averages
- **Bearish**: Strong downward deviation from moving averages
- **Strength**: 0.0-1.0 based on deviation magnitude

### Probability Signals
- **High Odds Bullish**: >70% historical success rate
- **High Odds Bearish**: <30% historical success rate
- **Confidence**: Statistical significance level

### Sentiment Signals
- **Social Momentum**: StockTwits watcher growth
- **Community Size**: Weighted by engagement
- **Trend Strength**: Rate of change in sentiment

## Risk Management

### Signal Validation
- Require multiple confirmation sources
- Use confidence thresholds (>0.7 for action)
- Validate against market hours
- Implement position size limits

### Credit Management
- Monitor usage in WallStreetOdds dashboard
- Implement request throttling
- Cache frequently used data
- Upgrade plan as needed

## Troubleshooting

### Common Issues

**"API key not configured"**
- Check `.env` file exists
- Verify key name: `WALLSTREETODDS_API_KEY`
- Ensure no extra spaces or characters

**"Rate limit exceeded"**
- Check credit usage in dashboard
- Implement caching for repeated requests
- Upgrade to higher plan if needed

**"Invalid symbol"**
- Verify symbol format (AAPL, TSLA, etc.)
- Check if symbol is actively traded
- Use supported exchanges only

**Network timeouts**
- Increase timeout in requests
- Use jsonarray format for speed
- Check internet connection stability

### Support Resources
- **WallStreetOdds Docs**: https://wallstreetodds.com/api-documentation/
- **API Dashboard**: Monitor usage and credits
- **Community Forum**: Get help from other users
- **AAC Integration**: Contact development team

## Performance Benchmarks

### Response Times (Approximate)
- Real-time prices: 200-500ms
- Technical data: 300-700ms
- Historical data: 500-2000ms
- Analyst ratings: 400-800ms

### Credit Costs (Per Request)
- Basic price data: 1 credit
- Technical indicators: 2 credits
- Historical data (100 days): 5 credits
- Full analyst feed: 10 credits

## Security Considerations

- Store API keys securely (never in code)
- Use environment variables for configuration
- Rotate keys periodically
- Monitor API usage for unauthorized access
- Implement proper error handling

## Next Steps

Once configured, you can:

1. **Real-time Trading**: Use live price feeds for execution
2. **Signal Enhancement**: Combine with existing AAC data
3. **Backtesting**: Validate strategies with historical data
4. **Risk Management**: Use probability analysis for position sizing
5. **Sentiment Analysis**: Incorporate social and news sentiment

The WallStreetOdds integration significantly enhances AAC's arbitrage capabilities with institutional-grade market data and advanced analytics.

---

**Accelerated Arbitrage Corp** - Professional Algorithmic Trading Integration
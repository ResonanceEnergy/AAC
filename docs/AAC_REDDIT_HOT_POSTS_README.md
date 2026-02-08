# AAC Reddit Hot Posts Integration

## Overview

This module demonstrates how to integrate the user's provided PRAW code pattern into the AAC arbitrage system for WallStreetBets sentiment analysis.

## User's Original Code Pattern

```python
import praw

reddit = praw.Reddit(client_id=config.client_id,
                    client_secret=config.client_secret,
                    user_agent=config.user_agent,
                    username=config.username,
                    password=config.password)

subreddit = reddit.subreddit("wallstreetbets")
for submission in subreddit.hot():
    print("{} -- {}".format(submission.id, submission.title))
```

## AAC Integration Features

### 🔍 **Automatic Ticker Extraction**
- Extracts stock tickers from post titles using regex patterns
- Supports `$TICKER`, `TICKER$`, and standalone ticker formats
- Filters out common non-ticker words

### 📊 **Sentiment Analysis**
- Analyzes post engagement and mention frequency
- Generates confidence scores for arbitrage signals
- Integrates with AAC's multi-source arbitrage engine

### ⏰ **Market Timing**
- Automatic market hours detection (9:30 AM - 4:00 PM ET)
- Rate limit management (600 requests per 10 minutes)
- Optimized data collection during active trading periods

### 🚀 **Arbitrage Signal Generation**
- Combines Reddit sentiment with World Bank economic data
- Multi-factor analysis for enhanced signal accuracy
- Real-time signal generation based on retail interest

## Setup Instructions

### 1. Reddit API Credentials

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the required fields:
   - **Name**: AAC-Arbitrage-Bot
   - **App type**: script
   - **Description**: Arbitrage signal generation from Reddit sentiment
   - **About URL**: (leave blank)
   - **Redirect URI**: http://localhost:8080

5. Copy the **client_id** (under the app name) and **secret**

### 2. Environment Configuration

Add to your `.env` file:

```bash
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=AAC-Arbitrage-Bot/1.0
REDDIT_USERNAME=your_reddit_username  # Optional
REDDIT_PASSWORD=your_reddit_password  # Optional
```

### 3. Install Dependencies

```bash
pip install praw python-dotenv
```

## Usage Examples

### Basic Usage

```python
from aac_reddit_hot_posts_demo import AACRedditHotPostsDemo

# Initialize client
demo = AACRedditHotPostsDemo()

# Fetch hot posts
posts = demo.fetch_hot_posts(limit=10)

# Analyze tickers
ticker_analysis = demo.analyze_ticker_mentions(posts)

# Generate signals
signals = demo.generate_arbitrage_signals(ticker_analysis)
```

### Integration with AAC System

```python
from praw_reddit_integration import PRAWRedditClient, PRAWConfig
from aac_reddit_hot_posts_demo import AACRedditHotPostsDemo

# Using the full AAC integration
config = PRAWConfig(
    client_id="your_client_id",
    client_secret="your_client_secret",
    user_agent="AAC-Arbitrage-Bot/1.0",
    subreddit="wallstreetbets"
)

client = PRAWRedditClient(config)
posts = client.get_hot_posts(limit=25)

# Process with AAC arbitrage engine
# (Integration with main arbitrage system)
```

## Expected Output

When credentials are configured, the demo will show:

```
📊 Retrieved 15 hot posts

🔍 Analyzing ticker mentions...
Top mentioned tickers:
  $AAPL: 5 mentions
  $TSLA: 3 mentions
  $NVDA: 2 mentions

📈 Generating arbitrage signals...
Generated 2 arbitrage signals:
  🚀 AAPL: 50.0% confidence
     High retail interest in $AAPL with 5 mentions
```

## Rate Limits & Best Practices

- **Authenticated users**: 600 requests per 10 minutes
- **Non-authenticated**: 60 requests per hour
- **Best practice**: Collect data during market hours (9:30 AM - 4:00 PM ET)
- **Data freshness**: Hot posts change rapidly - poll every 5-15 minutes

## Integration Points

### With World Bank Data
```python
# Combine Reddit sentiment with economic indicators
economic_data = world_bank_client.get_gdp_data()
reddit_signals = reddit_client.generate_arbitrage_signals()

combined_signal = analyze_multi_source_data(economic_data, reddit_signals)
```

### With Multi-Source Arbitrage
```python
# Feed into main arbitrage engine
arbitrage_engine.add_sentiment_signals(reddit_signals)
arbitrage_engine.add_economic_signals(world_bank_signals)

opportunities = arbitrage_engine.find_arbitrage_opportunities()
```

## Files in This Integration

- `aac_reddit_hot_posts_demo.py` - Main demo with user's code pattern
- `praw_reddit_integration.py` - Full AAC Reddit integration
- `reddit_api_documentation.py` - API endpoint documentation
- `reddit_api_setup.py` - Setup and configuration guide

## Troubleshooting

### Common Issues

1. **"Reddit API credentials not found"**
   - Ensure `.env` file exists with correct variable names
   - Check that credentials are valid and not expired

2. **Rate limit exceeded**
   - Wait 10 minutes for authenticated users
   - Reduce polling frequency
   - Consider using multiple accounts (with caution)

3. **Authentication failed**
   - Verify client_id and client_secret are correct
   - Check user_agent format
   - Ensure Reddit account is in good standing

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Notes

- Never commit API credentials to version control
- Use environment variables for sensitive data
- Consider IP rotation for high-volume data collection
- Respect Reddit's Terms of Service and API usage policies

## Performance Optimization

- Cache frequently accessed data
- Use async processing for multiple data sources
- Implement exponential backoff for rate limits
- Monitor API usage and adjust polling intervals

## Next Steps

1. Set up Reddit API credentials
2. Test the demo with real data
3. Integrate with existing AAC arbitrage engine
4. Add sentiment analysis and signal filtering
5. Implement production monitoring and alerting
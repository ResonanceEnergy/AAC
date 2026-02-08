# AAC Multi-Subreddit Continuous Scraper

## Overview

The AAC Multi-Subreddit Continuous Scraper is a comprehensive 24/7 Reddit monitoring system designed to track arbitrage opportunities, market sentiment, and trading signals across 60+ specialized subreddits.

## Features

- **60+ Subreddits Monitored**: Comprehensive coverage of trading, arbitrage, crypto, and financial subreddits
- **15-Minute Intervals**: Continuous scraping with 15-minute cycles (24/7 operation)
- **Advanced Sentiment Analysis**: TextBlob-powered sentiment scoring for posts
- **Ticker Extraction**: Automatic identification of stock/crypto tickers in posts
- **Arbitrage Signal Detection**: Pattern recognition for arbitrage opportunities
- **Insight Generation**: Automated analysis and insight creation
- **Data Integration**: Seamless integration with AAC arbitrage systems
- **Comprehensive Logging**: Full audit trail and monitoring

## Monitored Subreddits

### Trading & Arbitrage
- r/algotrading, r/ArbitrageEd, r/ARBITRAGE, r/ArbitrageFBA, r/statarb
- r/cryptoarbitrage, r/arbitrageExpert, r/arbitrageCT, r/ArbitrageTrading
- r/ArbiSwap, r/CentArbitrage, r/SearchArbitrageAFD

### Quantitative Finance
- r/quantfinance, r/FiverrArbitrage, r/Valuation, r/investing_discussion
- r/Flipping, r/quant, r/AfriqArbitrage, r/SideHustleGold

### Crypto & DeFi
- r/bitcointrader, r/algorithmictrading, r/defi, r/InvestmentsTrading
- r/cryptocentralai, r/hogefinance, r/CryptoCurrency, r/CryptoMoonShots
- r/CryptoMarkets, r/CryptoQuantTrade, r/SocialArbitrageTradin
- r/CryptoHopper, r/p2pcryptoexchanges

### WallStreetBets Ecosystem
- r/wallstreetbets, r/wallstreetdd, r/WallStreetOasis, r/WallStreetSiren
- r/wallstreet_apes, r/WallstreetSluts, r/WallStreetNYSE, r/wallstreetbetsGER
- r/WallStreetbetsELITE, r/Wallstreetbetsnew, r/wallstreetbetsOGs
- r/WallStreetBetsCrypto, r/WallstreetBreakers, r/WallStreetVR
- r/wallstreetbets2, r/WallStreetBaggers, r/wallstreetInvestment
- r/wallstreetbets_wins, r/wallstreetsmallcaps, r/Superstonk
- r/WallStreetBetsTopMost, r/GME, r/GMEJungle, r/DDintoGME, r/gmeoptions

### Options & Derivatives
- r/options, r/thetagang, r/PriceActionTrading, r/resellprofits
- r/IndianStreetBets, r/swingtrading, r/TradeVol

### Traditional Finance
- r/stocks, r/InvestingandTrading, r/algobetting, r/valueinvestorsclub
- r/Economics, r/ZeroExchange, r/ETFInvesting, r/ValueInvesting
- r/btc, r/BitcoinMarkets, r/highfreqtrading, r/ethtrader

## Setup

### 1. Reddit API Credentials

You need to create a Reddit app to get API credentials:

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the required fields
5. Note down the `client_id` (under the app name) and `client_secret`

### 2. Environment Variables

Add the following to your `.env` file:

```bash
# Reddit API (PRAW)
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=AAC-Arbitrage-Bot/1.0
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
```

### 3. Dependencies

Install required packages:

```bash
pip install praw apscheduler textblob
```

## Usage

### Single Test Run

To test the scraper with a single cycle:

```bash
python aac_multi_subreddit_scraper.py --single
```

### Continuous Operation

To run the scraper continuously (15-minute intervals):

```bash
python aac_multi_subreddit_scraper.py
```

The scraper will run indefinitely, scraping all subreddits every 15 minutes.

## Data Output

The scraper generates three types of output files in `data/reddit_scrapes/`:

### 1. Posts Data (`posts_TIMESTAMP.json`)
Raw scraped post data for all subreddits:
```json
{
  "algotrading": [
    {
      "id": "abc123",
      "title": "Post title",
      "sentiment_score": 0.8,
      "tickers_mentioned": ["AAPL", "TSLA"],
      "arbitrage_signals": ["statistical_arbitrage"],
      "scraped_at": "2024-01-01T12:00:00"
    }
  ]
}
```

### 2. Metrics Data (`metrics_TIMESTAMP.csv`)
Aggregated metrics per subreddit:
```csv
subreddit,posts_scraped,total_score,avg_sentiment,top_tickers,arbitrage_signals,scraped_at
algotrading,25,15420,0.15,"{'AAPL': 5, 'TSLA': 3}",2,2024-01-01T12:00:00
```

### 3. Insights Data (`insights_TIMESTAMP.json`)
Generated arbitrage insights:
```json
[
  {
    "insight_id": "AAPL_bullish_sentiment_20240101_120000",
    "type": "ticker_sentiment",
    "confidence": 0.8,
    "description": "Bullish sentiment for $AAPL (avg: 0.75, mentions: 12)",
    "tickers": ["AAPL"],
    "subreddits": ["wallstreetbets", "stocks"],
    "generated_at": "2024-01-01T12:00:00"
  }
]
```

## Integration with AAC System

The scraper integrates with the broader AAC arbitrage system through:

1. **Data Storage**: All scraped data is stored in structured formats for analysis
2. **Insight Generation**: Automated insights feed into the arbitrage engine
3. **Sentiment Analysis**: Real-time sentiment data for trading decisions
4. **Signal Detection**: Arbitrage opportunities automatically flagged

## Configuration

### Subreddit List
Modify the `self.subreddits` list in the `__init__` method to add/remove subreddits.

### Scraping Parameters
- `limit`: Number of posts to scrape per subreddit (default: 25)
- `interval`: Scraping interval in minutes (default: 15)

### Sentiment Analysis
The scraper uses TextBlob for sentiment analysis. Keywords are categorized as:
- **Bullish**: moon, to the moon, bullish, buy, long, calls, yolo, diamond hands
- **Bearish**: sell, short, bearish, puts, crash, dump, paper hands, rekt

## Monitoring

### Logs
All activity is logged to `aac_reddit_scraper.log` with the following levels:
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Critical failures

### Health Checks
Monitor the log file for:
- Successful scrape cycles
- API rate limit warnings
- Network connectivity issues
- Data export confirmations

## Troubleshooting

### Common Issues

1. **Reddit API Credentials**
   - Ensure all credentials are set in `.env`
   - Verify app has correct permissions
   - Check Reddit account status

2. **Rate Limiting**
   - Reddit API has rate limits (600 requests/hour for authenticated users)
   - Scraper includes built-in delays to respect limits

3. **Network Issues**
   - Ensure stable internet connection
   - Check firewall/proxy settings

4. **Dependencies**
   - Verify all packages are installed
   - Update packages if needed

### Stopping the Scraper

To stop continuous scraping, use Ctrl+C. The scraper will shut down gracefully.

## Performance

- **Typical Cycle Time**: 2-5 minutes for all 60+ subreddits
- **Data Volume**: ~1500 posts per cycle
- **Storage**: ~50MB per day of compressed data
- **API Usage**: ~60 API calls per cycle

## Security

- Credentials stored securely in `.env` (not committed to version control)
- No sensitive data logged
- Rate limiting prevents API abuse
- Error handling prevents crashes

## Future Enhancements

- Machine learning sentiment analysis
- Real-time alerting for high-confidence signals
- Integration with additional social media platforms
- Advanced arbitrage pattern recognition
- Predictive modeling for market movements
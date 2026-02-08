# AAC Advanced Reddit Web Scraping Integration

## Overview

This module implements advanced WallStreetBets sentiment analysis using Selenium browser automation and Pushshift API, based on the tutorial from [AlgoTrading101](https://algotrading101.com/learn/reddit-wallstreetbets-web-scraping/).

## Key Features

### 🔍 **Advanced Web Scraping**
- **Selenium Automation**: Browser-based scraping to handle JavaScript-heavy Reddit pages
- **Pushshift API**: Bulk comment data retrieval (no rate limits like Reddit API)
- **Daily Discussion Targeting**: Automated discovery of daily discussion threads
- **Weekend Thread Support**: Handles both weekday and weekend discussion formats

### 📊 **Comprehensive Sentiment Analysis**
- **Stock Ticker Extraction**: Regex patterns for $TICKER, TICKER$, and standalone tickers
- **Sentiment Classification**: Bullish/bearish/neutral analysis based on keywords
- **Context Preservation**: Stores example contexts for each ticker mention
- **Frequency Analysis**: Counts and ranks ticker mentions across thousands of comments

### 🚀 **AAC Arbitrage Integration**
- **Multi-Source Signals**: Combines with World Bank economic data
- **Market Timing**: Automatic market hours detection for signal relevance
- **Large-Scale Processing**: Handles 60k+ comments efficiently using NumPy
- **Export Capabilities**: CSV and pandas DataFrame outputs for analysis

## Architecture

### Core Components

1. **`AACRedditWebScraper`**: Main scraping orchestrator
2. **`ScrapedRedditPost`**: Data structure for scraped thread data
3. **`StockMention`**: Detailed ticker mention analysis
4. **Pushshift Integration**: Bulk comment retrieval API
5. **Selenium Automation**: Browser-based thread discovery

### Data Flow

```
Thread Discovery → Comment ID Retrieval → Bulk Comment Fetch → Ticker Analysis → Signal Generation
     ↓                    ↓                      ↓                ↓              ↓
  Selenium Search    Pushshift API         Pushshift API    Regex Processing  AAC Engine
```

## Setup Requirements

### Dependencies

```bash
pip install selenium webdriver-manager pandas numpy requests python-dateutil
```

### ChromeDriver Setup

The scraper uses Chrome WebDriver. Ensure Chrome browser is installed and ChromeDriver is available in PATH.

### Environment Variables

No API keys required - uses free Pushshift API and browser automation.

## Usage Examples

### Basic Scraping

```python
from aac_reddit_web_scraper import AACRedditWebScraper

# Initialize scraper
scraper = AACRedditWebScraper(headless=True)

# Scrape yesterday's Daily Discussion
result = scraper.scrape_daily_discussion()

if result:
    print(f"Found {len(result.tickers_mentioned)} tickers")
    print(f"Overall sentiment: {result.sentiment_score:.2f}")
```

### Advanced Analysis

```python
# Get detailed ticker analysis
ticker_analysis = scraper.analyze_ticker_mentions(comments_data)

# Export to CSV
scraper.export_to_csv(ticker_analysis, "sentiment_analysis.csv")

# Convert to DataFrame for analysis
df = scraper.export_to_dataframe(ticker_analysis)
top_tickers = df.nlargest(10, 'mentions')
```

## Technical Implementation

### Thread Discovery Algorithm

1. **Search Navigation**: Uses Reddit's search with flair filter for Daily Discussions
2. **Date Matching**: Parses thread titles to find yesterday's discussion
3. **Weekend Handling**: Special logic for Saturday/Sunday thread ranges
4. **URL Extraction**: Retrieves thread IDs for API processing

### Comment Processing Pipeline

1. **ID Retrieval**: Pushshift API gets all comment IDs for a thread
2. **Batch Processing**: Comments fetched in 1000-comment batches
3. **Text Extraction**: Body text extracted from comment JSON
4. **Ticker Scanning**: Regex patterns identify stock mentions
5. **Sentiment Analysis**: Keyword-based bullish/bearish classification

### Performance Optimizations

- **NumPy Arrays**: Efficient handling of large comment datasets
- **Batch API Calls**: Respectful rate limiting with Pushshift
- **Memory Management**: Streaming processing for large threads
- **Error Resilience**: Graceful handling of API failures

## Integration with AAC System

### Arbitrage Signal Generation

```python
# Combine with existing AAC components
from aac_reddit_web_scraper import AACRedditWebScraper
from world_bank_arbitrage_integration import WorldBankIntegration

# Get sentiment signals
reddit_scraper = AACRedditWebScraper()
sentiment_data = reddit_scraper.scrape_daily_discussion()

# Get economic data
wb_integration = WorldBankIntegration()
economic_signals = wb_integration.generate_arbitrage_signals()

# Combine for multi-factor analysis
combined_signals = analyze_multi_source_data(sentiment_data, economic_signals)
```

### Market Hours Integration

```python
from aac_timestamp_converter import AACArbitrageTiming

# Only process signals during market hours
if AACArbitrageTiming.is_market_hours():
    signals = generate_arbitrage_signals(sentiment_data)
    execute_trades(signals)
```

## Output Formats

### CSV Export

```csv
ticker,mention_count,bullish,bearish,neutral,top_context
AAPL,25,15,5,5,"$AAPL to the moon! 🚀🚀🚀"
TSLA,18,12,3,3,"TSLA earnings beat expectations"
NVDA,12,8,2,2,"Why $NVDA is undervalued"
```

### DataFrame Analysis

```python
# Sentiment ratio calculation
df['sentiment_ratio'] = (df['bullish'] - df['bearish']) / df['mentions']

# Filter high-confidence signals
high_conf_signals = df[
    (df['mentions'] >= 10) &
    (df['sentiment_ratio'] > 0.5)
].sort_values('sentiment_ratio', ascending=False)
```

## Advantages over PRAW API

| Feature | PRAW API | Web Scraping |
|---------|----------|--------------|
| Rate Limits | 600/10min | No limits (Pushshift) |
| Data Volume | Limited | 60k+ comments |
| Historical Data | Recent only | Full thread history |
| Setup Complexity | Simple | Requires Selenium |
| Reliability | High | Dependent on website changes |
| Cost | Free | Free |

## Challenges & Solutions

### JavaScript-Heavy Pages
**Challenge**: Reddit uses JavaScript for dynamic content
**Solution**: Selenium browser automation simulates full page loading

### Large Dataset Processing
**Challenge**: Daily threads can have 60,000+ comments
**Solution**: NumPy arrays and batch processing for memory efficiency

### Thread Discovery
**Challenge**: Finding the correct Daily Discussion thread by date
**Solution**: Advanced date parsing and weekend thread handling

### Rate Limiting
**Challenge**: Respectful API usage while getting comprehensive data
**Solution**: Batch processing with delays and Pushshift's generous limits

## Monitoring & Maintenance

### Health Checks

```python
def check_scraper_health():
    """Monitor scraper functionality"""
    try:
        scraper = AACRedditWebScraper()
        # Test thread discovery
        thread_id = scraper.find_daily_discussion_thread()
        if thread_id:
            # Test comment retrieval
            comment_ids = scraper.get_comment_ids(thread_id)
            return len(comment_ids) > 0
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
```

### Error Handling

- **Network Issues**: Automatic retry with exponential backoff
- **API Changes**: Logging for manual intervention when APIs change
- **Browser Issues**: ChromeDriver version checking and updates
- **Data Quality**: Validation of scraped content integrity

## Future Enhancements

### Real-time Processing
- WebSocket connections for live comment streaming
- Real-time sentiment analysis during market hours

### Machine Learning Integration
- Sentiment classification using trained models
- Topic modeling for discussion themes
- Predictive analytics for mention trends

### Multi-Platform Expansion
- Twitter sentiment analysis
- StockTwits integration
- News article sentiment

### Advanced Analytics
- Time-series sentiment tracking
- Correlation analysis with price movements
- Network analysis of user influence

## Security & Ethics

### Responsible Scraping
- Respects robots.txt and website terms
- Uses appropriate delays between requests
- Limits concurrent connections
- Academic/research use case

### Data Privacy
- No personal user data collection
- Aggregated sentiment analysis only
- No individual comment storage without consent

## Performance Benchmarks

### Typical Processing Times

- **Thread Discovery**: 5-15 seconds
- **Comment ID Retrieval**: 2-5 seconds
- **Bulk Comment Fetching**: 30-120 seconds (60k comments)
- **Ticker Analysis**: 10-30 seconds
- **Total Pipeline**: 1-3 minutes per Daily Discussion

### Resource Usage

- **Memory**: 500MB-2GB for large threads
- **CPU**: Moderate usage during analysis phase
- **Network**: 50-200 API calls per scraping session
- **Storage**: Minimal (results exported to CSV/DataFrame)

## Troubleshooting

### Common Issues

1. **ChromeDriver Not Found**
   ```
   Solution: Install ChromeDriver and add to PATH
   Alternative: Use webdriver-manager for automatic handling
   ```

2. **Thread Not Found**
   ```
   Cause: Weekend threads have different naming
   Solution: Check date parsing logic for weekend ranges
   ```

3. **Pushshift API Issues**
   ```
   Cause: API rate limits or service changes
   Solution: Implement retry logic and monitor API status
   ```

4. **Memory Errors**
   ```
   Cause: Large threads with 60k+ comments
   Solution: Process in smaller batches or use streaming
   ```

## Conclusion

The AAC Advanced Reddit Web Scraping Integration provides comprehensive sentiment analysis capabilities that significantly enhance the arbitrage system's ability to detect market sentiment shifts. By combining Selenium automation with Pushshift API, it overcomes traditional API limitations while maintaining ethical scraping practices.

The integration seamlessly fits into the existing AAC multi-source arbitrage framework, providing valuable sentiment signals alongside economic indicators and traditional market data.
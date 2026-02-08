# AAC Premium Market Data APIs

This document explains how to enhance the AAC system with premium market data APIs for better performance, higher rate limits, and more reliable data feeds.

## 🎯 Why Use Premium APIs?

The AAC system currently works with free data sources (Yahoo Finance, crypto exchanges), but premium APIs offer:

- **Higher Rate Limits**: Free APIs have strict call limits (e.g., 25 calls/day on Alpha Vantage)
- **More Frequent Updates**: Real-time data with minimal delays
- **Better Reliability**: Dedicated infrastructure vs. free public APIs
- **Additional Features**: Historical data, fundamentals, news, technical indicators
- **Priority Support**: Direct support channels for issues

## 📊 Available Premium APIs

### 1. Polygon.io ⭐ **Recommended First Choice**
- **Free Tier**: 5 calls/minute, 5M calls/month
- **Premium**: $99/month (Stocks Starter) for 5M calls/month
- **Best For**: Excellent free tier, comprehensive stock data, great documentation
- **Setup**: Visit [polygon.io](https://polygon.io) → Sign up → Get API key → Add to `.env`

### 2. Finnhub
- **Free Tier**: 60 calls/minute, 150 calls/day
- **Premium**: $9.99/month (Basic) for 300 calls/minute
- **Best For**: Real-time quotes, company fundamentals, news sentiment
- **Setup**: Visit [finnhub.io](https://finnhub.io) → Sign up → Get token → Add to `.env`

### 3. IEX Cloud
- **Free Tier**: 50,000 calls/month
- **Premium**: $9/month (Launch) for 5M calls/month
- **Best For**: US stock data, 15+ years historical data, crypto
- **Setup**: Visit [iexcloud.io](https://iexcloud.io) → Sign up → Get token → Add to `.env`

### 4. Twelve Data
- **Free Tier**: 800 calls/day
- **Premium**: $9.99/month (Basic) for 800 calls/day
- **Best For**: Multi-asset (stocks, forex, crypto), WebSocket streaming
- **Setup**: Visit [twelvedata.com](https://twelvedata.com) → Sign up → Get key → Add to `.env`

### 5. Intrinio (Institutional)
- **Free Tier**: None (Premium only)
- **Premium**: $75/month (Starter) for 50K calls/month
- **Best For**: Institutional-grade data, options, alternative data
- **Setup**: Visit [intrinio.com](https://intrinio.com) → Sign up → Get credentials → Add to `.env`

### 6. Alpha Vantage
- **Free Tier**: 25 calls/day, 5 calls/minute
- **Premium**: $49.99/month for 75 calls/minute
- **Best For**: Technical indicators, sector data, forex/crypto
- **Setup**: Visit [alphavantage.co](https://www.alphavantage.co/support/#api-key) → Get free key → Add to `.env`

## 🚀 Quick Start Setup

1. **Choose Your APIs**: Start with 2-3 complementary APIs for redundancy
   - Primary: Polygon.io (excellent free tier)
   - Secondary: Finnhub or IEX Cloud
   - Tertiary: Twelve Data or Alpha Vantage

2. **Get API Keys**: Follow setup steps for each API above

3. **Configure Environment**:
   ```bash
   # Edit .env file and add your keys:
   POLYGON_API_KEY=your_polygon_key_here
   FINNHUB_API_KEY=your_finnhub_token_here
   IEX_CLOUD_API_KEY=your_iex_token_here
   ```

4. **Test Configuration**:
   ```bash
   python premium_api_guide.py  # Check status
   python test_premium_apis.py  # Test connections
   python strategy_execution_demo.py  # Run with premium data
   ```

## 📈 Performance Comparison

| API | Free Calls/Month | Premium | Best Features |
|-----|------------------|---------|---------------|
| Polygon.io | 5M | $99/mo | Real-time aggregates, options |
| Finnhub | ~150K | $10/mo | Fundamentals, news sentiment |
| IEX Cloud | 50K | $9/mo | 15+ years history, crypto |
| Twelve Data | ~24K | $10/mo | Multi-asset, WebSocket |
| Intrinio | 0 | $75/mo | Institutional quality |
| Alpha Vantage | ~750 | $50/mo | Technical indicators |

## 🔧 Configuration Details

### Environment Variables

Add these to your `.env` file:

```env
# Polygon.io
POLYGON_API_KEY=your_key_here

# Finnhub
FINNHUB_API_KEY=your_token_here

# IEX Cloud
IEX_CLOUD_API_KEY=your_token_here

# Twelve Data
TWELVE_DATA_API_KEY=your_key_here

# Intrinio (requires username + key)
INTRINIO_API_KEY=your_key_here
INTRINIO_USERNAME=your_username_here

# Alpha Vantage
ALPHAVANTAGE_API_KEY=your_key_here
```

### Testing Your Setup

Run the included test scripts:

```bash
# Check configuration status
python premium_api_guide.py

# Test API connections
python test_premium_apis.py

# Run full system with premium data
python strategy_execution_demo.py
```

## 💰 Cost Optimization

### Free Tier Strategy (Recommended Start)
- Use Polygon.io + Finnhub free tiers
- Supplement with Yahoo Finance fallback
- Total cost: **$0/month**
- Rate limits: ~150-300 calls/minute combined

### Basic Premium Setup
- Polygon.io Starter ($99/mo) + Finnhub Basic ($10/mo)
- Total cost: **$109/month**
- Rate limits: ~5.3M calls/month
- Best balance of cost/performance

### Professional Setup
- Polygon.io Pro ($999/mo) + IEX Cloud Scale ($99/mo) + Intrinio Developer ($200/mo)
- Total cost: **$1,298/month**
- Rate limits: ~51M calls/month
- For high-frequency trading operations

## 🔄 Automatic Fallback

The AAC system automatically falls back through this priority order:

1. **Premium APIs** (if configured and working)
2. **Alpha Vantage** (if API key configured)
3. **Yahoo Finance** (always available)
4. **Simulated data** (for testing)

This ensures the system always has data, even if premium APIs fail.

## 📞 Support

- **Polygon.io**: Excellent documentation and community support
- **Finnhub**: Active Discord community
- **IEX Cloud**: Good documentation
- **Twelve Data**: Responsive support
- **Intrinio**: Institutional support
- **Alpha Vantage**: Basic forum support

## 🎯 Next Steps

1. Start with Polygon.io free tier (best bang for buck)
2. Add Finnhub for fundamentals/news data
3. Monitor usage and upgrade as needed
4. Consider IEX Cloud for extensive historical data needs

The system will automatically use premium APIs when configured, providing better data quality and higher reliability for your arbitrage strategies.